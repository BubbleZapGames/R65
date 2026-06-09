# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Trait dispatch chain coalescing analysis.

When a function makes back-to-back TraitDispatch calls on the same far-self
vreg, each dispatch redundantly re-emits the PHB / load-bank / PHA / PLB
bracket plus a Y reload. For chained calls on the same self, DBR can stay
set across the whole chain, saving ~10 cycles per chained dispatch.

The pass detects two kinds of chains:

  - far-self chains: coalesce the DBR bracket. Soundness predicate is
    DBR-independence — every gap instruction and every method impl in
    the chain must execute correctly under the chain's DBR.
  - near-self chains: coalesce the LDY reload. Soundness predicate is
    Y-preservation — every gap instruction and every method impl must
    leave Y unchanged.

Walks straight-line CFG paths, with a strict if/else diamond bridge
extension. Soundness is verified using the trait-impl-resolved CallGraph
(see analysis/call_graph.py): every impl in the trait's jump table must
satisfy the chain predicate, and every instruction in the gap between
chained dispatches must too.

Cast-transparency: two dispatches share a self if walking back through
trivial trait-pointer-to-trait-pointer casts (Move and same-size, same-
far/near TypeConvert) leads to the same root vreg. This lets dispatches
chain across different trait aliases of the same underlying object.
"""

from dataclasses import dataclass
from typing import Callable

from r65.compiler.mir.nodes import (
    ChainRole, MIRFunction, HardwareRegister,
    Load, Store, LoadIndirect, StoreIndirect,
    Call, TraitDispatch, MemoryLocation, VirtualRegister,
    Move, TypeConvert, Compare, BinaryOp, UnaryOp, BitTest, InlineAsm,
)


@dataclass(frozen=True)
class ChainKind:
    """Bundles the predicates that distinguish far-self vs near-self chains.

    Both chain kinds share the same path-walking + diamond-bridging
    skeleton. They differ only in:
      - ``is_chainable``: which TraitDispatch shapes start/extend a chain.
      - ``instr_ok``: per-instruction soundness check used both inside
        the gap between chain members and inside diamond arm bodies.
      - ``impls_ok``: trait-method-impls-soundness check for every
        method dispatched in the chain.
    """
    name: str
    is_chainable: Callable
    instr_ok: Callable
    impls_ok: Callable


# Cache attribute name for DBR-independence (set on MIRFunction).
_DBR_INDEP_ATTR = '_chain_dbr_independent'

# Cache attribute names for cast-transparency tracking on MIRFunction:
#   _chain_self_roots: Dict[VirtualRegister, VirtualRegister]
#       memoizes the canonical root self vreg for each vreg encountered.
#   _chain_vreg_def_map: Dict[int (vreg.id), MIRInstruction]
#       lazy single-defining-instr map used by _chain_self_root to walk
#       the def chain. Built once per function on first lookup.
_SELF_ROOT_ATTR = '_chain_self_roots'
_VREG_DEF_ATTR = '_chain_vreg_def_map'


def analyze_trait_dispatch_chains(mir_program, call_graph):
    """Detect runs of TraitDispatch on the same far-self vreg and assign
    ChainRole values to coalesce DBR brackets.

    Args:
        mir_program: MIRProgram to analyze (mutated in place).
        call_graph: CallGraph from CallGraphAnalyzer.analyze() with
            trait_dispatch_info loaded.
    """
    func_map = {f.name: f for f in mir_program.functions}
    # Track recursion via SCCs in the call graph. Functions in any cycle
    # cannot be definitively proven DBR-independent without iteration to
    # fixpoint; we treat them conservatively as NOT independent.
    from r65.compiler.analysis.call_graph import CallGraphAnalyzer
    # Build a temporary analyzer just to reuse find_cycles on this graph.
    cga = CallGraphAnalyzer(mir_program)
    cga.graph = call_graph
    cycles = cga.find_cycles()
    cyclic_funcs: set = set()
    for cycle in cycles:
        cyclic_funcs.update(cycle)

    for func in mir_program.functions:
        _analyze_function_chains(func, call_graph, func_map, cyclic_funcs)


def _analyze_function_chains(func, call_graph, func_map, cyclic_funcs):
    """Find chainable runs in a single function and assign ChainRole.

    Runs the path walker twice per function — once for far-self
    DBR-bracket coalescing, once for near-self Y-reload coalescing.
    Each pass uses its own visited_blocks set so a diamond bridged for
    one chain kind doesn't block the other.
    """
    for kind in (FAR, NEAR):
        visited_blocks: set = set()
        for entry_block_id in func.blocks:
            if entry_block_id in visited_blocks:
                continue
            path = _extended_path(
                func, entry_block_id, visited_blocks,
                call_graph, func_map, cyclic_funcs,
                kind=kind,
            )
            if not path:
                continue
            _coalesce_chains_along_path(
                func, path, call_graph, func_map, cyclic_funcs,
                kind=kind,
            )


def _extended_path(func, entry_block_id, visited_blocks,
                   call_graph, func_map, cyclic_funcs, kind):
    """Like ``_straight_line_path`` but additionally bridges across simple
    if/else diamonds where both arms satisfy ``kind.instr_ok`` and don't
    redefine the chain root self vreg.

    The diamond rule: when the straight-line walk terminates at a block
    ``A`` because ``A`` has two successors, accept the rejoin ``C`` as
    the next block in the path iff:
      - ``A`` has exactly two successors ``S1`` and ``S2``.
      - ``S1`` and ``S2`` each have exactly one successor and that
        successor is the SAME block ``C``.
      - ``C`` has exactly two predecessors and they are exactly
        ``{S1, S2}``.
      - Neither ``S1`` nor ``S2`` has further branching (each is a
        single-block arm) and each arm body is DBR-independent (no
        RAM access, no near deref, no nested TraitDispatch, no call
        to a non-DBR-independent function).
      - Neither ``S1`` nor ``S2`` is a back-edge into earlier path
        (we never revisit a block already in the path or
        ``visited_blocks``).

    The diamond's two arms are inserted into the path AFTER ``A`` and
    BEFORE ``C`` (in arm order S1 then S2). When the chain pass walks
    the resulting flat instruction list, instructions from BOTH arms
    appear between any chain-head dispatch in or before ``A`` and any
    chain-tail dispatch in ``C``. ``_between_is_dbr_independent``
    already rejects any TraitDispatch, RAM access, near deref, or
    self-vreg redef inside the gap — these constraints subsume the
    soundness conditions for the diamond.

    Loops, multi-arm switches, and any join with topology that doesn't
    match the strict diamond pattern are rejected — the walker
    terminates and returns the path so far.
    """
    path = []
    cur_id = entry_block_id
    while cur_id is not None and cur_id not in visited_blocks:
        block = func.blocks.get(cur_id)
        if block is None:
            break
        visited_blocks.add(cur_id)
        path.append(block)

        succs = block.successors
        if len(succs) == 1:
            next_id = succs[0]
            next_block = func.blocks.get(next_id)
            if next_block is None:
                break
            if len(next_block.predecessors) != 1:
                # Try to bridge a diamond rejoin reached from another
                # arm. Today the straight-line rule rejects any block
                # with >1 predecessor; the diamond extension only
                # applies when the *current* block has 2 successors.
                break
            if next_block.predecessors[0] != cur_id:
                break
            cur_id = next_id
            continue

        if len(succs) == 2:
            bridged = _try_bridge_diamond(
                func, block, visited_blocks, path,
                call_graph, func_map, cyclic_funcs,
                kind=kind,
            )
            if bridged is None:
                break
            arm_blocks, rejoin_id = bridged
            # Add arm blocks to path so their instructions participate in
            # the inter-dispatch DBR-independence check via the flat
            # walker. Mark them visited so the outer iteration doesn't
            # restart a fresh path inside an arm.
            for arm in arm_blocks:
                visited_blocks.add(arm.block_id)
                path.append(arm)
            cur_id = rejoin_id
            continue

        # 0 successors (terminal) or 3+ (switch) — stop.
        break

    return path


def _try_bridge_diamond(func, head_block, visited_blocks, path,
                        call_graph, func_map, cyclic_funcs, kind):
    """Detect a strict if/else diamond rooted at ``head_block`` and return
    ``(arm_blocks, rejoin_id)`` if bridgeable, else None.

    See ``_extended_path`` for the diamond shape and arm soundness rules.
    Each arm body must satisfy ``kind.instr_ok`` for every instruction.
    """
    succs = head_block.successors
    if len(succs) != 2:
        return None
    s1_id, s2_id = succs[0], succs[1]
    if s1_id == s2_id:
        return None
    s1 = func.blocks.get(s1_id)
    s2 = func.blocks.get(s2_id)
    if s1 is None or s2 is None:
        return None

    # Reject loops / back-edges: arm blocks must not already be in the
    # path or otherwise visited (which would mean we're looping back).
    for arm_id in (s1_id, s2_id):
        if arm_id in visited_blocks:
            return None

    # Each arm must have exactly the head as its sole predecessor.
    if list(s1.predecessors) != [head_block.block_id]:
        return None
    if list(s2.predecessors) != [head_block.block_id]:
        return None

    # Each arm must have exactly one successor, and they must match.
    if len(s1.successors) != 1 or len(s2.successors) != 1:
        return None
    rejoin_id = s1.successors[0]
    if s2.successors[0] != rejoin_id:
        return None

    rejoin = func.blocks.get(rejoin_id)
    if rejoin is None:
        return None

    # Rejoin block must have exactly the two arms as predecessors.
    rejoin_preds = set(rejoin.predecessors)
    if rejoin_preds != {s1_id, s2_id}:
        return None

    # Reject loops at the rejoin too.
    if rejoin_id in visited_blocks:
        return None

    # Each arm body must be safe to traverse under the chain's predicate.
    # The arm's terminator (an unconditional branch back to the rejoin)
    # isn't modeled as a DBR/Y-relevant op.
    for arm in (s1, s2):
        for instr in arm.instructions:
            if not kind.instr_ok(instr, call_graph, func_map, cyclic_funcs):
                return None

    return [s1, s2], rejoin_id


def _coalesce_chains_along_path(func, path, call_graph, func_map,
                                cyclic_funcs, kind):
    """Walk a path in order; identify same-self runs of TraitDispatch and
    assign ChainRole (and, for far chains, ``self_y_preloaded``).

    For ``kind=FAR``: DBR-bracket coalescing using the DBR-independence
    predicate. Additionally sets ``self_y_preloaded`` on MIDDLE/END
    members when every (trait, method) impl also preserves Y at exit.

    For ``kind=NEAR``: LDY-reload coalescing using the Y-preservation
    predicate. ChainRole alone controls Y emission (no DBR work).

    A run extends as long as every member shares the same chain root
    self vreg as the chain head and is the right kind of dispatch.
    Methods and traits may differ; the impl-set predicate is re-
    evaluated per-method.
    """
    # Flatten (block_idx, instr_idx, instr) so cross-block / cross-arm
    # gaps can be checked uniformly.
    flat = []
    for b_idx, block in enumerate(path):
        for i_idx, instr in enumerate(block.instructions):
            flat.append((b_idx, i_idx, instr))

    is_chainable = kind.is_chainable
    impls_pass = kind.impls_ok
    between_ok = lambda prev_idx, next_idx: _between_satisfies(
        flat, prev_idx, next_idx, kind.instr_ok,
        call_graph, func_map, cyclic_funcs, func=func,
    )

    n = len(flat)
    i = 0
    while i < n:
        instr = flat[i][2]
        if not is_chainable(instr):
            i += 1
            continue

        # Verify the impl set passes the chain predicate for THIS dispatch.
        if not impls_pass(instr, call_graph, func_map, cyclic_funcs):
            i += 1
            continue

        # Track the union of (trait_name, method_name) pairs visited in the
        # chain. Methods may differ within a chain, so the impl-set check
        # is re-evaluated for every member. Tracking the set explicitly
        # makes the soundness rule readable.
        chain_methods = {(instr.trait_name, instr.method_name)}

        # Try to extend the chain forward through `flat`. Skip past any
        # gap-safe non-dispatch instructions that lie between adjacent
        # chain candidates.
        run_indices = [i]
        j = i + 1
        while j < n:
            # Find the next chainable dispatch at or after j (matching
            # the same kind — near vs far).
            cand_idx = j
            while cand_idx < n and not is_chainable(flat[cand_idx][2]):
                cand_idx += 1
            if cand_idx >= n:
                break

            # Inter-dispatch instructions: from flat[run_indices[-1]+1] up
            # to flat[cand_idx-1]. Verify they pass the chain predicate
            # and don't redefine the chain root self vreg.
            if not between_ok(run_indices[-1], cand_idx):
                break

            cand = flat[cand_idx][2]
            # Pass func so _same_self_chain can consult the cast-
            # transparency walker; without it the comparison falls back
            # to plain vreg-identity equality.
            if not _same_self_chain(instr, cand, func):
                break
            # Re-check the candidate's method's impls.
            if not impls_pass(cand, call_graph, func_map, cyclic_funcs):
                break

            chain_methods.add((cand.trait_name, cand.method_name))
            run_indices.append(cand_idx)
            j = cand_idx + 1

        # Assign roles
        if len(run_indices) >= 2:
            # For FAR chains, additionally compute the Y-elision flag.
            # DBR-bracket coalescing (set above by role) and Y elision are
            # orthogonal — each FAR chain may have both, only DBR
            # coalescing, or neither. NEAR chains use ChainRole alone for
            # Y-elision; the FAR-only flag doesn't apply.
            y_preloaded = False
            if kind is FAR:
                y_preloaded = _far_chain_can_elide_y(
                    flat, run_indices, call_graph, func_map,
                    cyclic_funcs, func,
                )

            for k, run_idx in enumerate(run_indices):
                td = flat[run_idx][2]
                if k == 0:
                    td.self_chain_role = ChainRole.START
                elif k == len(run_indices) - 1:
                    td.self_chain_role = ChainRole.END
                else:
                    td.self_chain_role = ChainRole.MIDDLE
                # MIDDLE/END skip the Y reload only when the chain
                # passes the Y predicate. START always loads Y (no
                # prior dispatch put the address there).
                if y_preloaded and k > 0:
                    td.self_y_preloaded = True
            i = run_indices[-1] + 1
        else:
            # Single dispatch — leave as SOLO (default)
            i += 1


def _is_chainable_far_dispatch(instr):
    """True if instr is a far-self TraitDispatch with a vreg self_ptr."""
    if not isinstance(instr, TraitDispatch):
        return False
    if not instr.self_is_far:
        return False
    if not isinstance(instr.self_ptr, VirtualRegister):
        return False
    return True


def _is_chainable_near_dispatch(instr):
    """True if instr is a near-self TraitDispatch with a vreg self_ptr.

    Mirrors ``_is_chainable_far_dispatch`` but for near-self chains.
    """
    if not isinstance(instr, TraitDispatch):
        return False
    if instr.self_is_far:
        return False
    if not isinstance(instr.self_ptr, VirtualRegister):
        return False
    return True


def _same_self_chain(a, b, func=None):
    """True if two TraitDispatches can chain DBR-bracket-wise.

    Cast-transparency: dispatches across different trait aliases of
    the same underlying object share a self if walking back through
    trivial trait-pointer-to-trait-pointer casts (Move and zero-shift
    TypeConvert) leads to the same root vreg. The trait_name doesn't
    need to match — different trait-pointer aliases share bank+address,
    and the DBR bracket / Y reload work identically for both.

    Bank-compatibility is enforced by the cast walker
    (``_chain_self_root``): only same-bank, same-size casts are
    followed. Casts that change far/near or that involve pointer
    arithmetic (e.g. ``&obj.weapon as *Trait``) terminate the walk
    early — they yield distinct roots and therefore distinct chains.

    ``func`` is the enclosing MIRFunction; if None, the comparison
    falls back to plain vreg-identity equality.
    """
    if a.self_ptr is None or b.self_ptr is None:
        return False
    if not (isinstance(a.self_ptr, VirtualRegister)
            and isinstance(b.self_ptr, VirtualRegister)):
        return False

    if func is not None:
        root_a = _chain_self_root(a.self_ptr, func)
        root_b = _chain_self_root(b.self_ptr, func)
        return root_a.id == root_b.id

    if a.self_ptr is b.self_ptr:
        return True
    return a.self_ptr.id == b.self_ptr.id


def _chain_self_root(vreg: VirtualRegister, func: MIRFunction) -> VirtualRegister:
    """Walk back through trivial trait-pointer casts to the root self
    vreg.

    A cast is "trivial" (transparent) if it preserves bank+address:
      - `Move` from another VirtualRegister source — pure aliasing.
      - `TypeConvert` between two pointer types where:
          * source and target are both `PointerTypeInfo`
          * `source_type.is_far == target_type.is_far` (same far/near)
          * `source_type.size_bytes == target_type.size_bytes` (same
            on-the-wire layout — implied by same is_far for pointers,
            but we check both defensively)

    This covers trait-pointer ↔ trait-pointer casts and trait-pointer
    ↔ underlying struct-pointer casts. It does NOT cover:
      - far ↔ near pointer casts (different bank semantics + size)
      - pointer ↔ integer casts
      - pointer arithmetic / field-offset casts (these are emitted as
        BinaryOp / UnaryOp / different MIR shapes, not Move/TypeConvert,
        and would terminate the walk on the first non-cast def)

    Note on MIR shapes: HIRTypeCast lowering in
    `mir/lowerers/expression.py` emits a plain `Move` for same-size
    reinterpretation casts and a `TypeConvert` for size-changing or
    pointer↔integer casts (see lower_type_cast at ~line 274).
    Trait-pointer-to-trait-pointer casts are same-size and therefore
    arrive as a `Move`, which the walker follows.

    Termination: the walk stops on:
      - non-Move, non-trivial-TypeConvert def (Load, Call, BinaryOp,
        TraitDispatch return, etc.)
      - a vreg with no def in the function (parameter or pre-existing)
      - a vreg already visited (cycle guard — should never occur with
        SSA-like single-def, but defensive)
      - a Move/TypeConvert source that is not a VirtualRegister
        (Immediate, MemoryLocation, hardware register, etc.)
    """
    cache = getattr(func, _SELF_ROOT_ATTR, None)
    if cache is None:
        cache = {}
        setattr(func, _SELF_ROOT_ATTR, cache)

    if vreg.id in cache:
        return cache[vreg.id]

    visited = set()
    cur = vreg
    while True:
        if cur.id in visited:
            break
        visited.add(cur.id)
        # Already-cached intermediate root short-circuits the walk.
        if cur.id in cache and cache[cur.id].id != cur.id:
            cur = cache[cur.id]
            continue

        defining = _chain_lookup_def(cur, func)
        next_src = _trivial_cast_source(defining)
        if next_src is None:
            break
        cur = next_src

    # Cache every visited vreg as resolving to the same root for future
    # walks.
    for vid in visited:
        cache[vid] = cur
    return cur


def _chain_lookup_def(vreg: VirtualRegister, func: MIRFunction):
    """Return the (single) defining instruction of `vreg` in `func`, or
    None if no def is found in the function. Builds and caches a
    per-function vreg.id -> defining instr map on first call.

    If a vreg has multiple defs (rare in trait-self contexts but
    possible through phi-like structures or rebinding), we return None
    to be conservative — a multiply-defined vreg cannot be safely
    walked through.
    """
    def_map = getattr(func, _VREG_DEF_ATTR, None)
    if def_map is None:
        def_map = {}
        # `multi` tracks vregs with more than one definition; for those
        # we record None to signal "not safe to walk".
        multi = set()
        for block in func.blocks.values():
            for instr in block.instructions:
                # `dest` covers Move, TypeConvert, Load, BinaryOp, etc.
                dest = getattr(instr, 'dest', None)
                if isinstance(dest, VirtualRegister):
                    if dest.id in def_map and def_map[dest.id] is not instr:
                        multi.add(dest.id)
                    def_map[dest.id] = instr
                # `returns` covers Call/TraitDispatch return vregs.
                returns = getattr(instr, 'returns', None) or []
                for r in returns:
                    if isinstance(r, VirtualRegister):
                        if r.id in def_map and def_map[r.id] is not instr:
                            multi.add(r.id)
                        def_map[r.id] = instr
        for vid in multi:
            def_map[vid] = None
        setattr(func, _VREG_DEF_ATTR, def_map)

    return def_map.get(vreg.id)


def _trivial_cast_source(instr):
    """Return the VirtualRegister source of a trivial trait-pointer cast,
    or None if `instr` is not a transparent cast.

    A Move from a VirtualRegister source is always trivial (pure SSA
    rename). A TypeConvert is trivial only if both source and target
    are pointer types with matching is_far flags and matching sizes
    — see `_chain_self_root` for the rationale and what this rejects.
    """
    if instr is None:
        return None

    if isinstance(instr, Move):
        if isinstance(instr.source, VirtualRegister):
            return instr.source
        return None

    if isinstance(instr, TypeConvert):
        # Conservative: both types must be PointerTypeInfo with matching
        # is_far AND matching size_bytes. We import lazily to avoid a
        # circular dependency with the HIR types module.
        try:
            from r65.compiler.hir.types import PointerTypeInfo
        except ImportError:
            return None
        st = instr.source_type
        tt = instr.target_type
        if not (isinstance(st, PointerTypeInfo) and isinstance(tt, PointerTypeInfo)):
            return None
        if st.is_far != tt.is_far:
            return None
        # Defensive: size_bytes derives from is_far for pointers, but
        # check explicitly so future PointerTypeInfo extensions (e.g.
        # tagged pointers) don't silently slip through.
        if getattr(st, 'size_bytes', None) != getattr(tt, 'size_bytes', None):
            return None
        if isinstance(instr.source, VirtualRegister):
            return instr.source
        return None

    return None


def _between_satisfies(flat, prev_run_idx, next_idx, instr_ok,
                       call_graph, func_map, cyclic_funcs, func=None):
    """All instructions strictly between flat[prev_run_idx] and
    flat[next_idx] must satisfy ``instr_ok`` AND not redefine the chain
    root self vreg.

    Defining a fresh trivial-cast alias of the root is safe (Move /
    TypeConvert with a different dest vreg); only direct redefinition
    of the root or the dispatch's self_ptr breaks the chain.
    """
    prev_td = flat[prev_run_idx][2]
    self_vreg = prev_td.self_ptr
    root_vreg = None
    if func is not None and isinstance(self_vreg, VirtualRegister):
        root_vreg = _chain_self_root(self_vreg, func)
    for k in range(prev_run_idx + 1, next_idx):
        instr = flat[k][2]
        if not instr_ok(instr, call_graph, func_map, cyclic_funcs, func=func):
            return False
        if _instr_redefines_vreg(instr, self_vreg):
            return False
        if root_vreg is not None and root_vreg.id != self_vreg.id:
            if _instr_redefines_vreg(instr, root_vreg):
                return False
    return True


def _instr_redefines_vreg(instr, vreg):
    """True if instr writes to vreg (which would invalidate chained self)."""
    # Most instruction classes have a `dest` field that is a register.
    dest = getattr(instr, 'dest', None)
    if isinstance(dest, VirtualRegister) and dest.id == vreg.id:
        return True
    # Move's dest covers most rebinds; Call/TraitDispatch returns lists.
    returns = getattr(instr, 'returns', None)
    if returns:
        for r in returns:
            if isinstance(r, VirtualRegister) and r.id == vreg.id:
                return True
    return False


def _instr_is_dbr_independent(instr, call_graph, func_map, cyclic_funcs, func=None):
    """True if `instr` is safe to execute while DBR is set to self's bank
    (rather than the caller's DBR).

    `func` is accepted but unused — it's there so the signature matches
    ``_instr_preserves_y`` and both can be plugged into the same predicate
    helpers (``_function_satisfies``, ``_between_satisfies``).

    Disqualifiers (per docs/register_memory_config.md and the brief):
      - non-LONG RAM access (DBR-relative absolute load/store of RAM)
      - near pointer dereference
      - call to a non-DBR-independent function
      - nested far-self TraitDispatch on a DIFFERENT self vreg
    """
    if isinstance(instr, (Load, Store)):
        loc = instr.source if isinstance(instr, Load) else instr.dest
        if isinstance(loc, MemoryLocation):
            st = loc.storage_type
            if st in ('ram', 'lowram'):
                # RAM and lowram both use DBR-relative absolute addressing.
                # ROM/HW are emitted as LONG already; zeropage uses DP.
                # Lowram lives in bank $7E ($00-$1FFF) but the codegen
                # emits absolute (not LONG), so under a non-default DBR
                # it would address the wrong bank.
                return False
        return True

    # Some MIR instructions accept a MemoryLocation directly as an
    # operand (the codegen folds Load+Compare into a single CMP abs).
    # Reject any direct memory operand whose storage type is DBR-
    # dependent. Inline MemoryLocation operands are observed on
    # Compare, BitTest, BinaryOp, UnaryOp.
    if isinstance(instr, (Compare, BitTest, BinaryOp, UnaryOp)):
        for attr in ('left', 'right', 'source', 'value'):
            operand = getattr(instr, attr, None)
            if isinstance(operand, MemoryLocation):
                st = operand.storage_type
                if st in ('ram', 'lowram'):
                    return False
        return True

    if isinstance(instr, (LoadIndirect, StoreIndirect)):
        if not instr.is_far:
            # Near pointer deref — DBR-dependent.
            return False
        return True

    if isinstance(instr, Call):
        # Direct calls: recurse into callee.
        if isinstance(instr.function, str):
            return _function_is_dbr_independent(
                instr.function, call_graph, func_map, cyclic_funcs
            )
        # Indirect call via fn pointer: any function whose address is taken
        # is a possibility — too broad to prove independent. Reject.
        return False

    if isinstance(instr, TraitDispatch):
        # A nested TraitDispatch on the SAME self vreg is fine — it sets
        # DBR to the same bank. A different self is a disqualifier.
        # However, the "between" walker handles "same self" only at top
        # level (the chain itself); a nested dispatch in between would
        # re-bracket DBR. Conservatively reject any TraitDispatch inside
        # the gap region.
        return False

    # Other instructions (arithmetic, compare, branch, mode set, push/pull,
    # inline asm, save/restore reg, type convert, ...) are DBR-independent.
    return True


def _function_satisfies(func_name, *, attr, instr_ok, call_graph, func_map,
                        cyclic_funcs, extra_guard=None):
    """Recursively check whether ``func_name``'s body satisfies a
    per-instruction predicate, memoized on ``MIRFunction.<attr>``.

    Conservative rejections (shared by all predicates):
      - Function not found in ``func_map`` (external/builtin) -> False
      - Function in a recursive cycle -> False (no fixpoint analysis)
      - ``extra_guard(func)`` returns False (predicate-specific veto)

    The predicate is marked True optimistically before walking the body,
    so a self-call during recursion sees the in-progress True and
    terminates instead of looping. Real recursion is filtered upstream
    by ``cyclic_funcs``.
    """
    func = func_map.get(func_name)
    if func is None:
        return False  # external — can't analyze

    cached = getattr(func, attr, None)
    if cached is not None:
        return cached

    if func_name in cyclic_funcs:
        setattr(func, attr, False)
        return False

    if extra_guard is not None and not extra_guard(func):
        setattr(func, attr, False)
        return False

    setattr(func, attr, True)  # optimistic — breaks self-loops

    for block in func.blocks.values():
        for instr in block.instructions:
            if not instr_ok(instr, call_graph, func_map, cyclic_funcs, func=func):
                setattr(func, attr, False)
                return False

    return True


def _dbr_indep_function_guard(func):
    """Functions that play their own DBR/D games can't run under ours.

    SET_DBR functions assume their own DBR; D=S functions remap D in
    their prologue. Either way, splicing the body inside a chain whose
    DBR is set to a different bank is unsafe.
    """
    if func.has_far_ptr_stack_params:
        return False
    if func.self_far_uses_d_equals_s:
        return False
    return True


def _function_is_dbr_independent(func_name, call_graph, func_map, cyclic_funcs):
    """True if ``func_name``'s body executes DBR-independently. Memoized."""
    return _function_satisfies(
        func_name, attr=_DBR_INDEP_ATTR,
        instr_ok=_instr_is_dbr_independent,
        call_graph=call_graph, func_map=func_map, cyclic_funcs=cyclic_funcs,
        extra_guard=_dbr_indep_function_guard,
    )


def _trait_method_impls_all_independent(td, call_graph, func_map, cyclic_funcs):
    """True if every implementor of (td.trait_name, td.method_name) is
    DBR-independent. False if the impl set is empty/unknown — conservative.
    """
    impls = call_graph.resolve_trait_method(td.trait_name, td.method_name)
    if not impls:
        return False
    for impl_name in impls:
        if not _function_is_dbr_independent(
            impl_name, call_graph, func_map, cyclic_funcs
        ):
            return False
    return True


# ============================================================================
# Y-preservation predicate (used by chain coalescing)
# ============================================================================
#
# A function "preserves Y" iff Y at every exit holds the same value as
# at entry. The chain pass uses this to elide LDY reloads in both
# near-self and far-self chains.
#
# Pre-codegen we don't have register allocation results, so we use a
# conservative MIR-level approximation:
#
#   - For a trait-method impl using the DBR:Y leaf path (self_y_vreg
#     present, no D=S, no far-ptr-stack-params), the trait-method
#     ABI guarantees Y holds self throughout the body. Field accesses
#     read via $offset,Y (Y-USE only). Such methods preserve Y iff
#     no other instruction writes Y or calls a non-Y-preserving
#     function.
#
#   - For other functions, we conservatively reject any:
#       * Move / TypeConvert / Load / LoadIndirect with HW-Y dest
#       * vreg with register_hint == 'Y' (loop register promotion
#         intends to allocate it to Y)
#       * Call to a non-Y-preserving function
#       * TraitDispatch (we don't model whether the dispatch wrapper
#         preserves Y — conservatively reject)
#       * InlineAsm (we don't analyze asm)
#       * Recursive cycle (without fixpoint we can't tell)
#
# Memoized on `MIRFunction._chain_preserves_y`.

_PRESERVES_Y_ATTR = '_chain_preserves_y'


def _vreg_targets_y(vreg):
    """True if a vreg is intended for Y allocation (register_hint='Y')."""
    if not isinstance(vreg, VirtualRegister):
        return False
    return getattr(vreg, 'register_hint', None) == 'Y'


def _instr_writes_y(instr, func):
    """True if an instruction writes the Y register.

    The MIR shapes that produce a Y write:
      - dest is HardwareRegister('Y')
      - dest is the function's self_y_vreg (pre-allocated to Y)
      - dest is a vreg with register_hint='Y' (loop promotion target)
      - returns list contains a HW Y or Y-hinted vreg

    The trait-method entry-point Move(dest=self_y_vreg, source=HW_Y)
    is a no-op at codegen time (move_select.py treats it as such),
    but conceptually Y already holds self at function entry — so
    treating it as a Y-write would incorrectly flag every trait
    method as Y-clobbering. We therefore EXCLUDE that specific
    instruction shape from the writes-Y check.
    """
    self_y_id = func.self_y_vreg.id if func.self_y_vreg else None

    dest = getattr(instr, 'dest', None)
    if isinstance(dest, HardwareRegister) and dest.name == 'Y':
        return True
    if isinstance(dest, VirtualRegister):
        # The trait-method self_y_vreg is pre-allocated to Y but the
        # entry-point Move(dest=self_y_vreg, source=HW_Y) is a no-op
        # — Y already holds self. Subsequent writes to self_y_vreg
        # WOULD clobber Y, but the parameter binding is the only
        # such write in well-formed MIR, so excluding it is safe.
        if self_y_id is not None and dest.id == self_y_id:
            return False
        if _vreg_targets_y(dest):
            return True

    returns = getattr(instr, 'returns', None) or []
    for r in returns:
        if isinstance(r, HardwareRegister) and r.name == 'Y':
            return True
        if isinstance(r, VirtualRegister) and _vreg_targets_y(r):
            return True

    return False


def _instr_preserves_y(instr, call_graph, func_map, cyclic_funcs, func=None):
    """True if `instr` is safe to execute in the gap of a Y-preserving
    chain — Y is unchanged after `instr` runs.

    Disqualifiers:
      - any write to Y (see ``_instr_writes_y``)
      - call to a non-Y-preserving function
      - nested TraitDispatch (unconditionally clobbers Y in the
        dispatch wrapper / impl)
      - inline assembly (we don't analyze asm)
    """
    if isinstance(instr, InlineAsm):
        return False

    if isinstance(instr, TraitDispatch):
        # Nested dispatches always reload Y for their own self.
        return False

    if isinstance(instr, Call):
        if isinstance(instr.function, str):
            if not _function_preserves_y(
                instr.function, call_graph, func_map, cyclic_funcs
            ):
                return False
        else:
            # Indirect call via fn pointer — broad, conservatively no.
            return False
        # Even if the callee preserves Y, check that this call's
        # returns list doesn't write Y.
        if func is not None and _instr_writes_y(instr, func):
            return False
        return True

    if func is not None and _instr_writes_y(instr, func):
        return False
    return True


def _function_preserves_y(func_name, call_graph, func_map, cyclic_funcs):
    """True if ``func_name``'s body preserves Y at every exit. Memoized."""
    return _function_satisfies(
        func_name, attr=_PRESERVES_Y_ATTR,
        instr_ok=_instr_preserves_y,
        call_graph=call_graph, func_map=func_map, cyclic_funcs=cyclic_funcs,
    )


def _trait_method_impls_all_preserve_y(td, call_graph, func_map, cyclic_funcs):
    """True if every implementor of (td.trait_name, td.method_name)
    preserves Y at exit. False if the impl set is empty/unknown.
    """
    impls = call_graph.resolve_trait_method(td.trait_name, td.method_name)
    if not impls:
        return False
    for impl_name in impls:
        if not _function_preserves_y(
            impl_name, call_graph, func_map, cyclic_funcs
        ):
            return False
    return True


def _far_chain_can_elide_y(flat, run_indices, call_graph, func_map,
                           cyclic_funcs, func):
    """For a far-self chain, return True if every chain member's method
    impls and every inter-dispatch gap instruction preserve Y.

    The DBR-independence checks (from the chain assignment loop above)
    do NOT imply Y-preservation: a function may be DBR-independent but
    still clobber Y (e.g. with ``Y = 0``). Conversely, a function may
    preserve Y but not be DBR-independent (e.g. it does a non-LONG RAM
    read that requires DBR=$00). The two predicates are independent.
    """
    # Every method's impl set must preserve Y.
    seen_methods: set = set()
    for run_idx in run_indices:
        td = flat[run_idx][2]
        key = (td.trait_name, td.method_name)
        if key in seen_methods:
            continue
        seen_methods.add(key)
        if not _trait_method_impls_all_preserve_y(
            td, call_graph, func_map, cyclic_funcs
        ):
            return False
    # Every gap between consecutive members must preserve Y.
    for k in range(len(run_indices) - 1):
        prev_idx = run_indices[k]
        next_idx = run_indices[k + 1]
        if not _between_satisfies(
            flat, prev_idx, next_idx, _instr_preserves_y,
            call_graph, func_map, cyclic_funcs, func=func,
        ):
            return False
    return True


# ChainKind singletons. Defined at module bottom so the predicate
# functions they reference exist at import time.
FAR = ChainKind(
    name='far',
    is_chainable=_is_chainable_far_dispatch,
    instr_ok=_instr_is_dbr_independent,
    impls_ok=_trait_method_impls_all_independent,
)

NEAR = ChainKind(
    name='near',
    is_chainable=_is_chainable_near_dispatch,
    instr_ok=_instr_preserves_y,
    impls_ok=_trait_method_impls_all_preserve_y,
)
