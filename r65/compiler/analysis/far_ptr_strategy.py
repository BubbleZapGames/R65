# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Far pointer strategy analysis.

For functions with far pointer stack parameters, choose between two strategies:
- D_EQUALS_S: PHD/TSC/TCD sets D=S, enables [dp],Y indirect long addressing.
  Disables zeropage/scratch registers (DP no longer points to zeropage).
- SET_DBR: PHB/LDA/PHA/PLB sets DBR to pointer's bank, enables bare (d,S),Y
  addressing. Keeps DP intact, preserving scratch registers and zeropage access.

Cost model:
  D=S cost     = 13 + 1*N_zp + 13*N_calls
  SET_DBR cost = 19 + 1*N_rom + 1*N_hw + 1*N_ram + 14*N_calls

SET_DBR wins when: N_zp > N_rom + N_hw + N_ram + N_calls + 6
"""

from r65.compiler.mir.nodes import (
    FarPtrStrategy, ChainRole, MIRFunction, BasicBlock,
    Load, Store, LoadIndirect, StoreIndirect,
    Call, TraitDispatch, MemoryLocation, VirtualRegister,
    Move, TypeConvert,
)


def analyze_far_ptr_strategy(mir_program):
    """Analyze all functions and set far_ptr_strategy on those with far pointer stack params."""
    for func in mir_program.functions:
        if not func.has_far_ptr_stack_params:
            continue
        func.far_ptr_strategy = _choose_strategy(func)


def _choose_strategy(func: MIRFunction) -> FarPtrStrategy:
    """Choose the optimal far pointer access strategy for a function."""
    # D=S is incompatible with scratch params: D=S moves DP to the stack,
    # so DP addresses no longer reach zeropage scratch registers.
    # Force SET_DBR when scratch params are present.
    if func.scratch_param_addrs and _is_set_dbr_safe(func):
        return FarPtrStrategy.SET_DBR

    # Safety checks — force D=S if any disqualifier
    if not _is_set_dbr_safe(func):
        return FarPtrStrategy.D_EQUALS_S

    # Count access types
    n_zp, n_rom, n_hw, n_ram, n_calls = _count_accesses(func)

    # Cost comparison:
    # D=S cost     = 13 + n_zp + 13*n_calls
    # SET_DBR cost = 19 + n_rom + n_hw + n_ram + 14*n_calls
    # RAM needs LONG under SET_DBR because DBR may not be $7E
    # SET_DBR wins when: n_zp > n_rom + n_hw + n_ram + n_calls + 6
    d_equals_s_cost = 13 + n_zp + 13 * n_calls
    set_dbr_cost = 19 + n_rom + n_hw + n_ram + 14 * n_calls

    if set_dbr_cost < d_equals_s_cost:
        return FarPtrStrategy.SET_DBR
    return FarPtrStrategy.D_EQUALS_S


def _is_set_dbr_safe(func: MIRFunction) -> bool:
    """Check if SET_DBR strategy is safe for this function."""
    # 1. Multiple far pointer stack params — can't set DBR to two banks
    if len(func.far_ptr_param_indices) > 1:
        return False

    # 2. Trait methods with self_far_uses_d_equals_s — complex interaction
    if func.self_far_uses_d_equals_s:
        return False

    # 3. Check for near pointer derefs and non-param far pointer derefs
    far_param_vregs = set()
    for idx in func.far_ptr_param_indices:
        if idx in func.param_to_vreg:
            far_param_vregs.add(func.param_to_vreg[idx])

    for block in func.blocks.values():
        for instr in block.instructions:
            if isinstance(instr, (LoadIndirect, StoreIndirect)):
                # Near pointer dereference — changing DBR changes bank semantics
                if not instr.is_far:
                    ptr = instr.pointer if isinstance(instr, LoadIndirect) else instr.pointer
                    if isinstance(ptr, VirtualRegister):
                        return False

                # Far pointer dereference through non-param pointer
                if instr.is_far:
                    ptr = instr.pointer if isinstance(instr, LoadIndirect) else instr.pointer
                    if isinstance(ptr, VirtualRegister) and ptr not in far_param_vregs:
                        return False

    return True


def _count_accesses(func: MIRFunction):
    """Count memory access types for cost comparison.

    Returns (n_zp, n_rom, n_hw, n_ram, n_calls).
    """
    n_zp = 0
    n_rom = 0
    n_hw = 0
    n_ram = 0
    n_calls = 0

    for block in func.blocks.values():
        for instr in block.instructions:
            if isinstance(instr, (Load, Store)):
                loc = None
                if isinstance(instr, Load) and isinstance(instr.source, MemoryLocation):
                    loc = instr.source
                elif isinstance(instr, Store) and isinstance(instr.dest, MemoryLocation):
                    loc = instr.dest

                if loc is not None:
                    st = loc.storage_type
                    if st == 'zeropage':
                        n_zp += 1
                    elif st == 'rom':
                        n_rom += 1
                    elif st == 'hw':
                        n_hw += 1
                    elif st == 'ram':
                        n_ram += 1

            elif isinstance(instr, (Call, TraitDispatch)):
                n_calls += 1

    return n_zp, n_rom, n_hw, n_ram, n_calls


# ============================================================================
# Trait dispatch chain coalescing
# ============================================================================
#
# When a function makes back-to-back TraitDispatch calls on the same far-self
# vreg, each dispatch redundantly re-emits the PHB / load-bank / PHA / PLB
# bracket plus a Y reload. For chained calls on the same self, DBR can stay
# set across the whole chain, saving ~10 cycles per chained dispatch.
#
# This v1 pass coalesces only the DBR bracket, leaves Y reload alone,
# only handles same-trait/same-method runs, and only walks straight-line
# CFG paths (each block in the chain has exactly one successor whose only
# predecessor is that block). Soundness is verified using the trait-impl-
# resolved CallGraph (see analysis/call_graph.py): every impl in the
# trait's jump table must be DBR-independent for the chain to fire, and
# every instruction between dispatches must be DBR-independent too.

# Cache attribute name for DBR-independence (set on MIRFunction).
_DBR_INDEP_ATTR = '_chain_dbr_independent'

# Cache attribute names for v1.6 cast-transparency on MIRFunction:
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
    """Find chainable runs in a single function and assign ChainRole."""
    # Build straight-line block paths starting at every block whose chain
    # could produce a coalesceable run. We walk forward through fall-through
    # successors only.
    visited_blocks: set = set()

    for entry_block_id in func.blocks:
        if entry_block_id in visited_blocks:
            continue
        path = _straight_line_path(func, entry_block_id, visited_blocks)
        if not path:
            continue
        _coalesce_chains_along_path(
            func, path, call_graph, func_map, cyclic_funcs
        )


def _straight_line_path(func, entry_block_id, visited_blocks):
    """Return the maximal forward path of blocks starting at entry_block_id
    where each non-terminal block has exactly one successor whose only
    predecessor is that block. Marks blocks as visited.

    Returns a list of BasicBlock objects.
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
        if len(succs) != 1:
            break
        next_id = succs[0]
        next_block = func.blocks.get(next_id)
        if next_block is None:
            break
        if len(next_block.predecessors) != 1:
            break
        if next_block.predecessors[0] != cur_id:
            break
        cur_id = next_id

    return path


def _coalesce_chains_along_path(func, path, call_graph, func_map, cyclic_funcs):
    """Walk a straight-line block path in order; identify same-self runs of
    far-self TraitDispatch and assign ChainRole.

    A run extends as long as every member shares the same trait_name and
    self_ptr vreg as the chain head (method may differ — see v1.5). The
    impl-set DBR-independence predicate is evaluated over the UNION of
    impls across every (trait, method) pair in the chain so far plus the
    new candidate; if adding a candidate would make any impl in the union
    DBR-dependent, the chain is flushed before that candidate.
    """
    # Flatten (block_idx, instr_idx, instr) for every TraitDispatch in path
    # AND track all instructions in order so we can check inter-dispatch
    # DBR-independence by walking the flat list between dispatch positions.
    flat = []  # (block_idx, instr_idx, instr)
    for b_idx, block in enumerate(path):
        for i_idx, instr in enumerate(block.instructions):
            flat.append((b_idx, i_idx, instr))

    n = len(flat)
    i = 0
    while i < n:
        instr = flat[i][2]
        if not _is_chainable_far_dispatch(instr):
            i += 1
            continue

        # Verify the impl set is DBR-independent for THIS dispatch.
        if not _trait_method_impls_all_independent(
            instr, call_graph, func_map, cyclic_funcs
        ):
            i += 1
            continue

        # Track the union of (trait_name, method_name) pairs visited in the
        # chain. v1.5: methods may differ, so the impl-set check has to
        # cover every method seen so far. Re-checking impls per method (vs
        # over the union) is equivalent because each method's impl set is
        # independent of the others — but tracking the set explicitly makes
        # the soundness rule readable and matches the brief.
        chain_methods = {(instr.trait_name, instr.method_name)}

        # Try to extend the chain forward through `flat`.
        run_indices = [i]
        j = i + 1
        while j < n:
            # Inter-dispatch instructions: from flat[run_indices[-1]+1] up
            # to flat[j-1] (inclusive). Verify they are DBR-independent.
            if not _between_is_dbr_independent(
                flat, run_indices[-1], j, call_graph, func_map,
                cyclic_funcs, func=func,
            ):
                break

            cand = flat[j][2]
            if not _is_chainable_far_dispatch(cand):
                break
            # v1.6: pass func so _same_self_chain can consult the
            # cast-transparency walker; otherwise it falls back to
            # plain vreg-identity equality.
            if not _same_self_chain(instr, cand, func):
                break
            # Re-check the candidate's method's impls. If we've already
            # accepted this (trait, method) earlier in the chain, the
            # check is redundant but cheap (memoized via
            # `_chain_dbr_independent`).
            if not _trait_method_impls_all_independent(
                cand, call_graph, func_map, cyclic_funcs
            ):
                break

            chain_methods.add((cand.trait_name, cand.method_name))
            run_indices.append(j)
            j += 1

        # Assign roles
        if len(run_indices) >= 2:
            for k, run_idx in enumerate(run_indices):
                td = flat[run_idx][2]
                if k == 0:
                    td.self_chain_role = ChainRole.START
                elif k == len(run_indices) - 1:
                    td.self_chain_role = ChainRole.END
                else:
                    td.self_chain_role = ChainRole.MIDDLE
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


def _same_self_chain(a, b, func=None):
    """True if two TraitDispatches can chain DBR-bracket-wise.

    v1.6: chain across different traits via cast-transparency. Two
    dispatches share a self if walking back through trivial trait-
    pointer-to-trait-pointer casts (Move and zero-shift TypeConvert)
    leads to the same root vreg. The trait_name no longer needs to
    match — different trait pointer aliases of the same underlying
    object share bank+address, and the DBR bracket / Y reload work
    identically for both.

    Bank-compatibility is enforced via the cast walker
    (`_chain_self_root`): only same-bank, same-size casts are followed.
    Casts that change far/near or that involve pointer arithmetic
    (e.g. ``&obj.weapon as *Trait``) terminate the walk early — they
    yield distinct roots and therefore distinct chains.

    `func` is the enclosing MIRFunction; if None, we fall back to the
    pre-v1.6 vreg-identity comparison.
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

    # Fallback: identity-based (pre-v1.6 behavior).
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


def _between_is_dbr_independent(flat, prev_run_idx, next_idx,
                                call_graph, func_map, cyclic_funcs,
                                func=None):
    """Check that all instructions strictly between flat[prev_run_idx]
    (the previous TraitDispatch in the run) and flat[next_idx] (the next
    candidate) preserve DBR-independence.

    The previous self_ptr vreg must NOT be redefined in this range —
    otherwise the value Y reads from on the next dispatch differs from
    what's in DBR.

    v1.6: when `func` is supplied, also reject redefinition of the
    chain's *root* self vreg (the address-holder behind any
    cast-transparency aliases).
    """
    prev_td = flat[prev_run_idx][2]
    self_vreg = prev_td.self_ptr
    root_vreg = None
    if func is not None and isinstance(self_vreg, VirtualRegister):
        root_vreg = _chain_self_root(self_vreg, func)
    for k in range(prev_run_idx + 1, next_idx):
        instr = flat[k][2]
        if not _instr_is_dbr_independent(
            instr, call_graph, func_map, cyclic_funcs
        ):
            return False
        # Reject if self_ptr vreg is redefined.
        if _instr_redefines_vreg(instr, self_vreg):
            return False
        # Also reject if the cast-transparency root is redefined.
        # Defining a fresh trivial-cast alias of the root is safe
        # (it's a Move/TypeConvert, dest is a different vreg), so
        # this only fires when the root itself is rewritten.
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


def _instr_is_dbr_independent(instr, call_graph, func_map, cyclic_funcs):
    """True if `instr` is safe to execute while DBR is set to self's bank
    (rather than the caller's DBR).

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
            if st == 'ram':
                # RAM access uses DBR-relative absolute by default. ROM/HW
                # are emitted as LONG already; zeropage uses DP. RAM is
                # the disqualifier.
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


def _function_is_dbr_independent(func_name, call_graph, func_map, cyclic_funcs):
    """Recursively determine whether `func_name`'s body executes DBR-
    independently. Memoized on the MIRFunction.

    Conservative rejections:
      - Function not found in func_map (external/builtin) -> False
      - Function in a recursive cycle -> False (without fixpoint we can't tell)
      - Function uses SET_DBR for its own far-ptr stack params -> False
        (its body assumes its OWN DBR; running it under our DBR is unsafe)
      - Function has D=S far-ptr stack params -> conservatively False
        (its prologue PHD/TSC/TCD changes D; complicated to reason about)
    """
    func = func_map.get(func_name)
    if func is None:
        return False  # external — can't analyze

    cached = getattr(func, _DBR_INDEP_ATTR, None)
    if cached is not None:
        return cached

    if func_name in cyclic_funcs:
        setattr(func, _DBR_INDEP_ATTR, False)
        return False

    # Refuse functions that play their own DBR/D games.
    if func.has_far_ptr_stack_params:
        setattr(func, _DBR_INDEP_ATTR, False)
        return False
    if func.self_far_uses_d_equals_s:
        setattr(func, _DBR_INDEP_ATTR, False)
        return False

    # Mark as True optimistically to break self-loops via memoization
    # (cyclic_funcs already filters real cycles, this guards re-entry on
    # same node within a single resolution).
    setattr(func, _DBR_INDEP_ATTR, True)

    for block in func.blocks.values():
        for instr in block.instructions:
            if not _instr_is_dbr_independent(
                instr, call_graph, func_map, cyclic_funcs
            ):
                setattr(func, _DBR_INDEP_ATTR, False)
                return False

    return True


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
