# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Loop register promotion analysis.

Pre-pass that runs before codegen to promote stack parameters used in loops
to local virtual registers with hardware register hints. This avoids repeated
stack-relative addressing (LDA d,S / STA d,S) in hot loops by keeping values
in hardware registers (X, Y) or scratch locations.

A stack parameter is promoted if:
1. It is used (read or written) inside any loop body
2. It is NOT a pointer type (pointers need memory locations for indirect addressing)
"""

from typing import Dict, Set, List, Tuple, Optional
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, VirtualRegister, HardwareRegister, Immediate, Move, BinaryOp, UnaryOp, Compare, Store, StoreIndirect, LoadIndirect,
    Call, TraitDispatch, CondBranch, Jump, TypeConvert, ToBool,
    Rotate,
    JumpTable, LookupTable, iter_operands, map_operands,
    BlockCopy, MemoryFill, AggregateCopy, AGGREGATE_COPY_UNROLL_MAX,
)
from r65.compiler.hir.types import PointerTypeInfo
from r65.compiler.codegen.type_utils import get_type_size


def analyze_loop_promotion(mir_program: MIRProgram):
    """
    Analyze and promote loop variables to hardware registers.

    Mutates mir_program in place:
    - Stack parameters used in loops: creates new local vregs with register
      hints, inserts Move instructions at function entry
    - Local loop counters: sets register hints on existing vregs so the
      register allocator places them in X/Y instead of stack slots

    Args:
        mir_program: MIR program to analyze (mutated in place)
    """
    total_param_promoted = 0
    total_local_promoted = 0

    for func in mir_program.functions:
        promoted = _analyze_function(func)
        total_param_promoted += promoted

    for func in mir_program.functions:
        promoted = _promote_local_loop_counters(func)
        total_local_promoted += promoted

    # Safety: drop/relocate loop-counter hints whose loop body clobbers the index
    # register — jump/lookup-table dispatch (X), or MVN block moves / fills (X and
    # Y). Covers both promotion above and the HIR depth-1/2 for-loop X/Y hints.
    for func in mir_program.functions:
        _enforce_index_clobber_safety(func)

    # Eliminate temp copies in ALL functions with loops, even if no
    # promotion happened. This collapses `%T = %V + 1; %V = Move %T`
    # into `%V = %V + 1`, enabling INX/INY/DEX/DEY pattern matching
    # in codegen (avoids expensive REP/TXA/ADC/TAX sequences).
    for func in mir_program.functions:
        if _find_loops(func):
            _eliminate_temp_copies(func)

    if total_param_promoted > 0:
        print(f"Loop register promotion: {total_param_promoted} parameter(s) promoted")
    if total_local_promoted > 0:
        print(f"Loop register promotion: {total_local_promoted} local(s) promoted")


def _analyze_function(func: MIRFunction) -> int:
    """
    Analyze a single function for loop register promotion opportunities.

    Returns:
        Number of parameters promoted
    """
    if not func.stack_param_offsets:
        return 0  # No stack parameters

    # Step 1: Find loops via back-edges
    loops = _find_loops(func)
    if not loops:
        return 0  # No loops

    # Step 2: Collect all block IDs that are part of any loop
    loop_blocks: Set[int] = set()
    for header, body in loops:
        loop_blocks.update(body)

    # Step 3: Find stack param vregs used in loop blocks
    loop_used_vregs = _find_vregs_used_in_blocks(func, loop_blocks)

    # Step 4: Filter to eligible stack param vregs
    eligible: List[Tuple[int, VirtualRegister]] = []  # (param_idx, vreg)
    for param_idx in func.stack_param_offsets:
        vreg = func.param_to_vreg.get(param_idx)
        if vreg is None:
            continue
        if vreg.id not in loop_used_vregs:
            continue
        # Skip pointer types (need memory location for indirect addressing)
        param_type = func.parameters[param_idx].param_type
        if isinstance(param_type, PointerTypeInfo):
            continue
        eligible.append((param_idx, vreg))

    if not eligible:
        return 0

    # Step 5: Detect loop counters (decremented/incremented in loop body)
    loop_counter_vregs = _find_loop_counters(func, loop_blocks)

    # Step 6: Assign register hints and create replacement vregs
    replacements: Dict[int, VirtualRegister] = {}  # old_vreg_id -> new_vreg
    used_hints: Set[str] = set()

    for param_idx, vreg in eligible:
        param_size = get_type_size(func.parameters[param_idx].param_type)
        hint = None

        if param_size == 2:  # u16 params get X/Y hints
            if vreg.id in loop_counter_vregs:
                # Loop counters prefer Y first, then X
                if 'Y' not in used_hints:
                    hint = 'Y'
                    used_hints.add('Y')
                elif 'X' not in used_hints:
                    hint = 'X'
                    used_hints.add('X')
            else:
                # Non-counter u16 params: X first, then Y
                if 'X' not in used_hints:
                    hint = 'X'
                    used_hints.add('X')
                elif 'Y' not in used_hints:
                    hint = 'Y'
                    used_hints.add('Y')

        # Skip params that didn't get a register hint.
        # A u8 param without a hint ends up in a frame slot.
        # LDA frame_slot,S costs the same as LDA original_param,S.
        # Promoting it just creates an unnecessary local + frame + copy.
        if hint is None:
            continue

        # Create new local vreg
        new_vreg = func.vreg_allocator.alloc(
            type_info=vreg.type_info,
            hint=f"loop_promo_{vreg.hint or vreg.id}",
            register_hint=hint,
        )
        replacements[vreg.id] = new_vreg

        # Record the hw vreg mapping for pre-allocation in codegen
        func.loop_promoted_hw_vregs[hint] = new_vreg

    if not replacements:
        return 0

    # Step 7: Replace all references to old vregs throughout the function
    _replace_vregs(func, replacements)

    # Step 7b: Eliminate temporary copies created by compound assignments
    # count-- generates: %T = %V - 1; %V = Move %T
    # Collapse to:       %V = %V - 1  (enables DEY/DEX pattern matching)
    _eliminate_temp_copies(func)

    # Step 8: Insert Move instructions at entry block start
    # After replacement, the original param vregs are only referenced in
    # the new Move instructions we're about to insert
    entry_block = func.blocks[func.entry_block_id]
    moves_to_insert = []
    for param_idx, vreg in eligible:
        if vreg.id in replacements:
            new_vreg = replacements[vreg.id]
            move = Move(
                dest=new_vreg,
                source=vreg,  # Original param vreg (still pre-allocated to stack)
                type_info=vreg.type_info,
            )
            moves_to_insert.append(move)

    # Insert at the beginning of the entry block
    entry_block.instructions = moves_to_insert + entry_block.instructions

    return len(replacements)


def _promote_local_loop_counters(func: MIRFunction) -> int:
    """
    Promote local loop counter variables to hardware registers (X/Y).

    Unlike stack parameter promotion, this doesn't create new vregs or
    insert entry Moves — it just sets register_hint on existing vregs
    so the register allocator places them in X/Y instead of stack slots.

    A local vreg is eligible if:
    1. It is at most 2 bytes (u8 or u16 — X/Y are always 16-bit, so a u8
       counter rides in the low byte)
    2. It is a loop counter (incremented/decremented by 1 in loop body)
    3. It is NOT a parameter vreg (those are handled by _analyze_function)
    4. It is NOT already promoted

    Returns:
        Number of locals promoted
    """
    # Step 1: Find loops
    loops = _find_loops(func)
    if not loops:
        return 0

    # Step 2: Collect loop blocks
    loop_blocks: Set[int] = set()
    for header, body in loops:
        loop_blocks.update(body)

    # Step 3: Find loop counter vregs (checks both in-place and temp patterns)
    loop_counter_vregs = _find_loop_counters(func, loop_blocks)
    if not loop_counter_vregs:
        return 0

    # Step 3b: Eliminate temp copies BEFORE collecting uses.
    # MIR lowers `len++` as `%T = %len + 1; %len = Move %T`.
    # Without elimination, _uses_compatible_with_hw sees BinaryOp with
    # dest=%T (not %len) and rejects it. After elimination, the pattern
    # becomes `%len = %len + 1` which the compatibility check accepts.
    _eliminate_temp_copies(func)

    # Step 4: Build vreg_id -> VirtualRegister map from all instructions
    all_vregs: Dict[int, VirtualRegister] = {}
    for block in func.blocks.values():
        for instr in block.instructions:
            for vreg in _get_vregs_from_instr(instr):
                all_vregs[vreg.id] = vreg

    # Step 5: Collect all uses per vreg (for compatibility check)
    vreg_uses: Dict[int, List] = {}
    for block in func.blocks.values():
        for instr in block.instructions:
            for vreg in _get_vregs_from_instr(instr):
                if vreg.id not in vreg_uses:
                    vreg_uses[vreg.id] = []
                vreg_uses[vreg.id].append(instr)

    # Step 6: Filter to eligible local vregs
    param_vreg_ids = {v.id for v in func.param_to_vreg.values() if v is not None}
    already_promoted = {v.id for v in func.loop_promoted_hw_vregs.values()}
    used_hints = set(func.loop_promoted_hw_vregs.keys())

    promoted = 0
    for vreg_id in sorted(loop_counter_vregs):
        if vreg_id in param_vreg_ids:
            continue
        if vreg_id in already_promoted:
            continue
        vreg = all_vregs.get(vreg_id)
        if vreg is None:
            continue
        if vreg.type_info is None:
            continue
        vreg_size = get_type_size(vreg.type_info)
        if vreg_size > 2:
            continue
        # Check that all uses are compatible with hardware register allocation.
        # Store/StoreIndirect with our vreg as source needs LDA which can't
        # resolve a hardware register as memory operand. Call arguments may
        # also be incompatible depending on mechanism.
        if not _uses_compatible_with_hw(vreg_id, vreg_uses.get(vreg_id, [])):
            # Check if the only blocker is a Compare against a hw-promoted
            # parameter. Local counters are more valuable in hw registers
            # (used for [dp],Y indexing and INX/INY) than parameters that
            # are only used for comparison. Demoting the parameter lets the
            # counter use Y and the comparison becomes CPY $dp (parameter
            # stays on stack at a DP-relative offset when D=S).
            conflict = _find_compare_hw_conflict(vreg_id, vreg_uses.get(vreg_id, []))
            if conflict:
                _demote_hw_vreg(func, conflict, used_hints)
                # Re-check after demotion
                if not _uses_compatible_with_hw(vreg_id, vreg_uses.get(vreg_id, [])):
                    continue
            else:
                continue

        # Choose register: prefer X when vreg is used as array index
        # (Move to HW X for indexed addressing), otherwise prefer Y.
        prefers_x = _prefers_x_register(vreg_id, vreg_uses.get(vreg_id, []))
        if prefers_x and 'X' not in used_hints:
            hint = 'X'
        elif 'Y' not in used_hints:
            hint = 'Y'
        elif 'X' not in used_hints:
            hint = 'X'
        else:
            continue

        used_hints.add(hint)
        vreg.register_hint = hint
        func.loop_promoted_hw_vregs[hint] = vreg
        promoted += 1

    # Step 7: Register HIR-hinted loop counters that weren't promoted above.
    # The HIR builder sets register_hint='X'/'Y' on for-loop counter vregs, but
    # some do not reach the promotion above (rejected by the compatibility
    # check, or no register left). Register those with existing hints in
    # loop_promoted_hw_vregs so function_gen pre-allocates them to hardware
    # instead of creating unnecessary stack frame slots.
    for vreg_id in sorted(loop_counter_vregs):
        if vreg_id in param_vreg_ids:
            continue
        vreg = all_vregs.get(vreg_id)
        if vreg is None:
            continue
        hint = vreg.register_hint
        if hint in ('X', 'Y') and hint not in used_hints:
            if vreg.id not in {v.id for v in func.loop_promoted_hw_vregs.values()}:
                func.loop_promoted_hw_vregs[hint] = vreg
                used_hints.add(hint)
                promoted += 1

    if promoted > 0:
        _eliminate_temp_copies(func)

    return promoted


def _enforce_index_clobber_safety(func: MIRFunction) -> int:
    """
    Some instructions clobber index registers across a loop body, which would
    corrupt a loop counter pinned to that register (via the depth-1 for-loop hint
    or loop promotion):

    - JumpTable / LookupTable use ``TAX`` to index the dispatch table — they
      clobber **X only**. Y is untouched and still supports ``addr,Y`` indexing,
      so an unsafe X hint can be relocated to Y (when free) or dropped to a stack
      slot.
    - MVN block moves (BlockCopy, large AggregateCopy clones) and MemoryFill loops
      clobber **both X and Y**. A counter pinned to either must be dropped — there
      is no safe index register. (Small AggregateCopy clones unroll to A-only
      moves and are safe; see AGGREGATE_COPY_UNROLL_MAX.)

    Deliberately NOT covered: a StoreIndirect using ``(zp),Y`` addressing also
    needs Y, but only for the duration of that one store, and often the index it
    loads *is* the promoted counter (``PTR[i] = i``) — in which case pinning the
    counter to Y is exactly right. Blanket-demoting on the presence of an
    indexed indirect store would give that case up. The contention is instead
    left to the register allocator, which declines the hint when the index is a
    different value; the hint set here is advisory, not binding. Both outcomes
    are pinned by TestCounterAsIndirectStoreSource
    (tests/compiler/codegen/test_loop_promotion.py) and its emulator counterpart.

    Returns the number of hints rewritten.
    """
    loops = _find_loops(func)
    if not loops:
        return 0

    def clobbers_xy(instr) -> bool:
        if isinstance(instr, (BlockCopy, MemoryFill)):
            return True
        if isinstance(instr, AggregateCopy):
            return instr.count > AGGREGATE_COPY_UNROLL_MAX
        return False

    x_only_blocks: Set[int] = set()   # tables: X clobbered, Y safe
    xy_blocks: Set[int] = set()       # block moves / fills: X and Y clobbered
    for _header, body in loops:
        instrs = [instr for bid in body for instr in func.blocks[bid].instructions]
        if any(clobbers_xy(instr) for instr in instrs):
            xy_blocks |= body          # subsumes any table in the same loop
        elif any(isinstance(instr, (JumpTable, LookupTable)) for instr in instrs):
            x_only_blocks |= body
    if not x_only_blocks and not xy_blocks:
        return 0

    all_vregs: Dict[int, VirtualRegister] = {}
    for block in func.blocks.values():
        for instr in block.instructions:
            for vreg in _get_vregs_from_instr(instr):
                all_vregs[vreg.id] = vreg

    rewritten = 0

    # X+Y clobbered: no safe index register, so drop any X or Y hint used here.
    for vid in sorted(_find_vregs_used_in_blocks(func, xy_blocks)):
        vreg = all_vregs.get(vid)
        if vreg is None or vreg.register_hint not in ('X', 'Y'):
            continue
        reg = vreg.register_hint
        if func.loop_promoted_hw_vregs.get(reg) is vreg:
            del func.loop_promoted_hw_vregs[reg]
        vreg.register_hint = None
        rewritten += 1

    # X-only clobbered: relocate X->Y when free, else drop. Only relocate when no
    # block move exists anywhere in the function (otherwise Y may be clobbered by
    # one). Vregs already cleared above are skipped (hint no longer 'X').
    y_claimed = any(v.register_hint == 'Y' for v in all_vregs.values())
    can_use_y = not xy_blocks
    for vid in sorted(_find_vregs_used_in_blocks(func, x_only_blocks)):
        vreg = all_vregs.get(vid)
        if vreg is None or vreg.register_hint != 'X':
            continue
        if func.loop_promoted_hw_vregs.get('X') is vreg:
            del func.loop_promoted_hw_vregs['X']
        if can_use_y and not y_claimed:
            vreg.register_hint = 'Y'
            y_claimed = True
            func.loop_promoted_hw_vregs['Y'] = vreg
        else:
            vreg.register_hint = None
        rewritten += 1
    return rewritten


def _prefers_x_register(vreg_id: int, uses: list) -> bool:
    """
    Check if a loop counter vreg is used as an array index.

    When the counter is Moved to hardware register X for indexed addressing
    (LDA addr,X), placing it in X directly eliminates the TYX transfer.
    """
    for use in uses:
        if isinstance(use, Move):
            if (isinstance(use.source, VirtualRegister) and use.source.id == vreg_id and
                    isinstance(use.dest, HardwareRegister) and use.dest.name == 'X'):
                return True
    return False


def _uses_compatible_with_hw(vreg_id: int, uses: list) -> bool:
    """
    Check if all uses of a vreg are compatible with hardware register allocation.

    A vreg in X/Y can be used in:
    - Move (source or dest) — emits TXA/TAX, LDX, etc.
    - BinaryOp with +1/-1 where vreg is left operand — emits INX/INY/DEX/DEY
    - Compare (as left or right) — emits CPX/CPY
    - Return (as return value) — emits TXA/TYA
    - Store / StoreIndirect (as source) — both store paths test for a hardware
      source before resolving anything as a memory operand, and route it
      through A (_emit_store_from_reg, select_store_indirect).
    - CondBranch (as condition) — okay
    - Jump — okay

    Incompatible uses (codegen tries to resolve hw register as memory operand):
    - BinaryOp with operations other than +1/-1 (shift, XOR, OR, etc.)
    - BinaryOp where vreg is right operand (needs memory operand for CMP/SBC/etc.)
    - UnaryOp — needs value in A
    - LoadIndirect (as dest) — result comes from A, not X/Y
    - Call/TraitDispatch (as argument value) — may need memory operand
    - TypeConvert, ToBool, Rotate — need value in A
    """
    for use in uses:
        if isinstance(use, BinaryOp):
            # Only allow +1/-1 where our vreg is the left operand AND dest
            # (this is the in-place increment/decrement that maps to INX/INY/DEX/DEY)
            if isinstance(use.left, VirtualRegister) and use.left.id == vreg_id:
                is_inc_dec = (
                    use.op in ('+', '-') and
                    isinstance(use.right, Immediate) and
                    use.right.value == 1 and
                    isinstance(use.dest, VirtualRegister) and
                    use.dest.id == vreg_id
                )
                if not is_inc_dec:
                    return False
            # Vreg as right operand of BinaryOp — needs memory operand resolution
            if isinstance(use.right, VirtualRegister) and use.right.id == vreg_id:
                return False
        elif isinstance(use, UnaryOp):
            # UnaryOp needs value in A — incompatible
            if isinstance(use.operand, VirtualRegister) and use.operand.id == vreg_id:
                return False
        elif isinstance(use, Compare):
            # Compare: our vreg as LEFT can use CPX/CPY if right is
            # Immediate or a non-hw vreg (memory operand). But if right
            # is also a hw-promoted vreg, neither can be resolved as
            # memory operand → reject.
            # Our vreg as RIGHT: codegen resolves right as memory operand
            # for CMP/CPX/CPY → reject.
            is_left = isinstance(use.left, VirtualRegister) and use.left.id == vreg_id
            is_right = isinstance(use.right, VirtualRegister) and use.right.id == vreg_id
            if is_right:
                return False
            if is_left:
                # Right operand must be resolvable as memory operand
                if isinstance(use.right, VirtualRegister) and use.right.register_hint:
                    # Other operand is also hw-promoted → can't resolve as memory
                    return False
        elif isinstance(use, LoadIndirect):
            # LoadIndirect dest gets the result in A, not compatible with X/Y dest
            if isinstance(use.dest, VirtualRegister) and use.dest.id == vreg_id:
                return False
        elif isinstance(use, (Call, TraitDispatch)):
            for arg in use.args:
                if isinstance(arg.value, VirtualRegister) and arg.value.id == vreg_id:
                    return False
        elif isinstance(use, (TypeConvert, ToBool, Rotate)):
            # These need value in A
            if hasattr(use, 'source') and isinstance(use.source, VirtualRegister) and use.source.id == vreg_id:
                return False
            # The result is written with STA (memory operand), so a
            # hw-register dest can't be resolved either.
            if hasattr(use, 'dest') and isinstance(use.dest, VirtualRegister) and use.dest.id == vreg_id:
                return False
    return True


def _find_compare_hw_conflict(vreg_id: int, uses: list) -> Optional[VirtualRegister]:
    """
    Find a hw-promoted vreg that conflicts with promoting vreg_id.

    Returns the conflicting VirtualRegister if the ONLY reason
    _uses_compatible_with_hw fails is a Compare where the other operand
    has a register_hint (hw-promoted parameter).
    """
    conflict = None
    for use in uses:
        if isinstance(use, Compare):
            is_left = isinstance(use.left, VirtualRegister) and use.left.id == vreg_id
            if is_left and isinstance(use.right, VirtualRegister) and use.right.register_hint:
                conflict = use.right
    return conflict


def _demote_hw_vreg(func: MIRFunction, vreg: VirtualRegister, used_hints: set):
    """
    Demote a hw-promoted vreg back to a stack allocation.

    Fully reverses the promotion: clears the register_hint, removes
    the entry-block Move, and replaces all uses of the promoted vreg
    back to the original param vreg. This avoids allocating a separate
    stack slot for the promoted vreg (which would just be a redundant
    copy of the parameter).
    """
    old_hint = vreg.register_hint
    vreg.register_hint = None
    if old_hint and old_hint in func.loop_promoted_hw_vregs:
        if func.loop_promoted_hw_vregs[old_hint].id == vreg.id:
            del func.loop_promoted_hw_vregs[old_hint]
    if old_hint:
        used_hints.discard(old_hint)

    # Find the entry-block Move that copies from the original param vreg
    # into this promoted vreg, and reverse the promotion entirely.
    entry_block = func.blocks[func.entry_block_id]
    original_param = None
    move_idx = None
    for i, instr in enumerate(entry_block.instructions):
        if (isinstance(instr, Move) and
                isinstance(instr.dest, VirtualRegister) and
                instr.dest.id == vreg.id and
                isinstance(instr.source, VirtualRegister)):
            original_param = instr.source
            move_idx = i
            break

    if original_param is not None:
        # Remove the Move instruction
        entry_block.instructions.pop(move_idx)
        # Replace all uses of the promoted vreg back to the original param
        _replace_vregs(func, {vreg.id: original_param})


def _find_loops(func: MIRFunction) -> List[Tuple[int, Set[int]]]:
    """
    Find natural loops via back-edge detection.

    A back-edge is a Jump where target block_id <= source block_id.

    Returns:
        List of (header_block_id, set of block IDs in loop body)
    """
    loops = []

    for block_id, block in func.blocks.items():
        if not block.instructions:
            continue

        # Check terminator(s) for back-edges
        for instr in block.instructions:
            target = None
            if isinstance(instr, Jump):
                target = instr.target
            elif isinstance(instr, CondBranch):
                # Check both targets
                if instr.true_target <= block_id:
                    target = instr.true_target
                if instr.false_target <= block_id:
                    # Also check false target for back-edge
                    body = _find_loop_body(func, instr.false_target, block_id)
                    if body:
                        loops.append((instr.false_target, body))
                    if target is None:
                        continue

            if target is not None and target <= block_id:
                body = _find_loop_body(func, target, block_id)
                if body:
                    loops.append((target, body))

    return loops


def _find_loop_body(func: MIRFunction, header: int, back_edge_source: int) -> Set[int]:
    """
    Find the natural loop body using reverse walk from back-edge source to header.

    Returns:
        Set of block IDs in the loop body (including header)
    """
    body = {header}
    if header == back_edge_source:
        return body  # Single-block loop

    # Reverse walk from back_edge_source to header
    worklist = [back_edge_source]
    body.add(back_edge_source)

    while worklist:
        block_id = worklist.pop()
        block = func.blocks.get(block_id)
        if block is None:
            continue
        for pred_id in block.predecessors:
            if pred_id not in body:
                body.add(pred_id)
                worklist.append(pred_id)

    return body


def _find_vregs_used_in_blocks(func: MIRFunction, block_ids: Set[int]) -> Set[int]:
    """
    Find all VirtualRegister IDs used (read or written) in the given blocks.

    Returns:
        Set of vreg IDs
    """
    used = set()

    for block_id in block_ids:
        block = func.blocks.get(block_id)
        if block is None:
            continue
        for instr in block.instructions:
            for vreg in _get_vregs_from_instr(instr):
                used.add(vreg.id)

    return used


def _iter_vregs(instr):
    """Yield every VirtualRegister operand of instr (read and write positions)."""
    for _, v in iter_operands(instr):
        if isinstance(v, VirtualRegister):
            yield v


def _map_vregs(instr, fn):
    """Replace every VirtualRegister-bearing operand slot in place.

    fn passes non-register values through unchanged (_replace_vreg is a no-op on
    them). Operand coverage comes from the shared registry (mir/nodes.py), so it
    can never drift from liveness / the inliner.
    """
    map_operands(instr, fn)


def _get_vregs_from_instr(instr) -> List[VirtualRegister]:
    """Extract all VirtualRegister references from an instruction."""
    return list(_iter_vregs(instr))


def _find_loop_counters(func: MIRFunction, loop_blocks: Set[int]) -> Set[int]:
    """
    Find vreg IDs that are loop counters (incremented/decremented by 1 in loop body).

    Patterns:
    1. BinaryOp(dest=V, left=V, op='-'|'+', right=Immediate(1))  — in-place
    2. BinaryOp(dest=T, left=V, op='-'|'+', right=Immediate(1))  — via temporary
       followed by Move(dest=V, source=T)
    """
    counters = set()

    for block_id in loop_blocks:
        block = func.blocks.get(block_id)
        if block is None:
            continue
        instrs = block.instructions
        for i, instr in enumerate(instrs):
            if isinstance(instr, BinaryOp) and instr.op in ('+', '-'):
                if not (isinstance(instr.left, VirtualRegister) and
                        isinstance(instr.right, Immediate) and
                        instr.right.value == 1 and
                        isinstance(instr.dest, VirtualRegister)):
                    continue

                if instr.dest.id == instr.left.id:
                    # Pattern 1: in-place update
                    counters.add(instr.dest.id)
                else:
                    # Pattern 2: temp + copy back
                    # Check the next instruction for Move(dest=V, source=T)
                    if i + 1 < len(instrs):
                        next_instr = instrs[i + 1]
                        if (isinstance(next_instr, Move) and
                                isinstance(next_instr.dest, VirtualRegister) and
                                isinstance(next_instr.source, VirtualRegister) and
                                next_instr.source.id == instr.dest.id and
                                next_instr.dest.id == instr.left.id):
                            counters.add(instr.left.id)

    return counters


def _eliminate_temp_copies(func: MIRFunction):
    """
    Eliminate temporary copies from compound assignments (e.g., count--).

    MIR lowers `count--` as:
        %T = %V - 1       (BinaryOp with temporary dest)
        %V = Move %T       (copy back to original)

    This collapses them into:
        %V = %V - 1       (in-place, enables DEY/DEX pattern matching)

    Only applied when %T has no other uses in the entire function.
    """
    # Collect all vreg uses across the entire function
    all_vreg_uses: Dict[int, int] = {}  # vreg_id -> use count
    for block in func.blocks.values():
        for instr in block.instructions:
            for vreg in _get_vregs_from_instr(instr):
                all_vreg_uses[vreg.id] = all_vreg_uses.get(vreg.id, 0) + 1

    for block in func.blocks.values():
        instrs = block.instructions
        i = 0
        while i < len(instrs) - 1:
            instr = instrs[i]
            next_instr = instrs[i + 1]

            # Pattern: BinaryOp(dest=%T, left=%V, op, right) followed by Move(dest=%V, source=%T)
            if (isinstance(instr, BinaryOp) and
                    isinstance(next_instr, Move) and
                    isinstance(instr.dest, VirtualRegister) and
                    isinstance(instr.left, VirtualRegister) and
                    isinstance(next_instr.dest, VirtualRegister) and
                    isinstance(next_instr.source, VirtualRegister) and
                    instr.dest.id != instr.left.id and
                    next_instr.source.id == instr.dest.id and
                    next_instr.dest.id == instr.left.id):

                temp_id = instr.dest.id
                # %T appears exactly twice: as BinaryOp dest and Move source
                if all_vreg_uses.get(temp_id, 0) == 2:
                    # Collapse: BinaryOp dest=%V, left=%V (in-place)
                    instr.dest = next_instr.dest  # %V
                    # Delete the Move
                    instrs.pop(i + 1)
                    # Update use counts
                    all_vreg_uses[temp_id] = 0
                    continue  # Re-check same index in case of consecutive patterns

            i += 1


def _replace_vregs(func: MIRFunction, replacements: Dict[int, VirtualRegister]):
    """
    Replace all references to old vregs with new vregs throughout the function.

    Args:
        func: MIR function to modify
        replacements: Map of old_vreg_id -> new VirtualRegister
    """
    for block in func.blocks.values():
        for instr in block.instructions:
            _replace_in_instr(instr, replacements)


def _replace_vreg(val, replacements: Dict[int, VirtualRegister]):
    """Replace a VirtualRegister if it's in the replacements map."""
    if isinstance(val, VirtualRegister) and val.id in replacements:
        return replacements[val.id]
    return val


def _replace_in_instr(instr, replacements: Dict[int, VirtualRegister]):
    """Replace VirtualRegister references in a single instruction."""
    _map_vregs(instr, lambda v: _replace_vreg(v, replacements))
