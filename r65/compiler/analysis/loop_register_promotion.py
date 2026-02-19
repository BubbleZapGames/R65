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
    MIRProgram, MIRFunction, BasicBlock,
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Move, BinaryOp, UnaryOp, Compare, Store, StoreIndirect, Load, LoadIndirect,
    Return, Call, TraitDispatch, CondBranch, Jump, TypeConvert, ToBool,
    SaveRegister, RestoreRegister, Argument, BitTest, Rotate,
    JumpTable, LookupTable, StatusFlagRead, InlineAsm,
)
from r65.compiler.hir.types import PointerTypeInfo
from r65.compiler.codegen.type_utils import get_type_size


def analyze_loop_promotion(mir_program: MIRProgram):
    """
    Analyze and promote stack parameters used in loops to local vregs.

    Mutates mir_program in place:
    - Creates new local vregs with register hints for promoted params
    - Inserts Move instructions at function entry to copy param -> local
    - Replaces all references to param vregs with local vregs

    Args:
        mir_program: MIR program to analyze (mutated in place)
    """
    total_promoted = 0

    for func in mir_program.functions:
        promoted = _analyze_function(func)
        total_promoted += promoted

    if total_promoted > 0:
        print(f"Loop register promotion: {total_promoted} parameter(s) promoted")


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


def _get_vregs_from_instr(instr) -> List[VirtualRegister]:
    """Extract all VirtualRegister references from an instruction."""
    vregs = []

    def _collect(val):
        if isinstance(val, VirtualRegister):
            vregs.append(val)

    if isinstance(instr, Move):
        _collect(instr.dest)
        _collect(instr.source)
    elif isinstance(instr, BinaryOp):
        _collect(instr.dest)
        _collect(instr.left)
        _collect(instr.right)
    elif isinstance(instr, UnaryOp):
        _collect(instr.dest)
        _collect(instr.operand)
    elif isinstance(instr, Compare):
        _collect(instr.left)
        _collect(instr.right)
    elif isinstance(instr, Store):
        _collect(instr.source)
    elif isinstance(instr, StoreIndirect):
        _collect(instr.source)
        _collect(instr.pointer)
    elif isinstance(instr, Load):
        _collect(instr.dest)
    elif isinstance(instr, LoadIndirect):
        _collect(instr.dest)
        _collect(instr.pointer)
    elif isinstance(instr, Return):
        for v in instr.values:
            _collect(v)
    elif isinstance(instr, (Call, TraitDispatch)):
        for ret in instr.returns:
            _collect(ret)
        for arg in instr.args:
            _collect(arg.value)
        if isinstance(instr, TraitDispatch) and instr.self_ptr:
            _collect(instr.self_ptr)
    elif isinstance(instr, CondBranch):
        _collect(instr.condition)
    elif isinstance(instr, TypeConvert):
        _collect(instr.dest)
        _collect(instr.source)
    elif isinstance(instr, ToBool):
        _collect(instr.dest)
        _collect(instr.source)
    elif isinstance(instr, SaveRegister):
        _collect(instr.save_location)
    elif isinstance(instr, RestoreRegister):
        _collect(instr.save_location)
    elif isinstance(instr, BitTest):
        _collect(instr.value)
    elif isinstance(instr, Rotate):
        _collect(instr.dest)
        _collect(instr.source)
    elif isinstance(instr, JumpTable):
        _collect(instr.scrutinee)
    elif isinstance(instr, LookupTable):
        _collect(instr.dest)
        _collect(instr.scrutinee)
    elif isinstance(instr, StatusFlagRead):
        _collect(instr.dest)

    return vregs


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
    if isinstance(instr, Move):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.source = _replace_vreg(instr.source, replacements)
    elif isinstance(instr, BinaryOp):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.left = _replace_vreg(instr.left, replacements)
        instr.right = _replace_vreg(instr.right, replacements)
    elif isinstance(instr, UnaryOp):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.operand = _replace_vreg(instr.operand, replacements)
    elif isinstance(instr, Compare):
        instr.left = _replace_vreg(instr.left, replacements)
        instr.right = _replace_vreg(instr.right, replacements)
    elif isinstance(instr, Store):
        instr.source = _replace_vreg(instr.source, replacements)
    elif isinstance(instr, StoreIndirect):
        instr.source = _replace_vreg(instr.source, replacements)
        instr.pointer = _replace_vreg(instr.pointer, replacements)
    elif isinstance(instr, Load):
        instr.dest = _replace_vreg(instr.dest, replacements)
    elif isinstance(instr, LoadIndirect):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.pointer = _replace_vreg(instr.pointer, replacements)
    elif isinstance(instr, Return):
        instr.values = [_replace_vreg(v, replacements) for v in instr.values]
    elif isinstance(instr, (Call, TraitDispatch)):
        instr.returns = [_replace_vreg(r, replacements) for r in instr.returns]
        for arg in instr.args:
            arg.value = _replace_vreg(arg.value, replacements)
        if isinstance(instr, TraitDispatch) and instr.self_ptr:
            instr.self_ptr = _replace_vreg(instr.self_ptr, replacements)
    elif isinstance(instr, CondBranch):
        instr.condition = _replace_vreg(instr.condition, replacements)
    elif isinstance(instr, TypeConvert):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.source = _replace_vreg(instr.source, replacements)
    elif isinstance(instr, ToBool):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.source = _replace_vreg(instr.source, replacements)
    elif isinstance(instr, SaveRegister):
        instr.save_location = _replace_vreg(instr.save_location, replacements)
    elif isinstance(instr, RestoreRegister):
        instr.save_location = _replace_vreg(instr.save_location, replacements)
    elif isinstance(instr, BitTest):
        instr.value = _replace_vreg(instr.value, replacements)
    elif isinstance(instr, Rotate):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.source = _replace_vreg(instr.source, replacements)
    elif isinstance(instr, JumpTable):
        instr.scrutinee = _replace_vreg(instr.scrutinee, replacements)
    elif isinstance(instr, LookupTable):
        instr.dest = _replace_vreg(instr.dest, replacements)
        instr.scrutinee = _replace_vreg(instr.scrutinee, replacements)
    elif isinstance(instr, StatusFlagRead):
        instr.dest = _replace_vreg(instr.dest, replacements)
