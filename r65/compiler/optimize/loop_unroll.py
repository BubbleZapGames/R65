# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Loop Unrolling Optimization.

Fully unrolls static for-loops with constant bounds when the unrolled
body fits within size limits. The loop index is replaced with constant
values in each copy, enabling downstream constant folding.

Constraints:
  - Loop must have a compile-time constant trip count
  - Body must have >= 4 and <= 20 MIR operations
  - Total unrolled size (trip_count * body_ops) < 255
  - No break, continue (exits to outside the loop), or return in body
  - No nested loops
"""

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union

from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock, MIRInstruction,
    VirtualRegister, HardwareRegister, Immediate,
    Move, BinaryOp, Compare, CondBranch, Jump, Return,
    ReturnFromInterrupt, Call, TraitDispatch,
    Load, Store, LoadIndirect, StoreIndirect,
    UnaryOp, TypeConvert, ToBool, Rotate, BitTest,
    InlineAsm, SetMode, Push, Pull, SaveRegister, RestoreRegister,
    JumpTable, LookupTable, MemoryFill, BlockCopy,
)

# Size thresholds
MIN_BODY_OPS = 4
MAX_BODY_OPS = 20
MAX_UNROLLED_OPS = 255


class LoopUnroller:
    """Fully unrolls static loops at the MIR level."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def unroll(self, mir_program: MIRProgram) -> int:
        """Unroll eligible loops in all functions. Returns number of loops unrolled."""
        total = 0
        for func in mir_program.functions:
            total += self._unroll_in_function(func)
        return total

    def _unroll_in_function(self, func: MIRFunction) -> int:
        """Find and unroll eligible loops in a single function."""
        loops = _find_loops(func)
        unrolled = 0

        for header_id, body_ids in loops:
            result = self._try_unroll_loop(func, header_id, body_ids)
            if result:
                unrolled += 1

        return unrolled

    def _try_unroll_loop(self, func: MIRFunction, header_id: int,
                         body_ids: Set[int]) -> bool:
        """Attempt to unroll a single loop. Returns True if unrolled."""
        header = func.blocks.get(header_id)
        if header is None:
            return False

        # --- Detect the canonical for-loop pattern ---
        loop_info = _detect_for_loop(func, header_id, body_ids)
        if loop_info is None:
            return False

        (counter_vreg, start_val, end_val, step_val,
         comparison, body_block_ids, exit_block_id, increment_block_id) = loop_info

        # Trip count
        if step_val <= 0:
            return False
        trip_count = (end_val - start_val + step_val - 1) // step_val
        if trip_count <= 0:
            return False

        # --- Check body size constraints ---
        body_ops = _count_body_ops(func, body_block_ids, counter_vreg)
        if body_ops < MIN_BODY_OPS or body_ops > MAX_BODY_OPS:
            return False
        if trip_count * body_ops >= MAX_UNROLLED_OPS:
            return False

        # --- Check body is safe to unroll (no break/continue/return/nested loops) ---
        if not _body_is_safe(func, body_block_ids, header_id, exit_block_id,
                             increment_block_id):
            return False

        # --- Perform the unroll ---
        self._do_unroll(func, header_id, body_block_ids, exit_block_id,
                        increment_block_id, counter_vreg, start_val, end_val,
                        step_val, trip_count)
        return True

    def _do_unroll(self, func: MIRFunction, header_id: int,
                   body_block_ids: List[int], exit_block_id: int,
                   increment_block_id: int, counter_vreg: VirtualRegister,
                   start_val: int, end_val: int, step_val: int,
                   trip_count: int):
        """Replace the loop with unrolled copies of the body."""
        # Collect blocks that feed into the header (the pre-header edges)
        header = func.blocks[header_id]
        pre_header_preds = [p for p in header.predecessors
                            if p not in body_block_ids
                            and p != header_id]

        # Ordered body blocks (header is NOT included; it's the condition check)
        # body_block_ids includes header, so strip it
        pure_body_ids = [bid for bid in body_block_ids
                         if bid != header_id]

        # Find the entry body block (the true_target of the header's CondBranch)
        header_term = header.instructions[-1]
        if isinstance(header_term, CondBranch):
            body_entry_id = header_term.true_target
        else:
            return  # shouldn't happen

        # Build ordered chain of body blocks (BFS from body_entry following
        # successors within body, excluding header back-edge)
        ordered_body = _order_body_blocks(func, body_entry_id, pure_body_ids,
                                          header_id)

        # Allocate a single landing block that replaces the entire loop
        next_bid = max(func.blocks.keys()) + 1

        # Build unrolled sequence: for each iteration, clone body blocks
        # with counter_vreg replaced by constant
        all_new_blocks = []
        prev_chain_exit = None  # block ID whose Jump→header needs rewriting

        for iteration in range(trip_count):
            const_val = start_val + iteration * step_val
            vreg_map: Dict[int, VirtualRegister] = {}

            # Allocate fresh vregs for this iteration
            for bid in ordered_body:
                block = func.blocks[bid]
                for instr in block.instructions:
                    for vreg in _all_vregs_defined(instr):
                        if vreg.id not in vreg_map and vreg.id != counter_vreg.id:
                            new_vreg = func.vreg_allocator.alloc(
                                type_info=vreg.type_info,
                                hint=f"{vreg.hint}_u{iteration}" if vreg.hint else None,
                                register_hint=vreg.register_hint,
                            )
                            vreg_map[vreg.id] = new_vreg

            # Clone blocks
            block_id_map: Dict[int, int] = {}
            iteration_blocks = []
            for bid in ordered_body:
                new_bid = next_bid
                next_bid += 1
                block_id_map[bid] = new_bid

                old_block = func.blocks[bid]
                new_block = BasicBlock(block_id=new_bid)

                for instr in old_block.instructions:
                    new_instr = deepcopy(instr)
                    # Replace counter vreg with constant
                    _replace_vreg_with_immediate(new_instr, counter_vreg,
                                                 Immediate(const_val))
                    # Replace other vregs with iteration copies
                    _remap_vregs(new_instr, vreg_map)
                    new_block.instructions.append(new_instr)

                iteration_blocks.append(new_block)

            # Remap intra-body block references (Jump/CondBranch targets)
            for new_block in iteration_blocks:
                _remap_block_targets(new_block, block_id_map)

            # The last block in the body Jumps back to header; redirect it to
            # the first block of the NEXT iteration (or exit for last iteration)
            last_body_block = iteration_blocks[-1]
            _remove_back_edge_jump(last_body_block, header_id)
            # Also remove the increment instruction (counter += step) since
            # counter is now a constant
            _remove_counter_increment(last_body_block, counter_vreg)

            # Register blocks in function
            for nb in iteration_blocks:
                func.blocks[nb.block_id] = nb

            all_new_blocks.append(iteration_blocks)

            # Chain: previous iteration's last block → this iteration's first
            if prev_chain_exit is not None:
                prev_block = func.blocks[prev_chain_exit]
                first_new = iteration_blocks[0]
                prev_block.instructions.append(Jump(target=first_new.block_id))
                prev_block.successors.append(first_new.block_id)
                first_new.predecessors.append(prev_chain_exit)

            prev_chain_exit = last_body_block.block_id

        # Wire the last unrolled block to the exit
        if prev_chain_exit is not None:
            last_block = func.blocks[prev_chain_exit]
            last_block.instructions.append(Jump(target=exit_block_id))
            last_block.successors.append(exit_block_id)
            exit_block = func.blocks[exit_block_id]
            if prev_chain_exit not in exit_block.predecessors:
                exit_block.predecessors.append(prev_chain_exit)

        # Redirect pre-header → first unrolled block (bypass header)
        first_unrolled_block = all_new_blocks[0][0] if all_new_blocks else None
        if first_unrolled_block is None:
            return

        for pred_id in pre_header_preds:
            pred_block = func.blocks.get(pred_id)
            if pred_block is None:
                continue
            _redirect_targets(pred_block, header_id, first_unrolled_block.block_id)
            if header_id in pred_block.successors:
                pred_block.successors.remove(header_id)
            if first_unrolled_block.block_id not in pred_block.successors:
                pred_block.successors.append(first_unrolled_block.block_id)
            if pred_id not in first_unrolled_block.predecessors:
                first_unrolled_block.predecessors.append(pred_id)

        # Remove old loop blocks from function
        # (header + body blocks including increment block)
        old_blocks_to_remove = set(body_block_ids) | {header_id}
        for old_bid in old_blocks_to_remove:
            if old_bid in func.blocks:
                # Clean up edges pointing to exit
                old_block = func.blocks[old_bid]
                for succ_id in old_block.successors:
                    if succ_id not in old_blocks_to_remove:
                        succ = func.blocks.get(succ_id)
                        if succ and old_bid in succ.predecessors:
                            succ.predecessors.remove(old_bid)
                del func.blocks[old_bid]

        # Remove old header from exit predecessors
        exit_block = func.blocks.get(exit_block_id)
        if exit_block and header_id in exit_block.predecessors:
            exit_block.predecessors.remove(header_id)


# ============================================================================
# Loop Detection (reused from loop_register_promotion.py)
# ============================================================================

def _find_loops(func: MIRFunction) -> List[Tuple[int, Set[int]]]:
    """Find natural loops via back-edge detection."""
    loops = []
    for block_id, block in func.blocks.items():
        if not block.instructions:
            continue
        for instr in block.instructions:
            target = None
            if isinstance(instr, Jump):
                target = instr.target
            elif isinstance(instr, CondBranch):
                if instr.true_target <= block_id:
                    target = instr.true_target
                if instr.false_target <= block_id:
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


def _find_loop_body(func: MIRFunction, header: int,
                    back_edge_source: int) -> Set[int]:
    """Find the natural loop body by reverse walk from back-edge source."""
    body = {header}
    if header == back_edge_source:
        return body
    worklist = [back_edge_source]
    body.add(back_edge_source)
    while worklist:
        bid = worklist.pop()
        block = func.blocks.get(bid)
        if block is None:
            continue
        for pred_id in block.predecessors:
            if pred_id not in body:
                body.add(pred_id)
                worklist.append(pred_id)
    return body


# ============================================================================
# For-Loop Pattern Detection
# ============================================================================

def _detect_for_loop(func: MIRFunction, header_id: int,
                     body_ids: Set[int]):
    """
    Detect a canonical for-loop pattern in the CFG.

    Expected pattern:
      pre-header: Move(counter_vreg, Immediate(start))
      header:     Compare(counter_vreg, Immediate(end), '<')
                  CondBranch(cond, body_entry, exit)
      body:       [user code]
      increment:  BinaryOp(counter_vreg, counter_vreg, '+', Immediate(step))
                  Jump(header)

    Returns tuple or None:
      (counter_vreg, start, end, step, comparison,
       body_block_ids, exit_block_id, increment_block_id)
    """
    header = func.blocks.get(header_id)
    if header is None or len(header.instructions) < 2:
        return None

    # Header must end with Compare + CondBranch
    term = header.instructions[-1]
    if not isinstance(term, CondBranch):
        return None

    # Find the Compare just before the CondBranch
    compare = None
    for instr in header.instructions:
        if isinstance(instr, Compare):
            compare = instr

    if compare is None:
        return None

    # Counter must be a VirtualRegister compared against an Immediate
    if not isinstance(compare.left, VirtualRegister):
        return None
    if not isinstance(compare.right, Immediate):
        return None
    if compare.comparison != '<':
        return None

    counter_vreg = compare.left
    end_val = compare.right.value
    exit_block_id = term.false_target
    body_entry_id = term.true_target

    # Exit must be outside the loop
    if exit_block_id in body_ids:
        return None

    # --- Find initialization in a pre-header ---
    start_val = _find_counter_init(func, header_id, counter_vreg, body_ids)
    if start_val is None:
        return None

    # --- Find increment block (the block with Jump back to header) ---
    increment_block_id = None
    step_val = None
    for bid in body_ids:
        if bid == header_id:
            continue
        block = func.blocks.get(bid)
        if block is None:
            continue
        # Check if this block jumps back to header
        if not block.instructions:
            continue
        last = block.instructions[-1]
        if isinstance(last, Jump) and last.target == header_id:
            # Look for counter increment
            for instr in block.instructions:
                if (isinstance(instr, BinaryOp) and
                        isinstance(instr.dest, VirtualRegister) and
                        instr.dest.id == counter_vreg.id and
                        isinstance(instr.left, VirtualRegister) and
                        instr.left.id == counter_vreg.id and
                        instr.op == '+' and
                        isinstance(instr.right, Immediate)):
                    increment_block_id = bid
                    step_val = instr.right.value
                    break
        if increment_block_id is not None:
            break

    if increment_block_id is None or step_val is None:
        return None

    # Body block IDs excluding header (those are the ones we clone)
    body_block_ids = [bid for bid in body_ids if bid != header_id]

    return (counter_vreg, start_val, end_val, step_val,
            compare.comparison, body_block_ids, exit_block_id,
            increment_block_id)


def _find_counter_init(func: MIRFunction, header_id: int,
                       counter_vreg: VirtualRegister,
                       body_ids: Set[int]) -> Optional[int]:
    """Find the constant initialization of the counter before the loop."""
    header = func.blocks.get(header_id)
    if header is None:
        return None

    for pred_id in header.predecessors:
        if pred_id in body_ids:
            continue  # Skip back-edges
        pred = func.blocks.get(pred_id)
        if pred is None:
            continue
        # Walk instructions backward looking for Move(counter, Immediate)
        for instr in reversed(pred.instructions):
            if (isinstance(instr, Move) and
                    isinstance(instr.dest, VirtualRegister) and
                    instr.dest.id == counter_vreg.id and
                    isinstance(instr.source, Immediate)):
                return instr.source.value
    return None


# ============================================================================
# Body Analysis
# ============================================================================

def _count_body_ops(func: MIRFunction, body_block_ids: List[int],
                    counter_vreg: Optional[VirtualRegister] = None) -> int:
    """Count MIR operation instructions in the body.

    Excludes control flow (Jump/CondBranch) and the counter increment
    BinaryOp (i = i + step) since that is removed during unrolling.
    """
    count = 0
    for bid in body_block_ids:
        block = func.blocks.get(bid)
        if block is None:
            continue
        for instr in block.instructions:
            if isinstance(instr, (Jump, CondBranch)):
                continue
            # Skip the counter increment
            if (counter_vreg is not None and
                    isinstance(instr, BinaryOp) and
                    isinstance(instr.dest, VirtualRegister) and
                    instr.dest.id == counter_vreg.id and
                    instr.op == '+'):
                continue
            count += 1
    return count


def _body_is_safe(func: MIRFunction, body_block_ids: List[int],
                  header_id: int, exit_block_id: int,
                  increment_block_id: int) -> bool:
    """Check that the body has no breaks, returns, nested loops, or asm!."""
    body_set = set(body_block_ids)

    for bid in body_block_ids:
        block = func.blocks.get(bid)
        if block is None:
            continue
        for instr in block.instructions:
            # Reject returns
            if isinstance(instr, (Return, ReturnFromInterrupt)):
                return False
            # Reject inline assembly (unpredictable effects)
            if isinstance(instr, InlineAsm):
                return False
            # Reject calls (side effects, could be expensive)
            if isinstance(instr, (Call, TraitDispatch)):
                return False
            # Reject CondBranch that exits the loop (break)
            if isinstance(instr, CondBranch):
                if (instr.true_target not in body_set and
                        instr.true_target != header_id):
                    return False
                if (instr.false_target not in body_set and
                        instr.false_target != header_id):
                    return False

    # Reject nested loops (any back-edge within body that doesn't target
    # our header)
    for bid in body_block_ids:
        block = func.blocks.get(bid)
        if block is None:
            continue
        for instr in block.instructions:
            if isinstance(instr, Jump) and instr.target != header_id:
                if instr.target in body_set and instr.target <= bid:
                    return False  # Inner back-edge = nested loop

    return True


# ============================================================================
# Block Ordering and Cloning Helpers
# ============================================================================

def _order_body_blocks(func: MIRFunction, entry_id: int,
                       body_ids: List[int], header_id: int) -> List[int]:
    """Order body blocks by BFS from the body entry."""
    body_set = set(body_ids)
    ordered = []
    visited = set()
    queue = [entry_id]
    while queue:
        bid = queue.pop(0)
        if bid in visited or bid == header_id or bid not in body_set:
            continue
        visited.add(bid)
        ordered.append(bid)
        block = func.blocks.get(bid)
        if block:
            for succ in block.successors:
                if succ not in visited and succ != header_id:
                    queue.append(succ)
    # Add any blocks not reached by BFS (shouldn't happen in well-formed CFG)
    for bid in body_ids:
        if bid not in visited and bid != header_id:
            ordered.append(bid)
    return ordered


def _all_vregs_defined(instr: MIRInstruction) -> List[VirtualRegister]:
    """Get all VirtualRegisters defined (written to) by an instruction."""
    result = []
    if hasattr(instr, 'dest') and isinstance(getattr(instr, 'dest'), VirtualRegister):
        result.append(instr.dest)
    return result


def _replace_vreg_with_immediate(instr: MIRInstruction,
                                 target_vreg: VirtualRegister,
                                 replacement: Immediate):
    """Replace all reads of target_vreg with an Immediate in an instruction."""
    # Fields that can hold operands (reads)
    for field_name in ('source', 'left', 'right', 'operand', 'condition',
                       'pointer', 'value', 'scrutinee'):
        val = getattr(instr, field_name, None)
        if (isinstance(val, VirtualRegister) and val.id == target_vreg.id):
            # For fields that accept Immediate
            if field_name in ('source', 'left', 'right', 'operand',
                              'scrutinee'):
                setattr(instr, field_name, replacement)

    # For Compare: replace left/right
    if isinstance(instr, Compare):
        if (isinstance(instr.left, VirtualRegister) and
                instr.left.id == target_vreg.id):
            instr.left = replacement
        if (isinstance(instr.right, VirtualRegister) and
                instr.right.id == target_vreg.id):
            instr.right = replacement

    # If the dest is the counter vreg in a Move/BinaryOp, this is the
    # increment — will be removed separately. Don't replace dest.


def _remap_vregs(instr: MIRInstruction,
                 vreg_map: Dict[int, VirtualRegister]):
    """Replace VirtualRegister references using the given mapping."""
    for field_name in ('dest', 'source', 'left', 'right', 'operand',
                       'condition', 'pointer', 'value', 'scrutinee',
                       'save_location'):
        val = getattr(instr, field_name, None)
        if isinstance(val, VirtualRegister) and val.id in vreg_map:
            setattr(instr, field_name, vreg_map[val.id])

    # Handle list fields
    if hasattr(instr, 'values') and isinstance(instr.values, list):
        instr.values = [vreg_map.get(v.id, v) if isinstance(v, VirtualRegister)
                        else v for v in instr.values]
    if hasattr(instr, 'args') and isinstance(instr.args, list):
        for arg in instr.args:
            if (hasattr(arg, 'value') and
                    isinstance(arg.value, VirtualRegister) and
                    arg.value.id in vreg_map):
                arg.value = vreg_map[arg.value.id]


def _remap_block_targets(block: BasicBlock,
                         block_id_map: Dict[int, int]):
    """Remap Jump/CondBranch targets using the block ID mapping."""
    for instr in block.instructions:
        if isinstance(instr, Jump):
            if instr.target in block_id_map:
                instr.target = block_id_map[instr.target]
        elif isinstance(instr, CondBranch):
            if instr.true_target in block_id_map:
                instr.true_target = block_id_map[instr.true_target]
            if instr.false_target in block_id_map:
                instr.false_target = block_id_map[instr.false_target]


def _remove_back_edge_jump(block: BasicBlock, header_id: int):
    """Remove the Jump instruction that targets the loop header."""
    block.instructions = [
        instr for instr in block.instructions
        if not (isinstance(instr, Jump) and instr.target == header_id)
    ]
    if header_id in block.successors:
        block.successors.remove(header_id)


def _remove_counter_increment(block: BasicBlock,
                              counter_vreg: VirtualRegister):
    """Remove the BinaryOp that increments the loop counter.

    After vreg-to-immediate replacement, the dest may still be the counter
    vreg while left has become an Immediate. Match on dest + op only.
    """
    block.instructions = [
        instr for instr in block.instructions
        if not (isinstance(instr, BinaryOp) and
                isinstance(instr.dest, VirtualRegister) and
                instr.dest.id == counter_vreg.id and
                instr.op == '+')
    ]


def _redirect_targets(block: BasicBlock, old_target: int, new_target: int):
    """Redirect all Jump/CondBranch targets from old to new."""
    for instr in block.instructions:
        if isinstance(instr, Jump) and instr.target == old_target:
            instr.target = new_target
        elif isinstance(instr, CondBranch):
            if instr.true_target == old_target:
                instr.true_target = new_target
            if instr.false_target == old_target:
                instr.false_target = new_target
