# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Dead Code Elimination.

Removes unreachable code and dead assignments within functions.
"""

from typing import Set
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, MIRInstruction, VirtualRegister, MemoryLocation,
    Move, Load, BinaryOp, UnaryOp, TypeConvert, ToBool, LoadIndirect, Rotate,
    OperandRole, iter_operands,
)


# ============================================================================
# Operand access (the shared registry in mir/nodes.py)
# ============================================================================

# Types that are safe to eliminate (no side effects beyond writing dest)
_SIDE_EFFECT_FREE = frozenset({
    Move, Load, LoadIndirect, BinaryOp, UnaryOp,
    TypeConvert, ToBool, Rotate,
})


def _reads_volatile(instr: MIRInstruction) -> bool:
    """True for a read of a #[hw] location. The read is the side effect —
    $2137 latches the H/V counters, $4211 acknowledges the timer IRQ — so
    it stays even when nothing uses the value."""
    source = getattr(instr, 'source', None)
    return (type(instr) in (Load, Move) and isinstance(source, MemoryLocation)
            and source.is_volatile)


def _get_read_vreg_ids(instr: MIRInstruction) -> Set[int]:
    """Get the set of virtual register IDs read by an instruction."""
    return {v.id for _, v in iter_operands(instr, role=OperandRole.READ)
            if type(v) is VirtualRegister}


def _get_written_vreg_id(instr: MIRInstruction):
    """The virtual register ID an instruction writes, or None.

    Asked only of the side-effect-free types, which each write one dest.
    """
    for _, v in iter_operands(instr, role=OperandRole.WRITE):
        if type(v) is VirtualRegister:
            return v.id
    return None


# ============================================================================
# DeadCodeEliminator
# ============================================================================

class DeadCodeEliminator:
    """
    Eliminates dead code within MIR functions.

    Two phases:
    1. Unreachable block elimination (BFS from entry)
    2. Dead store elimination (fixed-point removal of unused vreg writes)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def eliminate(self, mir_program: MIRProgram) -> int:
        """Eliminate dead code from all functions. Returns total eliminations."""
        total = 0
        for func in mir_program.functions:
            total += self._eliminate_in_function(func)
        return total

    def _eliminate_in_function(self, func: MIRFunction) -> int:
        total = 0

        blocks_removed = self._eliminate_unreachable_blocks(func)
        total += blocks_removed
        if self.verbose and blocks_removed > 0:
            print(f"  {func.name}: removed {blocks_removed} unreachable block(s)")

        stores_removed = self._eliminate_dead_stores(func)
        total += stores_removed
        if self.verbose and stores_removed > 0:
            print(f"  {func.name}: removed {stores_removed} dead store(s)")

        return total

    # ====================================================================
    # Phase 1: Unreachable Block Elimination
    # ====================================================================

    def _eliminate_unreachable_blocks(self, func: MIRFunction) -> int:
        if not func.blocks:
            return 0

        reachable = set()
        worklist = [func.entry_block_id]
        while worklist:
            block_id = worklist.pop()
            if block_id in reachable:
                continue
            reachable.add(block_id)
            if block_id in func.blocks:
                for succ_id in func.blocks[block_id].successors:
                    if succ_id not in reachable:
                        worklist.append(succ_id)

        unreachable = set(func.blocks.keys()) - reachable
        if not unreachable:
            return 0

        for block_id in unreachable:
            del func.blocks[block_id]
        for block in func.blocks.values():
            block.predecessors = [p for p in block.predecessors if p in reachable]
        func.exit_block_ids = [b for b in func.exit_block_ids if b in reachable]

        return len(unreachable)

    # ====================================================================
    # Phase 2: Dead Store Elimination
    # ====================================================================

    def _eliminate_dead_stores(self, func: MIRFunction) -> int:
        """Remove writes to unused vregs. Iterates until fixed point."""
        total_removed = 0

        while True:
            # Collect all vreg IDs that are read anywhere
            used = set()
            for block in func.blocks.values():
                for instr in block.instructions:
                    used.update(_get_read_vreg_ids(instr))

            # Remove side-effect-free instructions that write to unused vregs
            removed = 0
            for block in func.blocks.values():
                new_instructions = []
                for instr in block.instructions:
                    written = _get_written_vreg_id(instr)
                    if (written is not None and
                        written not in used and
                        type(instr) in _SIDE_EFFECT_FREE and
                        not _reads_volatile(instr)):
                        removed += 1
                    else:
                        new_instructions.append(instr)
                block.instructions = new_instructions

            if removed == 0:
                break
            total_removed += removed

        return total_removed
