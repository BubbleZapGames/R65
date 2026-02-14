"""
Dead Code Elimination.

Removes unreachable code and dead assignments within functions.
"""

from typing import Set, Dict, List
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock, MIRInstruction,
    VirtualRegister, HardwareRegister,
    Move, Load, Store, BinaryOp, UnaryOp, Compare, TypeConvert, ToBool,
    Jump, CondBranch, Return, ReturnFromInterrupt, Call,
    LoadIndirect, StoreIndirect, Rotate, BitTest, JumpTable,
    Push, Pull, SaveRegister, RestoreRegister, InlineAsm, SetMode,
    MemoryFill, BlockCopy, LookupTable, TraitDispatch,
)


class DeadCodeEliminator:
    """
    Eliminates dead code within MIR functions.

    Two main optimizations:
    1. Unreachable block elimination: Remove blocks not reachable from entry
    2. Dead store elimination: Remove assignments to unused virtual registers
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the dead code eliminator.

        Args:
            verbose: If True, print information about eliminated code
        """
        self.verbose = verbose

    def eliminate(self, mir_program: MIRProgram) -> int:
        """
        Eliminate dead code from all functions in the MIR program.

        Args:
            mir_program: The MIR program to optimize

        Returns:
            Total number of eliminations (blocks + instructions)
        """
        total_eliminated = 0

        for func in mir_program.functions:
            eliminated = self._eliminate_in_function(func)
            total_eliminated += eliminated

        return total_eliminated

    def _eliminate_in_function(self, func: MIRFunction) -> int:
        """
        Eliminate dead code in a single function.

        Args:
            func: The MIR function to optimize

        Returns:
            Number of eliminations (blocks + instructions)
        """
        total = 0

        # Phase 1: Remove unreachable blocks
        blocks_removed = self._eliminate_unreachable_blocks(func)
        total += blocks_removed

        if self.verbose and blocks_removed > 0:
            print(f"  {func.name}: removed {blocks_removed} unreachable block(s)")

        # Phase 2: Remove dead stores (assignments to unused vregs)
        stores_removed = self._eliminate_dead_stores(func)
        total += stores_removed

        if self.verbose and stores_removed > 0:
            print(f"  {func.name}: removed {stores_removed} dead store(s)")

        return total

    # ========================================================================
    # Unreachable Block Elimination
    # ========================================================================

    def _eliminate_unreachable_blocks(self, func: MIRFunction) -> int:
        """
        Remove blocks that are not reachable from the entry block.

        Uses BFS from entry block to find all reachable blocks.

        Args:
            func: The MIR function

        Returns:
            Number of blocks removed
        """
        if not func.blocks:
            return 0

        # Find all reachable blocks via BFS from entry
        reachable = self._find_reachable_blocks(func)

        # Find unreachable blocks
        all_block_ids = set(func.blocks.keys())
        unreachable = all_block_ids - reachable

        if not unreachable:
            return 0

        # Remove unreachable blocks
        for block_id in unreachable:
            del func.blocks[block_id]

        # Update predecessor lists in remaining blocks
        for block in func.blocks.values():
            block.predecessors = [p for p in block.predecessors if p in reachable]

        # Update exit block IDs
        func.exit_block_ids = [b for b in func.exit_block_ids if b in reachable]

        return len(unreachable)

    def _find_reachable_blocks(self, func: MIRFunction) -> Set[int]:
        """
        Find all blocks reachable from the entry block.

        Args:
            func: The MIR function

        Returns:
            Set of reachable block IDs
        """
        reachable = set()
        worklist = [func.entry_block_id]

        while worklist:
            block_id = worklist.pop()

            if block_id in reachable:
                continue

            reachable.add(block_id)

            # Add successors to worklist
            if block_id in func.blocks:
                block = func.blocks[block_id]
                for succ_id in block.successors:
                    if succ_id not in reachable:
                        worklist.append(succ_id)

        return reachable

    # ========================================================================
    # Dead Store Elimination
    # ========================================================================

    def _eliminate_dead_stores(self, func: MIRFunction) -> int:
        """
        Remove stores to virtual registers that are never read.

        Iterates until no more dead stores are found (fixed point).

        Args:
            func: The MIR function

        Returns:
            Number of instructions removed
        """
        total_removed = 0

        # Iterate until fixed point
        while True:
            # Compute which vregs are used (read)
            used_vregs = self._compute_used_vregs(func)

            # Find and remove dead stores
            removed = self._remove_dead_stores(func, used_vregs)

            if removed == 0:
                break

            total_removed += removed

        return total_removed

    def _compute_used_vregs(self, func: MIRFunction) -> Set[int]:
        """
        Compute the set of virtual register IDs that are actually used (read).

        Args:
            func: The MIR function

        Returns:
            Set of virtual register IDs that are read
        """
        used = set()

        for block in func.blocks.values():
            for instr in block.instructions:
                # Collect all vregs that are read by this instruction
                read_vregs = self._get_read_vregs(instr)
                used.update(read_vregs)

        return used

    def _get_read_vregs(self, instr: MIRInstruction) -> Set[int]:
        """
        Get the set of virtual register IDs read by an instruction.

        Args:
            instr: The MIR instruction

        Returns:
            Set of virtual register IDs that are read
        """
        read = set()

        def add_if_vreg(operand):
            if isinstance(operand, VirtualRegister):
                read.add(operand.id)

        # Handle each instruction type
        if isinstance(instr, Move):
            add_if_vreg(instr.source)
        elif isinstance(instr, Load):
            # Load reads from memory, source is MemoryLocation (not vreg)
            pass
        elif isinstance(instr, Store):
            add_if_vreg(instr.source)
        elif isinstance(instr, LoadIndirect):
            add_if_vreg(instr.pointer)
        elif isinstance(instr, StoreIndirect):
            add_if_vreg(instr.source)
            add_if_vreg(instr.pointer)
        elif isinstance(instr, BinaryOp):
            add_if_vreg(instr.left)
            add_if_vreg(instr.right)
        elif isinstance(instr, UnaryOp):
            add_if_vreg(instr.operand)
        elif isinstance(instr, Compare):
            add_if_vreg(instr.left)
            add_if_vreg(instr.right)
        elif isinstance(instr, TypeConvert):
            add_if_vreg(instr.source)
        elif isinstance(instr, ToBool):
            add_if_vreg(instr.source)
        elif isinstance(instr, CondBranch):
            add_if_vreg(instr.condition)
        elif isinstance(instr, Return):
            for val in instr.values:
                add_if_vreg(val)
        elif isinstance(instr, (Call, TraitDispatch)):
            for arg in instr.args:
                add_if_vreg(arg.value)
            # TraitDispatch uses self_ptr
            if isinstance(instr, TraitDispatch) and isinstance(instr.self_ptr, VirtualRegister):
                read.add(instr.self_ptr.id)
            # Indirect call through vreg
            if isinstance(instr, Call) and isinstance(instr.function, VirtualRegister):
                read.add(instr.function.id)
        elif isinstance(instr, Rotate):
            add_if_vreg(instr.source)
        elif isinstance(instr, BitTest):
            add_if_vreg(instr.value)
        elif isinstance(instr, JumpTable):
            add_if_vreg(instr.scrutinee)
        elif isinstance(instr, LookupTable):
            add_if_vreg(instr.scrutinee)
        elif isinstance(instr, RestoreRegister):
            add_if_vreg(instr.save_location)
        # Jump, ReturnFromInterrupt, Push, Pull, SaveRegister,
        # InlineAsm, SetMode, MemoryFill, BlockCopy don't read vregs directly

        return read

    def _get_written_vreg(self, instr: MIRInstruction) -> int:
        """
        Get the virtual register ID written by an instruction, or None.

        Args:
            instr: The MIR instruction

        Returns:
            Virtual register ID if instruction writes to a vreg, else None
        """
        dest = None

        if isinstance(instr, (Move, Load, LoadIndirect, BinaryOp, UnaryOp, TypeConvert, ToBool, Rotate, LookupTable)):
            dest = instr.dest
        elif isinstance(instr, SaveRegister):
            dest = instr.save_location
        elif isinstance(instr, (Call, TraitDispatch)):
            # Calls can have multiple return registers
            # For simplicity, we don't eliminate call results (they may have side effects)
            return None

        if isinstance(dest, VirtualRegister):
            return dest.id

        return None

    def _is_side_effect_free(self, instr: MIRInstruction) -> bool:
        """
        Check if an instruction has no side effects other than writing to dest.

        Instructions with side effects (calls, stores, mode changes, etc.)
        should not be eliminated even if their result is unused.

        Args:
            instr: The MIR instruction

        Returns:
            True if the instruction can be safely eliminated
        """
        # These instructions only write to a destination register
        # and have no other side effects
        return isinstance(instr, (Move, Load, LoadIndirect, BinaryOp, UnaryOp,
                                   TypeConvert, ToBool, Rotate, SaveRegister))

    def _remove_dead_stores(self, func: MIRFunction, used_vregs: Set[int]) -> int:
        """
        Remove instructions that write to unused virtual registers.

        Args:
            func: The MIR function
            used_vregs: Set of virtual register IDs that are actually used

        Returns:
            Number of instructions removed
        """
        removed = 0

        for block in func.blocks.values():
            new_instructions = []

            for instr in block.instructions:
                written_vreg = self._get_written_vreg(instr)

                # Keep instruction if:
                # 1. It doesn't write to a vreg, OR
                # 2. It writes to a used vreg, OR
                # 3. It has side effects (not safe to remove)
                if (written_vreg is None or
                    written_vreg in used_vregs or
                    not self._is_side_effect_free(instr)):
                    new_instructions.append(instr)
                else:
                    removed += 1

            block.instructions = new_instructions

        return removed


def eliminate_dead_code(mir_program: MIRProgram, verbose: bool = False) -> int:
    """
    Convenience function to run dead code elimination.

    Args:
        mir_program: The MIR program to optimize
        verbose: If True, print information about eliminated code

    Returns:
        Total number of eliminations
    """
    eliminator = DeadCodeEliminator(verbose=verbose)
    return eliminator.eliminate(mir_program)
