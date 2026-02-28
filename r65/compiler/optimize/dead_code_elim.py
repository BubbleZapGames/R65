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


# ============================================================================
# _get_read_vregs dispatch helpers
# ============================================================================

def _read_move(instr):
    s = instr.source
    return {s.id} if type(s) is VirtualRegister else set()

def _read_load(instr):
    return set()

def _read_store(instr):
    s = instr.source
    return {s.id} if type(s) is VirtualRegister else set()

def _read_load_indirect(instr):
    p = instr.pointer
    return {p.id} if type(p) is VirtualRegister else set()

def _read_store_indirect(instr):
    read = set()
    if type(instr.source) is VirtualRegister:
        read.add(instr.source.id)
    if type(instr.pointer) is VirtualRegister:
        read.add(instr.pointer.id)
    return read

def _read_binary_op(instr):
    read = set()
    if type(instr.left) is VirtualRegister:
        read.add(instr.left.id)
    if type(instr.right) is VirtualRegister:
        read.add(instr.right.id)
    return read

def _read_unary_op(instr):
    o = instr.operand
    return {o.id} if type(o) is VirtualRegister else set()

def _read_compare(instr):
    read = set()
    if type(instr.left) is VirtualRegister:
        read.add(instr.left.id)
    if type(instr.right) is VirtualRegister:
        read.add(instr.right.id)
    return read

def _read_type_convert(instr):
    s = instr.source
    return {s.id} if type(s) is VirtualRegister else set()

def _read_to_bool(instr):
    s = instr.source
    return {s.id} if type(s) is VirtualRegister else set()

def _read_cond_branch(instr):
    c = instr.condition
    return {c.id} if type(c) is VirtualRegister else set()

def _read_return(instr):
    read = set()
    for val in instr.values:
        if type(val) is VirtualRegister:
            read.add(val.id)
    return read

def _read_call(instr):
    read = set()
    for arg in instr.args:
        if type(arg.value) is VirtualRegister:
            read.add(arg.value.id)
    # Indirect call through vreg
    if type(instr.function) is VirtualRegister:
        read.add(instr.function.id)
    return read

def _read_trait_dispatch(instr):
    read = set()
    for arg in instr.args:
        if type(arg.value) is VirtualRegister:
            read.add(arg.value.id)
    if type(instr.self_ptr) is VirtualRegister:
        read.add(instr.self_ptr.id)
    return read

def _read_rotate(instr):
    s = instr.source
    return {s.id} if type(s) is VirtualRegister else set()

def _read_bit_test(instr):
    v = instr.value
    return {v.id} if type(v) is VirtualRegister else set()

def _read_jump_table(instr):
    s = instr.scrutinee
    return {s.id} if type(s) is VirtualRegister else set()

def _read_lookup_table(instr):
    s = instr.scrutinee
    return {s.id} if type(s) is VirtualRegister else set()

def _read_restore_register(instr):
    s = instr.save_location
    return {s.id} if type(s) is VirtualRegister else set()

def _read_none(instr):
    return set()


# Dispatch table: instruction type -> read vregs extractor
_GET_READ_VREGS = {
    Move: _read_move,
    Load: _read_load,
    Store: _read_store,
    LoadIndirect: _read_load_indirect,
    StoreIndirect: _read_store_indirect,
    BinaryOp: _read_binary_op,
    UnaryOp: _read_unary_op,
    Compare: _read_compare,
    TypeConvert: _read_type_convert,
    ToBool: _read_to_bool,
    CondBranch: _read_cond_branch,
    Return: _read_return,
    Call: _read_call,
    TraitDispatch: _read_trait_dispatch,
    Rotate: _read_rotate,
    BitTest: _read_bit_test,
    JumpTable: _read_jump_table,
    LookupTable: _read_lookup_table,
    RestoreRegister: _read_restore_register,
    # These don't read vregs directly
    Jump: _read_none,
    ReturnFromInterrupt: _read_none,
    Push: _read_none,
    Pull: _read_none,
    SaveRegister: _read_none,
    InlineAsm: _read_none,
    SetMode: _read_none,
    MemoryFill: _read_none,
    BlockCopy: _read_none,
}

# Types whose dest field is a writable vreg
_DEST_TYPES = frozenset({Move, Load, LoadIndirect, BinaryOp, UnaryOp, TypeConvert, ToBool, Rotate, LookupTable})

# Types that are safe to eliminate (no side effects beyond writing dest)
_SIDE_EFFECT_FREE = frozenset({Move, Load, LoadIndirect, BinaryOp, UnaryOp, TypeConvert, ToBool, Rotate, SaveRegister})


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

        Uses dispatch dict for O(1) type lookup.

        Args:
            instr: The MIR instruction

        Returns:
            Set of virtual register IDs that are read
        """
        handler = _GET_READ_VREGS.get(type(instr))
        if handler:
            return handler(instr)
        return set()

    def _get_written_vreg(self, instr: MIRInstruction) -> int:
        """
        Get the virtual register ID written by an instruction, or None.

        Args:
            instr: The MIR instruction

        Returns:
            Virtual register ID if instruction writes to a vreg, else None
        """
        instr_type = type(instr)

        if instr_type in _DEST_TYPES:
            dest = instr.dest
        elif instr_type is SaveRegister:
            dest = instr.save_location
        elif instr_type is Call or instr_type is TraitDispatch:
            # Calls can have multiple return registers
            # For simplicity, we don't eliminate call results (they may have side effects)
            return None
        else:
            return None

        if type(dest) is VirtualRegister:
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
        return type(instr) in _SIDE_EFFECT_FREE

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
