# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
    LoadIndirect, StoreIndirect, Rotate, BitTest, JumpTable, BankByte,
    Push, Pull, SaveRegister, RestoreRegister, InlineAsm, SetMode,
    MemoryFill, BlockCopy, LookupTable, TraitDispatch,
)


# ============================================================================
# Read-vreg extraction (data-driven)
# ============================================================================

_EMPTY: frozenset = frozenset()

# Instruction types whose read operands are simple named fields.
# Maps type -> tuple of attribute names to check for VirtualRegister.
_READ_FIELDS = {
    Move: ('source',),
    Store: ('source',),
    LoadIndirect: ('pointer',),
    StoreIndirect: ('source', 'pointer'),
    BinaryOp: ('left', 'right'),
    UnaryOp: ('operand',),
    Compare: ('left', 'right'),
    TypeConvert: ('source',),
    ToBool: ('source',),
    BankByte: ('source',),
    CondBranch: ('condition',),
    Rotate: ('source',),
    BitTest: ('value',),
    JumpTable: ('scrutinee',),
    LookupTable: ('scrutinee',),
    RestoreRegister: ('save_location',),
}

# Types that read no vregs at all
_READ_NONE = frozenset({
    Load, Jump, ReturnFromInterrupt, Push, Pull, SaveRegister,
    InlineAsm, SetMode, MemoryFill, BlockCopy,
})

# Types whose dest field is a writable vreg
_DEST_TYPES = frozenset({
    Move, Load, LoadIndirect, BinaryOp, UnaryOp,
    TypeConvert, ToBool, Rotate, LookupTable,
})

# Types that are safe to eliminate (no side effects beyond writing dest)
_SIDE_EFFECT_FREE = frozenset({
    Move, Load, LoadIndirect, BinaryOp, UnaryOp,
    TypeConvert, ToBool, Rotate, SaveRegister,
})


def _get_read_vreg_ids(instr: MIRInstruction) -> Set[int]:
    """Get the set of virtual register IDs read by an instruction."""
    instr_type = type(instr)

    # Fast path: field-spec types (covers ~15 instruction types)
    fields = _READ_FIELDS.get(instr_type)
    if fields is not None:
        if len(fields) == 1:
            val = getattr(instr, fields[0])
            return {val.id} if type(val) is VirtualRegister else _EMPTY
        # Two fields
        read = set()
        for f in fields:
            val = getattr(instr, f)
            if type(val) is VirtualRegister:
                read.add(val.id)
        return read

    # Special cases with loop-based logic
    if instr_type is Call:
        read = set()
        for arg in instr.args:
            if type(arg.value) is VirtualRegister:
                read.add(arg.value.id)
        if type(instr.function) is VirtualRegister:
            read.add(instr.function.id)
        return read

    if instr_type is TraitDispatch:
        read = set()
        for arg in instr.args:
            if type(arg.value) is VirtualRegister:
                read.add(arg.value.id)
        if type(instr.self_ptr) is VirtualRegister:
            read.add(instr.self_ptr.id)
        return read

    if instr_type is Return:
        read = set()
        for val in instr.values:
            if type(val) is VirtualRegister:
                read.add(val.id)
        return read

    # No-read types (Load, Jump, InlineAsm, etc.)
    return _EMPTY


def _get_written_vreg_id(instr: MIRInstruction):
    """Get the virtual register ID written by an instruction, or None."""
    instr_type = type(instr)

    if instr_type in _DEST_TYPES:
        dest = instr.dest
    elif instr_type is SaveRegister:
        dest = instr.save_location
    elif instr_type is Call or instr_type is TraitDispatch:
        return None
    else:
        return None

    return dest.id if type(dest) is VirtualRegister else None


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
                        type(instr) in _SIDE_EFFECT_FREE):
                        removed += 1
                    else:
                        new_instructions.append(instr)
                block.instructions = new_instructions

            if removed == 0:
                break
            total_removed += removed

        return total_removed
