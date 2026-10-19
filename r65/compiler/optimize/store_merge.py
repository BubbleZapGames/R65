# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Merge the two byte stores of a 16-bit value into one word store.

The low byte of a u16 `v` stored at `M` and its high byte at `M+1` is the
little-endian word `v` at `M`::

    %a = TypeConvert %v u16 -> u8         ->   Store %v -> [M] : u16
    Store %a -> [M] : u8
    %t = %v >> #8
    %b = TypeConvert %t u16 -> u8
    Store %b -> [M+1] : u8

The case it is for is an OAM entry: `tile` and `attr` are adjacent u8
fields that `OamBuffer.set!` fills from one `chr` (bound once in the macro).
Two m8 byte stores with an `XBA` between become one m16 store.

Only a pair taken from one vreg merges - two constant bytes are already two
cheap immediates. The location must be the same but for the byte (same
storage, symbol and index register), not `#[hw]`, nothing from the first
conversion to the second store may write `v`, and nothing between the two
stores may touch memory or write an index register (the index set up for
the low store may come after the conversion); the conversions and the
shift must have no other reader.
"""

from typing import Dict, Optional

from r65.compiler.mir.nodes import (
    BinaryOp, HardwareRegister, Immediate, MemoryLocation, MIRProgram, Move,
    OperandRole, Store, TypeConvert, VirtualRegister, iter_operands,
)


def _int_name(type_info) -> Optional[str]:
    from r65.compiler.hir.types import BasicTypeInfo, strip_newtype
    t = strip_newtype(type_info) if type_info is not None else None
    return t.name if isinstance(t, BasicTypeInfo) else None


def _next_byte(a: MemoryLocation, b: MemoryLocation) -> bool:
    if not (isinstance(a, MemoryLocation) and isinstance(b, MemoryLocation)):
        return False
    if a.storage_type == 'hw' or b.storage_type != a.storage_type:
        return False
    if a.index_register != b.index_register or a.symbol is not b.symbol:
        return False
    if a.address is not None or b.address is not None:
        return (a.address is not None and b.address is not None
                and b.address == a.address + 1 and b.offset == a.offset)
    return b.offset == a.offset + 1


def _narrow_of(instr) -> Optional[VirtualRegister]:
    """The u16 vreg *instr* truncates to u8, or None."""
    if (isinstance(instr, TypeConvert) and isinstance(instr.source, VirtualRegister)
            and _int_name(instr.source_type) in ('u16', 'i16')
            and _int_name(instr.target_type) in ('u8', 'i8')):
        return instr.source
    return None


_PURE = (BinaryOp, TypeConvert, Move)


class StoreMerger:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.merged = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        self.merged = 0
        for func in mir_program.functions:
            reads: Dict[int, int] = {}
            for block in func.blocks.values():
                for instr in block.instructions:
                    for _, v in iter_operands(instr, role=OperandRole.READ):
                        if isinstance(v, VirtualRegister):
                            reads[v.id] = reads.get(v.id, 0) + 1
            for block in func.blocks.values():
                while self._merge_one(block.instructions, reads):
                    self.merged += 1
        return self.merged

    def _merge_one(self, instrs, reads) -> bool:
        defs = {}
        for k, instr in enumerate(instrs):
            for _, v in iter_operands(instr, role=OperandRole.WRITE):
                if isinstance(v, VirtualRegister):
                    defs.setdefault(v.id, []).append(k)

        def only_def(vreg) -> Optional[int]:
            d = defs.get(vreg.id, [])
            return d[0] if len(d) == 1 else None

        for j, hi in enumerate(instrs):
            if not (isinstance(hi, Store) and _int_name(hi.type_info) in ('u8', 'i8')
                    and isinstance(hi.source, VirtualRegister)):
                continue
            kb = only_def(hi.source)
            if kb is None or reads.get(hi.source.id) != 1:
                continue
            t = _narrow_of(instrs[kb])
            kt = only_def(t) if t is not None else None
            if kt is None or reads.get(t.id) != 1:
                continue
            shift = instrs[kt]
            if not (isinstance(shift, BinaryOp) and shift.op == '>>'
                    and isinstance(shift.left, VirtualRegister)
                    and isinstance(shift.right, Immediate) and shift.right.value == 8
                    and _int_name(shift.type_info) in ('u16',)):
                continue
            v = shift.left
            # the low-byte store: the latest u8 store of trunc(v) to the byte before
            for i in range(j - 1, -1, -1):
                lo = instrs[i]
                if not (isinstance(lo, Store) and isinstance(lo.source, VirtualRegister)
                        and _int_name(lo.type_info) in ('u8', 'i8')
                        and _next_byte(lo.dest, hi.dest)):
                    continue
                ka = only_def(lo.source)
                if ka is None or reads.get(lo.source.id) != 1 or _narrow_of(instrs[ka]) != v:
                    break
                if not self._quiet(instrs, min(ka, kt), i, j, v):
                    break
                instrs[i] = Store(source=v, dest=lo.dest, type_info=shift.type_info,
                                  source_loc=lo.source_loc)
                for k in sorted({j, kb, kt, ka}, reverse=True):
                    del instrs[k]
                reads[v.id] = reads.get(v.id, 0) - 1     # two reads became one
                return True
        return False

    @staticmethod
    def _quiet(instrs, start, i, j, v) -> bool:
        """The low store at i can take v and the high byte: nothing from the
        first conversion (start) to the high store (j) redefines v, and
        between the two stores there are only pure vreg defs - no memory
        access and no index register write."""
        for k in range(start, j):
            if k == i:
                continue
            instr = instrs[k]
            if any(isinstance(d, VirtualRegister) and d.id == v.id
                   for _, d in iter_operands(instr, role=OperandRole.WRITE)):
                return False
            if k > i and not (isinstance(instr, _PURE)
                              and isinstance(instr.dest, VirtualRegister)):
                return False
        return True
