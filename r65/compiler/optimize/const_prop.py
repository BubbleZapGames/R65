# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Constant propagation for locals assigned once.

A vreg whose only definition is ``Move %v = #k`` holds k wherever it is read.
The reads where an immediate is as good as a register - a store's value, a
conversion's or a move's source, a binary op's right operand - take k
directly; a conversion of a constant, and a binary op of two, fold to a
constant in turn, which feeds the next round. A Move nothing reads any more
is dropped. A result never read by name is read through the register it
lands in (an inlined call's result, picked up by the `Move A` after it), so
its instruction keeps its kind and is not folded or dropped::

    %2 = Move #0                          ->   Store #0 -> tile
    %3 = TypeConvert %2 u16 -> u8              Store #0 -> attr
    Store %3 -> tile
    %4 = %2 >> #8
    %5 = TypeConvert %4 u16 -> u8
    Store %5 -> attr

The case it is for is a macro that binds an argument once
(`OamBuffer.set!`'s `let oam_chr = $chr`): without it, a constant argument
went through a stack slot instead of two immediates.

Soundness. Vregs are not SSA, so a vreg qualifies only with exactly one
definition in the whole function; any read of it that the definition does
not reach would read an undefined value anyway. Only integer scalars of one
or two bytes fold, masked to their width; an arithmetic right shift, a
pointer and a bool are left alone.
"""

from dataclasses import replace
from typing import Dict, Optional

from r65.compiler.mir.nodes import (
    BinaryOp, Immediate, MIRProgram, Move, OperandRole, Store, TypeConvert,
    VirtualRegister, iter_operands,
)

_INT_TYPES = ('u8', 'u16', 'i8', 'i16')


def _int_type(type_info) -> Optional[str]:
    from r65.compiler.hir.types import BasicTypeInfo, strip_newtype
    t = strip_newtype(type_info) if type_info is not None else None
    return t.name if isinstance(t, BasicTypeInfo) and t.name in _INT_TYPES else None


def _plain(value) -> bool:
    return isinstance(value, Immediate) and value.symbol is None and isinstance(value.value, int)


def _mask(name: str) -> int:
    return 0xFF if name in ('u8', 'i8') else 0xFFFF


def _fold_convert(instr: TypeConvert) -> Optional[int]:
    src, dst = _int_type(instr.source_type), _int_type(instr.target_type)
    if src is None or dst is None or not _plain(instr.source):
        return None
    v = instr.source.value & _mask(src)
    if src == 'i8' and dst in ('u16', 'i16') and v & 0x80:
        v |= 0xFF00                         # sign-extend
    return v & _mask(dst)


def _fold_binary(instr: BinaryOp) -> Optional[int]:
    name = _int_type(instr.type_info)
    if name is None or not (_plain(instr.left) and _plain(instr.right)):
        return None
    a, b, m = instr.left.value & _mask(name), instr.right.value, _mask(name)
    if instr.op == '+':
        return (a + b) & m
    if instr.op == '-':
        return (a - b) & m
    if instr.op == '&':
        return a & b & m
    if instr.op == '|':
        return (a | b) & m
    if instr.op == '^':
        return (a ^ b) & m
    if instr.op == '<<' and 0 <= b < 16:
        return (a << b) & m
    if instr.op == '>>' and 0 <= b < 16 and name in ('u8', 'u16'):
        return a >> b
    return None


class ConstantPropagator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.propagated = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        self.propagated = 0
        for func in mir_program.functions:
            while self._round(func):
                pass
        return self.propagated

    def _round(self, func) -> bool:
        instrs = [i for b in func.blocks.values() for i in b.instructions]
        defs: Dict[int, list] = {}
        for instr in instrs:
            for _, v in iter_operands(instr, role=OperandRole.WRITE):
                if isinstance(v, VirtualRegister):
                    defs.setdefault(v.id, []).append(instr)
        consts = {vid: d[0].source.value for vid, d in defs.items()
                  if len(d) == 1 and isinstance(d[0], Move) and _plain(d[0].source)
                  and isinstance(d[0].dest, VirtualRegister) and d[0].dest.id == vid}

        read_before = self._reads(func)

        def const_of(value):
            if isinstance(value, VirtualRegister) and value.id in consts:
                return Immediate(consts[value.id])
            return None

        changed = False
        for block in func.blocks.values():
            for k, instr in enumerate(block.instructions):
                new = instr
                if isinstance(instr, (Store, TypeConvert, Move)):
                    imm = const_of(instr.source)
                    if imm is not None:
                        new = replace(instr, source=imm)
                elif isinstance(instr, BinaryOp):
                    imm = const_of(instr.right)
                    if imm is not None:
                        new = replace(instr, right=imm)
                    # the left only when the result folds outright
                    left = const_of(new.left)
                    if left is not None and _plain(new.right):
                        new = replace(new, left=left)
                if not self._named(new, read_before):
                    pass        # read through the register it lands in: keep its kind
                elif isinstance(new, TypeConvert):
                    v = _fold_convert(new)
                    if v is not None:
                        new = Move(dest=new.dest, source=Immediate(v), type_info=new.target_type)
                elif isinstance(new, BinaryOp):
                    v = _fold_binary(new)
                    if v is not None:
                        new = Move(dest=new.dest, source=Immediate(v), type_info=new.type_info)
                if new is not instr:
                    new.source_loc = instr.source_loc
                    block.instructions[k] = new
                    self.propagated += 1
                    changed = True

        # Drop the Moves of constants whose every read took the immediate. A
        # vreg never read by name may still be read through the register it
        # lands in (an inlined result picked up by a following `Move A`), so
        # it stays.
        read = self._reads(func)
        dead = {vid for vid in consts if vid in read_before and vid not in read}
        for block in func.blocks.values():
            kept = [i for i in block.instructions
                    if not (isinstance(i, Move) and isinstance(i.dest, VirtualRegister)
                            and i.dest.id in dead)]
            if len(kept) != len(block.instructions):
                block.instructions[:] = kept
                changed = True
        return changed

    @staticmethod
    def _named(instr, read) -> bool:
        dest = getattr(instr, 'dest', None)
        return isinstance(dest, VirtualRegister) and dest.id in read

    @staticmethod
    def _reads(func) -> set:
        read = set()
        for block in func.blocks.values():
            for instr in block.instructions:
                for _, v in iter_operands(instr, role=OperandRole.READ):
                    if isinstance(v, VirtualRegister):
                        read.add(v.id)
        return read
