# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Known-Bits Simplification Pass.

Codegen masks with AND wherever a value changes width: `AND #$00FF` to
zero-extend a u8, `XBA; AND #$00FF` / `XBA; AND #$FF00` for a u16 shift by
8 or more. It cannot see that the bits a mask clears are often 0 already,
or that nothing reads them. The MIR can, from two facts per u16 vreg:

  known zero  bits every definition leaves 0: a widened u8's high byte,
              the bits a constant `&` clears, the bits a shift brings in
  demanded    bits its readers look at: `& C` reads C, a narrowing to u8
              reads the low byte

and with them:

  * a result whose every bit is known zero is the constant 0 -
    `((x as u16) >> 8) & 1` on a u8 x;
  * a value known below 256, narrowed and widened again, is itself -
    `set_full!(slot as u8, ..)` on a slot already widened, an index
    `(w >> 8) as u8` moved into X;
  * a shift's mask goes when the byte it clears is known zero
    (`(c & $7C00) >> 10`, `(x & 31) << 8`) or unread (`(w >> 8) & 1`);
  * a zero-extend's mask goes when its high byte is unread
    (`(b as u16) << 8`, which the shift's own mask cleans).

The facts are per vreg, over all its definitions (a vreg is not SSA). Masks
dropped because their bits are unread leave those bits unknown, so that
step runs last and only where every reader re-masks or narrows them away.
"""

from dataclasses import replace
from typing import Dict, List

from r65.compiler.mir.liveness import liveness_defs, liveness_uses
from r65.compiler.mir.nodes import (
    BinaryOp, Compare, HardwareRegister, Immediate, MIRProgram, Move, OperandRole,
    OperandContainer, Store, TypeConvert, VirtualRegister, iter_memory_operands,
    iter_operands,
)
from r65.compiler.optimize.index_streams import drop_dead

FULL = 0xFFFF


def _name(type_info):
    from r65.compiler.hir.types import strip_newtype
    return getattr(strip_newtype(type_info), 'name', None)


def _widens(ins) -> bool:
    return (type(ins) is TypeConvert and _name(ins.source_type) == 'u8'
            and _name(ins.target_type) == 'u16')


def _narrows(ins) -> bool:
    return (type(ins) is TypeConvert and _name(ins.source_type) == 'u16'
            and _name(ins.target_type) == 'u8')


def _const(operand):
    if type(operand) is Immediate and isinstance(operand.value, int):
        return operand.value
    return None


def _u16_op(ins) -> bool:
    return (type(ins) is BinaryOp and _name(ins.type_info) == 'u16'
            and type(ins.dest) is VirtualRegister
            and not any(True for _ in iter_memory_operands(ins)))


def _byte_shift(ins):
    """`k` for a u16 shift by a constant 8..15 (the XBA path), else None."""
    if not (_u16_op(ins) and ins.op in ('<<', '>>') and type(ins.left) is VirtualRegister):
        return None
    k = _const(ins.right)
    return k if k is not None and 8 <= k < 16 else None


def _garbage(ins, k: int) -> int:
    """Bits a byte shift without its mask leaves undefined."""
    if ins.op == '>>':
        return 0xFF00 >> (k - 8)
    return (0x00FF << (k - 8)) & FULL


class KnownBitsSimplifier:
    """Drop the masks that known-zero or unread bits make moot."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.simplified = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Simplify every function. Returns how many instructions changed."""
        self.simplified = 0
        for func in mir_program.functions:
            self._exact_rewrites(func, self._known_zero(func))
            self._flag_known_zero_shifts(func, self._known_zero(func))
            self._flag_unread(func)
        return self.simplified

    # -- known-zero bits ---------------------------------------------------

    @staticmethod
    def _defs(func) -> Dict[int, List]:
        defs: Dict[int, List] = {}
        for block in func.blocks.values():
            for ins in block.instructions:
                for d in liveness_defs(ins):
                    if type(d) is VirtualRegister:
                        defs.setdefault(d.id, []).append(ins)
        return defs

    def _known_zero(self, func) -> Dict[int, int]:
        """vreg id -> the bits every definition leaves zero (a greatest
        fixpoint: start from all-zero, weaken until each def agrees)."""
        defs = self._defs(func)

        def modelled(ins) -> bool:
            if _widens(ins):
                return True
            if type(ins) is Move:
                return (type(ins.dest) is VirtualRegister
                        and type(ins.source) in (VirtualRegister, Immediate)
                        and _name(ins.type_info) == 'u16')
            return _u16_op(ins) and ins.op in ('&', '|', '^', '<<', '>>')

        kz = {vid: FULL for vid, ds in defs.items() if all(modelled(d) for d in ds)}

        def of(operand) -> int:
            c = _const(operand)
            if c is not None:
                return ~c & FULL
            if type(operand) is VirtualRegister:
                return kz.get(operand.id, 0)
            return 0

        def leaves(ins) -> int:
            if _widens(ins):
                return 0 if ins.high_undefined else 0xFF00
            if type(ins) is Move:
                return of(ins.source)
            a, b = of(ins.left), of(ins.right)
            if ins.op == '&':
                return a | b
            if ins.op in ('|', '^'):
                return a & b
            k = _const(ins.right)
            if k is None or not 0 <= k < 16 or ins.mask_free:
                return 0
            if ins.op == '<<':
                return ((a << k) | ((1 << k) - 1)) & FULL
            if _name(ins.type_info) != 'u16':
                return 0
            return ((a >> k) | (FULL << (16 - k))) & FULL

        changed = True
        while changed:
            changed = False
            for vid in kz:
                new = FULL
                for d in defs[vid]:
                    new &= leaves(d)
                if new != kz[vid]:
                    kz[vid] = new
                    changed = True
        return kz

    # -- phase 1: exact rewrites ---------------------------------------------

    def _exact_rewrites(self, func, kz: Dict[int, int]):
        sole = {vid for vid, ds in self._defs(func).items() if len(ds) == 1}
        zeros = set()

        def high_zero(operand) -> bool:
            return type(operand) is VirtualRegister and kz.get(operand.id, 0) & 0xFF00 == 0xFF00

        for block in func.blocks.values():
            instrs = block.instructions
            narrowed = {}       # u8 vreg id -> the u16 vreg it narrows
            dropped = []
            for i, ins in enumerate(instrs):
                # A result with every bit known zero is the constant 0.
                if (_u16_op(ins) and kz.get(ins.dest.id) == FULL
                        and ins.dest.id in sole):
                    instrs[i] = Move(dest=ins.dest, source=Immediate(0),
                                     type_info=ins.type_info, source_loc=ins.source_loc)
                    dropped += [v for v in (ins.left, ins.right) if type(v) is VirtualRegister]
                    zeros.add(ins.dest.id)
                    self.simplified += 1
                # A value known below 256, narrowed and widened again.
                src = getattr(ins, 'source', None)
                wide = narrowed.get(src.id) if type(src) is VirtualRegister else None
                if wide is not None and src.id in sole:
                    if _widens(ins):
                        instrs[i] = Move(dest=ins.dest, source=wide, type_info=ins.target_type,
                                         source_loc=ins.source_loc)
                        dropped.append(src)
                        self.simplified += 1
                    elif (type(ins) is Move and type(ins.dest) is HardwareRegister
                            and ins.dest.name in ('X', 'Y')):
                        instrs[i] = replace(ins, source=wide, type_info=wide.type_info)
                        dropped.append(src)
                        self.simplified += 1
                for d in liveness_defs(instrs[i]):
                    if type(d) is VirtualRegister:
                        narrowed.pop(d.id, None)
                        narrowed = {k: w for k, w in narrowed.items() if w.id != d.id}
                ins = instrs[i]
                if (_narrows(ins) and high_zero(ins.source)
                        and type(ins.dest) is VirtualRegister):
                    narrowed[ins.dest.id] = ins.source
            if dropped:
                drop_dead(func, block, dropped)
        if zeros:
            self._propagate_zeros(func, zeros)

    @staticmethod
    def _propagate_zeros(func, zeros):
        """Read a folded 0 as the immediate it is, so no vreg is kept for it."""
        takes_immediate = (Store, BinaryOp, Compare, Move, TypeConvert)
        for block in func.blocks.values():
            for ins in block.instructions:
                if type(ins) not in takes_immediate:
                    continue
                for spec, v in list(iter_operands(ins, role=OperandRole.READ)):
                    if (spec.container is OperandContainer.SCALAR
                            and type(v) is VirtualRegister and v.id in zeros):
                        setattr(ins, spec.field, Immediate(0))
            drop_dead(func, block, [VirtualRegister(id=z, type_info=None) for z in zeros])

    # -- phase 2: masks over bits already zero ---------------------------------

    def _flag_known_zero_shifts(self, func, kz: Dict[int, int]):
        for block in func.blocks.values():
            for i, ins in enumerate(block.instructions):
                k = _byte_shift(ins)
                if k is None or ins.mask_free:
                    continue
                if ins.op == '>>' and _name(ins.type_info) != 'u16':
                    continue
                cleared = 0x00FF if ins.op == '>>' else 0xFF00   # source byte XBA moves in
                if kz.get(ins.left.id, 0) & cleared == cleared:
                    block.instructions[i] = replace(ins, mask_free=True)
                    self.simplified += 1

    # -- phase 3: masks over bits no reader looks at --------------------------

    def _demanded(self, func) -> Dict[int, int]:
        """vreg id -> the bits some reader looks at (a least fixpoint, back
        from the readers that read everything: stores, compares, calls...)."""
        readers: Dict[int, List] = {}
        for block in func.blocks.values():
            for ins in block.instructions:
                for u in liveness_uses(ins):
                    if type(u) is VirtualRegister:
                        readers.setdefault(u.id, []).append(ins)
        demand = {vid: 0 for vid in readers}

        def through(r, vid: int) -> int:
            dest = getattr(r, 'dest', None)
            d = demand.get(dest.id, 0) if type(dest) is VirtualRegister else FULL
            if _narrows(r):
                return 0x00FF
            if type(r) is Move and type(dest) is VirtualRegister:
                return d
            if not _u16_op(r):
                return FULL
            is_left = type(r.left) is VirtualRegister and r.left.id == vid
            if r.op == '&':
                c = _const(r.right if is_left else r.left)
                return d & c if c is not None else d
            if r.op in ('|', '^'):
                return d
            if r.op in ('<<', '>>') and is_left and not r.mask_free:
                k = _const(r.right)
                if k is not None and 0 <= k < 16:
                    if r.op == '<<':
                        return d >> k
                    if _name(r.type_info) == 'u16':
                        return (d << k) & FULL
            return FULL

        changed = True
        while changed:
            changed = False
            for vid, rs in readers.items():
                new = 0
                for r in rs:
                    new |= through(r, vid)
                if new != demand[vid]:
                    demand[vid] = new
                    changed = True
        return demand

    def _flag_unread(self, func):
        demand = self._demanded(func)
        defs = self._defs(func)
        for block in func.blocks.values():
            i = -1
            while i + 1 < len(block.instructions):
                i += 1
                ins = block.instructions[i]
                dest = getattr(ins, 'dest', None)
                if type(dest) is not VirtualRegister or len(defs.get(dest.id, ())) != 1:
                    continue
                d = demand.get(dest.id)
                if d is None:
                    continue    # no reader at all: dead code is not this pass's job
                if _widens(ins):
                    if not ins.high_undefined and d & 0xFF00 == 0:
                        block.instructions[i] = replace(ins, high_undefined=True)
                        self.simplified += 1
                    continue
                if not _u16_op(ins):
                    continue
                if ins.op == '&' and type(ins.left) is VirtualRegister:
                    c = _const(ins.right)
                    if c is not None and d & ~c & FULL == 0:
                        self.simplified += 1
                        if self._forward(func, block, i, dest, ins.left):
                            i -= 1      # the def is gone; look at what took its place
                        else:
                            block.instructions[i] = Move(dest=dest, source=ins.left,
                                                         type_info=ins.type_info,
                                                         source_loc=ins.source_loc)
                    continue
                k = _byte_shift(ins)
                if k is None or ins.mask_free:
                    continue
                if ins.op == '>>' and _name(ins.type_info) != 'u16':
                    continue
                if d & _garbage(ins, k) == 0:
                    block.instructions[i] = replace(ins, mask_free=True)
                    self.simplified += 1

    @staticmethod
    def _forward(func, block, i: int, dest, source) -> bool:
        """Have every reader of *dest* read *source* instead and delete the
        def at *i* - when they all follow it in this block with *source*
        unchanged between. A copy left in its place costs a spill."""
        instrs = block.instructions
        reads = sum(1 for b in func.blocks.values() for ins in b.instructions
                    for u in liveness_uses(ins)
                    if type(u) is VirtualRegister and u.id == dest.id)
        found = []
        for j in range(i + 1, len(instrs)):
            ins = instrs[j]
            if any(type(u) is VirtualRegister and u.id == dest.id for u in liveness_uses(ins)):
                slots = [spec for spec, v in iter_operands(ins, role=OperandRole.READ)
                         if type(v) is VirtualRegister and v.id == dest.id]
                if any(spec.container is not OperandContainer.SCALAR for spec in slots):
                    return False
                found.append((ins, slots))
            if any(type(d) is VirtualRegister and d.id in (source.id, dest.id)
                   for d in liveness_defs(ins)):
                break
        if sum(len(slots) for _, slots in found) != reads:
            return False
        for ins, slots in found:
            for spec in slots:
                setattr(ins, spec.field, source)
        del instrs[i]
        return True
