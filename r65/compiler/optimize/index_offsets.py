# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Constant Index Offset Folding Pass.

`walkmap[t + 32]` lowers to an add feeding X, so neighbouring cells of one
array each recompute and reload their index::

    %a = %t + #1             X = Move %t
    X = Move %a              Store #0 -> [walkmap,X]
    Store #0 -> [walkmap,X]  Store #0 -> [walkmap+1,X]
    %b = %t + #32      ->    Store #0 -> [walkmap+32,X]
    X = Move %b
    Store #0 -> [walkmap,X]

The constant belongs in the address: `LDA walkmap+32,X` costs nothing
over `LDA walkmap,X`. This pass peels `+ #C` / `- #C` (through one `<< k`
scale, as a struct stride is) off the value an `X = Move` loads and adds it
to the offset of every access indexed by that X. Loads of X that then hold
what X already holds go, and so does the arithmetic nothing reads.

Only a u16 add is folded. A u8 index wraps at 256 - `ring[head + 1]` on a
256-byte ring relies on it - where the folded address would not. A u16
index wraps only past 65535, beyond any array, so either way the access is
outside it. A negative offset is kept off a direct-page or low-RAM static
whose address may be 0; a #[ram] or ROM base is never that low.
"""

from dataclasses import replace
from typing import List, Optional, Tuple

from r65.compiler.mir.liveness import liveness_defs
from r65.compiler.mir.nodes import (
    BinaryOp, Immediate, MIRProgram, VirtualRegister,
    count_vreg_reads, iter_memory_operands, map_memory_operands,
)
from r65.compiler.optimize.index_streams import (
    drop_dead, redundant_x_moves, x_segments,
)

_WRAP_SAFE = frozenset({'u16', 'i16'})
_LOW_STORAGE = frozenset({'zeropage', 'lowram'})


def _type_name(type_info) -> Optional[str]:
    return getattr(type_info, 'name', None)


class IndexOffsetFolder:
    """Move `+ constant` out of an X index and into the addresses."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Fold every constant it can. Returns how many were folded."""
        self.folded = 0
        for func in mir_program.functions:
            for block in func.blocks.values():
                if self._fold_block(func, block):
                    instrs = block.instructions
                    found, sources = redundant_x_moves(instrs, x_segments(instrs))
                    block.instructions = [ins for i, ins in enumerate(instrs)
                                          if i not in found]
                    drop_dead(func, block, sources)
        return self.folded

    def _fold_block(self, func, block) -> bool:
        changed = False
        while self._fold_one(func, block):
            changed = True
        return changed

    def _fold_one(self, func, block) -> bool:
        reads = count_vreg_reads(func)
        instrs = block.instructions
        for seg in x_segments(instrs):
            if seg.move is None or seg.reads_x or not seg.accesses:
                continue
            peeled = self._peel(instrs, seg.move, reads)
            if peeled is None:
                continue
            delta, rewrite = peeled
            accesses = [instrs[a] for a in seg.accesses]
            if not all(self._offset_ok(loc, delta)
                       for ins in accesses for loc in iter_memory_operands(ins)
                       if loc.index_register == 'X'):
                continue
            rewrite()
            for ins in accesses:
                map_memory_operands(ins, lambda loc: self._shift(loc, delta))
            self.folded += 1
            return True
        return False

    def _peel(self, instrs: List, move_idx: int, reads) -> Optional[Tuple[int, object]]:
        """`(delta, rewrite)` when the value `X = Move` loads is `t ± C`,
        or `(t ± C) << k`, computed in this block for X alone."""
        move = instrs[move_idx]
        src = move.source
        add_idx = self._single_def(instrs, move_idx, src, reads)
        if add_idx is None:
            return None
        shift = 0
        shift_idx = None
        ins = instrs[add_idx]
        if (type(ins) is BinaryOp and ins.op == '<<' and type(ins.right) is Immediate
                and type(ins.left) is VirtualRegister):
            shift_idx, shift = add_idx, ins.right.value
            add_idx = self._single_def(instrs, shift_idx, ins.left, reads)
            if add_idx is None:
                return None
            ins = instrs[add_idx]
        constant = self._constant_term(ins)
        if constant is None:
            return None
        base, c = constant
        if self._redefined(instrs, base, add_idx, move_idx):
            return None
        delta = c << shift
        add = ins

        def rewrite():
            if shift_idx is None:
                instrs[move_idx] = replace(move, source=base)
            else:
                instrs[shift_idx] = replace(instrs[shift_idx], left=base)
            del instrs[next(i for i, x in enumerate(instrs) if x is add)]
        return delta, rewrite

    @staticmethod
    def _single_def(instrs: List, before: int, operand, reads) -> Optional[int]:
        """Index of the instruction defining *operand* in this block, when
        that def is its only one and the reader at *before* its only read."""
        if type(operand) is not VirtualRegister or reads.get(operand.id, 0) != 1:
            return None
        defs = [i for i, ins in enumerate(instrs)
                if any(type(d) is VirtualRegister and d.id == operand.id
                       for d in liveness_defs(ins))]
        if len(defs) != 1 or defs[0] >= before:
            return None
        return defs[0]

    @staticmethod
    def _constant_term(ins) -> Optional[Tuple[VirtualRegister, int]]:
        """`(t, C)` for a u16 `t + C`, `C + t` or `t - C`."""
        if type(ins) is not BinaryOp or _type_name(ins.type_info) not in _WRAP_SAFE:
            return None
        left, right = ins.left, ins.right
        if ins.op == '+':
            if type(right) is Immediate and type(left) is VirtualRegister:
                return left, right.value
            if type(left) is Immediate and type(right) is VirtualRegister:
                return right, left.value
        if ins.op == '-' and type(right) is Immediate and type(left) is VirtualRegister:
            return left, -right.value
        return None

    @staticmethod
    def _redefined(instrs: List, vreg, start: int, end: int) -> bool:
        """Whether *vreg* is written between *start* and *end*: moving its
        read from the add to the later instruction would see another value."""
        return any(type(d) is VirtualRegister and d.id == vreg.id
                   for ins in instrs[start + 1:end] for d in liveness_defs(ins))

    @staticmethod
    def _offset_ok(loc, delta: int) -> bool:
        if delta >= 0:
            return True
        if loc.address is not None:
            return loc.address + delta >= 0
        return loc.storage_type not in _LOW_STORAGE

    @staticmethod
    def _shift(loc, delta: int):
        if loc.index_register != 'X':
            return loc
        if loc.address is not None:
            return replace(loc, address=loc.address + delta)
        return replace(loc, offset=loc.offset + delta)
