# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Constant-Index Folding Pass.

Rewrites a memory access indexed by X, where X holds a constant the block
set, into the absolute access at the constant offset, and drops the X loads
that are then dead::

    X = Move #2                          ->   %1 = Load [lowram:POOL__hp+2]
    %1 = Load [lowram:POOL__hp,X]

The case it is for is an inlined `#[soa]` method. A method takes its element
index in X (`self`), so its body reads every field as ``POOL__f,X``; inlined
at a call with a constant index - ``POOL[PLAYER].fly()`` - the inliner
substitutes the constant for `self`, and the body becomes ``LDX #0`` / ``LDA
POOL__f,X`` pairs (with ``LDA #0 / ASL A / TAX`` for a u16 plane's scaled
index). Folded, it is the same ``LDA POOL__f`` a direct ``POOL[PLAYER].f``
compiles to, so a method shared by every element of a pool costs nothing
over hand-written per-element code when its call names the element.

Soundness. The fold reads X's value from the latest `X = ...` in the block
before the access, through a whitelist of instructions known not to change
X (`_TRANSPARENT`); anything else - a call, inline asm, a block move, a
trait dispatch - forgets it. The MIR builder sets X before an indexed
access rather than leaving it set across unrelated code, so the latest
visible definition is the value the access used. Deleting an `X = Move`
needs X dead after it, which is a backward liveness over the CFG in which
every instruction outside the whitelist counts as a read of X - a call may
take an argument in X, asm may read it, a return may carry it out.
"""

from dataclasses import fields, replace
from typing import Dict, List, Optional, Set

from r65.compiler.mir.nodes import (
    BinaryOp, BitTest, Compare, CondBranch, HardwareRegister, Immediate,
    IncDecMemory, Jump, Load, MemoryLocation, MIRFunction, MIRProgram, Move,
    Return, Rotate, Store, ToBool, TypeConvert, UnaryOp, VirtualRegister, iter_memory_operands, map_memory_operands,
)

# Instructions whose only contact with X is through operands the pass sees:
# a memory operand's index, or X named as a register operand.
_TRANSPARENT = (Load, Store, IncDecMemory, Move, BinaryOp, UnaryOp, Compare,
                TypeConvert, ToBool, BitTest, Rotate, Jump, CondBranch)

_CONST_OPS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '<<': lambda a, b: a << b,
    '>>': lambda a, b: a >> b,
    '&': lambda a, b: a & b,
    '|': lambda a, b: a | b,
    '^': lambda a, b: a ^ b,
}


def _is_x(v) -> bool:
    return isinstance(v, HardwareRegister) and v.name == 'X'


def _writes_x(instr) -> bool:
    return isinstance(instr, (Move, BinaryOp, UnaryOp, TypeConvert, ToBool, Rotate)) \
        and _is_x(getattr(instr, 'dest', None))


def _reads_x(instr) -> bool:
    """Does *instr* read X? Anything outside the whitelist is assumed to,
    but a return only when it returns X itself: a value returned in X by
    the ABI is a vreg the codegen moves there, not the X this pass drops."""
    if isinstance(instr, Return):
        return any(_is_x(v) for v in instr.values)
    if not isinstance(instr, _TRANSPARENT):
        return True
    if any(m.index_register == 'X' for m in iter_memory_operands(instr)):
        return True
    for f in fields(instr):
        if f.name == 'dest' and _writes_x(instr):
            continue
        if _is_x(getattr(instr, f.name, None)):
            return True
    return False


class ConstIndexFolder:
    """Fold X-indexed accesses with a known constant X into absolute ones."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0

    def optimize(self, program: MIRProgram) -> int:
        self.folded = 0
        for func in program.functions:
            if self._fold_function(func):
                fed = self._drop_dead_x_moves(func)
                if fed:
                    self._drop_dead_constants(func, fed)
        return self.folded

    # -- the fold ---------------------------------------------------------

    def _fold_function(self, func: MIRFunction) -> bool:
        changed = False
        for block in func.blocks.values():
            known_x: Optional[int] = None
            consts: Dict[int, int] = {}
            for instr in block.instructions:
                if known_x is not None and isinstance(instr, _TRANSPARENT):
                    hits = sum(m.index_register == 'X' for m in iter_memory_operands(instr))
                    if hits:
                        map_memory_operands(instr, lambda m: self._at(m, known_x)
                                            if m.index_register == 'X' else m)
                        self.folded += hits
                        changed = True
                if not isinstance(instr, _TRANSPARENT):
                    known_x = None
                    consts.clear()
                    continue
                value = self._const_value(instr, consts)
                dest = getattr(instr, 'dest', None)
                if _writes_x(instr):
                    known_x = value
                elif isinstance(dest, VirtualRegister):
                    if value is None:
                        consts.pop(dest.id, None)
                    else:
                        consts[dest.id] = value
        return changed

    @staticmethod
    def _at(loc: MemoryLocation, k: int) -> MemoryLocation:
        """*loc* indexed by the constant *k*, as an absolute location."""
        if loc.address is not None:
            # An explicit address already carries any offset (instruction
            # selection does not add `offset` to it).
            return replace(loc, address=loc.address + k, index_register=None)
        return replace(loc, offset=(loc.offset or 0) + k, index_register=None)

    @staticmethod
    def _operand(v, consts: Dict[int, int]) -> Optional[int]:
        if isinstance(v, Immediate):
            return v.value
        if isinstance(v, VirtualRegister):
            return consts.get(v.id)
        return None

    def _const_value(self, instr, consts: Dict[int, int]) -> Optional[int]:
        """The constant *instr* writes to its dest, if it is one."""
        if isinstance(instr, Move):
            return self._operand(instr.source, consts)
        if isinstance(instr, BinaryOp) and instr.op in _CONST_OPS:
            a = self._operand(instr.left, consts)
            b = self._operand(instr.right, consts)
            if a is None or b is None:
                return None
            return _CONST_OPS[instr.op](a, b) & 0xFFFF
        return None

    # -- the clean-up -----------------------------------------------------

    def _drop_dead_x_moves(self, func: MIRFunction) -> Set[int]:
        """Delete `X = Move` whose X nothing reads, by liveness over the CFG.
        Returns the ids of the vregs those moves read."""
        fed: Set[int] = set()
        # The least fixpoint: start every block dead and let reads spread
        # backward. A block with no successor ends in its return (or an RTI,
        # which counts as a read), so nothing after it reads X; a successor
        # id with no block is assumed to.
        def live_out(block) -> bool:
            return any(live_in.get(s, True) for s in block.successors)

        live_in: Dict[int, bool] = {bid: False for bid in func.blocks}
        changed = True
        while changed:
            changed = False
            for bid, block in func.blocks.items():
                v = self._live_before(block.instructions, live_out(block))
                if v != live_in[bid]:
                    live_in[bid] = v
                    changed = True
        for block in func.blocks.values():
            live = live_out(block)
            kept: List = []
            for instr in reversed(block.instructions):
                if isinstance(instr, Move) and _is_x(instr.dest) and not live:
                    if isinstance(instr.source, VirtualRegister):
                        fed.add(instr.source.id)
                    continue               # dead: drop it
                if _writes_x(instr):
                    live = False
                if _reads_x(instr):
                    live = True
                kept.append(instr)
            kept.reverse()
            block.instructions = kept
        return fed

    @staticmethod
    def _live_before(instrs, live_after: bool) -> bool:
        live = live_after
        for instr in reversed(instrs):
            if _writes_x(instr):
                live = False
            if _reads_x(instr):
                live = True
        return live

    def _drop_dead_constants(self, func: MIRFunction, candidates: Set[int]) -> None:
        """Delete the pure definitions of *candidates* - the vregs a deleted
        X move read, and the operands of those definitions in turn - once
        nothing else references them."""
        candidates = set(candidates)
        while True:
            refs: Dict[int, int] = {}
            for block in func.blocks.values():
                for instr in block.instructions:
                    for vid in _vreg_ids(instr):
                        refs[vid] = refs.get(vid, 0) + 1
            removed = False
            for block in func.blocks.values():
                kept = []
                for instr in block.instructions:
                    dest = getattr(instr, 'dest', None)
                    pure = (isinstance(instr, Move) and isinstance(instr.source, (Immediate, VirtualRegister))) or \
                           (isinstance(instr, BinaryOp) and instr.op in _CONST_OPS
                            and not isinstance(instr.left, MemoryLocation)
                            and not isinstance(instr.right, MemoryLocation))
                    if pure and isinstance(dest, VirtualRegister) and dest.id in candidates \
                            and refs.get(dest.id, 0) == 1:
                        removed = True     # its only reference is this definition
                        for v in (getattr(instr, 'source', None), getattr(instr, 'left', None),
                                  getattr(instr, 'right', None)):
                            if isinstance(v, VirtualRegister):
                                candidates.add(v.id)
                        continue
                    kept.append(instr)
                block.instructions = kept
            if not removed:
                return


def _vreg_ids(instr) -> List[int]:
    """Every VirtualRegister id in an instruction's own fields, in a list
    field, or as an argument's value."""
    out: List[int] = []
    for f in fields(instr):
        v = getattr(instr, f.name, None)
        items = v if isinstance(v, (list, tuple)) else (v,)
        for item in items:
            if isinstance(item, VirtualRegister):
                out.append(item.id)
            elif isinstance(getattr(item, 'value', None), VirtualRegister):
                out.append(item.value.id)
    return out
