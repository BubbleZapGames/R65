# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Index Stride Reduction.

A u16 array indexed by a loop counter is addressed with the counter
doubled, rebuilt every pass::

    %t = %i << #1                   TYA / ASL A / TAX
    X = Move %t                     LDA s / CLC / ADC $0200,X / STA s
    ...                             INY
    %i = %i + #1                    CPY #$0100
    Compare %i < #256

When the counter is read by nothing but those doubles, its own step and
its test, it is carried doubled instead: started at twice its start,
stepped by 2 and tested against twice the bound::

    X = Move %i                     LDA s / CLC / ADC $0200,X / STA s
    ...                             INX / INX
    %i = %i + #2                    CPX #$0200
    Compare %i < #512

`i << 1` and the new `i` agree modulo 2^16 at every pass, so the index
reads are exact. Only the test compares a different number: `i < N` and
`2i < 2N` agree while no value the test sees is 2^15 or more, which a
constant start and bound can show. The counter must be dead outside the
loop - nothing else reads the doubled value.

Runs on MIR before loop promotion, which keeps a counter stepped by 2 in X
or Y (INX / INX).
"""

from typing import Dict, List, Optional, Set, Tuple

from r65.compiler.analysis.loop_analysis import find_loops
from r65.compiler.hir.types import BasicTypeInfo, strip_newtype
from r65.compiler.mir.nodes import (
    BinaryOp, Compare, CondBranch, Immediate, MIRFunction, MIRProgram, Move,
    OperandRole, VirtualRegister, iter_operands, map_operands,
)

# The shift that doubles an index into a u16 array.
_SHIFT = 1


def _is_u16(type_info) -> bool:
    t = strip_newtype(type_info) if type_info is not None else None
    return isinstance(t, BasicTypeInfo) and t.name == 'u16'


def _const(value) -> Optional[int]:
    if type(value) is Immediate and value.symbol is None and isinstance(value.value, int):
        return value.value
    return None


class IndexStrideReducer:
    """Carry a loop counter read only as a doubled index doubled."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def optimize(self, mir_program: MIRProgram) -> int:
        return sum(self._function(func) for func in mir_program.functions)

    def _function(self, func: MIRFunction) -> int:
        loops: Dict[int, Set[int]] = {}
        for header, body in find_loops(func):
            loops.setdefault(header, set()).update(b for b in body if b in func.blocks)
        if not loops:
            return 0
        reduced = 0
        for body in loops.values():
            sites = self._sites(func)
            for vid in sorted({v for v, (defs, _) in sites.items() if len(defs) == 2}):
                if self._reduce(func, body, vid, *sites[vid]):
                    reduced += 1
                    sites = self._sites(func)
        return reduced

    @staticmethod
    def _sites(func: MIRFunction) -> Dict[int, Tuple[List, List]]:
        """vreg id -> (defs, reads), each a list of (block_id, index, instr)."""
        sites: Dict[int, Tuple[List, List]] = {}
        for bid, block in func.blocks.items():
            for k, instr in enumerate(block.instructions):
                for role, slot in ((OperandRole.WRITE, 0), (OperandRole.READ, 1)):
                    for _, v in iter_operands(instr, role=role):
                        if type(v) is VirtualRegister:
                            sites.setdefault(v.id, ([], []))[slot].append((bid, k, instr))
        return sites

    def _reduce(self, func: MIRFunction, body: Set[int], vid: int,
                defs: List, reads: List) -> bool:
        outside = [d for d in defs if d[0] not in body]
        inside = [d for d in defs if d[0] in body]
        if len(outside) != 1 or len(inside) != 1:
            return False
        init = outside[0][2]
        start = _const(init.source) if type(init) is Move else None
        if start is None or start < 0 or not _is_u16(init.type_info):
            return False

        step = self._step(func, vid, inside[0])
        if step is None:
            return False
        step_op = step[0]

        # A compare may also sit outside the loop - the guard a rotation
        # left in front of it tests the start: every value it can see is
        # covered by the bounds check below.
        compares, doubles = [], []
        for bid, k, instr in reads:
            if instr is step_op:
                continue
            if (type(instr) is Compare and type(instr.left) is VirtualRegister
                    and instr.left.id == vid and instr.comparison in ('<', '<=')
                    and _const(instr.right) is not None and _is_u16(instr.type_info)
                    and self._branch_follows(func, bid, k, instr)):
                compares.append(instr)
            elif (bid in body and type(instr) is BinaryOp and instr.op == '<<'
                    and type(instr.left) is VirtualRegister and instr.left.id == vid
                    and _const(instr.right) == _SHIFT
                    and type(instr.dest) is VirtualRegister and _is_u16(instr.type_info)):
                doubles.append((bid, k, instr))
            else:
                return False
        if not doubles or not compares:
            return False

        # Every value a test sees is below the bound plus one step: doubled,
        # it, the start and each bound must stay below 2^16.
        for cmp in compares:
            bound = _const(cmp.right) + (1 if cmp.comparison == '<=' else 0)
            if (bound << _SHIFT) > 0xFFFF:
                return False
        if (start << _SHIFT) > 0xFFFF:
            return False

        init.source = Immediate(start << _SHIFT)
        step_op.right = Immediate(1 << _SHIFT)
        for cmp in compares:
            cmp.right = Immediate(_const(cmp.right) << _SHIFT)
        copies = []
        for bid, k, instr in doubles:
            copy = Move(dest=instr.dest, source=instr.left, type_info=instr.type_info,
                        source_loc=getattr(instr, 'source_loc', None))
            func.blocks[bid].instructions[k] = copy
            copies.append((bid, copy))
        for bid, copy in copies:
            self._forward_copy(func, bid, copy, vid)
        if self.verbose:
            print(f"  index-stride: {func.name} %{vid} carried doubled")
        return True

    @staticmethod
    def _step(func: MIRFunction, vid: int, site) -> Optional[Tuple[BinaryOp, Optional[Move]]]:
        """The counter's `+ 1`: `i = i + 1`, or `t = i + 1; i = t` with `t`
        read by nothing else."""
        bid, k, instr = site
        instrs = func.blocks[bid].instructions

        def plus_one(op):
            return (type(op) is BinaryOp and op.op == '+'
                    and type(op.left) is VirtualRegister and op.left.id == vid
                    and _const(op.right) == 1 and _is_u16(op.type_info))

        if plus_one(instr) and type(instr.dest) is VirtualRegister and instr.dest.id == vid:
            return instr, None
        if (type(instr) is Move and type(instr.source) is VirtualRegister and k > 0
                and plus_one(instrs[k - 1])
                and type(instrs[k - 1].dest) is VirtualRegister
                and instrs[k - 1].dest.id == instr.source.id):
            temp = instr.source.id
            others = [i for b in func.blocks.values() for i in b.instructions
                      if i is not instr and any(type(v) is VirtualRegister and v.id == temp
                                                for _, v in iter_operands(i, role=OperandRole.READ))]
            if not others:
                return instrs[k - 1], instr
        return None

    @staticmethod
    def _branch_follows(func: MIRFunction, bid: int, k: int, cmp: Compare) -> bool:
        """The Compare is read by the block's closing CondBranch, which tests
        the same comparison."""
        instrs = func.blocks[bid].instructions
        return (k + 1 < len(instrs) and type(instrs[k + 1]) is CondBranch
                and instrs[k + 1].condition is None
                and instrs[k + 1].comparison == cmp.comparison)

    @staticmethod
    def _forward_copy(func: MIRFunction, bid: int, copy: Move, vid: int) -> None:
        """Replace the reads of `copy.dest` with the counter and drop the copy,
        when its one def is this copy and every read follows it in the block
        before the counter is written again."""
        dest = copy.dest.id
        instrs = func.blocks[bid].instructions
        k = instrs.index(copy)
        later_reads = []
        for j in range(k + 1, len(instrs)):
            instr = instrs[j]
            if any(type(v) is VirtualRegister and v.id == dest
                   for _, v in iter_operands(instr, role=OperandRole.READ)):
                later_reads.append(instr)
            if any(type(v) is VirtualRegister and v.id in (dest, vid)
                   for _, v in iter_operands(instr, role=OperandRole.WRITE)):
                break
        total = sum(1 for b in func.blocks.values() for i in b.instructions
                    for role in (OperandRole.READ, OperandRole.WRITE)
                    for _, v in iter_operands(i, role=role)
                    if type(v) is VirtualRegister and v.id == dest)
        # The copy's own write plus the reads found here: nothing else.
        if total != 1 + sum(1 for i in later_reads for _, v in iter_operands(i, role=OperandRole.READ)
                            if type(v) is VirtualRegister and v.id == dest):
            return
        for instr in later_reads:
            map_operands(instr, lambda v: copy.source
                         if type(v) is VirtualRegister and v.id == dest else v,
                         role=OperandRole.READ)
        del instrs[k]
