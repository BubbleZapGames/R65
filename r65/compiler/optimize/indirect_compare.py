# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Indirect Compare Folding Pass.

Rewrites a byte loaded through a pointer parameter, when every reader is the
right-hand side of a Compare, into those compares' operand, so the CMP reads
it from memory::

    Y = Move %i
    %ca = LoadIndirect [%a],Y            Y = Move %i
    Y = Move %i                          %ca = LoadIndirect [%a],Y
    %cb = LoadIndirect [%b],Y     ->     Y = Move %i
    Compare %ca < %cb                    CompareIndirect %ca < [%b],Y
    CondBranch ...                       CondBranch ...
  next block:                          next block:
    Compare %ca > %cb                    CompareIndirect %ca > [%b],Y

which is what a hand-written strcmp loop looks like::

    LDA [a],Y
    CMP [b],Y                    ; was: LDA [a],Y / STA $03,S
    BCC less                     ;      LDA [b],Y / STA $05,S
                                 ;      LDA $03,S / CMP $05,S ...

As with memory-operand folding (memory_operand.py), most of the win is that
`%cb`'s load no longer overwrites A after `%ca` is defined, so `%ca` can stay
in A instead of both spilling to stack slots. A second compare reads the
byte again rather than keeping a copy, so a value read by two compares costs
two 7-cycle CMPs instead of a load, a store and two stack CMPs.

Only a 1-byte unsigned compare (or ==/!=) through an indexed pointer
parameter folds. A parameter is the pointer every addressing path takes in
one instruction: on the stack it is a far pointer the function's D=S or
SET_DBR strategy covers, or a near one read `(d,S),Y`; promoted, it is in
zero page. A local far pointer may instead need a PHB/PLB bracket, whose
PLB would destroy the flags. A signed ordered compare is an SBC, not a CMP.

Reading the byte later is only legal if nothing between the load and each
compare can change it, the pointer, or Y. Each compare must be in the load's
block or in one reached from it only through blocks with a single
predecessor, and the instructions on the way may be only compares and
branches. Not even a register move: loop promotion, after this pass, may pin
the vreg it defines to Y, and that pin is checked against Y's explicit writes
only, not against an index a compare further on still needs. See `_may_pass`.
"""

from typing import Dict, List, Optional, Set, Tuple

from r65.compiler.codegen.type_utils import is_signed_scalar
from r65.compiler.mir.nodes import (
    Compare, CompareIndirect, CondBranch, Jump, LoadIndirect, MIRFunction,
    MIRProgram, VirtualRegister, iter_operands, OperandRole,
)

Position = Tuple[int, int]  # (block id, instruction index)


def _type_size(type_info) -> Optional[int]:
    size = getattr(type_info, 'size_bytes', None)
    return size if isinstance(size, int) else None


class IndirectCompareFolder:
    """Fold pointer loads read only by compares into `CompareIndirect`."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rewrite every foldable load in the program. Returns the count."""
        self.folded = 0
        for func in mir_program.functions:
            self._optimize_function(func)
        return self.folded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _optimize_function(self, func: MIRFunction) -> None:
        # A trait method's self arrives in Y, which the index would share.
        if func.self_y_vreg is not None:
            return
        params = {v.id for v in func.param_to_vreg.values()}
        if not params:
            return

        defs: Dict[int, int] = {}
        uses: Dict[int, List[Position]] = {}
        for bid, block in func.blocks.items():
            for idx, instr in enumerate(block.instructions):
                for _, v in iter_operands(instr, role=OperandRole.WRITE):
                    if isinstance(v, VirtualRegister):
                        defs[v.id] = defs.get(v.id, 0) + 1
                for _, v in iter_operands(instr, role=OperandRole.READ):
                    if isinstance(v, VirtualRegister):
                        uses.setdefault(v.id, []).append((bid, idx))

        preds: Dict[int, Set[int]] = {bid: set() for bid in func.blocks}
        for bid, block in func.blocks.items():
            for succ in block.successors:
                preds.setdefault(succ, set()).add(bid)

        # Decide every fold before rewriting, so positions stay valid.
        folds = []
        for bid, block in func.blocks.items():
            for idx, instr in enumerate(block.instructions):
                if not self._is_candidate(instr, params, defs):
                    continue
                readers = uses.get(instr.dest.id, [])
                if readers and all(
                        self._use_folds(func, preds, (bid, idx), pos, instr)
                        for pos in readers):
                    folds.append(((bid, idx), readers))

        dead: Dict[int, Set[int]] = {}
        for (bid, idx), readers in folds:
            load = func.blocks[bid].instructions[idx]
            for ubid, uidx in readers:
                cmp = func.blocks[ubid].instructions[uidx]
                func.blocks[ubid].instructions[uidx] = CompareIndirect(
                    left=cmp.left, pointer=load.pointer, is_far=load.is_far,
                    comparison=cmp.comparison, type_info=cmp.type_info,
                    index_register=load.index_register, offset=load.offset,
                    source_loc=cmp.source_loc, comment=cmp.comment)
            dead.setdefault(bid, set()).add(idx)
            self.folded += 1
            if self.verbose:
                print(f"  indirect-compare: {load} folded into {len(readers)} compare(s)")
        for bid, idxs in dead.items():
            block = func.blocks[bid]
            block.instructions = [i for k, i in enumerate(block.instructions)
                                  if k not in idxs]

    @staticmethod
    def _is_candidate(instr, params: Set[int], defs: Dict[int, int]) -> bool:
        """A byte load through an indexed pointer parameter, defined once."""
        if not isinstance(instr, LoadIndirect):
            return False
        dest = instr.dest
        if not isinstance(dest, VirtualRegister) or dest.register_hint:
            return False
        if defs.get(dest.id, 0) != 1:
            return False
        if instr.pointer.id not in params:
            return False
        if instr.index_register != 'Y' or instr.offset:
            return False
        return _type_size(instr.type_info) == 1

    def _use_folds(self, func: MIRFunction, preds: Dict[int, Set[int]],
                   load_pos: Position, use_pos: Position, load: LoadIndirect) -> bool:
        """Whether the reader at *use_pos* is a Compare that can read the byte
        itself, with nothing on the way from the load able to change it."""
        ubid, uidx = use_pos
        cmp = func.blocks[ubid].instructions[uidx]
        if type(cmp) is not Compare:
            return False
        if not (isinstance(cmp.right, VirtualRegister) and cmp.right.id == load.dest.id):
            return False
        # The left goes in A for the CMP; X/Y have no `CPX [dp],Y`.
        left = cmp.left
        if not isinstance(left, VirtualRegister) or left.register_hint:
            return False
        if left.id == load.dest.id:
            return False
        if _type_size(cmp.type_info) != 1:
            return False
        if is_signed_scalar(cmp.type_info) and cmp.comparison not in ('==', '!='):
            return False
        path = self._path(func, preds, load_pos, use_pos)
        return path is not None and all(self._may_pass(i) for i in path)

    @staticmethod
    def _path(func: MIRFunction, preds: Dict[int, Set[int]],
              load_pos: Position, use_pos: Position) -> Optional[List]:
        """The instructions run between the load and the use, if every way to
        the use goes through the load: the use is later in the load's block,
        or in a block reached from it through blocks of one predecessor."""
        lbid, lidx = load_pos
        ubid, uidx = use_pos
        if ubid == lbid:
            if uidx <= lidx:
                return None
            return func.blocks[lbid].instructions[lidx + 1:uidx]
        chain = []
        bid = ubid
        seen = set()
        while bid != lbid:
            if bid in seen:
                return None
            seen.add(bid)
            if len(preds.get(bid, ())) != 1:
                return None
            chain.append(bid)
            (bid,) = preds[bid]
        path = list(func.blocks[lbid].instructions[lidx + 1:])
        for bid in reversed(chain[1:]):
            path.extend(func.blocks[bid].instructions)
        path.extend(func.blocks[ubid].instructions[:uidx])
        return path

    @staticmethod
    def _may_pass(instr) -> bool:
        """True if the byte may be read *after* this instruction instead of
        before it: it writes no memory, no register and no vreg - so neither
        the pointer nor Y can change, now or once registers are allocated."""
        return isinstance(instr, (Compare, CompareIndirect, CondBranch, Jump))
