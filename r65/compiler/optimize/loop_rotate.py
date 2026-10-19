# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Loop Rotation.

A `while` loop lowers top-tested: the header compares, the body jumps back
to it. Laid out, every iteration runs the body's `BRA header` as well as
the compare's branch::

    P:  i = 0; Jump H             P:  i = 0; Jump B
    H:  Compare i < N
        CondBranch B : E   ->
    B:  ...; Jump H               B:  ...; Compare i < N
                                      CondBranch B : E

When the header's test is known to pass on entry - the counter is set to
a constant just before the loop - the entry edges skip it and go to the
body, and the test moves to the end of each block that jumped to it (the
body's end, each `continue`): one branch per iteration instead of two. A
test kept as a block of its own would be entered in the function's mode
rather than the width the body ends in. The rest is codegen's: each edge
is explicit and each block is entered in the mode its predecessors leave
in.

A rotated loop whose counter steps up by one, is read only by its own
test and is dead once the loop ends counts down instead - from N - init
to zero, tested `!= 0` - which drops the compare's operand (`DEY; BNE`
rather than `INY; CPY #N; BCC`).
"""

from copy import copy
from typing import Dict, List, Optional, Set

from r65.compiler.analysis.loop_analysis import find_loops
from r65.compiler.hir.types import BasicTypeInfo, strip_newtype
from r65.compiler.mir.liveness import LivenessAnalyzer, liveness_defs, liveness_uses
from r65.compiler.mir.mode_tracker import reanalyze_function
from r65.compiler.mir.nodes import (
    BinaryOp, Compare, CondBranch, Immediate, Jump, MIRFunction, MIRProgram, Move,
    VirtualRegister, iter_block_targets, map_block_targets,
)
from r65.compiler.typeck.type_utils import get_type_range

# The test is copied into each block that jumps to it, up to this many.
_MAX_TEST_COPIES = 4

_HOLDS = {
    '<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
    '>': lambda a, b: a > b, '>=': lambda a, b: a >= b,
    '==': lambda a, b: a == b, '!=': lambda a, b: a != b,
}


def _reads(instr, vreg_id: int) -> bool:
    return any(type(v) is VirtualRegister and v.id == vreg_id
               for v in liveness_uses(instr))


def _writes(instr, vreg_id: int) -> bool:
    return any(type(v) is VirtualRegister and v.id == vreg_id
               for v in liveness_defs(instr))


class LoopRotator:
    """Rotate loops whose first test always passes; count some down."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.rotated = 0
        self.counted_down = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rotate every eligible loop. Returns how many were rotated."""
        self.rotated = self.counted_down = 0
        for func in mir_program.functions:
            # One entry per back edge: a `continue` makes a second one.
            loops: Dict[int, Set[int]] = {}
            for header_id, body in find_loops(func):
                loops.setdefault(header_id, set()).update(body)
            for header_id, body in loops.items():
                # Without the test block of an inner loop rotated before it.
                self._rotate(func, header_id, {b for b in body if b in func.blocks})
        return self.rotated

    def _rotate(self, func: MIRFunction, header_id: int, body: Set[int]) -> None:
        header = func.blocks.get(header_id)
        test = self._test(header, body)
        if test is None:
            return
        cmp, branch = test
        counter = cmp.left.id
        entries = [p for p in header.predecessors if p not in body]
        if not entries:
            return
        inits = {}
        for pred_id in entries:
            init = self._entry_init(func.blocks[pred_id], counter)
            if init is None or not self._holds(cmp, init.source.value):
                return
            inits[pred_id] = init

        first = branch.true_target
        for pred_id in entries:
            self._retarget(func, pred_id, header_id, first)
        if not reanalyze_function(func):
            for pred_id in entries:
                self._retarget(func, pred_id, first, header_id)
            reanalyze_function(func)
            return
        self.rotated += 1
        if self.verbose:
            print(f"  loop-rotate: {func.name} B{header_id} entered at B{first}")
        self._count_down(func, header_id, body, cmp, branch, inits)
        self._merge_test(func, header_id)

    @staticmethod
    def _merge_test(func: MIRFunction, header_id: int) -> None:
        """Give each block that jumps to the test - the body's end and each
        `continue` - its own copy of it. A block of its own would be entered
        in the function's mode rather than the width the body ends in, with
        a label in front of the compare."""
        header = func.blocks[header_id]
        preds = [func.blocks[p] for p in header.predecessors]
        if not preds or len(preds) > _MAX_TEST_COPIES or not all(
                pred.instructions and type(pred.instructions[-1]) is Jump
                and pred.instructions[-1].target == header_id for pred in preds):
            return
        for pred in preds:
            pred.instructions[-1:] = [copy(instr) for instr in header.instructions]
            pred.successors = list(header.successors)
        for succ_id in header.successors:
            succ = func.blocks[succ_id]
            succ.predecessors = [p for p in succ.predecessors if p != header_id]
            succ.predecessors.extend(pred.block_id for pred in preds)
        del func.blocks[header_id]
        reanalyze_function(func)

    @staticmethod
    def _test(header, body: Set[int]):
        """The header's (Compare, CondBranch) if it is nothing else: a
        counter against a constant, entering the loop when it holds."""
        if header is None or len(header.instructions) != 2:
            return None
        cmp, branch = header.instructions
        if not (type(cmp) is Compare and type(branch) is CondBranch
                and branch.condition is None
                and branch.comparison == cmp.comparison in _HOLDS
                and type(cmp.left) is VirtualRegister
                and type(cmp.right) is Immediate and cmp.right.symbol is None
                and isinstance(cmp.right.value, int)):
            return None
        if (branch.true_target not in body or branch.true_target == header.block_id
                or branch.false_target in body):
            return None
        return cmp, branch

    @staticmethod
    def _entry_init(pred, counter: int) -> Optional[Move]:
        """The `counter = constant` that last writes the counter in *pred*,
        with nothing reading or writing it after; None if there is none."""
        for instr in reversed(pred.instructions):
            if _writes(instr, counter):
                if (type(instr) is Move and type(instr.source) is Immediate
                        and instr.source.symbol is None
                        and isinstance(instr.source.value, int)):
                    return instr
                return None
            if _reads(instr, counter):
                return None
        return None

    @staticmethod
    def _type_range(type_info):
        payload = strip_newtype(type_info)
        if not isinstance(payload, BasicTypeInfo) or payload.name == 'bool':
            return None
        return get_type_range(payload.name)

    def _holds(self, cmp: Compare, init: int) -> bool:
        """Whether `init <cmp> bound` is true, as the compare evaluates it."""
        bounds = self._type_range(cmp.type_info)
        if bounds is None:
            return False
        low, high = bounds
        span = high - low + 1

        def value(v):
            v %= span
            return v - span if v > high else v

        return _HOLDS[cmp.comparison](value(init), value(cmp.right.value))

    @staticmethod
    def _retarget(func: MIRFunction, pred_id: int, old: int, new: int) -> None:
        pred = func.blocks[pred_id]
        for instr in pred.instructions:
            map_block_targets(instr, lambda t: new if t == old else t)
        pred.successors = list(dict.fromkeys(
            new if s == old else s for s in pred.successors))
        func.blocks[old].predecessors = [
            p for p in func.blocks[old].predecessors if p != pred_id]
        if pred_id not in func.blocks[new].predecessors:
            func.blocks[new].predecessors.append(pred_id)

    # ------------------------------------------------------------------
    # Count-down
    # ------------------------------------------------------------------

    def _count_down(self, func: MIRFunction, header_id: int, body: Set[int],
                    cmp: Compare, branch: CondBranch,
                    inits: Dict[int, Move]) -> None:
        """`i = init; .. i < N .. i + 1` as `i = N - init; .. i != 0 .. i - 1`.

        The two agree while the counter is between init and N at each test,
        which holds when it steps by one at most once per pass: its one
        update is in no inner loop. Unsigned only: a signed test
        against 0 branches on the sign.
        """
        bounds = self._type_range(cmp.type_info)
        if cmp.comparison != '<' or bounds is None or bounds[0] < 0:
            return
        counter = cmp.left.id
        # The new initial value must reach only the loop.
        if any(func.blocks[p].successors != [branch.true_target] for p in inits):
            return
        step = self._single_step(func, header_id, body, counter, branch.true_target)
        if step is None:
            return
        # Dead wherever the loop exits to.
        exits = {t for bid in body for instr in func.blocks[bid].instructions
                 for t in iter_block_targets(instr) if t not in body}
        live = LivenessAnalyzer(func).analyze()
        if any(counter in {v.id for v in live[e].live_in
                           if type(v) is VirtualRegister}
               for e in exits if e in live):
            return

        bound = cmp.right.value
        for init in inits.values():
            init.source = Immediate(bound - init.source.value)
        step.op = '-'
        cmp.right = Immediate(0)
        cmp.comparison = branch.comparison = '!='
        self.counted_down += 1
        if self.verbose:
            print(f"  loop-rotate: {func.name} B{header_id} counts down")

    @staticmethod
    def _single_step(func: MIRFunction, header_id: int, body: Set[int],
                     counter: int, first: int) -> Optional[BinaryOp]:
        """The loop's one `counter + 1`, if that and the test are the
        counter's only uses in it - `t = i + 1; i = t` included - and no
        inner loop contains it; else None."""
        step = step_block = None
        for bid in body:
            instrs = func.blocks[bid].instructions
            for k, instr in enumerate(instrs):
                if not (_reads(instr, counter) or _writes(instr, counter)):
                    continue
                if bid == header_id and type(instr) is Compare:
                    continue
                if (step is None and type(instr) is BinaryOp and instr.op == '+'
                        and type(instr.left) is VirtualRegister
                        and instr.left.id == counter
                        and type(instr.right) is Immediate
                        and instr.right.symbol is None and instr.right.value == 1
                        and type(instr.dest) is VirtualRegister):
                    if instr.dest.id == counter:
                        step, step_block = instr, bid
                        continue
                    nxt = instrs[k + 1] if k + 1 < len(instrs) else None
                    if (type(nxt) is Move and type(nxt.dest) is VirtualRegister
                            and nxt.dest.id == counter
                            and type(nxt.source) is VirtualRegister
                            and nxt.source.id == instr.dest.id
                            and _only_read_by(func, instr.dest.id, nxt)):
                        step, step_block = instr, bid
                        continue
                if (step is not None and step.dest.id != counter
                        and type(instr) is Move and type(instr.dest) is VirtualRegister
                        and instr.dest.id == counter
                        and type(instr.source) is VirtualRegister
                        and instr.source.id == step.dest.id):
                    continue  # the step's copy back
                return None
        if step is None:
            return None
        # An inner loop's header is in this one; rotated, this loop's own
        # back edge goes to *first*.
        for inner_header, inner_body in find_loops(func):
            if (inner_header != first and inner_header in body
                    and step_block in inner_body):
                return None
        return step


def _only_read_by(func: MIRFunction, vreg_id: int, reader) -> bool:
    """Whether *reader* is the one instruction in *func* reading *vreg_id*."""
    for block in func.blocks.values():
        for instr in block.instructions:
            if instr is not reader and _reads(instr, vreg_id):
                return False
    return True
