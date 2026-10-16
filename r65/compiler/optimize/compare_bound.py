# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Constant-Bound Compare Pass.

Rewrites `x > K` to `x >= K+1` and `x <= K` to `x < K+1` when K is a
constant whose successor still fits the compared type::

    Compare x > #5      ->   Compare x >= #6
    CondBranch >             CondBranch >=

`>` and `<=` need two branches on the 65816 (`BEQ` for the equal case,
then `BCS`/`BCC`), where `>=` and `<` need one, so the rewrite saves a
branch on every such test. It applies to signed compares as well, which
against a constant are biased to an unsigned compare; `x > -1` becomes
`x >= 0`, a single `BPL`.

K = 0 is left alone: unsigned `x > 0` is already one `BNE` with no compare,
and a signed compare with 0 branches on the sign without one. So is a K at
the type's maximum, whose successor does not exist.
"""

from r65.compiler.hir.types import BasicTypeInfo, strip_newtype
from r65.compiler.mir.nodes import Compare, CondBranch, Immediate, MIRProgram
from r65.compiler.typeck.type_utils import get_type_range

_BUMPED = {'>': '>=', '<=': '<'}


class CompareBoundFolder:
    """Turn a two-branch `>`/`<=` against a constant into one-branch form."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.folded = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """Rewrite every eligible compare in the program. Returns the count."""
        self.folded = 0
        for func in mir_program.functions:
            for block in func.blocks.values():
                instrs = block.instructions
                if len(instrs) >= 2:
                    self._fold(instrs[-2], instrs[-1])
        return self.folded

    def _fold(self, cmp, branch) -> None:
        if not (type(cmp) is Compare and type(branch) is CondBranch
                and branch.condition is None
                and cmp.comparison == branch.comparison in _BUMPED
                and type(cmp.right) is Immediate and isinstance(cmp.right.value, int)
                and type(cmp.left) is not Immediate):
            return
        type_info = strip_newtype(cmp.type_info)
        if not isinstance(type_info, BasicTypeInfo):
            return
        bounds = get_type_range(type_info.name)
        if bounds is None or type_info.name == 'bool':
            return
        low, high = bounds
        k = cmp.right.value
        if low < 0 and k > high:
            k -= high - low + 1  # A signed constant held as its bit pattern
        if k == 0 or not low <= k < high:
            return
        new_op = _BUMPED[cmp.comparison]
        if self.verbose:
            print(f"  compare-bound: {cmp.left} {cmp.comparison} {k} -> {new_op} {k + 1}")
        cmp.right = Immediate(k + 1)
        cmp.comparison = branch.comparison = new_op
        self.folded += 1
