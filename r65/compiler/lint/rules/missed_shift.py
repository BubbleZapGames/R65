"""L004: missed_shift — warn when `x * C` or `x / C` uses a power-of-2 constant."""

from r65.compiler.hir import HIRBinaryOp, HIRIntegerLiteral
from r65.compiler.lint.rule import LintContext, LintRule


def _is_pow2(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


def _shift_amount(n: int) -> int:
    return n.bit_length() - 1


class MissedShift(LintRule):
    """Warn when multiplication or division by a power-of-2 literal could be
    written as a shift. R65 already restricts ``*``/``/`` to power-of-2
    constants for a reason; making that restriction visible at the source
    level via ``<<``/``>>`` documents the hardware intent and matches the
    generated ASL/LSR directly.
    """

    def __init__(self):
        super().__init__(
            code="L004",
            name="missed_shift",
            description="power-of-2 multiply/divide could be a shift",
        )

    def visit_binary_op(self, expr: HIRBinaryOp, ctx: LintContext) -> None:
        if expr.op not in ("*", "/"):
            return
        const_literal = None
        if isinstance(expr.right, HIRIntegerLiteral):
            const_literal = expr.right
        elif expr.op == "*" and isinstance(expr.left, HIRIntegerLiteral):
            # Division is not commutative — only multiplication can have the
            # constant on the left.
            const_literal = expr.left
        if const_literal is None:
            return
        value = const_literal.value
        if not _is_pow2(value) or value > 256:
            return
        shift = _shift_amount(value)
        shift_op = "<<" if expr.op == "*" else ">>"
        ctx.emit(
            code=self.code,
            message=f"`{expr.op} {value}` could be written as `{shift_op} {shift}`",
            source_loc=expr.source_loc,
            hint=f"use `x {shift_op} {shift}` — same cost, clearer hardware intent",
        )
