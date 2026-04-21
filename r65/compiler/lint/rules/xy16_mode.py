"""L009: xy16_mode — warn on `STATUS.XY16 = false` writes."""

from r65.compiler.hir import (
    HIRAssignment,
    HIRBooleanLiteral,
    HIRStatusFlagAccess,
)
from r65.compiler.lint.rule import LintContext, LintRule


class Xy16Mode(LintRule):
    """Warn when code drops the CPU into 8-bit index (x8) mode.

    R65 assumes X/Y are always 16-bit. Compiler-generated code — array
    indexing, loop counters, struct/pointer access, function call/return —
    silently breaks if executed while ``STATUS.XY16 = false``.

    Legitimate uses (DP-indexed reads inside a tight ``asm!`` block) are
    rare but valid; this rule flags every entry into x8 so it's visible in
    review. The typeck-level ``E-XY16-REGION`` pass enforces the actual
    safety envelope (no calls, no control flow, paired restore).
    """

    def __init__(self):
        super().__init__(
            code="L009",
            name="xy16_mode",
            description="entering 8-bit index mode (STATUS.XY16 = false)",
        )

    def visit_assignment(self, expr: HIRAssignment, ctx: LintContext) -> None:
        target = expr.target
        value = expr.value
        if not isinstance(target, HIRStatusFlagAccess):
            return
        if target.flag_name != "XY16":
            return
        if not isinstance(value, HIRBooleanLiteral) or value.value is not False:
            return
        ctx.emit(
            code=self.code,
            message="entering 8-bit index mode — X/Y are assumed 16-bit by R65 codegen",
            source_loc=expr.source_loc or target.source_loc,
            hint="pair with `STATUS.XY16 = true` before any compiler-generated X/Y code (calls, indexing, control flow)",
        )
