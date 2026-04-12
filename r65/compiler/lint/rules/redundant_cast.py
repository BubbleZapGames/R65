"""L007: redundant_cast — warn on `(x as T) as T` chains."""

from r65.compiler.hir import HIRTypeCast
from r65.compiler.lint.rule import LintContext, LintRule


class RedundantCast(LintRule):
    """Warn when an outer type cast targets the same type as its inner cast.

    ``(x as u8) as u8`` is a no-op chain — the inner cast already produced
    a ``u8``, and the outer cast just re-asserts the same type. Almost
    always a refactor leftover.

    Type comparison uses ``TypeUtils.types_equal`` from the type checker so
    structural types (arrays, pointers, function types) compare correctly,
    not just basic-type names.
    """

    def __init__(self):
        super().__init__(
            code="L007",
            name="redundant_cast",
            description="cast `as T` is redundant — inner expression is already T",
        )

    def visit_type_cast(self, expr: HIRTypeCast, ctx: LintContext) -> None:
        inner = expr.expr
        if not isinstance(inner, HIRTypeCast):
            return
        outer_type = expr.target_type
        inner_type = inner.target_type
        if outer_type is None or inner_type is None:
            return
        from r65.compiler.typeck.type_utils import TypeUtils
        if not TypeUtils.types_equal(outer_type, inner_type):
            return
        ctx.emit(
            code=self.code,
            message=f"redundant cast to `{outer_type}` (inner expression is already `{inner_type}`)",
            source_loc=expr.source_loc or inner.source_loc,
            hint="remove the outer `as` cast",
        )
