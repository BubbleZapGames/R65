"""L006: empty_block — warn on empty `if`/`else` bodies."""

from r65.compiler.hir import HIRBlock, HIRIfStmt
from r65.compiler.lint.rule import LintContext, LintRule


class EmptyBlock(LintRule):
    """Warn on `if cond { }` and `else { }` with no statements.

    Usually a leftover after editing. We only flag the bodies of `if`/`else` —
    empty function bodies and empty `while` loops are intentional in some
    patterns (e.g. busy-wait polling).
    """

    def __init__(self):
        super().__init__(
            code="L006",
            name="empty_block",
            description="empty `if` or `else` body",
        )

    def visit_if(self, stmt: HIRIfStmt, ctx: LintContext) -> None:
        if stmt.then_block is not None and not stmt.then_block.statements:
            ctx.emit(
                code=self.code,
                message="empty `if` body",
                source_loc=stmt.then_block.source_loc or stmt.source_loc,
                hint="remove the empty block or add the intended statements",
            )
        else_block = stmt.else_block
        if isinstance(else_block, HIRBlock) and not else_block.statements:
            ctx.emit(
                code=self.code,
                message="empty `else` body",
                source_loc=else_block.source_loc or stmt.source_loc,
                hint="remove the empty block or add the intended statements",
            )
