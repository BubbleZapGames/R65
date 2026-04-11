"""L003: unreachable_code — warn on statements after a terminator in the same block."""

from r65.compiler.hir import (
    HIRBlock,
    HIRBreakStmt,
    HIRContinueStmt,
    HIRReturnStmt,
)
from r65.compiler.lint.rule import LintContext, LintRule


class UnreachableCode(LintRule):
    """Warn when a statement follows a ``return``/``break``/``continue`` in the
    same block. Emits only on the first statement after the terminator to
    avoid noise. More subtle cases (e.g. both branches of an ``if`` returning)
    are left for a future pass that tracks block termination.
    """

    def __init__(self):
        super().__init__(
            code="L003",
            name="unreachable_code",
            description="code after `return`/`break`/`continue` is unreachable",
        )

    def visit_block(self, block: HIRBlock, ctx: LintContext) -> None:
        stmts = block.statements
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, (HIRReturnStmt, HIRBreakStmt, HIRContinueStmt)):
                if i + 1 < len(stmts):
                    next_stmt = stmts[i + 1]
                    ctx.emit(
                        code=self.code,
                        message="unreachable statement",
                        source_loc=next_stmt.source_loc or stmt.source_loc,
                        hint="the previous statement transfers control out of this block",
                    )
                return
