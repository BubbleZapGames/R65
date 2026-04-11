"""L001: unused_mut — warn on `let mut x = …` where x is never reassigned."""

from r65.compiler.hir import HIRAssignment, HIRFunctionDecl, HIRIdentifier, HIRLetStmt
from r65.compiler.lint.rule import LintContext, LintRule


class UnusedMut(LintRule):
    """Warn when a `let mut` binding is never actually mutated.

    Compound assignments (`x += 1`), postfix increments (`x++`), and
    decrements (`x--`) all desugar to `HIRAssignment` earlier in the pipeline,
    so a single `visit_assignment` hook catches all three forms.

    Writes *through* a binding (array element, struct field, pointer deref)
    do not count as reassigning the binding itself — per R65 semantics those
    writes never required `mut` in the first place.
    """

    def __init__(self):
        super().__init__(
            code="L001",
            name="unused_mut",
            description="`let mut` binding is never reassigned",
        )
        # Map id(symbol) -> (name, source_loc) for mut bindings awaiting a write.
        self._pending: dict = {}

    def enter_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        self._pending = {}

    def visit_let(self, stmt: HIRLetStmt, ctx: LintContext) -> None:
        if not stmt.is_mutable or stmt.symbol is None:
            return
        self._pending[id(stmt.symbol)] = (stmt.name, stmt.source_loc)

    def visit_assignment(self, expr: HIRAssignment, ctx: LintContext) -> None:
        target = expr.target
        if isinstance(target, HIRIdentifier) and target.symbol is not None:
            self._pending.pop(id(target.symbol), None)

    def leave_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        for name, loc in self._pending.values():
            ctx.emit(
                code=self.code,
                message=f"variable `{name}` does not need to be mutable",
                source_loc=loc,
                hint="remove the `mut` keyword",
            )
        self._pending = {}
