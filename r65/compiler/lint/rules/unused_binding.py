"""L002: unused_binding — warn on `let x = …` where x is never read."""

from r65.compiler.hir import HIRFunctionDecl, HIRIdentifier, HIRLetStmt
from r65.compiler.lint.rule import LintContext, LintRule


class UnusedBinding(LintRule):
    """Warn when a local binding is never read.

    Names starting with `_` are exempt (convention: intentional unused).
    Compound assignment targets (``x += 1``) still count as reads because the
    RHS contains an identifier reference to ``x`` after desugaring. Pure
    assignment targets (``x = 5``) do NOT count because the walker suppresses
    ``visit_identifier`` on plain-identifier assignment targets.
    """

    def __init__(self):
        super().__init__(
            code="L002",
            name="unused_binding",
            description="local `let` binding is never read",
        )
        self._declared: dict = {}  # id(symbol) -> (name, source_loc)
        self._read: set = set()     # id(symbol)

    def enter_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        self._declared = {}
        self._read = set()

    def visit_let(self, stmt: HIRLetStmt, ctx: LintContext) -> None:
        if stmt.symbol is None:
            return
        if stmt.name.startswith("_"):
            return
        self._declared[id(stmt.symbol)] = (stmt.name, stmt.source_loc)

    def visit_identifier(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        if expr.symbol is not None:
            self._read.add(id(expr.symbol))

    def leave_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        for sym_id, (name, loc) in self._declared.items():
            if sym_id not in self._read:
                ctx.emit(
                    code=self.code,
                    message=f"unused variable `{name}`",
                    source_loc=loc,
                    hint=f"if this is intentional, prefix with `_`: `_{name}`",
                )
        self._declared = {}
        self._read = set()
