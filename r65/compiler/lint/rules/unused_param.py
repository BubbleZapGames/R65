"""L008: unused_param — warn on function parameters that are never read."""

from r65.compiler.hir import HIRFunctionDecl, HIRIdentifier
from r65.compiler.lint.rule import LintContext, LintRule


class UnusedParam(LintRule):
    """Warn when a function parameter is never read in the body.

    Sister rule to L002 ``unused_binding``, but operates on parameter
    symbols. Names starting with ``_`` are exempt — same convention as L002.

    For register-bound parameters (``param @ A: u8``), the symbol named
    ``param`` is what users reference in the body. If the user reads ``A``
    directly without ever naming ``param``, this rule still fires — that's
    the right call: the named parameter is dead even if the underlying
    register is being used elsewhere.
    """

    def __init__(self):
        super().__init__(
            code="L008",
            name="unused_param",
            description="function parameter is never read",
        )
        self._declared: dict = {}  # id(symbol) -> (name, source_loc)
        self._read: set = set()     # id(symbol)

    def enter_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        self._declared = {}
        self._read = set()
        for param in func.parameters:
            if param.symbol is None:
                continue
            if param.name.startswith("_"):
                continue
            loc = param.source_loc or func.source_loc
            self._declared[id(param.symbol)] = (param.name, loc)

    def visit_identifier(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        if expr.symbol is not None:
            self._read.add(id(expr.symbol))

    def leave_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        for sym_id, (name, loc) in self._declared.items():
            if sym_id not in self._read:
                ctx.emit(
                    code=self.code,
                    message=f"unused parameter `{name}`",
                    source_loc=loc,
                    hint=f"if this is intentional, prefix with `_`: `_{name}`",
                )
        self._declared = {}
        self._read = set()
