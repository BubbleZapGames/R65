"""
LintRule base class and LintContext.

Rules subclass :class:`LintRule`, override the relevant ``visit_*`` hooks, and
append diagnostics to the :class:`LintContext` supplied by the walker. Every
hook defaults to a no-op so rules only implement what they need.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from r65.compiler.errors import (
    Diagnostic,
    DiagnosticCollector,
    DiagnosticSeverity,
    SourceLocation,
)

if TYPE_CHECKING:
    from r65.compiler.hir import (
        HIRAsmStmt,
        HIRAssignment,
        HIRBinaryOp,
        HIRBlock,
        HIRFunctionDecl,
        HIRIdentifier,
        HIRIfStmt,
        HIRLetStmt,
        HIRProgram,
        HIRReturnStmt,
        HIRStaticDecl,
        HIRWhileStmt,
    )


@dataclass
class LintContext:
    """Shared state passed to every rule hook during a lint walk."""

    program: "HIRProgram"
    diagnostics: DiagnosticCollector = field(default_factory=DiagnosticCollector)
    current_function: Optional["HIRFunctionDecl"] = None
    _call_graph: Optional[object] = None  # Lazily built

    @property
    def call_graph(self):
        """Lazily built HIR call graph, shared across all rules in one lint run."""
        if self._call_graph is None:
            from r65.compiler.lint.call_graph import build_call_graph
            self._call_graph = build_call_graph(self.program)
        return self._call_graph

    def emit(
        self,
        code: str,
        message: str,
        source_loc: Optional[SourceLocation] = None,
        hint: Optional[str] = None,
        severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
    ) -> None:
        self.diagnostics.add(
            Diagnostic(
                severity=severity,
                message=message,
                source_loc=source_loc,
                code=code,
                hint=hint,
            )
        )


@dataclass
class LintRule:
    """Base class for all lint rules.

    Every visit hook defaults to a no-op. Rules override only the hooks they
    care about. The walker in :class:`R65Linter` dispatches to every enabled
    rule's hooks at each node in a single HIR traversal.
    """

    code: str = ""
    name: str = ""
    description: str = ""
    default_enabled: bool = True

    def setup(self, ctx: LintContext) -> None:
        """Called once before the HIR walk begins. Use for rule-level
        precomputation that needs access to the full program (e.g. building
        a set of forbidden symbol ids, snapshotting the call graph)."""
        pass

    def enter_function(self, func: "HIRFunctionDecl", ctx: LintContext) -> None:
        pass

    def leave_function(self, func: "HIRFunctionDecl", ctx: LintContext) -> None:
        pass

    def visit_block(self, block: "HIRBlock", ctx: LintContext) -> None:
        pass

    def visit_if(self, stmt: "HIRIfStmt", ctx: LintContext) -> None:
        pass

    def visit_while(self, stmt: "HIRWhileStmt", ctx: LintContext) -> None:
        pass

    def visit_let(self, stmt: "HIRLetStmt", ctx: LintContext) -> None:
        pass

    def visit_return(self, stmt: "HIRReturnStmt", ctx: LintContext) -> None:
        pass

    def visit_asm(self, stmt: "HIRAsmStmt", ctx: LintContext) -> None:
        pass

    def visit_identifier(self, expr: "HIRIdentifier", ctx: LintContext) -> None:
        """Called for every ``HIRIdentifier`` used as a READ.

        Plain-identifier assignment targets are dispatched to
        :meth:`visit_identifier_write` instead, so rules that only care about
        reads (e.g. L002 unused_binding) don't accidentally count writes."""
        pass

    def visit_identifier_write(
        self, expr: "HIRIdentifier", ctx: LintContext
    ) -> None:
        """Called for plain-identifier assignment targets (``x = 5``).

        Compound lvalues like ``arr[i] = 5`` or ``*ptr = 5`` do NOT fire this
        hook — their subexpressions are walked normally and the inner
        identifiers come through as reads via :meth:`visit_identifier`."""
        pass

    def visit_assignment(self, expr: "HIRAssignment", ctx: LintContext) -> None:
        pass

    def visit_binary_op(self, expr: "HIRBinaryOp", ctx: LintContext) -> None:
        pass

    def visit_static_decl(self, decl: "HIRStaticDecl", ctx: LintContext) -> None:
        pass

    def finalize(self, ctx: LintContext) -> None:
        """Called once after the full HIR walk completes. Use for program-level
        emit passes that need state collected across every function body."""
        pass
