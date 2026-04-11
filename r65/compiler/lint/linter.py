"""
R65Linter: orchestrates a single HIR walk and dispatches to every enabled
rule's visit hook at each relevant node.

The walker uses ``isinstance`` dispatch to match the idiom used elsewhere in
the compiler (see ``typeck/type_checker.py`` for the reference pattern). It
avoids introducing a visitor-pattern abstraction that the rest of the compiler
doesn't use.
"""

from typing import TYPE_CHECKING, List, Optional

from r65.compiler.errors import DiagnosticCollector

if TYPE_CHECKING:
    from r65.compiler.lint.config import LintConfig
from r65.compiler.hir import (
    HIRAsmStmt,
    HIRAssignment,
    HIRAddressOf,
    HIRArrayFillExpr,
    HIRArrayIndex,
    HIRArrayLiteralExpr,
    HIRBinaryOp,
    HIRBlock,
    HIRBreakStmt,
    HIRContinueStmt,
    HIRDereference,
    HIRExprStmt,
    HIRExpression,
    HIRFieldAccess,
    HIRFunctionCall,
    HIRFunctionDecl,
    HIRIdentifier,
    HIRIfStmt,
    HIRLetStmt,
    HIRMethodCall,
    HIRMultiAssignment,
    HIRProgram,
    HIRReturnStmt,
    HIRStaticDecl,
    HIRStatement,
    HIRStructLiteralExpr,
    HIRTupleLetStmt,
    HIRTypeCast,
    HIRUnaryOp,
    HIRWhileStmt,
)
from r65.compiler.hir.nodes import (
    HIRBlockExpression,
    HIRIfExpression,
    HIRLoopExpression,
    HIRMatchExpression,
    HIRMatchArm,
)

from r65.compiler.lint.rule import LintContext, LintRule


class R65Linter:
    """Walks the HIR once, calling every enabled rule's visit hooks per node."""

    def __init__(self, program: HIRProgram, rules: List[LintRule]):
        self.program = program
        self.rules = rules
        self.ctx = LintContext(program=program)

    def check(self) -> DiagnosticCollector:
        for rule in self.rules:
            rule.setup(self.ctx)
        for decl in self.program.declarations:
            self._walk_decl(decl)
        for rule in self.rules:
            rule.finalize(self.ctx)
        return self.ctx.diagnostics

    # ------------------------------------------------------------------ decls

    def _walk_decl(self, decl) -> None:
        if isinstance(decl, HIRFunctionDecl):
            self._walk_function(decl)
        elif isinstance(decl, HIRStaticDecl):
            for rule in self.rules:
                rule.visit_static_decl(decl, self.ctx)

    def _walk_function(self, func: HIRFunctionDecl) -> None:
        self.ctx.current_function = func
        for rule in self.rules:
            rule.enter_function(func, self.ctx)
        if func.body is not None:
            self._walk_block(func.body)
        for rule in self.rules:
            rule.leave_function(func, self.ctx)
        self.ctx.current_function = None

    # ------------------------------------------------------------------ stmts

    def _walk_block(self, block: HIRBlock) -> None:
        for rule in self.rules:
            rule.visit_block(block, self.ctx)
        for stmt in block.statements:
            self._walk_stmt(stmt)

    def _walk_stmt(self, stmt: HIRStatement) -> None:
        if isinstance(stmt, HIRBlock):
            self._walk_block(stmt)
        elif isinstance(stmt, HIRLetStmt):
            for rule in self.rules:
                rule.visit_let(stmt, self.ctx)
            if stmt.initializer is not None:
                self._walk_expr(stmt.initializer)
        elif isinstance(stmt, HIRTupleLetStmt):
            if stmt.initializer is not None:
                self._walk_expr(stmt.initializer)
        elif isinstance(stmt, HIRExprStmt):
            if stmt.expr is not None:
                self._walk_expr(stmt.expr)
        elif isinstance(stmt, HIRReturnStmt):
            for rule in self.rules:
                rule.visit_return(stmt, self.ctx)
            for value in stmt.values:
                self._walk_expr(value)
        elif isinstance(stmt, HIRIfStmt):
            for rule in self.rules:
                rule.visit_if(stmt, self.ctx)
            if stmt.condition is not None:
                self._walk_expr(stmt.condition)
            if stmt.then_block is not None:
                self._walk_block(stmt.then_block)
            if stmt.else_block is not None:
                if isinstance(stmt.else_block, HIRIfStmt):
                    self._walk_stmt(stmt.else_block)
                else:
                    self._walk_block(stmt.else_block)
        elif isinstance(stmt, HIRWhileStmt):
            for rule in self.rules:
                rule.visit_while(stmt, self.ctx)
            if stmt.condition is not None:
                self._walk_expr(stmt.condition)
            if stmt.body is not None:
                self._walk_block(stmt.body)
        elif isinstance(stmt, HIRAsmStmt):
            for rule in self.rules:
                rule.visit_asm(stmt, self.ctx)
        elif isinstance(stmt, (HIRBreakStmt, HIRContinueStmt)):
            pass  # leaves
        elif isinstance(stmt, HIRExpression):
            # R65 HIR intentionally plants bare HIRAssignment / HIRMultiAssignment
            # into statement lists for for-loop desugaring (see
            # hir/builder.py::_build_for). The type checker matches this with a
            # dedicated branch; mirror it here so the walker descends into the
            # increment expression.
            self._walk_expr(stmt)

    # ------------------------------------------------------------------ exprs

    def _walk_expr(self, expr: Optional[HIRExpression]) -> None:
        if expr is None:
            return
        if isinstance(expr, HIRIdentifier):
            for rule in self.rules:
                rule.visit_identifier(expr, self.ctx)
        elif isinstance(expr, HIRBinaryOp):
            for rule in self.rules:
                rule.visit_binary_op(expr, self.ctx)
            self._walk_expr(expr.left)
            self._walk_expr(expr.right)
        elif isinstance(expr, HIRUnaryOp):
            self._walk_expr(expr.operand)
        elif isinstance(expr, HIRAssignment):
            for rule in self.rules:
                rule.visit_assignment(expr, self.ctx)
            # Plain-identifier assignment targets are dispatched as writes
            # (visit_identifier_write), not reads. This lets L002 ignore them
            # while rules like reachability_forbidden_access can still flag
            # writes to forbidden symbols. Compound lvalues (arr[i], s.field,
            # *ptr) are walked normally — their subexpressions' identifiers
            # are legitimate reads.
            if isinstance(expr.target, HIRIdentifier):
                for rule in self.rules:
                    rule.visit_identifier_write(expr.target, self.ctx)
            else:
                self._walk_expr(expr.target)
            self._walk_expr(expr.value)
        elif isinstance(expr, HIRMultiAssignment):
            for target in expr.targets:
                if isinstance(target, HIRIdentifier):
                    for rule in self.rules:
                        rule.visit_identifier_write(target, self.ctx)
                else:
                    self._walk_expr(target)
            self._walk_expr(expr.value)
        elif isinstance(expr, HIRFunctionCall):
            self._walk_expr(expr.func)
            for arg in expr.args:
                self._walk_expr(arg)
        elif isinstance(expr, HIRMethodCall):
            self._walk_expr(expr.receiver)
            for arg in expr.args:
                self._walk_expr(arg)
        elif isinstance(expr, HIRArrayIndex):
            self._walk_expr(expr.array)
            self._walk_expr(expr.index)
        elif isinstance(expr, HIRFieldAccess):
            self._walk_expr(expr.base)
        elif isinstance(expr, HIRDereference):
            self._walk_expr(expr.pointer)
        elif isinstance(expr, HIRAddressOf):
            self._walk_expr(expr.operand)
        elif isinstance(expr, HIRTypeCast):
            self._walk_expr(expr.expr)
        elif isinstance(expr, HIRArrayLiteralExpr):
            for element in expr.elements:
                self._walk_expr(element)
        elif isinstance(expr, HIRArrayFillExpr):
            self._walk_expr(expr.fill_value)
        elif isinstance(expr, HIRStructLiteralExpr):
            for field_init in expr.fields:
                self._walk_expr(field_init.value)
        elif isinstance(expr, HIRBlockExpression):
            for stmt in expr.statements:
                self._walk_stmt(stmt)
            self._walk_expr(expr.final_expr)
        elif isinstance(expr, HIRIfExpression):
            self._walk_expr(expr.condition)
            self._walk_expr(expr.then_block)
            self._walk_expr(expr.else_block)
        elif isinstance(expr, HIRLoopExpression):
            if expr.body is not None:
                self._walk_block(expr.body)
        elif isinstance(expr, HIRMatchExpression):
            self._walk_expr(expr.scrutinee)
            for arm in expr.arms:
                if isinstance(arm.body, HIRExpression):
                    self._walk_expr(arm.body)
                elif isinstance(arm.body, HIRStatement):
                    self._walk_stmt(arm.body)
        # Leaf expressions (HIRIntegerLiteral, HIRBooleanLiteral, HIRStringLiteral,
        # HIRRegister, HIRStatusFlagAccess, HIRFunctionAddress, HIREnumVariantExpr,
        # HIRIncludeBytesExpr) have no sub-expressions to walk.


def run_lint(
    program: HIRProgram,
    config: Optional["LintConfig"] = None,
) -> DiagnosticCollector:
    """Run every enabled lint rule on ``program`` and return the diagnostics.

    If ``config`` is ``None``, a default config with all built-in rules
    enabled is used (no config file, no custom rules).
    """
    from r65.compiler.lint.config import LintConfig, default_config
    from r65.compiler.lint.rules import BUILTIN_RULES

    if config is None:
        config = default_config()

    rules = [cls() for cls in BUILTIN_RULES if cls().code in config.enabled_codes]
    rules.extend(config.custom_rules)

    linter = R65Linter(program, rules)
    return linter.check()
