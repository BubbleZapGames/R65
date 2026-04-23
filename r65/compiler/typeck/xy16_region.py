"""
XY16 region validator.

R65 codegen assumes X/Y are 16-bit (x16). When a developer writes
``STATUS.XY16 = false`` the CPU drops into 8-bit index mode, and any
subsequent compiler-generated X/Y code — indexed addressing, loop counters,
function call/return, array/struct/pointer access — silently produces
corrupt code.

This pass walks each function body and enforces a narrow safe envelope:
an x8 region must begin with ``STATUS.XY16 = false`` and end with
``STATUS.XY16 = true`` within the same straight-line block, containing
only ``asm!`` statements and scalar non-indexed operations in between.
Violations are raised as ``TypeCheckError`` (error code ``E-XY16-REGION``).

Diagnoses:
    - Function / method call inside region
    - Return statement inside region
    - Control flow (if / while / loop / break / continue) inside region
    - Array, field, pointer-dereference, or indexed access inside region
    - Region unclosed at end of function body (missing restore)
"""

from typing import Optional

from r65.compiler.errors import SourceLocation
from r65.compiler.hir import (
    HIRArrayIndex,
    HIRAsmStmt,
    HIRAssignment,
    HIRBinaryOp,
    HIRBlock,
    HIRBooleanLiteral,
    HIRBreakStmt,
    HIRContinueStmt,
    HIRDereference,
    HIRExprStmt,
    HIRExpression,
    HIRFieldAccess,
    HIRFunctionCall,
    HIRFunctionDecl,
    HIRIfStmt,
    HIRLetStmt,
    HIRMethodCall,
    HIRMultiAssignment,
    HIRReturnStmt,
    HIRStatement,
    HIRStatusFlagAccess,
    HIRMultiLetStmt,
    HIRTupleLetStmt,
    HIRTypeCast,
    HIRUnaryOp,
    HIRWhileStmt,
)
from r65.compiler.typeck.errors import TypeCheckError


def _is_xy16_set(stmt: HIRStatement, value: bool) -> bool:
    """True if stmt is exactly `STATUS.XY16 = <value>`."""
    if not isinstance(stmt, HIRExprStmt):
        return False
    expr = stmt.expr
    if not isinstance(expr, HIRAssignment):
        return False
    target = expr.target
    if not isinstance(target, HIRStatusFlagAccess):
        return False
    if target.flag_name != "XY16":
        return False
    lit = expr.value
    return isinstance(lit, HIRBooleanLiteral) and lit.value is value


class Xy16RegionChecker:
    """Straight-line x8 region validator for a single function body."""

    CODE = "E-XY16-REGION"

    def __init__(self, func: HIRFunctionDecl):
        self.func = func
        self.x8_start_loc: Optional[SourceLocation] = None

    def check(self) -> None:
        if self.func.body is None:
            return
        self._check_block(self.func.body, is_function_body=True)

    # ------------------------------------------------------------------

    def _check_block(self, block: HIRBlock, *, is_function_body: bool) -> None:
        # Each block is its own straight-line scope: region state from a parent
        # scope does NOT carry in (the parent's check already rejected opening
        # a region and entering nested control flow). Save/restore so sibling
        # scopes don't interfere.
        saved = self.x8_start_loc
        self.x8_start_loc = None
        for stmt in block.statements:
            self._check_stmt(stmt)
        if self.x8_start_loc is not None:
            raise TypeCheckError(
                f"[{self.CODE}] block ends while still in 8-bit index mode "
                f"(STATUS.XY16 = false) — must restore with "
                f"STATUS.XY16 = true before the block ends",
                source_loc=self.x8_start_loc,
                hint="add `STATUS.XY16 = true;` before the closing `}`",
            )
        self.x8_start_loc = saved

    def _check_stmt(self, stmt: HIRStatement) -> None:
        # Track region entry/exit first.
        if _is_xy16_set(stmt, False):
            if self.x8_start_loc is None:
                self.x8_start_loc = stmt.source_loc
            return
        if _is_xy16_set(stmt, True):
            self.x8_start_loc = None
            return

        in_x8 = self.x8_start_loc is not None

        if isinstance(stmt, HIRAsmStmt):
            # asm! is the only allowed non-scalar statement in an x8 region.
            return

        if isinstance(stmt, HIRBlock):
            # Nested blocks are straight-line too; just recurse with shared state.
            self._check_block(stmt, is_function_body=False)
            return

        if in_x8:
            # Control-flow constructs break the straight-line invariant.
            if isinstance(stmt, HIRReturnStmt):
                self._error(
                    stmt.source_loc,
                    "`return` inside 8-bit index region — caller assumes 16-bit X/Y",
                )
            if isinstance(stmt, (HIRIfStmt, HIRWhileStmt)):
                self._error(
                    stmt.source_loc,
                    "control flow inside 8-bit index region — loops/branches may "
                    "contain compiler-generated X/Y code",
                )
            if isinstance(stmt, (HIRBreakStmt, HIRContinueStmt)):
                self._error(
                    stmt.source_loc,
                    "`break`/`continue` inside 8-bit index region — jumps out of "
                    "the region without restoring 16-bit mode",
                )

        # Inspect sub-expressions (even outside x8: return/if-condition may
        # themselves contain assignments to XY16 that we still want to surface
        # as region errors when they occur mid-region). We only walk when
        # in_x8 to avoid spurious false positives on normal code.
        if isinstance(stmt, HIRLetStmt):
            if stmt.initializer is not None and in_x8:
                self._check_expr(stmt.initializer)
        elif isinstance(stmt, HIRMultiLetStmt):
            if stmt.initializer is not None and in_x8:
                self._check_expr(stmt.initializer)
        elif isinstance(stmt, HIRExprStmt):
            if stmt.expr is not None and in_x8:
                self._check_expr(stmt.expr)
        elif isinstance(stmt, HIRReturnStmt):
            for v in stmt.values:
                if in_x8:
                    self._check_expr(v)
        elif isinstance(stmt, HIRIfStmt):
            if stmt.condition is not None and in_x8:
                self._check_expr(stmt.condition)
            # Walk branches as independent straight-line scopes so nested
            # `STATUS.XY16 = false` writes are caught.
            if stmt.then_block is not None:
                self._check_block(stmt.then_block, is_function_body=False)
            if stmt.else_block is not None:
                if isinstance(stmt.else_block, HIRIfStmt):
                    self._check_stmt(stmt.else_block)
                else:
                    self._check_block(stmt.else_block, is_function_body=False)
        elif isinstance(stmt, HIRWhileStmt):
            if stmt.condition is not None and in_x8:
                self._check_expr(stmt.condition)
            if stmt.body is not None:
                self._check_block(stmt.body, is_function_body=False)
        elif isinstance(stmt, HIRExpression) and in_x8:
            # for-loop increment expressions land here per HIR builder.
            self._check_expr(stmt)

    # ------------------------------------------------------------------

    def _check_expr(self, expr: Optional[HIRExpression]) -> None:
        """Reject any X/Y-dependent compiler-emitted operation."""
        if expr is None:
            return

        # A call anywhere in an expression breaks the region.
        if isinstance(expr, HIRFunctionCall):
            self._error(
                expr.source_loc,
                "function call inside 8-bit index region — callee prologue "
                "assumes 16-bit X/Y",
            )
            return
        if isinstance(expr, HIRMethodCall):
            self._error(
                expr.source_loc,
                "method call inside 8-bit index region — callee prologue "
                "assumes 16-bit X/Y",
            )
            return

        # Compiler lowers indexing to LDX/LDY-indexed addressing (u16 index).
        if isinstance(expr, HIRArrayIndex):
            self._error(
                expr.source_loc,
                "array indexing inside 8-bit index region — uses 16-bit X/Y",
            )
            return
        if isinstance(expr, HIRFieldAccess):
            self._error(
                expr.source_loc,
                "struct field access inside 8-bit index region — may lower to "
                "indexed load using 16-bit X/Y",
            )
            return
        if isinstance(expr, HIRDereference):
            self._error(
                expr.source_loc,
                "pointer dereference inside 8-bit index region — lowers to "
                "indexed addressing with 16-bit X/Y",
            )
            return

        # Descend into sub-expressions.
        if isinstance(expr, HIRBinaryOp):
            self._check_expr(expr.left)
            self._check_expr(expr.right)
        elif isinstance(expr, HIRUnaryOp):
            self._check_expr(expr.operand)
        elif isinstance(expr, HIRTypeCast):
            self._check_expr(expr.expr)
        elif isinstance(expr, HIRAssignment):
            self._check_expr(expr.target)
            self._check_expr(expr.value)
        elif isinstance(expr, HIRMultiAssignment):
            for t in expr.targets:
                self._check_expr(t)
            self._check_expr(expr.value)
        # Leaf expressions (literals, identifiers, HIRStatusFlagAccess, HIRRegister)
        # are scalar — no X/Y risk.

    # ------------------------------------------------------------------

    def _error(self, source_loc: Optional[SourceLocation], message: str) -> None:
        raise TypeCheckError(
            f"[{self.CODE}] {message}",
            source_loc=source_loc or self.x8_start_loc,
            hint="move the operation out of the region, or restore 16-bit mode "
            "with `STATUS.XY16 = true` first",
        )


def check_xy16_regions(func: HIRFunctionDecl) -> None:
    """Entry point — validate x8 regions in a single function."""
    Xy16RegionChecker(func).check()
