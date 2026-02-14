"""
Compile-time constant expression evaluation for HIR nodes.

Evaluates HIR expressions at compile time for constant folding and dead code elimination.
This module provides functions for both MIR builder (dead code elimination) and
expression lowerer (constant folding).
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.hir.symbol_table import SymbolTable


def try_eval_const_int(expr, symbol_table: 'SymbolTable' = None) -> Optional[int]:
    """
    Try to evaluate an HIR expression to a constant integer at compile time.

    Args:
        expr: HIR expression to evaluate
        symbol_table: Symbol table for looking up const values

    Returns:
        Integer value if expression is compile-time constant, None otherwise
    """
    from r65.compiler.hir import (
        HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr,
        HIRTypeCast, HIRBinaryOp, HIRUnaryOp, HIRIdentifier
    )
    from r65.compiler.hir.symbol_table import SymbolKind

    if isinstance(expr, HIRIntegerLiteral):
        return expr.value

    elif isinstance(expr, HIRBooleanLiteral):
        return 1 if expr.value else 0

    elif isinstance(expr, HIREnumVariantExpr):
        return expr.value

    elif isinstance(expr, HIRIdentifier):
        # Try symbol from expression first, then fall back to symbol_table lookup
        symbol = getattr(expr, 'symbol', None)
        if symbol is None and symbol_table is not None:
            symbol = symbol_table.lookup(expr.name)
        if symbol and symbol.kind == SymbolKind.CONST and symbol.const_value is not None:
            return symbol.const_value
        return None

    elif isinstance(expr, HIRTypeCast):
        return try_eval_const_int(expr.expr, symbol_table)

    elif isinstance(expr, HIRUnaryOp):
        operand = try_eval_const_int(expr.operand, symbol_table)
        if operand is None:
            return None
        if expr.op == '-':
            return -operand
        elif expr.op == '~':
            return ~operand
        elif expr.op == '!':
            return 0 if operand else 1
        return None

    elif isinstance(expr, HIRBinaryOp):
        left = try_eval_const_int(expr.left, symbol_table)
        right = try_eval_const_int(expr.right, symbol_table)
        if left is None or right is None:
            return None
        return _eval_binary_int(expr.op, left, right)

    # Block expression: evaluate statements (must be no-ops for const), then final expr
    from r65.compiler.hir.nodes import HIRBlockExpression, HIRIfExpression
    if isinstance(expr, HIRBlockExpression):
        # For const eval, block statements must be empty (we can't evaluate arbitrary stmts)
        if expr.statements:
            return None
        return try_eval_const_int(expr.final_expr, symbol_table)

    if isinstance(expr, HIRIfExpression):
        cond = try_eval_const_bool(expr.condition, symbol_table)
        if cond is True:
            return try_eval_const_int(expr.then_block, symbol_table)
        elif cond is False:
            return try_eval_const_int(expr.else_block, symbol_table)
        return None

    return None


def try_eval_const_bool(expr, symbol_table: 'SymbolTable' = None) -> Optional[bool]:
    """
    Try to evaluate an HIR condition expression to a constant boolean at compile time.

    Used for dead code elimination in if/while statements when conditions are
    compile-time constants.

    Args:
        expr: HIR condition expression to evaluate
        symbol_table: Symbol table for looking up const values

    Returns:
        True/False if condition is compile-time constant, None otherwise
    """
    from r65.compiler.hir import (
        HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr,
        HIRTypeCast, HIRBinaryOp, HIRUnaryOp, HIRIdentifier
    )
    from r65.compiler.hir.symbol_table import SymbolKind

    if isinstance(expr, HIRBooleanLiteral):
        return expr.value

    elif isinstance(expr, HIRIntegerLiteral):
        # Non-zero is truthy
        return expr.value != 0

    elif isinstance(expr, HIREnumVariantExpr):
        return expr.value != 0

    elif isinstance(expr, HIRIdentifier):
        # Try symbol from expression first, then fall back to symbol_table lookup
        symbol = getattr(expr, 'symbol', None)
        if symbol is None and symbol_table is not None:
            symbol = symbol_table.lookup(expr.name)
        if symbol and symbol.kind == SymbolKind.CONST and symbol.const_value is not None:
            return symbol.const_value != 0
        return None

    elif isinstance(expr, HIRTypeCast):
        return try_eval_const_bool(expr.expr, symbol_table)

    elif isinstance(expr, HIRUnaryOp):
        if expr.op == '!':
            inner = try_eval_const_bool(expr.operand, symbol_table)
            if inner is not None:
                return not inner
        elif expr.op == '~':
            inner = try_eval_const_int(expr.operand, symbol_table)
            if inner is not None:
                return (~inner) != 0
        return None

    elif isinstance(expr, HIRBinaryOp):
        return _eval_binary_bool(expr, symbol_table)

    from r65.compiler.hir.nodes import HIRBlockExpression, HIRIfExpression
    if isinstance(expr, HIRBlockExpression):
        if expr.statements:
            return None
        return try_eval_const_bool(expr.final_expr, symbol_table)

    if isinstance(expr, HIRIfExpression):
        cond = try_eval_const_bool(expr.condition, symbol_table)
        if cond is True:
            return try_eval_const_bool(expr.then_block, symbol_table)
        elif cond is False:
            return try_eval_const_bool(expr.else_block, symbol_table)
        return None

    return None


def try_eval_const_binary_masked(expr, symbol_table: 'SymbolTable' = None) -> Optional[int]:
    """
    Try to evaluate a binary operation at compile time with type masking.

    Used by expression lowerer for constant folding where results should be
    masked to the expression's type size.

    Args:
        expr: HIR binary operation to evaluate
        symbol_table: Symbol table for looking up const values

    Returns:
        Integer result masked to type size if constant, None otherwise
    """
    from r65.compiler.hir import HIRBinaryOp

    if not isinstance(expr, HIRBinaryOp):
        return None

    left = try_eval_const_int(expr.left, symbol_table)
    right = try_eval_const_int(expr.right, symbol_table)

    if left is None or right is None:
        return None

    mask = _get_type_mask(expr.expr_type)
    result = _eval_binary_int(expr.op, left, right)
    if result is None:
        return None

    # Apply mask for arithmetic operations that can overflow
    if expr.op in ('+', '-', '*', '<<'):
        return result & mask
    return result


def _eval_binary_int(op: str, left: int, right: int) -> Optional[int]:
    """Evaluate a binary operation on two integer operands."""
    if op == '+':
        return left + right
    elif op == '-':
        return left - right
    elif op == '*':
        return left * right
    elif op == '/':
        return left // right if right != 0 else None
    elif op == '%':
        return left % right if right != 0 else None
    elif op == '&':
        return left & right
    elif op == '|':
        return left | right
    elif op == '^':
        return left ^ right
    elif op == '<<':
        return left << right
    elif op == '>>':
        return left >> right
    # Comparison operators return 0 or 1
    elif op == '==':
        return 1 if left == right else 0
    elif op == '!=':
        return 1 if left != right else 0
    elif op == '<':
        return 1 if left < right else 0
    elif op == '>':
        return 1 if left > right else 0
    elif op == '<=':
        return 1 if left <= right else 0
    elif op == '>=':
        return 1 if left >= right else 0
    return None


def _eval_binary_bool(expr, symbol_table: 'SymbolTable') -> Optional[bool]:
    """Evaluate a binary operation to a constant boolean."""
    op = expr.op

    # Logical operators with short-circuit potential
    if op == '&&':
        left = try_eval_const_bool(expr.left, symbol_table)
        if left is False:
            return False
        right = try_eval_const_bool(expr.right, symbol_table)
        if left is True and right is not None:
            return right
        return None

    elif op == '||':
        left = try_eval_const_bool(expr.left, symbol_table)
        if left is True:
            return True
        right = try_eval_const_bool(expr.right, symbol_table)
        if left is False and right is not None:
            return right
        return None

    # Comparison operators
    elif op in ('==', '!=', '<', '>', '<=', '>='):
        left = try_eval_const_int(expr.left, symbol_table)
        right = try_eval_const_int(expr.right, symbol_table)
        if left is None or right is None:
            return None

        if op == '==':
            return left == right
        elif op == '!=':
            return left != right
        elif op == '<':
            return left < right
        elif op == '>':
            return left > right
        elif op == '<=':
            return left <= right
        elif op == '>=':
            return left >= right

    # Arithmetic/bitwise operators - evaluate as int, convert to bool
    else:
        result = try_eval_const_int(expr, symbol_table)
        if result is not None:
            return result != 0

    return None


def _get_type_mask(type_info) -> int:
    """Get bitmask for type size."""
    from r65.compiler.hir.types import BasicTypeInfo

    if isinstance(type_info, BasicTypeInfo):
        if type_info.name in ('u8', 'i8', 'bool'):
            return 0xFF
        elif type_info.name in ('u16', 'i16'):
            return 0xFFFF
    return 0xFFFF  # Default to 16-bit
