"""
Compile-time constant expression evaluation for R65 HIR.

Evaluates constant expressions at compile time (array sizes, enum values, etc.).
"""

from typing import Any, Union
from r65.compiler.frontend import ast
from r65.compiler.hir.errors import *


class ConstEvaluator:
    """Evaluates constant expressions at compile time."""

    def __init__(self, symbol_table: Any):
        self.symbol_table = symbol_table

    def is_constant(self, expr: ast.Expression) -> bool:
        """
        Check if an expression is a constant expression.
        
        Returns: True if expression can be evaluated at compile time
        """
        try:
            self.eval(expr)
            return True
        except HIRError:
            return False

    def eval(self, expr: ast.Expression) -> Union[int, bool]:
        """
        Evaluate a constant expression.

        Returns: int or bool value
        Raises: HIRError if expression is not constant or evaluation fails
        """
        if isinstance(expr, ast.IntegerLiteral):
            return expr.value

        elif isinstance(expr, ast.BooleanLiteral):
            return expr.value

        elif isinstance(expr, ast.Identifier):
            # Look up const variable
            symbol = self.symbol_table.lookup(expr.name)
            if symbol is None:
                raise HIRError(f"Undefined identifier in const expression: {expr.name}")

            if symbol.kind.value != "const":
                raise HIRError(f"Identifier '{expr.name}' is not a const in const expression")

            if symbol.const_value is None:
                raise HIRError(f"Const '{expr.name}' has no evaluated value")

            return symbol.const_value

        elif isinstance(expr, ast.BinaryOp):
            return self._eval_binary_op(expr)

        elif isinstance(expr, ast.UnaryOp):
            return self._eval_unary_op(expr)

        elif isinstance(expr, ast.TypeCast):
            return self._eval_cast(expr)

        else:
            raise HIRError(f"Non-constant expression: {type(expr).__name__}")

    def _eval_binary_op(self, expr: ast.BinaryOp) -> Union[int, bool]:
        """Evaluate binary operation."""
        left = self.eval(expr.left)
        right = self.eval(expr.right)

        op = expr.op

        # Arithmetic operators
        if op == '+':
            return self._ensure_int(left) + self._ensure_int(right)
        elif op == '-':
            return self._ensure_int(left) - self._ensure_int(right)
        elif op == '*':
            return self._ensure_int(left) * self._ensure_int(right)
        elif op == '/':
            right_val = self._ensure_int(right)
            if right_val == 0:
                raise HIRError("Division by zero in const expression")
            return self._ensure_int(left) // right_val
        elif op == '%':
            right_val = self._ensure_int(right)
            if right_val == 0:
                raise HIRError("Modulo by zero in const expression")
            return self._ensure_int(left) % right_val

        # Bitwise operators
        elif op == '&':
            return self._ensure_int(left) & self._ensure_int(right)
        elif op == '|':
            return self._ensure_int(left) | self._ensure_int(right)
        elif op == '^':
            return self._ensure_int(left) ^ self._ensure_int(right)
        elif op == '<<':
            return self._ensure_int(left) << self._ensure_int(right)
        elif op == '>>':
            return self._ensure_int(left) >> self._ensure_int(right)

        # Comparison operators
        elif op == '==':
            return left == right
        elif op == '!=':
            return left != right
        elif op == '<':
            return self._ensure_int(left) < self._ensure_int(right)
        elif op == '<=':
            return self._ensure_int(left) <= self._ensure_int(right)
        elif op == '>':
            return self._ensure_int(left) > self._ensure_int(right)
        elif op == '>=':
            return self._ensure_int(left) >= self._ensure_int(right)

        # Logical operators
        elif op == '&&':
            return self._ensure_bool(left) and self._ensure_bool(right)
        elif op == '||':
            return self._ensure_bool(left) or self._ensure_bool(right)

        else:
            raise HIRError(f"Unsupported operator in const expression: {op}")

    def _eval_unary_op(self, expr: ast.UnaryOp) -> Union[int, bool]:
        """Evaluate unary operation."""
        operand = self.eval(expr.operand)
        op = expr.op

        if op == '-':
            return -self._ensure_int(operand)
        elif op == '!':
            return not self._ensure_bool(operand)
        elif op == '~':
            return ~self._ensure_int(operand)
        else:
            raise HIRError(f"Unsupported unary operator in const expression: {op}")

    def _eval_cast(self, expr: ast.TypeCast) -> Union[int, bool]:
        """Evaluate type cast."""
        value = self.eval(expr.expr)
        target_type = expr.target_type

        if isinstance(target_type, ast.BasicType):
            type_name = target_type.name

            # Cast to integer types
            if type_name in ['u8', 'i8', 'u16', 'i16']:
                int_val = self._ensure_int(value)

                # Apply type bounds
                if type_name == 'u8':
                    return int_val & 0xFF
                elif type_name == 'i8':
                    val = int_val & 0xFF
                    return val if val < 128 else val - 256
                elif type_name == 'u16':
                    return int_val & 0xFFFF
                elif type_name == 'i16':
                    val = int_val & 0xFFFF
                    return val if val < 32768 else val - 65536

            # Cast to bool
            elif type_name == 'bool':
                if isinstance(value, bool):
                    return value
                elif isinstance(value, int):
                    return value != 0
                else:
                    raise HIRError(f"Cannot cast {type(value).__name__} to bool")

            else:
                raise HIRError(f"Cannot cast to {type_name} in const expression")
        else:
            raise HIRError(f"Unsupported cast target in const expression: {type(target_type).__name__}")
        
        # This should never be reached due to the raises above
        raise HIRError("Unexpected path in _eval_cast")

    def _ensure_int(self, value: Any) -> int:
        """Ensure value is an integer."""
        if isinstance(value, bool):
            return 1 if value else 0
        elif isinstance(value, int):
            return value
        else:
            raise HIRError(f"Expected integer in const expression, got {type(value).__name__}")

    def _ensure_bool(self, value: Any) -> bool:
        """Ensure value is a boolean."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            return value != 0
        else:
            raise HIRError(f"Expected boolean in const expression, got {type(value).__name__}")
