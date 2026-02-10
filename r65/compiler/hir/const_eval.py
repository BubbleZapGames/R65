"""
Compile-time constant expression evaluation for R65 HIR.

Evaluates constant expressions at compile time (array sizes, enum values, etc.).
"""

from typing import Any, Union
from r65.compiler.frontend import ast
from r65.compiler.hir.errors import *
from r65.compiler.errors import compiler_assert
from r65.compiler.hir.unified_type_utils import get_unified_type_size
from r65.compiler.hir.types import ArrayTypeInfo
from r65.compiler.builtins.registry import BuiltinRegistry
from r65.compiler.frontend.ast import CfgIdentifier


class ConstEvaluator:
    """Evaluates constant expressions at compile time."""

    def __init__(self, symbol_table: Any, cfg_evaluator: Any = None):
        self.symbol_table = symbol_table
        self.cfg_evaluator = cfg_evaluator

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

    def eval(self, expr: ast.Expression) -> Union[int, bool, str]:
        """
        Evaluate a constant expression.

        Returns: int, bool, or str value
        Raises: HIRError if expression is not constant or evaluation fails
        """
        if isinstance(expr, ast.IntegerLiteral):
            return expr.value

        elif isinstance(expr, ast.BooleanLiteral):
            return expr.value

        elif isinstance(expr, ast.StringLiteral):
            return expr.value

        elif isinstance(expr, ast.Identifier):
            # Check for built-in type names (used in size_of contexts)
            if expr.name in ['u8', 'u16', 'i8', 'i16', 'bool']:
                raise HIRError(f"Type name '{expr.name}' cannot be used as a value in const expression")
            
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

        elif isinstance(expr, ast.FunctionCall):
            return self._eval_function_call(expr)

        else:
            raise HIRError(f"Non-constant expression: {type(expr).__name__}")

    def _eval_binary_op(self, expr: ast.BinaryOp) -> Union[int, bool, str]:
        """Evaluate binary operation."""
        left = self.eval(expr.left)
        right = self.eval(expr.right)

        op = expr.op

        # String concatenation
        if op == '+':
            # If either operand is a string, perform string concatenation
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            # Otherwise, perform arithmetic addition
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
            # String comparisons use lexicographic ordering
            if isinstance(left, str) or isinstance(right, str):
                return str(left) < str(right)
            return self._ensure_int(left) < self._ensure_int(right)
        elif op == '<=':
            if isinstance(left, str) or isinstance(right, str):
                return str(left) <= str(right)
            return self._ensure_int(left) <= self._ensure_int(right)
        elif op == '>':
            if isinstance(left, str) or isinstance(right, str):
                return str(left) > str(right)
            return self._ensure_int(left) > self._ensure_int(right)
        elif op == '>=':
            if isinstance(left, str) or isinstance(right, str):
                return str(left) >= str(right)
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

    def _eval_function_call(self, expr: ast.FunctionCall) -> Union[int, bool, str]:
        """Evaluate function call in const expression."""
        # Check if this is a method call (e.g., array.len())
        if isinstance(expr.func, ast.FieldAccess):
            return self._eval_method_call(expr)

        func_name = expr.func.name if isinstance(expr.func, ast.Identifier) else None

        if not func_name:
            raise HIRError("Only direct function calls allowed in const expressions")
        
        # Check if this is a built-in function
        if not BuiltinRegistry.is_builtin(func_name):
            raise HIRError(f"Function '{func_name}' is not a built-in const function")
        
        builtin = BuiltinRegistry.get_builtin(func_name)
        
        # Only allow type info built-ins in const expressions
        if builtin.kind.value != "type_info":
            raise HIRError(f"Built-in '{func_name}' is not allowed in const expressions")
        
        # Handle size_of specifically
        if func_name == "size_of":
            return self._eval_size_of(expr)

        # Handle offset_of specifically
        if func_name == "offset_of":
            return self._eval_offset_of(expr)

        # Handle cfg! specifically - requires cfg evaluator to be passed
        if func_name == "cfg":
            if not hasattr(self, 'cfg_evaluator') or self.cfg_evaluator is None:
                raise HIRError("cfg! function requires cfg configuration to be provided")
            return self._eval_cfg(expr)
        
        raise HIRError(f"Unsupported const built-in function: {func_name}")

    def _eval_size_of(self, expr: ast.FunctionCall) -> int:
        """Evaluate size_of builtin function using unified type utilities."""
        if len(expr.args) != 1:
            raise HIRError("size_of expects exactly 1 argument")

        arg = expr.args[0]

        try:
            return get_unified_type_size(arg, self.symbol_table)
        except Exception as e:
            # If type not yet available (e.g., struct declared later), defer evaluation
            raise HIRError(f"Cannot evaluate size_of at this time: {e}")

    def _eval_offset_of(self, expr: ast.FunctionCall) -> int:
        """Evaluate offset_of builtin function: offset_of(StructName, field_name)."""
        if len(expr.args) != 2:
            raise HIRError("offset_of expects exactly 2 arguments: offset_of(StructType, field)")

        # First arg: struct type name (must be an identifier)
        struct_arg = expr.args[0]
        if not isinstance(struct_arg, ast.Identifier):
            raise HIRError("offset_of first argument must be a struct type name")

        struct_name = struct_arg.name
        symbol = self.symbol_table.lookup(struct_name)
        if symbol is None:
            raise HIRError(f"Undefined struct in offset_of: {struct_name}")

        struct_def = symbol.definition
        if struct_def is None or not hasattr(struct_def, 'fields'):
            raise HIRError(f"'{struct_name}' is not a struct type")

        # Second arg: field name (must be an identifier)
        field_arg = expr.args[1]
        if not isinstance(field_arg, ast.Identifier):
            raise HIRError("offset_of second argument must be a field name")

        field_name = field_arg.name

        # Check if this is an HIR struct (has pre-computed offsets)
        from r65.compiler.hir.nodes import HIRStructDecl
        if isinstance(struct_def, HIRStructDecl):
            for field in struct_def.fields:
                if field.name == field_name:
                    return field.offset
            raise HIRError(f"Struct '{struct_name}' has no field '{field_name}'")

        # AST struct: compute offset by summing field sizes
        offset = 0
        for field in struct_def.fields:
            if field.name == field_name:
                return offset
            offset += get_unified_type_size(field.field_type, self.symbol_table)

        raise HIRError(f"Struct '{struct_name}' has no field '{field_name}'")

    def _eval_method_call(self, expr: ast.FunctionCall) -> int:
        """Evaluate method call in const expression (e.g., array.len())."""
        compiler_assert(
            isinstance(expr.func, ast.FieldAccess),
            f"_eval_method_call called with non-FieldAccess func: {type(expr.func).__name__}"
        )

        method_name = expr.func.field
        receiver = expr.func.base

        # Only len() is supported as a const method
        if method_name != 'len':
            raise HIRError(f"Method '{method_name}' is not allowed in const expressions")

        # Validate no arguments
        if len(expr.args) != 0:
            raise HIRError(f"len() takes no arguments, got {len(expr.args)}")

        # Get the receiver's type - must be an identifier for const evaluation
        if not isinstance(receiver, ast.Identifier):
            raise HIRError("len() receiver must be an identifier in const expressions")

        symbol = self.symbol_table.lookup(receiver.name)
        if symbol is None:
            raise HIRError(f"Undefined identifier: {receiver.name}")

        # Ensure it's an array type
        if symbol.var_type is None or not isinstance(symbol.var_type, ArrayTypeInfo):
            raise HIRError(f"len() requires array type, '{receiver.name}' is not an array")

        # Return the array size
        return symbol.var_type.size

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
