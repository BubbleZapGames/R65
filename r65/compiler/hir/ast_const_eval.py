# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Compile-time constant expression evaluation for R65 HIR.

Evaluates constant expressions at compile time (array sizes, enum values, etc.).
"""

from typing import Any, Optional, Union
from r65.compiler.frontend import ast
from r65.compiler.hir.errors import *
from r65.compiler.errors import compiler_assert
from r65.compiler.hir.unified_type_utils import get_unified_type_size, layout_fields
from r65.compiler.hir.types import ArrayTypeInfo, BUILTIN_TYPE_NAMES, INTEGER_TYPE_NAMES
from r65.compiler.builtins.registry import BuiltinRegistry, BuiltinKind
from r65.compiler.frontend.ast import CfgIdentifier
from r65.compiler.hir.hir_const_eval import trunc_div, trunc_mod, wrap_int
import copy
import math as _math
import sys


# =============================================================================
# Fixed-point const math — single source of truth.
#
# Used both by const-expression evaluation (_eval_const_math, which wraps the
# raised ZeroDivisionError/ValueError in an HIRError) and by const-fn execution
# (_build_const_fn_namespace, which exposes these directly to transpiled code).
# =============================================================================

def _clamp_i16(v):
    v = int(round(v))
    return max(-32768, min(32767, v))


def _clamp_u16(v):
    v = int(round(v))
    return max(0, min(65535, v))


def _fixed_sin(index, table_size, amplitude):
    if table_size == 0:
        raise ZeroDivisionError("fixed_sin: table_size must not be zero")
    return _clamp_i16(_math.sin(2 * _math.pi * index / table_size) * amplitude)


def _fixed_cos(index, table_size, amplitude):
    if table_size == 0:
        raise ZeroDivisionError("fixed_cos: table_size must not be zero")
    return _clamp_i16(_math.cos(2 * _math.pi * index / table_size) * amplitude)


def _fixed_atan2(y, x, table_size):
    if table_size == 0:
        raise ZeroDivisionError("fixed_atan2: table_size must not be zero")
    if y == 0 and x == 0:
        return 0
    angle = _math.atan2(y, x)
    if angle < 0:
        angle += 2 * _math.pi
    return _clamp_u16(angle / (2 * _math.pi) * table_size)


def _fixed_sqrt(value, scale):
    if value < 0:
        raise ValueError("fixed_sqrt: value must not be negative")
    return _clamp_u16(_math.sqrt(value) * scale)


def _fixed_log2(value, scale):
    if value <= 0:
        raise ValueError("fixed_log2: value must be positive")
    return _clamp_i16(_math.log2(value) * scale)


def _fixed_exp2(value, in_scale, out_scale):
    if in_scale == 0:
        raise ZeroDivisionError("fixed_exp2: in_scale must not be zero")
    return _clamp_u16(_math.pow(2, value / in_scale) * out_scale)


def _fixed_lerp(a, b, t, t_max):
    if t_max == 0:
        raise ZeroDivisionError("fixed_lerp: t_max must not be zero")
    return _clamp_i16(a + (b - a) * t / t_max)


def _fixed_clamp(value, min_val, max_val):
    if min_val > max_val:
        raise ValueError("fixed_clamp: min must not be greater than max")
    return _clamp_i16(max(min_val, min(max_val, value)))


def _fixed_color_bgr(red, green, blue):
    for name, val in [('red', red), ('green', green), ('blue', blue)]:
        if val < 0 or val > 255:
            raise ValueError(f"fixed_color_bgr: {name} must be 0-255, got {val}")
    r5 = (red >> 3) & 0x1F
    g5 = (green >> 3) & 0x1F
    b5 = (blue >> 3) & 0x1F
    return (b5 << 10) | (g5 << 5) | r5


_FIXED_MATH_FNS = {
    'fixed_sin': _fixed_sin, 'fixed_cos': _fixed_cos,
    'fixed_atan2': _fixed_atan2, 'fixed_sqrt': _fixed_sqrt,
    'fixed_log2': _fixed_log2, 'fixed_exp2': _fixed_exp2,
    'fixed_lerp': _fixed_lerp, 'fixed_clamp': _fixed_clamp,
    'fixed_color_bgr': _fixed_color_bgr,
}


class ConstEvaluator:
    """Evaluates constant expressions at compile time."""

    def __init__(self, symbol_table: Any, cfg_evaluator: Any = None):
        self.symbol_table = symbol_table
        self.cfg_evaluator = cfg_evaluator
        self._compiled_const_fns = {}  # Cache: func_name -> compiled Python callable
        self._compiling_const_fns = set()  # Names currently being compiled (for recursion)
        self.type_resolver = None      # Set by HIRBuilder once it has one
        # Set by HIRBuilder: const fns by name, and a callback declaring a
        # const that pass 1 has not reached yet (both for forward references).
        self.forward_const_fns = {}
        self.declare_forward_const = None

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

    def eval(self, expr: ast.Expression) -> Union[int, bool, str, list]:
        """
        Evaluate a constant expression.

        Returns: int, bool, str, list (array literal/fill) or dict (struct literal)
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
            if expr.name in BUILTIN_TYPE_NAMES:
                raise HIRError(f"Type name '{expr.name}' cannot be used as a value in const expression", source_loc=expr.source_loc)
            
            # Look up const variable
            symbol = self._lookup_const(expr.name)
            if symbol is None:
                raise HIRError(f"Undefined identifier in const expression: {expr.name}", source_loc=expr.source_loc)

            if symbol.kind.value != "const":
                raise HIRError(f"Identifier '{expr.name}' is not a const in const expression", source_loc=expr.source_loc)

            if symbol.const_value is None:
                raise HIRError(f"Const '{expr.name}' has no evaluated value", source_loc=expr.source_loc)

            return symbol.const_value

        elif isinstance(expr, ast.EnumVariantExpr):
            # Resolve enum variant or associated constant to its integer value
            qualified = f"{expr.enum_name}::{expr.variant_name}"
            symbol = self.symbol_table.lookup(qualified)
            if symbol is None:
                raise HIRError(f"Undefined enum variant in const expression: {qualified}", source_loc=expr.source_loc)
            if symbol.const_value is None:
                raise HIRError(f"'{qualified}' has no evaluated value", source_loc=expr.source_loc)
            return symbol.const_value

        elif isinstance(expr, ast.BinaryOp):
            return self._eval_binary_op(expr)

        elif isinstance(expr, ast.UnaryOp):
            return self._eval_unary_op(expr)

        elif isinstance(expr, ast.TypeCast):
            return self._eval_cast(expr)

        elif isinstance(expr, ast.FunctionCall):
            return self._eval_function_call(expr)

        elif isinstance(expr, ast.CfgFunctionCall):
            return self.eval_cfg(expr.condition, expr.source_loc)

        elif isinstance(expr, ast.MatchExpression):
            return self._eval_match_expr(expr)

        elif isinstance(expr, ast.BlockExpression):
            # Const eval: block expression with no statements, just final expr
            if expr.statements:
                raise HIRError(f"Block expression with statements is not const-evaluable", source_loc=expr.source_loc)
            return self.eval(expr.final_expr)

        elif isinstance(expr, ast.IfExpression):
            cond = self.eval(expr.condition)
            if not isinstance(cond, bool):
                cond = bool(cond)
            if cond:
                return self.eval(expr.then_block)
            else:
                return self.eval(expr.else_block)

        elif isinstance(expr, ast.ArrayLiteralExpr):
            return [self.eval(e) for e in expr.elements]

        elif isinstance(expr, ast.ArrayFillExpr):
            value = self.eval(expr.value)
            count = self.eval(expr.count)
            return [value] * count

        elif isinstance(expr, ast.StructLiteralExpr):
            return {f.name: self.eval(f.value) for f in expr.fields}

        elif isinstance(expr, ast.FieldAccess):
            base_val = self.eval(expr.base)
            if not isinstance(base_val, dict):
                raise HIRError(
                    f"Field access on non-struct value in const expression",
                    source_loc=getattr(expr, 'source_loc', None)
                )
            if expr.field not in base_val:
                raise HIRError(
                    f"No field '{expr.field}' in const struct value",
                    source_loc=getattr(expr, 'source_loc', None)
                )
            return base_val[expr.field]

        elif isinstance(expr, ast.NewtypeFieldAccess):
            # `T.0` is a retype, not a field load — see `_transpile_expr`.
            self._check_newtype_field_index(expr)
            return self.eval(expr.base)

        elif isinstance(expr, ast.ArrayIndex):
            array_val = self.eval(expr.array)
            index_val = self.eval(expr.index)
            if not isinstance(array_val, list):
                raise HIRError(
                    f"Index on non-array value in const expression",
                    source_loc=getattr(expr, 'source_loc', None)
                )
            if not isinstance(index_val, int):
                raise HIRError(
                    f"Non-constant index in const expression",
                    source_loc=getattr(expr, 'source_loc', None)
                )
            if index_val < 0 or index_val >= len(array_val):
                raise HIRError(
                    f"Index {index_val} out of bounds for const array of length {len(array_val)}",
                    source_loc=getattr(expr, 'source_loc', None)
                )
            return array_val[index_val]

        else:
            raise HIRError(f"Non-constant expression: {type(expr).__name__}", source_loc=getattr(expr, 'source_loc', None))

    def _lookup_const(self, name: str):
        """The symbol for *name*, declaring a const written further down
        the file first if that is what it names."""
        symbol = self.symbol_table.lookup(name)
        if symbol is None and self.declare_forward_const is not None:
            if self.declare_forward_const(name):
                symbol = self.symbol_table.lookup(name)
        return symbol

    def _check_newtype_field_index(self, expr: ast.NewtypeFieldAccess) -> None:
        """Reject `t.1` before it silently evaluates to the payload.

        The type checker gives this diagnostic on the HIR, but const evaluation
        runs first, so the check has to exist here too.
        """
        if expr.index != 0:
            raise HIRError(
                f"a newtype has only field '.0', not '.{expr.index}'",
                source_loc=getattr(expr, 'source_loc', None),
                hint="a newtype wraps exactly one value, so '.0' is the only "
                     "field there is")

    def _newtype_construct_arg(self, expr: ast.FunctionCall, func_name: str,
                               symbol) -> Optional[ast.Expression]:
        """The payload of a `Newtype(x)` call, or None if this is not one.

        `Newtype(x)` parses as a call but is a retype: the value is its own
        payload. Without this both const paths reach their const-fn check and
        reject it as "Function 'Color' is not a const fn" — naming a function
        the author never wrote, for the one spelling the docs recommend.
        Range-checking still happens in the type checker, on the HIR.

        Takes the already-resolved `symbol` so callers that go on to test for a
        const fn do not look the name up twice.
        """
        from r65.compiler.hir.symbol_table import SymbolKind

        if symbol is None or symbol.kind != SymbolKind.NEWTYPE:
            return None

        if len(expr.args) != 1:
            raise HIRError(
                f"newtype '{func_name}' takes exactly 1 value, "
                f"got {len(expr.args)}",
                source_loc=expr.source_loc,
                hint=f"write '{func_name}(value)'")
        return expr.args[0]

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
                raise HIRError("Division by zero in const expression", source_loc=expr.source_loc)
            return trunc_div(self._ensure_int(left), right_val)
        elif op == '%':
            right_val = self._ensure_int(right)
            if right_val == 0:
                raise HIRError("Modulo by zero in const expression", source_loc=expr.source_loc)
            return trunc_mod(self._ensure_int(left), right_val)

        # Bitwise operators
        elif op == '&':
            return self._ensure_int(left) & self._ensure_int(right)
        elif op == '|':
            return self._ensure_int(left) | self._ensure_int(right)
        elif op == '^':
            return self._ensure_int(left) ^ self._ensure_int(right)
        elif op in ('<<', '>>'):
            amount = self._ensure_int(right)
            if amount < 0:
                raise HIRError(f"negative shift amount {amount} in const expression",
                               source_loc=expr.source_loc)
            amount = min(amount, 32)   # every R65 integer is gone by then
            if op == '<<':
                return self._ensure_int(left) << amount
            return self._ensure_int(left) >> amount

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
            raise HIRError(f"Unsupported operator in const expression: {op}", source_loc=expr.source_loc)

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
            raise HIRError(f"Unsupported unary operator in const expression: {op}", source_loc=expr.source_loc)

    def _newtype_payload_name(self, type_name: str) -> Optional[str]:
        """The payload type name if `type_name` names a newtype, else None.

        Only scalar payloads answer: an enum or pointer payload has no const
        cast of its own, and letting the caller keep the original name gives a
        diagnostic that names what was written.
        """
        from r65.compiler.hir.symbol_table import SymbolKind
        from r65.compiler.hir.types import BasicTypeInfo

        symbol = self.symbol_table.lookup(type_name)
        if symbol is None or symbol.kind != SymbolKind.NEWTYPE:
            return None
        if isinstance(symbol.type_info, BasicTypeInfo):
            return symbol.type_info.name
        return None

    def _eval_cast(self, expr: ast.TypeCast) -> Union[int, bool]:
        """Evaluate type cast."""
        value = self.eval(expr.expr)
        target_type = expr.target_type

        if isinstance(target_type, ast.BasicType):
            type_name = target_type.name

            # `x as Newtype` truncates exactly as `x as <payload>` does — a
            # newtype is its payload at runtime, so there is no separate cast to
            # perform. Rewriting the name here also keeps the diagnostic on an
            # unsupported payload naming that payload.
            payload_name = self._newtype_payload_name(type_name)
            if payload_name is not None:
                type_name = payload_name

            # Cast to integer types
            if type_name in INTEGER_TYPE_NAMES:
                from r65.compiler.hir.hir_const_eval import wrap_int
                return wrap_int(self._ensure_int(value), type_name)

            # Cast to bool
            elif type_name == 'bool':
                if isinstance(value, bool):
                    return value
                elif isinstance(value, int):
                    return value != 0
                else:
                    raise HIRError(f"Cannot cast {type(value).__name__} to bool", source_loc=expr.source_loc)

            else:
                raise HIRError(f"Cannot cast to {type_name} in const expression", source_loc=expr.source_loc)
        elif isinstance(target_type, ast.PointerType):
            # Cast to pointer type: preserve integer value (it's the address)
            int_val = self._ensure_int(value)
            if target_type.is_far:
                return int_val & 0xFFFFFF  # 24-bit far pointer
            else:
                return int_val & 0xFFFF  # 16-bit near pointer
        else:
            raise HIRError(f"Unsupported cast target in const expression: {type(target_type).__name__}", source_loc=expr.source_loc)
        
        # This should never be reached due to the raises above
        raise HIRError("Unexpected path in _eval_cast", source_loc=expr.source_loc)

    def _eval_function_call(self, expr: ast.FunctionCall) -> Union[int, bool, str]:
        """Evaluate function call in const expression."""
        # Check if this is a method call (e.g., array.len())
        if isinstance(expr.func, ast.FieldAccess):
            return self._eval_method_call(expr)

        func_name = self._const_fn_name(expr)

        if not func_name:
            raise HIRError("Only direct function calls allowed in const expressions", source_loc=expr.source_loc)

        from r65.compiler.hir.symbol_table import SymbolKind

        symbol = self.symbol_table.lookup(func_name)

        payload = self._newtype_construct_arg(expr, func_name, symbol)
        if payload is not None:
            return self.eval(payload)

        forward = self.forward_const_fns.get(func_name) if symbol is None else None
        if forward is not None:
            arg_values = [self.eval(arg) for arg in expr.args]
            return self._eval_const_fn_call(forward, arg_values, func_name)

        # Check if this is a const fn (user-defined)
        if symbol and symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            func_def = symbol.definition
            if func_def and hasattr(func_def, 'is_const') and func_def.is_const:
                # Evaluate arguments
                arg_values = [self.eval(arg) for arg in expr.args]
                return self._eval_const_fn_call(func_def, arg_values, func_name)
            elif func_def and not BuiltinRegistry.is_builtin(func_name):
                raise HIRError(f"Function '{func_name}' is not a const fn and cannot be called in const expressions", source_loc=expr.source_loc)

        # Check if this is a built-in function
        if not BuiltinRegistry.is_builtin(func_name):
            raise HIRError(f"Function '{func_name}' is not a const fn or built-in const function", source_loc=expr.source_loc)

        builtin = BuiltinRegistry.get_builtin(func_name)

        # Only type info, cfg and const math built-ins are const-evaluable
        if builtin.kind not in (BuiltinKind.TYPE_INFO, BuiltinKind.CFG, BuiltinKind.CONST_MATH):
            raise HIRError(f"Built-in '{func_name}' is not allowed in const expressions", source_loc=expr.source_loc)

        # Handle const math builtins
        if builtin.kind is BuiltinKind.CONST_MATH:
            arg_values = [self.eval(arg) for arg in expr.args]
            return self._eval_const_math(func_name, arg_values, expr.source_loc)

        # Handle size_of specifically
        if func_name == "size_of":
            return self._eval_size_of(expr)

        # Handle offset_of specifically
        if func_name == "offset_of":
            return self._eval_offset_of(expr)

        if func_name == "cfg":
            if len(expr.args) != 1:
                raise HIRError("cfg expects exactly 1 argument", source_loc=expr.source_loc)
            return self.eval_cfg(expr.args[0], expr.source_loc)

        raise HIRError(f"Unsupported const built-in function: {func_name}", source_loc=expr.source_loc)

    def eval_cfg(self, condition: ast.Expression, source_loc=None) -> bool:
        """Evaluate the argument of `cfg(...)` / `cfg!(...)`.

        The argument is a flag name (`debug`) or a key-value test
        (`target = "snes"`). With no `--cfg` options every flag is unset, the
        same default `#[cfg]` attributes use.
        """
        from r65.compiler.hir.cfg import CfgEvaluator
        from r65.compiler.frontend.ast import CfgComparison

        if isinstance(condition, ast.Identifier):
            cfg_cond = CfgIdentifier(name=condition.name)
        elif (isinstance(condition, ast.Assignment)
              and isinstance(condition.target, ast.Identifier)
              and isinstance(condition.value, ast.StringLiteral)):
            cfg_cond = CfgComparison(key=condition.target.name, operator='=',
                                     value=condition.value.value)
        else:
            raise HIRError("cfg condition must be a flag name or `key = \"value\"`",
                           source_loc=source_loc)

        evaluator = self.cfg_evaluator or CfgEvaluator(set(), {})
        try:
            return evaluator.evaluate(cfg_cond)
        except CompilerError as e:
            raise HIRError(str(e), source_loc=source_loc)

    def _eval_size_of(self, expr: ast.FunctionCall) -> int:
        """Evaluate size_of builtin function using unified type utilities."""
        if len(expr.args) != 1:
            raise HIRError("size_of expects exactly 1 argument", source_loc=expr.source_loc)

        arg = expr.args[0]

        try:
            return get_unified_type_size(arg, self.symbol_table)
        except Exception as e:
            # Not sizable yet (e.g. a struct declared later): report as non-constant.
            raise HIRError(f"Cannot evaluate size_of at this time: {e}", source_loc=expr.source_loc)

    def _eval_offset_of(self, expr: ast.FunctionCall) -> int:
        """Evaluate offset_of builtin function: offset_of(StructName, field_name)."""
        if len(expr.args) != 2:
            raise HIRError("offset_of expects exactly 2 arguments: offset_of(StructType, field)", source_loc=expr.source_loc)

        # First arg: struct type name (must be an identifier)
        struct_arg = expr.args[0]
        if not isinstance(struct_arg, ast.Identifier):
            raise HIRError("offset_of first argument must be a struct type name", source_loc=expr.source_loc)

        struct_name = struct_arg.name
        symbol = self.symbol_table.lookup(struct_name)
        if symbol is None:
            raise HIRError(f"Undefined struct in offset_of: {struct_name}", source_loc=expr.source_loc)

        struct_def = symbol.definition
        if struct_def is None or not hasattr(struct_def, 'fields'):
            raise HIRError(f"'{struct_name}' is not a struct type", source_loc=expr.source_loc)

        # Second arg: field name (must be an identifier)
        field_arg = expr.args[1]
        if not isinstance(field_arg, ast.Identifier):
            raise HIRError("offset_of second argument must be a field name", source_loc=expr.source_loc)

        field_name = field_arg.name

        # Check if this is an HIR struct (has pre-computed offsets)
        from r65.compiler.hir.nodes import HIRStructDecl
        if isinstance(struct_def, HIRStructDecl):
            for field in struct_def.fields:
                if field.name == field_name:
                    return field.offset
            raise HIRError(f"Struct '{struct_name}' has no field '{field_name}'", source_loc=expr.source_loc)

        # AST struct or union: offsets come from the shared aggregate layout
        sizes = [get_unified_type_size(f.field_type, self.symbol_table) for f in struct_def.fields]
        offsets, _ = layout_fields(sizes, getattr(struct_def, 'is_union', False))
        for field, offset in zip(struct_def.fields, offsets):
            if field.name == field_name:
                return offset

        raise HIRError(f"Struct '{struct_name}' has no field '{field_name}'", source_loc=expr.source_loc)

    def eval_method_call(self, receiver: ast.Expression, method_name: str) -> int:
        """Evaluate a method call on a receiver expression at compile time.

        Called from expression_builder for methods like bank_byte().
        """
        if method_name == 'bank_byte':
            value = self.eval(receiver)
            int_val = self._ensure_int(value)
            return (int_val >> 16) & 0xFF
        raise HIRError(f"Method '{method_name}' is not const-evaluable", source_loc=receiver.source_loc)

    def _eval_method_call(self, expr: ast.FunctionCall) -> int:
        """Evaluate method call in const expression (e.g., array.len())."""
        compiler_assert(
            isinstance(expr.func, ast.FieldAccess),
            f"_eval_method_call called with non-FieldAccess func: {type(expr.func).__name__}"
        )

        method_name = expr.func.field
        receiver = expr.func.base

        # Handle bank_byte() on far pointers
        if method_name == 'bank_byte':
            return self.eval_method_call(receiver, 'bank_byte')

        # Only len() is supported as a const method (besides bank_byte)
        if method_name != 'len':
            raise HIRError(f"Method '{method_name}' is not allowed in const expressions", source_loc=expr.source_loc)

        # Validate no arguments
        if len(expr.args) != 0:
            raise HIRError(f"len() takes no arguments, got {len(expr.args)}", source_loc=expr.source_loc)

        # Get the receiver's type - must be an identifier for const evaluation
        if not isinstance(receiver, ast.Identifier):
            raise HIRError("len() receiver must be an identifier in const expressions", source_loc=expr.source_loc)

        symbol = self.symbol_table.lookup(receiver.name)
        if symbol is None:
            raise HIRError(f"Undefined identifier: {receiver.name}", source_loc=expr.source_loc)

        # Ensure it's an array type
        if symbol.var_type is None or not isinstance(symbol.var_type, ArrayTypeInfo):
            raise HIRError(f"len() requires array type, '{receiver.name}' is not an array", source_loc=expr.source_loc)

        # Return the array size
        return symbol.var_type.size

    def _eval_const_math(self, func_name: str, args: list, source_loc) -> int:
        """Evaluate a const math builtin (fixed_sin, fixed_cos, etc.)."""
        fn = _FIXED_MATH_FNS.get(func_name)
        if fn is None:
            raise HIRError(f"Unknown const math function: {func_name}", source_loc=source_loc)
        try:
            return fn(*args)
        except (ZeroDivisionError, ValueError) as e:
            raise HIRError(str(e), source_loc=source_loc)

    def _ensure_int(self, value: Any) -> int:
        """Ensure value is an integer."""
        if isinstance(value, bool):
            return 1 if value else 0
        elif isinstance(value, int):
            return value
        else:
            raise HIRError(f"Expected integer in const expression, got {type(value).__name__}", source_loc=None)

    def _ensure_bool(self, value: Any) -> bool:
        """Ensure value is a boolean."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            return value != 0
        else:
            raise HIRError(f"Expected boolean in const expression, got {type(value).__name__}", source_loc=None)

    def _eval_match_expr(self, expr: ast.MatchExpression) -> Union[int, bool]:
        """Evaluate a match expression in const context."""
        scrutinee = self.eval(expr.scrutinee)

        for arm in expr.arms:
            if self._pattern_matches(arm.pattern, scrutinee):
                return self.eval(arm.body)

        raise HIRError("Non-exhaustive match in const expression", source_loc=expr.source_loc)

    def _pattern_matches(self, pattern, value) -> bool:
        """Check if a pattern matches a value."""
        if isinstance(pattern, ast.WildcardPattern):
            return True
        elif isinstance(pattern, ast.IdentifierPattern):
            return True  # Always matches (binding not needed in const eval)
        elif isinstance(pattern, ast.LiteralPattern):
            return value == pattern.value
        elif isinstance(pattern, ast.RangePattern):
            if pattern.inclusive:
                return pattern.start <= value <= pattern.end
            return pattern.start <= value < pattern.end
        elif isinstance(pattern, ast.EnumPattern):
            qualified = f"{pattern.enum_name}::{pattern.variant_name}"
            symbol = self.symbol_table.lookup(qualified)
            if symbol and symbol.const_value is not None:
                return value == symbol.const_value
            raise HIRError(f"Cannot resolve enum variant '{qualified}' in const match", source_loc=getattr(pattern, 'source_loc', None))
        elif isinstance(pattern, ast.OrPattern):
            return any(self._pattern_matches(sub, value) for sub in pattern.patterns)
        else:
            raise HIRError(f"Unsupported pattern in const match: {type(pattern).__name__}", source_loc=getattr(pattern, 'source_loc', None))

    # =========================================================================
    # Const fn evaluation via Python transpilation
    #
    # A const fn body is transpiled to a Python function once and run on each
    # call. The transpiler tracks the R65 type of every local and expression,
    # so values wrap at their type's width exactly where the 65816 would wrap
    # them, and it gives every local a fresh Python name (`v3_x`), so R65
    # scoping and shadowing hold and no R65 name can collide with Python's.
    # Statements that R65 allows inside an expression (a block, `match`, an
    # `if` with statements, `loop` with `break value`) are hoisted into
    # statements ahead of the expression that uses them.
    # =========================================================================

    MAX_CONST_FN_ITERATIONS = 1_000_000
    MAX_CONST_FN_DEPTH = 256

    def _eval_const_fn_call(self, func_def, arg_values, func_name):
        """Evaluate a const fn call by transpiling to Python and executing."""
        expected = len(func_def.params)
        actual = len(arg_values)
        if actual != expected:
            raise HIRError(
                f"const fn '{func_name}' expects {expected} argument(s), got {actual}",
                source_loc=getattr(func_def, 'source_loc', None)
            )

        if func_name not in self._compiled_const_fns:
            self._compiled_const_fns[func_name] = self._compile_const_fn(func_def, func_name)
        compiled_fn = self._compiled_const_fns[func_name]
        namespace = self._build_const_fn_namespace()

        loc = getattr(func_def, 'source_loc', None)
        old_limit = sys.getrecursionlimit()
        # Two Python frames per const fn call, on top of the compiler's own.
        sys.setrecursionlimit(max(old_limit, 4 * self.MAX_CONST_FN_DEPTH + 2000))
        try:
            result = compiled_fn(*arg_values, **namespace)
        except HIRError:
            raise
        except _ConstFnFault as e:
            raise HIRError(f"{e} in const fn '{func_name}'", source_loc=loc)
        except RecursionError:
            raise HIRError(f"Infinite recursion in const fn '{func_name}'", source_loc=loc)
        except Exception as e:
            raise HIRError(f"Error evaluating const fn '{func_name}': {e}", source_loc=loc)
        finally:
            sys.setrecursionlimit(old_limit)

        if result is None:
            raise HIRError(f"const fn '{func_name}' did not return a value", source_loc=loc)
        return result

    def validate_const_fn(self, func_def, func_name):
        """Eagerly validate a const fn body at definition time.

        Attempts to compile (transpile) the const fn body so that errors
        like calling non-const functions or accessing runtime variables
        are caught even if the const fn is never called in a const context.
        """
        if func_name not in self._compiled_const_fns:
            self._compiled_const_fns[func_name] = self._compile_const_fn(func_def, func_name)

    def _compile_const_fn(self, func_def, func_name):
        """Transpile a const fn AST body to a Python callable."""
        # Track that we're compiling this function (for recursive calls)
        self._compiling_const_fns.add(func_name)
        saved_ctx = getattr(self, '_ctx', None)
        try:
            return self._compile_const_fn_inner(func_def, func_name)
        finally:
            self._ctx = saved_ctx
            self._compiling_const_fns.discard(func_name)

    def _compile_const_fn_inner(self, func_def, func_name):
        """Inner implementation of const fn compilation."""
        ret = func_def.return_type
        ret_key = None if ret is None or isinstance(ret, ast.NeverType) else self._type_key(ret)
        ctx = self._ctx = _FnCtx(func_name, ret is not None and not isinstance(ret, ast.NeverType), ret_key)

        py_params = []
        prologue = []
        for p in func_def.params:
            key = self._type_key(p.param_type)
            py = ctx.declare(p.name, key)
            py_params.append(py)
            if _is_aggregate_key(key):
                prologue.append(f"    {py} = _copy({py})")
            elif key in _INT_KEYS:
                prologue.append(f"    {py} = _{key}({py})")

        body_lines = self._transpile_stmts(func_def.body.statements, 1, tail=True)

        params_str = "".join(f"{p}, " for p in py_params)
        fn_source = f"def _const_fn_({params_str}**_ns_):\n"
        fn_source += "    " + "; ".join(f"{h} = _ns_['{h}']" for h in _HELPERS) + "\n"
        fn_source += "    _loop_count_ = 0\n"
        fn_source += "\n".join(prologue + body_lines) + ("\n" if prologue or body_lines else "")
        fn_source += "    return None\n"

        try:
            code = compile(fn_source, f"<const fn {func_name}>", "exec")
        except SyntaxError as e:
            raise HIRError(
                f"Failed to compile const fn '{func_name}': {e}",
                source_loc=getattr(func_def, 'source_loc', None)
            )
        local_ns = {}
        exec(code, {}, local_ns)
        raw = local_ns['_const_fn_']
        limit = self.MAX_CONST_FN_DEPTH

        def call(*args, **ns):
            depth = ns['_depth']
            depth[0] += 1
            try:
                if depth[0] > limit:
                    raise _ConstFnFault(f"recursion deeper than {limit} calls")
                return raw(*args, **ns)
            finally:
                depth[0] -= 1

        return call

    def _build_const_fn_namespace(self):
        """Build the namespace dict passed to compiled const fns."""
        def _bool(v):
            if isinstance(v, bool):
                return v
            return int(v) != 0

        def _idiv(a, b, line=None):
            if b == 0:
                raise _ConstFnFault(f"division by zero{_at(line)}")
            return trunc_div(int(a), int(b))

        def _imod(a, b, line=None):
            if b == 0:
                raise _ConstFnFault(f"modulo by zero{_at(line)}")
            return trunc_mod(int(a), int(b))

        def _shift_amount(n, line):
            n = int(n)
            if n < 0:
                raise _ConstFnFault(f"negative shift amount {n}{_at(line)}")
            return min(n, 32)   # every R65 integer is gone by then

        def _shl(a, n, line=None):
            return int(a) << _shift_amount(n, line)

        def _shr(a, n, line=None):
            return int(a) >> _shift_amount(n, line)

        def _ix(arr, i, line=None):
            if not isinstance(arr, list):
                raise _ConstFnFault(f"indexing a value that is not an array{_at(line)}")
            i = int(i)
            if i < 0 or i >= len(arr):
                raise _ConstFnFault(
                    f"index {i} out of bounds for an array of length {len(arr)}{_at(line)}")
            return i

        def _fill(v, n):
            return [copy.deepcopy(v) for _ in range(int(n))]

        def _fault(msg):
            raise _ConstFnFault(msg)

        ns = {
            '_u8': lambda v: wrap_int(int(v), 'u8'),
            '_u16': lambda v: wrap_int(int(v), 'u16'),
            '_i8': lambda v: wrap_int(int(v), 'i8'),
            '_i16': lambda v: wrap_int(int(v), 'i16'),
            '_bool': _bool, '_idiv': _idiv, '_imod': _imod,
            '_shl': _shl, '_shr': _shr, '_ix': _ix, '_fill': _fill,
            '_copy': copy.deepcopy, '_fault': _fault, '_LoopExit': _LoopExit,
            '_depth': [0],
            **_FIXED_MATH_FNS,
        }

        # Add all compiled const fns to namespace so they can call each other
        for name, fn in self._compiled_const_fns.items():
            ns[name] = fn

        return ns

    # ---- types ---------------------------------------------------------------

    def _type_key(self, ast_type):
        """The transpiler's key for an AST type (see `_key_of_info`)."""
        if ast_type is None:
            return None
        if isinstance(ast_type, ast.BasicType) and (
                ast_type.name in INTEGER_TYPE_NAMES or ast_type.name == 'bool'):
            return ast_type.name
        if isinstance(ast_type, ast.ArrayType):
            elem = self._type_key(ast_type.element_type)
            try:
                size = self._ensure_int(self.eval(ast_type.size))
            except HIRError:
                size = None
            return ('arr', elem, size)
        if isinstance(ast_type, ast.PointerType):
            return 'u24' if ast_type.is_far else 'u16'
        resolver = getattr(self, 'type_resolver', None)
        if resolver is None:
            return None
        try:
            info = resolver.resolve_type(ast_type)
        except Exception:
            return None
        return self._key_of_info(info)

    def _key_of_info(self, info):
        """The transpiler's key for an HIR TypeInfo.

        An integer type name, 'bool', ('arr', elem key, size) or
        ('struct', name); None when unknown. Newtypes and enums are keyed
        by their payload width, which is all const evaluation needs.
        """
        from r65.compiler.hir.types import (
            BasicTypeInfo, NewtypeTypeInfo, EnumTypeInfo, StructTypeInfo,
            PointerTypeInfo, strip_newtype)
        if info is None:
            return None
        if isinstance(info, NewtypeTypeInfo):
            return self._key_of_info(strip_newtype(info))
        if isinstance(info, BasicTypeInfo):
            return info.name if (info.name in INTEGER_TYPE_NAMES or info.name == 'bool') else None
        if isinstance(info, EnumTypeInfo):
            try:
                return 'u8' if info.size_bytes == 1 else 'u16'
            except Exception:
                return 'u8'
        if isinstance(info, ArrayTypeInfo):
            return ('arr', self._key_of_info(info.element_type), info.size)
        if isinstance(info, StructTypeInfo):
            return ('struct', info.name)
        if isinstance(info, PointerTypeInfo):
            return 'u24' if info.is_far else 'u16'
        return None

    def _field_key(self, struct_key, field_name):
        """The key of *field_name* in a ('struct', name) key, or None."""
        if not (isinstance(struct_key, tuple) and struct_key[0] == 'struct'):
            return None
        symbol = self.symbol_table.lookup(struct_key[1])
        definition = getattr(symbol, 'definition', None) if symbol else None
        for f in getattr(definition, 'fields', None) or []:
            if f.name != field_name:
                continue
            info = getattr(f, 'type_info', None) or getattr(f, 'field_type', None)
            if isinstance(info, ast.ASTNode):
                return self._type_key(info)
            return self._key_of_info(info)
        return None

    def _literal_default_key(self, expr):
        """The type the type checker gives `let x = <expr>;` when *expr* is
        untyped literal arithmetic: the smallest type that holds its value."""
        try:
            value = self.eval(expr)
        except HIRError:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        from r65.compiler.typeck.type_inference import TypeInference
        info = TypeInference.infer_integer_literal_type(value, None)
        return self._key_of_info(info)

    def _zero_value(self, key, loc):
        """Python source for the zero value of *key* (an uninitialized `let`)."""
        if key in _INT_KEYS or key in ('u24', None):
            return "0"
        if key == 'bool':
            return "False"
        if key[0] == 'arr':
            if key[2] is None:
                raise HIRError("array length of an uninitialized local is not constant",
                               source_loc=loc)
            return f"_fill({self._zero_value(key[1], loc)}, {key[2]})"
        if key[0] == 'struct':
            symbol = self.symbol_table.lookup(key[1])
            fields = getattr(getattr(symbol, 'definition', None), 'fields', None) or []
            parts = ", ".join(f"'{f.name}': {self._zero_value(self._field_key(key, f.name), loc)}"
                              for f in fields)
            return f"{{{parts}}}"
        return "0"

    # ---- statements ----------------------------------------------------------

    def _transpile_stmts(self, stmts, indent, tail=False):
        """Transpile statements to lines at *indent*, in a new scope.

        With *tail*, a trailing expression statement is the function's value,
        as `_add_implicit_return` treats it in the HIR builder.
        """
        ctx = self._ctx
        ctx.scopes.append({})
        lines = []
        try:
            for i, stmt in enumerate(stmts):
                is_tail = (tail and i == len(stmts) - 1 and ctx.returns_value
                           and isinstance(stmt, ast.ExprStmt)
                           and not isinstance(stmt.expr, (ast.Assignment, ast.CompoundAssignment)))
                if (is_tail and isinstance(stmt.expr, ast.MatchExpression)
                        and stmt.expr.origin == 'let_else'):
                    # The rest of the body sits in the first arm; its tail is ours.
                    lines.extend(self._with_pre(
                        lambda: self._match(stmt.expr, indent, None, tail=True), indent))
                    continue
                if is_tail and getattr(stmt.expr, 'origin', None) is None:
                    stmt = ast.ReturnStmt(values=[stmt.expr], source_loc=stmt.source_loc)
                lines.extend(self._with_pre(lambda: self._transpile_stmt(stmt, indent), indent))
        finally:
            ctx.scopes.pop()
        return lines

    def _with_pre(self, build, indent):
        """Run *build* (which returns lines) collecting hoisted lines before them."""
        ctx = self._ctx
        saved_pre, saved_indent = ctx.pre, ctx.indent
        ctx.pre, ctx.indent = [], indent
        try:
            body = build()
            return ctx.pre + body
        finally:
            ctx.pre, ctx.indent = saved_pre, saved_indent

    def _block_lines(self, stmts, indent):
        lines = self._transpile_stmts(stmts, indent)
        return lines or ["    " * indent + "pass"]

    def _transpile_stmt(self, stmt, indent):
        """Transpile a single AST statement to Python source lines."""
        ctx = self._ctx
        prefix = "    " * indent
        loc = getattr(stmt, 'source_loc', None)

        if isinstance(stmt, ast.LetStmt):
            declared = self._type_key(stmt.var_type) if stmt.var_type is not None else None
            if stmt.initializer is not None:
                code, key = self._tx(stmt.initializer)
                if declared is not None:
                    key = declared
                elif key is None:
                    key = self._literal_default_key(stmt.initializer)
                code = self._coerce(code, key)
            else:
                key = declared
                code = self._zero_value(key, loc)
            py = ctx.declare(stmt.name, key)
            return [f"{prefix}{py} = {code}"]

        if isinstance(stmt, ast.ReturnStmt):
            if not stmt.values:
                return [f"{prefix}return None"]
            if len(stmt.values) > 1:
                return [f"{prefix}_fault('returns {len(stmt.values)} values; "
                        f"a constant holds one{_at(_line(loc))}')"]
            code, _ = self._tx(stmt.values[0])
            return [f"{prefix}return {self._coerce(code, ctx.ret_key)}"]

        if isinstance(stmt, ast.IfStmt):
            cond, _ = self._tx(stmt.condition)
            lines = [f"{prefix}if {cond}:"]
            lines += self._block_lines(stmt.then_block.statements, indent + 1)
            if isinstance(stmt.else_block, ast.IfStmt):
                lines.append(f"{prefix}else:")
                lines += self._with_pre(
                    lambda: self._transpile_stmt(stmt.else_block, indent + 1), indent + 1)
            elif isinstance(stmt.else_block, ast.Block):
                lines.append(f"{prefix}else:")
                lines += self._block_lines(stmt.else_block.statements, indent + 1)
            return lines

        if isinstance(stmt, ast.WhileStmt):
            return self._loop(stmt.label, None, indent, stmt.condition,
                              lambda i: self._block_lines(stmt.body.statements, i))

        if isinstance(stmt, ast.LoopStmt):
            return self._loop(stmt.label, None, indent, None,
                              lambda i: self._block_lines(stmt.body.statements, i))

        if isinstance(stmt, ast.ForStmt):
            start, skey = self._tx(stmt.start)
            end, ekey = self._tx(stmt.end)
            if stmt.inclusive:
                end = f"({end}) + 1"
            ctx.scopes.append({})
            try:
                var = ctx.declare(stmt.variable, _unify(skey, ekey))
                return self._loop(stmt.label, None, indent, None,
                                  lambda i: self._block_lines(stmt.body.statements, i),
                                  header=f"for {var} in range({start}, {end}):")
            finally:
                ctx.scopes.pop()

        if isinstance(stmt, ast.BreakStmt):
            return self._break(stmt, prefix, loc)

        if isinstance(stmt, ast.ContinueStmt):
            target = self._loop_target(stmt.label, loc)
            if target is ctx.loops[-1]:
                return [f"{prefix}continue"]
            target.continue_exc = True
            return [f"{prefix}raise _LoopExit({target.id}, 'c')"]

        if isinstance(stmt, ast.ExprStmt):
            return self._expr_stmt(stmt.expr, indent)

        if isinstance(stmt, ast.Block):
            return self._transpile_stmts(stmt.statements, indent)

        if isinstance(stmt, ast.AsmStmt):
            raise HIRError("Inline assembly (asm!) cannot be used in const fn", source_loc=loc)

        if isinstance(stmt, ast.ConstAssertStmt):
            raise HIRError("const_assert! is not supported in const fn", source_loc=loc)

        if isinstance(stmt, ast.MultiLetStmt):
            raise HIRError("`let a, b = ...` is not supported in const fn: a const fn "
                           "returns one value", source_loc=loc)

        raise HIRError(f"Unsupported statement in const fn: {type(stmt).__name__}", source_loc=loc)

    def _expr_stmt(self, expr, indent):
        """An expression used as a statement: its value, if any, is dropped."""
        prefix = "    " * indent
        if isinstance(expr, ast.Assignment):
            place, key = self._place(expr.target)
            code, _ = self._tx(expr.value)
            return [f"{prefix}{place} = {self._coerce(code, key)}"]
        if isinstance(expr, ast.CompoundAssignment):
            place, key = self._place(expr.target)
            code, rkey = self._tx(expr.value)
            value, _ = self._binary(expr.operator, place, key, code, rkey, expr)
            return [f"{prefix}{place} = {self._coerce(value, key)}"]
        if isinstance(expr, ast.MatchExpression):
            return self._match(expr, indent, None)
        if isinstance(expr, ast.LoopExpression):
            return self._loop(expr.label, None, indent, None,
                              lambda i: self._block_lines(expr.body.statements, i))
        code, _ = self._tx(expr)
        return [f"{prefix}{code}"]

    # ---- loops -------------------------------------------------------------

    def _loop(self, label, tmp, indent, condition, body_fn, header=None):
        """A Python loop for an R65 loop; *body_fn(indent)* gives its body.

        A `break`/`continue` aimed at an enclosing loop has no Python form,
        so it raises _LoopExit, and the loop it names catches it.
        """
        ctx = self._ctx
        loop = _LoopCtx(ctx.next_id(), label, tmp)
        ctx.loops.append(loop)
        try:
            inner = indent + 1
            pad = "    " * inner
            guard = [f"{pad}_loop_count_ += 1",
                     f"{pad}if _loop_count_ > {self.MAX_CONST_FN_ITERATIONS}: "
                     f"_fault('exceeded maximum iteration limit "
                     f"({self.MAX_CONST_FN_ITERATIONS:,} loop passes)')"]
            if header is not None:
                head, cond_lines = header, []
            elif condition is None:
                head, cond_lines = "while True:", []
            else:
                cond_pre = []

                def build_cond():
                    code, _ = self._tx(condition)
                    cond_pre.append(code)
                    return []
                pre = self._with_pre(build_cond, inner)
                if pre:
                    head = "while True:"
                    cond_lines = pre + [f"{pad}if not ({cond_pre[0]}): break"]
                else:
                    head, cond_lines = f"while {cond_pre[0]}:", []
            body = body_fn(inner)
            if loop.continue_exc:
                body = ([f"{pad}try:"] + _indent(body) +
                        [f"{pad}except _LoopExit as _e_:",
                         f"{pad}    if _e_.args != ({loop.id}, 'c'): raise"])
            lines = ["    " * indent + head] + guard + cond_lines + body
            if loop.break_exc:
                prefix = "    " * indent
                lines = ([f"{prefix}try:"] + _indent(lines) +
                         [f"{prefix}except _LoopExit as _e_:",
                          f"{prefix}    if _e_.args != ({loop.id}, 'b'): raise"])
            return lines
        finally:
            ctx.loops.pop()

    def _loop_target(self, label, loc):
        ctx = self._ctx
        if not ctx.loops:
            raise HIRError("`break`/`continue` outside a loop in const fn", source_loc=loc)
        if label is None:
            return ctx.loops[-1]
        for loop in reversed(ctx.loops):
            if loop.label == label:
                return loop
        raise HIRError(f"no enclosing loop labeled '{label}' in const fn", source_loc=loc)

    def _break(self, stmt, prefix, loc):
        ctx = self._ctx
        target = self._loop_target(stmt.label, loc)
        lines = []
        if stmt.value is not None:
            if target.tmp is None:
                raise HIRError("`break` with a value outside a `loop` expression",
                               source_loc=loc)
            code, key = self._tx(stmt.value)
            target.keys.append(key)
            lines.append(f"{prefix}{target.tmp} = {code}")
        if target is ctx.loops[-1]:
            return lines + [f"{prefix}break"]
        target.break_exc = True
        return lines + [f"{prefix}raise _LoopExit({target.id}, 'b')"]

    # ---- expressions ---------------------------------------------------------

    def _hoist(self, line):
        """Emit *line* ahead of the statement being transpiled."""
        self._ctx.pre.append("    " * self._ctx.indent + line)

    def _coerce(self, code, key):
        """*code* converted to a place of type *key* (wrapping an integer)."""
        if key in _INT_KEYS:
            return f"_{key}({code})"
        if key == 'u24':
            return f"(int({code}) & 0xFFFFFF)"
        return code

    def _place(self, target):
        """(Python place, key) for an assignment target."""
        loc = getattr(target, 'source_loc', None)
        if isinstance(target, ast.Identifier):
            local = self._ctx.lookup(target.name)
            if local is None:
                self._reject_global(target)
                raise HIRError(f"cannot assign to '{target.name}' in const fn", source_loc=loc)
            return local
        if isinstance(target, ast.ArrayIndex):
            arr, akey = self._tx(target.array)
            idx, _ = self._tx(target.index)
            elem = akey[1] if isinstance(akey, tuple) and akey[0] == 'arr' else None
            return f"{arr}[_ix({arr}, {idx}, {_line(loc)})]", elem
        if isinstance(target, ast.FieldAccess):
            base, bkey = self._tx(target.base)
            return f"{base}['{target.field}']", self._field_key(bkey, target.field)
        if isinstance(target, ast.Register):
            raise HIRError(f"Cannot access hardware register '{target.name}' in const fn",
                           source_loc=loc)
        raise HIRError(f"cannot assign to this in const fn: {type(target).__name__}",
                       source_loc=loc)

    def _reject_global(self, expr):
        from r65.compiler.hir.symbol_table import SymbolKind
        symbol = self.symbol_table.lookup(expr.name)
        if symbol and symbol.kind == SymbolKind.REGISTER:
            raise HIRError(f"Cannot access hardware register '{expr.name}' in const fn",
                           source_loc=expr.source_loc)
        if symbol and symbol.kind == SymbolKind.STATIC_VAR:
            raise HIRError(f"Cannot access runtime variable '{expr.name}' in const fn",
                           source_loc=expr.source_loc)

    def _tx(self, expr):
        """Transpile an AST expression to (Python expression, type key)."""
        from r65.compiler.hir.symbol_table import SymbolKind
        loc = getattr(expr, 'source_loc', None)

        if isinstance(expr, ast.IntegerLiteral):
            return str(expr.value), getattr(expr, 'suffix', None)

        if isinstance(expr, ast.BooleanLiteral):
            return ("True" if expr.value else "False"), 'bool'

        if isinstance(expr, ast.Identifier):
            local = self._ctx.lookup(expr.name)
            if local is not None:
                return local
            symbol = self._lookup_const(expr.name)
            if symbol and symbol.kind in (SymbolKind.CONST, SymbolKind.IMPL_CONST) \
                    and symbol.const_value is not None:
                return repr(symbol.const_value), self._key_of_info(symbol.var_type)
            self._reject_global(expr)
            raise HIRError(f"Undefined identifier '{expr.name}' in const fn", source_loc=loc)

        if isinstance(expr, ast.EnumVariantExpr):
            qualified = f"{expr.enum_name}::{expr.variant_name}"
            symbol = self.symbol_table.lookup(qualified)
            if symbol and symbol.const_value is not None:
                return repr(symbol.const_value), self._key_of_info(symbol.var_type)
            raise HIRError(f"Cannot resolve '{qualified}' in const fn", source_loc=loc)

        if isinstance(expr, ast.BinaryOp):
            if expr.op in ('&&', '||'):
                return self._short_circuit(expr)
            left, lkey = self._tx(expr.left)
            right, rkey = self._tx(expr.right)
            return self._binary(expr.op, left, lkey, right, rkey, expr)

        if isinstance(expr, ast.UnaryOp):
            operand, key = self._tx(expr.operand)
            if expr.op == '!':
                return f"(not {operand})", 'bool'
            if expr.op == '~':
                return self._coerce(f"(~{operand})", key), key
            if expr.op == '-':
                return self._coerce(f"(-{operand})", key), key
            raise HIRError(f"Unsupported unary op in const fn: {expr.op}", source_loc=loc)

        if isinstance(expr, ast.TypeCast):
            inner, _ = self._tx(expr.expr)
            key = self._type_key(expr.target_type)
            if key == 'bool':
                return f"_bool({inner})", 'bool'
            if key in _INT_KEYS or key == 'u24':
                return self._coerce(inner, key), key
            raise HIRError(f"Unsupported cast in const fn: {expr.target_type}", source_loc=loc)

        if isinstance(expr, ast.CfgFunctionCall):
            return ("True" if self.eval_cfg(expr.condition, loc) else "False"), 'bool'

        if isinstance(expr, ast.FunctionCall):
            return self._call(expr)

        if isinstance(expr, (ast.Assignment, ast.CompoundAssignment)):
            raise HIRError("an assignment is a statement, not a value, in const fn",
                           source_loc=loc)

        if isinstance(expr, ast.BlockExpression):
            self._ctx.scopes.append({})
            try:
                for stmt in expr.statements:
                    self._ctx.pre.extend(self._with_pre(
                        lambda: self._transpile_stmt(stmt, self._ctx.indent), self._ctx.indent))
                return self._tx(expr.final_expr)
            finally:
                self._ctx.scopes.pop()

        if isinstance(expr, ast.IfExpression):
            return self._if_expr(expr)

        if isinstance(expr, ast.MatchExpression):
            tmp = self._ctx.temp()
            keys = []
            self._ctx.pre.extend(self._match(expr, self._ctx.indent, (tmp, keys)))
            return tmp, _unify(*keys)

        if isinstance(expr, ast.LoopExpression):
            tmp = self._ctx.temp()
            self._hoist(f"{tmp} = None")
            lines = self._loop(expr.label, tmp, self._ctx.indent, None,
                               lambda i: self._block_lines(expr.body.statements, i))
            self._ctx.pre.extend(lines)
            return tmp, None

        if isinstance(expr, ast.Register):
            raise HIRError(f"Cannot access hardware register '{expr.name}' in const fn",
                           source_loc=loc)

        if isinstance(expr, ast.StringLiteral):
            raise HIRError("String literals are not supported in const fn", source_loc=loc)

        if isinstance(expr, ast.ArrayIndex):
            arr, akey = self._tx(expr.array)
            idx, _ = self._tx(expr.index)
            elem = akey[1] if isinstance(akey, tuple) and akey[0] == 'arr' else None
            return f"{arr}[_ix({arr}, {idx}, {_line(loc)})]", elem

        if isinstance(expr, ast.FieldAccess):
            base, bkey = self._tx(expr.base)
            return f"{base}['{expr.field}']", self._field_key(bkey, expr.field)

        if isinstance(expr, ast.NewtypeFieldAccess):
            # `t.0` is a retype, not a field load: a newtype *is* its payload at
            # compile time, so the payload is the operand's own value.
            self._check_newtype_field_index(expr)
            return self._tx(expr.base)

        if isinstance(expr, ast.Dereference):
            raise HIRError("Pointer dereference is not supported in const fn", source_loc=loc)

        if isinstance(expr, ast.AddressOf):
            raise HIRError("Address-of operator is not supported in const fn", source_loc=loc)

        if isinstance(expr, ast.ArrayLiteralExpr):
            parts = [self._tx(e) for e in expr.elements]
            elem = _unify(*(k for _, k in parts))
            code = ", ".join(self._coerce(c, elem) for c, _ in parts)
            return f"[{code}]", ('arr', elem, len(parts))

        if isinstance(expr, ast.ArrayFillExpr):
            value, vkey = self._tx(expr.value)
            count, _ = self._tx(expr.count)
            return f"_fill({value}, {count})", ('arr', vkey, None)

        if isinstance(expr, ast.StructLiteralExpr):
            key = ('struct', expr.struct_name) if hasattr(expr, 'struct_name') else None
            fields = ", ".join(
                f"'{f.name}': {self._coerce(self._tx(f.value)[0], self._field_key(key, f.name))}"
                for f in expr.fields)
            return f"{{{fields}}}", key

        if isinstance(expr, ast.MultiAssignment):
            raise HIRError("Multiple assignment is not supported in const fn", source_loc=loc)

        if isinstance(expr, ast.IncludeBytesExpr):
            raise HIRError("include_bytes! is not supported in const fn", source_loc=loc)

        raise HIRError(f"Unsupported expression in const fn: {type(expr).__name__}",
                       source_loc=loc)

    def _binary(self, op, left, lkey, right, rkey, expr):
        """(code, key) for `left op right`, wrapped to the operation's type."""
        line = _line(getattr(expr, 'source_loc', None))
        if op in ('==', '!=', '<', '>', '<=', '>='):
            return f"({left} {op} {right})", 'bool'
        if op in ('<<', '>>'):
            helper = '_shl' if op == '<<' else '_shr'
            return self._coerce(f"{helper}({left}, {right}, {line})", lkey), lkey
        key = _unify(lkey, rkey)
        if op == '/':
            code = f"_idiv({left}, {right}, {line})"
        elif op == '%':
            code = f"_imod({left}, {right}, {line})"
        elif op in ('+', '-', '*', '&', '|', '^'):
            code = f"({left} {op} {right})"
        else:
            raise HIRError(f"Unsupported operator in const fn: {op}",
                           source_loc=getattr(expr, 'source_loc', None))
        if key == 'bool':
            return f"_bool({code})", key
        return self._coerce(code, key), key

    def _short_circuit(self, expr):
        """`&&`/`||`, keeping the right side unevaluated when it is skipped."""
        ctx = self._ctx
        left, _ = self._tx(expr.left)
        right_code = []

        def build_right():
            code, _ = self._tx(expr.right)
            right_code.append(code)
            return []
        right_pre = self._with_pre(build_right, ctx.indent + 1)
        py_op = 'and' if expr.op == '&&' else 'or'
        if not right_pre:
            return f"(_bool({left}) {py_op} _bool({right_code[0]}))", 'bool'
        tmp = ctx.temp()
        self._hoist(f"{tmp} = _bool({left})")
        self._hoist(f"if {'' if expr.op == '&&' else 'not '}{tmp}:")
        ctx.pre.extend(right_pre)
        ctx.pre.append("    " * (ctx.indent + 1) + f"{tmp} = _bool({right_code[0]})")
        return tmp, 'bool'

    def _branch_value(self, branch, tmp, indent):
        """Lines that evaluate an if-expression branch into *tmp*, and its key."""
        keys = []

        def build():
            code, key = self._tx(branch)
            keys.append(key)
            return ["    " * indent + f"{tmp} = {code}"]
        return self._with_pre(build, indent), keys[0]

    def _if_expr(self, expr):
        ctx = self._ctx
        cond, _ = self._tx(expr.condition)
        tmp = ctx.temp()
        then_lines, tkey = self._branch_value(expr.then_block, tmp, ctx.indent + 1)
        else_lines, ekey = self._branch_value(expr.else_block, tmp, ctx.indent + 1)
        key = _unify(tkey, ekey)
        if len(then_lines) == 1 and len(else_lines) == 1:
            then_code = then_lines[0].split(" = ", 1)[1]
            else_code = else_lines[0].split(" = ", 1)[1]
            return f"(({then_code}) if ({cond}) else ({else_code}))", key
        self._hoist(f"if {cond}:")
        ctx.pre.extend(then_lines)
        self._hoist("else:")
        ctx.pre.extend(else_lines)
        return tmp, key

    def _call(self, expr):
        """A call inside a const fn body: a const fn, a builtin or a newtype."""
        from r65.compiler.hir.symbol_table import SymbolKind
        loc = getattr(expr, 'source_loc', None)
        func_name = self._const_fn_name(expr)
        if func_name is None:
            raise HIRError(f"Unsupported function call in const fn: {expr}", source_loc=loc)

        if func_name in ('size_of', 'offset_of') and BuiltinRegistry.is_builtin(func_name):
            evaluate = self._eval_size_of if func_name == 'size_of' else self._eval_offset_of
            return str(evaluate(expr)), None

        builtin = BuiltinRegistry.get_builtin(func_name)
        if builtin and builtin.kind is BuiltinKind.CONST_MATH:
            args = ", ".join(self._tx(a)[0] for a in expr.args)
            return f"_ns_['{func_name}']({args})", None
        if builtin and builtin.kind is BuiltinKind.CFG:
            return ("True" if self.eval(expr) else "False"), 'bool'

        symbol = self.symbol_table.lookup(func_name)
        payload = self._newtype_construct_arg(expr, func_name, symbol)
        if payload is not None:
            code, _ = self._tx(payload)
            key = self._key_of_info(symbol.type_info) if symbol else None
            return self._coerce(code, key), key

        forward = self.forward_const_fns.get(func_name) if symbol is None else None
        if forward is not None or (symbol and symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)):
            func_def = forward if forward is not None else symbol.definition
            if func_def is not None and getattr(func_def, 'is_const', False):
                if len(expr.args) != len(func_def.params):
                    raise HIRError(
                        f"const fn '{func_name}' expects {len(func_def.params)} "
                        f"argument(s), got {len(expr.args)}", source_loc=loc)
                if (func_name not in self._compiled_const_fns
                        and func_name not in self._compiling_const_fns):
                    self._compiled_const_fns[func_name] = self._compile_const_fn(func_def, func_name)
                args = "".join(f"{self._tx(a)[0]}, " for a in expr.args)
                ret = func_def.return_type
                key = None if ret is None or isinstance(ret, ast.NeverType) else self._type_key(ret)
                return f"_ns_['{func_name}']({args}**_ns_)", key
            raise HIRError(
                f"Cannot call '{func_name}' from const fn: '{func_name}' is not a const fn",
                source_loc=loc)

        raise HIRError(f"Unsupported function call in const fn: {func_name}", source_loc=loc)

    def _const_fn_name(self, expr):
        """The symbol name a call resolves to: `f(..)` or `Type::method(..)`."""
        if isinstance(expr.func, ast.Identifier):
            return expr.func.name
        if isinstance(expr.func, ast.EnumVariantExpr):
            return f"{expr.func.enum_name}__{expr.func.variant_name}"
        return None

    # ---- match ---------------------------------------------------------------

    def _match(self, expr, indent, into, tail=False):
        """Statement-form `match`; *into* is (tmp, keys) when it has a value."""
        ctx = self._ctx
        prefix = "    " * indent
        loc = getattr(expr, 'source_loc', None)
        if not expr.arms:
            raise HIRError("Empty match expression in const fn", source_loc=loc)
        scrut = ctx.temp()
        lines = []

        def build_scrutinee():
            code, _ = self._tx(expr.scrutinee)
            return [f"{prefix}{scrut} = {code}"]
        lines += self._with_pre(build_scrutinee, indent)

        first = True
        for arm in expr.arms:
            cond, binding = self._pattern_cond(arm.pattern, scrut)
            if cond is None:
                lines.append(f"{prefix}else:" if not first else f"{prefix}if True:")
            else:
                lines.append(f"{prefix}{'if' if first else 'elif'} {cond}:")
            lines += self._arm_body(arm.body, binding, scrut, indent + 1, into,
                                    tail=tail and first)
            first = False
            if cond is None:
                return lines
        lines.append(f"{prefix}else:")
        lines.append(f"{prefix}    _fault('no match arm matches {{}}{_at(_line(loc))}'.format({scrut}))")
        return lines

    def _arm_body(self, body, binding, scrut, indent, into, tail=False):
        ctx = self._ctx
        pad = "    " * indent
        ctx.scopes.append({})
        try:
            lines = []
            if binding is not None:
                lines.append(f"{pad}{ctx.declare(binding, None)} = {scrut}")
            if isinstance(body, (ast.ReturnStmt, ast.BreakStmt, ast.ContinueStmt)):
                lines += self._with_pre(lambda: self._transpile_stmt(body, indent), indent)
            elif isinstance(body, ast.Block):
                lines += self._transpile_stmts(body.statements, indent, tail=tail)
            elif into is None:
                lines += self._with_pre(lambda: self._expr_stmt(body, indent), indent)
            else:
                tmp, keys = into

                def build():
                    code, key = self._tx(body)
                    keys.append(key)
                    return [f"{pad}{tmp} = {code}"]
                lines += self._with_pre(build, indent)
            return lines or [f"{pad}pass"]
        finally:
            ctx.scopes.pop()

    def _pattern_cond(self, pattern, scrut):
        """(Python condition or None for irrefutable, binding name or None)."""
        loc = getattr(pattern, 'source_loc', None)
        if isinstance(pattern, ast.WildcardPattern):
            return None, None
        if isinstance(pattern, ast.IdentifierPattern):
            return None, (None if pattern.name == '_' else pattern.name)
        if isinstance(pattern, ast.LiteralPattern):
            return f"{scrut} == {pattern.value!r}", None
        if isinstance(pattern, ast.RangePattern):
            op = '<=' if pattern.inclusive else '<'
            return f"({pattern.start!r} <= {scrut} {op} {pattern.end!r})", None
        if isinstance(pattern, ast.EnumPattern):
            if pattern.binding is not None:
                raise HIRError("tagged enum patterns are not supported in const fn",
                               source_loc=loc)
            qualified = f"{pattern.enum_name}::{pattern.variant_name}"
            symbol = self.symbol_table.lookup(qualified)
            if symbol and symbol.const_value is not None:
                return f"{scrut} == {symbol.const_value!r}", None
            raise HIRError(f"Cannot resolve enum variant '{qualified}' in const fn match",
                           source_loc=loc)
        if isinstance(pattern, ast.OrPattern):
            conds = []
            for sub in pattern.patterns:
                cond, binding = self._pattern_cond(sub, scrut)
                if binding is not None:
                    raise HIRError("a binding inside an or-pattern is not supported in const fn",
                                   source_loc=loc)
                if cond is None:
                    return None, None
                conds.append(cond)
            return "(" + " or ".join(conds) + ")", None
        raise HIRError(f"Unsupported pattern in const fn match: {type(pattern).__name__}",
                       source_loc=loc)


# ---- transpiler support ------------------------------------------------------

_INT_KEYS = frozenset(INTEGER_TYPE_NAMES)
_HELPERS = ('_u8', '_u16', '_i8', '_i16', '_bool', '_idiv', '_imod', '_shl', '_shr',
            '_ix', '_fill', '_copy', '_fault', '_LoopExit')
_WIDTH = {'u8': 1, 'i8': 1, 'u16': 2, 'i16': 2, 'u24': 3}


class _ConstFnFault(Exception):
    """An error raised by running const fn code (bad index, division by zero,
    no matching arm, iteration or recursion limit). Reported as an HIRError
    naming the const fn."""


class _LoopExit(Exception):
    """`break`/`continue` aimed past the innermost loop; args (loop id, kind)."""


class _LoopCtx:
    def __init__(self, loop_id, label, tmp):
        self.id = loop_id
        self.label = label
        self.tmp = tmp            # the loop expression's value, or None
        self.keys = []            # types of the values `break` gives it
        self.break_exc = False
        self.continue_exc = False


class _FnCtx:
    """Per-function transpiler state: scopes, loops, hoisted lines."""

    def __init__(self, func_name, returns_value, ret_key):
        self.func_name = func_name
        self.returns_value = returns_value
        self.ret_key = ret_key
        self.scopes = [{}]
        self.loops = []
        self.pre = []
        self.indent = 1
        self._counter = 0

    def next_id(self):
        self._counter += 1
        return self._counter

    def declare(self, name, key):
        py = f"v{self.next_id()}_{name}"
        self.scopes[-1][name] = (py, key)
        return py

    def temp(self):
        return f"_t{self.next_id()}_"

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None


def _unify(*keys):
    """The type of an operation on values of *keys*: the wider integer wins;
    an untyped literal (None) takes the other side's type."""
    result = None
    for key in keys:
        if key is None:
            continue
        if result is None:
            result = key
        elif result in _WIDTH and key in _WIDTH and _WIDTH[key] > _WIDTH[result]:
            result = key
    return result


def _is_aggregate_key(key):
    return isinstance(key, tuple)


def _line(loc):
    return getattr(loc, 'line', None)


def _at(line):
    return f" (line {line})" if line else ""


def _indent(lines):
    return ["    " + line for line in lines]
