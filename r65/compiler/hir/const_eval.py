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
        self._compiled_const_fns = {}  # Cache: func_name -> compiled Python callable
        self._compiling_const_fns = set()  # Names currently being compiled (for recursion)
        self._const_fn_depth = 0       # Recursion depth counter (max 64)

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

        elif isinstance(expr, ast.EnumVariantExpr):
            # Resolve enum variant to its integer value
            qualified = f"{expr.enum_name}::{expr.variant_name}"
            symbol = self.symbol_table.lookup(qualified)
            if symbol is None:
                raise HIRError(f"Undefined enum variant in const expression: {qualified}")
            if symbol.const_value is None:
                raise HIRError(f"Enum variant '{qualified}' has no evaluated value")
            return symbol.const_value

        elif isinstance(expr, ast.BinaryOp):
            return self._eval_binary_op(expr)

        elif isinstance(expr, ast.UnaryOp):
            return self._eval_unary_op(expr)

        elif isinstance(expr, ast.TypeCast):
            return self._eval_cast(expr)

        elif isinstance(expr, ast.FunctionCall):
            return self._eval_function_call(expr)

        elif isinstance(expr, ast.MatchExpression):
            return self._eval_match_expr(expr)

        elif isinstance(expr, ast.BlockExpression):
            # Const eval: block expression with no statements, just final expr
            if expr.statements:
                raise HIRError(f"Block expression with statements is not const-evaluable")
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

        # Check if this is a const fn (user-defined)
        from r65.compiler.hir.symbol_table import SymbolKind
        symbol = self.symbol_table.lookup(func_name)
        if symbol and symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            func_def = symbol.definition
            if func_def and hasattr(func_def, 'is_const') and func_def.is_const:
                # Evaluate arguments
                arg_values = [self.eval(arg) for arg in expr.args]
                return self._eval_const_fn_call(func_def, arg_values, func_name)
            elif func_def and not BuiltinRegistry.is_builtin(func_name):
                raise HIRError(f"Function '{func_name}' is not a const fn and cannot be called in const expressions")

        # Check if this is a built-in function
        if not BuiltinRegistry.is_builtin(func_name):
            raise HIRError(f"Function '{func_name}' is not a const fn or built-in const function")

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

    def _eval_match_expr(self, expr: ast.MatchExpression) -> Union[int, bool]:
        """Evaluate a match expression in const context."""
        scrutinee = self.eval(expr.scrutinee)

        for arm in expr.arms:
            if self._pattern_matches(arm.pattern, scrutinee):
                return self.eval(arm.body)

        raise HIRError("Non-exhaustive match in const expression")

    def _pattern_matches(self, pattern, value) -> bool:
        """Check if a pattern matches a value."""
        if isinstance(pattern, ast.WildcardPattern):
            return True
        elif isinstance(pattern, ast.IdentifierPattern):
            return True  # Always matches (binding not needed in const eval)
        elif isinstance(pattern, ast.LiteralPattern):
            return value == pattern.value
        elif isinstance(pattern, ast.EnumPattern):
            qualified = f"{pattern.enum_name}::{pattern.variant_name}"
            symbol = self.symbol_table.lookup(qualified)
            if symbol and symbol.const_value is not None:
                return value == symbol.const_value
            raise HIRError(f"Cannot resolve enum variant '{qualified}' in const match")
        elif isinstance(pattern, ast.OrPattern):
            return any(self._pattern_matches(sub, value) for sub in pattern.patterns)
        else:
            raise HIRError(f"Unsupported pattern in const match: {type(pattern).__name__}")

    # =========================================================================
    # Const fn evaluation via Python transpilation
    # =========================================================================

    def _eval_const_fn_call(self, func_def, arg_values, func_name):
        """Evaluate a const fn call by transpiling to Python and executing."""
        # Validate argument count
        expected = len(func_def.params)
        actual = len(arg_values)
        if actual != expected:
            raise HIRError(
                f"const fn '{func_name}' expects {expected} argument(s), got {actual}"
            )

        # Guard against deep recursion
        if self._const_fn_depth > 64:
            raise HIRError(
                f"const fn recursion depth exceeded (max 64) in '{func_name}'"
            )

        # Compile function if not cached
        if func_name not in self._compiled_const_fns:
            self._compiled_const_fns[func_name] = self._compile_const_fn(func_def, func_name)

        compiled_fn = self._compiled_const_fns[func_name]

        # Build execution namespace with helpers and other const fns
        namespace = self._build_const_fn_namespace()

        # Call the compiled function
        self._const_fn_depth += 1
        try:
            result = compiled_fn(*arg_values, **namespace)
            if result is None:
                raise HIRError(
                    f"const fn '{func_name}' did not return a value"
                )
            return result
        except HIRError:
            raise
        except ZeroDivisionError:
            raise HIRError(f"Division by zero in const fn '{func_name}'")
        except RecursionError:
            raise HIRError(f"Infinite recursion in const fn '{func_name}'")
        except RuntimeError as e:
            if "exceeded maximum iteration limit" in str(e):
                raise HIRError(
                    f"const fn '{func_name}' exceeded maximum iteration limit "
                    f"({ConstEvaluator.MAX_CONST_FN_ITERATIONS} iterations)"
                )
            raise HIRError(f"Error evaluating const fn '{func_name}': {e}")
        except Exception as e:
            raise HIRError(f"Error evaluating const fn '{func_name}': {e}")
        finally:
            self._const_fn_depth -= 1

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
        try:
            return self._compile_const_fn_inner(func_def, func_name)
        finally:
            self._compiling_const_fns.discard(func_name)

    def _compile_const_fn_inner(self, func_def, func_name):
        """Inner implementation of const fn compilation."""
        # Get parameter names
        param_names = [p.name for p in func_def.params]

        # Transpile body to Python source
        body_lines = self._transpile_block(func_def.body, indent=1)

        if not body_lines:
            body_lines = ["    pass"]

        # Build the Python function source
        # Use **_ns_ to receive namespace helpers without polluting positional args
        params_str = ", ".join(param_names)
        if params_str:
            fn_source = f"def _const_fn_({params_str}, **_ns_):\n"
        else:
            fn_source = f"def _const_fn_(**_ns_):\n"

        # Add namespace unpacking and loop counter at the top
        fn_source += "    _u8 = _ns_['_u8']; _u16 = _ns_['_u16']; _i8 = _ns_['_i8']; _i16 = _ns_['_i16']; _bool = _ns_['_bool']\n"
        fn_source += "    _idiv = _ns_['_idiv']; _imod = _ns_['_imod']\n"
        fn_source += "    _loop_count_ = 0\n"

        fn_source += "\n".join(body_lines) + "\n"

        # Compile and exec
        try:
            code = compile(fn_source, f"<const fn {func_name}>", "exec")
            local_ns = {}
            exec(code, {}, local_ns)
            return local_ns['_const_fn_']
        except SyntaxError as e:
            raise HIRError(
                f"Failed to compile const fn '{func_name}': {e}"
            )

    def _build_const_fn_namespace(self):
        """Build the namespace dict passed to compiled const fns."""
        def _u8(v):
            return int(v) & 0xFF

        def _u16(v):
            return int(v) & 0xFFFF

        def _i8(v):
            val = int(v) & 0xFF
            return val if val < 128 else val - 256

        def _i16(v):
            val = int(v) & 0xFFFF
            return val if val < 32768 else val - 65536

        def _bool(v):
            if isinstance(v, bool):
                return v
            return int(v) != 0

        def _idiv(a, b):
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return int(a) // int(b)

        def _imod(a, b):
            if b == 0:
                raise ZeroDivisionError("modulo by zero")
            return int(a) % int(b)

        ns = {
            '_u8': _u8, '_u16': _u16, '_i8': _i8, '_i16': _i16, '_bool': _bool,
            '_idiv': _idiv, '_imod': _imod,
        }

        # Add all compiled const fns to namespace so they can call each other
        for name, fn in self._compiled_const_fns.items():
            ns[name] = fn

        return ns

    MAX_CONST_FN_ITERATIONS = 10_000

    def _transpile_block(self, block, indent=1):
        """Transpile an AST Block to Python source lines."""
        lines = []
        for stmt in block.statements:
            lines.extend(self._transpile_stmt(stmt, indent))
        return lines

    @staticmethod
    def _loop_guard_lines(prefix):
        """Return Python lines that increment and check the shared loop counter."""
        return [
            f"{prefix}_loop_count_ += 1",
            f"{prefix}if _loop_count_ > {ConstEvaluator.MAX_CONST_FN_ITERATIONS}:"
            f" raise RuntimeError('exceeded maximum iteration limit')",
        ]

    def _transpile_stmt(self, stmt, indent):
        """Transpile a single AST statement to Python source lines."""
        prefix = "    " * indent

        if isinstance(stmt, ast.LetStmt):
            if stmt.initializer is not None:
                val = self._transpile_expr(stmt.initializer)
                return [f"{prefix}{stmt.name} = {val}"]
            else:
                return [f"{prefix}{stmt.name} = 0"]

        elif isinstance(stmt, ast.ReturnStmt):
            if stmt.values:
                val = self._transpile_expr(stmt.values[0])
                return [f"{prefix}return {val}"]
            else:
                return [f"{prefix}return"]

        elif isinstance(stmt, ast.IfStmt):
            lines = []
            cond = self._transpile_expr(stmt.condition)
            lines.append(f"{prefix}if {cond}:")
            body_lines = self._transpile_block(stmt.then_block, indent + 1)
            if not body_lines:
                body_lines = [f"{'    ' * (indent + 1)}pass"]
            lines.extend(body_lines)
            if stmt.else_block:
                if isinstance(stmt.else_block, ast.IfStmt):
                    # else if chain
                    else_lines = self._transpile_stmt(stmt.else_block, indent)
                    # Change 'if' to 'elif' for the first line
                    if else_lines:
                        else_lines[0] = else_lines[0].replace(f"{prefix}if ", f"{prefix}elif ", 1)
                    lines.extend(else_lines)
                elif isinstance(stmt.else_block, ast.Block):
                    lines.append(f"{prefix}else:")
                    else_body = self._transpile_block(stmt.else_block, indent + 1)
                    if not else_body:
                        else_body = [f"{'    ' * (indent + 1)}pass"]
                    lines.extend(else_body)
            return lines

        elif isinstance(stmt, ast.WhileStmt):
            lines = []
            cond = self._transpile_expr(stmt.condition)
            lines.append(f"{prefix}while {cond}:")
            inner = "    " * (indent + 1)
            guard = self._loop_guard_lines(inner)
            body_lines = self._transpile_block(stmt.body, indent + 1)
            if not body_lines:
                body_lines = [f"{inner}pass"]
            lines.extend(guard)
            lines.extend(body_lines)
            return lines

        elif isinstance(stmt, ast.LoopStmt):
            lines = []
            lines.append(f"{prefix}while True:")
            inner = "    " * (indent + 1)
            guard = self._loop_guard_lines(inner)
            body_lines = self._transpile_block(stmt.body, indent + 1)
            if not body_lines:
                body_lines = [f"{inner}pass"]
            lines.extend(guard)
            lines.extend(body_lines)
            return lines

        elif isinstance(stmt, ast.ForStmt):
            lines = []
            start = self._transpile_expr(stmt.start)
            end = self._transpile_expr(stmt.end)
            lines.append(f"{prefix}for {stmt.variable} in range({start}, {end}):")
            inner = "    " * (indent + 1)
            guard = self._loop_guard_lines(inner)
            body_lines = self._transpile_block(stmt.body, indent + 1)
            if not body_lines:
                body_lines = [f"{inner}pass"]
            lines.extend(guard)
            lines.extend(body_lines)
            return lines

        elif isinstance(stmt, ast.BreakStmt):
            return [f"{prefix}break"]

        elif isinstance(stmt, ast.ContinueStmt):
            return [f"{prefix}continue"]

        elif isinstance(stmt, ast.ExprStmt):
            expr_str = self._transpile_expr(stmt.expr)
            return [f"{prefix}{expr_str}"]

        elif isinstance(stmt, ast.Block):
            return self._transpile_block(stmt, indent)

        elif isinstance(stmt, ast.AsmStmt):
            raise HIRError("Inline assembly (asm!) cannot be used in const fn")

        elif isinstance(stmt, ast.ConstAssertStmt):
            raise HIRError("const_assert! is not supported in const fn")

        else:
            raise HIRError(f"Unsupported statement in const fn: {type(stmt).__name__}")

    def _transpile_expr(self, expr):
        """Transpile an AST expression to a Python expression string."""
        if isinstance(expr, ast.IntegerLiteral):
            return str(expr.value)

        elif isinstance(expr, ast.BooleanLiteral):
            return "True" if expr.value else "False"

        elif isinstance(expr, ast.Identifier):
            # Check if it's a const or enum variant
            from r65.compiler.hir.symbol_table import SymbolKind
            symbol = self.symbol_table.lookup(expr.name)
            if symbol:
                if symbol.kind == SymbolKind.CONST and symbol.const_value is not None:
                    return repr(symbol.const_value)
                elif symbol.kind == SymbolKind.IMPL_CONST and symbol.const_value is not None:
                    return repr(symbol.const_value)
                elif symbol.kind == SymbolKind.REGISTER:
                    raise HIRError(
                        f"Cannot access hardware register '{expr.name}' in const fn"
                    )
                elif symbol.kind == SymbolKind.STATIC_VAR:
                    raise HIRError(
                        f"Cannot access runtime variable '{expr.name}' in const fn"
                    )
            # Otherwise it's a local variable or parameter reference
            return expr.name

        elif isinstance(expr, ast.EnumVariantExpr):
            qualified = f"{expr.enum_name}::{expr.variant_name}"
            symbol = self.symbol_table.lookup(qualified)
            if symbol and symbol.const_value is not None:
                return str(symbol.const_value)
            raise HIRError(f"Cannot resolve enum variant '{qualified}' in const fn")

        elif isinstance(expr, ast.BinaryOp):
            left = self._transpile_expr(expr.left)
            right = self._transpile_expr(expr.right)
            op = expr.op
            if op == '&&':
                return f"(bool({left}) and bool({right}))"
            elif op == '||':
                return f"(bool({left}) or bool({right}))"
            elif op == '/':
                return f"_ns_['_idiv']({left}, {right})"
            elif op == '%':
                return f"_ns_['_imod']({left}, {right})"
            else:
                return f"({left} {op} {right})"

        elif isinstance(expr, ast.UnaryOp):
            operand = self._transpile_expr(expr.operand)
            if expr.op == '!':
                return f"(not {operand})"
            elif expr.op == '~':
                return f"(~{operand})"
            elif expr.op == '-':
                return f"(-{operand})"
            else:
                raise HIRError(f"Unsupported unary op in const fn: {expr.op}")

        elif isinstance(expr, ast.TypeCast):
            inner = self._transpile_expr(expr.expr)
            if isinstance(expr.target_type, ast.BasicType):
                type_name = expr.target_type.name
                if type_name in ('u8', 'u16', 'i8', 'i16'):
                    return f"_ns_['_{type_name}']({inner})"
                elif type_name == 'bool':
                    return f"_ns_['_bool']({inner})"
            raise HIRError(f"Unsupported cast in const fn: {expr.target_type}")

        elif isinstance(expr, ast.FunctionCall):
            if isinstance(expr.func, ast.Identifier):
                func_name = expr.func.name
                # Check for builtins that can be resolved at transpile time
                if func_name == 'size_of' and BuiltinRegistry.is_builtin(func_name):
                    try:
                        val = self._eval_size_of(expr)
                        return str(val)
                    except HIRError:
                        pass
                elif func_name == 'offset_of' and BuiltinRegistry.is_builtin(func_name):
                    try:
                        val = self._eval_offset_of(expr)
                        return str(val)
                    except HIRError:
                        pass

                # Check if it's a const fn call
                from r65.compiler.hir.symbol_table import SymbolKind
                symbol = self.symbol_table.lookup(func_name)
                if symbol and symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                    func_def = symbol.definition
                    if func_def and hasattr(func_def, 'is_const') and func_def.is_const:
                        # Compile it if needed (skip if currently being compiled — recursive)
                        if func_name not in self._compiled_const_fns and func_name not in self._compiling_const_fns:
                            self._compiled_const_fns[func_name] = self._compile_const_fn(func_def, func_name)
                        args_str = ", ".join(self._transpile_expr(a) for a in expr.args)
                        return f"_ns_['{func_name}']({args_str}, **_ns_)"
                    else:
                        raise HIRError(
                            f"Cannot call '{func_name}' from const fn: "
                            f"'{func_name}' is not a const fn"
                        )

            raise HIRError(f"Unsupported function call in const fn: {expr}")

        elif isinstance(expr, ast.Assignment):
            target = self._transpile_expr(expr.target)
            value = self._transpile_expr(expr.value)
            return f"{target} = {value}"

        elif isinstance(expr, ast.CompoundAssignment):
            target = self._transpile_expr(expr.target)
            value = self._transpile_expr(expr.value)
            op = expr.operator
            if op == '/':
                return f"{target} = _ns_['_idiv']({target}, {value})"
            elif op == '%':
                return f"{target} = _ns_['_imod']({target}, {value})"
            return f"{target} {op}= {value}"

        elif isinstance(expr, ast.BlockExpression):
            return self._transpile_block_expr(expr)

        elif isinstance(expr, ast.IfExpression):
            return self._transpile_if_expr(expr)

        elif isinstance(expr, ast.MatchExpression):
            return self._transpile_match_expr(expr)

        elif isinstance(expr, ast.Register):
            raise HIRError(
                f"Cannot access hardware register '{expr.name}' in const fn"
            )

        elif isinstance(expr, ast.StringLiteral):
            raise HIRError("String literals are not supported in const fn")

        elif isinstance(expr, ast.ArrayIndex):
            array = self._transpile_expr(expr.array)
            index = self._transpile_expr(expr.index)
            return f"{array}[{index}]"

        elif isinstance(expr, ast.FieldAccess):
            raise HIRError("Struct field access is not supported in const fn")

        elif isinstance(expr, ast.Dereference):
            raise HIRError("Pointer dereference is not supported in const fn")

        elif isinstance(expr, ast.AddressOf):
            raise HIRError("Address-of operator is not supported in const fn")

        elif isinstance(expr, ast.ArrayLiteralExpr):
            elements = ", ".join(self._transpile_expr(e) for e in expr.elements)
            return f"[{elements}]"

        elif isinstance(expr, ast.ArrayFillExpr):
            value = self._transpile_expr(expr.value)
            count = self._transpile_expr(expr.count)
            return f"[{value}] * {count}"

        elif isinstance(expr, ast.StructLiteralExpr):
            raise HIRError("Struct literals are not supported in const fn")

        elif isinstance(expr, ast.MultiAssignment):
            raise HIRError("Multiple assignment is not supported in const fn")

        elif isinstance(expr, ast.IncludeBytesExpr):
            raise HIRError("include_bytes! is not supported in const fn")

        else:
            raise HIRError(f"Unsupported expression in const fn: {type(expr).__name__}")

    def _transpile_block_expr(self, expr):
        """Transpile a block expression to a Python expression.

        If no statements, just transpile the final expression.
        Block expressions with statements cannot be transpiled to a single
        Python expression, so they raise an error in const fn context.
        """
        if not expr.statements:
            return self._transpile_expr(expr.final_expr)

        raise HIRError("Block expressions with statements are not supported in const fn")

    def _transpile_if_expr(self, expr):
        """Transpile an if expression to a Python ternary expression."""
        cond = self._transpile_expr(expr.condition)
        then_val = self._transpile_expr(expr.then_block)
        else_val = self._transpile_expr(expr.else_block)
        return f"(({then_val}) if ({cond}) else ({else_val}))"

    def _transpile_match_expr(self, expr):
        """Transpile a match expression to a Python lambda with chained ternaries.

        match x { 0 => a, 1 | 2 => b, name => c, _ => d }
        becomes:
        (lambda _m_: (a) if _m_ == 0 else (b) if _m_ == 1 or _m_ == 2
         else (lambda name: (c))(_m_) else (d))(scrutinee)
        """
        scrutinee = self._transpile_expr(expr.scrutinee)

        if not expr.arms:
            raise HIRError("Empty match expression in const fn")

        # Build list of (condition_or_None, body_str) for each arm
        parts = []
        for arm in expr.arms:
            body_str = self._transpile_expr(arm.body)
            pattern = arm.pattern

            if isinstance(pattern, ast.WildcardPattern):
                parts.append((None, body_str))
            elif isinstance(pattern, ast.IdentifierPattern):
                # Bind scrutinee to variable name via nested lambda
                parts.append((None, f"(lambda {pattern.name}: {body_str})(_m_)"))
            elif isinstance(pattern, ast.LiteralPattern):
                parts.append((f"_m_ == {repr(pattern.value)}", body_str))
            elif isinstance(pattern, ast.EnumPattern):
                cond = self._transpile_enum_pattern(pattern)
                parts.append((cond, body_str))
            elif isinstance(pattern, ast.OrPattern):
                or_conds = []
                for sub in pattern.patterns:
                    if isinstance(sub, ast.LiteralPattern):
                        or_conds.append(f"_m_ == {repr(sub.value)}")
                    elif isinstance(sub, ast.EnumPattern):
                        or_conds.append(self._transpile_enum_pattern(sub))
                    else:
                        raise HIRError(
                            f"Unsupported sub-pattern in or-pattern in const fn: "
                            f"{type(sub).__name__}"
                        )
                parts.append((" or ".join(or_conds), body_str))
            else:
                raise HIRError(
                    f"Unsupported pattern in const fn match: {type(pattern).__name__}"
                )

        # Build ternary chain from right to left
        result = None
        for condition, body in reversed(parts):
            if result is None:
                # Last arm
                if condition is None:
                    result = body
                else:
                    result = f"({body} if {condition} else None)"
            else:
                if condition is None:
                    # Unconditional catch-all before end — takes priority
                    result = body
                else:
                    result = f"({body} if {condition} else {result})"

        return f"(lambda _m_: {result})({scrutinee})"

    def _transpile_enum_pattern(self, pattern):
        """Transpile an enum pattern to a Python comparison string."""
        qualified = f"{pattern.enum_name}::{pattern.variant_name}"
        symbol = self.symbol_table.lookup(qualified)
        if symbol and symbol.const_value is not None:
            return f"_m_ == {symbol.const_value}"
        raise HIRError(
            f"Cannot resolve enum variant '{qualified}' in const fn match"
        )
