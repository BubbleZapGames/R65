"""
Function call validation for R65 type checker.

Handles validation of function calls, method calls, built-in calls,
and function address expressions.
"""

from typing import Callable, Optional
from r65.compiler.hir import (
    HIRFunctionCall, HIRMethodCall, HIRFunctionAddress, HIRFunctionDecl,
    HIRIdentifier, HIRArrayIndex, HIRIntegerLiteral,
    SymbolKind, BasicTypeInfo, StructTypeInfo, NeverTypeInfo
)
from r65.compiler.hir.types import TypeInfo, FunctionTypeInfo, ArrayTypeInfo
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.typeck.type_utils import TypeUtils
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeTransition


class CallValidator:
    """Validates function and method calls."""

    def __init__(self, symbol_table, lookup_function_decl_fn: Callable,
                 check_expression_fn: Callable, get_current_mode_fn: Callable,
                 get_current_function_fn: Callable):
        """
        Initialize with dependencies.

        Args:
            symbol_table: Symbol table for lookups
            lookup_function_decl_fn: Function to look up function declarations
            check_expression_fn: Function to type check expressions
            get_current_mode_fn: Function to get current processor mode
            get_current_function_fn: Function to get current function being checked
        """
        self.symbol_table = symbol_table
        self.lookup_function_decl = lookup_function_decl_fn
        self.check_expression = check_expression_fn
        self.get_current_mode = get_current_mode_fn
        self.get_current_function = get_current_function_fn

    def check_function_call(self, expr: HIRFunctionCall) -> TypeInfo:
        """
        Type check function call.

        Supports both:
        - Direct calls: expr.func is HIRIdentifier pointing to function
        - Indirect calls: expr.func is expression with function pointer type
        - Built-in calls: expr.builtin_name is set
        """
        # Check if this is a built-in function call
        if expr.builtin_name:
            return self._check_builtin_call(expr)

        # Handle direct call vs indirect call
        if isinstance(expr.func, HIRIdentifier) and expr.func.symbol.kind == SymbolKind.FUNCTION:
            # Direct call to a function
            func_symbol = expr.func.symbol
            func_decl = self.lookup_function_decl(func_symbol.name, expr.source_loc)

            # Check argument count
            if len(expr.args) != len(func_decl.parameters):
                param_names = [p.name for p in func_decl.parameters]
                raise TypeCheckError(
                    f"function '{func_symbol.name}' expects {len(func_decl.parameters)} argument(s), got {len(expr.args)}",
                    source_loc=expr.source_loc,
                    hint=f"parameters: {', '.join(param_names)}" if param_names else None
                )

            # Type check each argument
            for i, (arg, param) in enumerate(zip(expr.args, func_decl.parameters)):
                arg_type = self.check_expression(arg)
                if not isinstance(arg_type, NeverTypeInfo):
                    if not TypeUtils.types_compatible(arg_type, param.param_type):
                        raise TypeCheckError(
                            f"argument {i + 1} to '{func_symbol.name}' has wrong type: expected {param.param_type}, found {arg_type}",
                            source_loc=arg.source_loc if hasattr(arg, 'source_loc') else expr.source_loc,
                            hint=f"parameter '{param.name}' expects type {param.param_type}"
                        )

            # Check mode compatibility
            self._check_call_mode_compatibility(func_symbol.name, func_decl, expr.source_loc)

            # Check bank compatibility (near functions can't call near functions in different banks)
            self._check_call_bank_compatibility(func_symbol.name, func_decl, expr.source_loc)

            # Set return type
            if func_decl.return_type:
                expr.expr_type = func_decl.return_type
            else:
                expr.expr_type = BasicTypeInfo('void')

        else:
            # Indirect call (function pointer)
            func_type = self.check_expression(expr.func)

            if not isinstance(func_type, FunctionTypeInfo):
                raise TypeCheckError(
                    f"cannot call expression of type {func_type}",
                    source_loc=expr.source_loc,
                    hint="only functions and function pointers can be called"
                )

            # Check argument count
            if len(expr.args) != len(func_type.param_types):
                raise TypeCheckError(
                    f"function pointer expects {len(func_type.param_types)} argument(s), got {len(expr.args)}",
                    source_loc=expr.source_loc
                )

            # Type check each argument
            for i, (arg, param_type) in enumerate(zip(expr.args, func_type.param_types)):
                arg_type = self.check_expression(arg)
                if not isinstance(arg_type, NeverTypeInfo):
                    if not TypeUtils.types_compatible(arg_type, param_type):
                        raise TypeCheckError(
                            f"argument {i + 1} has wrong type: expected {param_type}, found {arg_type}",
                            source_loc=arg.source_loc if hasattr(arg, 'source_loc') else expr.source_loc,
                            hint=f"use cast if needed: (value as {param_type})"
                        )

            # Set return type
            if func_type.return_type:
                expr.expr_type = func_type.return_type
            else:
                expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def _check_builtin_call(self, expr: HIRFunctionCall) -> TypeInfo:
        """Type check built-in function call."""
        from r65.compiler.builtins import BuiltinRegistry

        builtin = BuiltinRegistry.get_builtin(expr.builtin_name)
        if not builtin:
            raise TypeCheckError(
                f"Unknown built-in function: {expr.builtin_name}",
                source_loc=expr.source_loc
            )

        # Type check arguments
        for arg in expr.args:
            # For size_of, don't type-check type identifiers
            if builtin.kind.value == "type_info" and expr.builtin_name == "size_of":
                if isinstance(arg, (HIRIdentifier, HIRArrayIndex)):
                    continue
            self.check_expression(arg)

        # Handle const built-ins (like size_of)
        if builtin.kind.value == "type_info":
            if expr.builtin_name == "size_of":
                return self._check_size_of_builtin(expr)
            else:
                raise TypeCheckError(
                    f"Unknown type info built-in: {expr.builtin_name}",
                    source_loc=expr.source_loc
                )

        # Set return type
        if builtin.returns_value:
            if expr.args:
                first_arg_type = expr.args[0].expr_type
                if isinstance(first_arg_type, BasicTypeInfo) and first_arg_type.name in ('u8', 'i8', 'u16', 'i16'):
                    expr.expr_type = first_arg_type
                else:
                    expr.expr_type = BasicTypeInfo('u8')
            else:
                expr.expr_type = BasicTypeInfo('u8')
        else:
            expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def check_method_call(self, expr: HIRMethodCall) -> TypeInfo:
        """Type check method call (e.g., value.rotate_left(3) or array.len())."""
        # Type check receiver
        receiver_type = self.check_expression(expr.receiver)

        # Handle len() method on arrays
        if expr.method_name == 'len':
            # Validate receiver is an array type
            if not isinstance(receiver_type, ArrayTypeInfo):
                raise TypeCheckError(
                    f"method 'len' requires array type, found {receiver_type}",
                    source_loc=expr.source_loc,
                    hint="len() only works on arrays"
                )

            # Validate no arguments
            if len(expr.args) != 0:
                raise TypeCheckError(
                    f"len() takes no arguments, got {len(expr.args)}",
                    source_loc=expr.source_loc,
                    hint="example: array.len()"
                )

            # Return type is u16 for arrays (can hold sizes up to 65535)
            expr.expr_type = BasicTypeInfo('u16')
            return expr.expr_type

        # Handle rotate methods on integers
        # Validate receiver is an integer type
        if not isinstance(receiver_type, BasicTypeInfo) or receiver_type.name not in ['u8', 'i8', 'u16', 'i16']:
            raise TypeCheckError(
                f"method '{expr.method_name}' requires integer type, found {receiver_type}",
                source_loc=expr.source_loc,
                hint="rotate methods only work on u8, i8, u16, or i16"
            )

        # Validate method name
        if expr.method_name not in ['rotate_left', 'rotate_right']:
            raise TypeCheckError(
                f"unknown method '{expr.method_name}' for type {receiver_type}",
                source_loc=expr.source_loc,
                hint="available methods: rotate_left, rotate_right"
            )

        # Type check argument (rotation count)
        if len(expr.args) != 1:
            raise TypeCheckError(
                f"{expr.method_name}() takes exactly 1 argument, got {len(expr.args)}",
                source_loc=expr.source_loc,
                hint="example: value.rotate_left(1)"
            )

        count_arg = expr.args[0]
        self.check_expression(count_arg)

        # Validate count is an integer literal
        if not isinstance(count_arg, HIRIntegerLiteral):
            raise TypeCheckError(
                f"{expr.method_name}() count must be a constant",
                source_loc=count_arg.source_loc,
                hint="rotation count must be a compile-time constant (1-8)"
            )

        # Validate count is in range 1-8
        count_value = count_arg.value
        if not (1 <= count_value <= 8):
            raise TypeCheckError(
                f"{expr.method_name}() count must be between 1 and 8, got {count_value}",
                source_loc=count_arg.source_loc,
                hint="valid rotation counts: 1, 2, 3, 4, 5, 6, 7, 8"
            )

        expr.expr_type = receiver_type
        return expr.expr_type

    def check_function_address(self, expr: HIRFunctionAddress) -> TypeInfo:
        """Type check function address expression."""
        func_symbol = expr.symbol
        if not func_symbol:
            raise TypeCheckError(
                f"Function '{expr.function_name}' not resolved",
                source_loc=expr.source_loc
            )

        func_decl = self.lookup_function_decl(func_symbol.name, expr.source_loc)

        param_types = [param.param_type for param in func_decl.parameters]
        func_type = FunctionTypeInfo(
            is_far=func_decl.is_far,
            param_types=param_types,
            return_type=func_decl.return_type
        )

        expr.expr_type = func_type
        return func_type

    def _check_call_mode_compatibility(self, func_name: str, func_decl: HIRFunctionDecl, source_loc):
        """Check mode compatibility between caller and callee."""
        # Get callee mode
        if func_decl.mode_attr:
            callee_mode = ProcessorMode.from_attribute(func_decl.mode_attr)
        else:
            callee_mode = ProcessorMode.unknown()

        caller_mode = self.get_current_mode()

        if caller_mode == callee_mode:
            return

        if not caller_mode.is_fully_known() or not callee_mode.is_fully_known():
            return

        mode_attr = func_decl.mode_attr
        if mode_attr and hasattr(mode_attr, 'transition'):
            transition = mode_attr.transition
        else:
            transition = ModeTransition.NONE

        if transition == ModeTransition.INLINE:
            if func_decl.preserves_attr and 'STATUS' in func_decl.preserves_attr.registers:
                raise TypeCheckError(
                    f"function '{func_name}' cannot use transition=inline with #[preserves(STATUS)]",
                    source_loc=source_loc,
                    hint="transition=inline modifies STATUS to switch modes, which conflicts with preservation"
                )

        if transition == ModeTransition.NONE:
            raise TypeCheckError(
                f"cannot call '{func_name}': mode mismatch (caller: {caller_mode}, callee: {callee_mode})",
                source_loc=source_loc,
                hint="add transition=inline or transition=caller to the callee's #[mode(...)] attribute"
            )

    def _check_call_bank_compatibility(self, func_name: str, func_decl: HIRFunctionDecl, source_loc):
        """
        Check bank compatibility between caller and callee.

        A non-far function cannot call another non-far function in a different bank,
        because JSR only uses a 16-bit address and cannot cross bank boundaries.
        """
        # If callee is a far function, any caller can call it (JSL handles cross-bank)
        if func_decl.is_far:
            return

        # Get caller function
        caller_func = self.get_current_function()
        if caller_func is None:
            return  # No caller context (shouldn't happen in normal flow)

        # Get bank numbers (default to 0 if not specified)
        caller_bank = 0
        if caller_func.bank_attr:
            caller_bank = caller_func.bank_attr.bank_number

        callee_bank = 0
        if func_decl.bank_attr:
            callee_bank = func_decl.bank_attr.bank_number

        # If banks differ and callee is not far, this is an error
        if caller_bank != callee_bank:
            raise TypeCheckError(
                f"cannot call near function '{func_name}' from bank {caller_bank}: "
                f"'function {func_name}' is in bank {callee_bank}",
                source_loc=source_loc,
                hint=f"near functions use JSR which cannot cross bank boundaries; "
                     f"declare '{func_name}' as 'far fn' to allow cross-bank calls"
            )

    def _check_size_of_builtin(self, expr: HIRFunctionCall) -> TypeInfo:
        """Type check size_of built-in function call."""
        if len(expr.args) != 1:
            raise TypeCheckError(
                "size_of expects exactly 1 argument",
                source_loc=expr.source_loc
            )

        arg = expr.args[0]

        if isinstance(arg, HIRIdentifier):
            type_name = arg.name

            # Basic types
            if type_name in ['u8', 'i8', 'bool']:
                expr.evaluated_size = 1
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type
            elif type_name in ['u16', 'i16']:
                expr.evaluated_size = 2
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type

            # Look up struct or enum
            symbol = self.symbol_table.lookup(type_name)
            if symbol is None:
                raise TypeCheckError(
                    f"Unknown type: {type_name}",
                    source_loc=expr.source_loc
                )

            if symbol.kind.value == "struct":
                struct_type = StructTypeInfo(name=type_name, definition=symbol.definition)
                try:
                    from r65.compiler.hir.unified_type_utils import get_unified_type_size
                    size = get_unified_type_size(struct_type, self.symbol_table)
                    expr.evaluated_size = size
                    expr.expr_type = BasicTypeInfo('u8') if size <= 255 else BasicTypeInfo('u16')
                    return expr.expr_type
                except Exception:
                    raise TypeCheckError(
                        f"Cannot determine size of struct '{type_name}'",
                        source_loc=expr.source_loc
                    )

            elif symbol.kind.value == "enum":
                expr.evaluated_size = 1
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type

            else:
                raise TypeCheckError(
                    f"'{type_name}' is not a valid type for size_of",
                    source_loc=arg.source_loc
                )

        elif isinstance(arg, HIRArrayIndex):
            element_type = arg.expr_type
            if not isinstance(element_type, ArrayTypeInfo):
                raise TypeCheckError(
                    "size_of expects a type identifier or array type",
                    source_loc=arg.source_loc
                )

            try:
                from r65.compiler.hir.unified_type_utils import get_unified_type_size
                element_size = get_unified_type_size(element_type.element_type, self.symbol_table)
                array_size = element_size * element_type.size
                expr.evaluated_size = array_size
                expr.expr_type = BasicTypeInfo('u8') if array_size <= 255 else BasicTypeInfo('u16')
                return expr.expr_type
            except Exception:
                raise TypeCheckError(
                    "Cannot determine size of array type",
                    source_loc=arg.source_loc
                )

        else:
            raise TypeCheckError(
                "size_of expects a type identifier (like u8, Player, etc.), not a value expression",
                source_loc=arg.source_loc
            )
