"""
Function call validation for R65 type checker.

Handles validation of function calls, method calls, built-in calls,
and function address expressions.
"""

from typing import Callable, Optional
from r65.compiler.hir import (
    HIRFunctionCall, HIRMethodCall, HIRFunctionAddress, HIRFunctionDecl,
    HIRIdentifier, HIRArrayIndex, HIRIntegerLiteral, HIRFieldAccess,
    HIRAddressOf,
    SymbolKind, BasicTypeInfo, StructTypeInfo, NeverTypeInfo,
    HIRError,
)
from r65.compiler.hir.types import TypeInfo, FunctionTypeInfo, ArrayTypeInfo, PointerTypeInfo, TraitTypeInfo
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.typeck.type_utils import TypeUtils
from r65.compiler.typeck.processor_mode import ProcessorMode


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

    def _validate_arguments(self, args, params, func_name: str, source_loc, is_indirect: bool = False):
        """
        Validate argument types against parameter types.

        Args:
            args: List of argument expressions
            params: List of parameters (HIRParameter objects) or param types (TypeInfo for indirect)
            func_name: Function name for error messages
            source_loc: Source location for error reporting
            is_indirect: True if params are raw TypeInfo (function pointer call)
        """
        for i, (arg, param) in enumerate(zip(args, params)):
            arg_type = self.check_expression(arg)
            if isinstance(arg_type, NeverTypeInfo):
                continue

            # Get param type - handle both HIRParameter objects and raw TypeInfo
            if is_indirect:
                param_type = param
                param_name = None
            else:
                param_type = param.param_type
                param_name = param.name

            if not TypeUtils.types_compatible(arg_type, param_type):
                hint = (f"parameter '{param_name}' expects type {param_type}"
                        if param_name else f"use cast if needed: (value as {param_type})")
                raise TypeCheckError(
                    f"argument {i + 1} to '{func_name}' has wrong type: expected {param_type}, found {arg_type}",
                    source_loc=arg.source_loc if hasattr(arg, 'source_loc') else source_loc,
                    hint=hint
                )

    def check_function_call(self, expr: HIRFunctionCall) -> TypeInfo:
        """
        Type check function call.

        Supports both:
        - Direct calls: expr.func is HIRIdentifier pointing to function
        - Indirect calls: expr.func is expression with function pointer type
        - Built-in calls: expr.builtin_name is set
        - Method calls: expr.func is HIRFieldAccess on struct type
        """
        # Check if this is a built-in function call
        if expr.builtin_name:
            return self._check_builtin_call(expr)

        # Check if this is a method call (receiver.method(args))
        if isinstance(expr.func, HIRFieldAccess):
            method_result = self._try_method_call(expr)
            if method_result is not None:
                return method_result

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
            self._validate_arguments(expr.args, func_decl.parameters, func_symbol.name, expr.source_loc)

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
            self._validate_arguments(expr.args, func_type.param_types, "function pointer", expr.source_loc, is_indirect=True)

            # Set return type
            if func_type.return_type:
                expr.expr_type = func_type.return_type
            else:
                expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def _try_method_call(self, expr: HIRFunctionCall) -> Optional[TypeInfo]:
        """
        Try to handle expr as a method call (receiver.method(args)).

        Returns the result type if this is a valid method call,
        or None if it's not a method call (should fall through to field access).
        """
        field_access = expr.func
        method_name = field_access.field_name

        # Get the type of the receiver (e.g., PLAYER has type Player)
        # We need to check the type without throwing an error for missing field
        try:
            receiver_type = self.check_expression(field_access.base)
        except TypeCheckError:
            return None

        # Determine struct name from receiver type
        struct_name = None
        receiver_is_pointer = False
        pointer_is_far = False

        if isinstance(receiver_type, StructTypeInfo):
            struct_name = receiver_type.name
        elif isinstance(receiver_type, PointerTypeInfo):
            if isinstance(receiver_type.pointee_type, StructTypeInfo):
                struct_name = receiver_type.pointee_type.name
                receiver_is_pointer = True
                pointer_is_far = receiver_type.is_far
            elif isinstance(receiver_type.pointee_type, TraitTypeInfo):
                # Trait pointer dispatch: *Drawable.method(args)
                return self._try_trait_method_call(expr, receiver_type, method_name)

        if not struct_name:
            return None

        # Look up method symbol using StructName.method_name key
        method_key = f"{struct_name}.{method_name}"
        method_symbol = self.symbol_table.lookup(method_key)

        if not method_symbol or method_symbol.kind != SymbolKind.METHOD:
            return None

        # Found a method! Get the mangled function name
        method_info = method_symbol.type_info
        if not method_info or not isinstance(method_info, dict):
            return None

        mangled_name = method_info.get('mangled_name')
        impl_is_far = method_info.get('impl_is_far', False)
        method_self_is_far = method_info.get('method_self_is_far', False)

        # Look up the actual function declaration
        func_decl = self.lookup_function_decl(mangled_name, expr.source_loc)
        if not func_decl:
            raise TypeCheckError(
                f"method '{method_name}' not found for struct '{struct_name}'",
                source_loc=expr.source_loc
            )

        # Check self pointer compatibility
        # impl far StructName -> expects far *self
        # impl StructName -> expects near *self
        expected_far_self = impl_is_far or method_self_is_far

        # If receiver is a pointer, check far/near compatibility
        if receiver_is_pointer:
            if expected_far_self and not pointer_is_far:
                raise TypeCheckError(
                    f"method '{method_name}' expects far pointer but got near pointer",
                    source_loc=expr.source_loc,
                    hint=f"method is defined in 'impl far {struct_name}'"
                )
            elif not expected_far_self and pointer_is_far:
                raise TypeCheckError(
                    f"method '{method_name}' expects near pointer but got far pointer",
                    source_loc=expr.source_loc,
                    hint=f"use 'impl far {struct_name}' for far pointer methods"
                )

        # Transform: receiver.method(args) -> mangled_name(&receiver, args)
        # Create address-of expression for the receiver (unless already a pointer)
        if receiver_is_pointer:
            self_arg = field_access.base
        else:
            self_arg = HIRAddressOf(
                operand=field_access.base,
                source_loc=field_access.base.source_loc if hasattr(field_access.base, 'source_loc') else None
            )
            # Set the type of the address-of expression
            self_arg.expr_type = PointerTypeInfo(
                pointee_type=receiver_type,
                is_far=expected_far_self
            )

        # Check argument count (method has self as first param)
        expected_args = len(func_decl.parameters) - 1  # Exclude self
        if len(expr.args) != expected_args:
            param_names = [p.name for p in func_decl.parameters[1:]]
            raise TypeCheckError(
                f"method '{method_name}' expects {expected_args} argument(s), got {len(expr.args)}",
                source_loc=expr.source_loc,
                hint=f"parameters: {', '.join(param_names)}" if param_names else None
            )

        # Type check self argument
        self_param = func_decl.parameters[0]
        if not TypeUtils.types_compatible(self_arg.expr_type, self_param.param_type):
            raise TypeCheckError(
                f"self argument has wrong type: expected {self_param.param_type}, found {self_arg.expr_type}",
                source_loc=expr.source_loc
            )

        # Type check remaining arguments
        self._validate_arguments(expr.args, func_decl.parameters[1:], method_name, expr.source_loc)

        # Check mode compatibility
        self._check_call_mode_compatibility(mangled_name, func_decl, expr.source_loc)

        # Check bank compatibility
        self._check_call_bank_compatibility(mangled_name, func_decl, expr.source_loc)

        # Transform the HIR node for code generation
        # Store the mangled name and self argument in the expression
        expr.method_call_info = {
            'mangled_name': mangled_name,
            'self_arg': self_arg,
            'func_decl': func_decl
        }

        # Set return type
        if func_decl.return_type:
            expr.expr_type = func_decl.return_type
        else:
            expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def _try_trait_method_call(self, expr: HIRFunctionCall, receiver_type: PointerTypeInfo, method_name: str) -> Optional[TypeInfo]:
        """Handle method call on trait pointer (dynamic dispatch)."""
        trait_type = receiver_type.pointee_type  # TraitTypeInfo
        trait_name = trait_type.name

        # Built-in type_id() method on trait pointers
        if method_name == 'type_id':
            if len(expr.args) != 0:
                raise TypeCheckError(
                    f"type_id() takes no arguments, got {len(expr.args)}",
                    source_loc=expr.source_loc
                )
            expr.expr_type = BasicTypeInfo('u8')
            field_access = expr.func
            expr.method_call_info = {
                'is_type_id': True,
                'self_arg': field_access.base,
            }
            return expr.expr_type

        # Look up trait definition from symbol table
        trait_symbol = self.symbol_table.lookup(trait_name)
        if not trait_symbol or trait_symbol.kind != SymbolKind.TRAIT:
            raise TypeCheckError(
                f"'{trait_name}' is not a trait",
                source_loc=expr.source_loc
            )

        trait_def = trait_symbol.definition
        if not trait_def:
            raise TypeCheckError(
                f"trait '{trait_name}' has no definition",
                source_loc=expr.source_loc
            )

        # Find the method in the trait's method list
        method_index = None
        trait_method = None
        for i, m in enumerate(trait_def.methods):
            if m.name == method_name:
                method_index = i
                trait_method = m
                break

        if trait_method is None:
            raise TypeCheckError(
                f"trait '{trait_name}' has no method '{method_name}'",
                source_loc=expr.source_loc,
                hint=f"available methods: {', '.join(m.name for m in trait_def.methods)}"
            )

        # Validate argument count (excluding self)
        expected_args = len(trait_method.params)
        if len(expr.args) != expected_args:
            raise TypeCheckError(
                f"trait method '{trait_name}::{method_name}' expects {expected_args} argument(s), got {len(expr.args)}",
                source_loc=expr.source_loc
            )

        # Type check arguments against trait method params
        for i, (arg, param) in enumerate(zip(expr.args, trait_method.params)):
            arg_type = self.check_expression(arg)
            if not TypeUtils.types_compatible(arg_type, param.param_type):
                raise TypeCheckError(
                    f"argument {i+1} to '{method_name}' has wrong type: expected {param.param_type}, got {arg_type}",
                    source_loc=arg.source_loc if hasattr(arg, 'source_loc') else expr.source_loc
                )

        # Build self argument (the trait pointer itself)
        field_access = expr.func
        self_arg = field_access.base

        # Determine if trait methods are far
        trait_is_far = trait_method.is_far

        # Store dispatch info for codegen
        expr.method_call_info = {
            'is_trait_dispatch': True,
            'trait_name': trait_name,
            'method_name': method_name,
            'method_index': method_index,
            'self_arg': self_arg,
            'trait_is_far': trait_is_far
        }

        # Set return type from trait method definition
        if trait_method.return_type:
            expr.expr_type = trait_method.return_type
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

        # Const math builtins - should be folded away, but set types for safety
        if builtin.kind.value == "const_math":
            if expr.builtin_name in ('fixed_sin', 'fixed_cos', 'fixed_log2', 'fixed_lerp', 'fixed_clamp'):
                expr.expr_type = BasicTypeInfo('i16')
            else:
                expr.expr_type = BasicTypeInfo('u16')
            return expr.expr_type

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

        # Handle type_id() on trait pointers
        if expr.method_name == 'type_id':
            if isinstance(receiver_type, PointerTypeInfo) and isinstance(receiver_type.pointee_type, TraitTypeInfo):
                if len(expr.args) != 0:
                    raise TypeCheckError(
                        f"type_id() takes no arguments, got {len(expr.args)}",
                        source_loc=expr.source_loc
                    )
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type
            else:
                raise TypeCheckError(
                    f"type_id() can only be called on trait pointers, found {receiver_type}",
                    source_loc=expr.source_loc
                )

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
        """Check mode compatibility between caller and callee.

        In the new design, mode transitions are handled automatically by the compiler:
        - Compiler inserts REP/SEP around 16-bit A operations
        - Call sites switch to callee's entry mode as needed

        This method no longer raises errors for mode mismatches - the compiler
        will generate the appropriate mode switch code.
        """
        # Mode transitions are now automatic - no validation needed
        # The MIR builder will insert REP/SEP instructions as needed at call sites
        pass

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
                except HIRError:
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
            except HIRError:
                raise TypeCheckError(
                    "Cannot determine size of array type",
                    source_loc=arg.source_loc
                )

        else:
            raise TypeCheckError(
                "size_of expects a type identifier (like u8, Player, etc.), not a value expression",
                source_loc=arg.source_loc
            )
