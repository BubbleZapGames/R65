# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Main type checker for R65 compiler.

Orchestrates type checking, mode tracking, and validation.
"""

from typing import Optional
from r65.compiler.hir import (
    HIRProgram, HIRFunctionDecl, HIRExpression, HIRStatement,
    HIRBinaryOp, HIRUnaryOp, HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr,
    HIRIdentifier, HIRFunctionAddress, HIRRegister, HIRStatusFlagAccess, HIRIncludeBytesExpr, HIRArrayFillExpr, HIRArrayLiteralExpr,
    HIRStringLiteral,
    HIRStructFieldInit, HIRStructLiteralExpr,
    HIRTypeCast, HIRFunctionCall,
    HIRMethodCall, HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf, HIRAssignment, HIRMultiAssignment,
    HIRLetStmt, HIRMultiLetStmt, HIRExprStmt, HIRReturnStmt, HIRIfStmt, HIRWhileStmt, HIRBreakStmt, HIRBlock,
    HIRStaticDecl, HIRConstDecl, HIRTypeAlias,
    HIRMatchExpression, HIRPattern, HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern, HIRIdentifierPattern, HIROrPattern,
    HIRBlockExpression, HIRIfExpression, HIRLoopExpression,
    BasicTypeInfo, TypeInfo, SymbolKind, NeverTypeInfo, MultiReturnTypeInfo,
    RegisterBinding, ArrayTypeInfo, StructTypeInfo, EnumTypeInfo,
    HIRError,
)
from r65.compiler.registers import REGISTERS
from r65.compiler.hir.types import (
    INTEGER_TYPE_NAMES,
    FunctionTypeInfo, PointerTypeInfo, NewtypeTypeInfo, strip_newtype
)
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState, XModeState
from r65.compiler.typeck.type_utils import (
    TypeRelations, TypeUtils, soa_static_of, value_fits_type, widen_implicitly)
from r65.compiler.typeck.operator_validator import OperatorValidator
from r65.compiler.typeck.preservation_checker import PreservationChecker
from r65.compiler.typeck.register_capabilities import (
    is_index_register,
    can_transfer_directly,
    get_transfer_error_hint,
)
from r65.compiler.typeck.type_inference import TypeInference
from r65.compiler.typeck.errors import TypeCheckError, TypeCheckWarning
from r65.compiler.typeck.string_validator import StringValidator
from r65.compiler.typeck.match_validator import MatchValidator
from r65.compiler.typeck.struct_validator import StructValidator
from r65.compiler.typeck.call_validator import CallValidator
from r65.compiler.typeck.pointer_validator import PointerValidator
from r65.compiler.typeck.far_pointer_params import FarPromotion, promote_far_pointer_params
from r65.compiler.hir.ast_const_eval import ConstEvaluator


class TypeChecker:
    """
    Main type checker orchestrator.

    Runs one program-wide pre-pass, `promote_far_pointer_params` (the one
    step that changes declarations; what it changed is kept in
    `far_pointer_promotions`), then walks each function once: checks types
    bottom-up (operator restrictions inline, via `check_binary_op`), then
    XY16 regions, then `#[preserves]`.
    """

    def __init__(self, program: HIRProgram):
        self.program = program
        self.symbol_table = program.symbol_table
        self.relations = TypeRelations(self.symbol_table)

        # Parameters `check()` retyped from `*T` to `far *T`
        self.far_pointer_promotions: list[FarPromotion] = []

        # Current context during type checking
        self.current_function: Optional[HIRFunctionDecl] = None
        self.current_mode: ProcessorMode = ProcessorMode.default()

        # Collect warnings during type checking
        self.warnings: list[TypeCheckWarning] = []

        # Const evaluator for bounds checking
        self.const_evaluator = ConstEvaluator(self.symbol_table)

        # Initialize validators (lazily created to avoid circular init)
        self._match_validator: Optional[MatchValidator] = None
        self._struct_validator: Optional[StructValidator] = None
        self._call_validator: Optional[CallValidator] = None
        self._pointer_validator: Optional[PointerValidator] = None

        # Track register aliases: symbol ID -> register name
        # Maps local variables bound to registers (e.g., let x @ X = 10)
        self._register_aliases: dict[int, str] = {}

    @property
    def match_validator(self) -> MatchValidator:
        """Lazy initialization of match validator."""
        if self._match_validator is None:
            self._match_validator = MatchValidator(
                self.symbol_table, self.check_expression, self.check_statement)
        return self._match_validator

    @property
    def struct_validator(self) -> StructValidator:
        """Lazy initialization of struct validator."""
        if self._struct_validator is None:
            self._struct_validator = StructValidator(
                self.symbol_table, self.check_expression, self._check_type_match
            )
        return self._struct_validator

    @property
    def call_validator(self) -> CallValidator:
        """Lazy initialization of call validator."""
        if self._call_validator is None:
            self._call_validator = CallValidator(
                self.symbol_table, self.relations, self._lookup_function_decl,
                self.check_expression, self.check_element_base,
                lambda: self.current_mode,
                lambda: self.current_function
            )
        return self._call_validator

    @property
    def pointer_validator(self) -> PointerValidator:
        """Lazy initialization of pointer validator."""
        if self._pointer_validator is None:
            self._pointer_validator = PointerValidator(self.check_expression,
                                                       self.program.hirom)
        return self._pointer_validator

    def warn(self, message: str, source_loc=None):
        """Emit a type checking warning."""
        warning = TypeCheckWarning(message, source_loc)
        self.warnings.append(warning)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _is_valid_register_type(self, register_name: str, type_name: str) -> bool:
        """
        Check if a type is valid for a given register.

        Args:
            register_name: Name of the register (A, B, X, Y, D, S, DBR, PBR, STATUS)
            type_name: Name of the type (u8, i8, u16, i16)

        Returns:
            True if the type is valid for the register
        """
        reg = REGISTERS.get(register_name)
        return reg is not None and type_name in reg.types

    def _lookup_function_decl(self, func_name: str, source_loc=None) -> HIRFunctionDecl:
        """
        Look up function declaration by name.

        Args:
            func_name: Name of function to find
            source_loc: Source location for error reporting

        Returns:
            HIRFunctionDecl

        Raises:
            TypeCheckError: If function not found
        """
        for decl in self.program.all_functions():
            if decl.name == func_name:
                return decl

        raise TypeCheckError(
            f"Function '{func_name}' not found",
            source_loc=source_loc
        )

    def _get_register_type_from_parameter(self, register_name: str) -> Optional[TypeInfo]:
        """
        Check if a register is bound to a parameter with an explicit type.

        This allows functions to use registers like X/Y without requiring
        mode annotations when the parameter explicitly declares the type.

        Args:
            register_name: Name of the register ("A", "X", "Y", etc.)

        Returns:
            TypeInfo if register is bound to a parameter, None otherwise
        """
        if not self.current_function:
            return None

        from r65.compiler.hir.nodes import RegisterBinding

        for param in self.current_function.parameters:
            if isinstance(param.binding, RegisterBinding):
                if param.binding.register_name == register_name:
                    return param.param_type

        return None

    def _require_boolean_type(self, expr_type: TypeInfo, context: str, source_loc=None):
        """
        Validate that a type is boolean, raise error if not.

        Args:
            expr_type: Type to check
            context: Context string for error message (e.g., "if condition")
            source_loc: Source location for error

        Raises:
            TypeCheckError: If type is not boolean
        """
        if not TypeUtils.is_boolean_type(expr_type):
            raise TypeCheckError(
                f"{context} must be boolean, found {expr_type}",
                source_loc=source_loc,
                hint=self._bool_payload_hint(expr_type)
            )

    @staticmethod
    def _bool_payload_hint(expr_type: TypeInfo) -> Optional[str]:
        """Hint for a newtype over `bool` used where a bool is wanted.

        Consuming a value *as* a bool is the value flowing out, which a newtype
        does not do implicitly — the same rule that rejects `let b: bool = f;`.
        A pattern is not a consumer, so `match f { true => ... }` still works on
        the wrapper; only the conditions need the unwrap, and the error should
        say so rather than leave the reader guessing which rule they hit.
        """
        if not isinstance(expr_type, NewtypeTypeInfo):
            return None
        if str(strip_newtype(expr_type)) != 'bool':
            return None
        return (f"'{expr_type}' wraps a bool but does not flow out as one; "
                f"unwrap it with '.0' (a 'match' on it needs no unwrap)")

    def _require_integer_type(self, expr_type: TypeInfo, context: str, source_loc=None):
        """
        Validate that a type is integer, raise error if not.

        Args:
            expr_type: Type to check
            context: Context string for error message (e.g., "array index")
            source_loc: Source location for error

        Raises:
            TypeCheckError: If type is not integer
        """
        if not TypeUtils.is_integer_type(expr_type):
            raise TypeCheckError(
                f"{context} must be integer, found {expr_type}",
                source_loc=source_loc
            )

    def _get_target_register(self, target: HIRExpression) -> Optional[str]:
        """
        Get the register name if target is a register or register-aliased variable.

        Args:
            target: The assignment target expression

        Returns:
            Register name ('A', 'X', 'Y', 'B') or None if not a register target
        """
        # Direct register reference (e.g., X = ...)
        if isinstance(target, HIRRegister):
            return target.name

        # Register-aliased variable (e.g., x = ... where let x @ X = ...)
        if isinstance(target, HIRIdentifier) and target.symbol:
            symbol_id = id(target.symbol)
            if symbol_id in self._register_aliases:
                return self._register_aliases[symbol_id]

        return None

    def _check_target_mutable(self, target: HIRExpression, source_loc):
        """
        Verify that an assignment target is mutable.

        Only direct identifier reassignment (x = 5, x += 1, x++) needs `mut`.
        Writes through a field, element or deref never do (CLAUDE.md
        §Mutability).
        """
        # Only check direct identifier assignments
        if isinstance(target, HIRIdentifier) and target.symbol is not None:
            if not target.symbol.is_mutable:
                raise TypeCheckError(
                    f"cannot assign to immutable variable '{target.name}'",
                    source_loc=source_loc,
                    hint=f"declare with 'let mut {target.name}' or 'static mut {target.name}' to allow mutation"
                )


    def _binary_op_uses_target(self, value: HIRExpression, target: HIRExpression) -> bool:
        """
        Check if a binary operation's left operand references the same target.

        This detects patterns like `X = X + 5` where the target is used in the value.

        Args:
            value: The value expression (should be HIRBinaryOp)
            target: The assignment target

        Returns:
            True if the binary op's left operand is the same as target
        """
        if not isinstance(value, HIRBinaryOp):
            return False

        left = value.left

        # Direct register comparison
        if isinstance(target, HIRRegister) and isinstance(left, HIRRegister):
            return target.name == left.name

        # Register-aliased variable comparison
        if isinstance(target, HIRIdentifier) and isinstance(left, HIRIdentifier):
            if target.symbol and left.symbol:
                return id(target.symbol) == id(left.symbol)

        # Mixed: target is register, left is aliased variable (or vice versa)
        target_reg = self._get_target_register(target)
        left_reg = self._get_target_register(left)
        if target_reg and left_reg:
            return target_reg == left_reg

        return False

    def _validate_comparison_operands(self, expr: HIRBinaryOp) -> None:
        """
        Validate that comparison operands are valid for hardware.

        Rejects comparing two index registers (X vs Y) because there's no
        direct CPX Y or CPY X instruction - it would require using an
        intermediate register.

        Args:
            expr: The comparison expression

        Raises:
            TypeCheckError: If comparing two index registers
        """
        left_reg = self._get_target_register(expr.left)
        right_reg = self._get_target_register(expr.right)

        # Check if both operands are index registers
        if left_reg and right_reg:
            if is_index_register(left_reg) and is_index_register(right_reg):
                raise TypeCheckError(
                    f"cannot compare {left_reg} with {right_reg} directly",
                    source_loc=expr.source_loc,
                    hint=f"no CPX {right_reg} or CPY {left_reg} instruction exists; "
                         f"store one register to a variable first, then compare"
                )

    def _check_no_aggregate_type(self, type_info: TypeInfo, context: str, source_loc=None,
                                   suggestion_suffix: str = "", verb: str = "used"):
        """
        Validate that a type is not an aggregate type (array or struct).

        Arrays and structs cannot be passed by value, returned by value, or assigned by value.
        This helper provides consistent error messages with appropriate suggestions.

        Args:
            type_info: Type to check
            context: Description of where the type appears (e.g., "Parameter 'x'", "Return type")
            source_loc: Source location for error reporting
            suggestion_suffix: Additional text to append to suggestion (e.g., "\\n  Or write to a pre-allocated output parameter")
            verb: The verb to use in the error message ("passed", "returned", or "used")

        Raises:
            TypeCheckError: If type is an aggregate type
        """
        if TypeUtils.is_aggregate_type(type_info):
            type_name = str(type_info)
            suggestion = f"*name: {type_name}"
            raise TypeCheckError(
                f"{context} has type '{type_name}' which cannot be {verb} by value\n"
                f"  Arrays and structs must be passed by reference\n"
                f"  Suggestion: Use a pointer type instead: {suggestion}{suggestion_suffix}",
                source_loc=source_loc
            )

    def _check_return_fits_a_register(self, func):
        """
        Reject return values wider than one register.

        The return ABI hands each value back in a register (A, then B or X,
        then Y), so a value has at most 2 bytes to travel in. A `far *T` is
        3 bytes: the callee would build it in its stack frame and return only
        the low byte, leaving the caller to read the other two out of the
        frame it just deallocated. That links cleanly and produces a wild
        pointer, so it is rejected here rather than miscompiled.

        Raises:
            TypeCheckError: If any returned value needs more than 2 bytes.
        """
        from r65.compiler.hir.types import MultiReturnTypeInfo

        ret_type = func.return_type
        parts = (ret_type.element_types
                 if isinstance(ret_type, MultiReturnTypeInfo) else [ret_type])

        for part in parts:
            size = getattr(part, 'size_bytes', None)
            if isinstance(size, int) and size > 2:
                raise TypeCheckError(
                    f"Function '{func.name}' returns '{part}', which is {size} bytes "
                    f"and does not fit in a return register\n"
                    f"  Return values travel in A, B/X, or Y - at most 2 bytes each\n"
                    f"  Suggestion: return a near pointer, or write the value "
                    f"through an output parameter instead of returning it",
                    source_loc=func.source_loc
                )

    def _check_type_match(self, expected_type: TypeInfo, actual_type: TypeInfo,
                          expr: HIRExpression, context: str, source_loc=None,
                          use_compatible: bool = False):
        """
        Check that actual_type matches expected_type, raising an error if not.

        This helper consolidates type mismatch checking with consistent error messages.

        Args:
            expected_type: The expected type
            actual_type: The actual type found
            expr: The expression being checked (for enhanced error messages)
            context: Description of the context (e.g., "let binding", "assignment")
            source_loc: Source location for error reporting
            use_compatible: If True, use assignable() instead of types_equal()

        Raises:
            TypeCheckError: If types don't match
        """
        # Every caller here is assignment-shaped (let / static / const /
        # assignment / if-arm / break value), so `expected_type` is the place
        # being written and the directional `assignable` rule applies. A newtype
        # takes it even without `use_compatible`: transparent in, opaque out.
        if (use_compatible or isinstance(expected_type, NewtypeTypeInfo)
                or isinstance(actual_type, NewtypeTypeInfo)):
            types_match = self.relations.assignable(actual_type, expected_type)
        else:
            types_match = TypeUtils.types_equal(expected_type, actual_type)

        if not types_match:
            self._raise_type_mismatch_error(
                expected_type=expected_type,
                actual_type=actual_type,
                expr=expr,
                context=context,
                source_loc=source_loc
            )

    def _raise_type_mismatch_error(self, expected_type: TypeInfo, actual_type: TypeInfo,
                                    expr: HIRExpression, context: str, source_loc=None):
        """
        Raise a type mismatch error with improved messages for literal overflow.

        Detects when the mismatch is due to an integer literal exceeding
        the target type's range and provides a clearer error message.
        """
        # Check if this is a literal overflow case
        if isinstance(expr, HIRIntegerLiteral) and isinstance(expected_type, BasicTypeInfo):
            from r65.compiler.typeck.type_utils import get_type_range
            value = expr.value
            type_name = expected_type.name

            # Get range for expected type
            range_info = get_type_range(type_name)
            if range_info is not None:
                min_val, max_val = range_info
                if value < min_val or value > max_val:
                    # This is a literal overflow!
                    if value > max_val:
                        raise TypeCheckError(
                            f"literal value {value} exceeds maximum for type {type_name} ({max_val})",
                            source_loc=source_loc,
                            hint=f"use a larger type like u16, or reduce the value (valid range: {min_val} to {max_val})"
                        )
                    else:  # value < min_val
                        raise TypeCheckError(
                            f"literal value {value} is below minimum for type {type_name} ({min_val})",
                            source_loc=source_loc,
                            hint=f"use a signed type like i8 or i16 for negative values (valid range: {min_val} to {max_val})"
                        )

        # Generate helpful hint based on context
        hint = self._generate_type_mismatch_hint(expected_type, actual_type, context)

        # Default type mismatch error
        raise TypeCheckError(
            f"type mismatch in {context}: expected {expected_type}, found {actual_type}",
            source_loc=source_loc,
            hint=hint
        )

    def _generate_type_mismatch_hint(self, expected_type: TypeInfo, actual_type: TypeInfo,
                                      context: str) -> Optional[str]:
        """Generate a helpful hint for type mismatch errors."""
        # Suggest cast for integer size mismatches
        if isinstance(expected_type, BasicTypeInfo) and isinstance(actual_type, BasicTypeInfo):
            expected_name = expected_type.name
            actual_name = actual_type.name

            # Integer type mismatches
            if expected_name in INTEGER_TYPE_NAMES and actual_name in INTEGER_TYPE_NAMES:
                return f"use explicit cast: (value as {expected_name})"

            # Bool to integer or vice versa
            if expected_name == 'bool' and actual_name in INTEGER_TYPE_NAMES:
                return "use comparison to convert integer to bool: (value != 0)"
            if expected_name in INTEGER_TYPE_NAMES and actual_name == 'bool':
                return f"use cast to convert bool to integer: (value as {expected_name})"

        # Pointer type hints
        if TypeUtils.drops_bank_byte(actual_type, expected_type):
            return (f"a far pointer does not narrow implicitly; declare the destination "
                    f"'far {expected_type}', or cast (value as {expected_type}) to drop the bank byte")
        if isinstance(expected_type, PointerTypeInfo) and not isinstance(actual_type, PointerTypeInfo):
            return f"expected a pointer; use &value to get address"

        return None

    def _get_promoted_type(self, left_type: TypeInfo, right_type: TypeInfo) -> Optional[TypeInfo]:
        """
        Get the result type when implicitly promoting operands in arithmetic.

        Supports automatic integer promotion for mixed-size operands:
        - u8 + u16 -> u16 (u8 widened to u16)
        - i8 + i16 -> i16 (i8 widened to i16)
        - u8 + i8 -> u8 (same-size mix prefers unsigned)

        Args:
            left_type: Type of left operand
            right_type: Type of right operand

        Returns:
            The promoted result type, or None if promotion is not allowed
        """
        if not isinstance(left_type, BasicTypeInfo) or not isinstance(right_type, BasicTypeInfo):
            return None

        left_name = left_type.name
        right_name = right_type.name

        # Define type hierarchy (smaller -> larger)
        unsigned_hierarchy = {'u8': 8, 'u16': 16}
        signed_hierarchy = {'i8': 8, 'i16': 16}

        # Both unsigned: promote to larger
        if left_name in unsigned_hierarchy and right_name in unsigned_hierarchy:
            left_bits = unsigned_hierarchy[left_name]
            right_bits = unsigned_hierarchy[right_name]
            if left_bits > right_bits:
                return left_type
            else:
                return right_type

        # Both signed: promote to larger
        if left_name in signed_hierarchy and right_name in signed_hierarchy:
            left_bits = signed_hierarchy[left_name]
            right_bits = signed_hierarchy[right_name]
            if left_bits > right_bits:
                return left_type
            else:
                return right_type

        # Mixed signed/unsigned of same size: allow (reinterpret)
        if (left_name == 'u8' and right_name == 'i8') or (left_name == 'i8' and right_name == 'u8'):
            return BasicTypeInfo(name='u8')  # Prefer unsigned for mixed 8-bit
        if (left_name == 'u16' and right_name == 'i16') or (left_name == 'i16' and right_name == 'u16'):
            return BasicTypeInfo(name='u16')  # Prefer unsigned for mixed 16-bit

        # Different sizes: the wider operand's type wins
        # (u8 + i16 -> i16, i8 + u16 -> u16)
        if left_name in INTEGER_TYPE_NAMES and right_name in INTEGER_TYPE_NAMES:
            left_bits = 16 if left_name in ['u16', 'i16'] else 8
            right_bits = 16 if right_name in ['u16', 'i16'] else 8
            if left_bits > right_bits:
                return left_type
            elif right_bits > left_bits:
                return right_type

        return None

    def _check_array_element(self, index: int, elem, elem_type: TypeInfo,
                             expected: TypeInfo):
        """Check one array literal element against the array's element type.

        Initializing from an array literal is assignment, so this is the same
        rule as `let`: same-size signed/unsigned still mix freely, a payload
        still flows implicitly into an array of a newtype, and a newtype does
        not flow back out into an array of its payload.
        """
        if self.relations.assignable(elem_type, expected):
            return
        raise TypeCheckError(
            f"Array element {index} has type {elem_type}, expected {expected}",
            source_loc=elem.source_loc
        )

    # ========================================================================
    # Main Type Checking
    # ========================================================================

    def _validate_function_type_no_aggregates(self, func_type: FunctionTypeInfo, context: str, source_loc=None):
        """
        Validate that a function type doesn't use aggregate types (arrays/structs) for parameters or return.

        Function pointer types cannot have aggregate parameters or return types since the underlying
        functions cannot pass/return aggregates by value.

        Args:
            func_type: FunctionTypeInfo to validate
            context: Context string for error message (e.g., "type alias 'Callback'")
            source_loc: Source location for error reporting
        """
        for i, param_type in enumerate(func_type.param_types):
            self._check_no_aggregate_type(
                param_type,
                f"Function type in {context}, parameter {i + 1}",
                source_loc,
                verb="passed"
            )

        if func_type.return_type:
            self._check_no_aggregate_type(
                func_type.return_type,
                f"Function type in {context}, return type",
                source_loc,
                verb="returned"
            )

    def check(self):
        """Perform type checking on entire program."""
        # Validate type aliases with function types
        for decl in self.program.declarations:
            if isinstance(decl, HIRTypeAlias):
                if isinstance(decl.aliased_type, FunctionTypeInfo):
                    self._validate_function_type_no_aggregates(
                        decl.aliased_type,
                        f"type alias '{decl.name}'",
                        decl.source_loc
                    )

        # Type check static initializers and validate function pointer types
        for decl in self.program.declarations:
            if isinstance(decl, HIRStaticDecl):
                # Validate function pointer type variables don't have aggregate params/returns
                if isinstance(decl.var_type, FunctionTypeInfo):
                    self._validate_function_type_no_aggregates(
                        decl.var_type,
                        f"static variable '{decl.name}'",
                        decl.source_loc
                    )

                if decl.initializer:
                    init_type = self.check_expression(decl.initializer, decl.var_type)
                    self._check_type_match(
                        decl.var_type, init_type, decl.initializer,
                        "static variable initializer", decl.source_loc,
                        use_compatible=True
                    )

            elif isinstance(decl, HIRConstDecl):
                if decl.value:
                    value_type = self.check_expression(decl.value, decl.const_type)
                    self._check_type_match(
                        decl.const_type, value_type, decl.value,
                        "const declaration", decl.source_loc
                    )

        # Retype near pointer params that a caller passes a far address. This
        # changes declarations, so it runs before any function body is checked.
        self.far_pointer_promotions = promote_far_pointer_params(self.program)

        # Type check every function body, impl methods included
        for func in self.program.all_functions():
            self.check_function(func)

    def check_function(self, func: HIRFunctionDecl):
        """Type check a single function."""
        self.current_function = func

        # Clear register aliases from previous function
        self._register_aliases.clear()

        # Track register aliases from function parameters
        from r65.compiler.hir.nodes import RegisterBinding
        for param in func.parameters:
            if isinstance(param.binding, RegisterBinding) and param.symbol:
                self._register_aliases[id(param.symbol)] = param.binding.register_name

        # Validate that parameters are not aggregate types (arrays/structs cannot be passed by value)
        for param in func.parameters:
            self._check_no_aggregate_type(
                param.param_type,
                f"Parameter '{param.name}'",
                func.source_loc,
                verb="passed"
            )

        # Validate that return type is not an aggregate type (arrays/structs cannot be returned by value)
        # Const fns are exempt — they execute at compile time, no hardware registers involved
        if func.return_type and not func.is_const:
            self._check_no_aggregate_type(
                func.return_type,
                f"Function '{func.name}' return type",
                func.source_loc,
                suggestion_suffix="\n  Or write to a pre-allocated output parameter",
                verb="returned"
            )
            self._check_return_fits_a_register(func)

        # Get entry mode from inferred mode (based on parameter types)
        # entry_m_mode is set by HIR builder based on A parameter type
        if func.entry_m_mode is not None:
            entry_mode = ProcessorMode(func.entry_m_mode, XModeState.X16)
        else:
            # Fallback: use default mode (m8, x16)
            entry_mode = ProcessorMode.default()

        self.current_mode = entry_mode

        if func.body is None:
            return  # External function

        # The mode is fixed for the whole body (asm! SEP/REP is not tracked).
        self.check_block(func.body)

        # Validate XY16 (x8) regions — reject unsafe patterns inside
        # `STATUS.XY16 = false` / `= true` pairs (calls, control flow, indexing,
        # missing restore). See typeck/xy16_region.py for the rules.
        from r65.compiler.typeck.xy16_region import check_xy16_regions
        check_xy16_regions(func)

        # Check register preservation
        if func.preserves_attr:
            preservation_checker = PreservationChecker(func)
            preservation_checker.check()

        self.current_function = None

    def check_block(self, block):
        """Type check a block of statements."""
        for stmt in block.statements:
            self.check_statement(stmt)

    def check_statement(self, stmt: HIRStatement):
        """Type check a statement."""
        if isinstance(stmt, HIRLetStmt):
            self.check_let_statement(stmt)

        elif isinstance(stmt, HIRMultiLetStmt):
            self.check_multi_let_statement(stmt)

        elif isinstance(stmt, HIRExprStmt):
            self.check_expression(stmt.expr)

        elif isinstance(stmt, HIRReturnStmt):
            self._check_return_statement(stmt)

        elif isinstance(stmt, HIRIfStmt):
            # Check condition
            cond_type = self.check_expression(stmt.condition)
            self._require_boolean_type(cond_type, "If condition", stmt.condition.source_loc)

            # Check branches
            self.check_block(stmt.then_block)
            if stmt.else_block:
                if isinstance(stmt.else_block, HIRIfStmt):
                    self.check_statement(stmt.else_block)
                else:
                    self.check_block(stmt.else_block)

        elif isinstance(stmt, HIRBreakStmt):
            if stmt.value is not None:
                self.check_expression(stmt.value)

        elif isinstance(stmt, HIRWhileStmt):
            if stmt.condition:
                cond_type = self.check_expression(stmt.condition)
                self._require_boolean_type(cond_type, "While condition", stmt.condition.source_loc)
            self.check_block(stmt.body)

        elif isinstance(stmt, HIRBlock):
            # Nested block (e.g., from for loop desugaring)
            self.check_block(stmt)

        elif isinstance(stmt, HIRAssignment):
            # Assignment statement (e.g., from for loop increment)
            self.check_assignment(stmt)

        elif isinstance(stmt, HIRMultiAssignment):
            # Multi-assignment statement
            self.check_multi_assignment(stmt)

    def _is_direct_clone(self, expr) -> bool:
        """True if expr is a `.clone()` sugar call (0-arg clone on a receiver)."""
        return (isinstance(expr, HIRFunctionCall)
                and isinstance(expr.func, HIRFieldAccess)
                and expr.func.field_name == 'clone'
                and len(expr.args) == 0)

    def _check_copy_value(self, expr, context_type=None, *, clone_ok: bool = True) -> TypeInfo:
        """Type check a let initializer or an assignment's value.

        This is the one position where `.clone()` sugar is legal (when
        `clone_ok`): the call itself is checked with that permission, and
        nothing nested inside it inherits it.
        """
        if clone_ok and self._is_direct_clone(expr):
            return self.call_validator.check_function_call(expr, clone_sugar_ok=True)
        return self.check_expression(expr, context_type)

    def _is_legal_aggregate_initializer(self, expr) -> bool:
        """Initializers that may legally produce an aggregate local.

        These are the forms MIR's `_emit_aggregate_init` can actually copy: a
        fresh literal (struct/array/string/array-fill) or an explicit `.clone()`.
        A bare aggregate *place* (identifier, field, index, deref) is NOT here —
        it would fall through to a scalar Store and silently leave the local
        uninitialized, so it must be rejected with clone guidance.
        """
        return (self._is_direct_clone(expr)
                or isinstance(expr, (HIRArrayFillExpr, HIRArrayLiteralExpr,
                                     HIRStringLiteral, HIRStructLiteralExpr)))

    def _check_return_statement(self, stmt: HIRReturnStmt):
        """Type check `return`, comparing each value against the declared type.

        Return is assignment-shaped — the value is stored into the return
        register — so the comparison is `assignable(actual, declared)`. That
        gives newtype opacity and rejects a far pointer narrowing to near.
        Integer narrowing is allowed and truncates, as in `let`
        (docs/type-system.md, Assignment Type Checking).

        The declared type is also passed as context when checking each value, so
        an out-of-range literal is caught the same way it is in a `let`.
        """
        func = self.current_function
        declared = func.return_type if func is not None else None

        if isinstance(declared, MultiReturnTypeInfo):
            expected = list(declared.element_types)
        elif declared is None or isinstance(declared, NeverTypeInfo):
            expected = []
        else:
            expected = [declared]

        self._check_return_arity(stmt, expected, declared)

        for i, val in enumerate(stmt.values):
            want = expected[i] if i < len(expected) else None
            actual = self.check_expression(val, want)

            if want is None or isinstance(actual, NeverTypeInfo):
                continue
            if self.relations.assignable(actual, want):
                stmt.values[i] = widen_implicitly(val, actual, want)
                continue

            where = f" (value {i + 1})" if len(expected) > 1 else ""
            raise TypeCheckError(
                f"returning '{actual}' from a function declared "
                f"'-> {declared}'{where}",
                source_loc=getattr(val, 'source_loc', stmt.source_loc),
                hint=(f"a far pointer does not narrow implicitly; cast with "
                      f"'as {want}' to drop the bank byte"
                      if TypeUtils.drops_bank_byte(actual, want) else
                      f"the return type is '{want}'; convert explicitly with "
                      f"'as {want}'")
            )

    @staticmethod
    def _check_return_arity(stmt: HIRReturnStmt, expected, declared):
        """Reject `return` handing back the wrong number of values.

        Only when both sides are non-empty. A bare `return;` is the documented
        implicit-A form — the value is already in the register, so there is
        nothing to count — and a `-> !` function has no declared type to count
        against. Without the guard both would start failing.

        Unchecked, the caller believes the signature: `let a, b = f();` against a
        `return 1;` reads a register the callee never wrote.
        """
        if not stmt.values or not expected:
            return
        if len(stmt.values) == len(expected):
            return

        got, want = len(stmt.values), len(expected)
        hint = (f"'-> {declared}' returns {want} values"
                if got < want else
                f"'-> {declared}' returns {want}, so drop the extra")
        if got < want:
            hint += "; a bare 'return;' returns whatever is already in A"
        raise TypeCheckError(
            f"returning {got} value{'s' if got != 1 else ''} from a function "
            f"declared '-> {declared}', which returns {want}",
            source_loc=stmt.source_loc,
            hint=hint
        )

    def check_let_statement(self, stmt: HIRLetStmt):
        """Type check let binding."""
        mode = self.current_mode

        # Determine variable type
        # Track if we inferred type from initializer (skip re-checking in that case)
        inferred_from_initializer = False

        if stmt.var_type:
            # Explicit type provided
            var_type = stmt.var_type
        elif isinstance(stmt.binding, RegisterBinding):
            # For register bindings without explicit type:
            # 1. Get the register's canonical type (e.g., u16 for X/Y, u8 for DBR)
            # 2. Use that as context when type-checking the initializer
            # This ensures literals get the correct type for the register
            reg_name = stmt.binding.register_name
            register_type = TypeInference.infer_register_alias_type(reg_name, mode)

            if stmt.initializer:
                # Infer type from initializer with register type as context
                # This allows `let x @ X = 100;` to work - literal 100 infers as u16
                init_type = self.check_expression(stmt.initializer, context_type=register_type)
                # Handle tuple: use first element type
                if isinstance(init_type, MultiReturnTypeInfo):
                    init_type = init_type.element_types[0]
                # Validate the inferred type is valid for the register
                if isinstance(init_type, BasicTypeInfo) and self._is_valid_register_type(reg_name, init_type.name):
                    var_type = init_type
                    inferred_from_initializer = True  # Already type-checked
                else:
                    # Initializer type doesn't match register - use register's type
                    # and let the type check below report the error
                    var_type = register_type
            else:
                # No initializer, use register type based on mode
                var_type = register_type
            if var_type is None:
                raise TypeCheckError(
                    f"Cannot determine type of register {stmt.binding.register_name} in unknown mode",
                    source_loc=stmt.source_loc
                )
            stmt.var_type = var_type  # Fill in inferred type
        elif stmt.initializer:
            # No explicit type — infer from initializer expression
            init_type = self._check_copy_value(stmt.initializer)
            if isinstance(init_type, MultiReturnTypeInfo):
                init_type = init_type.element_types[0]
            var_type = init_type
            inferred_from_initializer = True
            stmt.var_type = var_type
            if stmt.symbol:
                stmt.symbol.var_type = var_type
        else:
            # No type annotation and no initializer — cannot infer
            raise TypeCheckError(
                f"Variable '{stmt.name}' requires explicit type annotation",
                source_loc=stmt.source_loc
            )

        # Update symbol table with inferred type
        if stmt.symbol:
            stmt.symbol.var_type = var_type

        # Track register alias if this is a register-bound let
        if isinstance(stmt.binding, RegisterBinding) and stmt.symbol:
            self._register_aliases[id(stmt.symbol)] = stmt.binding.register_name

        # Check initializer type matches (skip if already checked during type inference)
        if stmt.initializer and not inferred_from_initializer:
            init_type = self._check_copy_value(stmt.initializer, var_type)
            # Handle tuple-to-scalar: let x: u8 = tuple_func() drops extra return values
            if isinstance(init_type, MultiReturnTypeInfo) and not isinstance(var_type, MultiReturnTypeInfo):
                first_elem_type = init_type.element_types[0]
                self._check_type_match(
                    var_type, first_elem_type, stmt.initializer,
                    "let binding (first element of tuple)", stmt.source_loc
                )
            else:
                # Auto-promote let binding from *T to far *T when initializer is far
                if (isinstance(var_type, PointerTypeInfo) and not var_type.is_far and
                        isinstance(init_type, PointerTypeInfo) and init_type.is_far and
                        TypeUtils._pointee_types_compatible(var_type.pointee_type, init_type.pointee_type)):
                    var_type = PointerTypeInfo(is_far=True, pointee_type=var_type.pointee_type)
                    stmt.var_type = var_type
                    if stmt.symbol:
                        stmt.symbol.var_type = var_type
                self._check_type_match(
                    var_type, init_type, stmt.initializer,
                    "let binding", stmt.source_loc, use_compatible=True
                )
                stmt.initializer = widen_implicitly(stmt.initializer, init_type, var_type)

        # Aggregates are not copied by a bare `let x = <place>`. Only a fresh
        # literal or an explicit `.clone()` may initialize an aggregate local;
        # a bare aggregate value (identifier/field/index/deref) would otherwise
        # fall through to a scalar Store in MIR and silently leave the local
        # uninitialized. Reject it here with the same clone guidance as
        # assignment-by-value. (Register bindings are never aggregate.)
        if (stmt.initializer is not None
                and TypeUtils.is_aggregate_type(var_type)
                and not self._is_legal_aggregate_initializer(stmt.initializer)):
            type_name = str(var_type)
            if isinstance(var_type, StructTypeInfo):
                clone_note = f"add `impl Clone for {var_type.name} {{}}` if the struct has none"
            else:  # array — clone is a built-in, no impl needed
                clone_note = "arrays clone built-in"
            src = (stmt.initializer.name
                   if isinstance(stmt.initializer, HIRIdentifier) else "src")
            raise TypeCheckError(
                f"cannot initialize '{stmt.name}' by copying a '{type_name}' by value\n"
                f"  structs and arrays are not copied by a bare 'let ='\n"
                f"  Suggestion: clone explicitly — `let {stmt.name} = {src}.clone()` ({clone_note})",
                source_loc=stmt.source_loc
            )

    def check_multi_let_statement(self, stmt: HIRMultiLetStmt):
        """Type check multi-binding let statement: let a, b = multi_return_func();

        Supports partial capture — binding fewer names than the return count.
        """
        init_type = self.check_expression(stmt.initializer)

        if not isinstance(init_type, MultiReturnTypeInfo):
            raise TypeCheckError(
                f"Multi-let binding requires a multi-return function call, got {init_type}",
                source_loc=stmt.source_loc
            )

        if len(stmt.names) > len(init_type.element_types):
            raise TypeCheckError(
                f"Cannot bind {len(stmt.names)} variables from a {len(init_type.element_types)}-value return",
                source_loc=stmt.source_loc
            )

        var_types = []
        for i, name in enumerate(stmt.names):
            elem_type = init_type.element_types[i]
            var_types.append(elem_type)
            if i < len(stmt.symbols):
                stmt.symbols[i].var_type = elem_type

        stmt.var_types = var_types

    def check_expression(self, expr: HIRExpression, context_type: Optional[TypeInfo] = None) -> TypeInfo:
        """
        Type check an expression and return its type.

        Args:
            expr: Expression to type check
            context_type: Expected type (for inference)

        Returns:
            The expression's type (sets expr.expr_type)
        """
        if isinstance(expr, HIRIntegerLiteral):
            # If type already set (e.g., from const evaluation), preserve it
            if expr.expr_type is not None:
                return expr.expr_type
            # A literal's range is a question about the payload, so a newtype
            # context answers for what it wraps. Without this `let t: TileId = 300;`
            # silently truncates where `let n: u8 = 300;` is an error.
            context_type = strip_newtype(context_type) if context_type is not None else None
            # Overflow check: if context type is a specific integer type and no suffix,
            # verify the value fits the declared type range.
            if (context_type is not None and isinstance(context_type, BasicTypeInfo)
                    and expr.suffix is None):
                from r65.compiler.typeck.type_utils import get_type_range
                range_info = get_type_range(context_type.name)
                if range_info is not None and not value_fits_type(expr.value, context_type.name):
                    min_val, max_val = range_info
                    raise TypeCheckError(
                        f"integer literal {expr.value} does not fit in type {context_type.name} "
                        f"(valid range: {min_val} to {max_val})",
                        source_loc=expr.source_loc,
                        hint=f"use a wider type, or cast with 'as {context_type.name}' to explicitly truncate"
                    )
            # Infer type from context, suffix, or default
            expr_type = TypeInference.infer_integer_literal_type(expr.value, context_type, suffix=expr.suffix)
            expr.expr_type = expr_type
            return expr_type

        elif isinstance(expr, HIRBooleanLiteral):
            expr.expr_type = BasicTypeInfo('bool')
            return expr.expr_type

        elif isinstance(expr, HIREnumVariantExpr):
            # Enum variant has its enum type. Carry the symbol: without it the
            # type has no variant list, so exhaustiveness checking silently
            # no-ops and the discriminant width falls back to one byte.
            symbol = self.symbol_table.lookup(expr.enum_name)
            expr.expr_type = EnumTypeInfo(
                name=expr.enum_name,
                definition=symbol.definition if symbol else None,
                symbol=symbol,
            )
            return expr.expr_type

        elif isinstance(expr, HIRIdentifier):
            # Look up symbol
            symbol = expr.symbol
            if symbol is None:
                raise TypeCheckError(
                    f"undefined identifier '{expr.name}'",
                    source_loc=expr.source_loc,
                    hint="check spelling, or add a declaration for this variable"
                )

            # Get type from symbol
            if symbol.kind in (SymbolKind.PARAMETER, SymbolKind.LOCAL_VAR, SymbolKind.STATIC_VAR, SymbolKind.CONST):
                expr.expr_type = symbol.var_type
            elif symbol.kind == SymbolKind.FUNCTION:
                # Function identifier used as value (function pointer)
                # Look up function declaration to get its type
                func_decl = self._lookup_function_decl(symbol.name, expr.source_loc)

                # Build FunctionTypeInfo from function signature
                from r65.compiler.hir.types import FunctionTypeInfo
                param_types = [param.param_type for param in func_decl.parameters]

                expr.expr_type = FunctionTypeInfo(
                    is_far=func_decl.is_far,
                    param_types=param_types,
                    return_type=func_decl.return_type
                )
            else:
                raise TypeCheckError(
                    f"'{expr.name}' is not a value",
                    source_loc=expr.source_loc
                )

            return expr.expr_type

        elif isinstance(expr, HIRRegister):
            # First check if this register is bound to a parameter with explicit type
            reg_type = self._get_register_type_from_parameter(expr.name)

            if reg_type is None:
                # Fall back to inferring from mode
                reg_type = self.current_mode.get_register_type(expr.name)

            if reg_type is None:
                raise TypeCheckError(
                    f"cannot determine type of register {expr.name} in unknown mode",
                    source_loc=expr.source_loc,
                    hint="mode is inferred from parameter types: u16 @ A means m16, otherwise m8"
                )

            expr.expr_type = reg_type
            return reg_type

        elif isinstance(expr, HIRFunctionAddress):
            return self.call_validator.check_function_address(expr)

        elif isinstance(expr, HIRBinaryOp):
            return self.check_binary_op(expr, context_type)

        elif isinstance(expr, HIRUnaryOp):
            return self.check_unary_op(expr, context_type)

        elif isinstance(expr, HIRTypeCast):
            return self.check_type_cast(expr)

        elif isinstance(expr, HIRFunctionCall):
            return self.call_validator.check_function_call(expr)

        elif isinstance(expr, HIRMethodCall):
            return self.call_validator.check_method_call(expr)

        elif isinstance(expr, HIRArrayIndex):
            return self.check_array_index(expr)

        elif isinstance(expr, HIRFieldAccess):
            return self.check_field_access(expr)

        elif isinstance(expr, HIRStatusFlagAccess):
            return self.check_status_flag_access(expr)

        elif isinstance(expr, HIRAssignment):
            return self.check_assignment(expr)

        elif isinstance(expr, HIRMultiAssignment):
            return self.check_multi_assignment(expr)

        elif isinstance(expr, HIRDereference):
            return self.pointer_validator.check_dereference(expr)

        elif isinstance(expr, HIRAddressOf):
            return self.pointer_validator.check_addressof(expr)

        elif isinstance(expr, HIRIncludeBytesExpr):
            # include_bytes! returns an array of bytes with the actual file size
            from r65.compiler.hir.types import ArrayTypeInfo
            elem_type = BasicTypeInfo(name='u8')
            # Use the actual file size stored during HIR building
            array_type = ArrayTypeInfo(element_type=elem_type, size=expr.size)
            expr.expr_type = array_type
            return array_type

        elif isinstance(expr, HIRArrayFillExpr):
            # Array fill: [value; count]
            from r65.compiler.hir.types import ArrayTypeInfo
            # Get expected element type from context if available
            expected_elem_type = None
            if context_type and isinstance(context_type, ArrayTypeInfo):
                expected_elem_type = context_type.element_type
            fill_type = self.check_expression(expr.fill_value, expected_elem_type)
            # If we have expected type, use it; otherwise use inferred type
            final_elem_type = expected_elem_type if expected_elem_type else fill_type
            array_type = ArrayTypeInfo(element_type=final_elem_type, size=expr.count)
            expr.expr_type = array_type
            return array_type

        elif isinstance(expr, HIRArrayLiteralExpr):
            # Array literal: [a, b, c, ...]
            from r65.compiler.hir.types import ArrayTypeInfo
            if not expr.elements:
                raise TypeCheckError(
                    "Empty array literals are not allowed",
                    source_loc=expr.source_loc
                )
            # Get expected element type from context if available
            expected_elem_type = None
            if context_type and isinstance(context_type, ArrayTypeInfo):
                expected_elem_type = context_type.element_type

            # Type check elements with context
            first_type = self.check_expression(expr.elements[0], expected_elem_type)

            # If we have expected type, use it; otherwise use inferred type
            final_elem_type = expected_elem_type if expected_elem_type else first_type

            # Element 1 is checked only when the element type came from context.
            # With no context it *is* the inference source, so there is nothing
            # to check it against.
            if expected_elem_type is not None:
                self._check_array_element(1, expr.elements[0], first_type,
                                          final_elem_type)
                expr.elements[0] = widen_implicitly(expr.elements[0], first_type,
                                                    final_elem_type)

            for i, elem in enumerate(expr.elements[1:], 2):
                elem_type = self.check_expression(elem, final_elem_type)
                self._check_array_element(i, elem, elem_type, final_elem_type)
                expr.elements[i - 1] = widen_implicitly(elem, elem_type, final_elem_type)
            array_type = ArrayTypeInfo(element_type=final_elem_type, size=len(expr.elements))
            expr.expr_type = array_type
            return array_type

        elif isinstance(expr, HIRStringLiteral):
            # String literal for byte array initialization
            return StringValidator.check_string_literal(expr, context_type)

        elif isinstance(expr, HIRStructLiteralExpr):
            # Struct literal: Player { x: 10, y: 20, health: 100 }
            return self.struct_validator.check_struct_literal(expr)

        elif isinstance(expr, HIRMatchExpression):
            return self.match_validator.check_match_expression(expr, context_type)

        elif isinstance(expr, HIRBlockExpression):
            return self.check_block_expression(expr, context_type)

        elif isinstance(expr, HIRIfExpression):
            return self.check_if_expression(expr, context_type)

        elif isinstance(expr, HIRLoopExpression):
            return self.check_loop_expression(expr, context_type)

        else:
            raise TypeCheckError(
                f"Unknown expression type: {type(expr).__name__}",
                source_loc=expr.source_loc
            )

    def _check_overloaded_comparison(self, expr: HIRBinaryOp, left_type, right_type) -> TypeInfo:
        """Rewrite an aggregate comparison `a OP b` into a primitive compare of a
        PartialEq::eq / PartialOrd::cmp call result against 0 (or 1)."""
        from r65.compiler.hir.lang_items import EQ_TRAIT, EQ_METHOD, ORD_TRAIT, ORD_METHOD
        from r65.compiler.hir.nodes import (
            HIRFunctionCall, HIRFieldAccess, HIRAddressOf, HIRIntegerLiteral)

        if not (isinstance(left_type, StructTypeInfo) and isinstance(right_type, StructTypeInfo)
                and left_type.name == right_type.name):
            raise TypeCheckError(
                f"cannot compare '{left_type}' with '{right_type}'",
                source_loc=expr.source_loc,
                hint="overloaded comparison requires both operands to be the same struct type"
            )
        struct_name = left_type.name
        op = expr.op
        if op in ('==', '!='):
            trait, method = EQ_TRAIT, EQ_METHOD          # eq(*self, other) -> bool
            cmp_op = '=='
            literal_val, literal_type, ret = (1 if op == '==' else 0), 'bool', 'bool'
        else:
            trait, method = ORD_TRAIT, ORD_METHOD        # cmp(*self, other) -> i8 (sign)
            cmp_op = op
            literal_val, literal_type, ret = 0, 'i8', 'i8'

        if self.symbol_table.lookup(f"{struct_name}.{method}") is None:
            raise TypeCheckError(
                f"type '{struct_name}' does not implement '{op}' (trait {trait})",
                source_loc=expr.source_loc,
                hint=f"add `impl {trait} for {struct_name} {{ fn {method}(*self, other: *{struct_name}) -> {ret} {{ ... }} }}`"
            )

        arg = HIRAddressOf(operand=expr.right, source_loc=getattr(expr.right, 'source_loc', None))
        call = HIRFunctionCall(
            func=HIRFieldAccess(base=expr.left, field_name=method, source_loc=expr.source_loc),
            args=[arg],
            source_loc=expr.source_loc,
        )
        self.call_validator.check_function_call(call)  # sets method_call_info + expr_type

        # Rewrite this node in place into `call cmp_op <literal>` so the existing
        # primitive comparison lowering/codegen handles it (signed for cmp's i8).
        lit = HIRIntegerLiteral(value=literal_val, source_loc=expr.source_loc)
        lit.expr_type = BasicTypeInfo(literal_type)
        expr.left = call
        expr.right = lit
        expr.op = cmp_op
        expr.expr_type = BasicTypeInfo('bool')
        return expr.expr_type

    # Operators whose result is a value rather than a bool. A newtype operand
    # makes the result that same newtype; comparisons still yield bool.
    _VALUE_PRODUCING_OPS = ('+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>')

    def check_binary_op(self, expr: HIRBinaryOp, context_type: Optional[TypeInfo] = None) -> TypeInfo:
        """Type check binary operation."""
        # The destination type is consulted only for machine-width decisions
        # (literal inference, and the shift-widening guard below), so a newtype
        # destination answers for its payload. Leave it wrapped and the widening
        # guard silently fails its `isinstance(..., BasicTypeInfo)` test, which
        # computes `n << 2` in m8 and stores a stale high byte into 2 bytes.
        if context_type is not None:
            context_type = strip_newtype(context_type)
        # Propagate context type ONLY for shift operators — needed so `const X: u16 = 0 << 2`
        # infers `0` as u16. Do NOT propagate for arithmetic/bitwise (+, -, *, etc.) because
        # intermediate values commonly exceed the target type (e.g., `let x: u8 = 256 - 200`).
        left_context = context_type if expr.op in ('<<', '>>') else None
        left_type = self.check_expression(expr.left, left_context)
        # For comparison operators, propagate left operand's type as context for right operand
        # This ensures `off >= 32 * 32` (where off is u16) evaluates 32*32 as u16 (1024), not u8 (0)
        right_context = left_type if expr.op in ['==', '!=', '<', '<=', '>', '>='] else None
        right_type = self.check_expression(expr.right, right_context)

        # A newtype inherits its payload's operators. Check the operation against
        # the payload types, then re-wrap the result so it stays nominal:
        # `TileId + 1` is a TileId, never a bare u8.
        result_newtype = self._newtype_binop_result(expr, left_type, right_type)
        if result_newtype is not None:
            result = self._check_binary_op_typed(
                expr, strip_newtype(left_type), strip_newtype(right_type), context_type)
            if expr.op in self._VALUE_PRODUCING_OPS:
                expr.expr_type = result_newtype
                return result_newtype
            return result

        return self._check_binary_op_typed(expr, left_type, right_type, context_type)

    def _newtype_binop_result(self, expr: HIRBinaryOp,
                              left_type: TypeInfo, right_type: TypeInfo) -> Optional[TypeInfo]:
        """The newtype an operator's result carries, or None if neither side is one.

        `NT op NT` and `NT op <payload>` both yield NT. Two *different* newtypes
        never mix — that is the opacity the type exists for.
        """
        left_nt = left_type if isinstance(left_type, NewtypeTypeInfo) else None
        right_nt = right_type if isinstance(right_type, NewtypeTypeInfo) else None
        if left_nt is None and right_nt is None:
            return None

        if (left_nt is not None and right_nt is not None
                and left_nt.newtype_name != right_nt.newtype_name):
            raise TypeCheckError(
                f"operator '{expr.op}' has mismatched types '{left_nt}' and '{right_nt}'",
                source_loc=expr.source_loc,
                hint=f"newtypes never mix; unwrap one side explicitly "
                     f"(e.g. '{right_nt}(lhs.0 {expr.op} rhs.0)')"
            )

        # `&&`/`||` are bool-only and never inherited.
        if expr.op in ('&&', '||'):
            return None

        return left_nt or right_nt

    @staticmethod
    def _widen_operands(expr: HIRBinaryOp, left_type: TypeInfo,
                        right_type: TypeInfo, promoted_type: TypeInfo) -> None:
        """Bring both operands of `expr` to `promoted_type`.

        A narrower operand is wrapped in an implicit cast; an integer literal is
        just retyped, since it widens at compile time.
        """
        for side, side_type in (('left', left_type), ('right', right_type)):
            if TypeUtils.types_equal(side_type, promoted_type):
                continue
            operand = getattr(expr, side)
            if isinstance(operand, HIRIntegerLiteral):
                operand.expr_type = promoted_type
                continue
            cast_node = HIRTypeCast(
                expr=operand,
                target_type=promoted_type,
                source_loc=operand.source_loc
            )
            cast_node.expr_type = promoted_type
            setattr(expr, side, cast_node)

    def _check_binary_op_typed(self, expr: HIRBinaryOp, left_type: TypeInfo,
                               right_type: TypeInfo,
                               context_type: Optional[TypeInfo] = None) -> TypeInfo:
        """Type check a binary operation given already-computed operand types."""
        operands_aggregate = (TypeUtils.is_aggregate_type(left_type)
                              or TypeUtils.is_aggregate_type(right_type))

        # Operator overloading (Tier B): comparisons on aggregate operands dispatch
        # to PartialEq::eq (== / !=) or PartialOrd::cmp (< <= > >=).
        if expr.op in ('==', '!=', '<', '<=', '>', '>=') and operands_aggregate:
            return self._check_overloaded_comparison(expr, left_type, right_type)

        # E-OVL-003: value-producing arithmetic/bitwise/shift on an aggregate is not
        # supported. Only compound-assignment (`OP=`) and comparison overloads exist;
        # `let c = a + b` would have to return a struct by value, which R65 forbids.
        if expr.op in ('+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>') and operands_aggregate:
            agg = left_type if TypeUtils.is_aggregate_type(left_type) else right_type
            raise TypeCheckError(
                f"operator '{expr.op}' is not overloadable for aggregate type '{agg}'",
                source_loc=expr.source_loc,
                hint=f"value-producing operators on structs are not supported; "
                     f"use the in-place form '{expr.op}=' instead"
            )

        # Validate primitive operator restrictions (power-of-2 mul/div, constant
        # shift). Skipped for aggregate operands (handled above) and for const fns
        # (evaluated at compile time).
        if not (self.current_function and self.current_function.is_const):
            OperatorValidator.validate_binary_op(expr)

        # Type rules for binary operators
        if expr.op in ['<<', '>>']:
            # Shift operators: left operand is value, right is shift amount
            # Shift amount can be any integer type (typically u8)
            # Result type is same as left operand
            if not TypeUtils.is_integer_type(left_type):
                raise TypeCheckError(
                    f"shift operator '{expr.op}' requires integer operand, found {left_type}",
                    source_loc=expr.source_loc,
                    hint="only integer types (u8, i8, u16, i16) can be shifted"
                )
            if not TypeUtils.is_integer_type(right_type):
                raise TypeCheckError(
                    f"shift amount must be an integer, found {right_type}",
                    source_loc=expr.source_loc,
                    hint="shift amount should be a constant like 1, 2, 4, etc."
                )
            # Promote unsuffixed literals if compile-time result overflows
            if (isinstance(expr.left, HIRIntegerLiteral) and expr.left.suffix is None
                    and isinstance(expr.right, HIRIntegerLiteral)
                    and isinstance(left_type, BasicTypeInfo)):
                if expr.op == '<<':
                    result_val = expr.left.value << expr.right.value
                else:  # >>
                    result_val = expr.left.value >> expr.right.value
                if not value_fits_type(result_val, left_type.name):
                    promoted_name = 'i16' if left_type.name in ('i8', 'i16') else 'u16'
                    promoted_type = BasicTypeInfo(name=promoted_name)
                    expr.left.expr_type = promoted_type
                    left_type = promoted_type

            # Widen a non-literal left operand to the destination type when
            # the assignment context is wider. Without this, the canonical
            # `let base: u16 = n << 2;` (n is u8) computes the shift in m8
            # and truncates for n >= 64 — the user wrote a u16 destination
            # exactly to avoid that. Mirrors the runtime widening that array
            # indexing already does in compute_scaled_index (mir/lowerers/multiply.py).
            if (context_type is not None
                    and isinstance(context_type, BasicTypeInfo)
                    and isinstance(left_type, BasicTypeInfo)
                    and TypeUtils.is_integer_type(context_type)
                    and TypeUtils.is_integer_type(left_type)
                    and not isinstance(expr.left, HIRIntegerLiteral)):
                ctx_size = 2 if context_type.name in ('u16', 'i16') else 1
                left_size = 2 if left_type.name in ('u16', 'i16') else 1
                if ctx_size > left_size:
                    cast_node = HIRTypeCast(
                        expr=expr.left,
                        target_type=context_type,
                        source_loc=expr.left.source_loc
                    )
                    cast_node.expr_type = context_type
                    expr.left = cast_node
                    left_type = context_type

            expr.expr_type = left_type
            return left_type

        elif expr.op in ['+', '-', '*', '/', '%', '&', '|', '^']:
            # Check for pointer arithmetic: pointer + integer or pointer - integer
            if expr.op in ['+', '-']:
                if isinstance(left_type, PointerTypeInfo) and TypeUtils.is_integer_type(right_type):
                    # pointer + int = pointer (result is same pointer type)
                    expr.expr_type = left_type
                    return left_type
                if isinstance(right_type, PointerTypeInfo) and TypeUtils.is_integer_type(left_type):
                    # int + pointer = pointer (result is same pointer type)
                    expr.expr_type = right_type
                    return right_type

            # Arithmetic and bitwise: operands must match or be implicitly promotable
            if not TypeUtils.types_equal(left_type, right_type):
                # Check for implicit integer promotion (u8 -> u16, i8 -> i16)
                promoted_type = self._get_promoted_type(left_type, right_type)
                if promoted_type is None:
                    raise TypeCheckError(
                        f"type mismatch in '{expr.op}' operation: {left_type} vs {right_type}",
                        source_loc=expr.source_loc,
                        hint=f"cast one operand to match: (value as {left_type})"
                    )
                self._widen_operands(expr, left_type, right_type, promoted_type)
                # Use the promoted type as the result
                expr.expr_type = promoted_type
                return promoted_type

            # Result is same type - but check for overflow with unsuffixed literals
            result_type = left_type
            if (expr.op in ['+', '-', '*']
                    and isinstance(expr.left, HIRIntegerLiteral) and expr.left.suffix is None
                    and isinstance(expr.right, HIRIntegerLiteral) and expr.right.suffix is None
                    and isinstance(result_type, BasicTypeInfo)):
                if expr.op == '+':
                    result_val = expr.left.value + expr.right.value
                elif expr.op == '-':
                    result_val = expr.left.value - expr.right.value
                else:  # *
                    result_val = expr.left.value * expr.right.value
                if not value_fits_type(result_val, result_type.name):
                    promoted_name = 'i16' if result_type.name in ('i8', 'i16') else 'u16'
                    result_type = BasicTypeInfo(name=promoted_name)
                    expr.left.expr_type = result_type
                    expr.right.expr_type = result_type

            expr.expr_type = result_type
            return result_type

        elif expr.op in ['==', '!=', '<', '<=', '>', '>=']:
            # Comparison: operands must be compatible, result is bool
            if not self.relations.types_compatible(left_type, right_type):
                raise TypeCheckError(
                    f"cannot compare {left_type} with {right_type}",
                    source_loc=expr.source_loc,
                    hint="comparison requires compatible types"
                )

            # Promote mismatched integer types (e.g., u16 >= u8 -> u16 >= u16)
            if not TypeUtils.types_equal(left_type, right_type):
                promoted_type = self._get_promoted_type(left_type, right_type)
                if promoted_type is not None:
                    self._widen_operands(expr, left_type, right_type, promoted_type)

            # Check for invalid index register comparison (X vs Y)
            # There's no direct CPX Y or CPY X instruction
            self._validate_comparison_operands(expr)

            expr.expr_type = BasicTypeInfo('bool')
            return expr.expr_type

        elif expr.op in ['&&', '||']:
            # Logical: operands must be bool
            self._require_boolean_type(left_type, f"Left operand of '{expr.op}'", expr.left.source_loc)
            self._require_boolean_type(right_type, f"Right operand of '{expr.op}'", expr.right.source_loc)

            expr.expr_type = BasicTypeInfo('bool')
            return expr.expr_type

        else:
            raise TypeCheckError(
                f"Unknown binary operator: {expr.op}",
                source_loc=expr.source_loc
            )

    def check_unary_op(self, expr: HIRUnaryOp, context_type: Optional[TypeInfo] = None) -> TypeInfo:
        """Type check unary operation."""
        # As in check_binary_op, the destination is consulted only for literal
        # width/signedness, so a newtype answers for its payload. Leave it
        # wrapped and `let q: Q10 = -32768;` is rejected as not fitting i16 —
        # a false rejection of the exact literal the escape hatch below exists for.
        if context_type is not None:
            context_type = strip_newtype(context_type)
        if expr.op == '!':
            # Logical NOT: operand must be bool (no context propagation needed)
            operand_type = self.check_expression(expr.operand)
            if not TypeUtils.is_boolean_type(operand_type):
                raise TypeCheckError(
                    f"logical NOT '!' requires boolean operand, found {operand_type}",
                    source_loc=expr.operand.source_loc,
                    hint=(self._bool_payload_hint(operand_type)
                          or "use comparison like (value != 0) to convert to bool")
                )
            expr.expr_type = BasicTypeInfo('bool')

        elif expr.op == '~':
            # Bitwise NOT: propagate context type to operand
            operand_type = self.check_expression(expr.operand, context_type)
            # A newtype inherits its payload's unary operators just as it does
            # the binary ones; `expr_type` stays the newtype so `~t` is a TileId.
            if not TypeUtils.is_integer_type(strip_newtype(operand_type)):
                raise TypeCheckError(
                    f"bitwise NOT '~' requires integer operand, found {operand_type}",
                    source_loc=expr.operand.source_loc,
                    hint="only integer types (u8, i8, u16, i16) support bitwise operations"
                )
            expr.expr_type = operand_type

        elif expr.op == '-':
            # Special case: -MIN_VALUE for signed types (e.g., -128 for i8, -32768 for i16).
            # The parser represents -128 as UnaryOp(-, IntegerLiteral(128)), but 128 doesn't
            # fit in i8. Pre-set the operand's type to bypass the overflow check.
            if (isinstance(expr.operand, HIRIntegerLiteral)
                    and expr.operand.expr_type is None
                    and expr.operand.suffix is None
                    and context_type is not None
                    and isinstance(context_type, BasicTypeInfo)
                    and context_type.name in ('i8', 'i16')):
                from r65.compiler.typeck.type_utils import get_type_range
                range_info = get_type_range(context_type.name)
                if range_info is not None and expr.operand.value == -range_info[0]:
                    expr.operand.expr_type = context_type
            # Negation: propagate context type to operand for proper literal typing
            operand_type = self.check_expression(expr.operand, context_type)
            if not TypeUtils.is_integer_type(strip_newtype(operand_type)):
                raise TypeCheckError(
                    f"negation '-' requires integer operand, found {operand_type}",
                    source_loc=expr.operand.source_loc,
                    hint="only integer types can be negated"
                )
            # Overflow check for negated literals: e.g., `let x: u8 = -1`.
            # Only applies when the literal itself fits operand_type (so it's a
            # clean negation). Wide literals like -100000 flow through bitwise
            # masks in stdlib macros (I32!, U32!) and are handled downstream.
            operand_payload = strip_newtype(operand_type)
            if (isinstance(expr.operand, HIRIntegerLiteral)
                    and isinstance(operand_payload, BasicTypeInfo)):
                from r65.compiler.typeck.type_utils import get_type_range, value_fits_type
                range_info = get_type_range(operand_payload.name)
                if range_info is not None and value_fits_type(expr.operand.value, operand_payload.name):
                    neg_val = -expr.operand.value
                    if not (range_info[0] <= neg_val <= range_info[1]):
                        # Try signed promotion when no explicit context was given
                        promoted = False
                        if context_type is None and operand_payload.name in ('u8', 'u16'):
                            for signed_name in ('i8', 'i16'):
                                signed_range = get_type_range(signed_name)
                                if signed_range is not None and signed_range[0] <= neg_val <= signed_range[1]:
                                    operand_type = BasicTypeInfo(signed_name)
                                    expr.operand.expr_type = operand_type
                                    promoted = True
                                    break
                        if not promoted:
                            raise TypeCheckError(
                                f"integer literal {neg_val} does not fit in type {operand_payload.name} "
                                f"(valid range: {range_info[0]} to {range_info[1]})",
                                source_loc=expr.source_loc,
                                hint=f"use a signed type (i8/i16) for negative values"
                            )
            expr.expr_type = operand_type

        else:
            raise TypeCheckError(
                f"unknown unary operator: {expr.op}",
                source_loc=expr.source_loc
            )

        return expr.expr_type

    def check_type_cast(self, expr: HIRTypeCast) -> TypeInfo:
        """Type check explicit cast, newtype construction, or payload access."""
        if expr.newtype_construct:
            return self._check_newtype_construct(expr)

        source_type = self.check_expression(expr.expr)

        if expr.newtype_field is not None:
            return self._check_newtype_field(expr, source_type)

        target_type = expr.target_type

        self._reject_dyn_cast_of_newtype(expr, source_type, target_type)

        if not TypeUtils.can_cast(source_type, target_type):
            raise TypeCheckError(
                f"cannot cast {source_type} to {target_type}",
                source_loc=expr.source_loc,
                hint="casts are only allowed between compatible types (integers, bools)"
            )

        expr.expr_type = target_type
        return target_type

    def _reject_dyn_cast_of_newtype(self, expr, source_type, target_type):
        """Reject `&newtype as *dyn Trait`.

        Dynamic dispatch reads a TypeId byte at offset 0 of the pointee to select
        an implementation. A newtype has no such byte — every byte it has is
        payload — so a dyn pointer to one would dispatch on the value itself.

        The implicit coercion already declines this, requiring a struct pointee,
        but an explicit cast would otherwise walk straight past it. A newtype may
        still implement the trait; only the dyn route is closed.
        """
        from r65.compiler.hir.types import TraitTypeInfo

        if not (isinstance(target_type, PointerTypeInfo)
                and isinstance(target_type.pointee_type, TraitTypeInfo)):
            return
        pointee = (source_type.pointee_type
                   if isinstance(source_type, PointerTypeInfo) else source_type)
        if not isinstance(pointee, NewtypeTypeInfo):
            return
        raise TypeCheckError(
            f"cannot form a '*dyn {target_type.pointee_type.name}' over newtype "
            f"'{pointee.newtype_name}'",
            source_loc=expr.source_loc,
            hint="dynamic dispatch reads a TypeId byte at offset 0, and a newtype "
                 "is all payload; call the method directly on the newtype instead"
        )

    def _check_newtype_construct(self, expr: HIRTypeCast) -> TypeInfo:
        """Type check `Newtype(x)`.

        Checked as an assignment into the payload, not as a cast: `TileId(300)`
        is rejected for the same reason `let t: TileId = 300;` is. Truncating
        stays possible, but has to be spelled `300 as TileId` — one operation,
        one meaning.
        """
        target_type = expr.target_type
        payload = strip_newtype(target_type)

        # Checking against the payload gives the literal its range check.
        source_type = self.check_expression(expr.expr, payload)

        if not self.relations.assignable(source_type, target_type):
            raise TypeCheckError(
                f"cannot make a '{target_type}' from '{source_type}'",
                source_loc=expr.source_loc,
                hint=f"'{target_type}' wraps '{payload}'; convert explicitly first "
                     f"(value as {payload}), or cast the whole value "
                     f"(value as {target_type}) to truncate"
            )
        expr.expr = widen_implicitly(expr.expr, source_type, payload)

        expr.expr_type = target_type
        return target_type

    def _check_newtype_field(self, expr: HIRTypeCast, source_type: TypeInfo) -> TypeInfo:
        """Type check `t.0`, filling in the retype the HIR builder left open."""
        index = expr.newtype_field

        if not isinstance(source_type, NewtypeTypeInfo):
            raise TypeCheckError(
                f"'{source_type}' is not a newtype, so it has no field '.{index}'",
                source_loc=expr.source_loc,
                hint="access struct and union fields by name (e.g. '.value')"
            )

        if index != 0:
            raise TypeCheckError(
                f"newtype '{source_type}' has only field '.0'",
                source_loc=expr.source_loc,
                hint="a newtype wraps exactly one value, so '.0' is the only "
                     "field there is"
            )

        expr.target_type = source_type.inner
        expr.expr_type = source_type.inner
        return source_type.inner


    def _check_const_index(self, expr: HIRArrayIndex, size: int):
        """Reject a constant index outside `0..size`. A dynamic one is not checked."""
        index_ast = expr.original_ast.index if expr.original_ast else None
        if index_ast is None or not self.const_evaluator.is_constant(index_ast):
            return
        try:
            index_value = self.const_evaluator.eval(index_ast)
        except HIRError:
            return
        if 0 <= index_value < size:
            return
        what = ("(negative index)" if index_value < 0
                else f"for array of size {size}")
        raise TypeCheckError(
            f"array index {index_value} is out of bounds {what}",
            source_loc=expr.index.source_loc,
            hint=f"valid indices are 0 to {size - 1}"
        )

    def check_element_base(self, expr: HIRExpression) -> TypeInfo:
        """Type check the base of a field access or the receiver of a method.

        That base is the one position where a whole #[soa] element `ARR[i]` is
        legal (`ARR[i].field`, `ARR[i].m()`); everywhere else
        check_array_index rejects it. Only `expr` itself gets the permission,
        so in `ARR[OTHER[k]].hp` the inner index is still checked normally.
        """
        if isinstance(expr, HIRArrayIndex):
            return self.check_array_index(expr, soa_element_ok=True)
        return self.check_expression(expr)

    def check_array_index(self, expr: HIRArrayIndex, *, soa_element_ok: bool = False) -> TypeInfo:
        """Type check array indexing.

        `soa_element_ok` is set only by check_element_base.
        """
        soa_decl = soa_static_of(expr.array)
        if soa_decl is not None and not soa_element_ok:
            # A tagged enum is planed the same way and lands here too, but it
            # has no fields to steer the author towards — what it has is a tag
            # and a payload, reachable by `match` and by a cast to an integer.
            element = getattr(soa_decl.var_type, 'element_type', soa_decl.var_type)
            if isinstance(element, EnumTypeInfo):
                raise TypeCheckError(
                    f"cannot use a whole element of tagged enum array "
                    f"'{soa_decl.name}'",
                    source_loc=expr.source_loc,
                    hint="a tagged enum element is a tag plus a payload in two "
                         "separate planes, with no contiguous value; `match "
                         f"{soa_decl.name}[i]` to reach the payload, or "
                         f"`{soa_decl.name}[i] as u8` for the discriminant"
                )
            raise TypeCheckError(
                f"cannot use a whole element of #[soa] array '{soa_decl.name}'",
                source_loc=expr.source_loc,
                hint="an SoA element is spread across one array per field, so it "
                     "has no address and no contiguous value; read or write a "
                     f"single field instead, e.g. `{soa_decl.name}[i].field`"
            )

        base_type = self.check_expression(expr.array)
        index_type = self.check_expression(expr.index)

        # Index must be integer
        self._require_integer_type(index_type, "Array index", expr.index.source_loc)

        # Handle pointer types: (*ptr: T)[idx] or (far *ptr: T)[idx]
        if isinstance(base_type, PointerTypeInfo):
            pointee = base_type.pointee_type
            # If pointer to array, result is element type
            if isinstance(pointee, ArrayTypeInfo):
                self._check_const_index(expr, pointee.size)
                expr.expr_type = pointee.element_type
                return expr.expr_type
            else:
                # Pointer to non-array: ptr[idx] is pointer arithmetic
                expr.expr_type = pointee
                return expr.expr_type

        # Array indexing: array[idx]
        if isinstance(base_type, ArrayTypeInfo):
            self._check_const_index(expr, base_type.size)

            # Result is element type
            expr.expr_type = base_type.element_type
            return expr.expr_type

        raise TypeCheckError(
            f"cannot index type {base_type}",
            source_loc=expr.array.source_loc,
            hint="indexing requires an array type or pointer"
        )

    def check_field_access(self, expr: HIRFieldAccess) -> TypeInfo:
        """Type check field access."""
        base_type = self.check_element_base(expr.base)

        # Auto-dereference pointers for field access (like C's `->`)
        # This allows `self.field` to work when self is *StructName
        from r65.compiler.hir import StructTypeInfo
        if isinstance(base_type, PointerTypeInfo):
            if isinstance(base_type.pointee_type, StructTypeInfo):
                # Mark that this access is through a pointer (for codegen)
                expr.auto_deref = True
                base_type = base_type.pointee_type
            else:
                raise TypeCheckError(
                    f"cannot access field '{expr.field_name}' on pointer to non-struct type {base_type.pointee_type}",
                    source_loc=expr.base.source_loc,
                    hint="pointer field access requires pointer to struct type"
                )

        # Base must be struct type
        if not isinstance(base_type, StructTypeInfo):
            raise TypeCheckError(
                f"cannot access field '{expr.field_name}' on type {base_type}",
                source_loc=expr.base.source_loc,
                hint="field access requires a struct type"
            )

        # HIR building points every StructTypeInfo at its HIR declaration.
        struct_def = base_type.definition
        if struct_def is None:
            raise TypeCheckError(
                f"struct '{base_type.name}' definition not found",
                source_loc=expr.source_loc,
                hint="ensure the struct is defined before use"
            )

        field = next((f for f in struct_def.fields if f.name == expr.field_name), None)

        if field is None:
            # Get available field names for hint (exclude synthetic __type_id)
            available_fields = [f.name for f in struct_def.fields if not f.name.startswith('__')]
            hint = f"available fields: {', '.join(available_fields)}" if available_fields else None
            raise TypeCheckError(
                f"struct '{base_type.name}' has no field '{expr.field_name}'",
                source_loc=expr.source_loc,
                hint=hint
            )

        expr.expr_type = field.field_type
        expr.field_offset = field.offset
        return expr.expr_type

    def check_status_flag_access(self, expr: HIRStatusFlagAccess) -> TypeInfo:
        """Type check STATUS flag access (e.g., STATUS.Carry)."""
        # STATUS flags are always boolean
        expr.expr_type = BasicTypeInfo(name='bool')
        return expr.expr_type

    def _check_status_flag_assignment(self, expr: HIRAssignment) -> TypeInfo:
        """Type check STATUS flag assignment (e.g., STATUS.Carry = true)."""
        from r65.compiler.hir.status_flags import STATUS_FLAGS, get_status_flag

        target = expr.target
        flag = get_status_flag(target.flag_name)

        # Check if the flag is writable
        if not flag.is_writable:
            writable_flags = ", ".join(
                name for name, f in STATUS_FLAGS.items() if f.is_writable)
            raise TypeCheckError(
                f"Cannot write to STATUS.{target.flag_name}\n"
                f"  This flag is set by CPU operations, not directly writable\n"
                f"  Writable flags: {writable_flags}",
                source_loc=expr.source_loc
            )

        # Check value type is boolean
        value_type = self.check_expression(expr.value)
        if not TypeUtils.is_boolean_type(value_type):
            raise TypeCheckError(
                f"STATUS flag assignment requires boolean value, got '{value_type}'",
                source_loc=expr.value.source_loc
            )

        expr.expr_type = BasicTypeInfo(name='bool')
        return expr.expr_type

    def _check_overloaded_compound_assign(self, expr: HIRAssignment, target_type) -> TypeInfo:
        """Redirect `a OP= b` on an aggregate to `a.<op>_assign(&b)`."""
        from r65.compiler.hir.lang_items import BINOP_ASSIGN
        from r65.compiler.hir.nodes import HIRFunctionCall, HIRFieldAccess, HIRAddressOf

        op = expr.compound_op
        if op not in BINOP_ASSIGN:
            raise TypeCheckError(
                f"operator '{op}=' cannot be overloaded for '{target_type}'",
                source_loc=expr.source_loc
            )
        trait_name, method_name = BINOP_ASSIGN[op]

        if not isinstance(target_type, StructTypeInfo):
            raise TypeCheckError(
                f"'{op}=' is not supported for type '{target_type}'",
                source_loc=expr.source_loc
            )
        struct_name = target_type.name
        if self.symbol_table.lookup(f"{struct_name}.{method_name}") is None:
            raise TypeCheckError(
                f"type '{struct_name}' does not implement '{op}=' (trait {trait_name})",
                source_loc=expr.source_loc,
                hint=f"add `impl {trait_name} for {struct_name} {{ fn {method_name}(*self, other: *{struct_name}) {{ ... }} }}`"
            )

        # The compound-assign desugar set value = (target OP rhs); recover rhs.
        rhs = expr.value.right if isinstance(expr.value, HIRBinaryOp) else expr.value

        # E-OVL-002: the operator method takes the same struct by reference, so the
        # right operand must be that same struct. A scalar/literal (e.g. `score += 5`
        # or `score += n` where n is a u16, on a U32) is not addable through the
        # operator — reject it here with an actionable message instead of letting the
        # synthesized `&rhs` call fail with a confusing lower-level error.
        rhs_type = self.check_expression(rhs)
        if not (isinstance(rhs_type, StructTypeInfo) and rhs_type.name == struct_name):
            raise TypeCheckError(
                f"operator '{op}=' on '{struct_name}' expects a '{struct_name}' "
                f"operand, found '{rhs_type}'",
                source_loc=expr.source_loc,
                hint=f"the right side of '{op}=' must be a '{struct_name}'; widen the "
                     f"value to '{struct_name}' first, or call a type-specific method "
                     f"directly (e.g. a scalar add/sub helper)"
            )

        arg = HIRAddressOf(operand=rhs, source_loc=getattr(rhs, 'source_loc', None))
        call = HIRFunctionCall(
            func=HIRFieldAccess(base=expr.target, field_name=method_name,
                                source_loc=expr.source_loc),
            args=[arg],
            source_loc=expr.source_loc,
        )
        # Resolves via _try_method_call -> sets call.method_call_info.
        self.call_validator.check_function_call(call)
        expr.opassign_call = call
        expr.expr_type = BasicTypeInfo('void')
        return expr.expr_type

    def check_assignment(self, expr: HIRAssignment) -> TypeInfo:
        """Type check assignment."""
        # Special handling for STATUS flag assignments
        if isinstance(expr.target, HIRStatusFlagAccess):
            return self._check_status_flag_assignment(expr)

        # Mutability check: reject assignments to immutable variables
        self._check_target_mutable(expr.target, expr.source_loc)

        # A and B are the two halves of one accumulator: the only way to move
        # B into A is XBA, which also moves A into B, so `A = B` cannot keep
        # B's value.
        if (isinstance(expr.target, HIRRegister) and expr.target.name == 'A'
                and isinstance(expr.value, HIRRegister) and expr.value.name == 'B'):
            raise TypeCheckError(
                "`A = B` is not supported: the XBA it needs also overwrites B",
                source_loc=expr.source_loc,
                hint="copy through a variable (`let hi: u8 = B; A = hi;`), or "
                     "swap the two halves explicitly with asm!(\"XBA\")")

        target_type = self.check_expression(expr.target)

        # Operator overloading (Tier A): an aggregate compound assignment
        # `a OP= b` is redirected to the operator-trait method `a.<op>_assign(&b)`
        # instead of the by-value primitive path (which structs cannot take).
        if getattr(expr, 'compound_op', None) and TypeUtils.is_aggregate_type(target_type):
            return self._check_overloaded_compound_assign(expr, target_type)

        # For register targets (A, X, Y, B, aliases), the compiler auto-widens the
        # operation (e.g., REP #$20 for 16-bit A). Do not propagate target_type as
        # context to avoid spurious overflow errors for constants like `A = 0x1234`.
        target_register = self._get_target_register(expr.target)
        value_context = None if target_register else target_type
        # `dst = src.clone()` is the sanctioned aggregate-copy assignment.
        value_type = self._check_copy_value(
            expr.value, value_context,
            clone_ok=TypeUtils.is_aggregate_type(target_type))

        # Validate register-specific operator restrictions
        # If target is a register (or register-aliased) and value is a binary op using that register,
        # validate that the register supports the operation
        if target_register and self._binary_op_uses_target(expr.value, expr.target):
            binary_op = expr.value
            OperatorValidator.validate_register_binary_op(
                op=binary_op.op,
                target_register=target_register,
                right_operand=binary_op.right,
                source_loc=expr.source_loc
            )

        # Validate register-to-register transfers
        # Some register pairs don't have direct transfer instructions (e.g., D to X)
        source_register = self._get_target_register(expr.value)
        if target_register and source_register:
            if not can_transfer_directly(source_register, target_register):
                hint = get_transfer_error_hint(source_register, target_register)
                raise TypeCheckError(
                    f"cannot transfer {source_register} to {target_register} directly",
                    source_loc=expr.source_loc,
                    hint=hint
                )

        # Arrays and structs are not copied by a bare `=` (copy cost stays
        # explicit). Point at `Clone`, the sanctioned aggregate-copy primitive.
        if TypeUtils.is_aggregate_type(target_type):
            # `dst = src.clone()` — sanctioned aggregate copy. The value carries
            # clone_info; MIR lowers it to an AggregateCopy into the target.
            if getattr(expr.value, 'clone_info', None):
                expr.expr_type = target_type
                return target_type
            type_name = str(target_type)
            if isinstance(target_type, StructTypeInfo):
                clone_hint = (
                    f"use `dst = src.clone()` or `dst.clone_from(&src)` "
                    f"(add `impl Clone for {target_type.name} {{}}` if the struct has none)")
            else:  # array — clone is a built-in, no impl needed
                clone_hint = "use `dst = src.clone()` or `dst.clone_from(&src)` (arrays clone built-in)"
            raise TypeCheckError(
                f"Cannot assign '{type_name}' by value\n"
                f"  structs and arrays are not copied by a bare '='\n"
                f"  Suggestion: {clone_hint}, or copy fields/elements individually",
                source_loc=expr.source_loc
            )

        # Handle tuple-to-scalar: A = tuple_func() drops extra return values
        if isinstance(value_type, MultiReturnTypeInfo) and not isinstance(target_type, MultiReturnTypeInfo):
            first_elem_type = value_type.element_types[0]
            self._check_type_match(
                target_type, first_elem_type, expr.value,
                "assignment (first element of tuple)", expr.source_loc, use_compatible=True
            )
            expr.expr_type = target_type
            return target_type

        # Register A mode switching: the value's type drives A's mode.
        # A = u16_expr → switch to m16; A = u8_expr → switch to m8.
        # Exception: register-bound variables (let v @ A: u16) keep their
        # declared type — only bare 'A' assignments trigger mode switching.
        if (target_register == 'A' and isinstance(expr.target, HIRRegister)
                and TypeUtils.is_integer_type(value_type)):
            expr.expr_type = value_type
            return value_type

        # Types must be compatible (allows enum/integer interop)
        self._check_type_match(
            target_type, value_type, expr.value,
            "assignment", expr.source_loc, use_compatible=True
        )
        # A move into X/Y already zero-extends a one-byte source in codegen,
        # without the cast's round trip through memory.
        if target_register not in ('X', 'Y'):
            expr.value = widen_implicitly(expr.value, value_type, target_type)

        expr.expr_type = target_type
        return target_type

    def check_multi_assignment(self, expr: HIRMultiAssignment) -> TypeInfo:
        """Type check multi-assignment (tuple destructuring).

        Handles: (A, X) = func() where func returns a tuple.
        """
        # Type check the value expression (should return a tuple)
        value_type = self.check_expression(expr.value)

        # Value must be a tuple type
        if not isinstance(value_type, MultiReturnTypeInfo):
            raise TypeCheckError(
                f"Multi-assignment requires a tuple value, got '{value_type}'",
                source_loc=expr.source_loc
            )

        # Number of targets must not exceed number of tuple elements
        # Partial assignment is allowed: (A,) = func() discards extra elements
        num_targets = len(expr.targets)
        num_elements = len(value_type.element_types)
        if num_targets > num_elements:
            raise TypeCheckError(
                f"Multi-assignment has {num_targets} targets but value only has {num_elements} elements",
                source_loc=expr.source_loc
            )

        # Type check each target against corresponding tuple element
        for i, (target, elem_type) in enumerate(zip(expr.targets, value_type.element_types)):
            target_type = self.check_expression(target)

            # Arrays and structs cannot be assigned by value
            if TypeUtils.is_aggregate_type(target_type):
                type_name = str(target_type)
                raise TypeCheckError(
                    f"Cannot assign '{type_name}' by value in multi-assignment\n"
                    f"  Arrays and structs cannot be copied by value",
                    source_loc=expr.source_loc
                )

            # Types must be compatible
            self._check_type_match(
                target_type, elem_type, target,
                f"multi-assignment element {i}", expr.source_loc, use_compatible=True
            )

        # Check for out-of-order register assignments
        self._check_tuple_register_order(expr)

        # The type of the expression is the tuple type
        expr.expr_type = value_type
        return value_type

    def check_block_expression(self, expr: HIRBlockExpression, context_type: Optional[TypeInfo] = None) -> TypeInfo:
        """Type check a block expression.

        Checks all statements in the block, then checks the final expression.
        The block's type is the type of the final expression.
        """
        # Check all statements
        for stmt in expr.statements:
            self.check_statement(stmt)

        # Check final expression - propagate context type for inference
        if expr.final_expr is not None:
            final_type = self.check_expression(expr.final_expr, context_type)
        else:
            # No tail expression: the block is void (it may or may not diverge)
            final_type = BasicTypeInfo(name='void')
        expr.expr_type = final_type
        return final_type

    def check_if_expression(self, expr: HIRIfExpression, context_type: Optional[TypeInfo] = None) -> TypeInfo:
        """Type check an if expression.

        Both branches must produce the same type.
        """
        # Check condition is boolean
        cond_type = self.check_expression(expr.condition)
        self._require_boolean_type(cond_type, "If expression condition", expr.condition.source_loc)

        # Check then branch
        then_type = self.check_expression(expr.then_block, context_type)

        # Check else branch (always present for if expressions)
        else_type = self.check_expression(expr.else_block, context_type or then_type)

        # Both branches must have the same type
        if not self.relations.types_compatible(then_type, else_type):
            raise TypeCheckError(
                f"if expression branches have different types: "
                f"then branch is {then_type}, else branch is {else_type}",
                source_loc=expr.source_loc,
                hint="both branches of an if expression must produce the same type"
            )

        expr.expr_type = then_type
        return then_type

    def check_loop_expression(self, expr: HIRLoopExpression, context_type: Optional[TypeInfo] = None) -> TypeInfo:
        """Type check a loop expression.

        Finds all break statements in the loop body (not nested loops),
        verifies they all have values, and that all value types are compatible.
        """
        # Check all statements in the body
        for stmt in expr.body.statements:
            self.check_statement(stmt)

        # Collect break value types from direct break statements
        break_types = []
        self._collect_break_types(expr.body, break_types, expr.label)

        if not break_types:
            raise TypeCheckError(
                "loop expression must have at least one break with a value",
                source_loc=expr.source_loc
            )

        # All break types must be compatible
        result_type = break_types[0]
        for i, bt in enumerate(break_types[1:], 1):
            if not self.relations.types_compatible(result_type, bt):
                raise TypeCheckError(
                    f"loop expression break values have different types: "
                    f"{result_type} vs {bt}",
                    source_loc=expr.source_loc,
                    hint="all break values in a loop expression must produce the same type"
                )

        expr.expr_type = result_type
        return result_type

    def _collect_break_types(self, node, break_types: list, loop_label: Optional[str],
                             nested: bool = False):
        """Collect the types of the break values that leave this loop.

        A break leaves it when it names its label, or is unlabeled and not
        inside a nested loop (an unlabeled break there leaves the inner
        loop). The walk goes everywhere a break can sit — if/else chains,
        match arms, blocks, nested loops — but stops at a nested loop with
        the same label, which shadows this one, and at declarations (a
        struct literal points back at its struct).
        """
        from dataclasses import fields, is_dataclass
        from r65.compiler.hir.nodes import (
            HIRNode, HIRDeclaration, HIRWhileStmt as _While,
            HIRLoopExpression as _LoopExpr,
        )
        if isinstance(node, list):
            for item in node:
                self._collect_break_types(item, break_types, loop_label, nested)
            return
        if not isinstance(node, HIRNode) or isinstance(node, HIRDeclaration):
            return
        if isinstance(node, HIRBreakStmt):
            if node.label == loop_label if node.label is not None else not nested:
                if node.value is None:
                    raise TypeCheckError(
                        "break in loop expression must have a value",
                        source_loc=node.source_loc
                    )
                break_types.append(node.value.expr_type)
            return
        if isinstance(node, (_While, _LoopExpr)):
            if loop_label is not None and node.label == loop_label:
                return
            nested = True
        if not is_dataclass(node):
            return
        for f in fields(node):
            child = getattr(node, f.name, None)
            if isinstance(child, (HIRNode, list)):
                self._collect_break_types(child, break_types, loop_label, nested)

    def _check_tuple_register_order(self, expr: HIRMultiAssignment):
        """
        Check that registers appearing in both return statement and assignment
        targets are at the same position.

        If a function returns (A, X) and we assign to (X, A), this creates a
        problematic swap situation. Enforce that overlapping registers must be
        at the same position.
        """
        # Get the function declaration if this is a direct function call
        if not isinstance(expr.value, HIRFunctionCall):
            return

        func_call = expr.value
        if not isinstance(func_call.func, HIRIdentifier):
            return  # Indirect call - can't check

        if func_call.func.symbol.kind != SymbolKind.FUNCTION:
            return

        func_decl = self._lookup_function_decl(func_call.func.symbol.name, expr.source_loc)
        if not func_decl:
            return

        # Extract register positions from return statements
        return_reg_positions = self._get_return_register_positions(func_decl)
        if not return_reg_positions:
            return  # No register returns to check

        # Extract register positions from assignment targets
        target_reg_positions = {}
        for i, target in enumerate(expr.targets):
            if isinstance(target, HIRRegister):
                target_reg_positions[target.name] = i

        # Check for conflicts: same register at different positions
        for reg_name, return_pos in return_reg_positions.items():
            if reg_name in target_reg_positions:
                target_pos = target_reg_positions[reg_name]
                if return_pos != target_pos:
                    raise TypeCheckError(
                        f"Register '{reg_name}' appears at position {return_pos} in return "
                        f"but position {target_pos} in assignment targets\n"
                        f"  This creates an impossible swap situation",
                        source_loc=expr.source_loc,
                        hint=f"Reorder assignment targets to match return order, "
                             f"or use intermediate variables"
                    )

    def _get_return_register_positions(self, func_decl: HIRFunctionDecl) -> dict:
        """
        Extract register positions from a function's return statements.

        Returns a dict mapping register name to position in return tuple.
        Only considers direct register returns (not expressions).
        """
        result = {}

        def visit_statement(stmt):
            if isinstance(stmt, HIRReturnStmt):
                for i, value in enumerate(stmt.values):
                    if isinstance(value, HIRRegister):
                        # Only record first occurrence of each register
                        if value.name not in result:
                            result[value.name] = i
            elif isinstance(stmt, HIRIfStmt):
                for s in stmt.then_block.statements:
                    visit_statement(s)
                if stmt.else_block:
                    for s in stmt.else_block.statements:
                        visit_statement(s)
            elif isinstance(stmt, HIRWhileStmt):
                for s in stmt.body.statements:
                    visit_statement(s)

        if func_decl.body:
            for stmt in func_decl.body.statements:
                visit_statement(stmt)

        return result
