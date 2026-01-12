"""
Main type checker for R65 compiler.

Orchestrates type checking, mode tracking, and validation.
"""

from typing import Optional
from r65.compiler.hir import (
    HIRProgram, HIRFunctionDecl, HIRExpression, HIRStatement,
    HIRBinaryOp, HIRUnaryOp, HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr,
    HIRIdentifier, HIRFunctionAddress, HIRRegister, HIRIncludeBytesExpr, HIRArrayFillExpr, HIRArrayLiteralExpr,
    HIRStringLiteral,
    HIRStructFieldInit, HIRStructLiteralExpr,
    HIRTypeCast, HIRFunctionCall,
    HIRMethodCall, HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf, HIRAssignment, HIRMultiAssignment,
    HIRLetStmt, HIRTupleLetStmt, HIRExprStmt, HIRReturnStmt, HIRIfStmt, HIRWhileStmt,
    HIRStaticDecl, HIRConstDecl, HIRTypeAlias,
    HIRMatchExpression, HIRPattern, HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern, HIRIdentifierPattern, HIROrPattern,
    BasicTypeInfo, TypeInfo, SymbolKind, NeverTypeInfo, TupleTypeInfo,
    RegisterLetBinding, ArrayTypeInfo, StructTypeInfo, EnumTypeInfo
)
from r65.compiler.hir.types import FunctionTypeInfo, PointerTypeInfo, SliceTypeInfo
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeTransition
from r65.compiler.typeck.mode_tracker import ModeTracker
from r65.compiler.typeck.cfg_builder import CFGBuilder
from r65.compiler.typeck.type_utils import TypeUtils
from r65.compiler.typeck.operator_validator import OperatorValidator
from r65.compiler.typeck.preservation_checker import PreservationChecker
from r65.compiler.typeck.type_inference import TypeInference
from r65.compiler.typeck.errors import TypeCheckError, TypeCheckWarning
from r65.compiler.typeck.string_validator import StringValidator
from r65.compiler.typeck.match_validator import MatchValidator
from r65.compiler.typeck.struct_validator import StructValidator
from r65.compiler.typeck.call_validator import CallValidator
from r65.compiler.typeck.pointer_validator import PointerValidator
from r65.compiler.hir.const_eval import ConstEvaluator


class TypeChecker:
    """
    Main type checker orchestrator.

    Multi-pass design:
    1. Build CFG for each function
    2. Perform mode analysis (track modes through CFG)
    3. Type check expressions (bottom-up with mode context)
    4. Validate operator restrictions
    5. Check register preservation
    """

    def __init__(self, program: HIRProgram):
        self.program = program
        self.symbol_table = program.symbol_table

        # Current context during type checking
        self.current_function: Optional[HIRFunctionDecl] = None
        self.current_mode: ProcessorMode = ProcessorMode.unknown()
        self.mode_tracker: Optional[ModeTracker] = None

        # Collect warnings during type checking
        self.warnings: list[TypeCheckWarning] = []

        # Const evaluator for bounds checking
        self.const_evaluator = ConstEvaluator(self.symbol_table)

        # Initialize validators (lazily created to avoid circular init)
        self._match_validator: Optional[MatchValidator] = None
        self._struct_validator: Optional[StructValidator] = None
        self._call_validator: Optional[CallValidator] = None
        self._pointer_validator: Optional[PointerValidator] = None

    @property
    def match_validator(self) -> MatchValidator:
        """Lazy initialization of match validator."""
        if self._match_validator is None:
            self._match_validator = MatchValidator(self.check_expression)
        return self._match_validator

    @property
    def struct_validator(self) -> StructValidator:
        """Lazy initialization of struct validator."""
        if self._struct_validator is None:
            self._struct_validator = StructValidator(
                self.symbol_table, self.const_evaluator,
                self.check_expression, self._check_type_match
            )
        return self._struct_validator

    @property
    def call_validator(self) -> CallValidator:
        """Lazy initialization of call validator."""
        if self._call_validator is None:
            self._call_validator = CallValidator(
                self.symbol_table, self._lookup_function_decl,
                self.check_expression, lambda: self.current_mode,
                lambda: self.current_function
            )
        return self._call_validator

    @property
    def pointer_validator(self) -> PointerValidator:
        """Lazy initialization of pointer validator."""
        if self._pointer_validator is None:
            self._pointer_validator = PointerValidator(self.check_expression)
        return self._pointer_validator

    def warn(self, message: str, source_loc=None):
        """Emit a type checking warning."""
        warning = TypeCheckWarning(message, source_loc)
        self.warnings.append(warning)

    # ========================================================================
    # Helper Methods
    # ========================================================================

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
        for decl in self.program.declarations:
            if isinstance(decl, HIRFunctionDecl) and decl.name == func_name:
                return decl

        raise TypeCheckError(
            f"Function '{func_name}' not found",
            source_loc=source_loc
        )

    def _get_mode_at(self, stmt_or_expr) -> ProcessorMode:
        """
        Get processor mode at a statement or expression.

        Args:
            stmt_or_expr: Statement or expression node

        Returns:
            ProcessorMode (from mode tracker if available, else current mode)
        """
        if self.mode_tracker:
            return self.mode_tracker.get_mode_at_statement(stmt_or_expr)
        return self.current_mode

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
                source_loc=source_loc
            )

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
            use_compatible: If True, use types_compatible() instead of types_equal()

        Raises:
            TypeCheckError: If types don't match
        """
        if use_compatible:
            types_match = TypeUtils.types_compatible(expected_type, actual_type)
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
            value = expr.value
            type_name = expected_type.name

            # Get range for expected type
            ranges = {
                'u8': (0, 255),
                'i8': (-128, 127),
                'u16': (0, 65535),
                'i16': (-32768, 32767),
            }

            if type_name in ranges:
                min_val, max_val = ranges[type_name]
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
            int_types = {'u8', 'i8', 'u16', 'i16'}
            if expected_name in int_types and actual_name in int_types:
                return f"use explicit cast: (value as {expected_name})"

            # Bool to integer or vice versa
            if expected_name == 'bool' and actual_name in int_types:
                return "use comparison to convert integer to bool: (value != 0)"
            if expected_name in int_types and actual_name == 'bool':
                return f"use cast to convert bool to integer: (value as {expected_name})"

        # Pointer type hints
        if isinstance(expected_type, PointerTypeInfo) and not isinstance(actual_type, PointerTypeInfo):
            return f"expected a pointer; use &value to get address"

        return None

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
                        "static variable initializer", decl.source_loc
                    )

            elif isinstance(decl, HIRConstDecl):
                if decl.value:
                    value_type = self.check_expression(decl.value, decl.const_type)
                    self._check_type_match(
                        decl.const_type, value_type, decl.value,
                        "const declaration", decl.source_loc
                    )

        # Type check all functions
        for decl in self.program.declarations:
            if isinstance(decl, HIRFunctionDecl):
                self.check_function(decl)

    def check_function(self, func: HIRFunctionDecl):
        """Type check a single function."""
        self.current_function = func

        # Validate that parameters are not aggregate types (arrays/structs cannot be passed by value)
        for param in func.parameters:
            self._check_no_aggregate_type(
                param.param_type,
                f"Parameter '{param.name}'",
                func.source_loc,
                verb="passed"
            )

        # Validate that return type is not an aggregate type (arrays/structs cannot be returned by value)
        if func.return_type:
            self._check_no_aggregate_type(
                func.return_type,
                f"Function '{func.name}' return type",
                func.source_loc,
                suggestion_suffix="\n  Or write to a pre-allocated output parameter",
                verb="returned"
            )

        # Validate interrupt handler mode transition
        if func.interrupt_attr and func.mode_attr:
            # Interrupt handlers with mode attributes MUST explicitly use transition=inline
            # because interrupts can fire from any mode and must restore properly
            if func.mode_attr.transition != ModeTransition.INLINE:
                raise TypeCheckError(
                    f"Interrupt handler '{func.name}' has #[mode] attribute but transition={func.mode_attr.transition.value}\n"
                    f"  Interrupt handlers with mode attributes MUST use transition=inline\n"
                    f"  Example: #[mode(m8, x8, transition=inline)]\n"
                    f"  Reason: Interrupts can fire from any mode and need automatic mode management",
                    source_loc=func.source_loc
                )

        # Get entry mode from function attribute
        entry_mode = ProcessorMode.from_attribute(func.mode_attr)
        self.current_mode = entry_mode

        if func.body is None:
            return  # External function

        # Phase 1: Build CFG
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build(func.body)

        # Phase 2: Mode analysis
        if entry_mode.is_fully_known():
            self.mode_tracker = ModeTracker(cfg, entry_mode)
            try:
                self.mode_tracker.analyze()
            except TypeCheckError as e:
                # Re-raise with function context
                raise TypeCheckError(
                    f"In function '{func.name}':\n{e.message}",
                    source_loc=e.source_loc or func.source_loc
                )

        # Phase 3: Type check function body
        self.check_block(func.body)

        # Phase 4: Check register preservation
        if func.preserves_attr:
            preservation_checker = PreservationChecker(func)
            preservation_checker.check()

        self.current_function = None
        self.mode_tracker = None

    def check_block(self, block):
        """Type check a block of statements."""
        for stmt in block.statements:
            self.check_statement(stmt)

    def check_statement(self, stmt: HIRStatement):
        """Type check a statement."""
        if isinstance(stmt, HIRLetStmt):
            self.check_let_statement(stmt)

        elif isinstance(stmt, HIRTupleLetStmt):
            self.check_tuple_let_statement(stmt)

        elif isinstance(stmt, HIRExprStmt):
            self.check_expression(stmt.expr)

        elif isinstance(stmt, HIRReturnStmt):
            for val in stmt.values:
                self.check_expression(val)

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

        elif isinstance(stmt, HIRWhileStmt):
            if stmt.condition:
                cond_type = self.check_expression(stmt.condition)
                self._require_boolean_type(cond_type, "While condition", stmt.condition.source_loc)
            self.check_block(stmt.body)

    def check_let_statement(self, stmt: HIRLetStmt):
        """Type check let binding."""
        # Get mode at this statement
        mode = self._get_mode_at(stmt)

        # Determine variable type
        if stmt.var_type:
            # Explicit type provided
            var_type = stmt.var_type
        elif isinstance(stmt.binding, RegisterLetBinding):
            # Infer from register type
            var_type = TypeInference.infer_register_alias_type(
                stmt.binding.register_name,
                mode
            )
            if var_type is None:
                raise TypeCheckError(
                    f"Cannot determine type of register {stmt.binding.register_name} in unknown mode",
                    source_loc=stmt.source_loc
                )
            stmt.var_type = var_type  # Fill in inferred type
        else:
            # Must have explicit type
            raise TypeCheckError(
                f"Variable '{stmt.name}' requires explicit type annotation",
                source_loc=stmt.source_loc
            )

        # Update symbol table with inferred type
        if stmt.symbol:
            stmt.symbol.var_type = var_type

        # Check initializer type matches
        if stmt.initializer:
            init_type = self.check_expression(stmt.initializer, var_type)
            # Handle tuple-to-scalar: let x: u8 = tuple_func() drops extra return values
            if isinstance(init_type, TupleTypeInfo) and not isinstance(var_type, TupleTypeInfo):
                first_elem_type = init_type.element_types[0]
                self._check_type_match(
                    var_type, first_elem_type, stmt.initializer,
                    "let binding (first element of tuple)", stmt.source_loc
                )
            else:
                self._check_type_match(
                    var_type, init_type, stmt.initializer,
                    "let binding", stmt.source_loc
                )

    def check_tuple_let_statement(self, stmt: HIRTupleLetStmt):
        """Type check tuple destructuring let binding.

        Example: let (a, b) = func_returning_tuple();

        Supports partial capture - binding fewer names than the tuple size.
        Extra return values are discarded.
        """
        # Check initializer type - must be a tuple
        init_type = self.check_expression(stmt.initializer)

        if not isinstance(init_type, TupleTypeInfo):
            raise TypeCheckError(
                f"Tuple destructuring requires a tuple type, got {init_type}",
                source_loc=stmt.source_loc
            )

        # Check we're not capturing more values than available
        if len(stmt.names) > len(init_type.element_types):
            raise TypeCheckError(
                f"Cannot destructure {len(init_type.element_types)}-element tuple "
                f"into {len(stmt.names)} bindings",
                source_loc=stmt.source_loc
            )

        # Infer types for each binding from tuple element types
        var_types = []
        for i, name in enumerate(stmt.names):
            elem_type = init_type.element_types[i]
            var_types.append(elem_type)

            # Update symbol with inferred type
            if i < len(stmt.symbols):
                stmt.symbols[i].var_type = elem_type

        # Store inferred types in statement
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
            # Infer type from context or default
            expr_type = TypeInference.infer_integer_literal_type(expr.value, context_type)
            expr.expr_type = expr_type
            return expr_type

        elif isinstance(expr, HIRBooleanLiteral):
            expr.expr_type = BasicTypeInfo('bool')
            return expr.expr_type

        elif isinstance(expr, HIREnumVariantExpr):
            # Enum variant has its enum type
            expr.expr_type = EnumTypeInfo(name=expr.enum_name)
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
                mode = self._get_mode_at(expr)
                reg_type = mode.get_register_type(expr.name)

            if reg_type is None:
                # Special error message for B register in wrong mode
                if expr.name == 'B':
                    from r65.compiler.typeck.processor_mode import ModeState
                    mode = self._get_mode_at(expr)
                    if mode.m_mode == ModeState.M16:
                        raise TypeCheckError(
                            f"B register only available in m8 mode (function has m16)",
                            source_loc=expr.source_loc,
                            hint="B is the high byte of 16-bit accumulator, only accessible in 8-bit mode"
                        )
                    else:
                        raise TypeCheckError(
                            f"B register only available in m8 mode",
                            source_loc=expr.source_loc,
                            hint="add #[mode(m8, ...)] attribute to the function"
                        )
                else:
                    raise TypeCheckError(
                        f"cannot determine type of register {expr.name} in unknown mode",
                        source_loc=expr.source_loc,
                        hint="add #[mode(m8/m16, x8/x16)] attribute to specify register sizes"
                    )

            expr.expr_type = reg_type
            return reg_type

        elif isinstance(expr, HIRFunctionAddress):
            return self.call_validator.check_function_address(expr)

        elif isinstance(expr, HIRBinaryOp):
            return self.check_binary_op(expr)

        elif isinstance(expr, HIRUnaryOp):
            return self.check_unary_op(expr)

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

        elif isinstance(expr, HIRAssignment):
            return self.check_assignment(expr)

        elif isinstance(expr, HIRMultiAssignment):
            return self.check_multi_assignment(expr)

        elif isinstance(expr, HIRDereference):
            return self.pointer_validator.check_dereference(expr)

        elif isinstance(expr, HIRAddressOf):
            return self.pointer_validator.check_addressof(expr)

        elif isinstance(expr, HIRIncludeBytesExpr):
            # include_bytes! returns an array of bytes
            # The exact type will be inferred from context (variable declaration)
            # For now, return a generic array type
            from r65.compiler.hir.types import ArrayTypeInfo
            elem_type = BasicTypeInfo(name='u8')
            # Size is unknown here - will be validated against variable type
            array_type = ArrayTypeInfo(element_type=elem_type, size=0)
            expr.expr_type = array_type
            return array_type

        elif isinstance(expr, HIRArrayFillExpr):
            # Array fill: [value; count]
            # Type check the fill value
            from r65.compiler.hir.types import ArrayTypeInfo
            fill_type = self.check_expression(expr.fill_value)
            # Create array type with the fill value type and count
            array_type = ArrayTypeInfo(element_type=fill_type, size=expr.count)
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
            # Type check first element to determine element type
            first_type = self.check_expression(expr.elements[0])
            # Type check remaining elements, ensuring they match
            for i, elem in enumerate(expr.elements[1:], 2):
                elem_type = self.check_expression(elem)
                if not TypeUtils.types_equal(first_type, elem_type):
                    raise TypeCheckError(
                        f"Array element {i} has type {elem_type}, expected {first_type}",
                        source_loc=elem.source_loc
                    )
            array_type = ArrayTypeInfo(element_type=first_type, size=len(expr.elements))
            expr.expr_type = array_type
            return array_type

        elif isinstance(expr, HIRStringLiteral):
            # String literal for byte array initialization
            return StringValidator.check_string_literal(expr, context_type)

        elif isinstance(expr, HIRStructLiteralExpr):
            # Struct literal: Player { x: 10, y: 20, health: 100 }
            return self.struct_validator.check_struct_literal(expr)

        elif isinstance(expr, HIRMatchExpression):
            return self.match_validator.check_match_expression(expr)

        else:
            raise TypeCheckError(
                f"Unknown expression type: {type(expr).__name__}",
                source_loc=expr.source_loc
            )

    def check_binary_op(self, expr: HIRBinaryOp) -> TypeInfo:
        """Type check binary operation."""
        # Validate operator restrictions
        OperatorValidator.validate_binary_op(expr)

        # Check operands
        left_type = self.check_expression(expr.left)
        right_type = self.check_expression(expr.right)

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
            expr.expr_type = left_type
            return left_type

        elif expr.op in ['+', '-', '*', '/', '%', '&', '|', '^']:
            # Arithmetic and bitwise: operands must match
            if not TypeUtils.types_equal(left_type, right_type):
                raise TypeCheckError(
                    f"type mismatch in '{expr.op}' operation: {left_type} vs {right_type}",
                    source_loc=expr.source_loc,
                    hint=f"cast one operand to match: (value as {left_type})"
                )

            # Result is same type
            expr.expr_type = left_type
            return left_type

        elif expr.op in ['==', '!=', '<', '<=', '>', '>=']:
            # Comparison: operands must be compatible, result is bool
            if not TypeUtils.types_compatible(left_type, right_type):
                raise TypeCheckError(
                    f"cannot compare {left_type} with {right_type}",
                    source_loc=expr.source_loc,
                    hint="comparison requires compatible types"
                )

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

    def check_unary_op(self, expr: HIRUnaryOp) -> TypeInfo:
        """Type check unary operation."""
        operand_type = self.check_expression(expr.operand)

        if expr.op == '!':
            # Logical NOT: operand must be bool
            if not TypeUtils.is_boolean_type(operand_type):
                raise TypeCheckError(
                    f"logical NOT '!' requires boolean operand, found {operand_type}",
                    source_loc=expr.operand.source_loc,
                    hint="use comparison like (value != 0) to convert to bool"
                )
            expr.expr_type = BasicTypeInfo('bool')

        elif expr.op == '~':
            # Bitwise NOT: operand must be integer
            if not TypeUtils.is_integer_type(operand_type):
                raise TypeCheckError(
                    f"bitwise NOT '~' requires integer operand, found {operand_type}",
                    source_loc=expr.operand.source_loc,
                    hint="only integer types (u8, i8, u16, i16) support bitwise operations"
                )
            expr.expr_type = operand_type

        elif expr.op == '-':
            # Negation: operand must be integer
            if not TypeUtils.is_integer_type(operand_type):
                raise TypeCheckError(
                    f"negation '-' requires integer operand, found {operand_type}",
                    source_loc=expr.operand.source_loc,
                    hint="only integer types can be negated"
                )
            expr.expr_type = operand_type

        else:
            raise TypeCheckError(
                f"unknown unary operator: {expr.op}",
                source_loc=expr.source_loc
            )

        return expr.expr_type

    def check_type_cast(self, expr: HIRTypeCast) -> TypeInfo:
        """Type check explicit cast."""
        source_type = self.check_expression(expr.expr)
        target_type = expr.target_type

        if not TypeUtils.can_cast(source_type, target_type):
            raise TypeCheckError(
                f"cannot cast {source_type} to {target_type}",
                source_loc=expr.source_loc,
                hint="casts are only allowed between compatible types (integers, bools)"
            )

        expr.expr_type = target_type
        return target_type

    def check_array_index(self, expr: HIRArrayIndex) -> TypeInfo:
        """Type check array indexing."""
        base_type = self.check_expression(expr.array)
        index_type = self.check_expression(expr.index)

        # Index must be integer
        self._require_integer_type(index_type, "Array index", expr.index.source_loc)

        # Handle pointer types: (*ptr: T)[idx] or (far *ptr: T)[idx]
        if isinstance(base_type, PointerTypeInfo):
            pointee = base_type.pointee_type
            # If pointer to array, result is element type
            if isinstance(pointee, ArrayTypeInfo):
                # Constant index bounds checking for pointer-to-array
                if expr.original_ast and self.const_evaluator.is_constant(expr.original_ast.index):
                    try:
                        index_value = self.const_evaluator.eval(expr.original_ast.index)
                        array_size = pointee.size

                        if index_value < 0:
                            raise TypeCheckError(
                                f"array index {index_value} is out of bounds (negative index)",
                                source_loc=expr.index.source_loc,
                                hint=f"valid indices are 0 to {array_size - 1}"
                            )
                        elif index_value >= array_size:
                            raise TypeCheckError(
                                f"array index {index_value} is out of bounds for array of size {array_size}",
                                source_loc=expr.index.source_loc,
                                hint=f"valid indices are 0 to {array_size - 1}"
                            )
                    except Exception:
                        # If const evaluation fails, skip bounds checking
                        pass
                expr.expr_type = pointee.element_type
                return expr.expr_type
            # If pointer to slice (unsized array), result is element type
            elif isinstance(pointee, SliceTypeInfo):
                # No bounds checking for slices (size unknown at compile time)
                expr.expr_type = pointee.element_type
                return expr.expr_type
            else:
                # Pointer to non-array: ptr[idx] is pointer arithmetic
                expr.expr_type = pointee
                return expr.expr_type

        # Array indexing: array[idx]
        if isinstance(base_type, ArrayTypeInfo):
            # Constant index bounds checking
            if expr.original_ast and self.const_evaluator.is_constant(expr.original_ast.index):
                try:
                    index_value = self.const_evaluator.eval(expr.original_ast.index)
                    array_size = base_type.size

                    if index_value < 0:
                        raise TypeCheckError(
                            f"array index {index_value} is out of bounds (negative index)",
                            source_loc=expr.index.source_loc,
                            hint=f"valid indices are 0 to {array_size - 1}"
                        )
                    elif index_value >= array_size:
                        raise TypeCheckError(
                            f"array index {index_value} is out of bounds for array of size {array_size}",
                            source_loc=expr.index.source_loc,
                            hint=f"valid indices are 0 to {array_size - 1}"
                        )
                except Exception:
                    # If const evaluation fails, skip bounds checking
                    pass

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
        base_type = self.check_expression(expr.base)

        # Base must be struct type
        from r65.compiler.hir import StructTypeInfo
        if not isinstance(base_type, StructTypeInfo):
            raise TypeCheckError(
                f"cannot access field '{expr.field_name}' on type {base_type}",
                source_loc=expr.base.source_loc,
                hint="field access requires a struct type"
            )

        # Look up struct definition from symbol table (not cached in StructTypeInfo)
        # This ensures we get the HIR definition, not the AST definition that may
        # have been cached during early type resolution in Pass 1.
        struct_symbol = self.program.symbol_table.lookup(base_type.name)
        if struct_symbol is None:
            raise TypeCheckError(
                f"struct '{base_type.name}' definition not found",
                source_loc=expr.source_loc,
                hint="ensure the struct is defined before use"
            )
        struct_def = struct_symbol.definition
        if struct_def is None:
            raise TypeCheckError(
                f"struct '{base_type.name}' definition not found",
                source_loc=expr.source_loc,
                hint="ensure the struct is defined before use"
            )

        field = None
        field_index = None
        for i, f in enumerate(struct_def.fields):
            if f.name == expr.field_name:
                field = f
                field_index = i
                break

        if field is None:
            # Get available field names for hint
            available_fields = [f.name for f in struct_def.fields]
            hint = f"available fields: {', '.join(available_fields)}" if available_fields else None
            raise TypeCheckError(
                f"struct '{base_type.name}' has no field '{expr.field_name}'",
                source_loc=expr.source_loc,
                hint=hint
            )

        expr.expr_type = field.field_type
        expr.field_index = field_index
        expr.field_offset = field.offset
        return expr.expr_type

    def check_assignment(self, expr: HIRAssignment) -> TypeInfo:
        """Type check assignment."""
        target_type = self.check_expression(expr.target)
        value_type = self.check_expression(expr.value, target_type)

        # Arrays and structs cannot be assigned by value
        # Note: Using specific message here since assignment has unique suggestion
        if TypeUtils.is_aggregate_type(target_type):
            type_name = str(target_type)
            raise TypeCheckError(
                f"Cannot assign '{type_name}' by value\n"
                f"  Arrays and structs cannot be copied by value\n"
                f"  Suggestion: Copy fields individually or use a pointer",
                source_loc=expr.source_loc
            )

        # Handle tuple-to-scalar: A = tuple_func() drops extra return values
        if isinstance(value_type, TupleTypeInfo) and not isinstance(target_type, TupleTypeInfo):
            first_elem_type = value_type.element_types[0]
            self._check_type_match(
                target_type, first_elem_type, expr.value,
                "assignment (first element of tuple)", expr.source_loc, use_compatible=True
            )
            expr.expr_type = target_type
            return target_type

        # Types must be compatible (allows enum/integer interop)
        self._check_type_match(
            target_type, value_type, expr.value,
            "assignment", expr.source_loc, use_compatible=True
        )

        expr.expr_type = target_type
        return target_type

    def check_multi_assignment(self, expr: HIRMultiAssignment) -> TypeInfo:
        """Type check multi-assignment (tuple destructuring).

        Handles: (A, X) = func() where func returns a tuple.
        """
        # Type check the value expression (should return a tuple)
        value_type = self.check_expression(expr.value)

        # Value must be a tuple type
        if not isinstance(value_type, TupleTypeInfo):
            raise TypeCheckError(
                f"Multi-assignment requires a tuple value, got '{value_type}'",
                source_loc=expr.source_loc
            )

        # Number of targets must match number of tuple elements
        num_targets = len(expr.targets)
        num_elements = len(value_type.element_types)
        if num_targets != num_elements:
            raise TypeCheckError(
                f"Multi-assignment has {num_targets} targets but value has {num_elements} elements",
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
