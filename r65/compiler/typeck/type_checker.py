"""
Main type checker for R65 compiler.

Orchestrates type checking, mode tracking, and validation.
"""

from typing import Optional
from r65.compiler.hir import (
    HIRProgram, HIRFunctionDecl, HIRExpression, HIRStatement,
    HIRBinaryOp, HIRUnaryOp, HIRIntegerLiteral, HIRBooleanLiteral,
    HIRIdentifier, HIRFunctionAddress, HIRRegister, HIRIncludeBytesExpr, HIRTypeCast, HIRFunctionCall,
    HIRMethodCall, HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf, HIRAssignment,
    HIRLetStmt, HIRExprStmt, HIRReturnStmt, HIRIfStmt, HIRWhileStmt,
    HIRStaticDecl, HIRConstDecl,
    HIRMatchExpression, HIRPattern, HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern, HIRIdentifierPattern, HIROrPattern,
    BasicTypeInfo, TypeInfo, SymbolKind,
    RegisterLetBinding, ArrayTypeInfo,
    ModeTransition
)
from r65.compiler.typeck.processor_mode import ProcessorMode
from r65.compiler.typeck.mode_tracker import ModeTracker
from r65.compiler.typeck.cfg_builder import CFGBuilder
from r65.compiler.typeck.type_utils import TypeUtils
from r65.compiler.typeck.operator_validator import OperatorValidator
from r65.compiler.typeck.preservation_checker import PreservationChecker
from r65.compiler.typeck.type_inference import TypeInference
from r65.compiler.typeck.errors import TypeCheckError, TypeCheckWarning
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
                            f"Literal value {value} exceeds maximum for type {type_name} ({max_val})\n"
                            f"  Valid range for {type_name}: {min_val} to {max_val}\n"
                            f"  Suggestion: Use a larger type (e.g., u16) or reduce the value",
                            source_loc=source_loc
                        )
                    else:  # value < min_val
                        raise TypeCheckError(
                            f"Literal value {value} is below minimum for type {type_name} ({min_val})\n"
                            f"  Valid range for {type_name}: {min_val} to {max_val}\n"
                            f"  Suggestion: Use a signed type (e.g., i8, i16) for negative values",
                            source_loc=source_loc
                        )

        # Default type mismatch error
        raise TypeCheckError(
            f"Type mismatch in {context}\n"
            f"  Expected: {expected_type}\n"
            f"  Found: {actual_type}",
            source_loc=source_loc
        )

    # ========================================================================
    # Main Type Checking
    # ========================================================================

    def check(self):
        """Perform type checking on entire program."""
        # Type check static initializers
        for decl in self.program.declarations:
            if isinstance(decl, HIRStaticDecl):
                if decl.initializer:
                    init_type = self.check_expression(decl.initializer, decl.var_type)
                    if not TypeUtils.types_equal(decl.var_type, init_type):
                        self._raise_type_mismatch_error(
                            expected_type=decl.var_type,
                            actual_type=init_type,
                            expr=decl.initializer,
                            context="static variable initializer",
                            source_loc=decl.source_loc
                        )

            elif isinstance(decl, HIRConstDecl):
                if decl.value:
                    value_type = self.check_expression(decl.value, decl.const_type)
                    if not TypeUtils.types_equal(decl.const_type, value_type):
                        self._raise_type_mismatch_error(
                            expected_type=decl.const_type,
                            actual_type=value_type,
                            expr=decl.value,
                            context="const declaration",
                            source_loc=decl.source_loc
                        )

        # Type check all functions
        for decl in self.program.declarations:
            if isinstance(decl, HIRFunctionDecl):
                self.check_function(decl)

    def check_function(self, func: HIRFunctionDecl):
        """Type check a single function."""
        self.current_function = func

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
            preservation_checker = PreservationChecker(func, cfg)
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

            if not TypeUtils.types_equal(var_type, init_type):
                self._raise_type_mismatch_error(
                    expected_type=var_type,
                    actual_type=init_type,
                    expr=stmt.initializer,
                    context="let binding",
                    source_loc=stmt.source_loc
                )

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

        elif isinstance(expr, HIRIdentifier):
            # Look up symbol
            symbol = expr.symbol
            if symbol is None:
                raise TypeCheckError(
                    f"Undefined identifier '{expr.name}'",
                    source_loc=expr.source_loc
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
            # Get register type from current mode
            mode = self._get_mode_at(expr)
            reg_type = mode.get_register_type(expr.name)
            if reg_type is None:
                # Special error message for B register in wrong mode
                if expr.name == 'B':
                    from r65.compiler.typeck.processor_mode import ModeState
                    if mode.m_mode == ModeState.M16:
                        raise TypeCheckError(
                            f"B register only available in m8 mode\n"
                            f"  Function has mode m16 where accumulator is 16-bit",
                            source_loc=expr.source_loc
                        )
                    else:
                        raise TypeCheckError(
                            f"B register only available in m8 mode\n"
                            f"  B requires #[mode(m8, ...)]",
                            source_loc=expr.source_loc
                        )
                else:
                    raise TypeCheckError(
                        f"Cannot determine type of register {expr.name} in unknown mode",
                        source_loc=expr.source_loc
                    )

            expr.expr_type = reg_type
            return reg_type

        elif isinstance(expr, HIRFunctionAddress):
            return self.check_function_address(expr)

        elif isinstance(expr, HIRBinaryOp):
            return self.check_binary_op(expr)

        elif isinstance(expr, HIRUnaryOp):
            return self.check_unary_op(expr)

        elif isinstance(expr, HIRTypeCast):
            return self.check_type_cast(expr)

        elif isinstance(expr, HIRFunctionCall):
            return self.check_function_call(expr)

        elif isinstance(expr, HIRMethodCall):
            return self.check_method_call(expr)

        elif isinstance(expr, HIRArrayIndex):
            return self.check_array_index(expr)

        elif isinstance(expr, HIRFieldAccess):
            return self.check_field_access(expr)

        elif isinstance(expr, HIRAssignment):
            return self.check_assignment(expr)

        elif isinstance(expr, HIRDereference):
            return self.check_dereference(expr)

        elif isinstance(expr, HIRAddressOf):
            return self.check_addressof(expr)

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

        elif isinstance(expr, HIRMatchExpression):
            return self.check_match_expression(expr)

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
        if expr.op in ['+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>']:
            # Arithmetic and bitwise: operands must match
            if not TypeUtils.types_equal(left_type, right_type):
                raise TypeCheckError(
                    f"Type mismatch in binary operation '{expr.op}'\n"
                    f"  Left: {left_type}\n"
                    f"  Right: {right_type}",
                    source_loc=expr.source_loc
                )

            # Result is same type
            expr.expr_type = left_type
            return left_type

        elif expr.op in ['==', '!=', '<', '<=', '>', '>=']:
            # Comparison: operands must match, result is bool
            if not TypeUtils.types_equal(left_type, right_type):
                raise TypeCheckError(
                    f"Type mismatch in comparison '{expr.op}'\n"
                    f"  Left: {left_type}\n"
                    f"  Right: {right_type}",
                    source_loc=expr.source_loc
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
            self._require_boolean_type(operand_type, "Operand of '!'", expr.operand.source_loc)
            expr.expr_type = BasicTypeInfo('bool')

        elif expr.op == '~':
            # Bitwise NOT: operand must be integer
            self._require_integer_type(operand_type, "Operand of '~'", expr.operand.source_loc)
            expr.expr_type = operand_type

        elif expr.op == '-':
            # Negation: operand must be integer
            self._require_integer_type(operand_type, "Operand of '-'", expr.operand.source_loc)
            expr.expr_type = operand_type

        else:
            raise TypeCheckError(
                f"Unknown unary operator: {expr.op}",
                source_loc=expr.source_loc
            )

        return expr.expr_type

    def check_type_cast(self, expr: HIRTypeCast) -> TypeInfo:
        """Type check explicit cast."""
        source_type = self.check_expression(expr.expr)
        target_type = expr.target_type

        if not TypeUtils.can_cast(source_type, target_type):
            raise TypeCheckError(
                f"Invalid cast from {source_type} to {target_type}",
                source_loc=expr.source_loc
            )

        expr.expr_type = target_type
        return target_type

    def check_function_call(self, expr: HIRFunctionCall) -> TypeInfo:
        """
        Type check function call.

        Supports both:
        - Direct calls: expr.func is HIRIdentifier pointing to function
        - Indirect calls: expr.func is expression with function pointer type
        - Built-in calls: expr.builtin_name is set

        Checks:
        - Argument types match parameters
        - Return type
        - Mode compatibility between caller and callee (for direct calls)
        """
        from r65.compiler.builtins import BuiltinRegistry

        # Check if this is a built-in function call
        if expr.builtin_name:
            return self._check_builtin_call(expr)

        # Handle direct call vs indirect call
        if isinstance(expr.func, HIRIdentifier) and expr.func.symbol.kind == SymbolKind.FUNCTION:
            # Direct call to a function
            func_symbol = expr.func.symbol

            # Look up HIR function declaration from program
            func_decl = self._lookup_function_decl(func_symbol.name, expr.source_loc)

            # Check argument count
            if len(expr.args) != len(func_decl.parameters):
                raise TypeCheckError(
                    f"Function '{func_symbol.name}' expects {len(func_decl.parameters)} arguments, got {len(expr.args)}",
                    source_loc=expr.source_loc
                )

            # Type check each argument
            for arg, param in zip(expr.args, func_decl.parameters):
                arg_type = self.check_expression(arg)
                # TODO: Check arg_type matches param.param_type

            # Check mode compatibility (only for direct calls)
            self._check_call_mode_compatibility(func_symbol.name, func_decl, expr.source_loc)

            # Set return type
            if func_decl.return_type:
                expr.expr_type = func_decl.return_type
            else:
                # Void function
                expr.expr_type = BasicTypeInfo('void')

        else:
            # Handle indirect call (function pointer) - type check the expression
            from r65.compiler.hir.types import FunctionTypeInfo
            func_type = self.check_expression(expr.func)

            if not isinstance(func_type, FunctionTypeInfo):
                raise TypeCheckError(
                    f"Cannot call expression of type {func_type}, expected function or function pointer",
                    source_loc=expr.source_loc
                )

            # Check argument count matches function type
            if len(expr.args) != len(func_type.param_types):
                raise TypeCheckError(
                    f"Function pointer expects {len(func_type.param_types)} arguments, got {len(expr.args)}",
                    source_loc=expr.source_loc
                )

            # Type check each argument against function type
            for arg, param_type in zip(expr.args, func_type.param_types):
                arg_type = self.check_expression(arg)
                # TODO: Check arg_type matches param_type

            # Set return type from function type
            if func_type.return_type:
                expr.expr_type = func_type.return_type
            else:
                expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def _check_builtin_call(self, expr: HIRFunctionCall) -> TypeInfo:
        """
        Type check built-in function call.

        Built-ins are validated at HIR construction, so we just need to:
        1. Type check arguments
        2. Set the return type

        Args:
            expr: HIRFunctionCall with builtin_name set

        Returns:
            Return type of the built-in
        """
        from r65.compiler.builtins import BuiltinRegistry

        builtin = BuiltinRegistry.get_builtin(expr.builtin_name)
        if not builtin:
            raise TypeCheckError(
                f"Unknown built-in function: {expr.builtin_name}",
                source_loc=expr.source_loc
            )

        # Type check arguments
        for arg in expr.args:
            self.check_expression(arg)

        # Set return type
        if builtin.returns_value:
            # Built-ins that return values return u8 or u16 depending on mode
            # For simplicity, assume u8 for now
            # TODO: Infer return type from argument types
            expr.expr_type = BasicTypeInfo('u8')
        else:
            # Void return
            expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def check_method_call(self, expr: HIRMethodCall) -> TypeInfo:
        """
        Type check method call (e.g., value.rotate_left(3)).

        Currently only supports rotate_left and rotate_right methods on integer types.

        Args:
            expr: HIRMethodCall to type check

        Returns:
            Return type of the method
        """
        from r65.compiler.hir import HIRIntegerLiteral

        # Type check receiver
        receiver_type = self.check_expression(expr.receiver)

        # Validate receiver is an integer type
        if not isinstance(receiver_type, BasicTypeInfo) or receiver_type.name not in ['u8', 'i8', 'u16', 'i16']:
            raise TypeCheckError(
                f"Method '{expr.method_name}' can only be called on integer types, not {receiver_type}",
                source_loc=expr.source_loc
            )

        # Validate method name
        if expr.method_name not in ['rotate_left', 'rotate_right']:
            raise TypeCheckError(
                f"Unknown method '{expr.method_name}' for type {receiver_type}",
                source_loc=expr.source_loc
            )

        # Type check argument (rotation count)
        if len(expr.args) != 1:
            raise TypeCheckError(
                f"{expr.method_name}() takes exactly 1 argument, got {len(expr.args)}",
                source_loc=expr.source_loc
            )

        count_arg = expr.args[0]
        count_type = self.check_expression(count_arg)

        # Validate count is an integer literal (compile-time constant)
        if not isinstance(count_arg, HIRIntegerLiteral):
            raise TypeCheckError(
                f"{expr.method_name}() count must be a constant integer literal",
                source_loc=count_arg.source_loc
            )

        # Validate count is in range 1-8
        count_value = count_arg.value
        if not (1 <= count_value <= 8):
            raise TypeCheckError(
                f"{expr.method_name}() count must be between 1 and 8, got {count_value}",
                source_loc=count_arg.source_loc
            )

        # Return type is same as receiver type
        expr.expr_type = receiver_type
        return expr.expr_type

    def check_function_address(self, expr: HIRFunctionAddress) -> TypeInfo:
        """
        Type check function address expression.

        Returns a FunctionTypeInfo representing the function pointer type.
        """
        from r65.compiler.hir.types import FunctionTypeInfo

        # Look up function symbol
        func_symbol = expr.symbol
        if not func_symbol:
            raise TypeCheckError(
                f"Function '{expr.function_name}' not resolved",
                source_loc=expr.source_loc
            )

        # Find function declaration
        func_decl = self._lookup_function_decl(func_symbol.name, expr.source_loc)

        # Build function type from declaration
        param_types = [param.param_type for param in func_decl.parameters]

        func_type = FunctionTypeInfo(
            is_far=func_decl.is_far,
            param_types=param_types,
            return_type=func_decl.return_type
        )

        expr.expr_type = func_type
        return func_type

    def _check_call_mode_compatibility(self, func_name: str, func_decl: HIRFunctionDecl, source_loc):
        """
        Check mode compatibility between caller and callee.

        Rules:
        - Mixed-mode calls are allowed
        - transition=none: No automatic mode switching (programmer handles)
        - transition=auto: Callee generates wrapper (PHP/SEP-REP/body/PLP/RTS)
        - transition=caller: Caller generates wrapper
        - transition=auto + preserves(STATUS) is an error (conflicting)
        """
        # Get callee mode
        if func_decl.mode_attr:
            callee_mode = ProcessorMode.from_attribute(func_decl.mode_attr)
        else:
            callee_mode = ProcessorMode.unknown()

        # Get caller mode (current mode in context)
        caller_mode = self.current_mode

        # Check if modes are compatible
        if caller_mode == callee_mode:
            # Same mode - no issue
            return

        # Check if both modes are fully known
        if not caller_mode.is_fully_known() or not callee_mode.is_fully_known():
            # Unknown mode - can't check compatibility
            return

        # Modes differ - check transition attribute
        mode_attr = func_decl.mode_attr
        if mode_attr and hasattr(mode_attr, 'transition'):
            transition = mode_attr.transition
        else:
            transition = ModeTransition.NONE  # Default

        # Validate transition=inline doesn't conflict with preserves(STATUS)
        if transition == ModeTransition.INLINE:
            if func_decl.preserves_attr and 'STATUS' in func_decl.preserves_attr.registers:
                raise TypeCheckError(
                    f"Function '{func_name}' cannot use transition=inline with #[preserves(STATUS)]\n"
                    f"  transition=inline requires modifying STATUS to switch modes, which conflicts with preservation",
                    source_loc=source_loc
                )

        # If modes don't match and transition=none, this is an error
        if transition == ModeTransition.NONE:
            raise TypeCheckError(
                f"Cannot call function '{func_name}' with mismatched processor modes\n"
                f"  Caller mode: {caller_mode}\n"
                f"  Callee mode: {callee_mode}\n"
                f"  Fix: Add transition attribute to callee: #[mode(..., transition=inline)] or #[mode(..., transition=caller)]",
                source_loc=source_loc
            )

    def check_array_index(self, expr: HIRArrayIndex) -> TypeInfo:
        """Type check array indexing."""
        array_type = self.check_expression(expr.array)
        index_type = self.check_expression(expr.index)

        # Array must be array type
        if not isinstance(array_type, ArrayTypeInfo):
            raise TypeCheckError(
                f"Cannot index non-array type {array_type}",
                source_loc=expr.array.source_loc
            )

        # Index must be integer
        self._require_integer_type(index_type, "Array index", expr.index.source_loc)

        # Constant index bounds checking
        if expr.original_ast and self.const_evaluator.is_constant(expr.original_ast.index):
            try:
                index_value = self.const_evaluator.eval(expr.original_ast.index)
                array_size = array_type.size
                
                if index_value < 0:
                    raise TypeCheckError(
                        f"Array index {index_value} is out of bounds for array of size {array_size} (negative index)",
                        source_loc=expr.index.source_loc
                    )
                elif index_value >= array_size:
                    raise TypeCheckError(
                        f"Array index {index_value} is out of bounds for array of size {array_size}",
                        source_loc=expr.index.source_loc
                    )
            except Exception:
                # If const evaluation fails, skip bounds checking
                pass

        # Result is element type
        expr.expr_type = array_type.element_type
        return expr.expr_type

    def check_field_access(self, expr: HIRFieldAccess) -> TypeInfo:
        """Type check field access."""
        base_type = self.check_expression(expr.base)

        # Base must be struct type
        from r65.compiler.hir import StructTypeInfo
        if not isinstance(base_type, StructTypeInfo):
            raise TypeCheckError(
                f"Cannot access field of non-struct type {base_type}",
                source_loc=expr.base.source_loc
            )

        # Find field in struct
        struct_def = base_type.definition
        if struct_def is None:
            raise TypeCheckError(
                f"Struct {base_type.name} definition not found",
                source_loc=expr.source_loc
            )

        field = None
        for f in struct_def.fields:
            if f.name == expr.field_name:
                field = f
                break

        if field is None:
            raise TypeCheckError(
                f"Struct {base_type.name} has no field '{expr.field_name}'",
                source_loc=expr.source_loc
            )

        expr.expr_type = field.field_type
        return expr.expr_type

    def check_dereference(self, expr: HIRDereference) -> TypeInfo:
        """Type check pointer dereference (*ptr)."""
        from r65.compiler.hir.types import PointerTypeInfo

        pointer_type = self.check_expression(expr.pointer)

        # Pointer must be a pointer type
        if not isinstance(pointer_type, PointerTypeInfo):
            raise TypeCheckError(
                f"Cannot dereference non-pointer type {pointer_type}",
                source_loc=expr.pointer.source_loc
            )

        # Dereference yields the pointee type
        expr.expr_type = pointer_type.pointee_type
        return expr.expr_type

    def check_addressof(self, expr: HIRAddressOf) -> TypeInfo:
        """Type check address-of operator (&variable)."""
        from r65.compiler.hir.types import PointerTypeInfo
        from r65.compiler.hir import HIRIdentifier, HIRArrayIndex, HIRFieldAccess

        operand_type = self.check_expression(expr.operand)

        # Operand must be an lvalue (identifier, array index, or field access)
        if not isinstance(expr.operand, (HIRIdentifier, HIRArrayIndex, HIRFieldAccess)):
            raise TypeCheckError(
                f"Cannot take address of non-lvalue expression",
                source_loc=expr.operand.source_loc
            )

        # Address-of yields a near pointer to the operand type
        # For now, all pointers are near (16-bit)
        # TODO: Support far pointers based on variable's storage attribute
        pointer_type = PointerTypeInfo(is_far=False, pointee_type=operand_type)
        expr.expr_type = pointer_type
        return expr.expr_type

    def check_match_expression(self, expr: HIRMatchExpression) -> TypeInfo:
        """Type check match expression."""
        from r65.compiler.hir import (HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern,
                                       HIRIdentifierPattern, HIROrPattern)

        # Check scrutinee type
        scrutinee_type = self.check_expression(expr.scrutinee)

        # Check each arm
        arm_types = []
        has_wildcard = False

        for arm in expr.arms:
            # Check pattern matches scrutinee type and check for wildcard/identifier
            if self._check_pattern(arm.pattern, scrutinee_type):
                has_wildcard = True

            # Check arm body
            body_type = self.check_expression(arm.body)
            arm_types.append(body_type)

        # All arms must return compatible types
        if not arm_types:
            raise TypeCheckError(
                "Match expression must have at least one arm",
                source_loc=expr.source_loc
            )

        # Use first arm's type as the expected type
        result_type = arm_types[0]
        for i, arm_type in enumerate(arm_types[1:], 1):
            if not TypeUtils.types_equal(result_type, arm_type):
                raise TypeCheckError(
                    f"Match arm {i} returns type {arm_type}, expected {result_type}",
                    source_loc=expr.arms[i].body.source_loc
                )

        # Basic exhaustiveness check: must have wildcard/identifier pattern or cover all cases
        if not has_wildcard:
            # For now, just warn - full exhaustiveness checking is complex
            # TODO: Implement proper exhaustiveness checking
            pass

        expr.expr_type = result_type
        return result_type

    def _check_pattern(self, pattern, scrutinee_type: TypeInfo) -> bool:
        """
        Check if pattern is valid for scrutinee type.
        Returns True if pattern is a catch-all (wildcard or identifier).
        """
        from r65.compiler.hir import (HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern,
                                       HIRIdentifierPattern, HIROrPattern)

        if isinstance(pattern, HIRLiteralPattern):
            # Literal must match scrutinee type
            if isinstance(pattern.value, bool):
                if scrutinee_type.name != 'bool':
                    raise TypeCheckError(f"Cannot match bool literal against {scrutinee_type}")
            elif isinstance(pattern.value, int):
                if scrutinee_type.name not in ('u8', 'i8', 'u16', 'i16'):
                    raise TypeCheckError(f"Cannot match integer literal against {scrutinee_type}")
            return False

        elif isinstance(pattern, HIREnumPattern):
            # Enum pattern must match enum type
            # scrutinee should be the enum's underlying integer type
            return False

        elif isinstance(pattern, HIRWildcardPattern):
            # Wildcard always matches
            return True

        elif isinstance(pattern, HIRIdentifierPattern):
            # Identifier pattern always matches and binds the value
            # Set the symbol's type to the scrutinee type
            pattern.symbol.var_type = scrutinee_type
            return True

        elif isinstance(pattern, HIROrPattern):
            # Or pattern: check all sub-patterns
            is_catchall = False
            for subpat in pattern.patterns:
                if self._check_pattern(subpat, scrutinee_type):
                    is_catchall = True
            return is_catchall

        else:
            raise TypeCheckError(f"Unknown pattern type: {type(pattern).__name__}")

    def check_assignment(self, expr: HIRAssignment) -> TypeInfo:
        """Type check assignment."""
        target_type = self.check_expression(expr.target)
        value_type = self.check_expression(expr.value, target_type)

        # Types must match exactly
        if not TypeUtils.types_equal(target_type, value_type):
            self._raise_type_mismatch_error(
                expected_type=target_type,
                actual_type=value_type,
                expr=expr.value,
                context="assignment",
                source_loc=expr.source_loc
            )

        expr.expr_type = target_type
        return target_type
