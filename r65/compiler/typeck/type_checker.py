"""
Main type checker for R65 compiler.

Orchestrates type checking, mode tracking, and validation.
"""

from typing import Optional
from r65.compiler.hir import (
    HIRProgram, HIRFunctionDecl, HIRExpression, HIRStatement,
    HIRBinaryOp, HIRUnaryOp, HIRIntegerLiteral, HIRBooleanLiteral,
    HIRIdentifier, HIRRegister, HIRTypeCast, HIRFunctionCall,
    HIRArrayIndex, HIRFieldAccess, HIRAssignment,
    HIRLetStmt, HIRExprStmt, HIRReturnStmt, HIRIfStmt, HIRWhileStmt,
    HIRStaticDecl, HIRConstDecl,
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

    def check(self):
        """Perform type checking on entire program."""
        # Type check static initializers
        for decl in self.program.declarations:
            if isinstance(decl, HIRStaticDecl):
                if decl.initializer:
                    self.check_expression(decl.initializer, decl.var_type)

            elif isinstance(decl, HIRConstDecl):
                if decl.value:
                    self.check_expression(decl.value, decl.const_type)

        # Type check all functions
        for decl in self.program.declarations:
            if isinstance(decl, HIRFunctionDecl):
                self.check_function(decl)

    def check_function(self, func: HIRFunctionDecl):
        """Type check a single function."""
        self.current_function = func

        # Validate interrupt handler mode transition
        if func.interrupt_attr and func.mode_attr:
            # Interrupt handlers with mode attributes MUST explicitly use transition=auto
            # because interrupts can fire from any mode and must restore properly
            if func.mode_attr.transition != ModeTransition.AUTO:
                raise TypeCheckError(
                    f"Interrupt handler '{func.name}' has #[mode] attribute but transition={func.mode_attr.transition.value}\n"
                    f"  Interrupt handlers with mode attributes MUST use transition=auto\n"
                    f"  Example: #[mode(m8, x8, transition=auto)]\n"
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
            if not TypeUtils.is_boolean_type(cond_type):
                raise TypeCheckError(
                    f"If condition must be boolean, found {cond_type}",
                    source_loc=stmt.condition.source_loc
                )

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
                if not TypeUtils.is_boolean_type(cond_type):
                    raise TypeCheckError(
                        f"While condition must be boolean, found {cond_type}",
                        source_loc=stmt.condition.source_loc
                    )
            self.check_block(stmt.body)

    def check_let_statement(self, stmt: HIRLetStmt):
        """Type check let binding."""
        # Get mode at this statement
        if self.mode_tracker:
            mode = self.mode_tracker.get_mode_at_statement(stmt)
        else:
            mode = self.current_mode

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
                raise TypeCheckError(
                    f"Type mismatch in let binding\n"
                    f"  Expected: {var_type}\n"
                    f"  Found: {init_type}",
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
            else:
                raise TypeCheckError(
                    f"'{expr.name}' is not a value",
                    source_loc=expr.source_loc
                )

            return expr.expr_type

        elif isinstance(expr, HIRRegister):
            # Get register type from current mode
            if self.mode_tracker:
                mode = self.mode_tracker.get_mode_at_statement(expr)
            else:
                mode = self.current_mode

            reg_type = mode.get_register_type(expr.name)
            if reg_type is None:
                raise TypeCheckError(
                    f"Cannot determine type of register {expr.name} in unknown mode",
                    source_loc=expr.source_loc
                )

            expr.expr_type = reg_type
            return reg_type

        elif isinstance(expr, HIRBinaryOp):
            return self.check_binary_op(expr)

        elif isinstance(expr, HIRUnaryOp):
            return self.check_unary_op(expr)

        elif isinstance(expr, HIRTypeCast):
            return self.check_type_cast(expr)

        elif isinstance(expr, HIRFunctionCall):
            return self.check_function_call(expr)

        elif isinstance(expr, HIRArrayIndex):
            return self.check_array_index(expr)

        elif isinstance(expr, HIRFieldAccess):
            return self.check_field_access(expr)

        elif isinstance(expr, HIRAssignment):
            return self.check_assignment(expr)

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
            if not TypeUtils.is_boolean_type(left_type):
                raise TypeCheckError(
                    f"Left operand of '{expr.op}' must be bool, found {left_type}",
                    source_loc=expr.left.source_loc
                )
            if not TypeUtils.is_boolean_type(right_type):
                raise TypeCheckError(
                    f"Right operand of '{expr.op}' must be bool, found {right_type}",
                    source_loc=expr.right.source_loc
                )

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
                    f"Operand of '!' must be bool, found {operand_type}",
                    source_loc=expr.operand.source_loc
                )
            expr.expr_type = BasicTypeInfo('bool')

        elif expr.op == '~':
            # Bitwise NOT: operand must be integer
            if not TypeUtils.is_integer_type(operand_type):
                raise TypeCheckError(
                    f"Operand of '~' must be integer, found {operand_type}",
                    source_loc=expr.operand.source_loc
                )
            expr.expr_type = operand_type

        elif expr.op == '-':
            # Negation: operand must be integer
            if not TypeUtils.is_integer_type(operand_type):
                raise TypeCheckError(
                    f"Operand of '-' must be integer, found {operand_type}",
                    source_loc=expr.operand.source_loc
                )
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

        Checks:
        - Argument types match parameters
        - Return type
        - Mode compatibility between caller and callee
        """
        # Get function symbol and declaration
        if not isinstance(expr.func, HIRIdentifier):
            raise TypeCheckError(
                "Function pointers not yet supported",
                source_loc=expr.func.source_loc if hasattr(expr.func, 'source_loc') else None
            )

        func_symbol = expr.func.symbol

        # Look up HIR function declaration from program
        func_decl = None
        for decl in self.program.declarations:
            if isinstance(decl, HIRFunctionDecl) and decl.name == func_symbol.name:
                func_decl = decl
                break

        if not func_decl:
            raise TypeCheckError(
                f"Function '{func_symbol.name}' not found",
                source_loc=expr.source_loc
            )

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

        # Check mode compatibility
        self._check_call_mode_compatibility(func_symbol.name, func_decl, expr.source_loc)

        # Set return type
        if func_decl.return_type:
            expr.expr_type = func_decl.return_type
        else:
            # Void function
            expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

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

        # Validate transition=auto doesn't conflict with preserves(STATUS)
        if transition == ModeTransition.AUTO:
            if func_decl.preserves_attr and 'STATUS' in func_decl.preserves_attr.registers:
                raise TypeCheckError(
                    f"Function '{func_name}' cannot use transition=auto with #[preserves(STATUS)]\n"
                    f"  transition=auto requires modifying STATUS to switch modes, which conflicts with preservation",
                    source_loc=source_loc
                )

        # Mixed-mode calls are allowed - code generation will handle wrappers based on transition mode

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
        if not TypeUtils.is_integer_type(index_type):
            raise TypeCheckError(
                f"Array index must be integer, found {index_type}",
                source_loc=expr.index.source_loc
            )

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

    def check_assignment(self, expr: HIRAssignment) -> TypeInfo:
        """Type check assignment."""
        target_type = self.check_expression(expr.target)
        value_type = self.check_expression(expr.value, target_type)

        # Types must match exactly
        if not TypeUtils.types_equal(target_type, value_type):
            raise TypeCheckError(
                f"Type mismatch in assignment\n"
                f"  Target: {target_type}\n"
                f"  Value: {value_type}",
                source_loc=expr.source_loc
            )

        expr.expr_type = target_type
        return target_type
