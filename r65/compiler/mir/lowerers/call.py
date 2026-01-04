"""
Call lowerer: HIR function/method calls → MIR instructions.

Handles function calls, method calls (rotate), mode transitions,
and argument passing mechanisms (register, variable-bound, stack).
"""

from typing import TYPE_CHECKING, Union, Optional, List, Any

from r65.compiler.hir import (
    HIRFunctionCall, HIRMethodCall, HIRFunctionDecl,
    HIRIdentifier, HIRIntegerLiteral,
    RegisterBinding, VariableBinding,
    ModeTransition,
)
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate,
    Move, Store, Call, Argument, ArgumentMechanism, Rotate,
    SetMode, Push, Pull,
)
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState, XModeState
from r65.compiler.errors import MIRLoweringError

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.mir.context import LoweringContext


class CallLowerer:
    """
    Lowers HIR function and method calls to MIR instructions.

    Handles:
    - Direct function calls
    - Indirect function calls (function pointers)
    - Method calls (rotate_left, rotate_right)
    - Mode transitions for cross-mode calls
    - Argument passing mechanisms (register, variable, stack)

    Calls back to builder.lower_expression() for sub-expression recursion.
    """

    def __init__(self, builder: 'MIRBuilder'):
        """
        Initialize call lowerer.

        Args:
            builder: Parent MIR builder for dispatch and helpers
        """
        self.builder = builder

    @property
    def ctx(self) -> 'LoweringContext':
        """Access the lowering context."""
        return self.builder.ctx

    def emit(self, instr):
        """Emit an instruction to the current block."""
        self.builder.emit(instr)

    # ========================================================================
    # Function Call Lowering
    # ========================================================================

    def lower_function_call(self, call_expr: HIRFunctionCall) -> Union[VirtualRegister, HardwareRegister, None]:
        """
        Lower function call with arguments and return values.

        Handles three parameter passing mechanisms:
        - Stack parameters (default)
        - Register alias parameters (param @ A)
        - Variable-bound parameters (param @ VAR)

        Args:
            call_expr: HIR function call expression

        Returns:
            VirtualRegister or HardwareRegister holding return value, or None for void
        """
        # Get function symbol and declaration
        # func is usually HIRIdentifier for direct calls
        # For indirect calls (function pointers), func is an expression with FunctionTypeInfo
        from r65.compiler.hir.symbol_table import SymbolKind
        is_direct_call = isinstance(call_expr.func, HIRIdentifier) and call_expr.func.symbol.kind == SymbolKind.FUNCTION
        is_indirect_call = not is_direct_call

        if is_direct_call:
            # Direct call to a function
            func_symbol = call_expr.func.symbol

            # Handle built-in function calls
            if call_expr.builtin_name:
                # Built-in call - no function declaration needed
                func_decl = None
                func_ptr_vreg = None
            else:
                # Regular function call - look up HIR function declaration
                func_decl = self.ctx.function_decls.get(func_symbol.name)
                if not func_decl:
                    raise MIRLoweringError(f"Function call to {func_symbol.name}: function not found in HIR")
                func_ptr_vreg = None
        else:
            # Indirect call (function pointer)
            # Lower the function expression to get the virtual register holding the pointer
            func_ptr_vreg = self.builder.lower_expression(call_expr.func)
            # For indirect calls, we don't have func_decl, so we'll use the function type
            # from the expression type
            func_decl = None

        # Prepare arguments using helper
        args = self._lower_call_arguments(call_expr, func_decl)

        # Allocate virtual register(s) for return value(s)
        returns = []
        return_type = None

        if func_decl:
            # Direct call - get return type from function declaration
            return_type = func_decl.return_type
        else:
            # Indirect call - get return type from function type
            from r65.compiler.hir.types import FunctionTypeInfo
            func_type = call_expr.func.expr_type
            if isinstance(func_type, FunctionTypeInfo):
                return_type = func_type.return_type

        if return_type:
            # For now, assume single return value
            # TODO: Handle multiple return values (tuples)
            call_name = func_decl.name if func_decl else "indirect_call"
            result_vreg = self.ctx.alloc_vreg(
                return_type,
                f"call_{call_name}_result"
            )
            returns.append(result_vreg)

        # Handle built-in calls
        if call_expr.builtin_name:
            # Built-in call - no mode transition handling needed
            self.emit(Call(
                function=func_symbol.name,
                args=args,
                returns=returns,
                is_far=False,
                bank_attr=None,
                builtin_name=call_expr.builtin_name
            ))
            return returns[0] if returns else None

        # For indirect calls, skip mode transition handling (we don't have mode info)
        if is_indirect_call:
            # Emit indirect call
            from r65.compiler.hir.types import FunctionTypeInfo
            func_type = call_expr.func.expr_type
            is_far = isinstance(func_type, FunctionTypeInfo) and func_type.is_far

            self.emit(Call(
                function=func_ptr_vreg,  # VirtualRegister holding function pointer
                args=args,
                returns=returns,
                is_far=is_far,
                bank_attr=None,  # No bank attribute for indirect calls
                builtin_name=call_expr.builtin_name
            ))
            return returns[0] if returns else None

        # Direct call - handle mode transitions using helper
        self._emit_call_with_mode_transition(func_decl, args, returns, call_expr.builtin_name)

        # Return result (or None for void functions)
        return returns[0] if returns else None

    # ========================================================================
    # Method Call Lowering
    # ========================================================================

    def lower_method_call(self, call_expr: HIRMethodCall) -> VirtualRegister:
        """
        Lower method call (e.g., value.rotate_left(3)).

        Currently only supports rotate_left and rotate_right methods.

        Args:
            call_expr: HIRMethodCall expression

        Returns:
            VirtualRegister holding the result
        """
        # Lower the receiver (value being rotated)
        receiver_value = self.builder.lower_expression(call_expr.receiver)

        # Get rotation count from argument (already validated as constant 1-8 in type checker)
        count_arg = call_expr.args[0]
        assert isinstance(count_arg, HIRIntegerLiteral), "Rotation count must be a constant"
        count = count_arg.value

        # Determine direction
        if call_expr.method_name == 'rotate_left':
            direction = 'left'
        elif call_expr.method_name == 'rotate_right':
            direction = 'right'
        else:
            raise MIRLoweringError(f"Unknown rotate method: {call_expr.method_name}")

        # Create result register
        result_vreg = self.ctx.alloc_vreg(call_expr.expr_type, "rotate_result")

        # Emit Rotate instruction
        self.emit(Rotate(
            dest=result_vreg,
            source=receiver_value,
            direction=direction,
            count=count,
            type_info=call_expr.expr_type
        ))

        return result_vreg

    # ========================================================================
    # Mode Transition Helpers
    # ========================================================================

    def _emit_mode_transition(self, from_mode: ProcessorMode, to_mode: ProcessorMode):
        """
        Emit instructions to transition from one processor mode to another.

        Generates SEP/REP instructions to change M and X flags.

        Args:
            from_mode: Current processor mode
            to_mode: Target processor mode
        """
        if from_mode == to_mode:
            return  # No transition needed

        # Build mask for mode changes
        sep_mask = 0  # Bits to set (8-bit mode)
        rep_mask = 0  # Bits to clear (16-bit mode)

        # M flag (bit 5, 0x20)
        if from_mode.m_mode != to_mode.m_mode:
            if to_mode.m_mode == ModeState.M8:
                sep_mask |= 0x20  # SEP #$20 for M8
            elif to_mode.m_mode == ModeState.M16:
                rep_mask |= 0x20  # REP #$20 for M16

        # X flag (bit 4, 0x10)
        if from_mode.x_mode != to_mode.x_mode:
            if to_mode.x_mode == XModeState.X8:
                sep_mask |= 0x10  # SEP #$10 for X8
            elif to_mode.x_mode == XModeState.X16:
                rep_mask |= 0x10  # REP #$10 for X16

        # Emit instructions
        if sep_mask:
            self.emit(SetMode(mask=sep_mask, is_set=True))
        if rep_mask:
            self.emit(SetMode(mask=rep_mask, is_set=False))

    def _emit_call_with_mode_transition(
        self,
        func_decl: HIRFunctionDecl,
        args: List[Argument],
        returns: List[VirtualRegister],
        builtin_name: Optional[str] = None
    ):
        """
        Emit function call with mode transition handling.

        Handles three cases:
        1. transition=none: No wrapper (default)
        2. transition=auto: Callee handles it
        3. transition=caller + mode mismatch: Caller handles it

        Args:
            func_decl: Function being called
            args: Prepared arguments
            returns: Virtual registers for return values
            builtin_name: Name of built-in function if this is a built-in call
        """
        caller_mode = self.ctx.current_mode
        callee_mode = ProcessorMode.from_attribute(func_decl.mode_attr) if func_decl.mode_attr else ProcessorMode.unknown()
        transition = func_decl.mode_attr.transition if func_decl.mode_attr and hasattr(func_decl.mode_attr, 'transition') else ModeTransition.NONE

        # Check if mode transition needed
        mode_mismatch = (
            caller_mode != callee_mode and
            caller_mode.is_fully_known() and
            callee_mode.is_fully_known()
        )

        if not mode_mismatch or transition != ModeTransition.CALLER:
            # No wrapper needed (transition=none, transition=auto, or same mode)
            self.emit(Call(
                function=func_decl.name,
                args=args,
                returns=returns,
                is_far=func_decl.is_far,
                bank_attr=func_decl.bank_attr,
                builtin_name=builtin_name
            ))
            return

        # Caller handles mode transition
        preserves_status = (
            func_decl.preserves_attr and
            'STATUS' in func_decl.preserves_attr.registers
        )

        if preserves_status:
            # Use SEP/REP before and after
            self._emit_mode_transition(caller_mode, callee_mode)
            self.emit(Call(
                function=func_decl.name,
                args=args,
                returns=returns,
                is_far=func_decl.is_far,
                bank_attr=func_decl.bank_attr,
                builtin_name=builtin_name
            ))
            self._emit_mode_transition(callee_mode, caller_mode)
        else:
            # Use PHP/PLP wrapper
            self.emit(Push(register=HardwareRegister('STATUS')))
            self._emit_mode_transition(caller_mode, callee_mode)
            self.emit(Call(
                function=func_decl.name,
                args=args,
                returns=returns,
                is_far=func_decl.is_far,
                bank_attr=func_decl.bank_attr,
                builtin_name=builtin_name
            ))
            self.emit(Pull(register=HardwareRegister('STATUS')))

    # ========================================================================
    # Argument Lowering
    # ========================================================================

    def _lower_call_arguments(
        self,
        call_expr: HIRFunctionCall,
        func_decl: Optional[HIRFunctionDecl]
    ) -> List[Argument]:
        """
        Lower function call arguments to MIR Arguments.

        Handles three parameter passing mechanisms:
        - Register alias (param @ A)
        - Variable-bound (param @ VAR)
        - Stack (default)

        Args:
            call_expr: HIR function call
            func_decl: Function declaration (None for indirect calls)

        Returns:
            List of MIR Arguments
        """
        args = []

        for i, arg_expr in enumerate(call_expr.args):
            arg_value = self.builder.lower_expression(arg_expr)

            # Determine mechanism based on parameter binding (if available)
            if func_decl and i < len(func_decl.parameters):
                param = func_decl.parameters[i]
                mechanism, location = self._get_argument_mechanism(param, arg_value)
            else:
                # Indirect call or no binding info - use stack
                mechanism = ArgumentMechanism.STACK
                location = None

            args.append(Argument(value=arg_value, mechanism=mechanism, location=location))

        return args

    def _get_argument_mechanism(
        self,
        param: Any,  # HIRParameter
        arg_value: Union[VirtualRegister, HardwareRegister, Immediate]
    ) -> tuple:
        """
        Determine argument passing mechanism and emit setup code.

        Args:
            param: Parameter declaration
            arg_value: Lowered argument value

        Returns:
            (mechanism, location) tuple
        """
        if isinstance(param.binding, RegisterBinding):
            # Register alias parameter
            # Note: Don't emit Move here - let Call instruction handler set up arguments
            # This avoids duplicate setup code
            mechanism = ArgumentMechanism.REGISTER
            location = HardwareRegister(param.binding.register_name)
            return mechanism, location

        elif isinstance(param.binding, VariableBinding):
            # Variable-bound parameter
            mechanism = ArgumentMechanism.VARIABLE
            location = self.builder.get_memory_location(param.binding.variable_symbol)

            # Store argument to variable location
            self.emit(Store(source=arg_value, dest=location, type_info=param.param_type))

            return mechanism, location

        else:
            # Stack parameter
            return ArgumentMechanism.STACK, None
