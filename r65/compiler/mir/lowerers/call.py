# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
)
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate,
    Move, Store, Call, Argument, ArgumentMechanism, Rotate, BankByte,
    SetMode, Push, Pull, TraitDispatch, LoadIndirect,
)
from r65.compiler.hir.types import ArrayTypeInfo, BasicTypeInfo
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState
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

        Also handles method calls (expr.method(args)) which have been transformed
        by the type checker with method_call_info set.

        Args:
            call_expr: HIR function call expression

        Returns:
            VirtualRegister or HardwareRegister holding return value, or None for void
        """
        # Check if this is a method call (set by type checker)
        if call_expr.method_call_info:
            if call_expr.method_call_info.get('is_type_id'):
                return self._lower_type_id_call(call_expr)
            if call_expr.method_call_info.get('is_trait_dispatch'):
                return self._lower_trait_dispatch_call(call_expr)
            return self._lower_method_call(call_expr)

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
                    raise MIRLoweringError(f"Function call to {func_symbol.name}: function not found in HIR", source_loc=call_expr.source_loc)
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

        is_tuple_return = False
        if return_type:
            # Skip vreg allocation for tuple returns - lower_multi_assignment handles
            # tuple destructuring directly from A, X, Y registers
            from r65.compiler.hir.types import TupleTypeInfo
            if isinstance(return_type, TupleTypeInfo):
                is_tuple_return = True
            else:
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
                mode_attr=None,
                bank_attr=None,
                builtin_name=call_expr.builtin_name
            ))
            if returns:
                return returns[0]
            return HardwareRegister('A') if is_tuple_return else None

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
                mode_attr=None,  # No mode attribute for indirect calls
                bank_attr=None,  # No bank attribute for indirect calls
                builtin_name=call_expr.builtin_name
            ))
            if returns:
                return returns[0]
            return HardwareRegister('A') if is_tuple_return else None

        # Direct call - handle mode transitions using helper
        self._emit_call_with_mode_transition(func_decl, args, returns, call_expr.builtin_name)

        # Return result (or None for void functions).
        # For tuple returns, the first element is always in A.
        if returns:
            return returns[0]
        return HardwareRegister('A') if is_tuple_return else None

    # ========================================================================
    # Method Call Lowering
    # ========================================================================

    def lower_method_call(self, call_expr: HIRMethodCall) -> VirtualRegister:
        """
        Lower method call (e.g., value.rotate_left(3) or array.len()).

        Currently supports:
        - rotate_left, rotate_right methods on integers
        - len() method on arrays

        Args:
            call_expr: HIRMethodCall expression

        Returns:
            VirtualRegister holding the result
        """
        # Handle bank_byte() method on far pointers
        if call_expr.method_name == 'bank_byte':
            receiver_value = self.builder.lower_expression(call_expr.receiver)
            result_vreg = self.ctx.alloc_vreg(BasicTypeInfo('u8'), "bank_byte")
            self.emit(BankByte(
                dest=result_vreg,
                source=receiver_value,
            ))
            return result_vreg

        # Handle len() method on arrays
        if call_expr.method_name == 'len':
            # Get the receiver's type (must be ArrayTypeInfo, validated in type checker)
            receiver_type = call_expr.receiver.expr_type
            if not isinstance(receiver_type, ArrayTypeInfo):
                raise MIRLoweringError(
                    f"len() receiver must be an array type, got {type(receiver_type).__name__}",
                    source_loc=call_expr.source_loc
                )

            # Get array size - this is always a compile-time constant
            array_size = receiver_type.size

            # Create result register (u16 to hold array lengths)
            result_vreg = self.ctx.alloc_vreg(BasicTypeInfo('u16'), "len_result")

            # Emit Move instruction to load the constant size
            self.emit(Move(
                dest=result_vreg,
                source=Immediate(array_size),
                type_info=BasicTypeInfo('u16')
            ))

            return result_vreg

        # Lower the receiver (value being rotated)
        receiver_value = self.builder.lower_expression(call_expr.receiver)

        # Get rotation count from argument (already validated as constant 1-8 in type checker)
        count_arg = call_expr.args[0]
        if not isinstance(count_arg, HIRIntegerLiteral):
            raise MIRLoweringError(
                f"rotation count must be a constant integer literal, got {type(count_arg).__name__}",
                source_loc=call_expr.source_loc
            )
        count = count_arg.value

        # Determine direction
        if call_expr.method_name == 'rotate_left':
            direction = 'left'
        elif call_expr.method_name == 'rotate_right':
            direction = 'right'
        else:
            raise MIRLoweringError(f"Unknown method: {call_expr.method_name}", source_loc=call_expr.source_loc)

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

    def _lower_method_call(self, call_expr: HIRFunctionCall) -> Union[VirtualRegister, HardwareRegister, None]:
        """
        Lower impl block method call.

        Method calls like `player.take_damage(5)` have been transformed by the
        type checker to have method_call_info containing:
        - mangled_name: The mangled function name (e.g., "Player__take_damage")
        - self_arg: The self argument (address of receiver)
        - func_decl: The HIR function declaration

        Args:
            call_expr: HIR function call with method_call_info set

        Returns:
            VirtualRegister or HardwareRegister holding return value, or None for void
        """
        method_info = call_expr.method_call_info
        mangled_name = method_info['mangled_name']
        self_arg = method_info['self_arg']
        func_decl = method_info['func_decl']

        # Lower self argument
        self_vreg = self.builder.lower_expression(self_arg)

        # Create argument list with self as first argument
        args = []

        # Self parameter (always stack-passed)
        self_param = func_decl.parameters[0]  # First parameter is self
        args.append(Argument(
            value=self_vreg,
            mechanism=ArgumentMechanism.STACK,
            location=None,
            param_type=self_param.param_type
        ))

        # Lower remaining arguments
        for i, arg_expr in enumerate(call_expr.args):
            param = func_decl.parameters[i + 1]  # +1 to skip self
            arg_vreg = self.builder.lower_expression(arg_expr)

            # Determine argument mechanism and location from parameter binding
            mechanism = ArgumentMechanism.STACK
            location = None
            if param.binding:
                if isinstance(param.binding, RegisterBinding):
                    mechanism = ArgumentMechanism.REGISTER
                    location = HardwareRegister(param.binding.register_name)
                elif isinstance(param.binding, VariableBinding):
                    mechanism = ArgumentMechanism.VARIABLE
                    location = param.binding.variable_symbol

            args.append(Argument(
                value=arg_vreg,
                mechanism=mechanism,
                location=location,
                param_type=param.param_type
            ))

        # Allocate virtual register for return value
        returns = []
        if func_decl.return_type:
            from r65.compiler.hir.types import TupleTypeInfo
            if not isinstance(func_decl.return_type, TupleTypeInfo):
                result_vreg = self.ctx.alloc_vreg(
                    func_decl.return_type,
                    f"call_{mangled_name}_result"
                )
                returns.append(result_vreg)

        # Emit call with mode transition handling
        self._emit_call_with_mode_transition(func_decl, args, returns, None)

        return returns[0] if returns else None

    def _lower_type_id_call(self, call_expr: HIRFunctionCall) -> VirtualRegister:
        """Lower type_id() call on trait pointer — loads TypeId byte at offset 0."""
        method_info = call_expr.method_call_info
        self_arg = method_info['self_arg']
        self_vreg = self.builder.lower_expression(self_arg)

        result = self.builder.current_function.vreg_allocator.alloc(
            BasicTypeInfo('u8'), 'type_id'
        )
        self.emit(LoadIndirect(
            dest=result,
            pointer=self_vreg,
            is_far=False,
            type_info=BasicTypeInfo('u8'),
            offset=0,
        ))
        return result

    def _lower_trait_dispatch_call(self, call_expr: HIRFunctionCall) -> Union[VirtualRegister, HardwareRegister, None]:
        """
        Lower trait method dispatch call.

        Trait dispatch calls are method calls on trait pointers (e.g., drawable.draw())
        that resolve at runtime via TypeId-indexed jump tables.

        method_call_info contains:
        - trait_name: Name of the trait (e.g., "Drawable")
        - method_name: Method being called (e.g., "draw")
        - method_index: Index of method in trait's method list
        - self_arg: The trait pointer expression
        - trait_is_far: Whether the trait method is far

        Args:
            call_expr: HIR function call with is_trait_dispatch set

        Returns:
            VirtualRegister holding return value, or None for void
        """
        method_info = call_expr.method_call_info
        trait_name = method_info['trait_name']
        method_name = method_info['method_name']
        method_index = method_info['method_index']
        self_arg = method_info['self_arg']
        trait_is_far = method_info['trait_is_far']

        # Lower self argument (trait pointer)
        self_vreg = self.builder.lower_expression(self_arg)

        # Build argument list: self pointer passed in Y register (DBR:Y addressing),
        # then user args on stack
        from r65.compiler.hir.types import PointerTypeInfo
        args = []
        self_ptr_type = self_arg.expr_type if hasattr(self_arg, 'expr_type') else PointerTypeInfo(pointee_type=BasicTypeInfo('u8'))
        args.append(Argument(
            value=self_vreg,
            mechanism=ArgumentMechanism.SELF_Y,
            location=HardwareRegister('Y'),
            param_type=self_ptr_type
        ))

        # Lower remaining arguments (trait methods can have params beyond self)
        # Look up trait definition for param types
        trait_symbol = self.builder._hir_program.symbol_table.lookup(trait_name)
        trait_def = trait_symbol.definition if trait_symbol else None
        trait_method = None
        if trait_def:
            for m in trait_def.methods:
                if m.name == method_name:
                    trait_method = m
                    break

        for i, arg_expr in enumerate(call_expr.args):
            arg_vreg = self.builder.lower_expression(arg_expr)
            param_type = None
            if trait_method and i < len(trait_method.params):
                param_type = trait_method.params[i].param_type
            args.append(Argument(
                value=arg_vreg,
                mechanism=ArgumentMechanism.STACK,
                location=None,
                param_type=param_type
            ))

        # Allocate return value register
        returns = []
        return_type = call_expr.expr_type if hasattr(call_expr, 'expr_type') else None
        if return_type:
            from r65.compiler.hir.types import TupleTypeInfo
            if isinstance(return_type, TupleTypeInfo):
                pass  # Tuple returns handled by caller
            elif return_type != BasicTypeInfo('void'):
                result_vreg = self.ctx.alloc_vreg(
                    return_type,
                    f"trait_dispatch_{trait_name}_{method_name}_result"
                )
                returns.append(result_vreg)

        # Determine if self pointer is far (24-bit)
        self_is_far = isinstance(self_ptr_type, PointerTypeInfo) and self_ptr_type.is_far

        # Emit TraitDispatch MIR node
        self.emit(TraitDispatch(
            trait_name=trait_name,
            method_name=method_name,
            method_index=method_index,
            self_ptr=self_vreg,
            args=args,
            returns=returns,
            is_far=trait_is_far,
            self_is_far=self_is_far,
            callee_return_type=return_type
        ))

        return returns[0] if returns else None

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

        # M flag (bit 5, 0x20) - X is always x16, no need to check
        if from_mode.m_mode != to_mode.m_mode:
            if to_mode.m_mode == ModeState.M8:
                sep_mask |= 0x20  # SEP #$20 for M8
            elif to_mode.m_mode == ModeState.M16:
                rep_mask |= 0x20  # REP #$20 for M16

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
        Emit function call.

        Mode transitions are now automatic - the callee's prologue handles
        REP #$20 for m16 entry mode, and epilogue handles SEP #$20 to restore.

        Args:
            func_decl: Function being called
            args: Prepared arguments
            returns: Virtual registers for return values
            builtin_name: Name of built-in function if this is a built-in call
        """
        # Compute Pascal result space bytes if needed
        pascal_result_bytes = 0
        from r65.compiler.codegen.abi_model import ABIKind
        if self.builder.abi_kind == ABIKind.PASCAL and func_decl.return_type:
            from r65.compiler.hir.types import BasicTypeInfo, NeverTypeInfo
            if (not isinstance(func_decl.return_type, NeverTypeInfo) and
                not (isinstance(func_decl.return_type, BasicTypeInfo) and
                     func_decl.return_type.name == 'void')):
                from r65.compiler.codegen.type_utils import get_type_size
                pascal_result_bytes = get_type_size(func_decl.return_type)

        self.emit(Call(
            function=func_decl.name,
            args=args,
            returns=returns,
            is_far=func_decl.is_far,
            mode_attr=func_decl.mode_attr,
            bank_attr=func_decl.bank_attr,
            builtin_name=builtin_name,
            callee_entry_m_mode=func_decl.entry_m_mode,
            callee_exit_m_mode=func_decl.exit_m_mode,
            callee_return_type=func_decl.return_type,
            preserves_attr=func_decl.preserves_attr,
            pascal_result_bytes=pascal_result_bytes,
        ))

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
            param_type = None
            if func_decl and i < len(func_decl.parameters):
                param = func_decl.parameters[i]
                mechanism, location = self._get_argument_mechanism(param, arg_value)
                param_type = param.param_type  # Get parameter type for correct stack push size
            else:
                # Indirect call or no binding info - use stack
                mechanism = ArgumentMechanism.STACK
                location = None

            args.append(Argument(value=arg_value, mechanism=mechanism, location=location, param_type=param_type))

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
        # Pascal ABI: all params go on stack regardless of binding
        from r65.compiler.codegen.abi_model import ABIKind
        if self.builder.abi_kind == ABIKind.PASCAL:
            return ArgumentMechanism.STACK, None

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
