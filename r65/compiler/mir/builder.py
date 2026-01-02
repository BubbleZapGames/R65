"""
HIR → MIR lowering.

Transforms type-checked HIR into MIR suitable for code generation.
"""

from typing import Optional, Dict, Any, List, Union
from r65.compiler.hir import (
    HIRProgram, HIRDeclaration, HIRFunctionDecl, HIRStaticDecl, HIRConstDecl,
    HIRStructDecl, HIREnumDecl,
    HIRStatement, HIRBlock, HIRLetStmt, HIRExprStmt, HIRReturnStmt,
    HIRIfStmt, HIRWhileStmt, HIRBreakStmt, HIRContinueStmt,
    HIRExpression, HIRIntegerLiteral, HIRBooleanLiteral, HIRIdentifier,
    HIRRegister, HIRBinaryOp, HIRUnaryOp, HIRTypeCast, HIRAssignment,
    HIRFunctionCall,
    RegisterLetBinding, VariableLetBinding,
    RegisterBinding, VariableBinding,
    SymbolKind,
    ModeTransition
)

from r65.compiler.mir.nodes import (
    MIRInstruction, MIRProgram, MIRFunction, BasicBlock,
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    Load, Store, Move, BinaryOp, UnaryOp,
    Jump, CondBranch, Return, ReturnFromInterrupt, Call, Argument, ArgumentMechanism,
    SetMode, SaveRegister, RestoreRegister,
    Push, Pull,
)

from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.mir.register_tracker import RegisterAliasTracker
from r65.compiler.mir.cfg import CFGBuilder
from r65.compiler.mir.mode_tracker import MIRModeTracker
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState, XModeState
from r65.compiler.hir import ModeTransition


class MIRBuilder:
    """
    Lowers HIR to MIR.

    Multi-pass transformation:
    1. Create MIR program structure
    2. Lower each function to MIR CFG
    3. Track register aliases
    4. Allocate virtual registers
    """

    def __init__(self):
        # Current lowering context
        self.current_function: Optional[MIRFunction] = None
        self.current_block: Optional[BasicBlock] = None
        self.cfg_builder: Optional[CFGBuilder] = None
        self.alias_tracker: Optional[RegisterAliasTracker] = None
        self.current_mode: ProcessorMode = ProcessorMode.unknown()

        # id(Symbol) → VirtualRegister mapping for current function
        self.symbol_to_vreg: Dict[int, VirtualRegister] = {}

        # Loop stack for break/continue (stack of (continue_target, break_target))
        self.loop_stack: List[tuple[int, int]] = []

        # Function name → HIRFunctionDecl mapping (for looking up during calls)
        self.function_decls: Dict[str, HIRFunctionDecl] = {}

        # Track if we generated __init_start() function
        self.has_init_start: bool = False

    def build_program(self, hir_program: HIRProgram) -> MIRProgram:
        """
        Lower HIR program to MIR.

        Args:
            hir_program: Type-checked HIR program

        Returns:
            MIRProgram ready for code generation
        """
        # Build function name → HIRFunctionDecl mapping
        for decl in hir_program.declarations:
            if isinstance(decl, HIRFunctionDecl):
                self.function_decls[decl.name] = decl

        mir_functions = []

        # Check if we need to generate __init_start() for static initialization
        # Include ALL statics with initializers (even zero values)
        # SNES RAM is NOT zeroed on power-on - contents are unpredictable
        statics_with_initializers = [
            d for d in hir_program.declarations
            if isinstance(d, HIRStaticDecl) and d.initializer is not None
        ]

        # Generate __init_start() if there are static initializers
        if statics_with_initializers:
            init_func = self._generate_init_start_function(statics_with_initializers)
            mir_functions.append(init_func)
            self.has_init_start = True

        # Lower each function
        for decl in hir_program.declarations:
            if isinstance(decl, HIRFunctionDecl) and decl.body:
                mir_func = self.lower_function(decl)
                mir_functions.append(mir_func)

        # Create MIR program (keep HIR declarations for statics, etc.)
        return MIRProgram(
            functions=mir_functions,
            statics=[d for d in hir_program.declarations if isinstance(d, HIRStaticDecl)],
            constants=[d for d in hir_program.declarations if isinstance(d, HIRConstDecl)],
            structs=[d for d in hir_program.declarations if isinstance(d, HIRStructDecl)],
            enums=[d for d in hir_program.declarations if isinstance(d, HIREnumDecl)],
            symbol_table=hir_program.symbol_table
        )

    def lower_function(self, hir_func: HIRFunctionDecl) -> MIRFunction:
        """
        Lower a single function to MIR.

        Args:
            hir_func: HIR function declaration

        Returns:
            MIRFunction with CFG
        """
        # Create MIR function structure
        mir_func = MIRFunction(
            name=hir_func.name,
            parameters=hir_func.parameters,
            return_type=hir_func.return_type,
            blocks={},
            entry_block_id=0,
            exit_block_ids=[],
            mode_attr=hir_func.mode_attr,
            preserves_attr=hir_func.preserves_attr,
            bank_attr=hir_func.bank_attr,
            interrupt_attr=hir_func.interrupt_attr,
            is_entry=hir_func.is_entry,
            is_far=hir_func.is_far,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=RegisterAliasTracker()
        )

        # Set current function context
        self.current_function = mir_func
        self.cfg_builder = CFGBuilder(mir_func)
        self.symbol_to_vreg.clear()
        self.loop_stack.clear()

        # Initialize current mode from function's mode attribute
        if hir_func.mode_attr:
            self.current_mode = ProcessorMode.from_attribute(hir_func.mode_attr)
        else:
            self.current_mode = ProcessorMode.unknown()

        # Create entry block
        entry_block = self.cfg_builder.new_block()
        mir_func.entry_block_id = entry_block.block_id
        self.current_block = entry_block

        # If this is an entry point function, call __init_start() first
        if hir_func.is_entry and self.has_init_start:
            # Emit call to __init_start()
            self.emit(Call(
                function="__init_start",
                args=[],
                returns=[],
                is_far=False
            ))

        # Generate interrupt handler entry wrapper if needed
        if hir_func.interrupt_attr:
            # Push all registers (automatic preservation for interrupts)
            self.emit(Push(register=HardwareRegister('STATUS')))  # PHP
            self.emit(Push(register=HardwareRegister('A')))       # PHA
            self.emit(Push(register=HardwareRegister('X')))       # PHX
            self.emit(Push(register=HardwareRegister('Y')))       # PHY
            self.emit(Push(register=HardwareRegister('D')))       # PHD
            self.emit(Push(register=HardwareRegister('DBR')))     # PHB

            # If handler has mode attribute, set the mode (transition=auto)
            # Interrupts can fire from any mode, so we force the handler's mode
            if hir_func.mode_attr and self.current_mode.is_fully_known():
                # Generate SEP/REP to force handler's mode
                # We don't know the incoming mode, so we just set all bits explicitly
                handler_mode = self.current_mode
                sep_mask = 0
                rep_mask = 0

                if handler_mode.m_mode == ModeState.M8:
                    sep_mask |= 0x20
                elif handler_mode.m_mode == ModeState.M16:
                    rep_mask |= 0x20

                if handler_mode.x_mode == XModeState.X8:
                    sep_mask |= 0x10
                elif handler_mode.x_mode == XModeState.X16:
                    rep_mask |= 0x10

                if sep_mask:
                    self.emit(SetMode(mask=sep_mask, is_set=True))
                if rep_mask:
                    self.emit(SetMode(mask=rep_mask, is_set=False))

        # Lower function body
        if hir_func.body:
            self.lower_block(hir_func.body)

        # Find exit blocks
        mir_func.exit_block_ids = self.cfg_builder.find_exit_blocks()

        # Perform mode tracking analysis
        mode_tracker = MIRModeTracker(mir_func)
        success = mode_tracker.analyze()
        if not success:
            raise Exception(f"Mode tracking failed for function '{mir_func.name}': mode conflicts detected")

        return mir_func

    def lower_block(self, block: HIRBlock):
        """
        Lower a block of statements.

        Args:
            block: HIR block
        """
        for stmt in block.statements:
            self.lower_statement(stmt)

    def lower_statement(self, stmt: HIRStatement):
        """
        Lower HIR statement to MIR instructions.

        Args:
            stmt: HIR statement
        """
        if isinstance(stmt, HIRLetStmt):
            self.lower_let_statement(stmt)
        elif isinstance(stmt, HIRExprStmt):
            self.lower_expression(stmt.expr)
        elif isinstance(stmt, HIRReturnStmt):
            self.lower_return_statement(stmt)
        elif isinstance(stmt, HIRIfStmt):
            self.lower_if_statement(stmt)
        elif isinstance(stmt, HIRWhileStmt):
            self.lower_while_statement(stmt)
        elif isinstance(stmt, HIRBreakStmt):
            self.lower_break_statement(stmt)
        elif isinstance(stmt, HIRContinueStmt):
            self.lower_continue_statement(stmt)
        else:
            # Unsupported statement type (placeholder for future expansion)
            pass

    def lower_let_statement(self, stmt: HIRLetStmt):
        """
        Lower let binding.

        Args:
            stmt: HIR let statement
        """
        if isinstance(stmt.binding, RegisterLetBinding):
            # Register alias: track in alias tracker
            hw_reg = HardwareRegister(stmt.binding.register_name)
            self.current_function.alias_tracker.add_alias(
                stmt.symbol,
                hw_reg,
                stmt.symbol.scope_id
            )

            # If there's an initializer, load it into hardware register
            if stmt.initializer:
                init_value = self.lower_expression(stmt.initializer)
                if not (isinstance(init_value, HardwareRegister) and init_value.name == hw_reg.name):
                    # Move to hardware register if not already there
                    self.emit(Move(
                        dest=hw_reg,
                        source=init_value,
                        type_info=stmt.var_type
                    ))

        else:
            # Regular variable: allocate virtual register or use memory
            if stmt.initializer:
                init_value = self.lower_expression(stmt.initializer)

                # Check if symbol has explicit memory location
                if self.has_explicit_location(stmt.symbol):
                    mem_loc = self.get_memory_location(stmt.symbol)
                    self.emit(Store(
                        source=init_value,
                        dest=mem_loc,
                        type_info=stmt.var_type
                    ))
                else:
                    # Allocate virtual register (will map to scratch later)
                    if isinstance(init_value, (VirtualRegister, HardwareRegister)):
                        # Reuse the virtual register from initializer
                        self.symbol_to_vreg[id(stmt.symbol)] = init_value
                    else:
                        # Allocate new virtual register for immediate
                        vreg = self.current_function.vreg_allocator.alloc(stmt.var_type, stmt.name)
                        self.symbol_to_vreg[id(stmt.symbol)] = vreg
                        self.emit(Move(dest=vreg, source=init_value, type_info=stmt.var_type))

    def lower_return_statement(self, stmt: HIRReturnStmt):
        """
        Lower return statement.

        Args:
            stmt: HIR return statement
        """
        # Lower return values
        return_values = []
        for val in stmt.values:
            lowered_val = self.lower_expression(val)
            return_values.append(lowered_val)

        # Check if this is an interrupt handler
        if self.current_function.interrupt_attr:
            # Interrupt handler exit sequence
            # Restore all registers (reverse order of push)
            self.emit(Pull(register=HardwareRegister('DBR')))     # PLB
            self.emit(Pull(register=HardwareRegister('D')))       # PLD
            self.emit(Pull(register=HardwareRegister('Y')))       # PLY
            self.emit(Pull(register=HardwareRegister('X')))       # PLX
            self.emit(Pull(register=HardwareRegister('A')))       # PLA
            self.emit(Pull(register=HardwareRegister('STATUS')))  # PLP (restores mode!)

            # Return from interrupt (RTI)
            # Note: Interrupt handlers shouldn't return values
            if return_values:
                raise Exception(f"Interrupt handler '{self.current_function.name}' cannot return values")
            self.emit(ReturnFromInterrupt())
        else:
            # Regular function return
            self.emit(Return(values=return_values))

    def lower_expression(self, expr: HIRExpression) -> Union[VirtualRegister, HardwareRegister, Immediate]:
        """
        Lower HIR expression, returning operand holding result.

        Args:
            expr: HIR expression

        Returns:
            VirtualRegister, HardwareRegister, or Immediate
        """
        if isinstance(expr, HIRIntegerLiteral):
            # Integer literal → Immediate
            return Immediate(expr.value)

        elif isinstance(expr, HIRBooleanLiteral):
            # Boolean literal → Immediate (0 or 1)
            return Immediate(1 if expr.value else 0)

        elif isinstance(expr, HIRIdentifier):
            symbol = expr.symbol

            # Check if aliased to hardware register
            hw_reg = self.current_function.alias_tracker.get_alias(symbol)
            if hw_reg:
                return hw_reg  # Direct hardware register reference

            # Check if already in a virtual register
            symbol_id = id(symbol)
            if symbol_id in self.symbol_to_vreg:
                return self.symbol_to_vreg[symbol_id]

            # Otherwise, load from memory
            vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, symbol.name)
            mem_loc = self.get_memory_location(symbol)
            self.emit(Load(dest=vreg, source=mem_loc, type_info=expr.expr_type))
            self.symbol_to_vreg[symbol_id] = vreg
            return vreg

        elif isinstance(expr, HIRRegister):
            # Direct hardware register reference
            return HardwareRegister(expr.name)

        elif isinstance(expr, HIRBinaryOp):
            return self.lower_binary_op(expr)

        elif isinstance(expr, HIRUnaryOp):
            return self.lower_unary_op(expr)

        elif isinstance(expr, HIRAssignment):
            return self.lower_assignment(expr)

        elif isinstance(expr, HIRFunctionCall):
            return self.lower_function_call(expr)

        else:
            # Unsupported expression type (placeholder)
            # Allocate placeholder virtual register
            vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, "unsupported")
            return vreg

    def lower_binary_op(self, expr: HIRBinaryOp) -> VirtualRegister:
        """
        Lower binary operation.

        Args:
            expr: HIR binary operation

        Returns:
            VirtualRegister holding result
        """
        # Lower operands
        left = self.lower_expression(expr.left)
        right = self.lower_expression(expr.right)

        # Allocate virtual register for result
        result = self.current_function.vreg_allocator.alloc(expr.expr_type, f"{expr.op}_result")

        # Emit binary operation
        self.emit(BinaryOp(
            dest=result,
            left=left,
            right=right,
            op=expr.op,
            type_info=expr.expr_type
        ))

        return result

    def lower_unary_op(self, expr: HIRUnaryOp) -> VirtualRegister:
        """
        Lower unary operation.

        Args:
            expr: HIR unary operation

        Returns:
            VirtualRegister holding result
        """
        # Lower operand
        operand = self.lower_expression(expr.operand)

        # Ensure operand is a register (not immediate)
        if isinstance(operand, Immediate):
            temp = self.current_function.vreg_allocator.alloc(expr.operand.expr_type, "unary_temp")
            self.emit(Move(dest=temp, source=operand, type_info=expr.operand.expr_type))
            operand = temp

        # Allocate virtual register for result
        result = self.current_function.vreg_allocator.alloc(expr.expr_type, f"{expr.op}_result")

        # Emit unary operation
        self.emit(UnaryOp(
            dest=result,
            operand=operand,
            op=expr.op,
            type_info=expr.expr_type
        ))

        return result

    def lower_assignment(self, expr: HIRAssignment) -> Union[VirtualRegister, HardwareRegister]:
        """
        Lower assignment.

        Args:
            expr: HIR assignment

        Returns:
            VirtualRegister or HardwareRegister with assigned value
        """
        # Lower value
        value = self.lower_expression(expr.value)

        # Lower target
        if isinstance(expr.target, HIRIdentifier):
            symbol = expr.target.symbol

            # Check if aliased to hardware register
            hw_reg = self.current_function.alias_tracker.get_alias(symbol)
            if hw_reg:
                # Move to hardware register
                if not (isinstance(value, HardwareRegister) and value.name == hw_reg.name):
                    self.emit(Move(dest=hw_reg, source=value, type_info=expr.expr_type))
                return hw_reg

            # Check if has explicit memory location
            if self.has_explicit_location(symbol):
                mem_loc = self.get_memory_location(symbol)
                self.emit(Store(source=value, dest=mem_loc, type_info=expr.expr_type))
                return value

            # Otherwise, update virtual register
            symbol_id = id(symbol)
            if symbol_id in self.symbol_to_vreg:
                vreg = self.symbol_to_vreg[symbol_id]
                if vreg != value:
                    self.emit(Move(dest=vreg, source=value, type_info=expr.expr_type))
                return vreg
            else:
                # Allocate new virtual register
                vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, symbol.name)
                self.symbol_to_vreg[symbol_id] = vreg
                if vreg != value:
                    self.emit(Move(dest=vreg, source=value, type_info=expr.expr_type))
                return vreg

        elif isinstance(expr.target, HIRRegister):
            # Direct hardware register assignment
            hw_reg = HardwareRegister(expr.target.name)
            if not (isinstance(value, HardwareRegister) and value.name == hw_reg.name):
                self.emit(Move(dest=hw_reg, source=value, type_info=expr.expr_type))
            return hw_reg

        else:
            # Unsupported target (array index, field access, etc.)
            # Placeholder for future expansion
            return value

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
        # func is usually HIRIdentifier
        if isinstance(call_expr.func, HIRIdentifier):
            func_symbol = call_expr.func.symbol
            # Look up HIR function declaration from our mapping
            func_decl = self.function_decls.get(func_symbol.name)
            if not func_decl:
                raise Exception(f"Function call to {func_symbol.name}: function not found in HIR")
        else:
            # Function pointer call - not yet implemented
            raise Exception("Function pointer calls not yet implemented")

        # Prepare arguments
        args = []
        for i, arg_expr in enumerate(call_expr.args):
            param = func_decl.parameters[i]
            arg_value = self.lower_expression(arg_expr)

            # Determine argument passing mechanism
            if isinstance(param.binding, RegisterBinding):
                # Register alias parameter: param @ A/X/Y
                mechanism = ArgumentMechanism.REGISTER
                location = HardwareRegister(param.binding.register_name)

                # Move argument to hardware register if not already there
                if not (isinstance(arg_value, HardwareRegister) and arg_value.name == location.name):
                    self.emit(Move(dest=location, source=arg_value, type_info=param.param_type))

            elif isinstance(param.binding, VariableBinding):
                # Variable-bound parameter: param @ VAR
                mechanism = ArgumentMechanism.VARIABLE
                location = self.get_memory_location(param.binding.variable_symbol)

                # Store argument to variable location
                self.emit(Store(source=arg_value, dest=location, type_info=param.param_type))

            else:
                # Stack parameter (default)
                mechanism = ArgumentMechanism.STACK
                location = None
                # Stack setup will be handled by code generator

            args.append(Argument(value=arg_value, mechanism=mechanism, location=location))

        # Allocate virtual register(s) for return value(s)
        returns = []
        if func_decl.return_type:
            # For now, assume single return value
            # TODO: Handle multiple return values (tuples)
            result_vreg = self.current_function.vreg_allocator.alloc(
                func_decl.return_type,
                f"call_{func_decl.name}_result"
            )
            returns.append(result_vreg)

        # Check for mode transition requirements
        caller_mode = self.current_mode
        callee_mode = ProcessorMode.from_attribute(func_decl.mode_attr) if func_decl.mode_attr else ProcessorMode.unknown()

        # Determine transition strategy
        transition = ModeTransition.NONE  # Default
        if func_decl.mode_attr and hasattr(func_decl.mode_attr, 'transition'):
            transition = func_decl.mode_attr.transition

        # Generate mode transition wrapper if needed
        mode_mismatch = (caller_mode != callee_mode and
                        caller_mode.is_fully_known() and
                        callee_mode.is_fully_known())

        if mode_mismatch and transition == ModeTransition.CALLER:
            # Caller handles mode transition
            # Check if callee preserves STATUS
            preserves_status = (func_decl.preserves_attr and
                              'STATUS' in func_decl.preserves_attr.registers)

            if preserves_status:
                # Callee preserves STATUS - use explicit SEP/REP before and after
                # Switch to callee's mode
                self._emit_mode_transition(caller_mode, callee_mode)

                # Emit call
                self.emit(Call(
                    function=func_decl.name,
                    args=args,
                    returns=returns,
                    is_far=func_decl.is_far
                ))

                # Restore caller's mode
                self._emit_mode_transition(callee_mode, caller_mode)
            else:
                # Callee doesn't preserve STATUS - use PHP/PLP
                # Save STATUS
                self.emit(Push(register=HardwareRegister('STATUS')))

                # Switch to callee's mode
                self._emit_mode_transition(caller_mode, callee_mode)

                # Emit call
                self.emit(Call(
                    function=func_decl.name,
                    args=args,
                    returns=returns,
                    is_far=func_decl.is_far
                ))

                # Restore STATUS (includes mode bits)
                self.emit(Pull(register=HardwareRegister('STATUS')))
        else:
            # No wrapper needed (transition=none or transition=auto or same mode)
            # For transition=auto, the callee will handle the wrapper
            self.emit(Call(
                function=func_decl.name,
                args=args,
                returns=returns,
                is_far=func_decl.is_far
            ))

        # Return result (or None for void functions)
        return returns[0] if returns else None

    def lower_if_statement(self, stmt: HIRIfStmt):
        r"""
        Lower if statement to conditional branches.

        Creates CFG:
            current_block
                |
                v
            [condition check]
                |
            CondBranch
            /        \
           /          \
        then_block   else_block (optional)
           \          /
            \        /
            merge_block
        """
        # Evaluate condition
        cond_value = self.lower_expression(stmt.condition)

        # Create blocks
        then_block = self.cfg_builder.new_block()
        merge_block = self.cfg_builder.new_block()

        if stmt.else_block:
            else_block = self.cfg_builder.new_block()
            # Emit conditional branch: if condition != 0 goto then, else goto else
            self.emit(CondBranch(
                condition=cond_value,
                true_target=then_block.block_id,
                false_target=else_block.block_id,
                comparison='!='
            ))
            # Add CFG edges
            self.cfg_builder.add_edge(self.current_block, then_block)
            self.cfg_builder.add_edge(self.current_block, else_block)
        else:
            # No else block: if condition != 0 goto then, else goto merge
            self.emit(CondBranch(
                condition=cond_value,
                true_target=then_block.block_id,
                false_target=merge_block.block_id,
                comparison='!='
            ))
            # Add CFG edges
            self.cfg_builder.add_edge(self.current_block, then_block)
            self.cfg_builder.add_edge(self.current_block, merge_block)

        # Lower then branch
        self.current_block = then_block
        self.lower_block(stmt.then_block)
        # Jump to merge (unless then block ends with return/break/continue)
        if not self._block_has_terminator():
            self.emit(Jump(target=merge_block.block_id))
            self.cfg_builder.add_edge(then_block, merge_block)

        # Lower else branch if present
        if stmt.else_block:
            self.current_block = else_block
            # Check if else_block is another if statement (else-if chain)
            if isinstance(stmt.else_block, HIRIfStmt):
                self.lower_if_statement(stmt.else_block)
            else:
                self.lower_block(stmt.else_block)
            # Jump to merge (unless else block ends with terminator)
            if not self._block_has_terminator():
                self.emit(Jump(target=merge_block.block_id))
                self.cfg_builder.add_edge(else_block, merge_block)

        # Continue at merge block
        self.current_block = merge_block

    def lower_while_statement(self, stmt: HIRWhileStmt):
        r"""
        Lower while/loop statement to conditional branches.

        Creates CFG:
            current_block
                |
                v
              Jump
                |
                v
            header_block (condition check)
                |
            CondBranch (or Jump for infinite loop)
            /        \
           /          \
        body_block   exit_block
           |
        [loop body]
           |
          Jump (back to header)
        """
        # Create blocks
        header_block = self.cfg_builder.new_block()
        body_block = self.cfg_builder.new_block()
        exit_block = self.cfg_builder.new_block()

        # Jump from current block to header
        self.emit(Jump(target=header_block.block_id))
        self.cfg_builder.add_edge(self.current_block, header_block)

        # Header: condition check (or infinite loop)
        self.current_block = header_block
        if stmt.condition:
            # while condition: conditional loop
            cond_value = self.lower_expression(stmt.condition)
            self.emit(CondBranch(
                condition=cond_value,
                true_target=body_block.block_id,
                false_target=exit_block.block_id,
                comparison='!='
            ))
            self.cfg_builder.add_edge(header_block, body_block)
            self.cfg_builder.add_edge(header_block, exit_block)
        else:
            # loop: infinite loop
            self.emit(Jump(target=body_block.block_id))
            self.cfg_builder.add_edge(header_block, body_block)

        # Body: track loop context for break/continue
        self.current_block = body_block
        # Push loop context: (continue_target=header, break_target=exit)
        self.loop_stack.append((header_block.block_id, exit_block.block_id))
        self.lower_block(stmt.body)
        self.loop_stack.pop()

        # Jump back to header (unless body ends with break/return)
        if not self._block_has_terminator():
            self.emit(Jump(target=header_block.block_id))
            self.cfg_builder.add_edge(body_block, header_block)

        # Continue at exit block
        self.current_block = exit_block

    def lower_break_statement(self, stmt: HIRBreakStmt):
        """
        Lower break statement.

        Jumps to the exit block of the innermost loop.
        """
        if not self.loop_stack:
            # Should have been caught by type checker, but be defensive
            raise Exception("Break statement outside of loop")

        _, break_target = self.loop_stack[-1]
        self.emit(Jump(target=break_target))

        # Add CFG edge
        break_block = self.cfg_builder.get_block(break_target)
        self.cfg_builder.add_edge(self.current_block, break_block)

    def lower_continue_statement(self, stmt: HIRContinueStmt):
        """
        Lower continue statement.

        Jumps to the header block of the innermost loop.
        """
        if not self.loop_stack:
            # Should have been caught by type checker, but be defensive
            raise Exception("Continue statement outside of loop")

        continue_target, _ = self.loop_stack[-1]
        self.emit(Jump(target=continue_target))

        # Add CFG edge
        continue_block = self.cfg_builder.get_block(continue_target)
        self.cfg_builder.add_edge(self.current_block, continue_block)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _block_has_terminator(self) -> bool:
        """
        Check if current block already has a terminator instruction.

        Terminator instructions: Return, Jump, CondBranch

        Returns:
            True if block ends with terminator, False otherwise
        """
        if not self.current_block or not self.current_block.instructions:
            return False

        last_instr = self.current_block.instructions[-1]
        return isinstance(last_instr, (Return, Jump, CondBranch))

    def emit(self, instruction: MIRInstruction):
        """
        Emit an instruction to the current block.

        Args:
            instruction: MIR instruction to emit
        """
        if self.current_block is not None:
            self.current_block.instructions.append(instruction)

    def has_explicit_location(self, symbol) -> bool:
        """
        Check if symbol has explicit memory location.

        Args:
            symbol: HIR Symbol

        Returns:
            True if symbol has explicit location (static variable)
        """
        return symbol.kind == SymbolKind.STATIC_VAR

    def get_memory_location(self, symbol) -> MemoryLocation:
        """
        Get memory location for symbol.

        Args:
            symbol: HIR Symbol

        Returns:
            MemoryLocation
        """
        # Get storage attribute from symbol's definition
        if symbol.kind == SymbolKind.STATIC_VAR:
            static_decl = symbol.definition
            # Check if definition has storage_attr (HIR node)
            if hasattr(static_decl, 'storage_attr') and static_decl.storage_attr:
                storage_attr = static_decl.storage_attr
                return MemoryLocation(
                    storage_type=storage_attr.storage_type,
                    address=storage_attr.address,
                    symbol=symbol,
                    is_volatile=storage_attr.storage_type == 'hw'
                )

        # Default: unknown storage (will be allocated later)
        return MemoryLocation(
            storage_type='unknown',
            address=None,
            symbol=symbol,
            is_volatile=False
        )

    def _generate_init_start_function(self, statics: List[HIRStaticDecl]) -> MIRFunction:
        """
        Generate __init_start() function for static initialization.

        This function initializes all static variables with non-zero initializers.
        It should be called at the beginning of the program's entry point.

        Args:
            statics: List of HIRStaticDecl with initializers

        Returns:
            MIRFunction for __init_start()
        """
        # Create MIR function structure for __init_start()
        mir_func = MIRFunction(
            name="__init_start",
            parameters=[],
            return_type=None,  # void return
            blocks={},
            entry_block_id=0,
            exit_block_ids=[],
            mode_attr=None,  # No specific mode requirement
            preserves_attr=None,
            bank_attr=None,
            interrupt_attr=None,
            is_entry=False,
            is_far=False,
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=RegisterAliasTracker()
        )

        # Set current function context
        self.current_function = mir_func
        self.cfg_builder = CFGBuilder(mir_func)
        self.symbol_to_vreg.clear()
        self.loop_stack.clear()
        self.current_mode = ProcessorMode.unknown()

        # Create entry block
        entry_block = self.cfg_builder.new_block()
        mir_func.entry_block_id = entry_block.block_id
        self.current_block = entry_block

        # Generate initialization code for each static variable
        for static_decl in statics:
            # Lower the initializer expression
            init_value = self.lower_expression(static_decl.initializer)

            # Get memory location for the static
            mem_loc = self.get_memory_location(static_decl.symbol)

            # Store the initialized value to the static's location
            self.emit(Store(
                source=init_value,
                dest=mem_loc,
                type_info=static_decl.var_type
            ))

        # Emit return instruction
        self.emit(Return(values=[]))

        # Find exit blocks
        mir_func.exit_block_ids = self.cfg_builder.find_exit_blocks()

        return mir_func
