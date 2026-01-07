"""
HIR → MIR lowering.

Transforms type-checked HIR into MIR suitable for code generation.
"""

from typing import Optional, Dict, Any, List, Union
from r65.compiler.hir import (
    HIRProgram, HIRDeclaration, HIRFunctionDecl, HIRStaticDecl, HIRConstDecl,
    HIRStructDecl, HIREnumDecl,
    HIRStatement, HIRBlock, HIRLetStmt, HIRTupleLetStmt, HIRExprStmt, HIRReturnStmt,
    HIRIfStmt, HIRWhileStmt, HIRBreakStmt, HIRContinueStmt, HIRAsmStmt,
    HIRExpression, HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr, HIRIdentifier,
    HIRFunctionAddress, HIRRegister, HIRBinaryOp, HIRUnaryOp, HIRTypeCast, HIRAssignment,
    HIRFunctionCall, HIRMethodCall, HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf,
    HIRArrayFillExpr, HIRArrayLiteralExpr, HIRStringLiteral, HIRStructLiteralExpr,
    HIRMatchExpression, HIRPattern, HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern, HIRIdentifierPattern, HIROrPattern,
    RegisterLetBinding, VariableLetBinding,
    RegisterBinding, VariableBinding,
    SymbolKind,
    ModeTransition
)
from r65.compiler.hir.attributes import StorageKind

from r65.compiler.mir.nodes import (
    MIRInstruction, MIRProgram, MIRFunction, BasicBlock,
    VirtualRegister, HardwareRegister, Immediate, FunctionPointer, MemoryLocation,
    Load, Store, LoadIndirect, StoreIndirect, Move, TypeConvert, BinaryOp, UnaryOp, Compare, BitTest,
    Jump, CondBranch, JumpTable, Return, ReturnFromInterrupt, Call, Argument, ArgumentMechanism,
    SetMode, SaveRegister, RestoreRegister,
    Push, Pull,
    MemoryFill, BlockCopy, ROMDataRef,
    InlineAsm,
)

from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.mir.register_tracker import RegisterAliasTracker
from r65.compiler.mir.cfg import CFGBuilder
from r65.compiler.mir.mode_tracker import MIRModeTracker
from r65.compiler.mir.builder_helpers import TypeSizeCalculator, MemoryLocationBuilder
from r65.compiler.mir.context import LoweringContext
from r65.compiler.mir.lowerers.expression import ExpressionLowerer
from r65.compiler.mir.lowerers.match import MatchLowerer
from r65.compiler.mir.lowerers.call import CallLowerer
from r65.compiler.mir.lowerers.assignment import AssignmentLowerer
from r65.compiler.mir.lowerers.condition import ConditionLowerer
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState, XModeState
from r65.compiler.hir import ModeTransition
from r65.compiler.errors import MIRLoweringError


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
        # Shared lowering context (mutable state)
        self.ctx = LoweringContext()

        # Initialize lowerers
        self.expr_lowerer = ExpressionLowerer(self)
        self.match_lowerer = MatchLowerer(self)
        self.call_lowerer = CallLowerer(self)
        self.assign_lowerer = AssignmentLowerer(self)
        self.cond_lowerer = ConditionLowerer(self)

        # ROM data section tracking for array literals
        self._rom_data_counter = 0
        self._rom_data_sections: List[ROMDataRef] = []

        # Current HIR program being lowered (for symbol table lookups)
        self._hir_program: Optional[HIRProgram] = None

    # ========================================================================
    # Context Property Accessors (for gradual migration)
    # ========================================================================
    # These properties delegate to ctx for backward compatibility during
    # the refactoring process. They will be removed once all code uses ctx.

    @property
    def current_function(self) -> Optional[MIRFunction]:
        return self.ctx.current_function

    @current_function.setter
    def current_function(self, value: Optional[MIRFunction]):
        self.ctx.current_function = value

    @property
    def current_block(self) -> Optional[BasicBlock]:
        return self.ctx.current_block

    @current_block.setter
    def current_block(self, value: Optional[BasicBlock]):
        self.ctx.current_block = value

    @property
    def cfg_builder(self) -> Optional[CFGBuilder]:
        return self.ctx.cfg_builder

    @cfg_builder.setter
    def cfg_builder(self, value: Optional[CFGBuilder]):
        self.ctx.cfg_builder = value

    @property
    def current_mode(self) -> ProcessorMode:
        return self.ctx.current_mode

    @current_mode.setter
    def current_mode(self, value: ProcessorMode):
        self.ctx.current_mode = value

    @property
    def symbol_to_vreg(self) -> Dict[int, VirtualRegister]:
        return self.ctx.symbol_to_vreg

    @property
    def loop_stack(self) -> List[tuple[int, int]]:
        return self.ctx.loop_stack

    @property
    def function_decls(self) -> Dict[str, HIRFunctionDecl]:
        return self.ctx.function_decls

    @property
    def has_init_start(self) -> bool:
        return self.ctx.has_init_start

    @has_init_start.setter
    def has_init_start(self, value: bool):
        self.ctx.has_init_start = value

    def build_program(self, hir_program: HIRProgram) -> MIRProgram:
        """
        Lower HIR program to MIR.

        Args:
            hir_program: Type-checked HIR program

        Returns:
            MIRProgram ready for code generation
        """
        # Reset ROM data tracking for this program
        self._rom_data_counter = 0
        self._rom_data_sections = []

        # Store reference to HIR program for symbol table lookups
        self._hir_program = hir_program

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
            statics=hir_program.statics,
            constants=hir_program.constants,
            structs=hir_program.structs,
            enums=hir_program.enums,
            symbol_table=hir_program.symbol_table,
            stack_attr=hir_program.stack_attr,
            rom_data_sections=self._rom_data_sections
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
            source_loc=hir_func.source_loc,  # Propagate source location
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

        # Register function parameters with alias tracker and allocate vregs
        # Three parameter types need handling:
        # 1. Register aliases (param @ A): track in alias tracker
        # 2. Variable-bound (param @ VAR): track as alias to static variable
        # 3. Stack parameters (param): allocate virtual register
        #
        # Track stack parameters for offset calculation
        stack_params = []

        for idx, param in enumerate(hir_func.parameters):
            if isinstance(param.binding, RegisterBinding):
                # Register-aliased parameter: add to alias tracker
                hw_reg = HardwareRegister(param.binding.register_name)
                self.current_function.alias_tracker.add_alias(
                    param.symbol,
                    hw_reg,
                    param.symbol.scope_id
                )
            elif isinstance(param.binding, VariableBinding):
                # Variable-bound parameter: treat it as an alias to the bound variable
                # When the parameter is referenced in the function, it should load from the bound variable
                # The bound variable already exists and has a memory allocation
                # No setup needed here - lowering expressions handles the load
                pass
            else:
                # Stack parameter: allocate a virtual register for it
                # The function prologue will load from stack into this vreg
                param_vreg = self.current_function.vreg_allocator.alloc(
                    param.param_type,
                    f"param_{param.name}"
                )
                self.symbol_to_vreg[id(param.symbol)] = param_vreg

                # Track for offset calculation
                stack_params.append((idx, param, param_vreg))

        # Calculate stack parameter offsets for prologue generation
        # Stack-relative addressing: LDA offset,S
        #
        # Stack layout after JSR/JSL (caller pushes params right-to-left):
        #     | param N             |  <- higher offset (pushed first)
        #     | ...                 |
        #     | param 0             |  <- SP + return_addr_size + 1 (pushed last)
        #     | return addr (2-3b)  |  <- SP + 1
        #     SP ->
        #
        # Parameters with lower index are at lower stack offsets
        if stack_params:
            return_addr_size = 3 if hir_func.is_far else 2
            current_offset = return_addr_size + 1  # First param starts after return address

            for idx, param, param_vreg in stack_params:
                mir_func.stack_param_offsets[idx] = current_offset
                mir_func.param_to_vreg[idx] = param_vreg

                # Advance offset by parameter size
                param_size = self._get_type_size(param.param_type)
                current_offset += param_size

        # If this is an entry point function, call __init_start() first
        if hir_func.is_entry and self.has_init_start:
            # Emit call to __init_start()
            self.emit(Call(
                function="__init_start",
                args=[],
                returns=[],
                is_far=False,
                bank_attr=None  # __init_start is always near, no DBR management
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

        # Add implicit Return for blocks that don't end with a terminator
        # This handles functions without explicit return statements
        if self.current_block and not self._block_has_terminator():
            # For interrupt handlers, emit ReturnFromInterrupt
            if hir_func.interrupt_attr:
                # Pop all registers (reverse order of pushes in prologue)
                self.emit(Pull(register=HardwareRegister('DBR')))     # PLB
                self.emit(Pull(register=HardwareRegister('D')))       # PLD
                self.emit(Pull(register=HardwareRegister('Y')))       # PLY
                self.emit(Pull(register=HardwareRegister('X')))       # PLX
                self.emit(Pull(register=HardwareRegister('A')))       # PLA
                self.emit(Pull(register=HardwareRegister('STATUS')))  # PLP
                self.emit(ReturnFromInterrupt())
            else:
                # Regular function: add implicit Return with no values
                self.emit(Return(values=[]))

        # Find exit blocks
        mir_func.exit_block_ids = self.cfg_builder.find_exit_blocks()

        # Perform mode tracking analysis
        mode_tracker = MIRModeTracker(mir_func)
        success = mode_tracker.analyze()
        if not success:
            raise MIRLoweringError(f"Mode tracking failed for function '{mir_func.name}': mode conflicts detected")

        return mir_func

    def lower_block(self, block: HIRBlock):
        """
        Lower a block of statements.

        Stops processing if current block gets a terminator (e.g., return/break).
        This handles dead code elimination when const-folded if statements
        contain return statements.

        Args:
            block: HIR block
        """
        for stmt in block.statements:
            # Stop if we already have a terminator (dead code elimination)
            if self._block_has_terminator():
                break
            self.lower_statement(stmt)

    def lower_statement(self, stmt: HIRStatement):
        """
        Lower HIR statement to MIR instructions.

        Args:
            stmt: HIR statement
        """
        if isinstance(stmt, HIRLetStmt):
            self.lower_let_statement(stmt)
        elif isinstance(stmt, HIRTupleLetStmt):
            self.lower_tuple_let_statement(stmt)
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
        elif isinstance(stmt, HIRAsmStmt):
            self.lower_asm_statement(stmt)
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

    def lower_tuple_let_statement(self, stmt: HIRTupleLetStmt):
        """
        Lower tuple destructuring let binding.

        Example: let (a, b) = func_returning_tuple();

        Return values are in A, X, Y registers (in order).
        We capture the values we need and ignore the rest.
        """
        # Evaluate the initializer (typically a function call)
        # This returns the first value; other values are in registers
        init_value = self.lower_expression(stmt.initializer)

        # Return registers in order: A, X, Y
        return_registers = ['A', 'X', 'Y']

        # Capture each binding from the corresponding return register
        for i, (name, symbol, var_type) in enumerate(zip(stmt.names, stmt.symbols, stmt.var_types)):
            if i >= len(return_registers):
                # Can't capture more than 3 return values
                break

            reg_name = return_registers[i]
            hw_reg = HardwareRegister(reg_name)

            # Allocate virtual register for this binding
            vreg = self.current_function.vreg_allocator.alloc(var_type, name)
            self.symbol_to_vreg[id(symbol)] = vreg

            # Move from hardware register to virtual register
            # For the first value (A), we can use the init_value directly if it's already A
            if i == 0 and isinstance(init_value, HardwareRegister) and init_value.name == 'A':
                # Already have it in the right place
                self.emit(Move(dest=vreg, source=init_value, type_info=var_type))
            else:
                # Read from the return register
                self.emit(Move(dest=vreg, source=hw_reg, type_info=var_type))

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
                raise MIRLoweringError(f"Interrupt handler '{self.current_function.name}' cannot return values")
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

        elif isinstance(expr, HIREnumVariantExpr):
            # Enum variant → Immediate (resolved value)
            return Immediate(expr.value)

        elif isinstance(expr, HIRIdentifier):
            symbol = expr.symbol

            # Check if this is a function identifier (function pointer)
            from r65.compiler.hir.symbol_table import SymbolKind
            from r65.compiler.hir.types import FunctionTypeInfo
            if symbol.kind == SymbolKind.FUNCTION:
                # Function identifier used as a value - load function address
                vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, f"fn_ptr_{symbol.name}")
                func_ptr = FunctionPointer(function_name=symbol.name)
                self.emit(Move(dest=vreg, source=func_ptr, type_info=expr.expr_type))
                return vreg

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

        elif isinstance(expr, HIRFunctionAddress):
            # Function address - load function pointer into virtual register
            # The actual address will be resolved during linking
            # For now, we create a virtual register and store a symbolic reference
            vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, f"fn_ptr_{expr.function_name}")
            # We'll emit a special "load function address" operation
            # The function name will be resolved during code generation
            func_ptr = FunctionPointer(function_name=expr.function_name)
            self.emit(Move(dest=vreg, source=func_ptr, type_info=expr.expr_type))
            return vreg

        elif isinstance(expr, HIRBinaryOp):
            return self.expr_lowerer.lower_binary_op(expr)

        elif isinstance(expr, HIRUnaryOp):
            return self.expr_lowerer.lower_unary_op(expr)

        elif isinstance(expr, HIRAssignment):
            return self.assign_lowerer.lower_assignment(expr)

        elif isinstance(expr, HIRFunctionCall):
            return self.call_lowerer.lower_function_call(expr)

        elif isinstance(expr, HIRMethodCall):
            return self.call_lowerer.lower_method_call(expr)

        elif isinstance(expr, HIRTypeCast):
            return self.expr_lowerer.lower_type_cast(expr)

        elif isinstance(expr, HIRArrayIndex):
            return self.expr_lowerer.lower_array_index(expr)

        elif isinstance(expr, HIRFieldAccess):
            return self.expr_lowerer.lower_field_access(expr)

        elif isinstance(expr, HIRDereference):
            return self.expr_lowerer.lower_dereference(expr)

        elif isinstance(expr, HIRAddressOf):
            return self.expr_lowerer.lower_addressof(expr)

        elif isinstance(expr, HIRMatchExpression):
            return self.match_lowerer.lower_match_expression(expr)

        else:
            # Unsupported expression type (placeholder)
            # Allocate placeholder virtual register
            vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, "unsupported")
            return vreg

    # ========================================================================
    # Expression Lowering (delegated to ExpressionLowerer)
    # ========================================================================
    # See lowerers/expression.py for: lower_binary_op, lower_unary_op,
    # lower_type_cast, lower_array_index, lower_field_access,
    # lower_dereference, lower_addressof

    # ========================================================================
    # Match Expression Lowering (delegated to MatchLowerer)
    # ========================================================================
    # See lowerers/match.py for: lower_match_expression, pattern matching,
    # jump table optimization, and conditional branch lowering

    # ========================================================================
    # Assignment Lowering (delegated to AssignmentLowerer)
    # ========================================================================
    # See lowerers/assignment.py for: lower_assignment, identifier/register/
    # field/array/dereference assignments

    # ========================================================================
    # Call Lowering (delegated to CallLowerer)
    # ========================================================================
    # See lowerers/call.py for: lower_function_call, lower_method_call,
    # mode transitions, and argument passing

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

        Optimization: If the condition is a compile-time constant,
        only emit the taken branch (dead code elimination).
        """
        # Try to evaluate condition at compile time
        const_result = self._try_eval_const_condition(stmt.condition)

        if const_result is True:
            # Condition is always true - only emit then branch
            self.lower_block(stmt.then_block)
            return

        if const_result is False:
            # Condition is always false - only emit else branch (if any)
            if stmt.else_block:
                self.lower_block(stmt.else_block)
            return

        # Non-constant condition - generate full control flow
        # Create target blocks
        then_block = self.cfg_builder.new_block()
        merge_block = self.cfg_builder.new_block()
        else_block = self.cfg_builder.new_block() if stmt.else_block else merge_block

        # Lower condition with short-circuit evaluation
        self.cond_lowerer.lower_condition(stmt.condition, then_block.block_id, else_block.block_id)

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
            self.lower_block(stmt.else_block)
            # Jump to merge
            if not self._block_has_terminator():
                self.emit(Jump(target=merge_block.block_id))
                self.cfg_builder.add_edge(else_block, merge_block)

        # Continue at merge block
        self.current_block = merge_block

    # ========================================================================
    # Condition Lowering (delegated to ConditionLowerer)
    # ========================================================================
    # See lowerers/condition.py for: lower_condition, bit test detection,
    # short-circuit evaluation

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

        # Try to evaluate condition at compile time
        const_cond = None
        if stmt.condition:
            const_cond = self._try_eval_const_condition(stmt.condition)

        if const_cond is True or stmt.is_infinite:
            # Infinite loop: `loop { }` or `while true { }`
            self.emit(Jump(target=body_block.block_id))
            self.cfg_builder.add_edge(header_block, body_block)
        elif const_cond is False:
            # Dead loop: `while false { }` - skip entirely
            self.emit(Jump(target=exit_block.block_id))
            self.cfg_builder.add_edge(header_block, exit_block)
            self.current_block = exit_block
            return
        elif stmt.condition:
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
            # loop: infinite loop (fallback)
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
            raise MIRLoweringError("Break statement outside of loop")

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
            raise MIRLoweringError("Continue statement outside of loop")

        continue_target, _ = self.loop_stack[-1]
        self.emit(Jump(target=continue_target))

        # Add CFG edge
        continue_block = self.cfg_builder.get_block(continue_target)
        self.cfg_builder.add_edge(self.current_block, continue_block)

    def lower_asm_statement(self, stmt: HIRAsmStmt):
        """
        Lower inline assembly statement.

        Emits raw assembly instructions verbatim.
        """
        self.emit(InlineAsm(instructions=stmt.instructions))

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _emit_conditional_set(
        self,
        condition_vreg: VirtualRegister,
        true_when_nonzero: bool,
        result_type: Any,
        hint: str = "cond_result"
    ) -> VirtualRegister:
        r"""
        Emit conditional set pattern: result = condition ? 1 : 0

        Creates control flow:
            condition check
                |
            CondBranch
            /        \
        true_block  false_block
            |          |
        result=1   result=0
            \          /
             merge_block

        Args:
            condition_vreg: Virtual register holding condition value
            true_when_nonzero: If True, result=1 when condition!=0
                               If False, result=1 when condition==0
            result_type: Type for result register
            hint: Name hint for result vreg

        Returns:
            VirtualRegister holding result (0 or 1)
        """
        result = self.current_function.vreg_allocator.alloc(result_type, hint)

        # Create blocks
        true_block = self.cfg_builder.new_block()
        false_block = self.cfg_builder.new_block()
        merge_block = self.cfg_builder.new_block()

        # Emit conditional branch
        if true_when_nonzero:
            true_target = true_block.block_id
            false_target = false_block.block_id
        else:
            true_target = false_block.block_id
            false_target = true_block.block_id

        self.emit(CondBranch(
            condition=condition_vreg,
            true_target=true_target,
            false_target=false_target,
            comparison='!='
        ))
        self.cfg_builder.add_edge(self.current_block, true_block)
        self.cfg_builder.add_edge(self.current_block, false_block)

        # True block: result = 1
        self.current_block = true_block
        self.emit(Move(dest=result, source=Immediate(1), type_info=result_type))
        self.emit(Jump(target=merge_block.block_id))
        self.cfg_builder.add_edge(true_block, merge_block)

        # False block: result = 0
        self.current_block = false_block
        self.emit(Move(dest=result, source=Immediate(0), type_info=result_type))
        self.emit(Jump(target=merge_block.block_id))
        self.cfg_builder.add_edge(false_block, merge_block)

        # Continue in merge block
        self.current_block = merge_block

        return result

    # Note: Call-related helper methods (_emit_call_with_mode_transition,
    # _lower_call_arguments, _get_argument_mechanism) moved to lowerers/call.py

    def _create_offset_memloc(
        self,
        base_memloc: MemoryLocation,
        offset: int,
        symbol: Any
    ) -> MemoryLocation:
        """
        Create a memory location with an offset from a base location.

        Used for array indexing and struct field access.

        Args:
            base_memloc: Base memory location
            offset: Byte offset from base
            symbol: Symbol for reference

        Returns:
            MemoryLocation at base + offset
        """
        if base_memloc.address is not None:
            # Address known - compute absolute address
            return MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=base_memloc.address + offset,
                symbol=symbol,
                is_volatile=base_memloc.is_volatile
            )
        else:
            # Address not known - store offset for later resolution
            return MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=offset,  # Just the offset
                symbol=symbol,
                is_volatile=base_memloc.is_volatile
            )

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
        # Handle variable-bound parameters
        if symbol.kind == SymbolKind.PARAMETER:
            # Find the HIR parameter in the current function's parameter list
            if self.current_function:
                for param in self.current_function.parameters:
                    if param.symbol == symbol:
                        # Check if parameter has a variable binding
                        if hasattr(param, 'binding') and param.binding:
                            if isinstance(param.binding, VariableBinding):
                                # Parameter is bound to a variable - return that variable's location
                                return self.get_memory_location(param.binding.variable_symbol)
                        # If no variable binding, fall through to default (stack parameter)
                        break

        # Get storage attribute from symbol's definition
        if symbol.kind == SymbolKind.STATIC_VAR:
            static_decl = symbol.definition
            # Check if definition has storage_attr (HIR node)
            if hasattr(static_decl, 'storage_attr') and static_decl.storage_attr:
                storage_attr = static_decl.storage_attr
                storage_type = storage_attr.storage_kind.value  # Get string value from enum
                return MemoryLocation(
                    storage_type=storage_type,
                    address=storage_attr.address,
                    symbol=symbol,
                    is_volatile=storage_attr.storage_kind == StorageKind.HW
                )

        # Default: unknown storage (will be allocated later)
        return MemoryLocation(
            storage_type='unknown',
            address=None,
            symbol=symbol,
            is_volatile=False
        )

    def _get_type_size(self, type_info) -> int:
        """
        Get size in bytes for a type.

        Delegates to TypeSizeCalculator for consistent type size handling.

        Args:
            type_info: TypeInfo

        Returns:
            Size in bytes
        """
        from r65.compiler.hir.types import FunctionTypeInfo

        # Handle function types specially (FunctionTypeInfo has is_far)
        if isinstance(type_info, FunctionTypeInfo):
            return 3 if type_info.is_far else 2

        # Use TypeSizeCalculator for standard types
        type_str = str(type_info)

        # Basic types
        if type_str in ('u8', 'i8', 'bool'):
            return 1
        elif type_str in ('u16', 'i16'):
            return 2

        # Pointer/function types by string
        if type_str.startswith('far'):
            return 3
        elif type_str.startswith('near') or type_str.startswith('fn'):
            return 2

        # Delegate to TypeSizeCalculator for complex types
        return TypeSizeCalculator.get_size(type_info)

    def _generate_init_start_function(self, statics: List[HIRStaticDecl]) -> MIRFunction:
        """
        Generate __init_start() function for static initialization.

        This function initializes all static variables with initializers.
        It should be called at the beginning of the program's entry point.

        Initialization strategies:
        - Array fill [value; count]: Loop fill (efficient for zero fills)
        - Array literal [a, b, c]: Block copy from ROM (MVN instruction)
        - Scalar values: Simple store

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
            # Get memory location for the static
            mem_loc = self.get_memory_location(static_decl.symbol)
            initializer = static_decl.initializer

            # Handle different initializer types
            if isinstance(initializer, HIRArrayFillExpr):
                # Array fill: use MemoryFill instruction (loop-based)
                self._emit_array_fill_init(static_decl, mem_loc, initializer)

            elif isinstance(initializer, HIRArrayLiteralExpr):
                # Array literal: use BlockCopy instruction (MVN from ROM)
                self._emit_array_literal_init(static_decl, mem_loc, initializer)

            elif isinstance(initializer, HIRStringLiteral):
                # String literal: use BlockCopy instruction (MVN from ROM)
                self._emit_string_literal_init(static_decl, mem_loc, initializer)

            elif isinstance(initializer, HIRStructLiteralExpr):
                # Struct literal: use BlockCopy instruction (MVN from ROM)
                self._emit_struct_literal_init(static_decl, mem_loc, initializer)

            elif self._is_function_pointer_init(initializer):
                # Function pointer: emit Store with FunctionPointer directly
                func_name = self._get_function_name(initializer)
                func_ptr = FunctionPointer(function_name=func_name)
                self.emit(Store(
                    source=func_ptr,
                    dest=mem_loc,
                    type_info=static_decl.var_type
                ))

            else:
                # Scalar value: simple store
                init_value = self.lower_expression(initializer)
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

    def _is_function_pointer_init(self, initializer: HIRExpression) -> bool:
        """Check if initializer is a function pointer (identifier or address-of function)."""
        # Direct function reference: handler
        if isinstance(initializer, HIRIdentifier):
            if initializer.symbol and initializer.symbol.kind == SymbolKind.FUNCTION:
                return True
        # Explicit function address: &handler (HIRFunctionAddress)
        if isinstance(initializer, HIRFunctionAddress):
            return True
        return False

    def _get_function_name(self, initializer: HIRExpression) -> str:
        """Extract function name from function pointer initializer."""
        if isinstance(initializer, HIRIdentifier):
            return initializer.name
        if isinstance(initializer, HIRFunctionAddress):
            return initializer.function_name
        raise MIRLoweringError(f"Cannot extract function name from {type(initializer).__name__}")

    def _emit_array_fill_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        fill_expr: 'HIRArrayFillExpr'
    ):
        """
        Emit MemoryFill instruction for array fill expression.

        Example: [0; 256] fills 256 elements with 0 using a loop.

        Args:
            static_decl: Static declaration being initialized
            mem_loc: Memory location of the array
            fill_expr: HIR array fill expression
        """
        from r65.compiler.hir.types import ArrayTypeInfo
        from r65.compiler.hir import HIRIntegerLiteral, HIRTypeCast

        # Get element type and size
        array_type = static_decl.var_type
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self._get_type_size(array_type.element_type)
        else:
            element_size = 1  # Default to 1 byte

        # Get fill value - must be constant for efficient code gen
        # Extract constant value without emitting instructions
        fill_value = self._extract_constant_value(fill_expr.fill_value)
        if fill_value is None:
            fill_value = 0  # Fallback

        self.emit(MemoryFill(
            dest=mem_loc,
            fill_value=fill_value,
            count=fill_expr.count,
            element_size=element_size
        ))

    def _extract_constant_value(self, expr: HIRExpression) -> Optional[int]:
        """
        Extract constant value from expression without emitting instructions.

        Handles integer literals, casts of literals, and boolean literals.

        Args:
            expr: HIR expression to evaluate

        Returns:
            Constant value if extractable, None otherwise
        """
        from r65.compiler.hir import HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr, HIRTypeCast

        if isinstance(expr, HIRIntegerLiteral):
            return expr.value
        elif isinstance(expr, HIRBooleanLiteral):
            return 1 if expr.value else 0
        elif isinstance(expr, HIREnumVariantExpr):
            return expr.value
        elif isinstance(expr, HIRTypeCast):
            # Recursively extract from the inner expression
            inner_value = self._extract_constant_value(expr.expr)
            return inner_value
        else:
            # Not a constant expression
            return None

    def _try_eval_const_condition(self, expr: HIRExpression) -> Optional[bool]:
        """
        Try to evaluate a condition expression at compile time.

        Used for dead code elimination in if/while statements when
        conditions are compile-time constants.

        Args:
            expr: HIR condition expression to evaluate

        Returns:
            True/False if condition is compile-time constant, None otherwise
        """
        from r65.compiler.hir import (
            HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr,
            HIRTypeCast, HIRBinaryOp, HIRUnaryOp, HIRIdentifier
        )

        if isinstance(expr, HIRBooleanLiteral):
            return expr.value

        elif isinstance(expr, HIRIntegerLiteral):
            # Non-zero is truthy
            return expr.value != 0

        elif isinstance(expr, HIREnumVariantExpr):
            return expr.value != 0

        elif isinstance(expr, HIRIdentifier):
            # Look up const values from symbol table
            if self._hir_program and self._hir_program.symbol_table:
                symbol = self._hir_program.symbol_table.lookup(expr.name)
                if symbol and symbol.kind == SymbolKind.CONST and symbol.const_value is not None:
                    return symbol.const_value != 0
            return None

        elif isinstance(expr, HIRTypeCast):
            inner = self._try_eval_const_condition(expr.expr)
            return inner

        elif isinstance(expr, HIRUnaryOp):
            if expr.op == '!':
                inner = self._try_eval_const_condition(expr.operand)
                if inner is not None:
                    return not inner
            elif expr.op == '~':
                inner = self._try_eval_const_int(expr.operand)
                if inner is not None:
                    return (~inner) != 0
            return None

        elif isinstance(expr, HIRBinaryOp):
            return self._try_eval_const_binary(expr)

        return None

    def _try_eval_const_int(self, expr: HIRExpression) -> Optional[int]:
        """
        Try to evaluate an expression to a constant integer at compile time.
        """
        from r65.compiler.hir import (
            HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr,
            HIRTypeCast, HIRBinaryOp, HIRUnaryOp, HIRIdentifier
        )

        if isinstance(expr, HIRIntegerLiteral):
            return expr.value

        elif isinstance(expr, HIRBooleanLiteral):
            return 1 if expr.value else 0

        elif isinstance(expr, HIREnumVariantExpr):
            return expr.value

        elif isinstance(expr, HIRIdentifier):
            if self._hir_program and self._hir_program.symbol_table:
                symbol = self._hir_program.symbol_table.lookup(expr.name)
                if symbol and symbol.kind == SymbolKind.CONST and symbol.const_value is not None:
                    return symbol.const_value
            return None

        elif isinstance(expr, HIRTypeCast):
            return self._try_eval_const_int(expr.expr)

        elif isinstance(expr, HIRUnaryOp):
            operand = self._try_eval_const_int(expr.operand)
            if operand is None:
                return None
            if expr.op == '-':
                return -operand
            elif expr.op == '~':
                return ~operand
            elif expr.op == '!':
                return 0 if operand else 1
            return None

        elif isinstance(expr, HIRBinaryOp):
            left = self._try_eval_const_int(expr.left)
            right = self._try_eval_const_int(expr.right)
            if left is None or right is None:
                return None

            op = expr.op
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left // right if right != 0 else None
            elif op == '%':
                return left % right if right != 0 else None
            elif op == '&':
                return left & right
            elif op == '|':
                return left | right
            elif op == '^':
                return left ^ right
            elif op == '<<':
                return left << right
            elif op == '>>':
                return left >> right
            # Comparison operators return 0 or 1
            elif op == '==':
                return 1 if left == right else 0
            elif op == '!=':
                return 1 if left != right else 0
            elif op == '<':
                return 1 if left < right else 0
            elif op == '>':
                return 1 if left > right else 0
            elif op == '<=':
                return 1 if left <= right else 0
            elif op == '>=':
                return 1 if left >= right else 0
            return None

        return None

    def _try_eval_const_binary(self, expr: HIRBinaryOp) -> Optional[bool]:
        """
        Try to evaluate a binary operation to a constant boolean.
        """
        op = expr.op

        # Logical operators with short-circuit potential
        if op == '&&':
            left = self._try_eval_const_condition(expr.left)
            if left is False:
                return False
            right = self._try_eval_const_condition(expr.right)
            if left is True and right is not None:
                return right
            return None

        elif op == '||':
            left = self._try_eval_const_condition(expr.left)
            if left is True:
                return True
            right = self._try_eval_const_condition(expr.right)
            if left is False and right is not None:
                return right
            return None

        # Comparison operators
        elif op in ('==', '!=', '<', '>', '<=', '>='):
            left = self._try_eval_const_int(expr.left)
            right = self._try_eval_const_int(expr.right)
            if left is None or right is None:
                return None

            if op == '==':
                return left == right
            elif op == '!=':
                return left != right
            elif op == '<':
                return left < right
            elif op == '>':
                return left > right
            elif op == '<=':
                return left <= right
            elif op == '>=':
                return left >= right

        # Arithmetic/bitwise operators - evaluate as int, convert to bool
        else:
            result = self._try_eval_const_int(expr)
            if result is not None:
                return result != 0

        return None

    def _emit_array_literal_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        literal_expr: 'HIRArrayLiteralExpr'
    ):
        """
        Emit BlockCopy instruction for array literal expression.

        Example: [1, 2, 3, 4] stores data in ROM and copies to RAM.
        Also handles arrays of struct literals like [Card { ... }, Card { ... }].

        Args:
            static_decl: Static declaration being initialized
            mem_loc: Memory location of the array
            literal_expr: HIR array literal expression
        """
        from r65.compiler.hir.types import ArrayTypeInfo, StructTypeInfo

        # Get element type and size
        array_type = static_decl.var_type
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self._get_type_size(array_type.element_type)
            element_type = array_type.element_type
        else:
            element_size = 1  # Default to 1 byte
            element_type = None

        # Extract constant values from all elements without emitting instructions
        data_bytes = []
        for elem in literal_expr.elements:
            # Check if element is a struct literal
            if isinstance(elem, HIRStructLiteralExpr):
                struct_bytes = self._extract_struct_literal_bytes(elem)
                data_bytes.extend(struct_bytes)
            else:
                value = self._extract_constant_value(elem)
                if value is None:
                    value = 0  # Fallback for non-constant

                # Store as little-endian bytes
                if element_size == 1:
                    data_bytes.append(value & 0xFF)
                elif element_size == 2:
                    data_bytes.append(value & 0xFF)
                    data_bytes.append((value >> 8) & 0xFF)
                else:
                    # Handle larger types if needed
                    for i in range(element_size):
                        data_bytes.append((value >> (i * 8)) & 0xFF)

        # Create ROM data reference using variable name
        label = f"__{static_decl.name}_data"

        rom_data = ROMDataRef(
            label=label,
            data=data_bytes,
            element_size=element_size
        )
        self._rom_data_sections.append(rom_data)

        # Emit block copy instruction
        self.emit(BlockCopy(
            dest=mem_loc,
            rom_data=rom_data,
            count=len(data_bytes)
        ))

    def _extract_struct_literal_bytes(self, struct_expr: 'HIRStructLiteralExpr') -> List[int]:
        """
        Extract constant bytes from a struct literal expression.

        Args:
            struct_expr: HIR struct literal expression

        Returns:
            List of bytes representing the struct data
        """
        from r65.compiler.frontend import ast

        # Find struct definition
        struct_decl = struct_expr.struct_decl
        if struct_decl is None:
            symbol = self._hir_program.symbol_table.lookup(struct_expr.struct_name)
            if symbol:
                struct_decl = symbol.definition

        if struct_decl is None:
            raise MIRLoweringError(f"Cannot find struct definition for {struct_expr.struct_name}")

        # Calculate field offsets and sizes
        total_size = 0
        field_info = {}  # name -> (offset, size)

        if isinstance(struct_decl, HIRStructDecl):
            for field in struct_decl.fields:
                field_size = self._get_type_size(field.field_type)
                field_info[field.name] = (field.offset, field_size)
                total_size = max(total_size, field.offset + field_size)
        elif isinstance(struct_decl, ast.StructDecl):
            from r65.compiler.hir.types import TypeResolver
            from r65.compiler.hir.const_eval import ConstEvaluator
            type_resolver = TypeResolver(self._hir_program.symbol_table, ConstEvaluator(self._hir_program.symbol_table))
            current_offset = 0
            for field in struct_decl.fields:
                field_type = type_resolver.resolve_type(field.field_type)
                field_size = self._get_type_size(field_type)
                field_info[field.name] = (current_offset, field_size)
                current_offset += field_size
            total_size = current_offset
        else:
            raise MIRLoweringError(f"Unexpected struct definition type: {type(struct_decl).__name__}")

        # Create byte array for struct data
        data_bytes = [0] * total_size

        # Fill in field values at their offsets
        for field_init in struct_expr.fields:
            if field_init.name not in field_info:
                continue

            offset, field_size = field_info[field_init.name]
            value = self._extract_constant_value(field_init.value)
            if value is None:
                value = 0  # Fallback for non-constant

            # Store as little-endian bytes at the field's offset
            for i in range(field_size):
                data_bytes[offset + i] = (value >> (i * 8)) & 0xFF

        return data_bytes

    def _emit_string_literal_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        string_literal: 'HIRStringLiteral'
    ):
        """
        Emit BlockCopy instruction for string literal initialization.

        String literals are converted to byte arrays. The processed_bytes field
        contains the escape-sequence-processed byte values from type checking.
        Zero-padding is applied to match the declared array size.

        Args:
            static_decl: Static declaration being initialized
            mem_loc: Memory location of the array
            string_literal: HIR string literal expression
        """
        from r65.compiler.hir.types import ArrayTypeInfo

        # Get the array size from the declared type
        array_type = static_decl.var_type
        if isinstance(array_type, ArrayTypeInfo):
            array_size = array_type.size
        else:
            # Shouldn't happen if type checking passed
            array_size = len(string_literal.processed_bytes)

        # Get processed bytes (escape sequences already handled by type checker)
        data_bytes = list(string_literal.processed_bytes)

        # Zero-pad to match array size
        while len(data_bytes) < array_size:
            data_bytes.append(0)

        # Create ROM data reference using variable name
        label = f"__{static_decl.name}_data"

        rom_data = ROMDataRef(
            label=label,
            data=data_bytes,
            element_size=1  # Strings are always u8 arrays
        )
        self._rom_data_sections.append(rom_data)

        # Emit block copy instruction
        self.emit(BlockCopy(
            dest=mem_loc,
            rom_data=rom_data,
            count=len(data_bytes)
        ))

    def _emit_struct_literal_init(
        self,
        static_decl: HIRStaticDecl,
        mem_loc: MemoryLocation,
        struct_expr: 'HIRStructLiteralExpr'
    ):
        """
        Emit BlockCopy instruction for struct literal expression.

        Example: Player { x: 10, y: 20, health: 100 } stores data in ROM and copies to RAM.

        Args:
            static_decl: Static declaration being initialized
            mem_loc: Memory location of the struct
            struct_expr: HIR struct literal expression
        """
        from r65.compiler.hir.types import StructTypeInfo
        from r65.compiler.hir import HIRStructDecl
        from r65.compiler.frontend import ast

        # Get struct definition to know field sizes and offsets
        struct_decl = struct_expr.struct_decl
        if struct_decl is None:
            # Look up from symbol table
            symbol = self._hir_program.symbol_table.lookup(struct_expr.struct_name)
            if symbol:
                struct_decl = symbol.definition

        if struct_decl is None:
            raise MIRLoweringError(f"Cannot find struct definition for {struct_expr.struct_name}")

        # Calculate total size of struct and field offsets
        # Handle both HIR and AST struct declarations
        total_size = 0
        field_info = {}  # name -> (offset, size)

        if isinstance(struct_decl, HIRStructDecl):
            # HIR struct has pre-computed offsets
            for field in struct_decl.fields:
                field_size = self._get_type_size(field.field_type)
                field_info[field.name] = (field.offset, field_size)
                total_size = max(total_size, field.offset + field_size)
        elif isinstance(struct_decl, ast.StructDecl):
            # AST struct - need to compute offsets
            from r65.compiler.hir.types import TypeResolver
            from r65.compiler.hir.const_eval import ConstEvaluator
            type_resolver = TypeResolver(self._hir_program.symbol_table, ConstEvaluator(self._hir_program.symbol_table))
            current_offset = 0
            for field in struct_decl.fields:
                field_type = type_resolver.resolve_type(field.field_type)
                field_size = self._get_type_size(field_type)
                field_info[field.name] = (current_offset, field_size)
                current_offset += field_size
            total_size = current_offset
        else:
            raise MIRLoweringError(f"Unexpected struct definition type: {type(struct_decl).__name__}")

        # Create byte array for struct data
        data_bytes = [0] * total_size

        # Fill in field values at their offsets
        for field_init in struct_expr.fields:
            if field_init.name not in field_info:
                continue

            offset, field_size = field_info[field_init.name]
            value = self._extract_constant_value(field_init.value)
            if value is None:
                value = 0  # Fallback for non-constant

            # Store as little-endian bytes at the field's offset
            for i in range(field_size):
                data_bytes[offset + i] = (value >> (i * 8)) & 0xFF

        # Create ROM data reference using variable name
        label = f"__{static_decl.name}_data"

        rom_data = ROMDataRef(
            label=label,
            data=data_bytes,
            element_size=1  # Struct is treated as a block of bytes
        )
        self._rom_data_sections.append(rom_data)

        # Emit block copy instruction
        self.emit(BlockCopy(
            dest=mem_loc,
            rom_data=rom_data,
            count=len(data_bytes)
        ))
