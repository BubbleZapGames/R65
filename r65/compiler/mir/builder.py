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
    HIRFunctionAddress, HIRRegister, HIRBinaryOp, HIRUnaryOp, HIRTypeCast, HIRAssignment,
    HIRFunctionCall, HIRMethodCall, HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf,
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
            stack_attr=hir_program.stack_attr
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

        # Register function parameters with alias tracker and allocate vregs
        # Three parameter types need handling:
        # 1. Register aliases (param @ A): track in alias tracker
        # 2. Variable-bound (param @ VAR): track as alias to static variable
        # 3. Stack parameters (param): allocate virtual register
        for param in hir_func.parameters:
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
                # We don't need to do anything special here - when the param symbol is referenced,
                # the HIR should already have resolved it to refer to the bound variable
                # Actually, we need to create a MemoryLocation for the bound variable
                # But that happens when lowering expressions - no setup needed here
                pass
            else:
                # Stack parameter: allocate a virtual register for it
                # The function prologue will load from stack into this vreg
                # For now, we'll just allocate the vreg and let the rest of the function use it
                param_vreg = self.current_function.vreg_allocator.alloc(
                    param.param_type,
                    f"param_{param.name}"
                )
                self.symbol_to_vreg[id(param.symbol)] = param_vreg

                # TODO: Emit prologue code to load from stack
                # The parameter is at a specific offset from the stack pointer
                # based on the calling convention (right-to-left push order)
                # For now, we're not emitting the load - the vreg will be allocated
                # but never initialized. This is a placeholder.

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
        """
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
