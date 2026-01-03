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
            return self.lower_binary_op(expr)

        elif isinstance(expr, HIRUnaryOp):
            return self.lower_unary_op(expr)

        elif isinstance(expr, HIRAssignment):
            return self.lower_assignment(expr)

        elif isinstance(expr, HIRFunctionCall):
            return self.lower_function_call(expr)

        elif isinstance(expr, HIRMethodCall):
            return self.lower_method_call(expr)

        elif isinstance(expr, HIRTypeCast):
            return self.lower_type_cast(expr)

        elif isinstance(expr, HIRArrayIndex):
            return self.lower_array_index(expr)

        elif isinstance(expr, HIRFieldAccess):
            return self.lower_field_access(expr)

        elif isinstance(expr, HIRDereference):
            return self.lower_dereference(expr)

        elif isinstance(expr, HIRAddressOf):
            return self.lower_addressof(expr)

        elif isinstance(expr, HIRMatchExpression):
            return self.lower_match_expression(expr)

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
        # Check if this is a comparison operator
        comparison_ops = {'==', '!=', '<', '<=', '>', '>='}

        if expr.op in comparison_ops:
            # Comparisons need special handling - convert to boolean value (0 or 1)
            left = self.lower_expression(expr.left)
            right = self.lower_expression(expr.right)

            if expr.op == '==':
                # For equality: compute (left ^ right) and check if zero
                # If left == right, then left ^ right == 0
                temp = self.current_function.vreg_allocator.alloc(expr.left.expr_type, "eq_temp")
                self.emit(BinaryOp(
                    dest=temp,
                    left=left,
                    right=right,
                    op='^',
                    type_info=expr.left.expr_type
                ))
                # Result = 1 when temp == 0 (i.e., when condition is zero)
                return self._emit_conditional_set(temp, true_when_nonzero=False, result_type=expr.expr_type, hint="eq_result")

            elif expr.op == '!=':
                # For inequality: compute (left ^ right) and check if non-zero
                temp = self.current_function.vreg_allocator.alloc(expr.left.expr_type, "ne_temp")
                self.emit(BinaryOp(
                    dest=temp,
                    left=left,
                    right=right,
                    op='^',
                    type_info=expr.left.expr_type
                ))
                # Result = 1 when temp != 0 (i.e., when condition is nonzero)
                return self._emit_conditional_set(temp, true_when_nonzero=True, result_type=expr.expr_type, hint="ne_result")

            else:
                # For <, <=, >, >=: use subtraction to set flags
                # TODO: This is a simplified placeholder implementation
                # A proper implementation needs to check processor flags (carry, negative, overflow)
                # For now, just use subtraction and check if result is non-zero
                temp = self.current_function.vreg_allocator.alloc(expr.left.expr_type, "cmp_temp")
                self.emit(BinaryOp(
                    dest=temp,
                    left=left,
                    right=right,
                    op='-',
                    type_info=expr.left.expr_type
                ))
                # Placeholder: result = 1 when temp != 0
                return self._emit_conditional_set(temp, true_when_nonzero=True, result_type=expr.expr_type, hint=f"{expr.op}_result")
        else:
            # Regular arithmetic/bitwise operation
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

    def lower_type_cast(self, expr: HIRTypeCast) -> VirtualRegister:
        """
        Lower type cast (explicit conversion).

        Handles:
        - Widening: u8→u16 (zero-extend), i8→i16 (sign-extend)
        - Narrowing: u16→u8 (truncate to low byte)
        - Same-size reinterpretation: u8↔i8, u16↔i16 (zero-cost)
        - Boolean conversions: any→bool (0=false, non-zero=true), bool→integer

        Args:
            expr: HIR type cast expression

        Returns:
            VirtualRegister holding converted value
        """
        # Lower the expression being cast
        source_operand = self.lower_expression(expr.expr)
        source_type = expr.expr.expr_type
        target_type = expr.target_type

        # Get type sizes
        source_size = self._get_type_size(source_type)
        target_size = self._get_type_size(target_type)

        # Check if source/target are signed
        source_signed = str(source_type).startswith('i')
        target_signed = str(target_type).startswith('i')

        # Allocate result register
        result = self.current_function.vreg_allocator.alloc(target_type, "cast_result")

        # Same size reinterpretation (zero-cost bit reinterpretation)
        if source_size == target_size:
            # Just move - same bits, different interpretation
            self.emit(Move(dest=result, source=source_operand, type_info=target_type))
            return result

        # Widening conversions (8-bit → 16-bit)
        elif source_size == 1 and target_size == 2:
            # Emit TypeConvert instruction for widening
            # Codegen will handle zero-extension (unsigned) or sign-extension (signed)
            self.emit(TypeConvert(
                dest=result,
                source=source_operand,
                source_type=source_type,
                target_type=target_type
            ))
            return result

        # Narrowing conversions (16-bit → 8-bit)
        elif source_size == 2 and target_size == 1:
            # Emit TypeConvert instruction for narrowing (truncation)
            self.emit(TypeConvert(
                dest=result,
                source=source_operand,
                source_type=source_type,
                target_type=target_type
            ))
            return result

        # Boolean conversions
        elif str(target_type) == 'bool':
            # Convert to boolean: 0 → false, non-zero → true
            # Use the source operand directly (or wrap in temp if it's immediate)
            if isinstance(source_operand, Immediate):
                # For immediate, we can optimize: just return 0 or 1 directly
                # But for consistency with the pattern, we'll still use the helper
                temp = self.current_function.vreg_allocator.alloc(source_type, "bool_temp")
                self.emit(Move(dest=temp, source=source_operand, type_info=source_type))
                source_operand = temp

            # Result = 1 when source_operand != 0
            return self._emit_conditional_set(source_operand, true_when_nonzero=True, result_type=target_type, hint="bool_result")

        elif str(source_type) == 'bool':
            # Convert from boolean to integer: normalize to 0 or 1, then extend if needed
            # Booleans are already 0 or 1, so just move
            self.emit(Move(dest=result, source=source_operand, type_info=target_type))
            return result

        else:
            # Unsupported conversion
            raise Exception(f"Unsupported type cast: {source_type} to {target_type}")

    def lower_array_index(self, expr: HIRArrayIndex) -> VirtualRegister:
        """
        Lower array indexing.

        Computes: array[index] → Load from (base_address + index * element_size)

        Args:
            expr: HIR array index expression

        Returns:
            VirtualRegister holding array element value
        """
        # Get element type and size
        element_type = expr.expr_type
        element_size = self._get_type_size(element_type)

        # Lower index expression
        index_operand = self.lower_expression(expr.index)

        # Get the array symbol (should be HIRIdentifier for static arrays)
        if not isinstance(expr.array, HIRIdentifier):
            raise Exception(f"Array indexing only supports static arrays currently, got: {type(expr.array)}")

        array_symbol = expr.array.symbol

        # Allocate result register
        result = self.current_function.vreg_allocator.alloc(element_type, "array_elem")

        # Calculate offset and create memory location
        if isinstance(index_operand, Immediate):
            # Constant index - compute offset at compile time
            offset = index_operand.value * element_size
            base_memloc = self.get_memory_location(array_symbol)

            # Create offset memory location using helper
            elem_memloc = self._create_offset_memloc(base_memloc, offset, array_symbol)

            # Emit load from the element location
            self.emit(Load(dest=result, source=elem_memloc, type_info=element_type))
            return result
        else:
            # Variable index - use indexed addressing mode
            # Strategy:
            # 1. Calculate offset = index * element_size (if element_size > 1)
            # 2. Move offset to X register
            # 3. Use indexed addressing: LDA base,X

            offset_operand = index_operand

            # If element size > 1, multiply index by element_size
            if element_size > 1:
                offset_vreg = self.current_function.vreg_allocator.alloc(
                    element_type, "array_offset"
                )
                # Check if element_size is power of 2 - use shift instead of multiply
                if element_size & (element_size - 1) == 0:  # Is power of 2
                    # Calculate shift amount: log2(element_size)
                    shift_amount = 0
                    temp = element_size
                    while temp > 1:
                        shift_amount += 1
                        temp >>= 1
                    shift_immediate = Immediate(shift_amount)
                    # offset = index << shift_amount
                    self.emit(BinaryOp(
                        dest=offset_vreg,
                        left=index_operand,
                        right=shift_immediate,
                        op='<<',
                        type_info=element_type
                    ))
                else:
                    # Non-power-of-2: use multiplication
                    size_immediate = Immediate(element_size)
                    self.emit(BinaryOp(
                        dest=offset_vreg,
                        left=index_operand,
                        right=size_immediate,
                        op='*',
                        type_info=element_type
                    ))
                offset_operand = offset_vreg

            # Move offset to X register for indexed addressing
            x_reg = HardwareRegister('X')
            self.emit(Move(dest=x_reg, source=offset_operand, type_info=element_type))

            # Create indexed memory location with X register
            base_memloc = self.get_memory_location(array_symbol)
            indexed_memloc = MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=base_memloc.address,
                symbol=array_symbol,
                is_volatile=base_memloc.is_volatile,
                index_register='X'  # Mark as indexed with X
            )

            # Emit load using indexed addressing (e.g., LDA $20,X)
            self.emit(Load(dest=result, source=indexed_memloc, type_info=element_type))
            return result

    def lower_field_access(self, expr: HIRFieldAccess) -> VirtualRegister:
        """
        Lower struct field access.

        Computes: struct.field → Load from (base_address + field_offset)

        Args:
            expr: HIR field access expression

        Returns:
            VirtualRegister holding field value
        """
        # Get field offset (computed during HIR construction)
        field_offset = expr.field_offset
        if field_offset is None:
            raise Exception(f"Field offset not computed for field: {expr.field_name}")

        # Get the struct symbol (should be HIRIdentifier for static structs)
        if not isinstance(expr.base, HIRIdentifier):
            raise Exception(f"Field access only supports static structs currently, got: {type(expr.base)}")

        struct_symbol = expr.base.symbol

        # Allocate result register
        result = self.current_function.vreg_allocator.alloc(expr.expr_type, f"field_{expr.field_name}")

        # Get base memory location
        base_memloc = self.get_memory_location(struct_symbol)

        # Create offset memory location using helper
        field_memloc = self._create_offset_memloc(base_memloc, field_offset, struct_symbol)

        # Emit load from the field location
        self.emit(Load(dest=result, source=field_memloc, type_info=expr.expr_type))
        return result

    def lower_dereference(self, expr: HIRDereference) -> VirtualRegister:
        """
        Lower pointer dereference (*ptr).

        Generates LoadIndirect instruction to read through pointer.

        Args:
            expr: HIR dereference expression

        Returns:
            VirtualRegister holding dereferenced value
        """
        from r65.compiler.hir.types import PointerTypeInfo

        # Lower the pointer expression
        ptr_operand = self.lower_expression(expr.pointer)

        # Get pointer type to determine if far or near
        pointer_type = expr.pointer.expr_type
        if not isinstance(pointer_type, PointerTypeInfo):
            raise Exception(f"Dereference of non-pointer type: {pointer_type}")

        # Allocate result register
        result = self.current_function.vreg_allocator.alloc(
            expr.expr_type,
            "deref_result"
        )

        # Emit LoadIndirect
        self.emit(LoadIndirect(
            dest=result,
            pointer=ptr_operand,
            is_far=pointer_type.is_far,
            type_info=expr.expr_type
        ))

        return result

    def lower_addressof(self, expr: HIRAddressOf) -> VirtualRegister:
        """
        Lower address-of operator (&variable).

        For static variables, loads the address as an immediate value.
        The address is determined by the memory allocator.

        Args:
            expr: HIR address-of expression

        Returns:
            VirtualRegister holding the address
        """
        from r65.compiler.hir import HIRIdentifier
        from r65.compiler.hir.types import PointerTypeInfo

        # Currently only support address-of static variables
        if not isinstance(expr.operand, HIRIdentifier):
            raise Exception(f"Address-of only supports static variables, got: {type(expr.operand)}")

        symbol = expr.operand.symbol

        # Get the memory location of the variable
        mem_loc = self.get_memory_location(symbol)

        # Allocate result register for the pointer
        result = self.current_function.vreg_allocator.alloc(
            expr.expr_type,
            f"addr_of_{symbol.name}"
        )

        # For now, we'll store the symbol reference in the virtual register
        # The actual address will be resolved during code generation
        # We need a way to represent "address of variable" in MIR
        # For simplicity, create an Immediate with a special marker
        # TODO: This is a simplification - need better address representation

        # Create a symbolic address immediate
        # The code generator will resolve this to the actual address
        from r65.compiler.mir.nodes import Immediate
        addr_immediate = Immediate(0)  # Placeholder - will be resolved in codegen
        # Store symbol info for codegen (hack for now)
        addr_immediate.symbol = symbol  # Add symbol attribute

        # Move the address into the result register
        self.emit(Move(dest=result, source=addr_immediate, type_info=expr.expr_type))

        return result

    def _analyze_match_for_jump_table(self, expr: HIRMatchExpression):
        """
        Analyze match expression to determine if jump table optimization is applicable.

        Returns:
            tuple: (use_jump_table, min_value, max_value, value_to_arm_index)
                   or (False, None, None, None) if not suitable
        """
        from r65.compiler.hir import HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern, HIRIdentifierPattern

        # Extract literal/enum values from patterns
        pattern_values = []
        has_catchall = False

        for i, arm in enumerate(expr.arms):
            if isinstance(arm.pattern, HIRLiteralPattern):
                if isinstance(arm.pattern.value, int):
                    pattern_values.append((arm.pattern.value, i))
                else:
                    # Non-integer literal (e.g., bool) - can't use jump table
                    return (False, None, None, None)
            elif isinstance(arm.pattern, HIREnumPattern):
                # Enum patterns map to integer values
                pattern_values.append((arm.pattern.variant_value, i))
            elif isinstance(arm.pattern, HIRWildcardPattern) or isinstance(arm.pattern, HIRIdentifierPattern):
                has_catchall = True
                # Keep track but don't add to pattern_values
            else:
                # Or-pattern or other complex pattern - skip jump table optimization
                return (False, None, None, None)

        if not pattern_values:
            return (False, None, None, None)

        # Check if patterns form a dense range
        values = [v for v, _ in pattern_values]
        min_val = min(values)
        max_val = max(values)
        range_size = max_val - min_val + 1
        num_patterns = len(pattern_values)

        # Heuristic: Use jump table if:
        # 1. Range is not too large (< 256 entries for 8-bit index)
        # 2. Density is good (>= 50% coverage)
        # 3. We have at least 3 patterns (otherwise linear is fine)
        MAX_JUMP_TABLE_SIZE = 256
        MIN_DENSITY = 0.5
        MIN_PATTERNS = 3

        if range_size > MAX_JUMP_TABLE_SIZE:
            return (False, None, None, None)

        density = num_patterns / range_size
        if density < MIN_DENSITY or num_patterns < MIN_PATTERNS:
            return (False, None, None, None)

        # Build value-to-arm-index mapping
        value_to_arm = {}
        for value, arm_index in pattern_values:
            value_to_arm[value] = arm_index

        return (True, min_val, max_val, value_to_arm)

    def lower_match_expression(self, expr: HIRMatchExpression) -> VirtualRegister:
        """
        Lower match expression to conditional branches or jump table.

        Analyzes patterns and chooses between:
        - Jump table for dense sequential integer patterns (O(1))
        - Conditional branch chain for sparse or complex patterns (O(n))

        Args:
            expr: HIR match expression

        Returns:
            VirtualRegister holding the match result
        """
        # Check if jump table optimization is applicable
        use_jump_table, min_val, max_val, value_to_arm = self._analyze_match_for_jump_table(expr)

        if use_jump_table:
            return self._lower_match_with_jump_table(expr, min_val, max_val, value_to_arm)
        else:
            return self._lower_match_with_branches(expr)

    def _lower_match_with_branches(self, expr: HIRMatchExpression) -> VirtualRegister:
        """
        Lower match expression to conditional branches (fallback/default strategy).

        Lowers to a chain of if-then-else blocks:
        - Compare scrutinee against each pattern
        - Branch to arm body if match
        - Fall through to next pattern if no match
        - Collect results into a result vreg

        Args:
            expr: HIR match expression

        Returns:
            VirtualRegister holding the match result
        """
        from r65.compiler.hir import (HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern,
                                       HIRIdentifierPattern, HIROrPattern)

        # Lower scrutinee once
        scrutinee_vreg = self.lower_expression(expr.scrutinee)
        scrutinee_type = expr.scrutinee.expr_type

        # Allocate result register
        result_vreg = self.current_function.vreg_allocator.alloc(
            expr.expr_type,
            "match_result"
        )

        # Create merge block (where all arms converge)
        merge_block = self.cfg_builder.new_block()

        # Lower each arm to a chain of conditional branches
        for i, arm in enumerate(expr.arms):
            is_last_arm = (i == len(expr.arms) - 1)

            # Create block for this arm's body
            arm_block = self.cfg_builder.new_block()

            # Create block for next pattern check (or merge if last arm)
            next_block = merge_block if is_last_arm else self.cfg_builder.new_block()

            # Emit pattern matching logic
            self._lower_pattern_match(arm.pattern, scrutinee_vreg, scrutinee_type, arm_block, next_block)

            # Emit arm body in arm_block
            self.current_block = arm_block

            # Handle identifier pattern binding
            if isinstance(arm.pattern, HIRIdentifierPattern):
                # Bind scrutinee to the pattern variable
                binding_vreg = self.current_function.vreg_allocator.alloc(
                    arm.pattern.symbol.var_type,
                    arm.pattern.name
                )
                self.symbol_to_vreg[id(arm.pattern.symbol)] = binding_vreg
                self.emit(Move(dest=binding_vreg, source=scrutinee_vreg, type_info=arm.pattern.symbol.var_type))

            # Lower arm body
            arm_result = self.lower_expression(arm.body)

            # Move result to result_vreg
            self.emit(Move(dest=result_vreg, source=arm_result, type_info=expr.expr_type))

            # Jump to merge block
            self.emit(Jump(target=merge_block.block_id))
            self.cfg_builder.add_edge(arm_block, merge_block)

            # Continue with next pattern (if not last)
            if not is_last_arm:
                self.current_block = next_block

        # Set current block to merge
        self.current_block = merge_block

        return result_vreg

    def _lower_match_with_jump_table(self, expr: HIRMatchExpression, min_val: int, max_val: int, value_to_arm: dict) -> VirtualRegister:
        """
        Lower match expression using jump table optimization.

        Generates a jump table for O(1) pattern matching on dense integer ranges.

        Args:
            expr: HIR match expression
            min_val: Minimum pattern value
            max_val: Maximum pattern value
            value_to_arm: Mapping from pattern value to arm index

        Returns:
            VirtualRegister holding the match result
        """
        from r65.compiler.hir import HIRWildcardPattern, HIRIdentifierPattern

        # Lower scrutinee once
        scrutinee_vreg = self.lower_expression(expr.scrutinee)
        scrutinee_type = expr.scrutinee.expr_type

        # Allocate result register
        result_vreg = self.current_function.vreg_allocator.alloc(
            expr.expr_type,
            "match_result"
        )

        # Create merge block
        merge_block = self.cfg_builder.new_block()

        # Create blocks for each arm
        arm_blocks = []
        for _ in expr.arms:
            arm_blocks.append(self.cfg_builder.new_block())

        # Find default arm (wildcard or identifier pattern)
        default_arm_index = None
        for i, arm in enumerate(expr.arms):
            if isinstance(arm.pattern, (HIRWildcardPattern, HIRIdentifierPattern)):
                default_arm_index = i
                break

        # Build jump table: array of block IDs indexed by (value - min_val)
        range_size = max_val - min_val + 1
        jump_table = []
        for offset in range(range_size):
            value = min_val + offset
            if value in value_to_arm:
                arm_index = value_to_arm[value]
                jump_table.append(arm_blocks[arm_index].block_id)
            elif default_arm_index is not None:
                # Use default arm for missing values
                jump_table.append(arm_blocks[default_arm_index].block_id)
            else:
                # No default - this shouldn't happen if exhaustiveness checking works
                # For now, jump to merge (unreachable in correct code)
                jump_table.append(merge_block.block_id)

        # Determine default target (for out-of-bounds)
        default_target = arm_blocks[default_arm_index].block_id if default_arm_index is not None else merge_block.block_id

        # Emit jump table instruction
        self.emit(JumpTable(
            scrutinee=scrutinee_vreg,
            base_value=min_val,
            targets=jump_table,
            default_target=default_target,
            type_info=scrutinee_type
        ))

        # Add CFG edges from current block to all possible targets
        for block_id in set(jump_table + [default_target]):
            if block_id in self.current_function.blocks:
                self.cfg_builder.add_edge(self.current_block, self.current_function.blocks[block_id])

        # Lower each arm body
        for i, arm in enumerate(expr.arms):
            arm_block = arm_blocks[i]
            self.current_block = arm_block

            # Handle identifier pattern binding
            if isinstance(arm.pattern, HIRIdentifierPattern):
                binding_vreg = self.current_function.vreg_allocator.alloc(
                    arm.pattern.symbol.var_type,
                    arm.pattern.name
                )
                self.symbol_to_vreg[id(arm.pattern.symbol)] = binding_vreg
                self.emit(Move(dest=binding_vreg, source=scrutinee_vreg, type_info=arm.pattern.symbol.var_type))

            # Lower arm body
            arm_result = self.lower_expression(arm.body)

            # Move result to result_vreg
            self.emit(Move(dest=result_vreg, source=arm_result, type_info=expr.expr_type))

            # Jump to merge
            self.emit(Jump(target=merge_block.block_id))
            self.cfg_builder.add_edge(arm_block, merge_block)

        # Set current block to merge
        self.current_block = merge_block

        return result_vreg

    def _lower_pattern_match(self, pattern, scrutinee_vreg, scrutinee_type, match_block, no_match_block):
        """
        Emit code to test if scrutinee matches pattern.
        Branch to match_block if matches, no_match_block otherwise.
        """
        from r65.compiler.hir import (HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern,
                                       HIRIdentifierPattern, HIROrPattern)

        if isinstance(pattern, HIRLiteralPattern):
            # Compare scrutinee with literal value
            literal_vreg = self.current_function.vreg_allocator.alloc(
                scrutinee_type,
                f"literal_{pattern.value}"
            )
            self.emit(Move(dest=literal_vreg, source=Immediate(pattern.value), type_info=scrutinee_type))

            # Emit comparison and conditional branch
            self.emit(Compare(left=scrutinee_vreg, right=literal_vreg, comparison="==", type_info=scrutinee_type))
            self.emit(CondBranch(
                condition=None,
                true_target=match_block.block_id,
                false_target=no_match_block.block_id,
                comparison="=="
            ))
            # Add CFG edges
            self.cfg_builder.add_edge(self.current_block, match_block)
            self.cfg_builder.add_edge(self.current_block, no_match_block)

        elif isinstance(pattern, HIREnumPattern):
            # Compare scrutinee with enum variant value
            variant_vreg = self.current_function.vreg_allocator.alloc(
                scrutinee_type,
                f"{pattern.enum_name}_{pattern.variant_name}"
            )
            self.emit(Move(dest=variant_vreg, source=Immediate(pattern.variant_value), type_info=scrutinee_type))

            # Emit comparison and conditional branch
            self.emit(Compare(left=scrutinee_vreg, right=variant_vreg, comparison="==", type_info=scrutinee_type))
            self.emit(CondBranch(
                condition=None,
                true_target=match_block.block_id,
                false_target=no_match_block.block_id,
                comparison="=="
            ))
            # Add CFG edges
            self.cfg_builder.add_edge(self.current_block, match_block)
            self.cfg_builder.add_edge(self.current_block, no_match_block)

        elif isinstance(pattern, HIRWildcardPattern):
            # Wildcard always matches - unconditional jump
            self.emit(Jump(target=match_block.block_id))
            self.cfg_builder.add_edge(self.current_block, match_block)

        elif isinstance(pattern, HIRIdentifierPattern):
            # Identifier always matches - unconditional jump
            # Binding happens in the arm block
            self.emit(Jump(target=match_block.block_id))
            self.cfg_builder.add_edge(self.current_block, match_block)

        elif isinstance(pattern, HIROrPattern):
            # Or pattern: try each sub-pattern, jump to match_block if any matches
            for i, subpat in enumerate(pattern.patterns):
                is_last = (i == len(pattern.patterns) - 1)
                next_subpat_block = no_match_block if is_last else self.cfg_builder.new_block()

                self._lower_pattern_match(subpat, scrutinee_vreg, scrutinee_type, match_block, next_subpat_block)

                if not is_last:
                    self.current_block = next_subpat_block

        else:
            raise Exception(f"Unknown pattern type in MIR lowering: {type(pattern).__name__}")

    def lower_assignment(self, expr: HIRAssignment) -> Union[VirtualRegister, HardwareRegister]:
        """
        Lower assignment.

        Args:
            expr: HIR assignment

        Returns:
            VirtualRegister or HardwareRegister with assigned value
        """
        # OPTIMIZATION: Detect pattern `target = target op value` for hardware registers
        # Generate BinaryOp(dest=target, left=target, op, right=value) directly
        # instead of temp = target op value; target = temp
        if isinstance(expr.value, HIRBinaryOp) and isinstance(expr.target, HIRRegister):
            binary_op = expr.value
            # Check if it's target = target op value
            if (isinstance(binary_op.left, HIRRegister) and
                binary_op.left.name == expr.target.name):
                # Direct hardware register op: X = X + 1 becomes BinaryOp(dest=X, left=X, right=1)
                hw_reg = HardwareRegister(expr.target.name)
                right = self.lower_expression(binary_op.right)

                self.emit(BinaryOp(
                    dest=hw_reg,
                    left=hw_reg,
                    op=binary_op.op,
                    right=right,
                    type_info=expr.expr_type
                ))
                return hw_reg

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

        elif isinstance(expr.target, HIRFieldAccess):
            # Field access assignment: struct.field = value
            field_access = expr.target
            field_offset = field_access.field_offset
            if field_offset is None:
                raise Exception(f"Field offset not computed for field: {field_access.field_name}")

            # Get the struct symbol
            if not isinstance(field_access.base, HIRIdentifier):
                raise Exception(f"Field access only supports static structs currently, got: {type(field_access.base)}")

            struct_symbol = field_access.base.symbol

            # Get base memory location
            base_memloc = self.get_memory_location(struct_symbol)

            # Create offset memory location
            field_memloc = self._create_offset_memloc(base_memloc, field_offset, struct_symbol)

            # Emit store to the field location
            self.emit(Store(source=value, dest=field_memloc, type_info=expr.expr_type))
            return value

        elif isinstance(expr.target, HIRArrayIndex):
            # Array index assignment: array[index] = value
            array_index = expr.target
            element_type = expr.expr_type
            element_size = self._get_type_size(element_type)

            # Lower index expression
            index_operand = self.lower_expression(array_index.index)

            # Get the array symbol
            if not isinstance(array_index.array, HIRIdentifier):
                raise Exception(f"Array indexing only supports static arrays currently, got: {type(array_index.array)}")

            array_symbol = array_index.array.symbol

            # Calculate offset and create memory location
            if isinstance(index_operand, Immediate):
                # Constant index - compute offset at compile time
                offset = index_operand.value * element_size
                base_memloc = self.get_memory_location(array_symbol)

                # Create offset memory location
                elem_memloc = self._create_offset_memloc(base_memloc, offset, array_symbol)

                # Emit store to the element location
                self.emit(Store(source=value, dest=elem_memloc, type_info=element_type))
                return value
            else:
                # Variable index - use indexed addressing mode
                offset_operand = index_operand

                # If element size > 1, multiply index by element_size
                if element_size > 1:
                    offset_vreg = self.current_function.vreg_allocator.alloc(
                        element_type, "array_offset"
                    )
                    # Check if element_size is power of 2 - use shift instead of multiply
                    if element_size & (element_size - 1) == 0:  # Is power of 2
                        # Calculate shift amount: log2(element_size)
                        shift_amount = 0
                        temp = element_size
                        while temp > 1:
                            shift_amount += 1
                            temp >>= 1
                        shift_immediate = Immediate(shift_amount)
                        # offset = index << shift_amount
                        self.emit(BinaryOp(
                            dest=offset_vreg,
                            left=index_operand,
                            right=shift_immediate,
                            op='<<',
                            type_info=element_type
                        ))
                    else:
                        # Non-power-of-2: use multiplication
                        size_immediate = Immediate(element_size)
                        self.emit(BinaryOp(
                            dest=offset_vreg,
                            left=index_operand,
                            right=size_immediate,
                            op='*',
                            type_info=element_type
                        ))
                    offset_operand = offset_vreg

                # Move offset to X register for indexed addressing
                x_reg = HardwareRegister('X')
                self.emit(Move(dest=x_reg, source=offset_operand, type_info=element_type))

                # Create indexed memory location with X register
                base_memloc = self.get_memory_location(array_symbol)
                indexed_memloc = MemoryLocation(
                    storage_type=base_memloc.storage_type,
                    address=base_memloc.address,
                    symbol=array_symbol,
                    is_volatile=base_memloc.is_volatile,
                    index_register='X'  # Mark as indexed with X
                )

                # Emit store using indexed addressing (e.g., STA $20,X)
                self.emit(Store(source=value, dest=indexed_memloc, type_info=element_type))
                return value

        elif isinstance(expr.target, HIRDereference):
            # Pointer dereference assignment: *ptr = value
            from r65.compiler.hir.types import PointerTypeInfo

            deref = expr.target
            pointer_type = deref.pointer.expr_type

            if not isinstance(pointer_type, PointerTypeInfo):
                raise Exception(f"Dereference of non-pointer type: {pointer_type}")

            # Lower the pointer expression to get the pointer value
            ptr_operand = self.lower_expression(deref.pointer)

            # Emit StoreIndirect
            self.emit(StoreIndirect(
                source=value,
                pointer=ptr_operand,
                is_far=pointer_type.is_far,
                type_info=expr.expr_type
            ))
            return value

        else:
            # Unsupported target
            raise Exception(f"Unsupported assignment target: {type(expr.target)}")

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
                func_decl = self.function_decls.get(func_symbol.name)
                if not func_decl:
                    raise Exception(f"Function call to {func_symbol.name}: function not found in HIR")
                func_ptr_vreg = None
        else:
            # Indirect call (function pointer)
            # Lower the function expression to get the virtual register holding the pointer
            func_ptr_vreg = self.lower_expression(call_expr.func)
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
            result_vreg = self.current_function.vreg_allocator.alloc(
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

    def lower_method_call(self, call_expr: HIRMethodCall) -> VirtualRegister:
        """
        Lower method call (e.g., value.rotate_left(3)).

        Currently only supports rotate_left and rotate_right methods.

        Args:
            call_expr: HIRMethodCall expression

        Returns:
            VirtualRegister holding the result
        """
        from r65.compiler.hir import HIRIntegerLiteral
        from r65.compiler.mir.nodes import Rotate

        # Lower the receiver (value being rotated)
        receiver_value = self.lower_expression(call_expr.receiver)

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
            raise Exception(f"Unknown rotate method: {call_expr.method_name}")

        # Create result register
        result_vreg = self.current_function.vreg_allocator.alloc(call_expr.expr_type, "rotate_result")

        # Emit Rotate instruction
        self.emit(Rotate(
            dest=result_vreg,
            source=receiver_value,
            direction=direction,
            count=count,
            type_info=call_expr.expr_type
        ))

        return result_vreg

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
        self._lower_condition(stmt.condition, then_block.block_id, else_block.block_id)

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

    def _detect_bit_test_pattern(self, condition: HIRExpression):
        """
        Detect if condition is a bit-testing pattern suitable for BIT instruction.

        Detects patterns:
        - (value & 0x80) != 0  =>  bit 7 test
        - (value & 0x40) != 0  =>  bit 6 test
        - (value & 0x80) == 0  =>  bit 7 test (inverted)
        - (value & 0x40) == 0  =>  bit 6 test (inverted)

        Returns:
            tuple: (value_expr, bit_number, inverted) or None
        """
        from r65.compiler.hir import HIRBinaryOp, HIRIntegerLiteral

        # Check if it's a comparison with 0
        if not isinstance(condition, HIRBinaryOp):
            return None

        if condition.op not in ('==', '!='):
            return None

        # Check pattern: (value & mask) op 0
        left = condition.left
        right = condition.right

        # Swap if needed: 0 op (value & mask)
        if isinstance(left, HIRIntegerLiteral) and left.value == 0:
            left, right = right, left

        # Now check: (value & mask) op 0
        if not isinstance(right, HIRIntegerLiteral) or right.value != 0:
            return None

        # Check if left is (value & mask)
        if not isinstance(left, HIRBinaryOp) or left.op != '&':
            return None

        # Check the mask value
        mask_expr = left.right
        if not isinstance(mask_expr, HIRIntegerLiteral):
            return None

        mask = mask_expr.value
        bit_number = None

        if mask == 0x80:
            bit_number = 7
        elif mask == 0x40:
            bit_number = 6
        else:
            return None  # Only support bit 6 and 7

        # Determine if test is inverted (== 0 means inverted)
        inverted = (condition.op == '==')

        return (left.left, bit_number, inverted)

    def _lower_condition(self, condition: HIRExpression, true_target: int, false_target: int):
        """
        Lower condition expression with short-circuit evaluation.

        Generates control flow that branches to true_target or false_target based
        on condition result, with short-circuit for && and || operators.

        Args:
            condition: Condition expression to evaluate
            true_target: Block ID to jump to if condition is true
            false_target: Block ID to jump to if condition is false
        """
        comparison_ops = {'==', '!=', '<', '<=', '>', '>='}

        # OPTIMIZATION 0: BIT instruction for bit testing
        bit_test = self._detect_bit_test_pattern(condition)
        if bit_test:
            value_expr, bit_number, inverted = bit_test

            # OPTIMIZATION: For direct variable access (especially hardware registers),
            # use MemoryLocation directly instead of loading the value
            # This allows: BIT $4212 instead of: LDA $4212; STA temp; BIT temp
            from r65.compiler.hir import HIRIdentifier

            if isinstance(value_expr, HIRIdentifier) and self.has_explicit_location(value_expr.symbol):
                # Direct variable access - use MemoryLocation to avoid load/store
                # This optimization only works for static variables (especially hardware registers)
                symbol = value_expr.symbol
                value = self.get_memory_location(symbol)
            else:
                # Complex expression - need to evaluate it first
                value = self.lower_expression(value_expr)

            # Only use BIT if value is not in a hardware register
            # BIT requires a memory operand
            if not isinstance(value, HardwareRegister):
                # Value is in memory - can use BIT optimization
                # Emit BitTest instruction
                self.emit(BitTest(
                    value=value,
                    test_bit=bit_number,
                    type_info=value_expr.expr_type
                ))

                # Branch based on bit value
                # BMI/BPL for bit 7, BVS/BVC for bit 6
                if inverted:
                    # Test is (value & mask) == 0, so bit is clear
                    # Swap targets: if bit clear goto true, else goto false
                    actual_true = true_target
                    actual_false = false_target
                else:
                    # Test is (value & mask) != 0, so bit is set
                    # Normal: if bit set goto true, else goto false
                    actual_true = true_target
                    actual_false = false_target

                # Emit conditional branch
                # We'll use a special comparison string to indicate BIT-based branch
                if bit_number == 7:
                    comparison = 'bit7_set' if not inverted else 'bit7_clear'
                else:  # bit_number == 6
                    comparison = 'bit6_set' if not inverted else 'bit6_clear'

                self.emit(CondBranch(
                    condition=None,  # Uses flags from BitTest
                    true_target=actual_true,
                    false_target=actual_false,
                    comparison=comparison
                ))

                # Add CFG edges
                self.cfg_builder.add_edge(self.current_block, self.current_function.blocks[true_target])
                self.cfg_builder.add_edge(self.current_block, self.current_function.blocks[false_target])
                return
            # If value is in hardware register, fall through to normal comparison handling

        # OPTIMIZATION 1: Short-circuit AND (&&)
        if isinstance(condition, HIRBinaryOp) and condition.op == '&&':
            # For: if (left && right)
            # - Evaluate left
            # - If left is false, jump to false_target (short-circuit)
            # - Otherwise, evaluate right and use its result
            right_eval_block = self.cfg_builder.new_block()

            # Evaluate left condition
            self._lower_condition(condition.left, right_eval_block.block_id, false_target)

            # If left was true, evaluate right
            self.current_block = right_eval_block
            self._lower_condition(condition.right, true_target, false_target)
            return

        # OPTIMIZATION 2: Short-circuit OR (||)
        elif isinstance(condition, HIRBinaryOp) and condition.op == '||':
            # For: if (left || right)
            # - Evaluate left
            # - If left is true, jump to true_target (short-circuit)
            # - Otherwise, evaluate right and use its result
            right_eval_block = self.cfg_builder.new_block()

            # Evaluate left condition
            self._lower_condition(condition.left, true_target, right_eval_block.block_id)

            # If left was false, evaluate right
            self.current_block = right_eval_block
            self._lower_condition(condition.right, true_target, false_target)
            return

        # OPTIMIZATION 3: Direct comparison - emit Compare + CondBranch
        elif isinstance(condition, HIRBinaryOp) and condition.op in comparison_ops:
            # Direct comparison - emit Compare instruction
            left = self.lower_expression(condition.left)
            right = self.lower_expression(condition.right)

            # Emit Compare instruction
            self.emit(Compare(
                left=left,
                right=right,
                comparison=condition.op,
                type_info=condition.left.expr_type
            ))

            # Emit conditional branch based on comparison flags
            self.emit(CondBranch(
                condition=None,  # No condition vreg - uses flags from Compare
                true_target=true_target,
                false_target=false_target,
                comparison=condition.op
            ))
            # Add CFG edges
            self.cfg_builder.add_edge(self.current_block, self.current_function.blocks[true_target])
            self.cfg_builder.add_edge(self.current_block, self.current_function.blocks[false_target])

        else:
            # General condition - evaluate to boolean and branch on != 0
            cond_value = self.lower_expression(condition)

            # Emit conditional branch: if condition != 0 goto true, else goto false
            self.emit(CondBranch(
                condition=cond_value,
                true_target=true_target,
                false_target=false_target,
                comparison='!='
            ))
            # Add CFG edges
            self.cfg_builder.add_edge(self.current_block, self.current_function.blocks[true_target])
            self.cfg_builder.add_edge(self.current_block, self.current_function.blocks[false_target])

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
        caller_mode = self.current_mode
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
            arg_value = self.lower_expression(arg_expr)

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
            mechanism = ArgumentMechanism.REGISTER
            location = HardwareRegister(param.binding.register_name)

            # Move argument to hardware register if needed
            if not (isinstance(arg_value, HardwareRegister) and arg_value.name == location.name):
                self.emit(Move(dest=location, source=arg_value, type_info=param.param_type))

            return mechanism, location

        elif isinstance(param.binding, VariableBinding):
            # Variable-bound parameter
            mechanism = ArgumentMechanism.VARIABLE
            location = self.get_memory_location(param.binding.variable_symbol)

            # Store argument to variable location
            self.emit(Store(source=arg_value, dest=location, type_info=param.param_type))

            return mechanism, location

        else:
            # Stack parameter
            return ArgumentMechanism.STACK, None

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

        Args:
            type_info: TypeInfo

        Returns:
            Size in bytes (1 for u8/i8/bool, 2 for u16/i16/near<T>/fn(), 3 for far<T>/far fn())
        """
        from r65.compiler.hir.types import FunctionTypeInfo

        type_str = str(type_info)

        # Basic types
        if type_str in ('u8', 'i8', 'bool'):
            return 1
        elif type_str in ('u16', 'i16'):
            return 2

        # Function pointers
        if isinstance(type_info, FunctionTypeInfo):
            return 3 if type_info.is_far else 2

        # Pointer types
        if type_str.startswith('near'):
            return 2
        elif type_str.startswith('far'):
            return 3

        # Function type (check string representation as fallback)
        if type_str.startswith('far fn'):
            return 3
        elif type_str.startswith('fn'):
            return 2

        # Array types - get element size * length
        # For now, return 1 as default
        # TODO: Handle arrays and structs properly
        return 1

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
