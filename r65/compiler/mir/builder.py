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
    HIRFunctionAddress, HIRRegister, HIRStatusFlagAccess, HIRBinaryOp, HIRUnaryOp, HIRTypeCast, HIRAssignment,
    HIRFunctionCall, HIRMethodCall, HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf, HIRMultiAssignment,
    HIRArrayFillExpr, HIRArrayLiteralExpr, HIRStringLiteral, HIRStructLiteralExpr,
    HIRMatchExpression, HIRPattern, HIRLiteralPattern, HIREnumPattern, HIRWildcardPattern, HIRIdentifierPattern, HIROrPattern,
    HIRBlockExpression, HIRIfExpression, HIRLoopExpression,
    RegisterLetBinding, VariableLetBinding,
    RegisterBinding, VariableBinding,
    SymbolKind,
)
from r65.compiler.hir.attributes import StorageKind

from r65.compiler.mir.nodes import (
    MIRInstruction, MIRProgram, MIRFunction, BasicBlock,
    VirtualRegister, HardwareRegister, Immediate, FunctionPointer, LabelRef, MemoryLocation,
    Load, Store, LoadIndirect, StoreIndirect, Move, TypeConvert, BinaryOp, UnaryOp, Compare, BitTest,
    Jump, CondBranch, JumpTable, LookupTable, Return, ReturnFromInterrupt, Call, Argument, ArgumentMechanism,
    StatusFlagRead,
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
from r65.compiler.mir.lowerers.static_init import StaticInitLowerer
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState, XModeState
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
        self.static_init_lowerer = StaticInitLowerer(self)

        # ROM data section tracking for array literals
        self._rom_data_counter = 0
        self._rom_data_sections: List[ROMDataRef] = []

        # Current HIR program being lowered (for symbol table lookups)
        self._hir_program: Optional[HIRProgram] = None

        # Promoted aggregate locals: maps original symbol id -> synthetic static symbol
        # Cleared per-function in lower_function()
        self._promoted_locals: Dict[int, Any] = {}

        # Counter for unique promoted local names (avoids collisions)
        self._promoted_local_counter = 0

        # Synthetic static declarations from promoted aggregate locals
        # Appended to MIRProgram.statics at program build time
        self._promoted_statics: List[Any] = []

        # Current source location for debug info propagation
        self._current_source_loc = None

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

        # Clear type size cache to prevent stale id()-based entries
        TypeSizeCalculator.clear_cache()

        # Store reference to HIR program for symbol table lookups
        self._hir_program = hir_program

        # Build function name → HIRFunctionDecl mapping
        from r65.compiler.hir import HIRImplDecl
        for decl in hir_program.declarations:
            if isinstance(decl, HIRFunctionDecl):
                self.function_decls[decl.name] = decl
            # Also register methods from impl blocks
            elif isinstance(decl, HIRImplDecl):
                for method in decl.methods:
                    self.function_decls[method.name] = method

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
            init_func = self.static_init_lowerer.generate_init_function(statics_with_initializers)
            mir_functions.append(init_func)
            self.has_init_start = True

        # Lower each function
        for decl in hir_program.declarations:
            if isinstance(decl, HIRFunctionDecl) and decl.body:
                mir_func = self.lower_function(decl)
                mir_functions.append(mir_func)
            # Also lower methods from impl blocks
            elif isinstance(decl, HIRImplDecl):
                for method in decl.methods:
                    if method.body:
                        mir_func = self.lower_function(method)
                        mir_functions.append(mir_func)

        # Create MIR program (keep HIR declarations for statics, etc.)
        # Include any synthetic statics from promoted aggregate locals
        all_statics = list(hir_program.statics) + self._promoted_statics
        return MIRProgram(
            functions=mir_functions,
            statics=all_statics,
            constants=hir_program.constants,
            structs=hir_program.structs,
            enums=hir_program.enums,
            symbol_table=hir_program.symbol_table,
            stack_attr=hir_program.stack_attr,
            snesrom_config=hir_program.snesrom_config,
            rom_data_sections=self._rom_data_sections,
            trait_dispatch_info=hir_program.trait_dispatch_info
        )

    def lower_function(self, hir_func: HIRFunctionDecl) -> MIRFunction:
        """
        Lower a single function to MIR.

        Args:
            hir_func: HIR function declaration

        Returns:
            MIRFunction with CFG
        """
        # Reset source location to function's own location for entry code
        # This ensures prologue instructions get the function's source_loc
        self._current_source_loc = hir_func.source_loc
        self.ctx.current_source_loc = hir_func.source_loc

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
            inline_attr=hir_func.inline_attr,
            is_entry=hir_func.is_entry,
            is_far=hir_func.is_far,
            is_trait_method=hir_func.is_trait_method,
            entry_m_mode=hir_func.entry_m_mode,  # Inferred entry mode
            exit_m_mode=hir_func.exit_m_mode,    # Inferred exit mode
            source_loc=hir_func.source_loc,  # Propagate source location
            vreg_allocator=VirtualRegisterAllocator(),
            alias_tracker=RegisterAliasTracker()
        )

        # Set current function context
        self.current_function = mir_func
        self.cfg_builder = CFGBuilder(mir_func)
        self.symbol_to_vreg.clear()
        self.loop_stack.clear()
        self._promoted_locals.clear()

        # Initialize current mode from function's inferred entry mode
        # entry_m_mode is set by HIR builder based on A parameter type
        if hir_func.entry_m_mode:
            self.current_mode = ProcessorMode(hir_func.entry_m_mode, XModeState.X16)
        else:
            # Default: m8, x16
            self.current_mode = ProcessorMode.default()

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

        # Count how many times each register-aliased parameter is used.
        # If the parameter is used at all, save it to a vreg at entry because
        # intermediate code (local variable initializers, etc.) may clobber A
        # before the parameter is used.
        param_usage_counts = self._count_param_usages(hir_func)

        # CRITICAL: Process A register parameters FIRST, before X/Y parameters.
        # Saving X/Y to memory requires TXA/TYA which clobbers A.
        # If A holds a parameter that's used later, we must save it first.
        # Process in two passes: first A, then X/Y.

        def process_register_param(param, hw_reg):
            """Process a register-bound parameter."""
            param_uses = param_usage_counts.get(id(param.symbol), 0)
            if param_uses >= 1:
                # Parameter is used - MUST save to vreg at entry
                vreg = self.current_function.vreg_allocator.alloc(
                    param.param_type,
                    f"saved_{param.name}"
                )
                self.symbol_to_vreg[id(param.symbol)] = vreg
                # Emit move from hardware register to vreg
                self.emit(Move(dest=vreg, source=hw_reg, type_info=param.param_type))
            else:
                # Parameter not used - just track as hw register alias
                self.current_function.alias_tracker.add_alias(
                    param.symbol,
                    hw_reg,
                    param.symbol.scope_id,
                    binding_type=param.param_type
                )

        # First pass: process A register parameters (save before X/Y clobber A)
        for param in hir_func.parameters:
            if isinstance(param.binding, RegisterBinding):
                hw_reg = HardwareRegister(param.binding.register_name)
                if hw_reg.name == 'A':
                    process_register_param(param, hw_reg)

        # Second pass: process B register parameters (accessed via XBA, so A must be saved first)
        for param in hir_func.parameters:
            if isinstance(param.binding, RegisterBinding):
                hw_reg = HardwareRegister(param.binding.register_name)
                if hw_reg.name == 'B':
                    process_register_param(param, hw_reg)

        # Third pass: process X/Y register parameters (may clobber A via TXA/TYA)
        for param in hir_func.parameters:
            if isinstance(param.binding, RegisterBinding):
                hw_reg = HardwareRegister(param.binding.register_name)
                if hw_reg.name in ('X', 'Y'):
                    process_register_param(param, hw_reg)

        # Fourth pass: process non-register parameters
        for idx, param in enumerate(hir_func.parameters):
            if isinstance(param.binding, RegisterBinding):
                pass  # Already processed above
            elif isinstance(param.binding, VariableBinding):
                # Variable-bound parameter: treat it as an alias to the bound variable
                # When the parameter is referenced in the function, it should load from the bound variable
                # The bound variable already exists and has a memory allocation
                # No setup needed here - lowering expressions handles the load
                pass
            elif hir_func.is_trait_method and idx == 0 and param.name == 'self':
                # Trait method self parameter: passed in Y register, not on stack
                # Allocate a vreg and emit Move from Y to self_vreg
                param_vreg = self.current_function.vreg_allocator.alloc(
                    param.param_type,
                    f"param_self_y"
                )
                self.symbol_to_vreg[id(param.symbol)] = param_vreg
                mir_func.self_y_vreg = param_vreg
                # Emit move from Y to self_vreg (will be pre-allocated to Y in codegen)
                self.emit(Move(dest=param_vreg, source=HardwareRegister('Y'), type_info=param.param_type))
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
            from r65.compiler.hir.types import PointerTypeInfo

            return_addr_size = 3 if hir_func.is_far else 2
            current_offset = return_addr_size + 1  # First param starts after return address

            for idx, param, param_vreg in stack_params:
                mir_func.stack_param_offsets[idx] = current_offset
                mir_func.param_to_vreg[idx] = param_vreg

                # Check if this is a far pointer parameter
                if isinstance(param.param_type, PointerTypeInfo) and param.param_type.is_far:
                    mir_func.has_far_ptr_stack_params = True
                    mir_func.far_ptr_param_indices.add(idx)

                # Advance offset by parameter size
                param_size = self._get_type_size(param.param_type)
                current_offset += param_size

            # Note: Far pointer stack params require x16 mode for [dp],Y addressing
            # In the simplified mode system, X/Y are always 16-bit (x16 mode),
            # so no validation is needed - far pointer params always work

        # If this is an entry point function, call __init_start() to initialize static data
        # Note: The SEI/CLC/XCE/REP sequence for switching to native mode is now
        # emitted in the prologue (function_gen.py) BEFORE frame allocation
        if hir_func.is_entry:
            if self.has_init_start:
                # Emit call to __init_start()
                self.emit(Call(
                    function="__init_start",
                    args=[],
                    returns=[],
                    is_far=False,
                    mode_attr=None,  # __init_start has no mode requirements
                    bank_attr=None   # __init_start is always near, no DBR management
                ))

        # Generate interrupt handler mode setup if needed
        # Note: Register saves (PHP, PHA, PHX, PHY, PHD, PHB) are emitted by codegen
        # in emit_prologue BEFORE frame allocation. This is critical because stack
        # frame allocation uses stack-relative addressing that would corrupt saved
        # registers if the pushes came after frame allocation.
        if hir_func.interrupt_attr:
            # Set the handler's mode (interrupts can fire from any mode)
            # Force the handler's inferred mode using SEP/REP
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

                # X is always x16 in the new design
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
            # Note: Register restores (PLB, PLD, PLY, PLX, PLA, PLP) are emitted by
            # codegen in select_return_from_interrupt AFTER frame deallocation.
            # This ensures correct stack ordering since prologue pushes registers
            # before allocating frame.
            if hir_func.interrupt_attr:
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
            raise MIRLoweringError(f"Mode tracking failed for function '{mir_func.name}': mode conflicts detected", source_loc=self._current_source_loc)

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
        # Capture source location for debug info propagation
        if hasattr(stmt, 'source_loc') and stmt.source_loc is not None:
            self._current_source_loc = stmt.source_loc
            self.ctx.current_source_loc = stmt.source_loc

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
        elif isinstance(stmt, HIRBlock):
            # Nested block - flatten by recursively lowering its statements
            self.lower_block(stmt)
        elif isinstance(stmt, HIRAssignment):
            # Assignment statement (e.g., from for loop increment)
            self.assign_lowerer.lower_assignment(stmt)
        elif isinstance(stmt, HIRMultiAssignment):
            # Multi-assignment statement
            self.assign_lowerer.lower_multi_assignment(stmt)
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
            # Register alias: track in alias tracker with binding type
            hw_reg = HardwareRegister(stmt.binding.register_name)
            self.current_function.alias_tracker.add_alias(
                stmt.symbol,
                hw_reg,
                stmt.symbol.scope_id,
                binding_type=stmt.var_type  # Track the type for mode optimization
            )

            # If there's an initializer, load it into hardware register
            if stmt.initializer:
                init_value = self.lower_expression(stmt.initializer)
                if not (isinstance(init_value, HardwareRegister) and init_value.name == hw_reg.name):
                    # Check if this is a u16 @ A binding that should keep m16 mode
                    persist_mode = False
                    if (stmt.binding.register_name == "A" and
                        stmt.var_type and
                        hasattr(stmt.var_type, 'name') and
                        stmt.var_type.name in ('u16', 'i16')):
                        persist_mode = True

                    # Move to hardware register if not already there
                    self.emit(Move(
                        dest=hw_reg,
                        source=init_value,
                        type_info=stmt.var_type,
                        persist_16bit_mode=persist_mode
                    ))

        else:
            # Check if this is an aggregate type (struct or array) that needs
            # promotion to static storage — the 65816 lacks stack-indexed
            # addressing modes needed for variable-index array access on stack
            from r65.compiler.hir.types import ArrayTypeInfo, StructTypeInfo
            if isinstance(stmt.var_type, (ArrayTypeInfo, StructTypeInfo)):
                self._promote_aggregate_local(stmt)
                return

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
                    # Get register hint if this is a loop variable
                    register_hint = getattr(stmt.symbol, 'register_hint', None)

                    if isinstance(init_value, VirtualRegister):
                        # Reuse the virtual register from initializer
                        # Propagate register hint to the reused vreg
                        if register_hint and not init_value.register_hint:
                            init_value.register_hint = register_hint
                        self.symbol_to_vreg[id(stmt.symbol)] = init_value
                    elif isinstance(init_value, HardwareRegister):
                        # MUST copy from hardware register - it can be clobbered later!
                        # Cannot just alias the symbol to the hw register.
                        vreg = self.current_function.vreg_allocator.alloc(
                            stmt.var_type, stmt.name, register_hint=register_hint)
                        self.symbol_to_vreg[id(stmt.symbol)] = vreg
                        self.emit(Move(dest=vreg, source=init_value, type_info=stmt.var_type))
                    else:
                        # Allocate new virtual register for immediate
                        vreg = self.current_function.vreg_allocator.alloc(
                            stmt.var_type, stmt.name, register_hint=register_hint)
                        self.symbol_to_vreg[id(stmt.symbol)] = vreg
                        self.emit(Move(dest=vreg, source=init_value, type_info=stmt.var_type))
            else:
                # Uninitialized variable - just allocate storage
                if not self.has_explicit_location(stmt.symbol):
                    # Allocate virtual register for later use
                    # Get register hint if this is a loop variable
                    register_hint = getattr(stmt.symbol, 'register_hint', None)
                    vreg = self.current_function.vreg_allocator.alloc(
                        stmt.var_type, stmt.name, register_hint=register_hint)
                    self.symbol_to_vreg[id(stmt.symbol)] = vreg

    def _promote_aggregate_local(self, stmt: HIRLetStmt):
        """
        Promote a local aggregate (struct/array) variable to static storage.

        The 65816 lacks stack-indexed addressing modes needed for variable-index
        array access on the stack. This silently promotes the local to an
        auto-allocated lowram static variable, emitting inline initialization
        code (rather than __init_start, since locals must re-initialize each call).

        Args:
            stmt: HIR let statement with aggregate type
        """
        from r65.compiler.hir.types import ArrayTypeInfo, StructTypeInfo
        from r65.compiler.hir.attributes import StorageAttribute, StorageKind
        from r65.compiler.hir.symbol_table import Symbol, SymbolKind

        # Generate a unique static name
        func_name = self.current_function.name
        var_name = stmt.name
        self._promoted_local_counter += 1
        unique_name = f"__local_{func_name}_{var_name}_{self._promoted_local_counter}"

        # Create a synthetic HIRStaticDecl
        synthetic_decl = HIRStaticDecl(
            name=unique_name,
            is_mutable=True,
            var_type=stmt.var_type,
            initializer=None,  # Init handled inline below
            storage_attr=StorageAttribute(
                name='lowram',
                storage_kind=StorageKind.LOWRAM,
                address=None  # Auto-allocated
            ),
            bank_attr=None,
            symbol=None  # Will be set below
        )

        # Create a synthetic Symbol
        new_symbol = Symbol(
            name=unique_name,
            kind=SymbolKind.STATIC_VAR,
            definition=synthetic_decl,
            scope_id=0,  # Global scope for statics
            var_type=stmt.var_type,
            is_mutable=True
        )
        synthetic_decl.symbol = new_symbol

        # Map original symbol to the new static symbol
        self._promoted_locals[id(stmt.symbol)] = new_symbol

        # Append the synthetic static to our tracking list
        self._promoted_statics.append(synthetic_decl)

        # Mark function as having promoted locals (for recursion checker)
        self.current_function.has_promoted_locals = True

        # Get memory location for the new static
        mem_loc = self.get_memory_location(new_symbol)

        # Handle initialization inline (runs each time the function is called)
        if stmt.initializer is not None:
            self._emit_aggregate_init(stmt, mem_loc, unique_name)

    def _emit_aggregate_init(self, stmt: HIRLetStmt, mem_loc: MemoryLocation, label_prefix: str):
        """
        Emit inline initialization code for a promoted aggregate local.

        Reuses patterns from StaticInitLowerer but emits directly into
        the current function's block (not __init_start).
        """
        initializer = stmt.initializer

        if isinstance(initializer, HIRArrayFillExpr):
            # Array fill: [0; 16] → MemoryFill
            from r65.compiler.hir.types import ArrayTypeInfo
            array_type = stmt.var_type
            if isinstance(array_type, ArrayTypeInfo):
                element_size = self._get_type_size(array_type.element_type)
            else:
                element_size = 1
            fill_value = self.static_init_lowerer._extract_constant_value(initializer.fill_value)
            if fill_value is None:
                fill_value = 0
            self.emit(MemoryFill(
                dest=mem_loc,
                fill_value=fill_value,
                count=initializer.count,
                element_size=element_size
            ))

        elif isinstance(initializer, HIRArrayLiteralExpr):
            # Array literal: [1, 2, 3] → ROMDataRef + BlockCopy
            self._emit_inline_array_literal(stmt, mem_loc, initializer, label_prefix)

        elif isinstance(initializer, HIRStringLiteral):
            # String literal → ROMDataRef + BlockCopy
            self._emit_inline_string_literal(stmt, mem_loc, initializer, label_prefix)

        elif isinstance(initializer, HIRStructLiteralExpr):
            # Struct literal → ROMDataRef + BlockCopy
            self._emit_inline_struct_literal(stmt, mem_loc, initializer, label_prefix)

        else:
            # Scalar/other — shouldn't normally happen for aggregates
            init_value = self.lower_expression(initializer)
            self.emit(Store(
                source=init_value,
                dest=mem_loc,
                type_info=stmt.var_type
            ))

    def _emit_inline_array_literal(self, stmt, mem_loc, literal_expr, label_prefix):
        """Emit ROMDataRef + BlockCopy for an inline array literal initializer."""
        from r65.compiler.hir.types import ArrayTypeInfo

        array_type = stmt.var_type
        if isinstance(array_type, ArrayTypeInfo):
            element_size = self._get_type_size(array_type.element_type)
        else:
            element_size = 1

        data_bytes = []
        for elem in literal_expr.elements:
            if isinstance(elem, HIRStructLiteralExpr):
                struct_bytes = self.static_init_lowerer._extract_struct_literal_bytes(elem)
                data_bytes.extend(struct_bytes)
            else:
                value = self.static_init_lowerer._extract_constant_value(elem)
                if value is None:
                    value = 0
                if element_size == 1:
                    data_bytes.append(value & 0xFF)
                elif element_size == 2:
                    data_bytes.append(value & 0xFF)
                    data_bytes.append((value >> 8) & 0xFF)
                else:
                    for i in range(element_size):
                        data_bytes.append((value >> (i * 8)) & 0xFF)

        label = f"__{label_prefix}_data"
        rom_data = ROMDataRef(label=label, data=data_bytes, element_size=element_size)
        self._rom_data_sections.append(rom_data)
        self.emit(BlockCopy(dest=mem_loc, rom_data=rom_data, count=len(data_bytes)))

    def _emit_inline_string_literal(self, stmt, mem_loc, string_literal, label_prefix):
        """Emit ROMDataRef + BlockCopy for an inline string literal initializer."""
        from r65.compiler.hir.types import ArrayTypeInfo

        array_type = stmt.var_type
        if isinstance(array_type, ArrayTypeInfo):
            array_size = array_type.size
        else:
            array_size = len(string_literal.processed_bytes)

        data_bytes = list(string_literal.processed_bytes)
        while len(data_bytes) < array_size:
            data_bytes.append(0)

        label = f"__{label_prefix}_data"
        rom_data = ROMDataRef(label=label, data=data_bytes, element_size=1)
        self._rom_data_sections.append(rom_data)
        self.emit(BlockCopy(dest=mem_loc, rom_data=rom_data, count=len(data_bytes)))

    def _emit_inline_struct_literal(self, stmt, mem_loc, struct_expr, label_prefix):
        """Emit ROMDataRef + BlockCopy for an inline struct literal initializer."""
        data_bytes = self.static_init_lowerer._extract_struct_literal_bytes(struct_expr)

        label = f"__{label_prefix}_data"
        rom_data = ROMDataRef(label=label, data=data_bytes, element_size=1)
        self._rom_data_sections.append(rom_data)
        self.emit(BlockCopy(dest=mem_loc, rom_data=rom_data, count=len(data_bytes)))

    def lower_tuple_let_statement(self, stmt: HIRTupleLetStmt):
        """
        Lower tuple destructuring let binding.

        Example: let (a, b) = func_returning_tuple();

        Return values are in registers determined by callee's return type.
        For (u8, u8) tuples in m8 mode, uses A, B, X, Y order.
        Otherwise uses A, X, Y order.
        """
        # Evaluate the initializer (typically a function call)
        # This returns the first value; other values are in registers
        init_value = self.lower_expression(stmt.initializer)

        # Determine return register order from callee's return type
        return_registers = self._get_callee_return_registers(stmt.initializer)

        # Capture each binding from the corresponding return register
        for i, (name, symbol, var_type) in enumerate(zip(stmt.names, stmt.symbols, stmt.var_types)):
            if i >= len(return_registers):
                # Can't capture more than available return registers
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
            # Return from interrupt (RTI)
            # Note: Interrupt handlers shouldn't return values
            # Register restores (PLB, PLD, PLY, PLX, PLA, PLP) are emitted by
            # codegen in select_return_from_interrupt AFTER frame deallocation.
            # This ensures correct stack ordering since prologue pushes registers
            # before allocating frame.
            if return_values:
                raise MIRLoweringError(f"Interrupt handler '{self.current_function.name}' cannot return values", source_loc=self._current_source_loc)
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
        # Capture source location for debug info propagation
        if hasattr(expr, 'source_loc') and expr.source_loc is not None:
            self._current_source_loc = expr.source_loc
            self.ctx.current_source_loc = expr.source_loc

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

            # Check if this is a constant - return immediate value
            from r65.compiler.hir.symbol_table import SymbolKind
            from r65.compiler.hir.types import FunctionTypeInfo
            if symbol.kind == SymbolKind.CONST and symbol.const_value is not None:
                if isinstance(symbol.const_value, dict):
                    raise MIRLoweringError(
                        f"const struct '{symbol.name}' cannot be used as a value; "
                        f"access individual fields (e.g., {symbol.name}.field_name)"
                    )
                if isinstance(symbol.const_value, list):
                    raise MIRLoweringError(
                        f"const array '{symbol.name}' cannot be used as a value; "
                        f"access individual elements (e.g., {symbol.name}[index])"
                    )
                return Immediate(symbol.const_value)

            # Check if this is a function identifier (function pointer)
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

        elif isinstance(expr, HIRMultiAssignment):
            return self.assign_lowerer.lower_multi_assignment(expr)

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

        elif isinstance(expr, HIRStatusFlagAccess):
            return self._lower_status_flag_read(expr)

        elif isinstance(expr, HIRDereference):
            return self.expr_lowerer.lower_dereference(expr)

        elif isinstance(expr, HIRAddressOf):
            return self.expr_lowerer.lower_addressof(expr)

        elif isinstance(expr, HIRMatchExpression):
            return self.match_lowerer.lower_match_expression(expr)

        elif isinstance(expr, HIRBlockExpression):
            return self._lower_block_expression(expr)

        elif isinstance(expr, HIRIfExpression):
            return self._lower_if_expression(expr)

        elif isinstance(expr, HIRLoopExpression):
            return self._lower_loop_expression(expr)

        elif isinstance(expr, HIRStringLiteral):
            return self._lower_inline_string_literal(expr)

        else:
            # Unsupported expression type (placeholder)
            # Allocate placeholder virtual register
            vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, "unsupported")
            return vreg

    def _lower_status_flag_read(self, expr: HIRStatusFlagAccess) -> VirtualRegister:
        """
        Lower STATUS flag read to MIR.

        For reads like: let x = STATUS.Carry;
        Emits StatusFlagRead instruction which will generate:
        PHP; PLA; AND #mask; (normalize to 0/1)
        """
        from r65.compiler.hir.types import BasicTypeInfo

        result = self.current_function.vreg_allocator.alloc(
            BasicTypeInfo('bool'),
            f"status_{expr.flag_name.lower()}"
        )
        self.emit(StatusFlagRead(
            dest=result,
            flag_name=expr.flag_name,
            bit_mask=expr.bit_mask
        ))
        return result

    def _lower_inline_string_literal(self, expr: HIRStringLiteral) -> VirtualRegister:
        """
        Lower an inline string literal to a *u8 pointer to ROM data.

        Generates a ROM data section for the string bytes and emits
        a LabelRef move to load the label address into a vreg.
        """
        # Generate unique label
        label = f"__str_{self._rom_data_counter}"
        self._rom_data_counter += 1

        # Create ROM data section from processed bytes
        rom_data = ROMDataRef(label=label, data=list(expr.processed_bytes), element_size=1)
        self._rom_data_sections.append(rom_data)

        # Allocate vreg for the pointer
        vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, f"str_ptr")

        # Emit move from label reference to vreg
        self.emit(Move(dest=vreg, source=LabelRef(label_name=label), type_info=expr.expr_type))

        return vreg

    def _lower_block_expression(self, expr: HIRBlockExpression) -> Union[VirtualRegister, HardwareRegister, Immediate]:
        """
        Lower block expression: { stmts; final_expr }

        Lowers all statements sequentially, then lowers the final expression
        and returns its result.
        """
        # Lower all statements
        for stmt in expr.statements:
            if self._block_has_terminator():
                break
            self.lower_statement(stmt)

        # Lower final expression and return its result
        return self.lower_expression(expr.final_expr)

    def _lower_if_expression(self, expr: HIRIfExpression) -> Union[VirtualRegister, HardwareRegister, Immediate]:
        r"""
        Lower if expression to conditional branches with a result value.

        Creates CFG:
            current_block
                |
            [condition check]
                |
            CondBranch
            /        \
        then_block   else_block
            |          |
        result=X   result=Y
            \        /
            merge_block

        Returns a virtual register holding the result.
        """
        # Try to evaluate condition at compile time
        const_result = self._try_eval_const_condition(expr.condition)

        if const_result is True:
            # Condition is always true - only evaluate then branch
            return self.lower_expression(expr.then_block)

        if const_result is False:
            # Condition is always false - only evaluate else branch
            return self.lower_expression(expr.else_block)

        # Non-constant condition - generate full control flow
        # Allocate result register
        result_vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, "if_result")

        # Create blocks
        then_block = self.cfg_builder.new_block()
        else_block = self.cfg_builder.new_block()
        merge_block = self.cfg_builder.new_block()

        # Lower condition with short-circuit evaluation
        self.cond_lowerer.lower_condition(expr.condition, then_block.block_id, else_block.block_id)

        # Lower then branch
        self.current_block = then_block
        then_result = self.lower_expression(expr.then_block)
        # Move result to result_vreg (must be done before jump, in the then block)
        if not self._block_has_terminator():
            self.emit(Move(dest=result_vreg, source=then_result, type_info=expr.expr_type))
            self.emit(Jump(target=merge_block.block_id))
            self.cfg_builder.add_edge(then_block, merge_block)

        # Lower else branch
        self.current_block = else_block
        else_result = self.lower_expression(expr.else_block)
        if not self._block_has_terminator():
            self.emit(Move(dest=result_vreg, source=else_result, type_info=expr.expr_type))
            self.emit(Jump(target=merge_block.block_id))
            self.cfg_builder.add_edge(else_block, merge_block)

        # Continue at merge block
        self.current_block = merge_block

        return result_vreg

    def _lower_loop_expression(self, expr: HIRLoopExpression) -> VirtualRegister:
        """
        Lower loop expression to MIR.

        Similar to lower_while_statement but allocates a result vreg
        that break statements write to before jumping to exit.
        """
        # Allocate result register
        result_vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, "loop_result")

        # Create header and exit blocks (same as infinite loop)
        header_block = self.cfg_builder.new_block()
        body_block = self.cfg_builder.new_block()
        exit_block = self.cfg_builder.new_block()

        # Jump to header
        self.emit(Jump(target=header_block.block_id))
        self.cfg_builder.add_edge(self.current_block, header_block)

        # Header: jump to body (infinite loop)
        self.current_block = header_block
        self.emit(Jump(target=body_block.block_id))
        self.cfg_builder.add_edge(header_block, body_block)

        # Body: track loop context with result vreg
        self.current_block = body_block
        self.loop_stack.append((header_block.block_id, exit_block.block_id, expr.label, result_vreg))
        self.lower_block(expr.body)
        self.loop_stack.pop()

        # Jump back to header (unless body ends with break/return)
        if not self._block_has_terminator():
            self.emit(Jump(target=header_block.block_id))
            self.cfg_builder.add_edge(body_block, header_block)

        # Continue at exit block
        self.current_block = exit_block

        return result_vreg

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
                # Handle else-if chains (else_block can be HIRIfStmt)
                if isinstance(stmt.else_block, HIRIfStmt):
                    self.lower_if_statement(stmt.else_block)
                else:
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
            # Handle else-if chains (else_block can be HIRIfStmt)
            if isinstance(stmt.else_block, HIRIfStmt):
                self.lower_if_statement(stmt.else_block)
            else:
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
            # Use condition lowerer for proper Compare + CondBranch generation
            self.cond_lowerer.lower_condition(stmt.condition, body_block.block_id, exit_block.block_id)
        else:
            # loop: infinite loop (fallback)
            self.emit(Jump(target=body_block.block_id))
            self.cfg_builder.add_edge(header_block, body_block)

        # Body: track loop context for break/continue
        self.current_block = body_block
        # Push loop context: (continue_target=header, break_target=exit, label, result_vreg)
        self.loop_stack.append((header_block.block_id, exit_block.block_id, stmt.label, None))
        self.lower_block(stmt.body)
        self.loop_stack.pop()

        # Jump back to header (unless body ends with break/return)
        if not self._block_has_terminator():
            self.emit(Jump(target=header_block.block_id))
            self.cfg_builder.add_edge(body_block, header_block)

        # Continue at exit block
        self.current_block = exit_block

    def _find_loop_target(self, label: str = None) -> tuple:
        """
        Find the loop target for break/continue.

        If label is None, returns the innermost loop.
        If label is specified, searches the stack for matching label.

        Returns: (continue_target, break_target, label, result_vreg_or_None)
        """
        if not self.loop_stack:
            raise MIRLoweringError("Break/continue statement outside of loop", source_loc=self._current_source_loc)

        if label is None:
            # Use innermost loop
            return self.loop_stack[-1]

        # Search for labeled loop (from innermost to outermost)
        for loop_ctx in reversed(self.loop_stack):
            if loop_ctx[2] == label:
                return loop_ctx

        raise MIRLoweringError(f"Label '{label}' not found in enclosing loops", source_loc=self._current_source_loc)

    def lower_break_statement(self, stmt: HIRBreakStmt):
        """
        Lower break statement.

        Jumps to the exit block of the target loop.
        If labeled, jumps to the exit block of the labeled loop.
        If break has a value (loop expression), emit Move to result vreg before jump.
        """
        continue_target, break_target, _, result_vreg = self._find_loop_target(stmt.label)

        # If break has value and loop has result vreg, emit Move
        if stmt.value is not None and result_vreg is not None:
            val = self.lower_expression(stmt.value)
            self.emit(Move(dest=result_vreg, source=val, type_info=stmt.value.expr_type))

        self.emit(Jump(target=break_target))

        # Add CFG edge
        break_block = self.cfg_builder.get_block(break_target)
        self.cfg_builder.add_edge(self.current_block, break_block)

    def lower_continue_statement(self, stmt: HIRContinueStmt):
        """
        Lower continue statement.

        Jumps to the header block of the target loop.
        If labeled, jumps to the header block of the labeled loop.
        """
        continue_target, break_target, _, _ = self._find_loop_target(stmt.label)
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
            symbol: Symbol for reference (base_memloc.symbol used for allocation lookup)

        Returns:
            MemoryLocation at base + offset
        """
        # Use the base's symbol for allocation resolution — this is critical for
        # promoted aggregate locals where get_memory_location() already redirected
        # to the synthetic static symbol
        alloc_symbol = base_memloc.symbol

        if base_memloc.address is not None:
            # Address known - compute absolute address
            return MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=base_memloc.address + offset,
                symbol=alloc_symbol,
                is_volatile=base_memloc.is_volatile
            )
        else:
            # Address not known at MIR time (auto-allocated) -
            # keep address=None and store offset for codegen resolution
            return MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=None,
                symbol=alloc_symbol,
                is_volatile=base_memloc.is_volatile,
                offset=offset
            )

    def _count_param_usages(self, hir_func: HIRFunctionDecl) -> Dict[int, int]:
        """
        Count how many times each parameter symbol is used in the function body.

        This is needed to determine if register-aliased parameters need to be
        saved to a vreg at function entry (if used more than once).

        Args:
            hir_func: HIR function declaration

        Returns:
            Dict mapping symbol id to usage count
        """
        counts: Dict[int, int] = {}

        # Build set of parameter symbol ids
        param_ids = set()
        for param in hir_func.parameters:
            if isinstance(param.binding, RegisterBinding):
                param_ids.add(id(param.symbol))

        if not param_ids:
            return counts

        def count_in_expr(expr):
            """Recursively count parameter uses in an expression."""
            if expr is None:
                return

            if isinstance(expr, HIRIdentifier):
                sym_id = id(expr.symbol)
                if sym_id in param_ids:
                    counts[sym_id] = counts.get(sym_id, 0) + 1
            elif isinstance(expr, HIRBinaryOp):
                count_in_expr(expr.left)
                count_in_expr(expr.right)
            elif isinstance(expr, HIRUnaryOp):
                count_in_expr(expr.operand)
            elif isinstance(expr, HIRTypeCast):
                count_in_expr(expr.expr)
            elif isinstance(expr, HIRAssignment):
                count_in_expr(expr.value)
                # Don't count target - writing to param doesn't need reading it
            elif isinstance(expr, HIRFunctionCall):
                for arg in expr.args:
                    count_in_expr(arg)
            elif isinstance(expr, HIRMethodCall):
                count_in_expr(expr.receiver)
                for arg in expr.args:
                    count_in_expr(arg)
            elif isinstance(expr, HIRArrayIndex):
                count_in_expr(expr.array)
                count_in_expr(expr.index)
            elif isinstance(expr, HIRFieldAccess):
                count_in_expr(expr.base)
            elif isinstance(expr, HIRDereference):
                count_in_expr(expr.operand)
            elif isinstance(expr, HIRAddressOf):
                count_in_expr(expr.operand)
            elif isinstance(expr, HIRBlockExpression):
                for s in expr.statements:
                    count_in_stmt(s)
                count_in_expr(expr.final_expr)
            elif isinstance(expr, HIRIfExpression):
                count_in_expr(expr.condition)
                count_in_expr(expr.then_block)
                count_in_expr(expr.else_block)

        def count_in_stmt(stmt):
            """Recursively count parameter uses in a statement."""
            if stmt is None:
                return

            if isinstance(stmt, HIRLetStmt):
                count_in_expr(stmt.initializer)
            elif isinstance(stmt, HIRExprStmt):
                count_in_expr(stmt.expr)
            elif isinstance(stmt, HIRReturnStmt):
                for val in stmt.values:
                    count_in_expr(val)
            elif isinstance(stmt, HIRIfStmt):
                count_in_expr(stmt.condition)
                count_in_block(stmt.then_block)
                if stmt.else_block:
                    if isinstance(stmt.else_block, HIRIfStmt):
                        count_in_stmt(stmt.else_block)
                    else:
                        count_in_block(stmt.else_block)
            elif isinstance(stmt, HIRWhileStmt):
                count_in_expr(stmt.condition)
                count_in_block(stmt.body)

        def count_in_block(block):
            """Count parameter uses in a block."""
            if block is None:
                return
            for stmt in block.statements:
                count_in_stmt(stmt)

        # Count in function body
        count_in_block(hir_func.body)

        return counts

    def _block_has_terminator(self) -> bool:
        """
        Check if current block already has a terminator instruction.

        Terminator instructions: Return, ReturnFromInterrupt, Jump, CondBranch

        Returns:
            True if block ends with terminator, False otherwise
        """
        if not self.current_block or not self.current_block.instructions:
            return False

        last_instr = self.current_block.instructions[-1]
        return isinstance(last_instr, (Return, ReturnFromInterrupt, Jump, CondBranch))

    def emit(self, instruction: MIRInstruction):
        """
        Emit an instruction to the current block.

        Args:
            instruction: MIR instruction to emit
        """
        # Propagate source location to instruction if not already set
        if instruction.source_loc is None and self._current_source_loc is not None:
            instruction.source_loc = self._current_source_loc

        if self.current_block is not None:
            self.current_block.instructions.append(instruction)

    def has_explicit_location(self, symbol) -> bool:
        """
        Check if symbol has explicit memory location.

        Args:
            symbol: HIR Symbol

        Returns:
            True if symbol has explicit location (static or promoted aggregate local)
        """
        return symbol.kind == SymbolKind.STATIC_VAR or id(symbol) in self._promoted_locals

    def get_memory_location(self, symbol) -> MemoryLocation:
        """
        Get memory location for symbol.

        Args:
            symbol: HIR Symbol

        Returns:
            MemoryLocation
        """
        # Check if this is a promoted aggregate local — redirect to synthetic static
        if id(symbol) in self._promoted_locals:
            return self.get_memory_location(self._promoted_locals[id(symbol)])

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
            # Immutable statics without storage attribute are ROM
            elif hasattr(static_decl, 'is_mutable') and not static_decl.is_mutable:
                return MemoryLocation(
                    storage_type='rom',
                    address=None,
                    symbol=symbol,
                    is_volatile=False
                )

        # Default: unknown storage (will be allocated later)
        return MemoryLocation(
            storage_type='unknown',
            address=None,
            symbol=symbol,
            is_volatile=False
        )

    def _get_callee_return_registers(self, expr) -> list:
        """
        Determine the return register order for a callee expression.

        Looks up the callee's return type and entry mode to determine
        if B register should be used for (u8, u8) tuple returns.

        Args:
            expr: The initializer expression (typically HIRFunctionCall)

        Returns:
            List of register names in order, e.g. ['A', 'B', 'X', 'Y']
        """
        from r65.compiler.codegen.constants import get_return_registers

        if isinstance(expr, HIRFunctionCall):
            # Look up the callee function declaration
            func_name = None
            if isinstance(expr.func, HIRIdentifier):
                func_name = expr.func.name
            elif isinstance(expr.func, HIRFunctionAddress):
                func_name = expr.function_name

            if func_name and func_name in self.function_decls:
                func_decl = self.function_decls[func_name]
                return get_return_registers(
                    func_decl.return_type,
                    func_decl.entry_m_mode
                )

        return ['A', 'X', 'Y']

    def _get_type_size(self, type_info) -> int:
        """Get size in bytes for a type. Delegates to TypeSizeCalculator."""
        return TypeSizeCalculator.get_size(type_info)

    def _try_eval_const_condition(self, expr: HIRExpression) -> Optional[bool]:
        """Try to evaluate a condition expression at compile time for dead code elimination."""
        from r65.compiler.hir.hir_const_eval import try_eval_const_bool
        symbol_table = self._hir_program.symbol_table if self._hir_program else None
        return try_eval_const_bool(expr, symbol_table)
