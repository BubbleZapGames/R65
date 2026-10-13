# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
HIR → MIR lowering.

Transforms type-checked HIR into MIR suitable for code generation.
"""

from typing import Optional, Dict, Any, List, Union
from r65.compiler.hir import (
    HIRProgram, HIRFunctionDecl, HIRStaticDecl, HIRStatement, HIRBlock, HIRLetStmt, HIRMultiLetStmt, HIRExprStmt, HIRReturnStmt,
    HIRIfStmt, HIRWhileStmt, HIRBreakStmt, HIRContinueStmt, HIRAsmStmt,
    HIRExpression, HIRIntegerLiteral, HIRBooleanLiteral, HIREnumVariantExpr, HIRIdentifier,
    HIRFunctionAddress, HIRRegister, HIRStatusFlagAccess, HIRBinaryOp, HIRUnaryOp, HIRTypeCast, HIRAssignment,
    HIRFunctionCall, HIRMethodCall, HIRArrayIndex, HIRFieldAccess, HIRDereference, HIRAddressOf, HIRMultiAssignment,
    HIRArrayFillExpr, HIRArrayLiteralExpr, HIRStringLiteral, HIRStructLiteralExpr,
    HIRMatchExpression, HIRBlockExpression, HIRIfExpression, HIRLoopExpression,
    RegisterBinding, VariableBinding,
    SymbolKind,
)
from r65.compiler.hir.attributes import StorageKind

from r65.compiler.mir.nodes import (
    MIRInstruction, MIRProgram, MIRFunction, BasicBlock,
    VirtualRegister, HardwareRegister, Immediate, FunctionPointer, LabelRef, MemoryLocation,
    Load, Store, Move, Jump, CondBranch, Return, ReturnFromInterrupt, Call, StatusFlagRead,
    MemoryFill, BlockCopy, AggregateCopy, ROMDataRef,
    InlineAsm, TERMINATOR_TYPES,
)

from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.mir.register_tracker import RegisterAliasTracker
from r65.compiler.mir.register_alias_check import check_register_aliases
from r65.compiler.mir.cfg import CFGBuilder
from r65.compiler.mir.mode_tracker import reanalyze_function as _reanalyze_modes
from r65.compiler.hir.unified_type_utils import get_unified_type_size
from r65.compiler.mir.context import LoweringContext
from r65.compiler.mir.lowerers.expression import ExpressionLowerer
from r65.compiler.mir.lowerers.match import MatchLowerer
from r65.compiler.mir.lowerers.call import CallLowerer
from r65.compiler.mir.lowerers.assignment import AssignmentLowerer
from r65.compiler.mir.lowerers.condition import ConditionLowerer
from r65.compiler.mir.lowerers.static_init import StaticInitLowerer
from r65.compiler.mir.lowerers.multiply import compute_scaled_index
from r65.compiler.typeck.processor_mode import ProcessorMode, XModeState
from r65.compiler.errors import MIRLoweringError, InternalCompilerError
from r65.compiler.typeck.type_utils import TypeUtils
from r65.compiler.hir.nodes import walk_hir


class MIRBuilder:
    """
    Lowers HIR to MIR.

    Lowers one function at a time: vregs are allocated and register aliases
    recorded during lowering, then `check_register_aliases` runs on each
    finished function.
    """

    def __init__(self, abi_kind=None):
        # ABI kind for calling convention decisions during MIR lowering
        # When Pascal, all parameters become stack parameters
        from r65.compiler.codegen.abi_model import ABIKind
        self.abi_kind = abi_kind or ABIKind.DEFAULT

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
        # Per enclosing loop: {(id(symbol), stride): vreg} for a multi-index's
        # inner index, scaled once in the loop's preheader (see
        # _hoist_invariant_index_terms).
        self._hoisted_index_terms: List[Dict[Any, Any]] = []

        # Decomposed struct locals: maps original symbol id -> dict of field_name -> VirtualRegister
        # Small flat structs are decomposed into per-field vregs instead of promoted to static storage
        # Cleared per-function in lower_function()
        self._decomposed_structs: Dict[int, Dict[str, VirtualRegister]] = {}

        # Counter for unique promoted local names (avoids collisions)
        self._promoted_local_counter = 0

        # Synthetic static declarations from promoted aggregate locals
        # Appended to MIRProgram.statics at program build time
        self._promoted_statics: List[Any] = []

        # Current source location for debug info propagation
        self._current_source_loc = None

    # ========================================================================
    # Context Property Accessors
    # ========================================================================
    # Delegates to `self.ctx`, so lowerers and builder code share one state
    # object.

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
        # One path for both routes into a block switch — `match` lowering calls
        # ctx.set_current_block directly, so the invalidation cannot live here.
        self.ctx.set_current_block(value)

    @property
    def _x_index_cache(self) -> Optional[tuple]:
        return self.ctx.x_index_cache

    @_x_index_cache.setter
    def _x_index_cache(self, value: Optional[tuple]):
        self.ctx.x_index_cache = value

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

        # Lower each function. A const fn returning an aggregate has no
        # runtime form (an aggregate is never returned by value): every call
        # to it is folded.
        for decl in hir_program.declarations:
            if (isinstance(decl, HIRFunctionDecl) and decl.is_const and
                    TypeUtils.is_aggregate_type(decl.return_type)):
                continue
            if isinstance(decl, HIRFunctionDecl) and decl.body:
                mir_func = self.lower_function(decl)
                check_register_aliases(mir_func)
                mir_functions.append(mir_func)
            # Also lower methods from impl blocks
            elif isinstance(decl, HIRImplDecl):
                for method in decl.methods:
                    if method.body:
                        mir_func = self.lower_function(method)
                        check_register_aliases(mir_func)
                        mir_functions.append(mir_func)

        # Create MIR program (keep HIR declarations for statics, etc.)
        # Include any synthetic statics from promoted aggregate locals
        all_statics = list(hir_program.statics) + self._promoted_statics
        from r65.compiler.hir import HIRIncludeAsm
        asm_includes = [d for d in hir_program.declarations if isinstance(d, HIRIncludeAsm)]
        return MIRProgram(
            functions=mir_functions,
            statics=all_statics,
            constants=hir_program.constants,
            symbol_table=hir_program.symbol_table,
            stack_attr=hir_program.stack_attr,
            snesrom_config=hir_program.snesrom_config,
            rom_data_sections=self._rom_data_sections,
            trait_dispatch_info=hir_program.trait_dispatch_info,
            asm_includes=asm_includes,
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
            is_used=hir_func.is_used,
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
        self._hoisted_index_terms.clear()
        self._promoted_locals.clear()
        self._decomposed_structs.clear()

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

        # Save A-bound params first (B is read via XBA and X/Y via TXA/TYA,
        # both through A), then B, then X/Y.

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

        # Skip register parameter passes for Pascal ABI — all params go to stack
        from r65.compiler.codegen.abi_model import ABIKind
        if self.abi_kind != ABIKind.PASCAL:
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
        # In Pascal ABI, ALL parameters go on the stack (register/variable bindings ignored)
        from r65.compiler.codegen.abi_model import ABIKind
        is_pascal = self.abi_kind == ABIKind.PASCAL

        for idx, param in enumerate(hir_func.parameters):
            if is_pascal:
                # Pascal ABI: all params are stack params
                param_vreg = self.current_function.vreg_allocator.alloc(
                    param.param_type,
                    f"param_{param.name}"
                )
                self.symbol_to_vreg[id(param.symbol)] = param_vreg
                stack_params.append((idx, param, param_vreg))
            elif isinstance(param.binding, RegisterBinding):
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
        if stack_params:
            from r65.compiler.hir.types import PointerTypeInfo, FunctionTypeInfo
            from r65.compiler.codegen.abi import ABIInfo

            abi = ABIInfo(is_far=hir_func.is_far)

            if self.abi_kind == ABIKind.PASCAL:
                # Pascal convention: left-to-right push order.
                # Caller pushes: [result_space] [param0] [param1] ... [paramN] [JSR]
                # So paramN is closest to return addr, param0 is deepest.
                #
                # Stack layout (from SP upward):
                #     SP -> [return addr]
                #           [paramN]          <- offset: ret_addr_size + 1
                #           [param(N-1)]
                #           ...
                #           [param0]
                #           [result_space]    <- if non-void return
                #
                # Compute total param bytes first
                total_param_bytes = sum(
                    self._get_type_size(p.param_type) for _, p, _ in stack_params
                )

                # Assign offsets: paramN at base, working backwards to param0
                current_offset = abi.return_addr_size + 1
                for idx, param, param_vreg in reversed(stack_params):
                    mir_func.stack_param_offsets[idx] = current_offset
                    mir_func.param_to_vreg[idx] = param_vreg

                    if isinstance(param.param_type, PointerTypeInfo) and param.param_type.is_far:
                        mir_func.has_far_ptr_stack_params = True
                        mir_func.far_ptr_param_indices.add(idx)
                    elif isinstance(param.param_type, FunctionTypeInfo) and param.param_type.is_far:
                        # Track far fn pointer params separately. Do NOT flip
                        # has_far_ptr_stack_params here — that decision is made
                        # later by the cost-modeled strategy analysis.
                        mir_func.fn_ptr_param_indices.add(idx)

                    param_size = self._get_type_size(param.param_type)
                    current_offset += param_size

                # Track total param bytes and result space for callee cleanup
                mir_func.pascal_total_param_bytes = total_param_bytes
                if hir_func.return_type:
                    from r65.compiler.hir.types import BasicTypeInfo, NeverTypeInfo
                    if (not isinstance(hir_func.return_type, NeverTypeInfo) and
                        not (isinstance(hir_func.return_type, BasicTypeInfo) and
                             hir_func.return_type.name == 'void')):
                        mir_func.pascal_result_space_bytes = self._get_type_size(hir_func.return_type)
            else:
                # Default/FixedStack convention: right-to-left push order.
                # Stack layout after JSR/JSL (caller pushes params right-to-left):
                #     | param N             |  <- higher offset (pushed first)
                #     | ...                 |
                #     | param 0             |  <- SP + return_addr_size + 1 (pushed last)
                #     | return addr (2-3b)  |  <- SP + 1
                #     SP ->
                #
                # Parameters with lower index are at lower stack offsets
                current_offset = abi.return_addr_size + 1  # First param starts after return address

                for idx, param, param_vreg in stack_params:
                    mir_func.stack_param_offsets[idx] = current_offset
                    mir_func.param_to_vreg[idx] = param_vreg

                    # Check if this is a far pointer parameter
                    if isinstance(param.param_type, PointerTypeInfo) and param.param_type.is_far:
                        mir_func.has_far_ptr_stack_params = True
                        mir_func.far_ptr_param_indices.add(idx)
                    elif isinstance(param.param_type, FunctionTypeInfo) and param.param_type.is_far:
                        # Track far fn pointer params separately. Do NOT flip
                        # has_far_ptr_stack_params here — that decision is made
                        # later by the cost-modeled strategy analysis.
                        mir_func.fn_ptr_param_indices.add(idx)

                    # Advance offset by parameter size
                    param_size = self._get_type_size(param.param_type)
                    current_offset += param_size

        # If this is an entry point function, call __init_start() to initialize static data
        # (The native-mode switch is emitted by FunctionGenerator.emit_prologue,
        # before frame allocation.)
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

        # Perform mode tracking analysis. Going through the shared
        # reanalyze_function helper keeps the post-build path and the
        # post-inline path on the same code (clears any block.entry_mode
        # left from a hypothetical earlier pass, then runs the fixpoint).
        success = _reanalyze_modes(mir_func)
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
        elif isinstance(stmt, HIRMultiLetStmt):
            self.lower_multi_let_statement(stmt)
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
            raise InternalCompilerError(
                f"no MIR lowering for statement {type(stmt).__name__}",
                getattr(stmt, 'source_loc', None))

    def lower_let_statement(self, stmt: HIRLetStmt):
        """
        Lower let binding.

        Args:
            stmt: HIR let statement
        """
        # Defensive: if a binding reuses the cached index symbol, the cache
        # is stale. (Register-bound lets that touch X are already caught by
        # the emit-time X-clobber check.)
        self.x_index_cache_invalidate_symbol(getattr(stmt, 'symbol', None))

        if isinstance(stmt.binding, RegisterBinding):
            # Register alias: track in alias tracker with binding type
            hw_reg = HardwareRegister(stmt.binding.register_name)
            self.current_function.alias_tracker.add_alias(
                stmt.symbol,
                hw_reg,
                stmt.symbol.scope_id,
                binding_type=stmt.var_type  # Track the type for mode optimization
            )

            # If there's an initializer, load it into hardware register. The
            # Move's dest carries the alias (a definition, for
            # register_alias_check); with no Move, the definition is noted
            # where it falls.
            hw_reg = self.current_function.alias_tracker.get_alias(stmt.symbol)
            if stmt.initializer:
                init_value = self.lower_expression(stmt.initializer)
                if isinstance(init_value, HardwareRegister) and init_value.name == hw_reg.name:
                    self.current_function.alias_tracker.record_def(
                        stmt.symbol, self.current_block.block_id,
                        len(self.current_block.instructions),
                        value=init_value, source_loc=stmt.source_loc)
                else:
                    # Move to hardware register if not already there
                    self.emit(Move(
                        dest=hw_reg,
                        source=init_value,
                        type_info=stmt.var_type,
                    ))

        else:
            # Check if this is an aggregate type (struct or array) that needs
            # promotion to static storage — the 65816 lacks stack-indexed
            # addressing modes needed for variable-index array access on stack
            from r65.compiler.hir.types import ArrayTypeInfo, StructTypeInfo
            if isinstance(stmt.var_type, (ArrayTypeInfo, StructTypeInfo)):
                # A clone initializer (`let c = a.clone()`) needs a real memory
                # location to copy into, so force promotion (skip decomposition).
                if stmt.initializer is not None and getattr(stmt.initializer, 'clone_info', None):
                    self._promote_aggregate_local(stmt)
                elif self._is_stack_eligible_struct(stmt):
                    self._decompose_struct_local(stmt)
                else:
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
                        # Check if init_value is already bound to another symbol.
                        # If so, reusing it would alias two symbols to the same
                        # slot — mutations of one (e.g. cy--) would corrupt reads
                        # of the other (e.g. height).  Create a fresh vreg + Move
                        # so each symbol has its own storage.  The slot allocator's
                        # coalescing pass merges them back when safe.
                        #
                        # When init_value is an unbound temporary (function call
                        # result, array load, etc.), reusing is safe since no
                        # other symbol references it.
                        source_is_bound = any(
                            v is init_value
                            for v in self.symbol_to_vreg.values()
                        )
                        if source_is_bound:
                            vreg = self.current_function.vreg_allocator.alloc(
                                stmt.var_type, stmt.name, register_hint=register_hint)
                            self.symbol_to_vreg[id(stmt.symbol)] = vreg
                            self.emit(Move(dest=vreg, source=init_value, type_info=stmt.var_type))
                        else:
                            # Safe to reuse — propagate register hint
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

        # Clone initializer: `let c = a.clone()` — copy the source aggregate's bytes.
        if getattr(initializer, 'clone_info', None):
            self._emit_clone_init(stmt, mem_loc, initializer)
            return

        fill_value = None
        if isinstance(initializer, HIRArrayFillExpr):
            fill_value = self.static_init_lowerer._extract_constant_value(initializer.fill_value)
            if fill_value is None:
                # An aggregate fill value (`[[7; 2]; 3]`) is no single value
                # to store: copy it as the literal it expands to.
                initializer = HIRArrayLiteralExpr(
                    elements=[initializer.fill_value] * initializer.count,
                    source_loc=initializer.source_loc,
                )

        if fill_value is not None:
            # Array fill of a constant scalar: [0; 16] → MemoryFill
            from r65.compiler.hir.types import ArrayTypeInfo
            array_type = stmt.var_type
            if isinstance(array_type, ArrayTypeInfo):
                element_size = self._get_type_size(array_type.element_type)
            else:
                element_size = 1
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
        data_bytes, element_size = self.static_init_lowerer.array_literal_bytes(
            literal_expr, stmt.var_type, f"element of local array `{stmt.name}`")
        self._emit_inline_rom_copy(mem_loc, data_bytes, element_size, label_prefix)

    def _emit_inline_string_literal(self, stmt, mem_loc, string_literal, label_prefix):
        """Emit ROMDataRef + BlockCopy for an inline string literal initializer."""
        data_bytes = self.static_init_lowerer.string_literal_bytes(string_literal, stmt.var_type)
        self._emit_inline_rom_copy(mem_loc, data_bytes, 1, label_prefix)

    def _emit_inline_struct_literal(self, stmt, mem_loc, struct_expr, label_prefix):
        """Emit ROMDataRef + BlockCopy for an inline struct literal initializer."""
        data_bytes = self.static_init_lowerer._extract_struct_literal_bytes(struct_expr)
        self._emit_inline_rom_copy(mem_loc, data_bytes, 1, label_prefix)

    def _emit_inline_rom_copy(self, mem_loc, data_bytes, element_size, label_prefix):
        """Copy *data_bytes* into a promoted local from the function's own ROM
        data, emitted next to its body."""
        rom_data = ROMDataRef(label=f"__{label_prefix}_data", data=data_bytes,
                              element_size=element_size)
        self.current_function.local_rom_data.append(rom_data)
        self.emit(BlockCopy(dest=mem_loc, rom_data=rom_data, count=len(data_bytes)))

    def _clone_place_memloc(self, expr) -> Optional[MemoryLocation]:
        """Resolve a clone operand (a place, or &place) to a static MemoryLocation.

        Clone operands are force-promoted, so identifiers resolve to a static or a
        promoted-local static. Returns None for forms not yet supported.
        """
        if isinstance(expr, HIRAddressOf):
            expr = expr.operand
        if isinstance(expr, HIRIdentifier):
            symbol = expr.symbol
            if id(symbol) in self._promoted_locals:
                symbol = self._promoted_locals[id(symbol)]
            return self.get_memory_location(symbol)
        return None

    def _emit_clone_init(self, stmt: HIRLetStmt, dest_loc: MemoryLocation, clone_expr):
        """Emit the copy for `let c = a.clone()` (intrinsic auto-struct / array)."""
        info = clone_expr.clone_info
        src_loc = self._clone_place_memloc(clone_expr.func.base)
        if src_loc is None:
            raise MIRLoweringError(
                "clone source must be a named aggregate variable or static",
                source_loc=getattr(clone_expr, 'source_loc', None)
            )
        count = self._get_type_size(info['agg_type'])
        self.emit(AggregateCopy(dest=dest_loc, src=src_loc, count=count))

    def _lower_intrinsic_clone_from(self, call_expr):
        """Lower `dst.clone_from(&src)` for intrinsic (auto-struct / array) clones."""
        info = call_expr.clone_info
        dest_loc = self._clone_place_memloc(call_expr.func.base)
        src_loc = self._clone_place_memloc(call_expr.args[0])
        if dest_loc is None or src_loc is None:
            raise MIRLoweringError(
                "clone_from operands must be named aggregate variables or statics",
                source_loc=getattr(call_expr, 'source_loc', None)
            )
        count = self._get_type_size(info['agg_type'])
        self.emit(AggregateCopy(dest=dest_loc, src=src_loc, count=count))
        return None

    def _lower_clone_assignment(self, expr):
        """Lower `dst = src.clone()` to an aggregate copy into the target place."""
        clone_expr = expr.value
        info = clone_expr.clone_info
        dest_loc = self._clone_place_memloc(expr.target)
        src_loc = self._clone_place_memloc(clone_expr.func.base)
        if dest_loc is None or src_loc is None:
            raise MIRLoweringError(
                "clone assignment operands must be named aggregate variables or statics",
                source_loc=getattr(expr, 'source_loc', None)
            )
        count = self._get_type_size(info['agg_type'])
        self.emit(AggregateCopy(dest=dest_loc, src=src_loc, count=count))
        return None

    def _is_stack_eligible_struct(self, stmt: HIRLetStmt) -> bool:
        """
        Check if a local struct can be decomposed into per-field vregs.

        Eligible structs are small, flat (all-scalar fields), and their address
        is never taken. This allows them to be stack-allocated via vregs instead
        of promoted to static storage, making them recursion-safe.

        Args:
            stmt: HIR let statement

        Returns:
            True if the struct can be decomposed into per-field vregs
        """
        from r65.compiler.hir.types import (
            StructTypeInfo, BasicTypeInfo, PointerTypeInfo, EnumTypeInfo, NewtypeTypeInfo
        )
        if not isinstance(stmt.var_type, StructTypeInfo):
            return False
        struct_decl = stmt.var_type.definition
        if struct_decl is None:
            return False
        # Unions are never decomposed: their fields deliberately share storage, and
        # giving each one its own vreg would break the aliasing that is the whole
        # point of a union.
        if getattr(struct_decl, 'is_union', False):
            return False
        # Check total size < 16 bytes (struct layout — see layout_fields)
        total_size = sum(self._get_type_size(f.field_type) for f in struct_decl.fields)
        if total_size >= 16:
            return False
        # All fields must be scalar (no arrays, no nested structs). A newtype is a
        # scalar, so a field of one decomposes like the payload it wraps.
        for f in struct_decl.fields:
            if not isinstance(f.field_type,
                              (BasicTypeInfo, PointerTypeInfo, EnumTypeInfo, NewtypeTypeInfo)):
                return False
        # Check address-of is not taken in the function body
        if self._symbol_address_taken(stmt.symbol):
            return False
        return True

    def _symbol_address_taken(self, symbol) -> bool:
        """Whether the current function's body takes *symbol*'s address.

        `&sym`, a method-style call on it (the receiver is passed or copied by
        address), and an operator-trait compound assignment to it all pin it
        to a memory location.
        """
        func_decl = self.function_decls.get(self.current_function.name)
        if func_decl is None or not func_decl.body:
            return False

        def is_sym(expr):
            return isinstance(expr, HIRIdentifier) and expr.symbol is symbol

        # An operator-trait call hangs off its assignment as an attribute, not
        # a field, so the walk is restarted from it.
        roots = [func_decl.body]
        while roots:
            for node in walk_hir(roots.pop()):
                if isinstance(node, HIRAddressOf) and is_sym(node.operand):
                    return True
                if (isinstance(node, HIRFunctionCall)
                        and isinstance(node.func, HIRFieldAccess)
                        and is_sym(node.func.base)):
                    return True
                call = getattr(node, 'opassign_call', None)
                if isinstance(node, HIRAssignment) and call is not None:
                    if is_sym(node.target):
                        return True
                    roots.append(call)
        return False

    def _decompose_struct_local(self, stmt: HIRLetStmt):
        """
        Decompose a small flat struct into per-field virtual registers.

        Instead of promoting to static storage, each field gets its own vreg.
        This makes the struct recursion-safe and potentially faster (fields can
        be register-allocated).

        Args:
            stmt: HIR let statement with eligible struct type
        """
        struct_decl = stmt.var_type.definition
        field_vregs = {}

        for f in struct_decl.fields:
            vreg = self.current_function.vreg_allocator.alloc(
                f.field_type, f"{stmt.name}__{f.name}"
            )
            field_vregs[f.name] = vreg

        self._decomposed_structs[id(stmt.symbol)] = field_vregs

        # Handle initializer
        if stmt.initializer is not None and isinstance(stmt.initializer, HIRStructLiteralExpr):
            for field_init in stmt.initializer.fields:
                field_vreg = field_vregs.get(field_init.name)
                if field_vreg is not None:
                    init_value = self.lower_expression(field_init.value)
                    self.emit(Move(dest=field_vreg, source=init_value, type_info=field_vreg.type_info))

    def lower_multi_let_statement(self, stmt: HIRMultiLetStmt):
        """
        Lower multi-binding let statement: let a, b = multi_return_func();

        Return values are in registers determined by callee's return type.
        For (u8, u8) in m8 mode: A, B, X, Y order.
        Otherwise: A, X, Y order.
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

            # Check if this is a decomposed struct (cannot be used as scalar)
            if id(symbol) in self._decomposed_structs:
                raise MIRLoweringError(
                    f"Cannot use decomposed struct '{symbol.name}' as a scalar value; "
                    f"access individual fields (e.g., {symbol.name}.field_name)",
                    source_loc=expr.source_loc
                )

            # Check if aliased to hardware register
            hw_reg = self.current_function.alias_tracker.get_alias(symbol)
            if hw_reg:
                return hw_reg  # Direct hardware register reference

            # Check if this is a volatile (hardware) symbol — must re-read every time
            is_volatile_sym = (
                symbol.kind == SymbolKind.STATIC_VAR
                and hasattr(symbol.definition, 'storage_attr')
                and symbol.definition.storage_attr is not None
                and symbol.definition.storage_attr.storage_kind == StorageKind.HW
            )

            # Check if already in a virtual register (skip cache for volatile)
            symbol_id = id(symbol)
            if not is_volatile_sym and symbol_id in self.symbol_to_vreg:
                return self.symbol_to_vreg[symbol_id]

            # Otherwise, load from memory
            vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, symbol.name)
            mem_loc = self.get_memory_location(symbol)
            self.emit(Load(dest=vreg, source=mem_loc, type_info=expr.expr_type))
            if not is_volatile_sym:
                self.symbol_to_vreg[symbol_id] = vreg
                # Track that this is a memory-location cache entry so it can
                # be invalidated at branch merge points.
                self.ctx.memloc_cached_symbols.add(symbol_id)
            return vreg

        elif isinstance(expr, HIRRegister):
            # Direct hardware register reference
            return HardwareRegister(expr.name)

        elif isinstance(expr, HIRFunctionAddress):
            # Function address: a Move from a FunctionPointer, resolved by the
            # assembler.
            vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, f"fn_ptr_{expr.function_name}")
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
            result = self.call_lowerer.lower_function_call(expr)
            # Invalidate cached memloc vregs — the callee may have
            # modified any static variable via its memory location.
            self._invalidate_memloc_cache()
            return result

        elif isinstance(expr, HIRMethodCall):
            result = self.call_lowerer.lower_method_call(expr)
            self._invalidate_memloc_cache()
            return result

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
            raise InternalCompilerError(
                f"no MIR lowering for expression {type(expr).__name__}",
                getattr(expr, 'source_loc', None))

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
        # For diverging blocks (e.g. { return 1; }), final_expr is None
        if expr.final_expr is None:
            # Return a dummy vreg — this block diverges so the result is never used
            return self.current_function.vreg_allocator.alloc(expr.expr_type, "diverging_block")
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

        result_vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, "if_result")

        def arm(body):
            def lower():
                value = self.lower_expression(body)
                if not self._block_has_terminator():
                    self.emit(Move(dest=result_vreg, source=value, type_info=expr.expr_type))
            return lower

        self._lower_branches(expr.condition, arm(expr.then_block), arm(expr.else_block))
        return result_vreg

    def _lower_loop_expression(self, expr: HIRLoopExpression) -> VirtualRegister:
        """`loop { .. }` as a value: an infinite loop whose breaks write the
        result vreg before jumping to the exit."""
        result_vreg = self.current_function.vreg_allocator.alloc(expr.expr_type, "loop_result")
        self._lower_loop(expr.body, expr.label, None, True, result_vreg)
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

        def lower_else():
            # Handle else-if chains (else_block can be HIRIfStmt)
            if isinstance(stmt.else_block, HIRIfStmt):
                self.lower_if_statement(stmt.else_block)
            else:
                self.lower_block(stmt.else_block)

        self._lower_branches(stmt.condition,
                             lambda: self.lower_block(stmt.then_block),
                             lower_else if stmt.else_block else None)

    def _lower_branches(self, condition, lower_then, lower_else):
        """Branch on *condition* to a then block and (if *lower_else*) an else
        block, each lowered by its callback, then continue at the merge.

        Cached loads of statics are dropped between the arms - a load in the
        then arm must not be reused by the else arm, a sibling rather than a
        successor - and again at the merge, where either arm may have written
        them. Local-variable vregs persist: they ARE the variable's storage.
        Each arm's closing jump leaves from `current_block`, wherever its
        body ended.
        """
        then_block = self.cfg_builder.new_block()
        merge_block = self.cfg_builder.new_block()
        else_block = self.cfg_builder.new_block() if lower_else else merge_block

        self.cond_lowerer.lower_condition(condition, then_block.block_id, else_block.block_id)

        saved_memloc = {
            sym_id: self.symbol_to_vreg[sym_id]
            for sym_id in self.ctx.memloc_cached_symbols
            if sym_id in self.symbol_to_vreg
        }

        self.current_block = then_block
        lower_then()
        if not self._block_has_terminator():
            self.jump_to(merge_block)

        if lower_else:
            for sym_id in list(self.ctx.memloc_cached_symbols):
                if sym_id in self.symbol_to_vreg:
                    del self.symbol_to_vreg[sym_id]
            for sym_id, vreg in saved_memloc.items():
                self.symbol_to_vreg[sym_id] = vreg
            self.current_block = else_block
            lower_else()
            if not self._block_has_terminator():
                self.jump_to(merge_block)

        self._invalidate_memloc_cache()
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
        self._lower_loop(stmt.body, stmt.label, stmt.condition, stmt.is_infinite, None)

    def _lower_loop(self, body, label, condition, is_infinite: bool, result_vreg):
        """Header, body and exit blocks for a while, loop or loop expression.
        *result_vreg* is where a `break value` lands (None for statements)."""
        header_block = self.cfg_builder.new_block()
        body_block = self.cfg_builder.new_block()
        exit_block = self.cfg_builder.new_block()

        self._hoist_invariant_index_terms(body)
        self.jump_to(header_block)
        self.current_block = header_block

        const_cond = self._try_eval_const_condition(condition) if condition else None
        if const_cond is False:
            # Dead loop: `while false { }` - skip entirely
            self.jump_to(exit_block)
            self.current_block = exit_block
            self._hoisted_index_terms.pop()
            return
        if condition and const_cond is not True and not is_infinite:
            self.cond_lowerer.lower_condition(condition, body_block.block_id, exit_block.block_id)
        else:
            # `loop { }`, `while true { }`
            self.jump_to(body_block)

        # Loop context for break/continue: (continue=header, break=exit, ...)
        self.current_block = body_block
        self.loop_stack.append((header_block.block_id, exit_block.block_id, label, result_vreg))
        self.lower_block(body)
        self.loop_stack.pop()
        self._hoisted_index_terms.pop()

        # Back to the header, unless the body ended with break/return
        if not self._block_has_terminator():
            self.jump_to(header_block)
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
        self.jump_to(merge_block)

        # False block: result = 0
        self.current_block = false_block
        self.emit(Move(dest=result, source=Immediate(0), type_info=result_type))
        self.jump_to(merge_block)

        # Continue in merge block
        self.current_block = merge_block

        return result

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
                symbol=alloc_symbol
            )
        else:
            # Address not known at MIR time (auto-allocated) -
            # keep address=None and store offset for codegen resolution.
            # Add to any existing base offset so nested aggregate accesses
            # (e.g. STRUCT.inner_arr[i]) accumulate correctly.
            return MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=None,
                symbol=alloc_symbol,
                offset=base_memloc.offset + offset
            )

    def resolve_array_base_memloc(self, array_expr):
        """Resolve an array lvalue expression to (base_memloc, reuse_base_key, terms).

        Supports:
          - HIRIdentifier: a bare static array symbol.
          - HIRFieldAccess: an array that is a field of a statically-located
            aggregate, e.g. STRUCT.array_field (including nested
            STRUCT.inner.array_field). The field offset is folded into the
            base memory location.
          - HIRArrayIndex: an array that is itself an element, e.g. the
            `T[i]` of `T[i][j]` or the `ROWS[i]` of `ROWS[i].name[j]`. Its
            index cannot ride on a MemoryLocation, so it is returned as a
            `(index_hir, stride)` term; the caller adds every term's scaled
            index into X (see multi_index_memloc). An `#[soa]` field reached
            this way (`POOL[i].arr`) is rebased onto its plane, and the term
            takes the plane's stride.

        terms is empty for a bare identifier. A field access passes through
        its base's terms (so `ROWS[i].name` carries `i`).

        reuse_base_key is a stable, hashable identity for the array base — the
        "array" half of an x-index reuse key (see x_index_reuse_key). It is
        derived structurally, not from node identity, so two separate accesses
        to the same array base (`s.arr[i].x = ..; s.arr[i].y = ..`) produce
        equal keys and can share the scaled index in X. The scaled index
        (index * element_size) is identical whether the array is top-level or a
        struct field — the differing base address is folded into the absolute
        address, leaving X reusable. Distinct field paths get distinct keys, so
        same-struct arrays of different element size (sprites[i] vs ext[i])
        never falsely share X. None disables reuse.
        """
        if isinstance(array_expr, HIRIdentifier):
            sym = array_expr.symbol
            return self.get_memory_location(sym), id(sym), []

        if isinstance(array_expr, HIRFieldAccess) and not getattr(array_expr, 'auto_deref', False):
            field_offset = array_expr.field_offset
            if field_offset is None:
                raise MIRLoweringError(
                    f"Field offset not computed for array field: {array_expr.field_name}",
                    source_loc=getattr(array_expr, 'source_loc', None)
                )
            base_memloc, base_key, terms = self.resolve_array_base_memloc(array_expr.base)
            if isinstance(array_expr.base, HIRArrayIndex):
                soa = self._soa_plane_for(base_memloc, field_offset)
                if soa is not None:
                    base_memloc, field_offset, stride, _ = self._soa_rebase(
                        soa, base_memloc, field_offset, None)
                    terms = terms[:-1] + [(terms[-1][0], stride)]
            offset_memloc = self._create_offset_memloc(
                base_memloc, field_offset, base_memloc.symbol)
            reuse_base_key = ('field', base_key, field_offset) if base_key is not None else None
            return offset_memloc, reuse_base_key, terms

        if isinstance(array_expr, HIRArrayIndex):
            from r65.compiler.hir.types import PointerTypeInfo
            if (isinstance(array_expr.array.expr_type, PointerTypeInfo)
                    or self.try_pointer_deref_array_base(array_expr.array) is not None):
                raise MIRLoweringError(
                    "indexing into an element reached through a pointer is not supported",
                    source_loc=getattr(array_expr, 'source_loc', None)
                )
            base_memloc, _key, terms = self.resolve_array_base_memloc(array_expr.array)
            stride = self._get_type_size(array_expr.expr_type)
            if isinstance(array_expr.array, HIRIdentifier):
                stride = self.plane_stride(array_expr.array.symbol, stride)
            # X will hold a sum, which the single-index reuse cache cannot key.
            return base_memloc, None, terms + [(array_expr.index, stride)]

        raise MIRLoweringError(
            f"Array base must be a static array or struct field, got: {type(array_expr)}",
            source_loc=getattr(array_expr, 'source_loc', None)
        )

    def sum_index_terms(self, terms, last_index, last_operand, last_stride):
        """`Σ index*stride` over an array base's runtime terms (from
        resolve_array_base_memloc) plus the outermost index *last_index*,
        already lowered into *last_operand*.

        Returns `(const, total)`: the constant indices folded to a byte count,
        and a u16 operand holding the sum of the variable ones (None if every
        index is constant).
        """
        from r65.compiler.mir.nodes import BinaryOp, HardwareRegister, Immediate, Move
        from r65.compiler.hir.types import BasicTypeInfo

        u16 = BasicTypeInfo('u16')
        const = 0
        total = None
        operands = [(last_index, last_operand, last_stride)]
        for index, stride in terms:
            hoisted = self._hoisted_index_term(index, stride)
            if hoisted is not None:
                # Scaled once, before the loop: add it as it is.
                operands.append((index, hoisted, 1))
            else:
                operands.append((index, self.lower_expression(index), stride))
        for index, operand, stride in operands:
            if isinstance(operand, Immediate):
                const += operand.value * stride
                continue
            if isinstance(operand, HardwareRegister):
                # Lowering the next index may go through this register.
                held = self.ctx.alloc_vreg(index.expr_type, 'idx')
                self.emit(Move(dest=held, source=operand, type_info=index.expr_type))
                operand = held
            scaled = self.scale_index(operand, stride, index.expr_type)
            if total is None:
                total = scaled
            else:
                summed = self.ctx.alloc_vreg(u16, 'idx_sum')
                self.emit(BinaryOp(dest=summed, left=scaled, right=total,
                                   op='+', type_info=u16))
                total = summed
        return const, total

    def _hoisted_index_term(self, index, stride):
        """The vreg holding *index* * *stride*, if a loop around this point
        scaled it in its preheader, else None."""
        if not isinstance(index, HIRIdentifier):
            return None
        key = (id(index.symbol), stride)
        for hoisted in reversed(self._hoisted_index_terms):
            if key in hoisted:
                return hoisted[key]
        return None

    def _hoist_invariant_index_terms(self, body):
        """Scale a loop's invariant inner indices once, before the loop.

        Called with the current block the loop's preheader. For every
        `T[i][j]` / `POOL[i].arr[j]` in *body* whose inner index `i` is a local
        or parameter the body never defines, emits `i * stride` here and
        records it, so sum_index_terms adds the vreg instead of rescaling `i`
        on every pass. Always pushes a (possibly empty) frame; the loop
        lowerer pops it.
        """
        from r65.compiler.mir.nodes import Move

        hoisted: Dict[Any, Any] = {}
        defined, array_indices = set(), []
        for node in walk_hir(body):
            if isinstance(node, HIRAssignment):
                if isinstance(node.target, HIRIdentifier):
                    defined.add(id(node.target.symbol))
            elif isinstance(node, HIRMultiAssignment):
                defined.update(id(t.symbol) for t in node.targets
                               if isinstance(t, HIRIdentifier))
            elif isinstance(node, HIRLetStmt):
                defined.add(id(node.symbol))
            elif isinstance(node, HIRMultiLetStmt):
                defined.update(id(sym) for sym in node.symbols)
            elif isinstance(node, HIRAddressOf):
                if isinstance(node.operand, HIRIdentifier):
                    defined.add(id(node.operand.symbol))
            elif isinstance(node, HIRArrayIndex):
                array_indices.append(node)

        for node in array_indices:
            try:
                _memloc, _key, terms = self.resolve_array_base_memloc(node.array)
            except MIRLoweringError:
                continue
            for index, stride in terms:
                if not isinstance(index, HIRIdentifier):
                    continue
                symbol = index.symbol
                key = (id(symbol), stride)
                if (key in hoisted or id(symbol) in defined
                        or self._hoisted_index_term(index, stride) is not None):
                    continue
                if symbol.kind == SymbolKind.LOCAL_VAR:
                    if self.current_function.alias_tracker.get_alias(symbol):
                        continue          # a `let x @ R`: nothing preserves it
                elif symbol.kind != SymbolKind.PARAMETER:
                    continue              # a static: a call may change it
                operand = self.lower_expression(index)
                if isinstance(operand, HardwareRegister):
                    held = self.ctx.alloc_vreg(index.expr_type, 'idx')
                    self.emit(Move(dest=held, source=operand, type_info=index.expr_type))
                    operand = held
                hoisted[key] = self.scale_index(operand, stride, index.expr_type)
        self._hoisted_index_terms.append(hoisted)

    def multi_index_memloc(self, base_memloc, terms, last_index, last_operand,
                           last_stride, extra_offset=0):
        """The MemoryLocation of `base + extra_offset + Σ index*stride`.

        Constant indices fold into the address; the variable ones are summed
        (sum_index_terms) and moved to X. With no variable index the location
        is absolute.
        """
        from r65.compiler.mir.nodes import HardwareRegister, MemoryLocation, Move
        from r65.compiler.hir.types import BasicTypeInfo

        const, total = self.sum_index_terms(terms, last_index, last_operand, last_stride)
        memloc = self._create_offset_memloc(base_memloc, const + extra_offset,
                                            base_memloc.symbol)
        if total is None:
            return memloc
        self.emit(Move(dest=HardwareRegister('X'), source=total,
                       type_info=BasicTypeInfo('u16')))
        self.x_index_cache_set(None)
        return MemoryLocation(
            storage_type=memloc.storage_type,
            address=memloc.address,
            symbol=memloc.symbol,
            index_register='X',
            offset=memloc.offset,
        )

    @staticmethod
    def plane_stride(symbol, element_size: int) -> int:
        """Bytes per element of the static *symbol*, honouring a padded plane.

        An `#[soa]` field plane pads a 3- or 7-byte element to 4 or 8
        (`soa_stride`), so its array type under-reports the stride. A plane
        reached through its parent gets the stride from `soa_planes`; one
        reached by name — an `#[soa]` payload's overlay plane, which the HIR
        rewrite indexes directly — has only the decl to say so.
        """
        definition = getattr(symbol, 'definition', None) if symbol else None
        stride = getattr(definition, 'soa_stride', None)
        return stride if stride is not None else element_size

    @staticmethod
    def _soa_plane_for(base_memloc, field_offset):
        """The `SoaPlane` owning *field_offset*.

        `peel_field_chain` hands the funnel a *summed* offset, so which
        top-level field a chain started at is recovered by range: struct layout
        is strictly packed with no zero-sized types, so exactly one plane's
        [offset, offset+size) contains it. A union nested *inside* the element
        still aliases correctly, which is why the union restriction is on the
        element type alone.
        """
        symbol = getattr(base_memloc, 'symbol', None)
        definition = getattr(symbol, 'definition', None) if symbol else None
        planes = getattr(definition, 'soa_planes', None)
        if not planes:
            return None
        for index, plane in enumerate(planes):
            nxt = planes[index + 1].aos_offset if index + 1 < len(planes) else None
            if plane.aos_offset <= field_offset and (nxt is None or field_offset < nxt):
                return plane
        return None

    def _soa_rebase(self, plane, base_memloc, field_offset, reuse_base_key):
        """Point a field access at its SoA plane instead of the parent array."""
        aos_offset, stride, child = plane.aos_offset, plane.stride, plane.child
        plane_memloc = self.get_memory_location(child.symbol)
        # The chain's residual offset within the plane's element, e.g. the `.y`
        # of `POOL[i].pos.y` where `pos` is the plane.
        residual = field_offset - aos_offset
        # X now holds index * stride(field), not index * sizeof(element), so the
        # reuse key must distinguish widths or a same-index access to a
        # different-width field would reuse the wrong scaled value.
        key = ('soa', reuse_base_key, stride) if reuse_base_key is not None else None
        return plane_memloc, residual, stride, key

    def require_addressable_aggregate(self, symbol, expr):
        """Assert *symbol* has a memory location rather than per-field vregs.

        Nested field access (`outer.inner.leaf`) computes one address off the
        base, which only works if the base actually lives in memory. Today it
        always does: `_is_stack_eligible_struct` refuses to decompose any
        aggregate with a non-scalar field, and a chain can only exist through
        such a field. This guard exists so that relaxing that rule fails loudly
        here instead of silently addressing storage nothing else maintains —
        `get_memory_location` knows about promoted locals but not decomposed
        ones, so it would hand back an unrelated location.
        """
        if id(symbol) in self._decomposed_structs:
            raise MIRLoweringError(
                f"nested field access on decomposed local '{getattr(symbol, 'name', '?')}' "
                f"— the base has no address",
                source_loc=getattr(expr, 'source_loc', None)
            )

    def peel_field_chain(self, field_expr):
        """Fold a chain of inline field accesses into ``(base, total_offset)``.

        ``outer.inner.leaf`` is a single address computation: every link but the
        innermost contributes only a constant offset within its parent, because
        struct and union fields are laid out inline (no indirection between
        levels). Summing those constants leaves one base that needs real code.

        Peeling stops at an ``auto_deref``'d field access — there the parent is
        reached through a pointer, so the returned base is that node and the
        caller loads through ``node.base``.

        Args:
            field_expr: The outermost HIRFieldAccess node.

        Returns:
            (base, total_offset) where base is the innermost non-inline node.
        """
        offset = 0
        node = field_expr
        while isinstance(node, HIRFieldAccess) and not node.auto_deref:
            if node.field_offset is None:
                raise MIRLoweringError(
                    f"Field offset not computed for field: {node.field_name}",
                    source_loc=getattr(node, 'source_loc', None)
                )
            offset += node.field_offset
            node = node.base
        return node, offset

    def try_pointer_deref_array_base(self, array_expr):
        """Recognise `(*p).arr` / `(*p)` array bases that need indirect codegen.

        Returns `(ptr_expr, base_field_offset)` when *array_expr* is a
        pointer-deref'd array — either an auto_deref ``HIRFieldAccess``
        (`self.arr[..]`) or an explicit ``HIRDereference`` (`(*p)[..]`),
        possibly wrapped in intermediate static (non-auto_deref)
        ``HIRFieldAccess`` nodes. The latter handles chained shapes like
        ``self.outer.inner_arr[..]`` where the pointer-deref happens at an
        inner field-access (`.outer`) and the array is reached through
        further inline struct fields (`.inner_arr`); each wrapper's
        ``field_offset`` is folded into the returned constant.

        Returns ``None`` for any other shape (caller falls through to the
        static-base path via :py:meth:`resolve_array_base_memloc`).
        """
        offset = 0
        node = array_expr
        # Peel non-deref'd HIRFieldAccess nodes, summing each field_offset
        # into the running constant. Stops at the first auto_deref'd
        # field-access (the pointer is its `.base`) or an explicit
        # HIRDereference, or bails out (returns None) for a purely static
        # chain so the static path keeps its existing handling.
        while isinstance(node, HIRFieldAccess) and not node.auto_deref:
            offset += node.field_offset if node.field_offset is not None else 0
            node = node.base
        if isinstance(node, HIRFieldAccess) and node.auto_deref:
            own = node.field_offset if node.field_offset is not None else 0
            return node.base, offset + own
        if isinstance(node, HIRDereference):
            return node.pointer, offset
        return None

    def emit_pointer_deref_array_access(
        self,
        *,
        ptr_expr,
        index_expr,
        element_size: int,
        const_offset: int,
        result_type,
        is_load: bool,
        source=None,
        dest=None,
    ):
        """Emit a Load/StoreIndirect against a pointer-deref'd array element.

        Used by both lowerers when :py:meth:`resolve_array_base_memloc`
        can't fold the base into a ``MemoryLocation`` because the array is
        reached through a pointer (auto-deref `self.arr[i]` or explicit
        `(*p)[i]`). Folds the compile-time `const_offset` (= outer field
        offset within the pointee struct + any inner ``.field`` offset
        within the element) and the runtime ``index * element_size`` into
        Y, then emits the indirect instruction against the pointer.

        Args:
            ptr_expr: HIR node yielding the pointer (e.g. ``self`` for
                auto-deref, the operand of an explicit ``*p``).
            index_expr: HIR expression for the array index.
            element_size: Byte size of one array element.
            const_offset: Compile-time byte offset to add to the base
                address — outer field + any inner field offset.
            result_type: ``type_info`` for the emitted Load/Store; usually
                the element type for plain element access, or the
                field type for ``arr[i].field`` access.
            is_load: ``True`` to emit ``LoadIndirect``, ``False`` to emit
                ``StoreIndirect``.
            source: For stores, the value being written.
            dest: For loads, the destination vreg (caller-allocated so it
                carries the right type metadata).

        Returns:
            ``dest`` (for loads) or ``source`` (for stores).
        """
        from r65.compiler.mir.nodes import (
            BinaryOp, HardwareRegister, Immediate, LoadIndirect, Move,
            StoreIndirect,
        )
        from r65.compiler.hir.types import BasicTypeInfo, PointerTypeInfo

        ptr_vreg = self.lower_expression(ptr_expr)
        ptr_type = ptr_expr.expr_type
        is_far = isinstance(ptr_type, PointerTypeInfo) and ptr_type.is_far

        index_operand = self.lower_expression(index_expr)

        # Fully constant index: fold the whole byte offset, no Y needed.
        if isinstance(index_operand, Immediate):
            total_offset = const_offset + index_operand.value * element_size
            if is_load:
                self.emit(LoadIndirect(
                    dest=dest, pointer=ptr_vreg, is_far=is_far,
                    type_info=result_type, offset=total_offset,
                ))
            else:
                self.emit(StoreIndirect(
                    source=source, pointer=ptr_vreg, is_far=is_far,
                    type_info=result_type, offset=total_offset,
                ))
            return dest if is_load else source

        # Variable index → Y = scaled_index + const_offset. memory_select
        # doesn't combine an explicit `offset` with an existing
        # index_register, so the constant lives inside the Y value.
        scaled = self.scale_index(index_operand, element_size, index_expr.expr_type)
        u16 = BasicTypeInfo('u16')
        if const_offset != 0:
            sum_vreg = self.ctx.alloc_vreg(u16, 'arr_off')
            self.emit(BinaryOp(
                dest=sum_vreg, left=scaled, right=Immediate(const_offset),
                op='+', type_info=u16,
            ))
            y_source = sum_vreg
        else:
            y_source = scaled
        self.emit(Move(dest=HardwareRegister('Y'), source=y_source, type_info=u16))
        if is_load:
            self.emit(LoadIndirect(
                dest=dest, pointer=ptr_vreg, is_far=is_far,
                index_register='Y', type_info=result_type, offset=0,
            ))
        else:
            self.emit(StoreIndirect(
                source=source, pointer=ptr_vreg, is_far=is_far,
                index_register='Y', type_info=result_type, offset=0,
            ))
        return dest if is_load else source

    def emit_indirect_field_access(
        self, ptr_expr, *, field_offset, result_type, is_load, dest=None, source=None,
    ):
        """Emit a Load/StoreIndirect for ``(ptr).field`` access.

        ptr_expr is the HIR node yielding the pointer — the auto-deref base for
        ``self.field`` or the operand of an explicit ``(*p).field``; is_far is
        derived from its type. Shared by the pointer-field and deref-field
        load/store lowerers (which differ only in load-vs-store and whether the
        pointer is reached directly or through an HIRDereference).
        """
        from r65.compiler.mir.nodes import LoadIndirect, StoreIndirect
        from r65.compiler.hir.types import PointerTypeInfo

        ptr_vreg = self.lower_expression(ptr_expr)
        ptr_type = ptr_expr.expr_type
        is_far = isinstance(ptr_type, PointerTypeInfo) and ptr_type.is_far
        if is_load:
            self.emit(LoadIndirect(
                dest=dest, pointer=ptr_vreg, is_far=is_far,
                type_info=result_type, offset=field_offset,
            ))
        else:
            self.emit(StoreIndirect(
                source=source, pointer=ptr_vreg, is_far=is_far,
                type_info=result_type, offset=field_offset,
            ))
        return dest if is_load else source

    def emit_static_array_field_access(
        self, array_index_expr, *, field_offset, result_type, is_load, dest=None, source=None,
    ):
        """Emit a Load/Store for ``arr[i].field`` where ``arr`` resolves to a static base.

        Handles the pointer-deref'd-base fast path, constant-index folding, and
        the variable-index X-reuse path. Shared by the array-field load and
        store lowerers, which differ only in load-vs-store and which type drives
        ``result_type`` (element field type for loads, assigned type for stores).
        """
        from r65.compiler.mir.nodes import (
            HardwareRegister, Immediate, Load, MemoryLocation, Move, Store,
        )
        from r65.compiler.hir.types import BasicTypeInfo

        # Pointer-deref'd array base (`self.sprites[i].field` via auto-deref,
        # or `(*p)[i].field`). Folds outer and inner field offsets into Y and
        # accesses indirect through the pointer.
        deref = self.try_pointer_deref_array_base(array_index_expr.array)
        if deref is not None:
            ptr_expr, base_field_offset = deref
            return self.emit_pointer_deref_array_access(
                ptr_expr=ptr_expr,
                index_expr=array_index_expr.index,
                element_size=self._get_type_size(array_index_expr.expr_type),
                const_offset=base_field_offset + field_offset,
                result_type=result_type,
                is_load=is_load, dest=dest, source=source,
            )

        # Static array base — a bare static array or an array field of a
        # statically-located struct (STRUCT.array_field[i].field).
        base_memloc, reuse_base_key, terms = self.resolve_array_base_memloc(
            array_index_expr.array)
        struct_type = array_index_expr.expr_type  # element (struct) type of the array
        struct_size = self._get_type_size(struct_type)
        if not terms:
            struct_size = self.plane_stride(getattr(base_memloc, 'symbol', None), struct_size)

        # #[soa]: redirect to the plane that owns this field, and scale by the
        # field rather than the element. Must run before anything reads
        # base_memloc — the parent static is allocated no bytes, and
        # get_memory_location does not fail on it, so a missed substitution
        # reaches the assembler as an undefined symbol the user never wrote.
        soa = self._soa_plane_for(base_memloc, field_offset)
        if soa is not None:
            base_memloc, field_offset, struct_size, reuse_base_key = self._soa_rebase(
                soa, base_memloc, field_offset, reuse_base_key)

        index_operand = self.lower_expression(array_index_expr.index)

        if terms:
            # `T[i][j].field`: more than one index, summed into X.
            field_memloc = self.multi_index_memloc(
                base_memloc, terms, array_index_expr.index, index_operand,
                struct_size, extra_offset=field_offset)
            if is_load:
                self.emit(Load(dest=dest, source=field_memloc, type_info=result_type))
            else:
                self.emit(Store(source=source, dest=field_memloc, type_info=result_type))
            return dest if is_load else source

        if isinstance(index_operand, Immediate):
            # Constant index: compute the byte offset at compile time.
            total_offset = (index_operand.value * struct_size) + field_offset
            field_memloc = self._create_offset_memloc(base_memloc, total_offset, base_memloc.symbol)
            if is_load:
                self.emit(Load(dest=dest, source=field_memloc, type_info=result_type))
            else:
                self.emit(Store(source=source, dest=field_memloc, type_info=result_type))
            return dest if is_load else source

        # Variable index. X holds index * struct_size (the field offset is folded
        # into the address constant), so it is identical for every field of the
        # same element — consecutive `arr[i].a; arr[i].b` can reuse X when the
        # reuse cache says X is still valid.
        reuse_key = self.x_index_reuse_key(reuse_base_key, array_index_expr.index)
        if base_memloc.address is not None:
            indexed_memloc = MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=base_memloc.address + field_offset,
                symbol=base_memloc.symbol,
                index_register='X',
            )
        else:
            indexed_memloc = MemoryLocation(
                storage_type=base_memloc.storage_type,
                address=None,
                symbol=base_memloc.symbol,
                index_register='X',
                offset=base_memloc.offset + field_offset,
            )

        if not self.x_index_cache_hit(reuse_key):
            # The scaled index is a byte offset (u16), NOT the struct type —
            # sizing the Move as the struct corrupts the index high byte.
            scaled_operand = self.scale_index(
                index_operand, struct_size, array_index_expr.index.expr_type)
            self.emit(Move(dest=HardwareRegister('X'), source=scaled_operand,
                           type_info=BasicTypeInfo('u16')))
            self.x_index_cache_set(reuse_key)

        if is_load:
            self.emit(Load(dest=dest, source=indexed_memloc, type_info=result_type))
        else:
            self.emit(Store(source=source, dest=indexed_memloc, type_info=result_type))
        return dest if is_load else source

    def _count_param_usages(self, hir_func: HIRFunctionDecl) -> Dict[int, int]:
        """
        Count how many times each register-bound parameter is read in the body.

        This is needed to determine if register-aliased parameters need to be
        saved to a vreg at function entry (if read at all: intermediate code
        may clobber the register first).

        Returns:
            Dict mapping symbol id to usage count
        """
        param_ids = {id(p.symbol) for p in hir_func.parameters
                     if isinstance(p.binding, RegisterBinding)}
        counts: Dict[int, int] = {}
        if not param_ids or hir_func.body is None:
            return counts

        # A bare identifier target is a pure write - `x = 5` does not read x.
        # Any compound target does (`arr[i] = v` reads i, `*p = v` reads p), so
        # only the bare one is skipped. The walk yields the assignment before
        # its target, so the skip is recorded in time.
        writes = set()
        for node in walk_hir(hir_func.body):
            if isinstance(node, HIRAssignment) and isinstance(node.target, HIRIdentifier):
                writes.add(id(node.target))
            elif (isinstance(node, HIRIdentifier) and id(node) not in writes
                    and id(node.symbol) in param_ids):
                counts[id(node.symbol)] = counts.get(id(node.symbol), 0) + 1
        return counts

    def _invalidate_memloc_cache(self):
        """Remove symbol_to_vreg entries for symbols with explicit memory
        locations (statics, promoted locals).  Local-variable vregs that
        ARE the variable storage are kept."""
        for sym_id in list(self.ctx.memloc_cached_symbols):
            if sym_id in self.symbol_to_vreg:
                del self.symbol_to_vreg[sym_id]

    def _block_has_terminator(self) -> bool:
        """
        Check if current block already ends with a terminator
        (`TERMINATOR_TYPES`).
        """
        if not self.current_block or not self.current_block.instructions:
            return False

        return isinstance(self.current_block.instructions[-1], TERMINATOR_TYPES)

    def jump_to(self, block: BasicBlock):
        """End the current block with a Jump to *block* and record the edge.

        The edge leaves `current_block` - wherever lowering has got to - not
        the block a construct began in: an arm that holds its own branches
        ends in a later block than it started in.
        """
        self.emit(Jump(target=block.block_id))
        self.cfg_builder.add_edge(self.current_block, block)

    def emit(self, instruction: MIRInstruction):
        """
        Emit an instruction to the current block.

        Args:
            instruction: MIR instruction to emit
        """
        # Propagate source location to instruction if not already set
        if instruction.source_loc is None and self._current_source_loc is not None:
            instruction.source_loc = self._current_source_loc

        if self._x_index_cache is not None and self._instr_clobbers_x(instruction):
            self._x_index_cache = None

        if self.current_block is not None:
            self.current_block.instructions.append(instruction)

    # Which nodes write X is `fixed_clobbers` / CLOBBERS_ALL in mir/nodes.py,
    # shared with the alias check and loop promotion. Missing one there means
    # a stale X is reused and an indexed access lands at the wrong address.
    @staticmethod
    def _instr_clobbers_x(instr) -> bool:
        """True if `instr` may change the X register.

        Indexed Load/Store (index_register='X') only READ X, so they do not
        count — that is exactly what the reuse cache relies on.
        """
        from r65.compiler.mir.nodes import (
            CLOBBERS_ALL, fixed_clobbers, HardwareRegister,
        )
        if isinstance(instr, CLOBBERS_ALL) or 'X' in fixed_clobbers(instr):
            return True
        dest = getattr(instr, 'dest', None)
        if isinstance(dest, HardwareRegister) and dest.name == 'X':
            return True
        return False

    def x_index_reuse_key(self, reuse_base_key, index_hir):
        """Reuse key for `array[index]` if it is safe to keep X live across
        consecutive accesses, else None.

        `reuse_base_key` is the array-base identity from
        resolve_array_base_memloc (an int for a top-level array, a tuple for a
        struct-field array, or None to disable reuse).

        Conservative: the index must be a bare identifier bound to a
        variable/parameter (pure, no side effects, stable until explicitly
        reassigned). Complex index expressions are never reused. The index
        symbol id stays at position [1] so x_index_cache_invalidate_symbol can
        clear the cache when the index variable is reassigned.
        """
        if reuse_base_key is None or index_hir is None:
            return None
        if not isinstance(index_hir, HIRIdentifier):
            return None
        sym = getattr(index_hir, 'symbol', None)
        if sym is None:
            return None
        return (reuse_base_key, id(sym))

    def x_index_cache_hit(self, key) -> bool:
        return key is not None and self._x_index_cache == key

    def x_index_cache_set(self, key):
        # key is None for non-reusable indices: X now holds an unkeyable
        # scaled index, so the cache must be cleared, not left stale.
        self._x_index_cache = key

    def x_index_cache_invalidate_symbol(self, symbol):
        """Drop the cache if `symbol` is the cached index variable."""
        if (self._x_index_cache is not None and symbol is not None
                and id(symbol) == self._x_index_cache[1]):
            self._x_index_cache = None

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
            # extern statics resolve to a label in an included .s file —
            # route through the label path regardless of mutability.
            if getattr(static_decl, 'is_extern', False):
                return MemoryLocation(
                    storage_type='rom',
                    address=None,
                    symbol=symbol,
                )
            # Check if definition has storage_attr (HIR node)
            if hasattr(static_decl, 'storage_attr') and static_decl.storage_attr:
                storage_attr = static_decl.storage_attr
                storage_type = storage_attr.storage_kind.value  # Get string value from enum
                return MemoryLocation(
                    storage_type=storage_type,
                    address=storage_attr.address,
                    symbol=symbol
                )
            # Immutable statics without storage attribute are ROM
            elif hasattr(static_decl, 'is_mutable') and not static_decl.is_mutable:
                return MemoryLocation(
                    storage_type='rom',
                    address=None,
                    symbol=symbol
                )

        # Default: unknown storage (will be allocated later)
        return MemoryLocation(
            storage_type='unknown',
            address=None,
            symbol=symbol
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

    def scale_index(self, index_operand, element_size: int, index_type):
        """`index * element_size` as a u16 byte offset; see compute_scaled_index."""
        return compute_scaled_index(index_operand, element_size, index_type,
                                    self.ctx, self.emit, self.call_lowerer)

    def _get_type_size(self, type_info) -> int:
        """Get size in bytes for a type."""
        return get_unified_type_size(type_info)

    def _try_eval_const_condition(self, expr: HIRExpression) -> Optional[bool]:
        """Try to evaluate a condition expression at compile time for dead code elimination."""
        from r65.compiler.hir.hir_const_eval import try_eval_const_bool
        symbol_table = self._hir_program.symbol_table if self._hir_program else None
        return try_eval_const_bool(expr, symbol_table)
