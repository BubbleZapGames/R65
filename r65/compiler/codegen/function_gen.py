"""
Function code generation: MIR functions → assembly.

Generates complete function bodies with headers, labels, and instructions.
"""

from typing import List, Set, Optional
from r65.compiler.mir.nodes import (
    MIRFunction, Return, Call, TraitDispatch, ArgumentMechanism,
    VirtualRegister, MemoryLocation,
    Load, Store, Move, BinaryOp, UnaryOp, Compare, BitTest, Rotate,
    Jump, JumpTable, LookupTable, CondBranch, TypeConvert, ToBool,
    LoadIndirect, StoreIndirect, StatusFlagRead, StatusFlagSet, StatusFlagTest,
    SaveRegister, RestoreRegister, SetMode, Push, Pull, ReturnFromInterrupt,
    MemoryFill, BlockCopy, InlineAsm,
)
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.register_alloc import ScratchRegisterPool, RegisterAllocator, PhysicalLocation, LocationKind
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.codegen.constants import DEFAULT_STACK_UPPER, M_FLAG, X_FLAG
from r65.compiler.codegen.opcodes import Opcode, PUSH_OPCODES, PULL_OPCODES
from r65.compiler.codegen.asm_nodes import Immediate, Address, StackOffset
from r65.compiler.codegen.abi import ABIInfo, StackFrameLayout


class FunctionCodeGenerator:
    """
    Generates complete assembly functions from MIR.

    Orchestrates function-level code generation including:
    - Function headers with metadata
    - Basic block ordering and labels
    - Instruction selection
    - Register allocation
    """

    def __init__(self,
                 emitter: AssemblyEmitter,
                 memory_allocator: MemoryAllocator):
        """
        Initialize function code generator.

        Args:
            emitter: Assembly emitter
            memory_allocator: Memory allocator for static variables
        """
        self.emitter = emitter
        self.mem_alloc = memory_allocator

    # ========================================================================
    # Emission Helpers
    # ========================================================================

    def _emit_instr(self, opcode: Opcode, operand=None, comment: str = None):
        """Emit an instruction using the node emitter."""
        self.emitter.emit_instr(opcode, operand, comment)

    # ========================================================================
    # Main Generation
    # ========================================================================

    def generate_function(self,
                          mir_func: MIRFunction,
                          scratch_pool: ScratchRegisterPool = None,
                          abi_model=None):
        """
        Generate complete assembly for MIR function.

        Args:
            mir_func: MIR function to generate
            scratch_pool: Optional scratch register pool (if None, no scratch registers available)
            abi_model: ABIModel instance (default: ABI_DEFAULT)
        """
        from r65.compiler.codegen.abi_model import ABI_DEFAULT
        if abi_model is None:
            abi_model = ABI_DEFAULT
        self.abi_model = abi_model
        # Setup register allocator for this function
        if scratch_pool is None:
            scratch_pool = ScratchRegisterPool()  # Empty pool if not provided

        # Reset scratch pool for this function (each function gets fresh allocation)
        scratch_pool.reset()

        # Mark scratch registers occupied by scratch-promoted parameters
        # These must be reserved before normal allocation so other vregs don't use them
        for param_idx, scratch_addr in mir_func.scratch_param_addrs.items():
            for scratch in scratch_pool.scratches:
                if scratch.address == scratch_addr:
                    scratch.is_free = False
                    break

        # Mark ALL globally-used scratch param addresses as occupied.
        # In FixedStack mode, a caller's scratch params persist while the callee
        # runs. If we allow local variables at those addresses, the callee's locals
        # overwrite the caller's params. Globally reserving all param addresses
        # prevents this collision.
        global_addrs = getattr(mir_func, '_global_scratch_param_addrs', None)
        if global_addrs:
            for scratch in scratch_pool.scratches:
                if scratch.address in global_addrs:
                    scratch.is_free = False

        # Analyze far self trait methods to determine if D=S prologue is needed
        # Must run before ABIInfo creation since it may set has_far_ptr_stack_params
        self._analyze_far_self_trait_method(mir_func)

        # Build ABI info and calculate prologue bytes BEFORE creating register allocator
        # This allows stack params to be allocated at their passed locations
        abi_info = ABIInfo.from_mir_function(mir_func)
        mir_func.abi_info = abi_info
        prologue_bytes = abi_info.prologue_stack_bytes

        # NOTE: We no longer need to add A register parameter save bytes.
        # Stack params are accessed directly at their passed locations without
        # copying, so there's no need to save/restore A during the prologue.

        # Create instruction-level liveness analyzer for precise liveness tracking
        from r65.compiler.mir.liveness import InstructionLivenessAnalyzer
        instr_liveness = InstructionLivenessAnalyzer(mir_func)

        outgoing_arg_bytes = mir_func.max_outgoing_arg_bytes

        # Build preliminary layout (local_frame_size filled in after allocation)
        layout = StackFrameLayout(
            abi=abi_info,
            local_frame_size=0,
            outgoing_arg_bytes=outgoing_arg_bytes,
        )

        reg_alloc = RegisterAllocator(
            scratch_pool=scratch_pool,
            mir_func=mir_func,
            prologue_stack_bytes=prologue_bytes,
            instr_liveness=instr_liveness,
            outgoing_arg_bytes=outgoing_arg_bytes,
            layout=layout,
        )

        # Pre-allocate scratch-promoted parameter vregs to their scratch locations
        # This must happen before allocate_all so the slot allocator sees them as pre-allocated
        for param_idx, scratch_addr in mir_func.scratch_param_addrs.items():
            vreg = mir_func.param_to_vreg.get(param_idx)
            if vreg:
                vreg_size = get_type_size(vreg.type_info)
                location = PhysicalLocation(
                    kind=LocationKind.SCRATCH,
                    scratch_addr=scratch_addr,
                    size=vreg_size
                )
                reg_alloc.allocations[vreg.id] = location

        # Pre-allocate loop-promoted vregs to hardware registers
        # These vregs were assigned X/Y hints by loop register promotion;
        # pre-allocating them prevents the slot allocator from creating
        # unnecessary frame slots.
        for hw_reg, vreg in list(mir_func.loop_promoted_hw_vregs.items()):
            # Check for conflicts: LoadIndirect/StoreIndirect clobber Y,
            # and explicit Move-to-HW defs can clobber X or Y.
            if reg_alloc._hint_conflicts_with_hw_defs(vreg, hw_reg):
                del mir_func.loop_promoted_hw_vregs[hw_reg]
                vreg.register_hint = None
                continue
            location = PhysicalLocation(
                kind=LocationKind.HARDWARE,
                hw_register=hw_reg,
                size=2  # X/Y are always 16-bit
            )
            reg_alloc.allocations[vreg.id] = location
            hw_alloc = reg_alloc.get_hw_alloc(hw_reg)
            hw_alloc.allocated_vreg = vreg
            hw_alloc.is_bound = True

        # Set register hints for hw-promoted parameter vregs (from FixedStack ABI promotion)
        # Do NOT pre-allocate to HARDWARE — let slot_allocator's coalescence analysis
        # determine whether the vreg can safely stay in the hw register. If A is clobbered
        # between the parameter Move and a later use, coalescence will correctly allocate
        # the vreg to stack/scratch instead.
        for param_idx, hw_reg_name in mir_func.hw_param_regs.items():
            vreg = mir_func.param_to_vreg.get(param_idx)
            if vreg:
                vreg.register_hint = hw_reg_name

        # Pre-allocate self pointer vreg for trait methods
        if mir_func.is_trait_method and mir_func.self_y_vreg:
            if mir_func.self_far_uses_d_equals_s:
                # D=S path: self pointer pushed to stack by PHB+PHY in prologue
                # Offset computed after frame_size known — mark as pre-allocated
                # with a placeholder so the slot allocator skips it
                location = PhysicalLocation(
                    kind=LocationKind.STACK,
                    stack_offset=0,  # Placeholder, computed post-allocation
                    size=3  # Far pointer: 3 bytes
                )
                reg_alloc.allocations[mir_func.self_y_vreg.id] = location
            else:
                # DBR:Y path (near self or far self leaf method): self stays in Y
                location = PhysicalLocation(
                    kind=LocationKind.HARDWARE,
                    hw_register='Y',
                    size=2  # Pointer address is always 16-bit
                )
                reg_alloc.allocations[mir_func.self_y_vreg.id] = location
                # Also track in hw_allocs so spill logic knows Y is occupied
                hw_alloc = reg_alloc.get_hw_alloc('Y')
                hw_alloc.allocated_vreg = mir_func.self_y_vreg
                hw_alloc.is_bound = True

        # Allocate all virtual registers in function
        self._allocate_function_registers(mir_func, reg_alloc)

        # Get frame size from allocator - all functions allocate frames for locals if needed
        # Note: The unified slot allocator already computes final param offsets
        # accounting for frame_size, so no post-hoc adjustment is needed.
        local_frame_size = reg_alloc.get_stack_frame_size()

        # Update layout with actual local frame size from allocation
        layout.local_frame_size = local_frame_size
        frame_size = layout.total_frame_size

        # Update register allocator with frame info
        reg_alloc.frame_size = frame_size
        reg_alloc.has_frame_allocation = layout.has_frame

        # Fix up far self D=S vreg offset now that frame_size is known
        # The far self pointer (3 bytes) was pushed by PHB+PHY before frame alloc.
        # From SP after all prologue: offset = prologue_stack_bytes - 2 + frame_size
        # (prologue_stack_bytes includes the 3 bytes for PHB+PHY, minus those 3 gives
        # the bytes above the self ptr; + frame_size for the frame; +1 for SP+1 base)
        if mir_func.self_far_uses_d_equals_s and mir_func.self_y_vreg:
            self_offset = prologue_bytes - 3 + frame_size + 1
            location = PhysicalLocation(
                kind=LocationKind.STACK,
                stack_offset=self_offset,
                size=3
            )
            reg_alloc.allocations[mir_func.self_y_vreg.id] = location

        # Store stack usage on MIR function for stack depth analysis
        mir_func.codegen_frame_size = frame_size
        mir_func.codegen_prologue_bytes = prologue_bytes

        # Store frame-aware liveness metric for stack depth analysis
        if reg_alloc.slot_allocation:
            max_live = reg_alloc.slot_allocation.max_live_frame_bytes_at_calls
            mir_func.codegen_max_live_frame_bytes_at_calls = max_live
            # Determine if partial frame dealloc is possible:
            # - Frame has reclaimable bytes (max_live < frame_size)
            # - No stack parameters (would shift param offsets)
            has_stack_params = bool(reg_alloc.slot_allocation.param_offsets)
            if frame_size > 0 and max_live < frame_size and not has_stack_params:
                mir_func.codegen_frame_dead_before_calls = True

        # Count Return instructions to decide if we need a shared epilogue.
        # When multiple returns exist, they all branch to a shared epilogue
        # label instead of emitting duplicate epilogue code inline.
        return_count = sum(
            1 for block in mir_func.blocks.values()
            for instr in block.instructions
            if isinstance(instr, Return)
        )
        shared_epilogue_label = (
            f"{mir_func.name}__epilogue" if return_count > 1 else None
        )

        # Create instruction selector with current function context
        instr_selector = InstructionSelector(self.emitter, reg_alloc, self.mem_alloc, mir_func, func_gen=self, abi_model=abi_model)

        # Initialize region-based spilling for this function
        instr_selector.call_selector.initialize_regions_for_function()

        # Set shared epilogue label if function has multiple returns
        instr_selector.control_flow_selector.shared_epilogue_label = shared_epilogue_label

        # Emit function header comment
        self.emit_function_header(mir_func)

        # Emit function label
        self.emitter.emit_label(mir_func.name)

        # Emit mode directives for WLA-DX assembler
        self._emit_mode_directives(mir_func)

        # Emit prologue (if needed)
        self.emit_prologue(mir_func, reg_alloc, frame_size)

        # Generate basic blocks
        block_order = self._compute_block_order(mir_func)

        # Track codegen exit modes per block to detect mode mismatches
        # between runtime predecessors (which may differ from emission order)
        codegen_exit_modes = {}

        for block_id in block_order:
            block = mir_func.blocks[block_id]

            # Initialize region tracking for this block (for optimized spilling)
            instr_selector.call_selector.initialize_regions_for_block(block_id)

            # Emit block label (except entry block which uses function label)
            if block_id != mir_func.entry_block_id:
                self.emitter.emit_label(f"{mir_func.name}__L{block_id}")

            # Determine if we need to force a mode switch at block entry.
            # This is needed when:
            # 1. A predecessor hasn't been emitted yet (back-edge from loop)
            #    - We can't know what mode it will exit in, so force the switch
            # 2. An emitted predecessor exits in a different mode than expected
            #    - The codegen may have switched modes for u16 ops without
            #      switching back (by design, to avoid redundant SEP/REP)
            force_mode = False
            if block.entry_mode is not None:
                from r65.compiler.typeck.processor_mode import ModeState
                expected_is_m16 = (block.entry_mode.m_mode == ModeState.M16)
                for pred_id in block.predecessors:
                    if pred_id not in codegen_exit_modes:
                        # Back-edge: predecessor not yet emitted.
                        # Check if it's safe to skip: if the predecessor
                        # enters in the same mode and doesn't contain
                        # mode-changing operations, it will exit in the
                        # same mode as our expected mode.
                        if self._back_edge_may_change_mode(
                            mir_func, pred_id, expected_is_m16
                        ):
                            force_mode = True
                            break
                    else:
                        pred_is_m16 = (codegen_exit_modes[pred_id] == 16)
                        if pred_is_m16 != expected_is_m16:
                            # Emitted predecessor exits in wrong mode
                            force_mode = True
                            break

            # Emit mode switch at block entry if needed
            self._emit_block_entry_mode_switch(block, instr_selector,
                                               force=force_mode)

            # Emit instructions in block
            for instr in block.instructions:
                instr_selector.select_instruction(instr)

            # Record this block's codegen exit mode
            codegen_exit_modes[block_id] = self.emitter.get_accu_mode()

        # Emit shared epilogue if function has multiple returns.
        # Each Return instruction branched here instead of emitting inline.
        if shared_epilogue_label:
            self.emitter.emit_label(shared_epilogue_label)
            self.emit_epilogue(mir_func, reg_alloc)
            instr_selector.control_flow_selector._emit_return_instruction()

        # Blank line after function
        self.emitter.emit_blank_line()

    # ========================================================================
    # Function Header
    # ========================================================================

    def emit_function_header(self, mir_func: MIRFunction):
        """
        Emit function header comment with metadata.

        Args:
            mir_func: MIR function
        """
        # Main divider
        divider = "-" * 76
        self.emitter.emit_comment(divider)

        # Function name
        self.emitter.emit_comment(f"{mir_func.name}")

        # Source location (if available)
        if mir_func.source_loc:
            loc = mir_func.source_loc
            self.emitter.emit_comment(f"Source: {loc.file_path}:{loc.line}")
            # Show include chain if this is from an included file
            if loc.included_from:
                parent = loc.included_from
                while parent:
                    self.emitter.emit_comment(f"  included from {parent.file_path}:{parent.line}")
                    parent = parent.included_from

        self.emitter.emit_comment("")

        # Parameters
        if mir_func.parameters:
            self.emitter.emit_comment("Parameters:")
            for param in mir_func.parameters:
                param_desc = f"  {param.name}: {param.param_type}"
                self.emitter.emit_comment(param_desc)
            self.emitter.emit_comment("")

        # Return type
        if mir_func.return_type:
            self.emitter.emit_comment(f"Returns: {mir_func.return_type}")
            self.emitter.emit_comment("")

        # Attributes
        if mir_func.mode_attr:
            mode_str = f"Mode: {mir_func.mode_attr}"
            self.emitter.emit_comment(mode_str)

        if mir_func.preserves_attr:
            preserves = ", ".join(mir_func.preserves_attr.registers)
            self.emitter.emit_comment(f"Preserves: {preserves}")

        if mir_func.is_entry:
            self.emitter.emit_comment("Entry: true")

        if mir_func.is_far:
            self.emitter.emit_comment("Far: true (JSL/RTL)")

        # Closing divider
        self.emitter.emit_comment(divider)

    # ========================================================================
    # Basic Block Ordering
    # ========================================================================

    def _compute_block_order(self, mir_func: MIRFunction) -> List[int]:
        """
        Compute optimal ordering of basic blocks.

        Uses DFS traversal from entry block with a layout heuristic:
        blocks containing Return instructions are visited last among
        successors. This places loop bodies linearly before exit paths,
        reducing mode-switch overhead at block boundaries.

        Args:
            mir_func: MIR function

        Returns:
            List of block IDs in emission order
        """
        from r65.compiler.mir.nodes import Return as MIRReturn

        # Pre-compute which blocks contain Return instructions
        return_blocks: Set[int] = set()
        for block_id, block in mir_func.blocks.items():
            for instr in block.instructions:
                if isinstance(instr, MIRReturn):
                    return_blocks.add(block_id)
                    break

        visited: Set[int] = set()
        order: List[int] = []

        def visit(block_id: int):
            if block_id in visited:
                return

            visited.add(block_id)
            order.append(block_id)

            # Visit successors, placing return blocks last so loop bodies
            # are laid out linearly (hot path as fallthrough)
            block = mir_func.blocks.get(block_id)
            if block:
                successors = sorted(
                    block.successors,
                    key=lambda sid: sid in return_blocks,
                )
                for successor_id in successors:
                    visit(successor_id)

        # Start from entry block
        visit(mir_func.entry_block_id)

        # Visit any unreachable blocks (shouldn't happen, but be safe)
        for block_id in mir_func.blocks.keys():
            if block_id not in visited:
                visit(block_id)

        return order

    def _back_edge_may_change_mode(
        self, mir_func: MIRFunction, pred_id: int, expected_is_m16: bool
    ) -> bool:
        """
        Check if a back-edge predecessor might exit in a different mode
        than expected. Used to avoid redundant SEP/REP at loop headers.

        Returns True (conservative, force mode switch) if:
        - The predecessor block's entry mode differs from expected
        - The predecessor contains instructions that may change accumulator mode
        - The predecessor can't be analyzed (missing data)

        Returns False (safe to skip mode switch) if the predecessor enters
        in the same mode as expected and contains no mode-changing instructions.
        """
        from r65.compiler.typeck.processor_mode import ModeState
        from r65.compiler.mir.nodes import (
            BinaryOp, UnaryOp, Compare, Return, TypeConvert, Call,
            TraitDispatch, LoadIndirect, Load, Store,
        )
        from r65.compiler.codegen.type_utils import get_type_size

        pred_block = mir_func.blocks.get(pred_id)
        if not pred_block:
            return True  # Can't analyze

        # Check if predecessor's entry mode matches our expected mode
        if pred_block.entry_mode is None:
            return True
        pred_is_m16 = (pred_block.entry_mode.m_mode == ModeState.M16)
        if pred_is_m16 != expected_is_m16:
            return True  # Different entry mode

        # Check if the block contains instructions that change accumulator mode.
        # Operations on u16 values switch to m16; returns may switch for
        # return value setup. Calls can leave mode in any state.
        for instr in pred_block.instructions:
            if isinstance(instr, (BinaryOp, UnaryOp)):
                if instr.type_info and get_type_size(instr.type_info) >= 2:
                    return True
            elif isinstance(instr, Compare):
                if instr.type_info and get_type_size(instr.type_info) >= 2:
                    return True
            elif isinstance(instr, Return):
                return True  # May switch mode for return value
            elif isinstance(instr, TypeConvert):
                return True  # May involve mode changes
            elif isinstance(instr, (Call, TraitDispatch)):
                return True  # Callee may leave mode in any state
            elif isinstance(instr, (LoadIndirect, Load, Store)):
                # Check if the operation involves u16 types
                if hasattr(instr, 'type_info') and instr.type_info:
                    if get_type_size(instr.type_info) >= 2:
                        return True

        return False  # Block is mode-preserving

    def _emit_block_entry_mode_switch(self, block, instr_selector, force=False):
        """
        Emit mode switch at block entry if the block's expected mode differs
        from the current tracked mode.

        This handles cases like loop back-edges where the predecessor block
        may have switched modes (e.g., for u16 operations) but the loop header
        expects a different mode.

        When force=True, always emit the mode switch even if the emitter's
        tracked mode appears correct. This is needed for loop headers and
        blocks with predecessors that exit in different modes, because the
        emitter tracks emission-order mode (not runtime execution order).

        Args:
            block: MIR basic block
            instr_selector: InstructionSelector with mode tracking
            force: If True, always emit mode switch regardless of tracked mode
        """
        from r65.compiler.typeck.processor_mode import ModeState

        # Get block's expected entry mode
        if not hasattr(block, 'entry_mode') or block.entry_mode is None:
            return

        block_entry_mode = block.entry_mode
        if hasattr(block_entry_mode, 'm_mode'):
            expected_m_mode = block_entry_mode.m_mode
        else:
            return

        # Get current tracked mode from emitter
        current_mode_bits = self.emitter.get_accu_mode()
        current_is_m16 = (current_mode_bits == 16)
        expected_is_m16 = (expected_m_mode == ModeState.M16)

        # Emit mode switch if needed (or forced for back-edge targets)
        if force or current_is_m16 != expected_is_m16:
            if expected_is_m16:
                self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG),
                               "Restore m16 mode for block")
                self.emitter.emit_accu_mode(16)
            else:
                self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG),
                               "Restore m8 mode for block")
                self.emitter.emit_accu_mode(8)

    # ========================================================================
    # Register Allocation
    # ========================================================================

    def _allocate_function_registers(self,
                                    mir_func: MIRFunction,
                                    reg_alloc: RegisterAllocator):
        """
        Allocate all virtual registers in function.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
        """
        # Collect all virtual registers used in function
        vregs = set()

        for block in mir_func.blocks.values():
            for instr in block.instructions:
                # Extract virtual registers from instruction
                # This is simplified - actual implementation would use visitor pattern
                vregs.update(self._extract_vregs_from_instruction(instr))

        # Allocate all at once
        reg_alloc.allocate_all(list(vregs))

    # Static field-name table per instruction type for vreg extraction.
    # Maps type → tuple of field names that may hold VirtualRegister values.
    # None means special handling (variable-length lists).
    _VREG_FIELDS = {
        Load: ('dest',),
        Store: ('source',),
        Move: ('dest', 'source'),
        BinaryOp: ('dest', 'left', 'right'),
        UnaryOp: ('dest', 'operand'),
        Compare: ('left', 'right'),
        BitTest: ('value',),
        Rotate: ('dest', 'source'),
        TypeConvert: ('dest', 'source'),
        ToBool: ('dest', 'source'),
        LoadIndirect: ('dest', 'pointer'),
        StoreIndirect: ('source', 'pointer'),
        LookupTable: ('dest', 'scrutinee'),
        JumpTable: ('scrutinee',),
        CondBranch: ('condition',),
        SaveRegister: ('save_location',),
        RestoreRegister: ('save_location',),
        StatusFlagRead: ('dest',),
        # Types with no vreg fields
        Jump: (),
        SetMode: (),
        Push: (),
        Pull: (),
        ReturnFromInterrupt: (),
        MemoryFill: (),
        BlockCopy: (),
        InlineAsm: (),
        StatusFlagSet: (),
        StatusFlagTest: (),
        # Special cases handled below
        Return: None,
        Call: None,
        TraitDispatch: None,
    }

    def _extract_vregs_from_instruction(self, instr) -> Set:
        """
        Extract virtual registers from instruction.

        Uses static field-name table for O(1) type lookup instead of dir() reflection.

        Args:
            instr: MIR instruction

        Returns:
            Set of VirtualRegister objects
        """
        vregs = set()
        fields = self._VREG_FIELDS.get(type(instr))

        if fields is None:
            # Special cases: Return, Call, TraitDispatch
            instr_type = type(instr)
            if instr_type is Return:
                for v in instr.values:
                    if isinstance(v, VirtualRegister):
                        vregs.add(v)
            elif instr_type is Call:
                if isinstance(instr.function, VirtualRegister):
                    vregs.add(instr.function)
                for arg in instr.args:
                    if isinstance(arg.value, VirtualRegister):
                        vregs.add(arg.value)
                for ret in instr.returns:
                    if isinstance(ret, VirtualRegister):
                        vregs.add(ret)
            elif instr_type is TraitDispatch:
                if isinstance(instr.self_ptr, VirtualRegister):
                    vregs.add(instr.self_ptr)
                for arg in instr.args:
                    if isinstance(arg.value, VirtualRegister):
                        vregs.add(arg.value)
                for ret in instr.returns:
                    if isinstance(ret, VirtualRegister):
                        vregs.add(ret)
        elif fields:
            for field_name in fields:
                val = getattr(instr, field_name, None)
                if isinstance(val, VirtualRegister):
                    vregs.add(val)

        return vregs

    # ========================================================================
    # Scratch Pool Creation
    # ========================================================================

    def _create_scratch_pool(self, mir_program) -> ScratchRegisterPool:
        """
        Create scratch register pool from user-defined register variables.

        Scans static variables for those marked with register=true attribute.
        Memory management is the programmer's responsibility - the compiler
        only uses scratch registers explicitly defined by the programmer.

        Args:
            mir_program: MIR program containing static variable declarations

        Returns:
            ScratchRegisterPool populated with user-defined registers
        """
        pool = ScratchRegisterPool()

        # Scan static variables for register-marked variables
        for static_var in mir_program.statics:
            if hasattr(static_var, 'storage_attr') and static_var.storage_attr:
                storage_attr = static_var.storage_attr

                # Only use variables explicitly marked as registers
                if storage_attr.is_register:
                    # Get the variable's address from memory allocator
                    alloc = self.mem_alloc.get_allocation(static_var.symbol)
                    if alloc:
                        # Determine size from type
                        size = self._get_variable_size(static_var.var_type)

                        # Add to scratch pool
                        pool.add_scratch(
                            address=alloc.address,
                            size=size,
                            name=static_var.name
                        )

        return pool

    def _get_variable_size(self, type_info) -> int:
        """Get size of variable in bytes from type info."""
        return get_type_size(type_info)

    # ========================================================================
    # Prologue/Epilogue
    # ========================================================================

    def _emit_mode_directives(self, mir_func: MIRFunction):
        """
        Emit WLA-DX mode directives to inform the assembler of the expected processor mode.

        These directives (.ACCU and .INDEX) tell WLA-DX what size the accumulator and
        index registers are, so it can assemble instructions correctly. They don't
        emit any code - they're just for the assembler.

        In the simplified mode system:
        - X/Y are always 16-bit (x16)
        - A is m8 by default, m16 if function has u16 @ A parameter

        Args:
            mir_func: MIR function
        """
        from r65.compiler.typeck.processor_mode import ModeState

        # Emit accumulator mode directive based on inferred entry mode
        if mir_func.entry_m_mode == ModeState.M16:
            self.emitter.emit_directive("    .ACCU 16")
        else:
            # Default: m8 (8-bit accumulator)
            self.emitter.emit_directive("    .ACCU 8")

        # Emit index mode directive - always 16-bit in R65
        self.emitter.emit_directive("    .INDEX 16")

    def emit_prologue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator, frame_size: int = 0):
        """
        Emit function prologue.

        Prologue may include (in order):
        1. Stack pointer initialization (entry functions with custom stack)
        2. Entry function setup (SEI/CLC/XCE/REP - switch to native mode)
        3. Stack frame allocation (if needed)
        4. DBR management (databank=inline)
        5. Mode transitions
        6. Register preservation

        Stack frame allocation comes BEFORE register saves so that saved registers
        and stack parameters are at consistent offsets regardless of local variable count.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
            frame_size: Number of bytes to allocate for local stack frame
        """
        # Initialize stack pointer for entry functions with custom stack region
        if mir_func.is_entry and self.mem_alloc.stack_upper is not None:
            if self.mem_alloc.stack_upper != DEFAULT_STACK_UPPER:
                stack_addr = self.mem_alloc.stack_upper
                self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "16-bit A for stack setup")
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(stack_addr), "Stack top")
                self._emit_instr(Opcode.TCS, comment="Set stack pointer")
                self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "Restore 8-bit A")

        # Entry function setup: switch from emulation mode to native mode
        # This must happen BEFORE frame allocation since PHA behavior differs between modes
        if mir_func.is_entry:
            self._emit_entry_setup(mir_func)

        # For interrupt handlers, emit register saves BEFORE frame allocation
        # This is critical: frame allocation uses stack-relative addressing that
        # would corrupt saved registers if done before the pushes.
        # Order: register saves -> scratch saves -> frame allocation -> mode setup -> body
        if mir_func.interrupt_attr:
            self._emit_interrupt_register_saves()
            self._emit_interrupt_scratch_saves(reg_alloc.scratch_pool)

        # For far self D=S path: push DBR (bank) and Y (addr) to stack BEFORE frame alloc
        # This places the 3-byte far self pointer on the stack where D=S can access it
        # Stack layout after PHB+PHY: [PHY_lo, PHY_hi, PHB_bank] (3 bytes)
        if mir_func.self_far_uses_d_equals_s:
            self._emit_instr(Opcode.PHB, comment="Push self bank byte (DBR = object's bank)")
            self._emit_instr(Opcode.PHY, comment="Push self address (Y = object addr)")

        # Detect which hardware registers hold parameters.
        from r65.compiler.hir import RegisterBinding
        a_has_param = False
        x_has_param = False
        y_has_param = False
        for param in mir_func.parameters:
            if isinstance(param.binding, RegisterBinding):
                reg = param.binding.register_name
                if reg == 'A':
                    a_has_param = True
                elif reg == 'X':
                    x_has_param = True
                elif reg == 'Y':
                    y_has_param = True

        # Determine if/how to save A across prologue code that clobbers it.
        # Possible clobber sources: frame alloc (TSC/SBC/TCS for large frames),
        # DBR inline management (LDA #bank), far pointer D setup (TSC/TCD).
        # a_save_method: 'Y' = TAY/TYA, 'X' = TAX/TXA, 'push' = force push-based
        # frame alloc (no save needed), None = no save needed.
        a_save_method = None
        force_direct = mir_func.interrupt_attr is not None

        if a_has_param:
            frame_clobbers_a = False
            if frame_size > 0:
                frame_clobbers_a = (
                    frame_size > self.abi_model.frame_alloc_clobbers_a_threshold
                    or force_direct
                )
            dbr_clobbers_a = False
            if mir_func.is_far and mir_func.mode_attr and mir_func.bank_attr:
                from r65.compiler.hir.attributes import DataBankMode
                if mir_func.mode_attr.databank == DataBankMode.INLINE:
                    dbr_clobbers_a = True
            fptr_clobbers_a = getattr(mir_func, 'has_far_ptr_stack_params', False)

            if frame_clobbers_a or dbr_clobbers_a or fptr_clobbers_a:
                if not y_has_param:
                    a_save_method = 'Y'
                elif not x_has_param:
                    a_save_method = 'X'
                else:
                    # All three registers occupied — force push-based frame
                    # allocation which doesn't clobber any register.
                    a_save_method = 'push'

        # Save A param before any prologue code that clobbers A.
        # This must happen before frame allocation AND D=S setup (both use A).
        if a_save_method == 'Y':
            self._emit_instr(Opcode.TAY, comment="Save A param before prologue")
        elif a_save_method == 'X':
            self._emit_instr(Opcode.TAX, comment="Save A param before prologue (Y has param)")

        # Allocate stack frame for functions with locals.
        if frame_size > 0:
            if a_save_method == 'push':
                # Push-based allocation (PHX/PHY) doesn't clobber A, X, or Y.
                self.abi_model._emit_register_push_alloc(self._emit_instr, frame_size)
            else:
                self._emit_frame_allocation(frame_size, force_direct_stack=force_direct)

        # Handle DBR management for far functions with databank=inline
        if mir_func.is_far and mir_func.mode_attr and mir_func.bank_attr:
            from r65.compiler.hir.attributes import DataBankMode

            if mir_func.mode_attr.databank == DataBankMode.INLINE:
                # Save current DBR and set to function's bank
                # Sequence: PHB, LDA #bank, PHA, PLB
                self._emit_instr(Opcode.PHB, comment="Save current data bank")
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(mir_func.bank_attr.bank_number),
                                "Load function's bank number")
                self._emit_instr(Opcode.PHA, comment="Push bank number")
                self._emit_instr(Opcode.PLB, comment="Set data bank register")

        # Set up processor mode based on inferred entry mode
        # In R65, X/Y are always x16, A mode is based on parameter types
        # Mode transitions are now automatic - compiler inserts REP/SEP as needed
        from r65.compiler.typeck.processor_mode import ModeState

        # For functions with m16 entry (u16 @ A parameter), set up 16-bit A mode
        # The default execution mode is m8, x16 (set by the bootstrap/interrupt context)
        # If this function requires m16, we need to switch to it
        if mir_func.entry_m_mode == ModeState.M16:
            # REP #$20 to set 16-bit accumulator mode
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "Set m16 mode for u16 @ A parameter")
        # If m8 (default), no prologue mode switch needed - we assume caller is in m8 mode
        # For now, we trust that the caller will set up the correct mode

        # Emit register saves for #[preserves(...)]
        # Registers are pushed in order: STATUS, A, X, Y, D, DBR
        # (Popped in reverse order in epilogue/select_return)
        if mir_func.preserves_attr:
            preserved_regs = mir_func.preserves_attr.registers

            # Push in defined order (reverse order for pop)
            push_order = ['STATUS', 'A', 'X', 'Y', 'D', 'DBR']
            for reg in push_order:
                if reg in preserved_regs:
                    push_opcode = PUSH_OPCODES.get(reg)
                    if push_opcode:
                        self._emit_instr(push_opcode, comment=f"Preserve {reg}")

        # Set up far pointer access for stack parameters
        # Must come AFTER all other prologue pushes so offsets are final
        if mir_func.has_far_ptr_stack_params:
            from r65.compiler.mir.nodes import FarPtrStrategy
            if mir_func.far_ptr_strategy == FarPtrStrategy.SET_DBR:
                # SET_DBR: save DBR, load bank byte from far pointer param, set DBR
                # This preserves DP, keeping scratch registers and zeropage available
                bank_offset = self._get_far_ptr_bank_stack_offset(mir_func, frame_size)
                self._emit_instr(Opcode.PHB, comment="Save Data Bank Register")
                self._emit_instr(Opcode.LDA_STACK, StackOffset(bank_offset),
                                "Load far pointer bank byte from stack")
                self._emit_instr(Opcode.PHA, comment="Push bank byte")
                self._emit_instr(Opcode.PLB, comment="Set DBR to far pointer bank")
            else:
                # D_EQUALS_S: enables [dp],Y addressing for 24-bit pointers
                self._emit_instr(Opcode.PHD, comment="Save Direct Page register")
                self._emit_instr(Opcode.TSC, comment="Transfer Stack to A")
                self._emit_instr(Opcode.TCD, comment="Transfer A to Direct Page (D = S)")

        # Restore A param AFTER all prologue code that clobbers A.
        # The current mode (set by mode setup above) ensures the transfer
        # operates at the correct width (m8 for u8 @ A, m16 for u16 @ A).
        if a_save_method == 'Y':
            self._emit_instr(Opcode.TYA, comment="Restore A param after prologue")
        elif a_save_method == 'X':
            self._emit_instr(Opcode.TXA, comment="Restore A param after prologue")
        # 'push' and None: no restore needed (A was never clobbered)

    def _get_far_ptr_bank_stack_offset(self, mir_func: MIRFunction, frame_size: int) -> int:
        """Compute stack offset to the far pointer's bank byte for SET_DBR prologue.

        At the point SET_DBR code runs (after frame alloc + preserves, before PHB),
        the offset from SP to the bank byte = param_offset + 2 + (prologue_bytes - 1) + frame_size.
        The -1 accounts for PHB not having been pushed yet.
        """
        # Get the single far pointer param index
        far_idx = next(iter(mir_func.far_ptr_param_indices))
        base_offset = mir_func.stack_param_offsets[far_idx]

        # Bank byte is at +2 from param start (3-byte pointer: lo, hi, bank)
        bank_byte_base = base_offset + 2

        # Adjust for everything pushed before this point:
        # prologue_stack_bytes includes 1 for PHB, but PHB hasn't happened yet
        abi = ABIInfo.from_mir_function(mir_func)
        pre_phb_bytes = abi.prologue_stack_bytes - 1  # Everything except PHB itself
        return bank_byte_base + pre_phb_bytes + frame_size

    def _emit_entry_setup(self, mir_func: MIRFunction):
        """
        Emit entry function setup code.

        SNES boots in 6502 emulation mode - must switch to 65816 native mode first.
        This must happen BEFORE frame allocation because PHA behavior differs
        between emulation mode (8-bit push) and native mode (8-bit or 16-bit push).

        After XCE, CPU is in native mode with M=1, X=1 (8-bit mode).
        We then set up the R65 default mode: m8 (8-bit A), x16 (16-bit X/Y).

        Args:
            mir_func: MIR function
        """
        # Emit the mode switch sequence using raw assembly
        # This is cleaner than emitting individual instructions since these
        # are special bootstrap instructions that don't need comments
        self.emitter.emit_raw("SEI")   # Disable interrupts
        self.emitter.emit_raw("CLC")   # Clear carry for XCE
        self.emitter.emit_raw("XCE")   # Enter native mode

        # After XCE, set up x16 mode (always) and m16 mode (if requested)
        # X/Y are ALWAYS 16-bit in R65, so we always emit REP #$10
        from r65.compiler.typeck.processor_mode import ModeState
        from r65.compiler.codegen.constants import M_FLAG, X_FLAG

        rep_mask = X_FLAG  # Clear X flag for 16-bit index mode

        # If function has m16 entry mode, also clear M flag
        if mir_func.entry_m_mode == ModeState.M16:
            rep_mask |= M_FLAG  # Clear M flag for 16-bit accumulator

        # Emit REP to set up x16 mode (always) and m16 mode (if requested)
        self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(rep_mask))

    def _emit_interrupt_register_saves(self):
        """
        Emit register saves for interrupt handlers.

        This must be called BEFORE frame allocation to ensure correct stack layout.
        After this, the stack has the saved registers on top, and stack-relative
        addressing for local variables will work correctly.

        CRITICAL: The 65816 has a hidden high byte of A (the "B accumulator") that
        is NOT preserved by PHA in m8 mode (PHA only pushes the low byte). If the
        handler body switches to m16 and performs 16-bit operations, the high byte
        gets clobbered. When the interrupted code later uses REP #$20, the corrupted
        high byte causes wrong results.

        Fix: Push PHP FIRST (saves mode), then REP #$20 (force m16), then PHA
        (pushes full 16-bit A). This always saves both bytes of A regardless of
        what mode the CPU was in when the interrupt fired.

        Order of pushes: PHP, [REP #$20], PHA(16-bit), PHX, PHY, PHD, PHB
        (Corresponding pops in epilogue: PLB, PLD, PLY, PLX, [REP #$20], PLA(16-bit), PLP)
        """
        self._emit_instr(Opcode.PHP, comment="Save processor status (first - before mode change)")
        self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "Force 16-bit A to save full accumulator")
        self._emit_instr(Opcode.PHA, comment="Save A (full 16-bit, includes hidden high byte)")
        self._emit_instr(Opcode.PHX, comment="Save X")
        self._emit_instr(Opcode.PHY, comment="Save Y")
        self._emit_instr(Opcode.PHD, comment="Save Direct Page")
        self._emit_instr(Opcode.PHB, comment="Save Data Bank")
        self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "Restore 8-bit A for handler body")

    def _emit_interrupt_scratch_saves(self, scratch_pool: ScratchRegisterPool):
        """
        Save scratch register WRAM contents to the stack for interrupt handlers.

        When an NMI fires mid-execution, the interrupted code may be using
        scratch registers for local variables. The interrupt handler also uses
        these scratch registers for its own temporaries. Without saving/restoring,
        the interrupted code's values get corrupted.

        Uses absolute addressing (not DP) because D's value is unknown when an
        interrupt fires — the interrupted code may have set D = S for far pointer
        stack params, making DP addressing read from the wrong location.

        Called after _emit_interrupt_register_saves() which leaves us in m8 mode.

        Args:
            scratch_pool: Pool of scratch registers to save
        """
        if not scratch_pool or not scratch_pool.scratches:
            return

        for scratch in scratch_pool.scratches:
            if scratch.size == 1:
                # 1-byte scratch: LDA abs / PHA (already in m8)
                self._emit_instr(Opcode.LDA_ABSOLUTE, Address(scratch.address),
                                f"Save scratch {scratch.name}")
                self._emit_instr(Opcode.PHA)
            elif scratch.size == 2:
                # 2-byte scratch: REP #$20 / LDA abs / PHA / SEP #$20
                self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "16-bit A for 2-byte scratch save")
                self._emit_instr(Opcode.LDA_ABSOLUTE, Address(scratch.address),
                                f"Save scratch {scratch.name}")
                self._emit_instr(Opcode.PHA)
                self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "Restore 8-bit A")

    def _emit_frame_allocation(self, frame_size: int, force_direct_stack: bool = False):
        """
        Emit stack frame allocation code.

        Delegates to the ABIModel which decides between PHB-per-byte
        (small/FixedStack) and TSC/SBC/TCS (large/Default).

        Args:
            frame_size: Number of bytes to allocate
            force_direct_stack: If True, always use TSC/SBC/TCS approach (for interrupt
                handlers where mode is unknown after register saves)
        """
        self.abi_model.emit_frame_alloc(self._emit_instr, frame_size, force_direct_stack)

    def _emit_frame_deallocation(self, frame_size: int):
        """
        Emit stack frame deallocation code.

        Uses TSC/ADC/TCS to add to stack pointer. We avoid PLB because frame
        bytes get overwritten by local variables during function execution,
        and PLB would load those garbage bytes into DBR.

        Args:
            frame_size: Number of bytes to deallocate
        """
        if frame_size <= 0:
            return

        # Always use TSC/ADC/TCS - PLB would corrupt DBR with overwritten frame data
        self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "16-bit A for frame cleanup")
        self._emit_instr(Opcode.TSC, comment="Get stack pointer")
        self._emit_instr(Opcode.CLC, comment="Clear carry for addition")
        self._emit_instr(Opcode.ADC_IMMEDIATE, Immediate(frame_size), f"Deallocate {frame_size} bytes")
        self._emit_instr(Opcode.TCS, comment="Update stack pointer")
        self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "Restore 8-bit A")

    def _get_prologue_stack_bytes(self, mir_func: MIRFunction,
                                   scratch_pool: ScratchRegisterPool = None) -> int:
        """
        Calculate bytes pushed by prologue that affect stack parameter offsets.

        Delegates to ABIInfo.prologue_stack_bytes for the actual computation.
        Kept for backward compatibility with any callers outside generate_function().

        Args:
            mir_func: MIR function
            scratch_pool: Optional scratch register pool (unused, kept for API compat)

        Returns:
            Number of bytes pushed by prologue
        """
        abi = ABIInfo.from_mir_function(mir_func)
        return abi.prologue_stack_bytes

    def _offset_location(self, location, offset: int):
        """Create new location offset from given location."""
        from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind

        if location.kind == LocationKind.SCRATCH:
            return PhysicalLocation(
                kind=LocationKind.SCRATCH,
                scratch_addr=location.scratch_addr + offset,
                size=1
            )
        elif location.kind == LocationKind.MEMORY:
            return PhysicalLocation(
                kind=LocationKind.MEMORY,
                memory_addr=location.memory_addr + offset,
                size=1
            )
        elif location.kind == LocationKind.STACK:
            return PhysicalLocation(
                kind=LocationKind.STACK,
                stack_offset=location.stack_offset + offset,
                size=1
            )
        else:
            raise ValueError(f"Cannot offset location kind: {location.kind}")

    def _analyze_far_self_trait_method(self, mir_func: MIRFunction):
        """Analyze trait method with far *self to determine access mode.

        For trait methods where self is a far pointer (24-bit), determine whether
        the method can use the fast DBR:Y path (leaf methods with no ROM/HW access)
        or needs the D=S fallback (methods with calls, ROM statics, or HW registers).

        DBR:Y path (fast): DBR is set to object's bank by dispatch caller,
        field access via LDA $offset,Y. Works when function doesn't need DBR
        for anything else (no ROM data, no HW, no function calls).

        D=S path: PHB+PHY push far self pointer to stack, then D=S infrastructure
        provides [dp],Y indirect long addressing for self access. DBR stays at
        caller's bank, so ROM/HW/calls work normally.
        """
        if not mir_func.is_trait_method or not mir_func.self_y_vreg:
            return

        # Check if self pointer is far
        self_type = mir_func.self_y_vreg.type_info
        from r65.compiler.hir.types import PointerTypeInfo
        if not isinstance(self_type, PointerTypeInfo) or not self_type.is_far:
            return

        # Already has far ptr stack params from other sources
        if mir_func.has_far_ptr_stack_params:
            mir_func.self_far_uses_d_equals_s = True
            return

        # Analyze MIR instructions to determine if DBR:Y is safe
        needs_d_equals_s = False

        for block in mir_func.blocks.values():
            for instr in block.instructions:
                # Any function call means we need D=S (callee expects normal DBR)
                if isinstance(instr, (Call, TraitDispatch)):
                    needs_d_equals_s = True
                    break

                # ROM static access uses absolute addressing which depends on DBR
                if isinstance(instr, (Load, Store)):
                    if hasattr(instr, 'source') and isinstance(instr.source, MemoryLocation):
                        if instr.source.storage_type in ('rom', 'hw'):
                            needs_d_equals_s = True
                            break
                    if hasattr(instr, 'dest') and isinstance(instr.dest, MemoryLocation):
                        if instr.dest.storage_type in ('rom', 'hw'):
                            needs_d_equals_s = True
                            break

            if needs_d_equals_s:
                break

        if needs_d_equals_s:
            mir_func.self_far_uses_d_equals_s = True
            mir_func.has_far_ptr_stack_params = True

    def emit_epilogue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
        """
        Emit function epilogue.

        Epilogue includes (in order):
        1. D restore for far pointer stack params (PLD)
        2. Preserved register restoration (reverse of prologue push order)
        3. DBR restoration (for databank=inline)
        4. Mode restoration (for transition=inline)
        5. Stack frame deallocation (non-entry functions)

        Note: Return value loading and RTS/RTL are handled separately
        by the return instruction.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
        """
        # Restore register saved by far pointer prologue
        # This must come first since it was the last push in prologue (LIFO)
        if mir_func.has_far_ptr_stack_params:
            from r65.compiler.mir.nodes import FarPtrStrategy
            if mir_func.far_ptr_strategy == FarPtrStrategy.SET_DBR:
                self._emit_instr(Opcode.PLB, comment="Restore Data Bank Register")
            else:
                self._emit_instr(Opcode.PLD, comment="Restore Direct Page register")

        self._emit_preserved_register_restores(mir_func)
        self._emit_dbr_restore(mir_func)
        self._emit_mode_restore(mir_func)

        # Note: Frame deallocation is NOT done here - it's combined with
        # the SP adjustment in _emit_stack_param_cleanup (control_flow_select.py)
        # to avoid issues with return address offsets.

    def _emit_preserved_register_restores(self, mir_func: MIRFunction):
        """
        Restore preserved registers in reverse order of prologue pushes.

        Prologue pushes: STATUS, A, X, Y, D, DBR
        Epilogue pops:   DBR, D, Y, X, A, STATUS

        Note: Interrupt handlers use select_return_from_interrupt() instead of
        this method. Their register save/restore is handled separately because
        they always save/restore the full 16-bit A (via REP #$20 before PHA/PLA).

        PLA is mode-sensitive: it pulls 1 byte in m8 mode, 2 bytes in m16 mode.
        """
        if not mir_func.preserves_attr:
            return

        preserved_regs = mir_func.preserves_attr.registers
        pop_order = ['DBR', 'D', 'Y', 'X', 'A', 'STATUS']

        for reg in pop_order:
            if reg in preserved_regs:
                pull_opcode = PULL_OPCODES.get(reg)
                if pull_opcode:
                    self._emit_instr(pull_opcode, comment=f"Restore {reg}")

    def _emit_dbr_restore(self, mir_func: MIRFunction):
        """Restore DBR for databank=inline functions."""
        if not (mir_func.is_far and mir_func.mode_attr):
            return

        from r65.compiler.hir.attributes import DataBankMode
        if mir_func.mode_attr.databank == DataBankMode.INLINE:
            self._emit_instr(Opcode.PLB, comment="Restore data bank")

    def _emit_mode_restore(self, mir_func: MIRFunction):
        """Mode restore placeholder.

        Mode switching for return values is now handled in select_return's
        _switch_to_exit_mode(), which runs BEFORE loading return values.
        This ensures 16-bit return values are loaded in the correct mode.

        This method is kept for API compatibility but no longer emits code.
        """
        # Mode switching is now handled in control_flow_select.py:_switch_to_exit_mode()
        pass


class ProgramFunctionGenerator:
    """
    Generates all functions in a program.

    Orchestrates function generation for entire MIR program.
    """

    def __init__(self,
                 emitter: AssemblyEmitter,
                 memory_allocator: MemoryAllocator):
        """
        Initialize program function generator.

        Args:
            emitter: Assembly emitter
            memory_allocator: Memory allocator
        """
        self.emitter = emitter
        self.mem_alloc = memory_allocator
        self.func_gen = FunctionCodeGenerator(emitter, memory_allocator)

    def generate_all_functions(self, mir_program):
        """
        Generate all functions in MIR program.

        Args:
            mir_program: MIRProgram to generate
        """
        # Create scratch pool from user-defined register variables
        scratch_pool = self.func_gen._create_scratch_pool(mir_program)

        # Emit section header
        self.emitter.emit_section_header("Functions")

        # Generate each function with the same scratch pool
        for mir_func in mir_program.functions:
            self.func_gen.generate_function(mir_func, scratch_pool=scratch_pool)

        # Blank line after all functions
        self.emitter.emit_blank_line()

    def generate_initialization_function(self, mir_program):
        """
        Generate __init_start function if needed.

        Args:
            mir_program: MIR program
        """
        # Check if __init_start exists in functions
        init_func = None
        for func in mir_program.functions:
            if func.name == "__init_start":
                init_func = func
                break

        if init_func:
            self.func_gen.generate_function(init_func)
