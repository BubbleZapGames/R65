"""
Function code generation: MIR functions → assembly.

Generates complete function bodies with headers, labels, and instructions.
"""

from typing import List, Set, Optional
from r65.compiler.mir.nodes import MIRFunction
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.register_alloc import ScratchRegisterPool, RegisterAllocator, PhysicalLocation, LocationKind
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.instruction_select_helpers import RegisterMappings
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.codegen.constants import DEFAULT_STACK_UPPER, M_FLAG, X_FLAG
from r65.compiler.codegen.opcodes import Opcode
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

        # DEBUG: Check scratch pool contents
        # print(f"DEBUG: generate_function({mir_func.name}), scratch_pool.scratches={len(scratch_pool.scratches) if scratch_pool.scratches else 0}")

        # Reset scratch pool for this function (each function gets fresh allocation)
        scratch_pool.reset()

        # Mark scratch registers occupied by scratch-promoted parameters
        # These must be reserved before normal allocation so other vregs don't use them
        for param_idx, scratch_addr in mir_func.scratch_param_addrs.items():
            for scratch in scratch_pool.scratches:
                if scratch.address == scratch_addr:
                    scratch.is_free = False
                    break

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

        # Pre-allocate hw-promoted parameter vregs (from FixedStack ABI promotion)
        for param_idx, hw_reg_name in mir_func.hw_param_regs.items():
            vreg = mir_func.param_to_vreg.get(param_idx)
            if vreg:
                vreg_size = get_type_size(vreg.type_info)
                location = PhysicalLocation(
                    kind=LocationKind.HARDWARE,
                    hw_register=hw_reg_name,
                    size=vreg_size
                )
                reg_alloc.allocations[vreg.id] = location
                hw_alloc = reg_alloc.get_hw_alloc(hw_reg_name)
                hw_alloc.allocated_vreg = vreg
                hw_alloc.is_bound = True

        # Pre-allocate self pointer vreg to Y register for trait methods
        # Self is passed in Y by the caller/dispatch wrapper
        if mir_func.is_trait_method and mir_func.self_y_vreg:
            location = PhysicalLocation(
                kind=LocationKind.HARDWARE,
                hw_register='Y',
                size=2  # Near pointer is always 16-bit
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

        # Create instruction selector with current function context
        instr_selector = InstructionSelector(self.emitter, reg_alloc, self.mem_alloc, mir_func, func_gen=self, abi_model=abi_model)

        # Initialize region-based spilling for this function
        instr_selector.call_selector.initialize_regions_for_function()

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
                        # Back-edge: predecessor not yet emitted
                        force_mode = True
                        break
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

        # Emit epilogue (if needed)
        # Note: Epilogue is emitted BEFORE the Return instruction in each block
        # So we don't emit it here. Instead, we handle it in the Return instruction
        # self.emit_epilogue(mir_func, reg_alloc)

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

        For now, simple DFS traversal from entry block.
        Future: optimize for fall-through and minimize jumps.

        Args:
            mir_func: MIR function

        Returns:
            List of block IDs in emission order
        """
        visited: Set[int] = set()
        order: List[int] = []

        def visit(block_id: int):
            if block_id in visited:
                return

            visited.add(block_id)
            order.append(block_id)

            # Visit successors
            block = mir_func.blocks.get(block_id)
            if block:
                for successor_id in block.successors:
                    visit(successor_id)

        # Start from entry block
        visit(mir_func.entry_block_id)

        # Visit any unreachable blocks (shouldn't happen, but be safe)
        for block_id in mir_func.blocks.keys():
            if block_id not in visited:
                visit(block_id)

        return order

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

    def _extract_vregs_from_instruction(self, instr) -> Set:
        """
        Extract virtual registers from instruction.

        This is a simplified implementation. A real implementation
        would use a visitor pattern or instruction introspection.

        Args:
            instr: MIR instruction

        Returns:
            Set of VirtualRegister objects
        """
        from r65.compiler.mir.nodes import VirtualRegister
        vregs = set()

        # Check all attributes of instruction for VirtualRegisters
        for attr_name in dir(instr):
            if attr_name.startswith('_'):
                continue

            attr = getattr(instr, attr_name)

            if isinstance(attr, VirtualRegister):
                vregs.add(attr)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, VirtualRegister):
                        vregs.add(item)

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

                # Debug: print register detection (comment out for production)
                # print(f"DEBUG: static {static_var.name}, is_register={storage_attr.is_register}")

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

        # Check if A holds a register parameter that would be clobbered
        # by frame allocation exceeding the ABI's push-based threshold.
        a_has_param = False
        if frame_size > self.abi_model.frame_alloc_clobbers_a_threshold:
            force_direct = mir_func.interrupt_attr is not None
            if not force_direct:
                from r65.compiler.hir import RegisterBinding
                for param in mir_func.parameters:
                    if (isinstance(param.binding, RegisterBinding) and
                        param.binding.register_name == 'A'):
                        a_has_param = True
                        break

        # Allocate stack frame for functions with locals.
        # For large frames (>4 bytes), TSC/SBC/TCS is used which clobbers A.
        # If A holds a register parameter, we must save it first.
        if frame_size > 0:
            force_direct = mir_func.interrupt_attr is not None

            if a_has_param:
                # Save A in Y before frame allocation.
                # TYA to restore is deferred to after ALL prologue code that
                # clobbers A (mode setup, preserves, PHD/TSC/TCD).
                self._emit_instr(Opcode.TAY, comment="Save A param before frame alloc")
                self._emit_frame_allocation(frame_size, force_direct_stack=force_direct)
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
                    push_opcode = RegisterMappings.PUSH_OPCODES.get(reg)
                    if push_opcode:
                        self._emit_instr(push_opcode, comment=f"Preserve {reg}")

        # Set up D = S for far pointer stack parameters
        # This enables [dp],Y addressing to access 24-bit pointers on the stack
        # Must come AFTER all other prologue pushes so D reflects final SP
        if mir_func.has_far_ptr_stack_params:
            self._emit_instr(Opcode.PHD, comment="Save Direct Page register")
            self._emit_instr(Opcode.TSC, comment="Transfer Stack to A")
            self._emit_instr(Opcode.TCD, comment="Transfer A to Direct Page (D = S)")

        # Restore A param from Y AFTER all prologue code that clobbers A.
        # Frame allocation (TSC/SBC/TCS) and PHD/TSC/TCD both overwrite A,
        # so TYA must be deferred to here. The current mode (set by mode setup
        # above) ensures TYA transfers the correct width:
        # - m8 functions: TYA transfers 8-bit (sufficient for u8 @ A)
        # - m16 functions: TYA transfers full 16-bit (required for u16 @ A)
        if frame_size > 0 and a_has_param:
            self._emit_instr(Opcode.TYA, comment="Restore A param after frame alloc")

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
        # Restore D register if we set up D = S for far pointer params
        # This must come first since PHD was the last push in prologue
        if mir_func.has_far_ptr_stack_params:
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
                pull_opcode = RegisterMappings.PULL_OPCODES.get(reg)
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
