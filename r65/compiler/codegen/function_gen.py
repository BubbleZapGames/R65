# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Function code generation: MIR functions → assembly.

Generates complete function bodies with headers, labels, and instructions.
"""

from typing import Dict, FrozenSet, List, Set
from r65.compiler.mir.nodes import (
    CallBase,
    MIRFunction, Return, VirtualRegister,
    MemoryLocation, Load, Store, Move, BinaryOp, UnaryOp, Compare, BitTest,
    Jump, CondBranch, TypeConvert, LoadIndirect, StoreIndirect, Push,
    MemoryFill, BlockCopy, InlineAsm, iter_operands, ModifyMemory,
    CompareIndirect,
)
from r65.compiler.codegen.clobber_summary import WRITABLE, code_writes
from r65.compiler.codegen.emitter import AssemblyEmitter, emit_node
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.register_alloc import ScratchRegisterPool, RegisterAllocator, PhysicalLocation, LocationKind, group_interrupt_scratch_pushes
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.codegen.constants import DEFAULT_STACK_UPPER, M_FLAG, X_FLAG
from r65.compiler.codegen.opcodes import (
    Opcode, PUSH_OPCODES, PULL_OPCODES,
)
from r65.compiler.codegen.asm_nodes import (
    Immediate, Address, StackOffset, Instruction, ModeChange, Comment)
from r65.compiler.codegen.abi import ABIInfo, StackFrameLayout
from r65.compiler.codegen.abi_model import emit_d_equals_s_keeping_c
from r65.compiler.errors import CodegenError
from r65.compiler.analysis.far_ptr_strategy import far_self_body_needs_dbr
from r65.compiler.analysis.width_demand import WidthAssignment


def _block_mode_switch(bits: int):
    """(opcode, operand, comment) of the block-entry switch to m`bits`."""
    opcode = Opcode.REP_IMMEDIATE if bits == 16 else Opcode.SEP_IMMEDIATE
    return opcode, Immediate(M_FLAG), f"REQUIRED: restore m{bits} mode for block"

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
        # What each generated function's code writes (clobber_summary), and
        # every function label; filled in as functions are generated.
        self.register_summaries: Dict[str, FrozenSet[str]] = {}
        self.function_labels: FrozenSet[str] = frozenset()
        self.call_graph = None   # analysis/call_graph, trait implementors included

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
        # For multi-byte params (e.g. u16 at $00), mark ALL overlapping scratches
        for param_idx, scratch_addr in mir_func.scratch_param_addrs.items():
            param_size = get_type_size(mir_func.parameters[param_idx].param_type)
            param_end = scratch_addr + param_size
            for scratch in scratch_pool.scratches:
                scratch_end = scratch.address + scratch.size
                # Overlap: scratch range intersects param range
                if scratch.address < param_end and scratch_end > scratch_addr:
                    scratch.is_free = False

        # Mark ALL globally-used scratch param addresses as occupied.
        # A caller's scratch params persist while the callee runs (under both
        # ABIs that promote to scratch). If we allow local variables at those addresses, the callee's locals
        # overwrite the caller's params. Globally reserving all param addresses
        # prevents this collision.
        for scratch in scratch_pool.scratches:
            if scratch.address in mir_func.global_scratch_param_bytes:
                scratch.is_free = False

        # Mark trait dispatch scratch addresses as occupied (caller-scoped).
        # These are only reserved in functions that contain TraitDispatch calls,
        # keeping the pool available for functions that don't use trait dispatch.
        for scratch in scratch_pool.scratches:
            if scratch.address in mir_func.trait_dispatch_scratch_bytes:
                scratch.is_free = False

        # Analyze far self trait methods to determine if D=S prologue is needed
        # Must run before ABIInfo creation since it may set has_far_ptr_stack_params
        self._analyze_far_self_trait_method(mir_func)

        # Check if databank=inline function actually needs DBR setup
        # Must run before ABIInfo creation since it affects prologue_stack_bytes
        self._analyze_dbr_inline_needed(mir_func)

        # Build ABI info and calculate prologue bytes BEFORE creating register allocator
        # This allows stack params to be allocated at their passed locations
        abi_info = ABIInfo.from_mir_function(mir_func)
        mir_func.abi_info = abi_info
        prologue_bytes = abi_info.prologue_stack_bytes

        # Create instruction-level liveness analyzer for precise liveness tracking
        from r65.compiler.mir.liveness import InstructionLivenessAnalyzer
        instr_liveness = InstructionLivenessAnalyzer(mir_func)

        # Build preliminary layout (local_frame_size filled in after allocation)
        layout = StackFrameLayout(abi=abi_info, local_frame_size=0)

        reg_alloc = RegisterAllocator(
            scratch_pool=scratch_pool,
            mir_func=mir_func,
            prologue_stack_bytes=prologue_bytes,
            instr_liveness=instr_liveness,
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
            if mir_func.self_far_uses_scratch:
                # FixedStack scratch path: self lives in a 3-byte zeropage scratch
                # slot. Prologue copies Y (lo/hi) and saved DBR (bank) into this
                # slot; body accesses self via [scratch],Y.
                assert mir_func.self_scratch_addr is not None
                location = PhysicalLocation(
                    kind=LocationKind.SCRATCH,
                    scratch_addr=mir_func.self_scratch_addr,
                    size=3
                )
                reg_alloc.allocations[mir_func.self_y_vreg.id] = location
            elif mir_func.self_far_uses_d_equals_s:
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
        frame_size = layout.local_frame_size

        # Update register allocator with frame info
        reg_alloc.frame_size = frame_size
        reg_alloc.has_frame_allocation = layout.has_frame

        # Fix up far self D=S vreg offset now that frame_size is known
        # The far self pointer (3 bytes) was pushed by PHB+PHY before frame alloc.
        # It sits just above the frame and the post-frame pushes:
        # offset = post_frame_prologue_bytes + frame_size + 1.
        # Skip for the FixedStack scratch path — that pre-allocates to SCRATCH and
        # stays put.
        if (mir_func.self_far_uses_d_equals_s and mir_func.self_y_vreg
                and not mir_func.self_far_uses_scratch):
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
        self.emitter.stack_depth = 0

        # Emit mode directives for WLA-DX assembler
        self._emit_mode_directives(mir_func)

        # Emit prologue (if needed)
        self.emit_prologue(mir_func, reg_alloc, frame_size)
        self.emitter.start_body()

        # Generate basic blocks
        block_order = self._compute_block_order(mir_func)

        # Track codegen exit modes per block to detect mode mismatches
        # between runtime predecessors (which may differ from emission order)
        codegen_exit_modes = {}
        # Whether X == Y at each emitted block's exit (see emitter.x_equals_y).
        codegen_exit_x_equals_y = {}
        # The stack depth and X's stack-slot contents at each emitted block's
        # exit, and the depth each block was entered at (emitter.x_slot).
        codegen_exit_slots = {}
        entry_depth = {}
        # Loop headers entered in their own mode from every emitted
        # predecessor: (node index, expected bits, back-edge predecessors).
        # Their mode switch is inserted afterwards, only if a back edge
        # actually exits in another mode.
        pending_back_edge_switches = []
        # The width each block is entered in (analysis/width_demand.py).
        def locate(vreg):
            location = reg_alloc.allocations.get(vreg.id)
            if location is None:
                return None
            if location.kind == LocationKind.STACK:
                return 'stack'
            if location.kind == LocationKind.HARDWARE:
                return location.hw_register
            return None
        entry_widths = WidthAssignment(mir_func, locate).entry

        for block_id in block_order:
            block = mir_func.blocks[block_id]

            # Initialize region tracking for this block (for optimized spilling)
            instr_selector.call_selector.initialize_regions_for_block(block_id)

            # Emit block label (except entry block which uses function label)
            if block_id != mir_func.entry_block_id:
                self.emitter.emit_label(f"{mir_func.name}__L{block_id}")
                # Emit .ACCU directive at every branch target so WLA-DX's
                # accumulator size tracking stays in sync. WLA-DX tracks
                # REP/SEP linearly through the text — it doesn't follow
                # branches — so a block that textually follows a 16-bit
                # epilogue will be assembled with wrong immediate sizes
                # unless we explicitly reset .ACCU here.
                self.emitter.emit_accu_mode(entry_widths[block_id])
                # X == Y holds at a join only if every predecessor has been
                # emitted (a back-edge source has not) and left it holding.
                preds = getattr(block, 'predecessors', None) or []
                self.emitter.x_equals_y = bool(preds) and all(
                    codegen_exit_x_equals_y.get(p, False) for p in preds)
                # Stack depth carries over when the emitted predecessors agree;
                # X's slot only when every predecessor is emitted and agrees.
                exits = [codegen_exit_slots[p] for p in preds if p in codegen_exit_slots]
                depths = {d for d, _x in exits}
                # A block is entered with the stack balanced: one that follows
                # a return in the text must not inherit its depth.
                self.emitter.stack_depth = (depths.pop() if len(depths) == 1
                                            else self.emitter.frame_base)
                xs = {x for _d, x in exits}
                if exits and len(exits) == len(preds) and len(xs) == 1:
                    self.emitter.x_slot = xs.pop()

            # Determine if we need to force a mode switch at block entry.
            # This is needed when an emitted predecessor exits in a different
            # mode than expected: the codegen may have switched modes for u16
            # ops without switching back (by design, to avoid redundant
            # SEP/REP). A predecessor not emitted yet (a back edge) is decided
            # once it has been, from its actual exit mode (see below).
            expected_bits = entry_widths[block_id]
            back_edge_preds = [p for p in block.predecessors
                               if p not in codegen_exit_modes]
            force_mode = any(codegen_exit_modes[p] != expected_bits
                             for p in block.predecessors
                             if p in codegen_exit_modes)
            if (back_edge_preds and not force_mode
                    and self.emitter.get_accu_mode() == expected_bits):
                pending_back_edge_switches.append(
                    (len(self.emitter.nodes), expected_bits, back_edge_preds))

            # Emit mode switch at block entry if needed
            if force_mode or self.emitter.get_accu_mode() != expected_bits:
                self._emit_instr(*_block_mode_switch(expected_bits))
                self.emitter.nodes[-1].required = True
                self.emitter.emit_accu_mode(expected_bits)

            entry_depth[block_id] = self.emitter.stack_depth
            block_start = len(self.emitter.nodes)

            # Emit instructions in block
            for idx, instr in enumerate(block.instructions):
                instr_selector._current_pos = (block_id, idx)
                instr_selector._next_instr = (block.instructions[idx + 1]
                                              if idx + 1 < len(block.instructions) else None)
                try:
                    instr_selector.select_instruction(instr)
                except CodegenError as e:
                    self._annotate_codegen_error(e, mir_func, instr)
                    raise

            self._switch_for_successors(mir_func, block, block_start,
                                        entry_widths, codegen_exit_modes)

            # Record this block's codegen exit mode
            codegen_exit_modes[block_id] = self.emitter.get_accu_mode()
            codegen_exit_x_equals_y[block_id] = self.emitter.x_equals_y
            codegen_exit_slots[block_id] = (self.emitter.stack_depth, self.emitter.x_slot)
            # A back edge into a block emitted earlier must leave the stack
            # where that block assumed it: slot names there were computed
            # from that depth.
            for succ in getattr(block, 'successors', None) or []:
                if succ in entry_depth and entry_depth[succ] != self.emitter.stack_depth:
                    raise CodegenError(
                        f"internal: stack depth {self.emitter.stack_depth} leaving "
                        f"block {block_id} of {mir_func.name} disagrees with the "
                        f"{entry_depth[succ]} block {succ} was emitted with")

        self._insert_back_edge_mode_switches(pending_back_edge_switches,
                                             codegen_exit_modes)

        # Emit shared epilogue if function has multiple returns.
        # Each Return instruction branched here instead of emitting inline.
        if shared_epilogue_label:
            self.emitter.emit_label(shared_epilogue_label)
            self.emit_epilogue(mir_func, reg_alloc)
            instr_selector.control_flow_selector._emit_return_instruction()

        if mir_func.interrupt_attr:
            self._reconcile_interrupt_saves(mir_func, instr_selector)

        # Record the body's stack high-water for the static stack-usage
        # analyzer: every push past the frame, spills and arguments alike.
        mir_func.codegen_max_region_spill_bytes = self.emitter.peak_displacement

        # Blank line after function
        self.emitter.emit_blank_line()

    @staticmethod
    def _annotate_codegen_error(error: CodegenError,
                                mir_func: MIRFunction,
                                instr) -> None:
        """
        Attach codegen-context to a CodegenError raised during instruction
        selection so it points at the offending source and MIR instruction.

        Errors from instruction selection (e.g. "Cannot resolve hardware
        register X as memory operand") are often raised without a source
        location; this fills in, when missing:
          - source_loc: the failing instruction's location, falling back to
            the enclosing function's definition location.
          - message: the enclosing function name.
          - hint: the MIR instruction being selected.

        Annotation is applied at most once, so re-raising through nested
        codegen layers does not stack duplicate context.
        """
        if getattr(error, '_codegen_annotated', False):
            return
        error._codegen_annotated = True

        if error.source_loc is None:
            error.source_loc = getattr(instr, 'source_loc', None) or mir_func.source_loc

        error.message = f"{error.message} (in function '{mir_func.name}')"

        if not error.hint:
            error.hint = f"while selecting MIR instruction: {instr!r}"

        # Refresh the Exception's own str() so tracebacks (e.g. under -v)
        # show the enriched message, not the original bare one.
        error.args = (error._format_message(),)

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
        """Block emission order (see compute_block_order)."""
        return compute_block_order(mir_func)

    def _insert_back_edge_mode_switches(self, pending, codegen_exit_modes):
        """Insert the mode switch at each pending loop header that a back edge
        reaches in the other mode (or from a block never emitted).

        Emission past the header is the same either way: the switch only
        brings the CPU to the mode the header was emitted in.
        """
        for index, bits, preds in reversed(pending):
            if all(codegen_exit_modes.get(p) == bits for p in preds):
                continue
            self.emitter.nodes[index:index] = [
                Instruction(*_block_mode_switch(bits), required=True),
                ModeChange("ACCU", bits),
            ]

    def _switch_for_successors(self, mir_func: MIRFunction, block, block_start: int,
                               entry_widths, codegen_exit_modes) -> None:
        """Switch to the width the block's successors are entered in, in
        front of its closing branches, so the switch runs on the edges that
        need it rather than at the top of a successor every edge enters.

        The width is that of the successors already emitted (back edges:
        their entry is fixed) when they agree, else that of all of them
        when they agree. A successor not yet emitted that wants the other
        width switches at its own top. SEP/REP of M leave N/Z/C, so the
        switch can sit between a compare and its branch.
        """
        from r65.compiler.codegen.opcodes import BRANCH_OPCODES, JUMP_OPCODES
        if not block.instructions or not isinstance(
                block.instructions[-1], (Jump, CondBranch)):
            return
        succs = [s for s in block.successors if s in mir_func.blocks]
        emitted = {entry_widths[s] for s in succs if s in codegen_exit_modes}
        wanted = emitted or {entry_widths[s] for s in succs}
        if len(wanted) != 1:
            return
        bits = wanted.pop()
        if self.emitter.get_accu_mode() == bits:
            return
        nodes = self.emitter.nodes
        k = len(nodes)
        while k > block_start:
            node = nodes[k - 1]
            if isinstance(node, Instruction) and (
                    node.opcode in BRANCH_OPCODES or node.opcode in JUMP_OPCODES):
                k -= 1
            elif isinstance(node, (Comment, ModeChange)):
                k -= 1
            else:
                break
        nodes[k:k] = [Instruction(*_block_mode_switch(bits), required=True),
                      ModeChange("ACCU", bits)]
        self.emitter.set_accu_mode_tracking(bits)

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
                vregs.update(v for _, v in iter_operands(instr)
                             if isinstance(v, VirtualRegister))

        # Allocate all at once
        reg_alloc.allocate_all(list(vregs))

    # ========================================================================
    # Scratch Pool Creation
    # ========================================================================

    def _create_scratch_pool(self, mir_program) -> ScratchRegisterPool:
        """
        Create scratch register pool from user-defined register variables.

        Scans static variables for those marked with the `register` flag
        (`#[zeropage(addr, register)]`).
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
            self.emitter.emit_accu_mode(16)
        else:
            # Default: m8 (8-bit accumulator)
            self.emitter.emit_accu_mode(8)

        # Emit index mode directive - always 16-bit in R65
        self.emitter.emit_index_mode(16)

    def emit_prologue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator, frame_size: int = 0):
        """
        Emit function prologue.

        Prologue may include (in order):
        1. Stack pointer initialization (entry functions with custom stack)
        2. Entry function setup (SEI/CLC/XCE/REP - switch to native mode)
        3. Interrupt register and scratch-pool saves
        4. Far-self pushes (PHB/PHY), then the data-bank load
        5. A-parameter save (TAY/TAX) when later steps clobber A
        6. Stack frame allocation (if needed)
        7. DBR management (databank=inline)
        8. REP #$20 for an m16 entry
        9. Register preservation (#[preserves])
        10. Far-pointer setup (PHD/TSC/TCD, or PHB/LDA/PHA/PLB)
        11. A-parameter restore

        Stack frame allocation comes BEFORE the #[preserves] pushes so saved
        registers sit at fixed offsets from S regardless of frame size.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
            frame_size: Number of bytes to allocate for local stack frame
        """
        # Entry function setup: switch from emulation mode to native mode,
        # then set S for a custom stack region.
        # This must happen BEFORE frame allocation since PHA behavior differs between modes
        if mir_func.is_entry:
            self._emit_entry_setup(mir_func)

        # For interrupt handlers, emit register saves BEFORE frame allocation
        # This is critical: frame allocation uses stack-relative addressing that
        # would corrupt saved registers if done before the pushes.
        # Order: register saves -> scratch saves -> frame allocation -> mode setup -> body
        if mir_func.interrupt_attr:
            self._mark_dead_volatile_loads(mir_func)
            # Track if we have scratches — D must be saved/restored if so
            # (scratch save resets D=0, which clobbers interrupted code's D)
            self._interrupt_has_scratches = bool(
                reg_alloc.scratch_pool and reg_alloc.scratch_pool.scratches)
            # Everything, until the emitted body says what it writes
            # (_reconcile_interrupt_saves).
            modified_regs = set(WRITABLE)
            mir_func.interrupt_modified_regs = modified_regs
            mir_func.interrupt_restore_runs = []
            start = len(self.emitter.nodes)
            self._emit_interrupt_register_saves(modified_regs)
            mir_func.interrupt_save_run = self.emitter.nodes[start:]
            self._emit_interrupt_scratch_saves(reg_alloc.scratch_pool)

        # For far self D=S path: push DBR (bank) and Y (addr) to stack BEFORE frame alloc
        # This places the 3-byte far self pointer on the stack where D=S can access it
        # Stack layout after PHB+PHY: [PHY_lo, PHY_hi, PHB_bank] (3 bytes)
        #
        # After PHB saves the caller's hijacked DBR=self_bank, we reset DBR to
        # the data bank for the body. D=S accesses self via [dp],Y indirect
        # long (self bank encoded in the on-stack pointer), so DBR isn't needed
        # for self. The data bank lets the body's absolute HW-register and
        # low-RAM accesses, and nested far-fn calls (e.g. U32::div_u8 from
        # U32::to_string), reach I/O — otherwise the hijacked DBR=$7E
        # propagates and STA $4204 hits WRAM at $7E:4204 instead of the HW
        # divider. (PBR is not a substitute: HiROM code runs in $C0-$FF,
        # which has no I/O or low RAM.)
        # The saved byte is discarded with the frame by the return cleanup (no
        # PLB); the dispatching caller restores its own DBR.
        if mir_func.self_far_uses_d_equals_s:
            self._emit_instr(Opcode.PHB, comment="Save caller's DBR (was hijacked to self bank)")
            self._emit_instr(Opcode.PHY, comment="Push self address (Y = object addr)")
            self.emitter.emit_data_bank_load("Set DBR to the data bank for the body")

        # For far self FixedStack scratch path: copy self (Y addr + caller DBR bank)
        # into a 3-byte zeropage scratch slot. DP stays at 0 so other scratch
        # params/registers remain reachable via [dp] addressing.
        # Stack effect: PHB pushes 1 byte (discarded by the return cleanup, no
        # PLB); STY/STA touch only zeropage. DBR is then set to the data bank
        # for the body, as on the D=S path.
        if mir_func.self_far_uses_scratch:
            scratch_addr = mir_func.self_scratch_addr
            assert scratch_addr is not None
            self._emit_instr(Opcode.PHB, comment="Save caller's DBR (was hijacked to self bank)")
            self._emit_instr(Opcode.STY_DP, Address(scratch_addr),
                             comment=f"Store self addr lo+hi to scratch ${scratch_addr:02X}")
            self._emit_instr(Opcode.LDA_STACK, StackOffset(1),
                             comment="Load self bank byte (= caller DBR just pushed)")
            self._emit_instr(Opcode.STA_DP, Address(scratch_addr + 2),
                             comment=f"Store self bank to scratch ${scratch_addr + 2:02X}")
            self.emitter.emit_data_bank_load("Set DBR to the data bank for the body")

        # Detect which hardware registers hold parameters.
        from r65.compiler.hir import RegisterBinding
        a_has_param = False
        b_has_param = False
        x_has_param = False
        y_has_param = False
        for param in mir_func.parameters:
            if isinstance(param.binding, RegisterBinding):
                reg = param.binding.register_name
                if reg == 'A':
                    a_has_param = True
                elif reg == 'B':
                    b_has_param = True
                elif reg == 'X':
                    x_has_param = True
                elif reg == 'Y':
                    y_has_param = True
        for reg in mir_func.hw_param_regs.values():
            if reg == 'A':
                a_has_param = True
            elif reg == 'B':
                b_has_param = True
            elif reg == 'X':
                x_has_param = True
            elif reg == 'Y':
                y_has_param = True

        # Keep an `@ A` or `@ B` parameter alive through the prologue. The
        # steps that write A are a TSC/SBC/TCS frame allocation, the
        # databank=inline DBR load, and far-pointer setup (LDA bank,S or
        # TSC/TCD); TSC writes B as well. The DBR load switches to a PEA when
        # A is live (_emit_dbr_inline_setup), so it never forces a save. For
        # the other two, `a_save` says where A is kept:
        #   'Y' / 'X' - TAY/TYA (TAX/TXA) around the whole prologue;
        #   'stack'   - X and Y hold parameters too, or B does (an m8 TYA
        #               restores only A): the frame is allocated with
        #               register pushes, and the far-pointer setup parks C
        #               on the stack around itself;
        #   None      - no prologue step writes A.
        # A parameter register is either explicitly bound or promoted by the
        # FixedStack pass (`hw_param_regs`).
        from r65.compiler.typeck.processor_mode import ModeState
        entry_m16 = mir_func.entry_m_mode == ModeState.M16
        force_direct = mir_func.interrupt_attr is not None
        a_save = None
        if a_has_param or b_has_param:
            frame_clobbers_a = frame_size > 0 and (
                frame_size > self.abi_model.frame_alloc_clobbers_a_threshold
                or force_direct
            )
            if frame_clobbers_a or mir_func.has_far_ptr_stack_params:
                if b_has_param:
                    a_save = 'stack'
                elif not y_has_param:
                    a_save = 'Y'
                elif not x_has_param:
                    a_save = 'X'
                else:
                    a_save = 'stack'
        a_live = (a_has_param or b_has_param) and a_save not in ('Y', 'X')

        if a_save == 'Y':
            self._emit_instr(Opcode.TAY, comment="Save A param before prologue")
        elif a_save == 'X':
            self._emit_instr(Opcode.TAX, comment="Save A param before prologue (Y has param)")

        # Allocate stack frame for functions with locals.
        if frame_size > 0:
            if a_save == 'stack':
                # Push-based allocation (PHX/PHY) doesn't clobber A, X, or Y.
                self.abi_model._emit_register_push_alloc(self._emit_instr, frame_size)
            else:
                self._emit_frame_allocation(frame_size, force_direct_stack=force_direct)

        # Handle DBR management for far functions with databank=inline
        if mir_func.is_far and mir_func.mode_attr and mir_func.bank_attr:
            from r65.compiler.hir.attributes import DataBankMode

            if mir_func.mode_attr.databank == DataBankMode.INLINE:
                if not getattr(mir_func, '_skip_dbr_inline', False):
                    self._emit_dbr_inline_setup(mir_func, keep_a=a_live or entry_m16)

        # An m16 entry is pinned by the caller before the JSR
        # (call_select._emit_entry_mode_switch); the REP restates it.
        if entry_m16:
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG), "Set m16 mode for u16 @ A parameter")

        # Emit register saves for #[preserves(...)]
        # Registers are pushed in order: STATUS, A, X, Y, D, DBR
        # (Popped in reverse order in epilogue/select_return)
        if mir_func.preserves_attr:
            preserved_regs = mir_func.preserves_attr.registers

            # Push in defined order (reverse order for pop)
            push_order = ['STATUS', 'A', 'X', 'Y', 'D', 'DBR']
            for reg in push_order:
                if reg in preserved_regs:
                    self._emit_instr(PUSH_OPCODES[reg], comment=f"Preserve {reg}")

        # Set up far pointer access for stack parameters
        # Must come AFTER all other prologue pushes so offsets are final
        if mir_func.has_far_ptr_stack_params:
            self._emit_far_ptr_setup(mir_func, frame_size,
                                     park_a=a_save == 'stack', m16=entry_m16)

        # Restore A param AFTER all prologue code that clobbers A.
        # The current mode (set by mode setup above) ensures the transfer
        # operates at the correct width (m8 for u8 @ A, m16 for u16 @ A).
        if a_save == 'Y':
            self._emit_instr(Opcode.TYA, comment="Restore A param after prologue")
        elif a_save == 'X':
            self._emit_instr(Opcode.TXA, comment="Restore A param after prologue")

    def _emit_dbr_inline_setup(self, mir_func: MIRFunction, keep_a: bool):
        """Save DBR and point it at the function's bank (databank=inline).

        `LDA #bank / PHA / PLB` is the cheap form, but it writes A, and in
        m16 its PHA pushes a byte the PLB never pulls. With `keep_a` (A holds
        a live value, or the function is entered in m16) the bank byte goes
        through `PEA` instead, which touches no register and pushes two bytes
        for the two PLBs.
        """
        # bank_attr.bank_number is a WLA bank *index*; the 65816 bank *byte*
        # is that index offset by the ROM base (.BASE $80 for LoROM FastROM,
        # $C0 for HiROM).
        dbr_bank = (getattr(self.emitter, 'rom_base', 0)
                    + mir_func.bank_attr.bank_number)
        self._emit_instr(Opcode.PHB, comment="Save current data bank")
        if keep_a:
            self._emit_instr(Opcode.PEA, Immediate((dbr_bank << 8) | dbr_bank),
                             "Push function's bank number (twice)")
            self._emit_instr(Opcode.PLB)
            self._emit_instr(Opcode.PLB, comment="Set data bank register")
        else:
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(dbr_bank),
                             "Load function's bank number")
            self._emit_instr(Opcode.PHA, comment="Push bank number")
            self._emit_instr(Opcode.PLB, comment="Set data bank register")

    def _emit_far_ptr_setup(self, mir_func: MIRFunction, frame_size: int,
                            park_a: bool, m16: bool):
        """Set up far-pointer stack-parameter access (SET_DBR or D=S).

        Both strategies go through A. With `park_a`, A holds a parameter that
        no free register can hold, so it is pushed before and pulled after;
        the pair is balanced, so the stack the body sees is unchanged.
        """
        from r65.compiler.mir.nodes import FarPtrStrategy
        if mir_func.far_ptr_strategy == FarPtrStrategy.SET_DBR:
            # SET_DBR: save DBR, load bank byte from far pointer param, set DBR
            # This preserves DP, keeping scratch registers and zeropage available
            bank_offset = self._get_far_ptr_bank_stack_offset(mir_func, frame_size)
            self._emit_instr(Opcode.PHB, comment="Save Data Bank Register")
            if park_a:
                self._emit_instr(Opcode.PHA, comment="Park A param across DBR setup")
                bank_offset += 2 if m16 else 1
            if m16:
                # A 16-bit load and push: the bank byte rides in the high
                # byte, so two PLBs leave it in DBR with the stack balanced.
                self._emit_instr(Opcode.LDA_STACK, StackOffset(bank_offset - 1),
                                 "Load far pointer bank byte (high) from stack")
                self._emit_instr(Opcode.PHA, comment="Push bank byte")
                self._emit_instr(Opcode.PLB)
                self._emit_instr(Opcode.PLB, comment="Set DBR to far pointer bank")
            else:
                self._emit_instr(Opcode.LDA_STACK, StackOffset(bank_offset),
                                 "Load far pointer bank byte from stack")
                self._emit_instr(Opcode.PHA, comment="Push bank byte")
                self._emit_instr(Opcode.PLB, comment="Set DBR to far pointer bank")
            if park_a:
                self._emit_instr(Opcode.PLA, comment="Restore A param")
        else:
            # D_EQUALS_S: enables [dp],Y indirect long addressing for 24-bit pointers
            self._emit_instr(Opcode.PHD, comment="Save Direct Page register")
            if park_a:
                emit_d_equals_s_keeping_c(self.emitter)
            else:
                self._emit_instr(Opcode.TSC, comment="Transfer Stack to A")
                self._emit_instr(Opcode.TCD, comment="Transfer A to Direct Page (D = S)")

    def _get_far_ptr_bank_stack_offset(self, mir_func: MIRFunction, frame_size: int) -> int:
        """Compute stack offset to the far pointer's bank byte for SET_DBR prologue.

        At the point the LDA runs, PHB has already been pushed (the SET_DBR
        prologue emits PHB before this LDA), so we need the full prologue_stack_bytes.
        """
        # Get the single far pointer param index
        far_idx = next(iter(mir_func.far_ptr_param_indices))
        base_offset = mir_func.stack_param_offsets[far_idx]

        # Bank byte is at +2 from param start (3-byte pointer: lo, hi, bank)
        bank_byte_base = base_offset + 2

        # Adjust for everything pushed before this point:
        # PHB has already been pushed, so include full prologue_stack_bytes
        return bank_byte_base + mir_func.abi_info.prologue_stack_bytes + frame_size

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
        self.emitter.emit_raw("SEI")   # Disable interrupts
        self.emitter.emit_raw("CLC")   # Clear carry for XCE
        self.emitter.emit_raw("XCE")   # Enter native mode

        # Reset leaves DBR=$00. Under FastROM the data bank is $80, so absolute
        # ROM reads go through the fast mirror too, not just opcode fetch;
        # hardware registers ($80:2100) and low RAM ($80:0000-1FFF) mirror
        # there as well. This belongs here rather than in the reset stub
        # because that still runs with PBR=$00 -- only after its JML is PBR
        # the code bank, which in LoROM is the data bank (PHK/PLB).
        if self.emitter.data_bank != 0x00:
            for opcode, operand in self.emitter.data_bank_load():
                self.emitter.emit_raw(emit_node(Instruction(opcode, operand), indent=""))

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

        # A custom #[stack] region: S is set only now. In emulation mode S
        # is pinned to page 1 and M to 8 bits, so before the XCE a
        # `REP #$20 / LDA #top / TCS` decodes LDA with a one-byte immediate
        # and never reaches the TCS, while the low-RAM allocator has already
        # placed statics where the declared stack was not. X is 16-bit
        # after the REP, and LDX/TXS leave A and its width alone.
        stack_top = self.mem_alloc.stack_upper
        if stack_top is not None and stack_top != DEFAULT_STACK_UPPER:
            self._emit_instr(Opcode.LDX_IMMEDIATE, Immediate(stack_top), "Stack top")
            self._emit_instr(Opcode.TXS, comment="Set stack pointer")

    def _mark_dead_volatile_loads(self, mir_func: MIRFunction):
        """Mark the interrupt handler's volatile reads whose value is never
        used (`mir_func.dead_volatile_loads`), so they are emitted as `BIT`,
        which reads the register without writing A.

        The scan stops at the first call, `asm!` or block move or fill.
        """
        from r65.compiler.mir.nodes import (
            Load, Store, StoreIndirect,
            HardwareRegister, VirtualRegister,
            InlineAsm, MemoryFill, BlockCopy,
        )

        # Collect all vreg IDs that are actually USED (appear as source/operand)
        used_vregs = set()
        for block in mir_func.blocks.values():
            for instr in block.instructions:
                # Check all fields for VirtualRegister references (as sources)
                for attr in ('source', 'left', 'right', 'operand', 'condition',
                             'address', 'ptr', 'function'):
                    val = getattr(instr, attr, None)
                    if isinstance(val, VirtualRegister):
                        used_vregs.add(val.id)
                # Check args list
                if hasattr(instr, 'args'):
                    for arg in instr.args:
                        if isinstance(getattr(arg, 'value', None), VirtualRegister):
                            used_vregs.add(arg.value.id)
                # Store/StoreIndirect source
                if isinstance(instr, Store) and isinstance(instr.source, VirtualRegister):
                    used_vregs.add(instr.source.id)
                if isinstance(instr, StoreIndirect) and isinstance(instr.source, VirtualRegister):
                    used_vregs.add(instr.source.id)

        def mark(instr):
            if not hasattr(mir_func, 'dead_volatile_loads'):
                mir_func.dead_volatile_loads = set()
            mir_func.dead_volatile_loads.add(id(instr))

        for block in mir_func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, (CallBase, InlineAsm, MemoryFill, BlockCopy)):
                    return
                if not (isinstance(instr, Load) and instr.source.is_volatile):
                    continue
                dest = instr.dest
                if isinstance(dest, HardwareRegister) and dest.name == 'A':
                    # A dead read into A: nothing later in the block reads A.
                    later = block.instructions[block.instructions.index(instr) + 1:]
                    if not any(isinstance(getattr(l, f, None), HardwareRegister)
                               and getattr(l, f).name == 'A'
                               for l in later for f in ('source', 'left')):
                        mark(instr)
                elif isinstance(dest, VirtualRegister) and dest.id not in used_vregs:
                    mark(instr)

    def _emit_interrupt_register_saves(self, modified_regs: set = None):
        """
        Emit register saves for interrupt handlers.

        This must be called BEFORE frame allocation to ensure correct stack layout.
        After this, the stack has the saved registers on top, and stack-relative
        addressing for local variables will work correctly.

        The 65816 has a hidden high byte of A (the "B accumulator") that PHA
        in m8 does not push. If the handler body does 16-bit work, that byte
        gets clobbered, and the interrupted code's next REP #$20 sees it. So A
        is saved under REP #$30 — both bytes, whatever width was interrupted.
        No PHP: RTI restores P from the CPU-pushed frame.

        Order of pushes: REP #$30, [PHA(16-bit)], PHX, PHY, PHD, PHB, SEP #$20
        (Corresponding pops in epilogue: PLB, PLD, PLY, PLX, [REP #$20, PLA(16-bit)])

        An interrupt keeps the M and X flags of the code it interrupted, so
        the handler cannot assume either width. The REP/SEP pair is
        unconditional: even a body that touches no register (a bare
        `INC counter`, a `BIT $4211`) reads or writes two bytes in m16, and
        an index register left x8 would truncate X and Y in the body. RTI
        puts the interrupted widths back.
        """
        save_all = modified_regs is None
        saves_a = save_all or 'A' in modified_regs
        # PHP is unnecessary — RTI restores P from the CPU-pushed stack frame.
        # PHA is only needed if the handler body modifies A.
        if saves_a:
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(M_FLAG | X_FLAG),
                             "16-bit A and X/Y: save the full registers")
            self._emit_instr(Opcode.PHA, comment="Save A (full 16-bit, includes hidden high byte)")
        else:
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(X_FLAG), "16-bit X/Y for handler body")
        # Only save X, Y, D, DBR if actually modified by the handler body
        if save_all or 'X' in modified_regs:
            self._emit_instr(Opcode.PHX, comment="Save X")
        if save_all or 'Y' in modified_regs:
            self._emit_instr(Opcode.PHY, comment="Save Y")
        if save_all or 'D' in modified_regs or self._interrupt_has_scratches:
            self._emit_instr(Opcode.PHD, comment="Save Direct Page")
        if save_all or 'DBR' in modified_regs:
            self._emit_instr(Opcode.PHB, comment="Save Data Bank")
        # Reset D=0 before the mode switch, so the scratch pushes that
        # follow can reach the pool through direct-page addressing. The
        # interrupted code may have left D anywhere (D=S for far pointer
        # stack params, say); PHD above saved it.
        #
        # `LDA #$0000` / `TCD` is 5 cycles to `PEA`/`PLD`'s 13, but it
        # goes through A — so it needs both a saved A and the known m16
        # the 16-bit PHA left us in. Without that, fall back to the
        # mode-independent push/pull pair.
        if self._interrupt_has_scratches:
            if saves_a:
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0), "D = 0 for scratch DP access")
                self._emit_instr(Opcode.TCD)
            else:
                self._emit_instr(Opcode.PEA, Immediate(0), "Push 0 for D reset")
                self._emit_instr(Opcode.PLD, comment="D = 0 (safe for scratch DP access)")
        # SEP #$20: m8 for the handler body, whatever width was interrupted.
        # The scratch saves that follow depend on it too: an `LDA dp`/`PHA`
        # for a scratch byte without a `PEI` partner pushes one byte in m8
        # but two in m16, while the epilogue always pulls back one.
        self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "8-bit A for handler body")
        # Keep the emitter's tracked width (and .ACCU) in step with the SEP.
        self.emitter.emit_accu_mode(8)

    def _reconcile_interrupt_saves(self, mir_func: MIRFunction, instr_selector):
        """Save exactly the registers the emitted handler writes.

        The prologue's save run and each RTI's restore run are emitted for
        every register, before the body exists. Here the emitted body is the
        authority (clobber_summary.code_writes, a call taking its callee's
        summary), and the runs are rebuilt for what it writes. The scratch
        pool is put back with PLA/STA (and an unpaired byte saved with
        LDA/PHA), so A changes even when the body never touches it - a body
        of nothing but `INC counter` would hand the interrupted code a
        scrambled A.
        """
        saved = mir_func.interrupt_modified_regs
        if saved is None or not mir_func.interrupt_save_run:
            return
        nodes = self.emitter.nodes
        save_run = mir_func.interrupt_save_run
        start = next(i for i, n in enumerate(nodes) if n is save_run[0])
        # The restore runs only pull what was saved; leave them out.
        in_restores = {id(n) for run in mir_func.interrupt_restore_runs for n in run}
        body = [n for n in nodes[start + len(save_run):] if id(n) not in in_restores]
        new_saved = set(code_writes(body, self.register_summaries, self.function_labels))
        if self._interrupt_has_scratches:
            new_saved.add('A')
        if new_saved == saved:
            return
        mir_func.interrupt_modified_regs = new_saved

        accu_mode = self.emitter.get_accu_mode()

        def capture(emit):
            outer = self.emitter.nodes
            self.emitter.nodes = []
            try:
                emit()
                return self.emitter.nodes
            finally:
                self.emitter.nodes = outer

        # Rebuild back to front so the earlier positions stay valid.
        runs = [save_run] + mir_func.interrupt_restore_runs
        replacements = [capture(lambda: self._emit_interrupt_register_saves(new_saved))]
        for _ in mir_func.interrupt_restore_runs:
            replacements.append(capture(
                lambda: instr_selector.emit_interrupt_return(new_saved)))
        positions = [next(i for i, n in enumerate(nodes) if n is run[0]) for run in runs]
        for pos, run, repl in sorted(zip(positions, runs, replacements), reverse=True,
                                     key=lambda t: t[0]):
            nodes[pos:pos + len(run)] = repl
        mir_func.interrupt_save_run = replacements[0]
        mir_func.interrupt_restore_runs = replacements[1:]
        self.emitter.set_accu_mode_tracking(accu_mode)

    def _emit_interrupt_scratch_saves(self, scratch_pool: ScratchRegisterPool):
        """
        Save the zeropage scratch pool to the stack for interrupt handlers.

        When an NMI fires mid-execution, the interrupted code may be using
        scratch registers for local variables. The interrupt handler also uses
        these scratch registers for its own temporaries. Without saving/restoring,
        the interrupted code's values get corrupted.

        D was reset to 0 by `_emit_interrupt_register_saves`, so PEI/LDA dp
        reach the pool whatever D the interrupted code had.

        Called after _emit_interrupt_register_saves() which leaves us in m8 mode.

        Args:
            scratch_pool: Pool of scratch registers to save
        """
        if not scratch_pool or not scratch_pool.scratches:
            return

        # D was already reset to 0 by _emit_interrupt_register_saves, so
        # the pool is reachable through direct-page addressing here.
        #
        # PEI pushes a direct-page word whatever the accumulator width
        # is, and without going through A — so the whole save runs in
        # the m8 the handler body wants, with no mode switching, at half
        # the per-byte cost of LDA/PHA.
        for kind, address in group_interrupt_scratch_pushes(scratch_pool.scratches):
            if kind == 'word':
                self._emit_instr(Opcode.PEI, Address(address),
                                 f"Save scratch word ${address:02X}")
            else:
                self._emit_instr(Opcode.LDA_DP, Address(address),
                                 f"Save scratch ${address:02X}")
                self._emit_instr(Opcode.PHA)

    def _emit_frame_allocation(self, frame_size: int, force_direct_stack: bool = False):
        """
        Emit stack frame allocation code.

        Delegates to the ABIModel which decides between push-based allocation
        (PHB, or PHX/PHY/PHB for Default) and TSC/SBC/TCS.

        Args:
            frame_size: Number of bytes to allocate
            force_direct_stack: If True, always use TSC/SBC/TCS (set for
                interrupt handlers)
        """
        self.abi_model.emit_frame_alloc(self._emit_instr, frame_size, force_direct_stack)


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

    def _analyze_dbr_inline_needed(self, mir_func: MIRFunction):
        """Check if a databank=inline function actually needs DBR setup.

        Scans MIR instructions to determine if the function accesses any
        bank-dependent memory (ROM data, HW registers, anything through a
        near pointer), makes near calls (which inherit DBR and may access
        bank-dependent memory) or contains inline asm (opaque).

        Sets mir_func._skip_dbr_inline = True if DBR setup can be safely
        skipped because the function only accesses DBR-independent memory
        (zeropage, lowram, RAM via long addressing, far pointers) and makes no
        near calls.
        """
        from r65.compiler.hir.attributes import DataBankMode

        if not (mir_func.is_far and mir_func.mode_attr and mir_func.bank_attr):
            return
        if mir_func.mode_attr.databank != DataBankMode.INLINE:
            return

        for block in mir_func.blocks.values():
            for instr in block.instructions:
                # ROM/HW static access uses absolute addressing which depends on DBR
                if isinstance(instr, Load):
                    if isinstance(instr.source, MemoryLocation):
                        if instr.source.storage_type in ('rom', 'hw'):
                            return
                elif isinstance(instr, Store):
                    if isinstance(instr.dest, MemoryLocation):
                        if instr.dest.storage_type in ('rom', 'hw'):
                            return
                elif isinstance(instr, ModifyMemory):
                    if instr.location.storage_type in ('rom', 'hw'):
                        return

                # A near pointer is DBR-relative by definition
                elif isinstance(instr, (LoadIndirect, StoreIndirect, CompareIndirect)):
                    if not instr.is_far:
                        return

                # asm! may use absolute addressing
                elif isinstance(instr, InlineAsm):
                    return

                # Near calls inherit DBR — callee may access ROM/HW data
                elif isinstance(instr, CallBase):
                    if not instr.is_far:
                        return

        # No bank-dependent accesses found — skip DBR setup
        mir_func._skip_dbr_inline = True

    def _analyze_far_self_trait_method(self, mir_func: MIRFunction):
        """Analyze trait method with far *self to determine access mode.

        For trait methods where self is a far pointer (24-bit), determine whether
        the method can use the fast DBR:Y path (leaf methods with no ROM/HW access)
        or needs the D=S fallback (methods with calls, ROM statics, or HW registers).

        DBR:Y path (fast): DBR is set to object's bank by dispatch caller,
        field access via LDA $offset,Y. Works when function doesn't need DBR
        for anything else (no ROM data, no HW, no function calls).

        D=S path: PHB+PHY push far self pointer to stack, then D=S infrastructure
        provides [dp],Y indirect long addressing for self access. The prologue
        then resets DBR to the data bank, so HW/low-RAM accesses and calls
        see the DBR they expect.
        """
        if not mir_func.is_trait_method or not mir_func.self_y_vreg:
            return

        # Check if self pointer is far
        self_type = mir_func.self_y_vreg.type_info
        from r65.compiler.hir.types import PointerTypeInfo
        if not isinstance(self_type, PointerTypeInfo) or not self_type.is_far:
            return

        # FixedStack pre-pass already chose the scratch-slot strategy — don't
        # also flip into D=S which would push self onto the stack.
        if mir_func.self_far_uses_scratch:
            return

        # Already has far ptr stack params from other sources
        if mir_func.has_far_ptr_stack_params:
            mir_func.self_far_uses_d_equals_s = True
            return

        if far_self_body_needs_dbr(mir_func):
            mir_func.self_far_uses_d_equals_s = True
            mir_func.has_far_ptr_stack_params = True

    def emit_epilogue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
        """
        Emit function epilogue.

        Epilogue includes (in order):
        1. PLD/PLB for far pointer stack params
        2. Preserved register restoration (reverse of prologue push order)
        3. PLB for databank=inline

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
        # FixedStack far-self scratch path: the PHB byte sits between the
        # frame and the return address; it's deallocated by the cleanup path
        # (see control_flow_select._emit_stack_param_cleanup) — no PLB here.
        # The caller is expected to manage its own DBR around the call, matching
        # the D=S protocol.

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
                self._emit_instr(PULL_OPCODES[reg], comment=f"Restore {reg}")

    def _emit_dbr_restore(self, mir_func: MIRFunction):
        """Restore DBR for databank=inline functions."""
        if not (mir_func.is_far and mir_func.mode_attr):
            return

        from r65.compiler.hir.attributes import DataBankMode
        if mir_func.mode_attr.databank == DataBankMode.INLINE:
            if not getattr(mir_func, '_skip_dbr_inline', False):
                self._emit_instr(Opcode.PLB, comment="Restore data bank")

    def _emit_mode_restore(self, mir_func: MIRFunction):
        """Emits nothing.

        Mode switching for return values happens in select_return's
        _switch_to_exit_mode() (control_flow_select.py), which runs BEFORE
        loading return values so 16-bit values load in the correct mode.
        """
        pass


class ProgramFunctionGenerator:
    """
    Holder for the per-program FunctionCodeGenerator.
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


def compute_block_order(mir_func: MIRFunction) -> List[int]:
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
