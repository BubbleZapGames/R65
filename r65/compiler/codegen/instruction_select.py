"""
Instruction selection: MIR → 65816 assembly.

Converts MIR instructions to WLA-DX assembly mnemonics with proper
addressing modes and register usage.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.codegen.function_gen import FunctionCodeGenerator
from r65.compiler.mir.nodes import (
    MIRFunction, MIRInstruction,
    Load, Store, LoadIndirect, StoreIndirect,
    Move, Return, Jump, JumpTable, CondBranch, Call,
    BinaryOp, UnaryOp, Compare, BitTest, Rotate, SetMode, TypeConvert, ToBool,
    Push, Pull, SaveRegister, RestoreRegister, ReturnFromInterrupt,
    StatusFlagTest, StatusFlagSet, StatusFlagRead,
    MemoryFill, BlockCopy, InlineAsm,
    VirtualRegister, HardwareRegister, Immediate as MIRImmediate, MemoryLocation
)
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.register_alloc import (
    RegisterAllocator, PhysicalLocation, LocationKind
)
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.errors import (
    unsupported_addressing_mode, unknown_value, missing_allocation,
    requires_constant, unsupported_operation
)
from r65.compiler.codegen.instruction_select_helpers import (
    XBAState, XBAStateManager
)
from r65.compiler.codegen.hw_register_tracker import (
    HardwareRegisterTracker, compute_vreg_last_uses
)
from r65.compiler.codegen.control_flow_select import ControlFlowInstructionSelector
from r65.compiler.codegen.call_select import CallInstructionSelector
from r65.compiler.codegen.memory_select import MemoryOperationSelector
from r65.compiler.codegen.move_select import MoveOperationSelector
from r65.compiler.codegen.type_conversion_select import TypeConversionSelector
from r65.compiler.codegen.compare_select import CompareSelector
from r65.compiler.codegen.opcodes import (
    Opcode, OPCODE_VARIANTS, PUSH_OPCODES, PULL_OPCODES,
    TRANSFER_OPCODES, LOAD_IMMEDIATE_OPCODES, STORE_DP_OPCODES, POWER_OF_2_SHIFTS
)
from r65.compiler.codegen.constants import (
    M_FLAG, X_FLAG, MX_FLAGS, BYTE_MASK, WORD_MASK, DP_BOUNDARY,
    WRAM_BANK, WRAM_BANK2, WRAM_BANK_START, WRAM_BANK2_START
)
from r65.compiler.codegen.location_resolver import (
    LocationResolver, StoreResolver, default_resolver
)
from r65.compiler.codegen.selector_context import SelectorContext
from r65.compiler.codegen.asm_nodes import Immediate, Address, StackOffset


class InstructionSelector:
    """
    Selects and emits 65816 instructions for MIR.

    Converts high-level MIR operations to actual 65816 assembly,
    handling addressing modes, register allocation, and instruction
    selection.
    """

    # Class-level counter for globally unique labels across all functions
    _global_label_counter = 0

    def __init__(self,
                 emitter: AssemblyEmitter,
                 register_allocator: RegisterAllocator,
                 memory_allocator: MemoryAllocator,
                 current_function: 'MIRFunction' = None,
                 func_gen: 'FunctionCodeGenerator' = None):
        """
        Initialize instruction selector.

        Args:
            emitter: Assembly emitter
            register_allocator: Register allocator for virtual registers
            memory_allocator: Memory allocator for static variables
            current_function: Current MIR function being generated (for far/near context)
            func_gen: Function code generator (for epilogue emission)
        """
        self.emitter = emitter
        self.reg_alloc = register_allocator
        self.mem_alloc = memory_allocator
        self.current_function = current_function
        self.func_gen = func_gen

        # Location resolver for addressing mode and opcode selection
        self._resolver = default_resolver

        # Shared context for composed selectors
        self._context = SelectorContext(
            emitter=emitter,
            register_allocator=register_allocator,
            memory_allocator=memory_allocator,
            resolver=self._resolver,
            current_function=current_function
        )
        # Wire up callbacks for context
        self._context.set_a_modified_callback(self._mark_a_modified)
        self._context.set_operand_location_callback(self._get_operand_location)

        # Helper classes for modular instruction selection
        self.xba_manager = XBAStateManager(emitter)
        self.control_flow_selector = ControlFlowInstructionSelector(self)
        self.call_selector = CallInstructionSelector(self)
        self.memory_selector = MemoryOperationSelector(self)
        self.move_selector = MoveOperationSelector(self)
        self.type_conversion_selector = TypeConversionSelector(self)
        self.compare_selector = CompareSelector(self)

        # Track type info from last Compare instruction for signed/unsigned branching
        self.last_comparison_type = None

        # Hardware register state tracker for optimization
        self.hw_tracker = HardwareRegisterTracker()
        self._instruction_index = 0

        # Pending SEP/REP mask for combining sequential mode flag sets
        # _pending_sep_mask: bits to set (8-bit mode), _pending_rep_mask: bits to clear (16-bit mode)
        self._pending_sep_mask = 0
        self._pending_rep_mask = 0

        # Initialize tracker from function parameters if available
        if current_function:
            self._init_hw_tracker(current_function)

        # Current source location for debug info propagation
        self._current_source_loc = None

    # ========================================================================
    # Pending Mode Flag Optimization
    # ========================================================================

    def _flush_pending_mode_flags(self):
        """
        Emit any pending SEP/REP instructions.

        This optimizes sequential STATUS.A16/XY16 assignments by combining them
        into a single SEP or REP instruction with a combined mask.

        Semantics:
            STATUS.A16 = true  -> 16-bit accumulator -> REP #$20 (clear M flag)
            STATUS.A16 = false -> 8-bit accumulator  -> SEP #$20 (set M flag)
            STATUS.XY16 = true  -> 16-bit index     -> REP #$10 (clear X flag)
            STATUS.XY16 = false -> 8-bit index      -> SEP #$10 (set X flag)

        For example:
            STATUS.A16 = false; STATUS.XY16 = false;  // Switch to 8-bit mode
        Becomes:
            SEP #$30  (instead of SEP #$20; SEP #$10)
        """
        if self._pending_sep_mask:
            mask = self._pending_sep_mask
            comment_parts = []
            if mask & M_FLAG:
                comment_parts.append("M")
            if mask & X_FLAG:
                comment_parts.append("X")
            comment = f"Set {'+'.join(comment_parts)} flag{'s' if len(comment_parts) > 1 else ''} (8-bit mode)"
            self.emitter.emit_instr(Opcode.SEP_IMMEDIATE, Immediate(mask), comment=comment)
            self._pending_sep_mask = 0

        if self._pending_rep_mask:
            mask = self._pending_rep_mask
            comment_parts = []
            if mask & M_FLAG:
                comment_parts.append("M")
            if mask & X_FLAG:
                comment_parts.append("X")
            comment = f"Clear {'+'.join(comment_parts)} flag{'s' if len(comment_parts) > 1 else ''} (16-bit mode)"
            self.emitter.emit_instr(Opcode.REP_IMMEDIATE, Immediate(mask), comment=comment)
            self._pending_rep_mask = 0

    def flush_pending_mode_flags(self):
        """Public method to flush pending mode flags (for function epilogue etc.)."""
        self._flush_pending_mode_flags()

    def _ensure_m8_mode(self):
        """
        Ensure accumulator is in 8-bit mode, switching if necessary.

        Call this before operations that require m8 mode when we might
        currently be in m16 mode. This is more efficient than switching
        back to m8 after every 16-bit operation.
        """
        if self.emitter.get_accu_mode() == 16:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "8-bit A")
            self.emitter.emit_accu_mode(8)

    def _ensure_m16_mode(self):
        """
        Ensure accumulator is in 16-bit mode, switching if necessary.

        Call this before operations that require m16 mode.
        """
        if self.emitter.get_accu_mode() != 16:
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A")
            self.emitter.emit_accu_mode(16)

    def _has_16bit_binding(self, register_name: str) -> bool:
        """
        Check if a hardware register has a 16-bit alias binding.

        This is used to determine if operations on a register should be
        done in 16-bit mode, even when the MIR instruction type says u8.

        Example: After `let w @ A : u16 = 1000`, operations on A should
        use 16-bit mode because A is bound to a u16 alias.

        Args:
            register_name: Hardware register name ('A', 'X', 'Y', etc.)

        Returns:
            True if register has an active u16/i16 binding
        """
        if not self.current_function:
            return False

        alias_tracker = getattr(self.current_function, 'alias_tracker', None)
        if not alias_tracker:
            return False

        binding_type = alias_tracker.get_register_binding_type(register_name)
        if binding_type and hasattr(binding_type, 'name'):
            return binding_type.name in ('u16', 'i16')

        return False

    # ========================================================================
    # Opcode Selection Helpers
    # ========================================================================

    def _get_opcode_for_location(self, mnemonic: str, location: PhysicalLocation) -> tuple[Opcode, Address | StackOffset]:
        """
        Get the appropriate Opcode variant and operand for a memory location.

        Delegates to the LocationResolver for unified addressing mode handling.

        When the current function has far pointer stack params (D = S setup),
        scratch/zeropage locations must use absolute addressing instead of DP
        because D no longer points to page 0.

        Args:
            mnemonic: Base instruction mnemonic (e.g., 'LDA', 'STA')
            location: Physical memory location

        Returns:
            Tuple of (Opcode variant, Operand)
        """
        # When D = S, DP addressing is unavailable - force absolute addressing
        # for both scratch registers and zeropage memory locations
        if (self.current_function and
            self.current_function.has_far_ptr_stack_params):
            if location.kind == LocationKind.SCRATCH:
                # Convert scratch to memory location with absolute addressing
                abs_location = PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_addr=location.scratch_addr,
                    size=location.size,
                    index_register=location.index_register
                )
                return self._resolver.resolve_and_get_opcode(mnemonic, abs_location)
            elif location.kind == LocationKind.MEMORY and location.memory_addr < 0x100:
                # Zeropage memory location - force absolute addressing
                # by using a resolver that treats it as non-DP
                from r65.compiler.codegen.location_resolver import ResolvedLocation, AddressingMode
                resolved = ResolvedLocation(
                    mode=self._resolver._get_indexed_mode(location.index_register, is_dp=False),
                    operand=Address(location.memory_addr),
                    address=location.memory_addr,
                    is_dp=False
                )
                opcode = self._resolver.get_opcode(mnemonic, resolved)
                return opcode, resolved.operand

        return self._resolver.resolve_and_get_opcode(mnemonic, location)

    def _emit_load(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """Emit a load instruction with the appropriate addressing mode."""
        opcode, operand = self._get_opcode_for_location(mnemonic, location)
        self.emitter.emit_instr(opcode, operand, comment, self._current_source_loc)

    def _emit_store(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """
        Emit a store instruction with the appropriate addressing mode.

        Uses StoreResolver to handle STX/STY addressing mode limitations.
        """
        if StoreResolver.needs_workaround(mnemonic, location):
            # Transfer to A first, then use STA
            transfer_op = StoreResolver.get_transfer_opcode(mnemonic)
            self.emitter.emit_instr(transfer_op, None, f"Transfer to A (no {mnemonic} with this addressing)", self._current_source_loc)
            opcode, operand = self._get_opcode_for_location('STA', location)
            self.emitter.emit_instr(opcode, operand, comment, self._current_source_loc)
            self._mark_a_modified()
            return

        opcode, operand = self._get_opcode_for_location(mnemonic, location)
        self.emitter.emit_instr(opcode, operand, comment, self._current_source_loc)

    def _emit_op(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """Emit an ALU operation with the appropriate addressing mode."""
        opcode, operand = self._get_opcode_for_location(mnemonic, location)
        self.emitter.emit_instr(opcode, operand, comment, self._current_source_loc)

    def _emit_implied(self, opcode: Opcode, comment: str = None):
        """Emit an implied addressing mode instruction."""
        self.emitter.emit_instr(opcode, None, comment, self._current_source_loc)

    def _emit_immediate(self, opcode: Opcode, value: int, comment: str = None):
        """Emit an immediate addressing mode instruction."""
        self.emitter.emit_instr(opcode, Immediate(value), comment, self._current_source_loc)

    def _emit_branch(self, opcode: Opcode, label: str, comment: str = None):
        """Emit a branch instruction to a label."""
        self.emitter.emit_instr(opcode, Address(label), comment, self._current_source_loc)

    def _block_label(self, block_id: int) -> str:
        """Format a block label with function-scoped naming."""
        func_name = self.current_function.name if self.current_function else ""
        return f"{func_name}__L{block_id}"

    @property
    def xba_state(self) -> XBAState:
        """XBA state (delegated to manager for backward compatibility)."""
        return self.xba_manager.state

    @xba_state.setter
    def xba_state(self, value: XBAState):
        """Set XBA state (delegated to manager)."""
        self.xba_manager.state = value

    def _get_unique_label(self) -> str:
        """Generate a unique label for inline branching."""
        InstructionSelector._global_label_counter += 1
        return f"__SCMP{InstructionSelector._global_label_counter}"

    # ========================================================================
    # XBA State Management (delegates to XBAStateManager)
    # ========================================================================

    def _invalidate_xba_state(self):
        """Invalidate XBA state at control flow boundaries."""
        self.xba_manager.invalidate()

    def _emit_xba(self, comment: str = None):
        """Emit XBA instruction with state tracking."""
        self.xba_manager.emit_xba(comment)

    def _ensure_xba_state_normal(self, comment: str = None):
        """Ensure A and B are in normal positions (A=A, B=B)."""
        self.xba_manager.ensure_normal(comment)

    def _ensure_xba_state_swapped(self, comment: str = None):
        """Ensure A and B are swapped (A=B, B=A)."""
        self.xba_manager.ensure_swapped(comment)

    def _mark_a_modified(self):
        """Mark A register as modified."""
        self.xba_manager.mark_a_modified()

    def _mark_b_modified(self):
        """Mark B register as modified."""
        self.xba_manager.mark_b_modified()

    def _access_b_value_in_a(self):
        """Make B register value available in A for reading."""
        self.xba_manager.access_b_value_in_a()

    def _store_to_b_from_a(self):
        """Store current A value into B register."""
        self.xba_manager.store_to_b_from_a()

    # ========================================================================
    # Hardware Register Tracking
    # ========================================================================

    def _init_hw_tracker(self, mir_func: 'MIRFunction'):
        """
        Initialize hardware register tracker from function parameters.

        For parameters with register aliases (e.g., `idx @ X`), marks
        the register as containing that parameter value and computes
        when that value is last used.

        Args:
            mir_func: MIR function being generated
        """
        # Compute last use for each virtual register
        vreg_last_uses = compute_vreg_last_uses(mir_func)

        # Also track parameter symbols' last use
        # This requires scanning the alias tracker
        if hasattr(mir_func, 'alias_tracker') and mir_func.alias_tracker:
            # For now, we'll compute liveness for aliased parameters
            # by treating them similarly to vregs
            pass

        # Initialize tracker
        self.hw_tracker.initialize_from_parameters(mir_func, vreg_last_uses)

    def _advance_instruction(self):
        """
        Advance to the next instruction.

        Updates the hardware register tracker's liveness information.
        """
        self._instruction_index += 1
        self.hw_tracker.advance_to(self._instruction_index)

    def _hw_reg_contains_vreg(self, reg_name: str, vreg) -> bool:
        """
        Check if a hardware register contains a specific virtual register value.

        Args:
            reg_name: Hardware register name ('A', 'X', 'Y')
            vreg: VirtualRegister to check for

        Returns:
            True if the register contains that vreg's value
        """
        return self.hw_tracker.contains_vreg(reg_name, vreg)

    def _hw_reg_is_free(self, reg_name: str) -> bool:
        """
        Check if a hardware register is free for scratch use.

        A register is free if its value is no longer needed (dead).

        Args:
            reg_name: Hardware register name

        Returns:
            True if register can be used as scratch
        """
        return self.hw_tracker.is_free(reg_name)

    def _find_vreg_in_hw_reg(self, vreg) -> str:
        """
        Find which hardware register (if any) contains a virtual register.

        Args:
            vreg: VirtualRegister to look for

        Returns:
            Register name ('A', 'X', 'Y') or None if not in any register
        """
        return self.hw_tracker.find_register_containing(vreg)

    def _mark_hw_reg_clobbered(self, reg_name: str):
        """
        Mark a hardware register as clobbered (contents unknown).

        Called when an instruction modifies a register.

        Args:
            reg_name: Register name ('A', 'X', 'Y')
        """
        self.hw_tracker.mark_clobbered(reg_name)

    def _mark_hw_reg_contains_vreg(self, reg_name: str, vreg, last_use: int = -1):
        """
        Mark that a hardware register now contains a virtual register value.

        Args:
            reg_name: Register name
            vreg: VirtualRegister that was loaded
            last_use: Last instruction index where vreg is used
        """
        self.hw_tracker.mark_contains_vreg(reg_name, vreg, last_use)

    # ========================================================================
    # Temporary Storage Management
    # ========================================================================

    def _get_temp_location(self) -> PhysicalLocation:
        """
        Get a safe temporary location for instruction selection.

        Returns a scratch register location if available, otherwise a stack slot.
        This is used when we need to temporarily store a hardware register value
        for operations that can't use hardware registers directly.

        Returns:
            PhysicalLocation for temporary storage
        """
        # First, try to find a free scratch register
        if hasattr(self.reg_alloc, 'scratch_pool'):
            for scratch in self.reg_alloc.scratch_pool.scratches:
                if scratch.is_free:
                    # Use this scratch register (mark temporarily busy)
                    return PhysicalLocation(
                        kind=LocationKind.SCRATCH,
                        scratch_addr=scratch.address,
                        size=scratch.size
                    )

        # No scratch available - require user to define scratch registers
        raise InstructionSelectionError(
            "No scratch register available for temporary storage. "
            "Define a scratch register using: #[zeropage(addr, register)] static mut SCRATCH: u8;"
        )

    def _get_temp_address(self) -> Address:
        """
        Get an Address object for the temp location.

        This is a convenience method for cases where we need an Address
        rather than a PhysicalLocation.

        Returns:
            Address for the temp location
        """
        temp_loc = self._get_temp_location()
        if temp_loc.kind == LocationKind.SCRATCH:
            return Address(temp_loc.scratch_addr)
        else:
            # Stack-relative - return as stack offset address
            # Note: This requires stack-relative addressing mode
            return StackOffset(temp_loc.stack_offset)

    # ========================================================================
    # Main Dispatch
    # ========================================================================

    def select_instruction(self, instr: MIRInstruction):
        """
        Select and emit assembly for MIR instruction.

        Args:
            instr: MIR instruction to convert
        """
        # Capture source location for debug info propagation
        self._current_source_loc = instr.source_loc

        # Flush pending mode flags before any non-StatusFlagSet instruction
        # This enables combining sequential STATUS.A16/XY16 assignments
        if not isinstance(instr, StatusFlagSet):
            self._flush_pending_mode_flags()

        if isinstance(instr, Load):
            self.memory_selector.select_load(instr)
        elif isinstance(instr, Store):
            self.memory_selector.select_store(instr)
        elif isinstance(instr, LoadIndirect):
            self.memory_selector.select_load_indirect(instr)
        elif isinstance(instr, StoreIndirect):
            self.memory_selector.select_store_indirect(instr)
        elif isinstance(instr, Move):
            self.move_selector.select_move(instr)
        elif isinstance(instr, TypeConvert):
            self.type_conversion_selector.select_type_convert(instr)
        elif isinstance(instr, ToBool):
            self.select_to_bool(instr)
        elif isinstance(instr, BinaryOp):
            self.select_binary_op(instr)
        elif isinstance(instr, UnaryOp):
            self.select_unary_op(instr)
        elif isinstance(instr, Compare):
            self.compare_selector.select_compare(instr)
        elif isinstance(instr, BitTest):
            self.compare_selector.select_bit_test(instr)
        elif isinstance(instr, Rotate):
            self.compare_selector.select_rotate(instr)
        elif isinstance(instr, Jump):
            self.control_flow_selector.select_jump(instr)
        elif isinstance(instr, JumpTable):
            self.control_flow_selector.select_jump_table(instr)
        elif isinstance(instr, CondBranch):
            self.control_flow_selector.select_cond_branch(instr)
        elif isinstance(instr, Return):
            self.control_flow_selector.select_return(instr)
        elif isinstance(instr, Call):
            self.call_selector.select_call(instr)
        elif isinstance(instr, SetMode):
            self.select_set_mode(instr)
        elif isinstance(instr, SaveRegister):
            self.select_save_register(instr)
        elif isinstance(instr, RestoreRegister):
            self.select_restore_register(instr)
        elif isinstance(instr, Push):
            self.select_push(instr)
        elif isinstance(instr, Pull):
            self.select_pull(instr)
        elif isinstance(instr, ReturnFromInterrupt):
            self.select_return_from_interrupt(instr)
        elif isinstance(instr, MemoryFill):
            self.select_memory_fill(instr)
        elif isinstance(instr, BlockCopy):
            self.select_block_copy(instr)
        elif isinstance(instr, InlineAsm):
            self.select_inline_asm(instr)
        elif isinstance(instr, StatusFlagTest):
            self.select_status_flag_test(instr)
        elif isinstance(instr, StatusFlagSet):
            self.select_status_flag_set(instr)
        elif isinstance(instr, StatusFlagRead):
            self.select_status_flag_read(instr)
        else:
            raise InstructionSelectionError(f"Unsupported MIR instruction: {type(instr).__name__}")

        # Advance instruction index for liveness tracking
        self._advance_instruction()

    # ========================================================================
    # Memory/Move/TypeConvert Operations (delegated)
    # ========================================================================
    # See memory_select.py for: select_load, select_store, select_load_indirect,
    # select_store_indirect
    # See move_select.py for: select_move
    # See type_conversion_select.py for: select_type_convert

    def select_to_bool(self, instr: ToBool):
        """
        Generate branchless boolean conversion.

        Converts value to boolean: 0 = false (0), non-zero = true (1).

        Uses branchless sequence:
            LDA source   ; Load value
            CMP #1       ; C=1 if A >= 1 (non-zero), C=0 if A = 0
            LDA #0       ; Clear A
            ADC #0       ; A = 0 + 0 + C = C (0 or 1)
            STA dest     ; Store result

        Args:
            instr: ToBool instruction
        """
        source = instr.source
        dest_loc = self._get_operand_location(instr.dest)

        # Load source value into A
        if isinstance(source, MIRImmediate):
            # Constant folding: evaluate at compile time
            result = 1 if source.value != 0 else 0
            self.emitter.emit_instr(Opcode.LDA_IMMEDIATE, result, comment="ToBool constant")
        else:
            # Load the source value
            src_loc = self._get_operand_location(source)
            src_opcode, src_operand = self._get_opcode_for_location('LDA', src_loc)
            self.emitter.emit_instr(src_opcode, src_operand)

            # Branchless conversion: CMP #1 / LDA #0 / ADC #0
            self.emitter.emit_instr(Opcode.CMP_IMMEDIATE, Immediate(1), comment="C=1 if non-zero")
            self.emitter.emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0))
            self.emitter.emit_instr(Opcode.ADC_IMMEDIATE, Immediate(0), comment="A = carry (0 or 1)")

        # Store result
        dest_opcode, dest_operand = self._get_opcode_for_location('STA', dest_loc)
        self.emitter.emit_instr(dest_opcode, dest_operand)

    # ========================================================================
    # Arithmetic Operations
    # ========================================================================

    def select_binary_op(self, instr: BinaryOp):
        """
        Generate code for BinaryOp instruction.

        dest = left op right

        Args:
            instr: BinaryOp instruction
        """
        op = instr.op
        is_u16 = self._is_16bit(instr.type_info)

        # OPTIMIZATION: Detect register increment/decrement patterns
        # reg = reg + 1  →  INX/INY/INC A
        # reg = reg - 1  →  DEX/DEY/DEC A
        # Check this BEFORE getting operand locations
        if (op in ('+', '-') and
            isinstance(instr.right, MIRImmediate) and
            instr.right.value == 1 and
            isinstance(instr.left, HardwareRegister) and
            isinstance(instr.dest, HardwareRegister) and
            instr.left.name == instr.dest.name):

            register = instr.dest.name
            if op == '+':
                # Increment
                if register == 'X':
                    self._emit_implied(Opcode.INX, f"{register}++")
                    return
                elif register == 'Y':
                    self._emit_implied(Opcode.INY, f"{register}++")
                    return
                elif register == 'A':
                    self._emit_implied(Opcode.INC, "A++")
                    return
            else:  # op == '-'
                # Decrement
                if register == 'X':
                    self._emit_implied(Opcode.DEX, f"{register}--")
                    return
                elif register == 'Y':
                    self._emit_implied(Opcode.DEY, f"{register}--")
                    return
                elif register == 'A':
                    self._emit_implied(Opcode.DEC, "A--")
                    return

        # Get operand locations
        left_loc = self._get_operand_location(instr.left)
        dest_loc = self._get_operand_location(instr.dest)

        # Check if immediate value exceeds 8-bit range
        has_large_immediate = (
            isinstance(instr.right, MIRImmediate) and instr.right.value > 0xFF
        )

        # Check if operation involves X or Y registers (always 16-bit in x16 mode)
        # Note: Direct X/Y arithmetic like `X = X + 5` is rejected by the type checker
        # (X/Y only support increment/decrement). This flag handles valid patterns like:
        # - `X = TEMP + 5` (dest is X, result transferred via TAX)
        # - `A = X + 5` (left is X, value transferred via TXA, operation in A)
        # In these cases, A must be in 16-bit mode to preserve the full 16-bit X/Y value.
        involves_index_register = (
            (left_loc.kind == LocationKind.HARDWARE and left_loc.hw_register in ('X', 'Y')) or
            (dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register in ('X', 'Y'))
        )

        # Check if we should use 16-bit mode because A is already in 16-bit mode
        # and the left operand is in A. This preserves 16-bit values that were
        # loaded by previous operations (e.g., after `let w @ A : u16 = 1000`).
        # Only apply this when left operand is specifically in A, not when loading
        # a new value from memory.
        a_already_in_16bit = (
            left_loc.kind == LocationKind.HARDWARE and
            left_loc.hw_register == 'A' and
            self.emitter.get_accu_mode() == 16
        )

        # Determine if we need 16-bit mode
        # Use 16-bit mode for ALL 16-bit operations, including memory-to-memory.
        # In 16-bit mode, LDA/STA/ADC etc. automatically handle both bytes.
        # This is more efficient than byte-by-byte operations in 8-bit mode.
        needs_16bit_mode = (
            (is_u16 or has_large_immediate or involves_index_register or a_already_in_16bit) and
            op in ('+', '-', '&', '|', '^', '<<', '>>')
        )

        # Switch to appropriate mode for operation
        # Note: We do NOT switch back immediately after the operation.
        # The mode will be switched when the next operation needs a different mode,
        # or at function return. This avoids redundant REP/SEP pairs.
        if needs_16bit_mode:
            self._ensure_m16_mode()
        else:
            # For 8-bit operations, ensure we're in m8 mode
            self._ensure_m8_mode()

        # Load left operand into A (if not already there)
        if left_loc.kind == LocationKind.HARDWARE and left_loc.hw_register == 'A':
            # Left operand is already in A, no need to load
            pass
        elif left_loc.kind == LocationKind.HARDWARE:
            # Transfer from other hardware register to A
            self._emit_register_transfer(left_loc.hw_register, 'A')
        elif left_loc.kind == LocationKind.IMMEDIATE:
            self._emit_load('LDA', left_loc)
        else:
            # OPTIMIZATION: Check if vreg value is already in X or Y
            # If so, use TXA/TYA instead of loading from memory
            hw_reg = None
            if isinstance(instr.left, VirtualRegister):
                hw_reg = self._find_vreg_in_hw_reg(instr.left)

            if hw_reg and hw_reg in ('X', 'Y'):
                self._emit_register_transfer(hw_reg, 'A')
            else:
                # Load left operand from memory/stack into A
                self._emit_load('LDA', left_loc)

        # Perform operation
        if op == '+':
            self._emit_add(instr.right, is_u16)
        elif op == '-':
            self._emit_sub(instr.right, is_u16)
        elif op == '&':
            self._emit_and(instr.right, is_u16)
        elif op == '|':
            self._emit_or(instr.right, is_u16)
        elif op == '^':
            self._emit_xor(instr.right, is_u16)
        elif op == '<<':
            self._emit_shift_left(instr.right, is_u16)
        elif op == '>>':
            self._emit_shift_right(instr.right, is_u16)
        elif op == '*':
            self._emit_multiply(instr.right, is_u16)
        elif op == '/':
            self._emit_divide(instr.right, is_u16)
        else:
            raise unsupported_operation("binary operation", op)

        # Store result from A (if destination is not A)
        if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
            # Result is already in A, no need to store
            pass
        elif dest_loc.kind == LocationKind.HARDWARE:
            # Transfer from A to other hardware register
            self._emit_register_transfer('A', dest_loc.hw_register)
        else:
            # Store result from A to memory/stack
            self._emit_store('STA', dest_loc)

        # Note: We do NOT switch back to 8-bit mode here.
        # The mode will be restored when needed by the next operation
        # or at function return. This avoids redundant REP/SEP pairs.

    def select_unary_op(self, instr: UnaryOp):
        """
        Generate code for UnaryOp instruction.

        dest = op operand

        Args:
            instr: UnaryOp instruction
        """
        op = instr.op
        is_u16 = self._is_16bit(instr.type_info)
        operand_loc = self._get_operand_location(instr.operand)
        dest_loc = self._get_operand_location(instr.dest)

        # Ensure correct mode for operation
        if is_u16:
            self._ensure_m16_mode()
        else:
            self._ensure_m8_mode()

        # Load operand
        self._emit_load('LDA', operand_loc)

        # Perform operation
        if op == '!':
            # Logical NOT: convert to 0 or 1, then XOR with 1
            self._emit_immediate(Opcode.CMP_IMMEDIATE, 0, "Check if zero")
            self._emit_branch(Opcode.BEQ, "+", "Branch if zero")
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 0, "Was non-zero, result = 0")
            self._emit_branch(Opcode.BRA, "++")
            self.emitter.emit_label("+")
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 1, "Was zero, result = 1")
            self.emitter.emit_label("++")
        elif op == '~':
            # Bitwise NOT
            mask = 0xFFFF if is_u16 else 0xFF
            self._emit_immediate(Opcode.EOR_IMMEDIATE, mask, "Bitwise complement")
        elif op == '-':
            # Negation
            mask = 0xFFFF if is_u16 else 0xFF
            self._emit_immediate(Opcode.EOR_IMMEDIATE, mask, "Complement")
            self._emit_implied(Opcode.INC, "Add 1 for two's complement")
        else:
            raise unsupported_operation("unary operation", op)

        # Store result
        self._emit_store('STA', dest_loc)

    # ========================================================================
    # Compare/BitTest/Rotate Operations (delegated)
    # ========================================================================
    # See compare_select.py for: select_compare, select_bit_test, select_rotate

    # ========================================================================
    # Arithmetic Helpers
    # ========================================================================

    def _emit_add(self, right_operand, is_u16: bool):
        """Emit addition operation."""
        self._emit_implied(Opcode.CLC)
        self._emit_binary_operation_with_operand("ADC", right_operand, is_u16)

    def _emit_sub(self, right_operand, is_u16: bool):
        """Emit subtraction operation."""
        self._emit_implied(Opcode.SEC)
        self._emit_binary_operation_with_operand("SBC", right_operand, is_u16)

    def _emit_and(self, right_operand, is_u16: bool):
        """Emit bitwise AND operation."""
        self._emit_binary_operation_with_operand("AND", right_operand, is_u16)

    def _emit_or(self, right_operand, is_u16: bool):
        """Emit bitwise OR operation."""
        self._emit_binary_operation_with_operand("ORA", right_operand, is_u16)

    def _emit_xor(self, right_operand, is_u16: bool):
        """Emit bitwise XOR operation."""
        self._emit_binary_operation_with_operand("EOR", right_operand, is_u16)

    # ========================================================================
    # Shift/Multiply/Divide Helpers
    # ========================================================================

    def _require_immediate(self, operand, operation: str) -> int:
        """Validate operand is immediate and return its value."""
        if not isinstance(operand, MIRImmediate):
            raise requires_constant(operation)
        return operand.value

    def _emit_repeated_opcode(self, opcode: Opcode, count: int):
        """Emit an implied opcode repeated count times."""
        for _ in range(count):
            self._emit_implied(opcode)

    def _emit_shift_left(self, right_operand, is_u16: bool):
        """Emit left shift operation (A << count)."""
        count = self._require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 0x00,
                f"Shift by {count} >= {bit_width} bits = 0")
            return

        self._emit_repeated_opcode(Opcode.ASL, count)

    def _emit_shift_right(self, right_operand, is_u16: bool):
        """Emit right shift operation (A >> count)."""
        count = self._require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 0x00,
                f"Shift by {count} >= {bit_width} bits = 0")
            return

        self._emit_repeated_opcode(Opcode.LSR, count)

    def _emit_multiply(self, right_operand, is_u16: bool):
        """Emit multiply by power of 2 (A * 1/2/4/8) using ASL instructions."""
        value = self._require_immediate(right_operand, "Multiply")
        shift_count = POWER_OF_2_SHIFTS.get(value)

        if shift_count is None:
            raise InstructionSelectionError(
                f"Multiply operator only supports 1, 2, 4, 8 (got {value}). "
                f"Use mul() for general multiplication.")

        self._emit_repeated_opcode(Opcode.ASL, shift_count)

    def _emit_divide(self, right_operand, is_u16: bool):
        """Emit divide by power of 2 (A / 1/2/4/8) using LSR instructions."""
        value = self._require_immediate(right_operand, "Divide")
        shift_count = POWER_OF_2_SHIFTS.get(value)

        if shift_count is None:
            raise InstructionSelectionError(
                f"Divide operator only supports 1, 2, 4, 8 (got {value}). "
                f"Use div() for general division.")

        self._emit_repeated_opcode(Opcode.LSR, shift_count)

    # ========================================================================
    # Control Flow (delegated to ControlFlowInstructionSelector)
    # ========================================================================
    # See control_flow_select.py for: select_jump, select_jump_table,
    # select_cond_branch, select_return

    # ========================================================================
    # Function Calls (delegated to CallInstructionSelector)
    # ========================================================================
    # See call_select.py for: select_call, built-in handling, indirect calls

    # ========================================================================
    # Mode Control
    # ========================================================================

    def select_set_mode(self, instr: SetMode):
        """
        Generate code for SetMode instruction.

        Args:
            instr: SetMode instruction
        """
        if instr.is_set:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, instr.mask)
        else:
            self._emit_immediate(Opcode.REP_IMMEDIATE, instr.mask)

    # ========================================================================
    # Register Save/Restore
    # ========================================================================

    def select_save_register(self, instr: SaveRegister):
        """
        Generate code for SaveRegister instruction.

        Args:
            instr: SaveRegister instruction
        """
        reg_name = instr.register.name
        push_opcode = PUSH_OPCODES.get(reg_name)
        if push_opcode:
            self._emit_implied(push_opcode)
        else:
            raise InstructionSelectionError(f"Cannot push register: {reg_name}")

    def select_restore_register(self, instr: RestoreRegister):
        """
        Generate code for RestoreRegister instruction.

        Args:
            instr: RestoreRegister instruction
        """
        reg_name = instr.register.name
        pull_opcode = PULL_OPCODES.get(reg_name)
        if pull_opcode:
            self._emit_implied(pull_opcode)
        else:
            raise InstructionSelectionError(f"Cannot pull register: {reg_name}")

    # ========================================================================
    # Interrupt Handler Instructions
    # ========================================================================

    def select_push(self, instr: Push):
        """
        Generate code for Push instruction (save register to stack).

        Args:
            instr: Push instruction
        """
        reg = instr.register.name
        push_opcode = PUSH_OPCODES.get(reg)
        if push_opcode:
            self._emit_implied(push_opcode)
        else:
            raise InstructionSelectionError(f"Cannot push register: {reg}")

    def select_pull(self, instr: Pull):
        """
        Generate code for Pull instruction (restore register from stack).

        Args:
            instr: Pull instruction
        """
        reg = instr.register.name
        pull_opcode = PULL_OPCODES.get(reg)
        if pull_opcode:
            self._emit_implied(pull_opcode)
        else:
            raise InstructionSelectionError(f"Cannot pull register: {reg}")

    def select_return_from_interrupt(self, instr: ReturnFromInterrupt):
        """
        Generate code for ReturnFromInterrupt instruction.

        Epilogue sequence for interrupt handlers:
        1. Deallocate stack frame (if any)
        2. Restore all registers (reverse order of prologue pushes)
        3. RTI

        The order is critical: prologue pushes registers THEN allocates frame,
        so epilogue must deallocate frame THEN restore registers.

        Args:
            instr: ReturnFromInterrupt instruction
        """
        # 1. Deallocate stack frame using direct stack manipulation
        # Always use TSC/CLC/ADC/TCS for interrupt handlers since the mode
        # after the handler body is unknown, and we need explicit mode control
        frame_size = 0
        if self.reg_alloc:
            frame_size = self.reg_alloc.frame_size

        if frame_size > 0:
            self._emit_immediate(Opcode.REP_IMMEDIATE, 0x20, "16-bit A for stack adjustment")
            self._emit_implied(Opcode.TSC, "Get stack pointer")
            self._emit_implied(Opcode.CLC)
            self._emit_immediate(Opcode.ADC_IMMEDIATE, frame_size, f"Deallocate {frame_size} bytes")
            self._emit_implied(Opcode.TCS, "Set stack pointer")

        # 2. Restore registers (reverse order of prologue: PHA PHX PHY PHD PHB PHP)
        # Pop order: PLP PLB PLD PLY PLX PLA
        # CRITICAL: Restore P first so A is restored in its original mode
        self._emit_implied(Opcode.PLP, "Restore processor status (first - restores mode)")
        self._emit_implied(Opcode.PLB, "Restore Data Bank")
        self._emit_implied(Opcode.PLD, "Restore Direct Page")
        self._emit_implied(Opcode.PLY, "Restore Y")
        self._emit_implied(Opcode.PLX, "Restore X")
        self._emit_implied(Opcode.PLA, "Restore A (in original mode)")

        # 3. Return from interrupt
        self._emit_implied(Opcode.RTI)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_temp_location(self, size: int = 1) -> PhysicalLocation:
        """
        Get a safe temporary location for instruction selection.

        Tries to use a scratch register from the pool first, falls back to
        a dedicated stack slot if no scratch is available.

        Args:
            size: Size in bytes (1 or 2)

        Returns:
            PhysicalLocation for temporary storage
        """
        # Try to find a free scratch register from the pool
        if self.reg_alloc.scratch_pool:
            for scratch in self.reg_alloc.scratch_pool.scratches:
                if scratch.is_free and scratch.size >= size:
                    return PhysicalLocation(
                        kind=LocationKind.SCRATCH,
                        scratch_addr=scratch.address,
                        size=size
                    )

        # No scratch available - require user to define scratch registers
        raise InstructionSelectionError(
            "No scratch register available for temporary storage. "
            "Define a scratch register using: #[zeropage(addr, register)] static mut SCRATCH: u8;"
        )

    def _get_temp_address(self) -> Address | None:
        """
        Get a safe temporary direct page address for storing hardware registers.

        This is used when we need to store X/Y registers which don't support
        stack-relative addressing.

        Returns:
            Address for temporary storage, or None if no scratch available
        """
        if self.reg_alloc.scratch_pool:
            for scratch in self.reg_alloc.scratch_pool.scratches:
                if scratch.is_free:
                    return Address(scratch.address)
        return None

    def _emit_binary_operation_with_operand(self, operation: str, right_operand, is_u16: bool):
        """
        Emit a binary operation with right operand.

        Handles immediate, memory, and hardware register operands.
        For hardware registers, stores to temp location first.

        Args:
            operation: Instruction mnemonic (ADC, SBC, AND, ORA, EOR)
            right_operand: Right operand (MIRImmediate, VirtualRegister, HardwareRegister)
            is_u16: Whether this is a 16-bit operation
        """
        # Get the immediate opcode for this operation
        immediate_opcode = OPCODE_VARIANTS[operation]['IMMEDIATE']

        if isinstance(right_operand, MIRImmediate):
            # Use the operand value directly - any necessary masking for 8-bit mode
            # is handled by the caller (select_binary_op) for memory-to-memory operations
            # For 16-bit ops with hardware registers, the full value is needed
            value = right_operand.value
            self._emit_immediate(immediate_opcode, value)
        else:
            right_loc = self._get_operand_location(right_operand)
            if right_loc.kind == LocationKind.HARDWARE:
                # Hardware register - must store to temp location first
                # (65816 can't use hardware registers as operands for these ops)
                if right_loc.hw_register == 'B':
                    # B register requires XBA to access - can use stack or scratch
                    temp_loc = self._get_temp_location()
                    self._access_b_value_in_a()
                    self._emit_store('STA', temp_loc, "Store B to temp")
                    self._ensure_xba_state_normal("Restore A")
                    self._emit_op(operation, temp_loc)
                elif right_loc.hw_register == 'A':
                    # A can use stack-relative addressing
                    temp_loc = self._get_temp_location()
                    self._emit_store('STA', temp_loc, "Store A to temp")
                    self._emit_op(operation, temp_loc)
                elif right_loc.hw_register in ['X', 'Y']:
                    # X/Y don't support stack-relative for STX/STY
                    # Try scratch first, fall back to push/pop
                    temp_addr = self._get_temp_address()
                    if temp_addr:
                        # Use scratch register
                        temp_loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=temp_addr.value, size=1)
                        store_opcode = STORE_DP_OPCODES[right_loc.hw_register]
                        self.emitter.emit_instr(store_opcode, temp_addr, f"Store {right_loc.hw_register} to temp")
                        self._emit_op(operation, temp_loc)
                    else:
                        # No scratch available - use push/pop pattern
                        # PHX/PHY pushes value to stack, then we can use stack-relative
                        push_opcode = Opcode.PHX if right_loc.hw_register == 'X' else Opcode.PHY
                        pull_opcode = Opcode.PLX if right_loc.hw_register == 'X' else Opcode.PLY
                        self._emit_implied(push_opcode, f"Push {right_loc.hw_register} for temp")
                        # Stack-relative location at offset 1 (just pushed)
                        temp_loc = PhysicalLocation(kind=LocationKind.STACK, stack_offset=1, size=1)
                        self._emit_op(operation, temp_loc)
                        self._emit_implied(pull_opcode, f"Restore {right_loc.hw_register}")
                else:
                    raise InstructionSelectionError(f"Cannot use hardware register in operation: {right_loc.hw_register}")
            else:
                # Memory location
                self._emit_op(operation, right_loc)

    def _emit_16bit_mem_to_mem(self, src_loc: PhysicalLocation, dest_loc: PhysicalLocation, comment: str = None):
        """
        Emit 16-bit memory-to-memory move using 16-bit mode.

        Switches to 16-bit accumulator mode if needed and performs single LDA/STA.
        Does NOT switch back after - mode will be restored when needed.

        Args:
            src_loc: Source memory location
            dest_loc: Destination memory location
            comment: Optional comment for first instruction
        """
        self._ensure_m16_mode()
        self._emit_load('LDA', src_loc, comment)
        self._emit_store('STA', dest_loc)
        # Note: Do NOT switch back here - mode will be restored when needed

    def _emit_16bit_immediate_store(self, value: int, dest_loc: PhysicalLocation):
        """
        Emit 16-bit immediate store using 16-bit mode.

        Switches to 16-bit mode if needed. Does NOT switch back after -
        mode will be restored when needed.

        Args:
            value: 16-bit immediate value
            dest_loc: Destination memory location
        """
        self._ensure_m16_mode()
        self._emit_immediate(Opcode.LDA_IMMEDIATE, value)
        self._emit_store('STA', dest_loc)
        # Note: Do NOT switch back here - mode will be restored when needed

    def _emit_register_transfer(self, src_reg: str, dest_reg: str):
        """
        Emit register-to-register transfer.

        Handles all valid 65816 register transfer combinations.
        For indirect transfers (e.g., X to Y), routes through A.
        B register transfers use XBA instruction.
        STATUS register transfers use PHP/PLP instructions.

        Args:
            src_reg: Source register name ('A', 'X', 'Y', 'B', 'STATUS')
            dest_reg: Destination register name ('A', 'X', 'Y', 'B', 'STATUS')
        """
        if src_reg == dest_reg:
            # No-op
            return

        # Special handling for STATUS register
        if src_reg == 'STATUS' or dest_reg == 'STATUS':
            if src_reg == 'STATUS' and dest_reg == 'A':
                # STATUS -> A: Push status, pull to A (requires m8)
                self._ensure_m8_mode()
                self._emit_implied(Opcode.PHP, "Push STATUS")
                self._emit_implied(Opcode.PLA, "Pull STATUS to A")
                return
            elif src_reg == 'A' and dest_reg == 'STATUS':
                # A -> STATUS: Push A, pull to status (requires m8)
                self._ensure_m8_mode()
                self._emit_implied(Opcode.PHA, "Push A")
                self._emit_implied(Opcode.PLP, "Pull to STATUS")
                return
        # Special handling for PBR register (read-only, via stack)
        if src_reg == 'PBR' and dest_reg == 'A':
            # PBR -> A: Push PBR, pull to A (requires m8)
            self._ensure_m8_mode()
            self._emit_implied(Opcode.PHK, "Push PBR")
            self._emit_implied(Opcode.PLA, "Pull PBR to A")
            return
        # Special handling for DBR register (via stack)
        if src_reg == 'DBR' or dest_reg == 'DBR':
            if src_reg == 'DBR' and dest_reg == 'A':
                # DBR -> A: Push DBR, pull to A (requires m8)
                self._ensure_m8_mode()
                self._emit_implied(Opcode.PHB, "Push DBR")
                self._emit_implied(Opcode.PLA, "Pull DBR to A")
                return
            elif src_reg == 'A' and dest_reg == 'DBR':
                # A -> DBR: Push A, pull to DBR (requires m8)
                self._ensure_m8_mode()
                self._emit_implied(Opcode.PHA, "Push A")
                self._emit_implied(Opcode.PLB, "Pull to DBR")
                return
        # Special handling for B register
        if src_reg == 'B' or dest_reg == 'B':
            if src_reg == 'B' and dest_reg == 'A':
                # B to A: If we're swapped, A already has B's value
                self._ensure_xba_state_swapped("Transfer B to A")
                return
            elif src_reg == 'A' and dest_reg == 'B':
                # A to B: XBA swaps them
                self._emit_xba("Transfer A to B")
                return
            elif src_reg == 'B':
                # B to X/Y: B -> A -> X/Y
                self._access_b_value_in_a()
                self._emit_register_transfer('A', dest_reg)
                self._ensure_xba_state_normal("Restore A")
                return
            elif dest_reg == 'B':
                # X/Y to B: X/Y -> A -> B
                # Save current A
                self._emit_implied(Opcode.PHA, "Save A")
                self._emit_register_transfer(src_reg, 'A')
                self._emit_xba("Move to B")
                self._emit_implied(Opcode.PLA, "Restore A")
                self._invalidate_xba_state()  # State unknown after PLA
                return

        # Direct transfers
        transfer_opcode = TRANSFER_OPCODES.get((src_reg, dest_reg))

        if transfer_opcode:
            self._emit_implied(transfer_opcode)
        else:
            # Indirect transfer through A (e.g., X to Y)
            if src_reg != 'A':
                self._emit_register_transfer(src_reg, 'A')
            if dest_reg != 'A':
                self._emit_register_transfer('A', dest_reg)

    def _emit_load_immediate_to_register(self, reg: str, value: int, is_u16: bool,
                                          persist_16bit_mode: bool = False):
        """
        Emit load immediate into hardware register.

        Switches to appropriate mode if needed. Does NOT switch back after -
        mode will be restored when needed by subsequent operations.

        Args:
            reg: Register name ('A', 'X', 'Y')
            value: Immediate value
            is_u16: Whether to use 16-bit format
            persist_16bit_mode: Deprecated - mode is always persisted now.
                               Kept for API compatibility.
        """
        load_opcode = LOAD_IMMEDIATE_OPCODES[reg]

        # Handle 16-bit load to A register - need mode switch
        # Also switch if value exceeds 8-bit range (0-255)
        needs_16bit = is_u16 or (reg == 'A' and value > 0xFF)
        if needs_16bit and reg == 'A':
            self._ensure_m16_mode()
            self._emit_immediate(load_opcode, value & 0xFFFF)
            # Note: Do NOT switch back here - mode will be restored when needed
        else:
            # 8-bit load - make sure we're in m8 mode for A register
            if reg == 'A':
                self._ensure_m8_mode()
            self._emit_immediate(load_opcode, value)

    def _get_operand_location(self, operand) -> PhysicalLocation:
        """
        Get physical location for an operand.

        Args:
            operand: VirtualRegister, HardwareRegister, MemoryLocation, or Immediate

        Returns:
            PhysicalLocation for the operand
        """
        if isinstance(operand, VirtualRegister):
            return self.reg_alloc.get_location(operand)
        elif isinstance(operand, HardwareRegister):
            return self.reg_alloc.get_hw_location(operand)
        elif isinstance(operand, MemoryLocation):
            # Check if this is a #[rom] static with a ROM label - access directly from ROM
            if (operand.symbol and
                hasattr(operand.symbol, 'rom_label') and
                operand.symbol.rom_label and
                operand.storage_type == 'rom'):
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_label=operand.symbol.rom_label,
                    size=1,  # Size determined by context
                    index_register=operand.index_register  # Pass indexed addressing info
                )
            # Check if MemoryLocation has explicit address (for offsets)
            elif operand.address is not None:
                # Use explicit address (already includes offset for arrays/structs)
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_addr=operand.address,
                    size=1,  # Size determined by context
                    index_register=operand.index_register  # Pass indexed addressing info
                )
            else:
                # Get address from memory allocator
                alloc = self.mem_alloc.get_allocation(operand.symbol)
                if alloc:
                    return PhysicalLocation(
                        kind=LocationKind.MEMORY,
                        memory_addr=alloc.address,
                        size=alloc.size,
                        index_register=operand.index_register  # Pass indexed addressing info
                    )
                else:
                    raise missing_allocation(operand.symbol.name)
        elif isinstance(operand, MIRImmediate):
            # Immediate value - return as immediate location
            return PhysicalLocation(
                kind=LocationKind.IMMEDIATE,
                immediate_value=operand.value,
                size=1  # Will be determined by context
            )
        else:
            raise InstructionSelectionError(f"Unknown operand type: {type(operand)}")

    def _format_operand(self, location: PhysicalLocation) -> str:
        """
        Format physical location as assembly operand.

        Args:
            location: Physical location

        Returns:
            Formatted operand string
        """
        if location.kind == LocationKind.HARDWARE:
            # Hardware register - can't be used as memory operand
            # This shouldn't happen in normal code generation
            raise InstructionSelectionError(f"Cannot use hardware register as memory operand: {location.hw_register}")
        elif location.kind == LocationKind.SCRATCH:
            base = f"${location.scratch_addr:02X}"
            if location.index_register:
                return f"{base},{location.index_register}"
            return base
        elif location.kind == LocationKind.MEMORY:
            # Check if using a ROM label (for #[rom] data accessed directly)
            if location.memory_label:
                base = location.memory_label
            elif location.memory_addr is not None:
                if location.memory_addr < DP_BOUNDARY:
                    # Zero-page
                    base = f"${location.memory_addr:02X}"
                else:
                    # Absolute
                    base = f"${location.memory_addr:04X}"
            else:
                raise InstructionSelectionError("Memory location has neither address nor label")
            # Add index register if present (e.g., "$20,X" or "LABEL,X")
            if location.index_register:
                return f"{base},{location.index_register}"
            return base
        elif location.kind == LocationKind.STACK:
            # Stack-relative addressing using 65816 stack-relative mode
            # Format: $XX,S where XX is the offset from stack pointer
            return f"${location.stack_offset:02X},S"
        elif location.kind == LocationKind.IMMEDIATE:
            # Immediate value
            return f"#{location.immediate_value}"
        else:
            raise InstructionSelectionError(f"Unknown location kind: {location.kind}")

    def _offset_location(self, location: PhysicalLocation, offset: int) -> PhysicalLocation:
        """
        Create new location offset from given location.

        Args:
            location: Base location
            offset: Byte offset

        Returns:
            New PhysicalLocation at offset
        """
        if location.kind == LocationKind.SCRATCH:
            return PhysicalLocation(
                kind=LocationKind.SCRATCH,
                scratch_addr=location.scratch_addr + offset,
                size=1
            )
        elif location.kind == LocationKind.MEMORY:
            if location.memory_addr is not None:
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_addr=location.memory_addr + offset,
                    size=1
                )
            elif location.memory_label is not None:
                # Label-based location - create new location with offset applied to label
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_label=f"{location.memory_label}+{offset}",
                    size=1
                )
            else:
                # Get all attributes for debugging
                attrs = {k: v for k, v in vars(location).items() if not k.startswith('_')}
                raise InstructionSelectionError(
                    f"Cannot offset MEMORY location with no address or label. Location attrs: {attrs}"
                )
        elif location.kind == LocationKind.STACK:
            return PhysicalLocation(
                kind=LocationKind.STACK,
                stack_offset=location.stack_offset + offset,
                size=1
            )
        elif location.kind == LocationKind.IMMEDIATE:
            # For immediate values, shift to get the high byte
            new_value = (location.immediate_value >> (offset * 8)) & 0xFF
            return PhysicalLocation(
                kind=LocationKind.IMMEDIATE,
                immediate_value=new_value,
                size=1
            )
        else:
            raise InstructionSelectionError(f"Cannot offset location kind: {location.kind}")

    def _is_16bit(self, type_info) -> bool:
        """
        Check if type is 16-bit.

        Args:
            type_info: Type information

        Returns:
            True if 16-bit type (u16, i16, near pointer, or near function pointer)
        """
        if hasattr(type_info, 'name'):
            return type_info.name in ('u16', 'i16')
        # Near pointers are 16-bit (far pointers are 24-bit)
        if hasattr(type_info, 'pointee_type'):
            return not getattr(type_info, 'is_far', False)
        # Near function pointers are 16-bit (far function pointers are 24-bit)
        from r65.compiler.hir.types import FunctionTypeInfo
        if isinstance(type_info, FunctionTypeInfo):
            return not type_info.is_far
        return False

    # ========================================================================
    # Array Initialization Operations
    # ========================================================================

    def _emit_indexed_store(self, base_addr: int, index_reg: str, comment: str = None):
        """Emit a store with indexed addressing: STA base,X or STA base,Y."""
        is_dp = base_addr < DP_BOUNDARY
        if index_reg == 'X':
            opcode = Opcode.STA_DP_X if is_dp else Opcode.STA_ABSOLUTE_X
        else:  # 'Y'
            opcode = Opcode.STA_ABSOLUTE_Y  # No STA dp,Y on 65816
        self.emitter.emit_instr(opcode, Address(base_addr), comment)

    def select_memory_fill(self, instr: MemoryFill):
        """
        Generate code for MemoryFill instruction.

        Fills a memory region with a constant value using a loop.
        Used for array fill expressions like [0; 256].

        For 8-bit elements:
            LDA #fill_value
            LDX #count-1
        .loop:
            STA dest,X
            DEX
            BPL .loop

        Args:
            instr: MemoryFill instruction
        """
        dest_loc = self._get_operand_location(instr.dest)
        fill_value = instr.fill_value
        count = instr.count
        element_size = instr.element_size

        # Total bytes to fill
        total_bytes = count * element_size

        # Generate unique label for this loop
        loop_label = self._get_unique_label()

        self.emitter.emit_comment(f"Fill {count} x {element_size}B elements with #{fill_value}")

        base_addr = dest_loc.memory_addr if dest_loc.kind == LocationKind.MEMORY else dest_loc.scratch_addr

        if element_size == 1:
            # 8-bit element fill
            if total_bytes <= 256:
                # Can use X as counter (0-255)
                self._emit_immediate(Opcode.LDA_IMMEDIATE, fill_value & 0xFF)
                self._emit_immediate(Opcode.LDX_IMMEDIATE, total_bytes - 1)
                self.emitter.emit_label(loop_label)
                self._emit_indexed_store(base_addr, 'X')
                self._emit_implied(Opcode.DEX)
                self._emit_branch(Opcode.BPL, loop_label)
            else:
                # Large array - use 16-bit counter
                # Use Y for counting, X for offset
                self._emit_immediate(Opcode.LDA_IMMEDIATE, fill_value & 0xFF)
                self._emit_immediate(Opcode.LDX_IMMEDIATE, 0x00)
                self._emit_immediate(Opcode.LDY_IMMEDIATE, total_bytes & 0xFFFF)
                self.emitter.emit_label(loop_label)
                self._emit_indexed_store(base_addr, 'X')
                self._emit_implied(Opcode.INX)
                self._emit_implied(Opcode.DEY)
                self._emit_branch(Opcode.BNE, loop_label)

        elif element_size == 2:
            # 16-bit element fill - need to fill low and high bytes
            low_byte = fill_value & 0xFF
            high_byte = (fill_value >> 8) & 0xFF

            # Use X as index (forward), Y as counter (decrement)
            self._emit_immediate(Opcode.LDX_IMMEDIATE, 0x00)
            self._emit_immediate(Opcode.LDY_IMMEDIATE, count)
            self.emitter.emit_label(loop_label)
            # Store low byte at base+X
            self._emit_immediate(Opcode.LDA_IMMEDIATE, low_byte)
            self._emit_indexed_store(base_addr, 'X')
            self._emit_implied(Opcode.INX)
            # Store high byte at base+X+1
            self._emit_immediate(Opcode.LDA_IMMEDIATE, high_byte)
            self._emit_indexed_store(base_addr, 'X')
            self._emit_implied(Opcode.INX)
            # Decrement counter and loop
            self._emit_implied(Opcode.DEY)
            self._emit_branch(Opcode.BNE, loop_label)

    def select_block_copy(self, instr: BlockCopy):
        """
        Generate code for BlockCopy instruction.

        Copies a block of data from ROM to RAM using MVN instruction.
        Used for array literal expressions like [1, 2, 3, 4].

        Generated code:
            LDA #count-1          ; A = byte count - 1
            LDX #<source_addr     ; X = source low word
            LDY #<dest_addr       ; Y = dest low word
            MVN src_bank, dst_bank

        Note: MVN copies A+1 bytes from src_bank:X to dst_bank:Y.

        Args:
            instr: BlockCopy instruction
        """
        from r65.compiler.codegen.asm_nodes import BlockMove

        dest_loc = self._get_operand_location(instr.dest)
        rom_label = instr.rom_data.label
        count = instr.count

        self.emitter.emit_comment(f"Block copy {count} bytes from ROM to RAM")

        # Set up for MVN: A = count - 1, X = source, Y = dest
        # For 16-bit index mode, we need REP #$10 first
        self._emit_immediate(Opcode.REP_IMMEDIATE, MX_FLAGS, "16-bit A and index")

        # Load count - 1 into A
        self._emit_immediate(Opcode.LDA_IMMEDIATE, count - 1)

        # Load source address (ROM data label) - use raw emission for label operand
        self.emitter.emit_instr(Opcode.LDX_IMMEDIATE, Address(rom_label))

        # Load destination address
        self._emit_immediate(Opcode.LDY_IMMEDIATE, dest_loc.memory_addr & WORD_MASK)

        # Perform block move
        # MVN src_bank, dst_bank
        # Assuming ROM is in bank 0 and RAM destination bank is $7E
        # For now, use bank 0 for ROM and calculate destination bank from address
        dest_addr = dest_loc.memory_addr
        if dest_addr >= WRAM_BANK2_START:
            dest_bank = WRAM_BANK2
        elif dest_addr >= WRAM_BANK_START:
            dest_bank = WRAM_BANK
        else:
            dest_bank = 0x00  # Low RAM or zeropage

        self.emitter.emit_instr(Opcode.MVN, BlockMove(0x00, dest_bank))

        # Restore 8-bit mode if needed (depends on context)
        self._emit_immediate(Opcode.SEP_IMMEDIATE, MX_FLAGS, "Restore 8-bit mode")

    # ========================================================================
    # Inline Assembly
    # ========================================================================

    def select_inline_asm(self, instr: InlineAsm):
        """
        Generate code for InlineAsm instruction.

        Emits raw assembly instructions verbatim. The compiler assumes all
        registers may be clobbered after inline assembly.

        Args:
            instr: InlineAsm instruction containing list of assembly strings
        """
        for asm_instr in instr.instructions:
            # Emit each assembly instruction as a raw line
            # The instruction string may contain operands, e.g., "LDA #$42"
            # Use raw emission since this is user-provided assembly
            self.emitter.emit_raw(asm_instr)

    # ========================================================================
    # STATUS Flag Operations
    # ========================================================================

    def select_status_flag_test(self, instr: StatusFlagTest):
        """
        Generate code for StatusFlagTest instruction.

        For branchable flags (Carry, Zero, Overflow, Negative):
            - No code needed; branch instruction directly tests the flag

        For non-branchable flags (Irq, Decimal, Index, Accumulator):
            - Emits: PHP; PLA; AND #mask
            - Result in A is 0 if flag clear, non-zero if flag set
        """
        from r65.compiler.hir.status_flags import is_branchable_flag
        from r65.compiler.codegen.opcodes import Opcode

        if is_branchable_flag(instr.flag_name):
            # No-op for branchable flags - branch instruction handles it
            self.emitter.emit_comment(f"StatusFlagTest {instr.flag_name} (direct branch)")
        else:
            # Non-branchable flag - need to test via PHP; PLA; AND #mask
            self.emitter.emit_comment(f"StatusFlagTest {instr.flag_name}")
            self.emitter.emit_instr(Opcode.PHP, comment="Push STATUS to stack")
            self.emitter.emit_instr(Opcode.PLA, comment="Pull STATUS into A")
            self.emitter.emit_instr(Opcode.AND_IMMEDIATE, instr.bit_mask,
                                    comment=f"Test {instr.flag_name} flag (bit {instr.bit_position})")

    def select_status_flag_set(self, instr: StatusFlagSet):
        """
        Generate code for StatusFlagSet instruction.

        Emits the appropriate instruction to set or clear a STATUS flag:
        - Carry: SEC / CLC
        - Irq: SEI / CLI
        - Decimal: SED / CLD
        - XY16: SEP #$10 / REP #$10 (accumulated for combining)
        - A16: SEP #$20 / REP #$20 (accumulated for combining)

        For A16/XY16 flags, masks are accumulated and emitted together when:
        - A non-StatusFlagSet instruction is encountered
        - The direction changes (set vs clear)
        """
        from r65.compiler.hir.status_flags import get_status_flag

        flag = get_status_flag(instr.flag_name)
        if not flag:
            raise unknown_value("STATUS flag", instr.flag_name)

        # For A16/XY16, accumulate masks for combining
        # Note: A16=true means "16-bit mode" which requires M flag=0 (REP clears)
        #       A16=false means "8-bit mode" which requires M flag=1 (SEP sets)
        if instr.flag_name in ('A16', 'XY16'):
            mask = M_FLAG if instr.flag_name == 'A16' else X_FLAG

            if instr.value:
                # A16/XY16 = true means 16-bit mode, use REP to clear the flag bit
                if self._pending_sep_mask:
                    self._flush_pending_mode_flags()
                self._pending_rep_mask |= mask
            else:
                # A16/XY16 = false means 8-bit mode, use SEP to set the flag bit
                if self._pending_rep_mask:
                    self._flush_pending_mode_flags()
                self._pending_sep_mask |= mask
            return

        # For other flags, emit immediately (and flush pending mode flags first)
        self._flush_pending_mode_flags()

        action = "Set" if instr.value else "Clear"
        self.emitter.emit_comment(f"StatusFlag{action} {instr.flag_name}")

        if instr.value:
            # Set the flag
            if instr.flag_name == 'Carry':
                self.emitter.emit_instr(Opcode.SEC, comment="Set Carry flag")
            elif instr.flag_name == 'Irq':
                self.emitter.emit_instr(Opcode.SEI, comment="Set Interrupt disable flag")
            elif instr.flag_name == 'Decimal':
                self.emitter.emit_instr(Opcode.SED, comment="Set Decimal mode flag")
        else:
            # Clear the flag
            if instr.flag_name == 'Carry':
                self.emitter.emit_instr(Opcode.CLC, comment="Clear Carry flag")
            elif instr.flag_name == 'Irq':
                self.emitter.emit_instr(Opcode.CLI, comment="Clear Interrupt disable flag")
            elif instr.flag_name == 'Decimal':
                self.emitter.emit_instr(Opcode.CLD, comment="Clear Decimal mode flag")

    def select_status_flag_read(self, instr: StatusFlagRead):
        """
        Generate code for StatusFlagRead instruction.

        Reads a STATUS flag into a virtual register as boolean (0/1).
        Emits: PHP; PLA; AND #mask; (normalize to 0/1 if needed)
        """
        from r65.compiler.codegen.opcodes import Opcode

        self.emitter.emit_comment(f"StatusFlagRead {instr.flag_name}")

        # Get STATUS into A
        self.emitter.emit_instr(Opcode.PHP, comment="Push STATUS to stack")
        self.emitter.emit_instr(Opcode.PLA, comment="Pull STATUS into A")
        self.emitter.emit_instr(Opcode.AND_IMMEDIATE, instr.bit_mask,
                                comment=f"Isolate {instr.flag_name} flag")

        # Normalize to 0/1 (result is 0 if flag clear, or bit_mask if flag set)
        # For bit 0 (Carry), result is already 0 or 1
        # For other bits, we need to normalize
        if instr.bit_mask != 0x01:
            # Use BEQ to skip if already 0, otherwise load 1
            norm_label = self._get_unique_label()
            self.emitter.emit_instr(Opcode.BEQ, norm_label, comment="Skip if flag clear")
            self.emitter.emit_instr(Opcode.LDA_IMMEDIATE, 1, comment="Normalize to 1")
            self.emitter.emit_label(norm_label)

        # Store result to destination
        dest_loc = self._get_operand_location(instr.dest)
        self._emit_store('STA', dest_loc, comment=f"Store {instr.flag_name} flag value")
