"""
Call instruction selector: Function calls and built-ins.

Handles function call generation including argument setup, call emission,
return value collection, and built-in function expansion.

Includes hardware register spill/reload around calls based on liveness
and callee's #[preserves()] attribute.

Region-based spilling (Phase 2):
Instead of spilling around each call, we identify "clobber regions" -
maximal sequences of calls where a register is live but not used. We save
once at the start of the region and restore once at the end, reducing
redundant push/pull operations.
"""

from typing import List, Set, Dict, NamedTuple, Optional, TYPE_CHECKING
from r65.compiler.mir.nodes import Call, VirtualRegister, ArgumentMechanism, Immediate as MIRImmediate

if TYPE_CHECKING:
    from r65.compiler.mir.liveness import ClobberRegion
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.codegen.opcodes import (
    Opcode, TRANSFER_OPCODES, PUSH_OPCODES, PULL_OPCODES,
    LOAD_IMMEDIATE_OPCODES, STORE_MNEMONICS, BUILTIN_OPCODES
)
from r65.compiler.codegen.asm_nodes import BlockMove
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.errors import (
    unknown_value, argument_count_error, requires_constant
)
from r65.compiler.codegen.base_selector import BaseSelector


class SpillInfo(NamedTuple):
    """Information about a hardware register that needs spilling."""
    vreg: Optional[VirtualRegister]  # The virtual register allocated to the hw reg, or None for direct hw reg usage
    hw_reg: str            # Hardware register name ('A', 'X', 'Y')
    # Stack spill location will be managed via push/pull


class ActiveRegionState:
    """
    Tracks active clobber regions during code generation.

    For each hardware register (X, Y), tracks whether we're currently inside
    a clobber region (have saved but not yet restored).

    Also tracks the current "spill offset" - the number of bytes pushed onto
    the stack for spilling. This is used to adjust stack-relative accesses
    to local variables while spills are active.
    """

    def __init__(self):
        # Map from hw_reg ('X', 'Y') to the ClobberRegion we're inside, or None
        self.active_regions: Dict[str, 'ClobberRegion'] = {}
        # Pre-computed regions for current block: block_id -> {hw_reg -> [ClobberRegion]}
        self.block_regions: Dict[int, Dict[str, List['ClobberRegion']]] = {}
        # Current block ID
        self.current_block_id: Optional[int] = None
        # Current spill offset - bytes pushed onto stack for spilling
        # Used to adjust stack-relative accesses while spills are active
        self.spill_offset: int = 0

    def set_block_regions(self, block_id: int, regions: Dict[str, List['ClobberRegion']]):
        """Set pre-computed regions for a block."""
        self.block_regions[block_id] = regions
        self.current_block_id = block_id
        # Clear active regions when entering new block
        self.active_regions.clear()
        # Reset spill offset for new block
        self.spill_offset = 0

    def get_region_for_call(self, hw_reg: str, call_idx: int) -> Optional['ClobberRegion']:
        """
        Get the clobber region that contains this call for the given register.

        Args:
            hw_reg: Hardware register ('X' or 'Y')
            call_idx: Instruction index of the call

        Returns:
            ClobberRegion if call is in a region, None otherwise
        """
        if self.current_block_id is None:
            return None

        regions = self.block_regions.get(self.current_block_id, {}).get(hw_reg, [])
        for region in regions:
            if call_idx in region.clobbering_calls:
                return region
        return None

    def is_first_call_in_region(self, hw_reg: str, call_idx: int) -> bool:
        """Check if this call is the first clobbering call in its region."""
        region = self.get_region_for_call(hw_reg, call_idx)
        if region is None:
            return False
        return region.clobbering_calls[0] == call_idx

    def is_last_call_in_region(self, hw_reg: str, call_idx: int) -> bool:
        """Check if this call is the last clobbering call in its region."""
        region = self.get_region_for_call(hw_reg, call_idx)
        if region is None:
            return False
        return region.clobbering_calls[-1] == call_idx

    def mark_region_active(self, hw_reg: str, region: 'ClobberRegion'):
        """Mark that we've saved for this region (entered it)."""
        self.active_regions[hw_reg] = region

    def is_region_active(self, hw_reg: str) -> bool:
        """Check if we're inside an active region for this register."""
        return hw_reg in self.active_regions

    def get_active_region(self, hw_reg: str) -> Optional['ClobberRegion']:
        """Get the active region for this register, if any."""
        return self.active_regions.get(hw_reg)

    def clear_active_region(self, hw_reg: str):
        """Mark that we've restored for this region (exited it)."""
        if hw_reg in self.active_regions:
            del self.active_regions[hw_reg]


class CallInstructionSelector(BaseSelector):
    """
    Handles call instruction selection.

    Manages generation of function calls, built-in expansions,
    and indirect call trampolines.

    Uses region-based spilling (Phase 2): instead of spilling around each call,
    identifies "clobber regions" and saves once at region start, restores once
    at region end.
    """

    def __init__(self, parent):
        """Initialize call instruction selector with region tracking state."""
        super().__init__(parent)
        # Region tracking state for optimized spilling
        self._region_state = ActiveRegionState()
        # Pre-computed regions for current function: {block_id: {hw_reg: [ClobberRegion]}}
        self._function_regions: Optional[Dict[int, Dict[str, List]]] = None

    def initialize_regions_for_function(self):
        """
        Pre-compute clobber regions for all blocks in the current function.

        Should be called once per function before processing any blocks.
        """
        reg_alloc = self.parent.reg_alloc
        if not reg_alloc or not reg_alloc.instr_liveness:
            self._function_regions = None
            return

        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        # Build preserves map from function signatures in the current function
        preserves_map = self._build_preserves_map()

        analyzer = ClobberRegionAnalyzer(reg_alloc.instr_liveness)
        self._function_regions = analyzer.analyze_function(preserves_map)

    def initialize_regions_for_block(self, block_id: int):
        """
        Initialize region tracking state for a specific block.

        Should be called when starting to process a new block.

        Args:
            block_id: The block ID being processed
        """
        if self._function_regions is None:
            # No pre-computed regions, will fall back to per-call spilling
            self._region_state = ActiveRegionState()
            return

        regions = self._function_regions.get(block_id, {'X': [], 'Y': []})
        self._region_state.set_block_regions(block_id, regions)

    def _build_preserves_map(self) -> Dict[str, Set[str]]:
        """
        Build a map from function names to their preserved registers.

        Returns:
            Dictionary mapping function name to set of preserved register names
        """
        preserves_map: Dict[str, Set[str]] = {}

        # Scan all call instructions in the function to build the map
        if self.parent.current_function:
            for block in self.parent.current_function.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call) and isinstance(instr.function, str):
                        if instr.preserves_attr:
                            func_name = instr.function
                            preserves_map[func_name] = set(instr.preserves_attr.registers)

        return preserves_map

    # ========================================================================
    # Call-Specific Emission Helpers
    # ========================================================================

    def _emit_transfer(self, source: str, dest: str):
        """Emit a register transfer instruction."""
        opcode = TRANSFER_OPCODES.get((source, dest))
        if opcode:
            self._emit_implied(opcode)

    def _emit_push(self, reg: str, comment: str = None):
        """Emit a push instruction."""
        opcode = PUSH_OPCODES.get(reg)
        if opcode:
            self._emit_implied(opcode, comment)

    def _emit_pull(self, reg: str, comment: str = None):
        """Emit a pull instruction."""
        opcode = PULL_OPCODES.get(reg)
        if opcode:
            self._emit_implied(opcode, comment)

    def _emit_load_immediate(self, reg: str, value: int, comment: str = None):
        """Emit a load immediate instruction."""
        opcode = LOAD_IMMEDIATE_OPCODES.get(reg)
        if opcode:
            self._emit_immediate(opcode, value, comment)

    def _ensure_m8_mode(self, comment: str = "8-bit A"):
        """Switch to 8-bit accumulator mode if not already there."""
        from r65.compiler.codegen.constants import M_FLAG
        if self.parent.emitter.get_accu_mode() != 8:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, comment)
            self.parent.emitter.emit_accu_mode(8)

    def _push_multibyte_value(self, arg_loc, source_size: int, target_size: int, is_stack: bool):
        """
        Push a multi-byte value to stack with zero-extension as needed.

        Handles stack drift compensation for stack-based source locations.
        Bytes are pushed high-to-low (bank first for 24-bit, high first for 16-bit).

        Args:
            arg_loc: Source location
            source_size: Size of source value in bytes
            target_size: Size of target parameter in bytes
            is_stack: True if arg_loc is on stack (needs drift compensation)
        """
        push_count = 0  # Track pushes for stack drift compensation

        # Push bytes from high to low
        for byte_idx in range(target_size - 1, -1, -1):
            if byte_idx >= source_size:
                # Zero-extend: push 0 for bytes beyond source size
                byte_name = {2: "bank", 1: "high", 0: "low"}.get(byte_idx, f"byte{byte_idx}")
                self._emit_load_immediate('A', 0, f"Zero-extend {byte_name} byte")
                self._emit_push('A', f"Push {byte_name} byte (zero)")
            else:
                # Load from source location
                byte_name = {2: "bank", 1: "high", 0: "low"}.get(byte_idx, f"byte{byte_idx}")
                offset = byte_idx
                if is_stack:
                    offset += push_count  # Compensate for stack drift
                if offset > 0:
                    loc = self.parent._offset_location(arg_loc, offset)
                else:
                    loc = arg_loc
                self.parent._emit_load('LDA', loc, f"Load {byte_name} byte")
                self._emit_push('A', f"Push {byte_name} byte")
            push_count += 1

    # ========================================================================
    # Main Call Selection
    # ========================================================================

    def select_call(self, instr: Call):
        """
        Generate code for Call instruction.

        Handles argument setup, call, and return value collection.
        Also handles built-in function calls.

        Cross-mode call handling:
        - Before call: switch to callee's entry mode if different from current
        - After call: caller receives return value in callee's exit mode
        - After using return value: switch back to m8 (default mode)

        Hardware register spilling:
        - Before call: spill hw registers that are live across call and not preserved
        - After call: reload those registers

        Args:
            instr: Call instruction
        """
        # Handle built-in functions
        if instr.builtin_name:
            self._emit_builtin_call(instr)
            return

        # Check if we need D register management for far pointer params
        needs_d_management = (
            self.parent.current_function and
            self.parent.current_function.has_far_ptr_stack_params
        )

        # Step 0: Compute hardware register spills needed
        # Spill hw registers that are live across this call and not preserved by callee
        spills = self._compute_hw_spills(instr)

        # Step 0.5: Emit spills BEFORE argument setup (to avoid clobbering args)
        self._emit_hw_spills(spills)

        # Step 1: Set up arguments
        stack_bytes_pushed = self._emit_argument_setup(instr)

        # Step 2: Handle caller-managed DBR (databank=caller)
        needs_dbr_restore = self._emit_caller_dbr_setup(instr)

        # Step 2.5: Restore D register before call (for far pointer functions)
        # The called function may use zeropage, so D must be restored to original value
        if needs_d_management:
            self._emit_d_restore_before_call()

        # Step 2.6: Switch to callee's entry mode if needed
        # Callee's prologue expects to be in entry_m_mode
        self._emit_entry_mode_switch(instr)

        # Step 3: Make the call
        self._emit_call_instruction(instr)

        # Invalidate XBA state after call (function may have modified A/B)
        self.parent._invalidate_xba_state()

        # Step 3.5: Update emitter mode to reflect callee's exit mode
        # This is critical - the callee may exit in a different mode than entry
        self._update_mode_after_call(instr)

        # Step 4: Restore DBR if caller-managed
        if needs_dbr_restore:
            self._emit_pull('B', "Restore data bank (caller)")

        # Step 5: Stack arguments are cleaned up by callee (not caller)
        # The callee's epilogue handles stack parameter cleanup before RTS/RTL

        # Step 6: Collect return values (in callee's exit mode)
        self._emit_return_value_collection(instr)

        # Step 7: Restore mode after receiving return value
        # If callee exited in m16 (u16 return), switch back to m8
        self._emit_exit_mode_restore(instr)

        # Step 7.5: Reload spilled hardware registers (region-based)
        # Only reload registers where this is the last call in the region
        reloads = self._compute_hw_reloads(instr)
        self._emit_hw_reloads(reloads)

        # Step 8: Restore D to stack and optionally re-establish D = S
        if needs_d_management:
            # We used PLD before the call, so we must push D back
            # If there are more far pointer dereferences, also set D = S
            if self._has_far_ptr_derefs_after_call(instr):
                self._emit_d_equals_s_restore()  # PHD + TSC + TCD
            else:
                self._emit_d_push_only()  # Just PHD (for epilogue)

    # ========================================================================
    # Hardware Register Spill/Reload (Region-Based)
    # ========================================================================

    def _compute_hw_spills(self, instr: Call) -> List[SpillInfo]:
        """
        Compute which hardware registers need spilling for this call.

        Uses region-based analysis when available: only returns registers that
        need to START a new region at this call. Registers that are already
        in an active region (saved earlier) are not included.

        Falls back to per-call analysis when region analysis is unavailable.

        Args:
            instr: The Call instruction

        Returns:
            List of SpillInfo for registers that need spilling at THIS call
        """
        spills: List[SpillInfo] = []

        # Get callee's preserved registers
        preserved: Set[str] = set()
        if instr.preserves_attr:
            preserved = set(instr.preserves_attr.registers)

        # Check each hardware register
        reg_alloc = self.parent.reg_alloc
        if not reg_alloc:
            return spills

        # Get instruction position for region lookup
        instr_idx = None
        if reg_alloc.instr_liveness:
            pos = reg_alloc.instr_liveness.get_instruction_position(instr)
            if pos:
                _, instr_idx = pos

        for reg_name in ['A', 'X', 'Y']:
            # Skip if callee preserves this register
            if reg_name in preserved:
                continue

            # For X/Y, use region-based spilling if available
            if reg_name in ('X', 'Y') and instr_idx is not None and self._function_regions is not None:
                # Check if this call is in a region for this register
                region = self._region_state.get_region_for_call(reg_name, instr_idx)

                if region is not None:
                    # This call is in a clobber region
                    if self._region_state.is_first_call_in_region(reg_name, instr_idx):
                        # First call in region - need to spill
                        spills.append(SpillInfo(vreg=None, hw_reg=reg_name))
                        # Mark region as active
                        self._region_state.mark_region_active(reg_name, region)
                    # Else: already in active region, no spill needed
                    continue

            # For A register or when region analysis unavailable, use per-call analysis
            hw_alloc = reg_alloc.get_hw_alloc(reg_name)
            vreg = hw_alloc.allocated_vreg

            # Case 1: Vreg allocated to this hw register
            if vreg is not None:
                # Check if the vreg is live after this call
                if reg_alloc.instr_liveness:
                    pos = reg_alloc.instr_liveness.get_instruction_position(instr)
                    if pos:
                        block_id, idx = pos
                        if reg_alloc.instr_liveness.is_live_after(vreg, block_id, idx):
                            spills.append(SpillInfo(vreg=vreg, hw_reg=reg_name))
                else:
                    # Conservative: assume live if we have an allocation
                    if hw_alloc.is_bound:
                        spills.append(SpillInfo(vreg=vreg, hw_reg=reg_name))
                continue

            # Case 2: Direct hardware register usage (X and Y only, not A)
            # For X/Y without region info, fall back to per-call
            if reg_name in ('X', 'Y') and reg_alloc.instr_liveness and self._function_regions is None:
                pos = reg_alloc.instr_liveness.get_instruction_position(instr)
                if pos:
                    block_id, idx = pos
                    if reg_alloc.instr_liveness.is_hw_reg_live_after(reg_name, block_id, idx):
                        spills.append(SpillInfo(vreg=None, hw_reg=reg_name))

        return spills

    def _compute_hw_reloads(self, instr: Call) -> List[SpillInfo]:
        """
        Compute which hardware registers need reloading after this call.

        Uses region-based analysis: only returns registers where this call
        is the LAST clobbering call in the region.

        Falls back to returning all spilled registers when region analysis
        is unavailable (per-call spilling behavior).

        Args:
            instr: The Call instruction

        Returns:
            List of SpillInfo for registers that need reloading after THIS call
        """
        reloads: List[SpillInfo] = []

        reg_alloc = self.parent.reg_alloc
        if not reg_alloc or not reg_alloc.instr_liveness:
            return reloads

        # Get instruction position
        pos = reg_alloc.instr_liveness.get_instruction_position(instr)
        if not pos:
            return reloads

        _, instr_idx = pos

        # Check each active region
        for hw_reg in ('X', 'Y'):
            if self._region_state.is_region_active(hw_reg):
                if self._region_state.is_last_call_in_region(hw_reg, instr_idx):
                    # Last call in region - need to reload
                    reloads.append(SpillInfo(vreg=None, hw_reg=hw_reg))
                    # Clear active region
                    self._region_state.clear_active_region(hw_reg)

        return reloads

    def _emit_hw_spills(self, spills: List[SpillInfo]):
        """
        Emit push instructions to spill hardware registers.

        For region-based spilling, this is only called at region start.
        Updates the spill offset to track stack growth for adjusting
        stack-relative accesses.

        Args:
            spills: List of registers to spill
        """
        for spill in spills:
            self._emit_push(spill.hw_reg, f"Spill {spill.hw_reg} (region start)")
            # Track stack growth: X/Y are 16-bit (2 bytes), A is 8-bit (1 byte)
            if spill.hw_reg in ('X', 'Y'):
                self._region_state.spill_offset += 2
            else:
                # A register size depends on mode, but typically 1 byte in m8
                self._region_state.spill_offset += 1

    def _emit_hw_reloads(self, spills: List[SpillInfo]):
        """
        Emit pull instructions to reload spilled hardware registers.

        Must be called in reverse order of spills due to stack behavior (LIFO).
        Updates the spill offset to track stack shrinkage.

        For region-based spilling, this is only called at region end.

        Args:
            spills: List of registers to reload (will be processed in reverse)
        """
        for spill in reversed(spills):
            self._emit_pull(spill.hw_reg, f"Reload {spill.hw_reg} (region end)")
            # Track stack shrinkage: X/Y are 16-bit (2 bytes), A is 8-bit (1 byte)
            if spill.hw_reg in ('X', 'Y'):
                self._region_state.spill_offset -= 2
            else:
                self._region_state.spill_offset -= 1

    def get_current_spill_offset(self) -> int:
        """
        Get the current spill offset for adjusting stack-relative accesses.

        Returns:
            Number of bytes currently pushed for spilling
        """
        return self._region_state.spill_offset

    # ========================================================================
    # Argument Setup
    # ========================================================================

    def _emit_argument_setup(self, instr: Call) -> int:
        """
        Set up call arguments in correct order.

        Process in specific order to avoid clobbering:
        1. Stack arguments (pushed in REVERSE order - last param first per calling convention)
        2. Variable-bound arguments
        3. B register arguments (these clobber A via XBA)
        4. X and Y register arguments
        5. A register arguments (set up last to avoid being clobbered)

        Args:
            instr: Call instruction

        Returns:
            Number of bytes pushed on stack (for cleanup)
        """
        from r65.compiler.codegen.type_utils import get_type_size

        stack_bytes_pushed = 0

        # Separate stack arguments from others and reverse them
        # Stack params must be pushed right-to-left (last param first) per calling convention
        stack_args = [arg for arg in instr.args if arg.mechanism == ArgumentMechanism.STACK]
        other_args = [arg for arg in instr.args if arg.mechanism != ArgumentMechanism.STACK]

        # Push stack arguments in reverse order (last param first)
        for arg in reversed(stack_args):
            arg_loc = self.parent._get_operand_location(arg.value)

            # CRITICAL: Adjust stack-relative source locations for bytes already pushed
            # by previous arguments. Each push shifts the stack pointer, so source
            # locations that were computed relative to the original SP need adjustment.
            if arg_loc.kind == LocationKind.STACK and stack_bytes_pushed > 0:
                arg_loc = self.parent._offset_location(arg_loc, stack_bytes_pushed)

            self._emit_stack_argument(arg, arg_loc)
            # Track bytes pushed based on argument size - prefer param_type
            arg_size = 1
            if arg.param_type is not None:
                arg_size = get_type_size(arg.param_type)
            elif hasattr(arg.value, 'type_info') and arg.value.type_info:
                arg_size = get_type_size(arg.value.type_info)
            stack_bytes_pushed += arg_size

        # Process other arguments (variable-bound and register) in sorted order
        sorted_other_args = sorted(other_args, key=self._arg_sort_key)
        for arg in sorted_other_args:
            arg_loc = self.parent._get_operand_location(arg.value)

            # CRITICAL: Adjust stack-relative source locations for bytes already pushed
            # by stack arguments. Register and variable-bound args loading from stack
            # need this adjustment too.
            if arg_loc.kind == LocationKind.STACK and stack_bytes_pushed > 0:
                arg_loc = self.parent._offset_location(arg_loc, stack_bytes_pushed)

            if arg.mechanism == ArgumentMechanism.REGISTER:
                self._emit_register_argument(arg, arg_loc)

            elif arg.mechanism == ArgumentMechanism.VARIABLE:
                self._emit_variable_argument(arg, arg_loc)

        return stack_bytes_pushed

    def _arg_sort_key(self, arg):
        """Sort key for argument processing order."""
        if arg.mechanism == ArgumentMechanism.STACK:
            return 0  # Stack first
        elif arg.mechanism == ArgumentMechanism.VARIABLE:
            return 1  # Variable-bound second
        elif arg.mechanism == ArgumentMechanism.REGISTER:
            target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)
            if target_reg == 'B':
                return 2  # B third (clobbers A)
            elif target_reg in ['X', 'Y']:
                return 3  # X, Y fourth
            elif target_reg == 'A':
                return 4  # A last (to avoid being clobbered)
        return 5

    def _emit_stack_argument(self, arg, arg_loc):
        """Emit stack argument (push onto stack).

        IMPORTANT: This method ensures correct accumulator mode for pushing.
        For 8-bit arguments, we must be in m8 mode so PHA pushes 1 byte.
        For 16-bit arguments loaded byte-by-byte, we use m8 mode.
        For 16-bit arguments already in A, we use m16 mode (single PHA).

        IMPORTANT: When source value is smaller than parameter type, we must
        zero-extend. For example, an 8-bit loop variable passed to a u16
        parameter needs its high byte set to 0, not loaded from garbage memory.
        """
        from r65.compiler.codegen.type_utils import get_type_size
        from r65.compiler.codegen.constants import M_FLAG

        # Determine PARAMETER size (what we need to push)
        param_size = 1
        if arg.param_type is not None:
            param_size = get_type_size(arg.param_type)
        elif hasattr(arg.value, 'type_info') and arg.value.type_info:
            param_size = get_type_size(arg.value.type_info)
        elif isinstance(arg.value, MIRImmediate):
            # Fallback: infer size from immediate value range
            value = arg.value.value
            if value > 0xFFFF or value < -32768:
                param_size = 3  # 24-bit
            elif value > 0xFF or value < -128:
                param_size = 2  # 16-bit
            # else: 8-bit (default)

        # Determine SOURCE size (what we're loading from)
        # This is critical for proper zero-extension
        source_size = param_size  # Default to param size
        if hasattr(arg.value, 'type_info') and arg.value.type_info:
            source_size = get_type_size(arg.value.type_info)
        elif isinstance(arg.value, MIRImmediate):
            # Immediates are already the right size (computed above)
            source_size = param_size

        # Use param_size for how many bytes to push (arg_size variable for compatibility)
        arg_size = param_size

        # Track current mode for proper restoration
        current_mode = self.parent.emitter.get_accu_mode()

        if arg_size == 3:
            # 24-bit value (far pointer): push all 3 bytes (bank, high, low order for stack)
            # IMPORTANT: Must be in m8 mode so each PHA pushes exactly 1 byte
            self._ensure_m8_mode("8-bit A for byte push")

            if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
                raise InstructionSelectionError("Cannot push 24-bit value from A register")
            else:
                is_stack = arg_loc.kind == LocationKind.STACK
                self._push_multibyte_value(arg_loc, source_size, 3, is_stack)

        elif arg_size == 2:
            # 16-bit value: push both bytes (high first, then low)
            if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
                # Value is in A - use 16-bit push if in m16, otherwise handle specially
                if current_mode == 16:
                    self._emit_push('A', "Push 16-bit stack arg")
                else:
                    # In m8, need to switch to m16 for single 16-bit push
                    self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for u16 push")
                    self.parent.emitter.emit_accu_mode(16)
                    self._emit_push('A', "Push 16-bit stack arg")
            elif isinstance(arg.value, MIRImmediate):
                # Immediate 16-bit value: push byte by byte in m8 mode
                self._ensure_m8_mode("8-bit A for byte push")
                value = arg.value.value
                high_byte = (value >> 8) & 0xFF
                low_byte = value & 0xFF
                self._emit_load_immediate('A', high_byte)
                self._emit_push('A', "Push high byte")
                self._emit_load_immediate('A', low_byte)
                self._emit_push('A', "Push low byte")
            else:
                # Memory locations: push byte by byte with zero-extension as needed
                self._ensure_m8_mode("8-bit A for byte push")
                is_stack = arg_loc.kind == LocationKind.STACK
                self._push_multibyte_value(arg_loc, source_size, 2, is_stack)

        else:
            # 8-bit value: single push
            # IMPORTANT: Must be in m8 mode so PHA pushes exactly 1 byte
            self._ensure_m8_mode("8-bit A for u8 push")

            if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
                pass  # Already in A
            elif arg_loc.kind == LocationKind.HARDWARE:
                if arg_loc.hw_register == 'X':
                    self._emit_transfer('X', 'A')
                elif arg_loc.hw_register == 'Y':
                    self._emit_transfer('Y', 'A')
            elif isinstance(arg.value, MIRImmediate):
                self._emit_load_immediate('A', arg.value.value)
            else:
                self.parent._emit_load('LDA', arg_loc)

            self._emit_push('A', "Push stack arg")

    def _emit_register_argument(self, arg, arg_loc):
        """Emit register argument (move to specified register)."""
        target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)

        if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == target_reg:
            return  # Already in correct register

        if target_reg == 'B':
            self._emit_b_register_argument(arg, arg_loc)
        elif isinstance(arg.value, MIRImmediate):
            self._emit_immediate_to_register(arg.value.value, target_reg)
        elif arg_loc.kind == LocationKind.HARDWARE:
            # Source is a hardware register - emit transfer
            self._emit_hw_to_register(arg_loc.hw_register, target_reg)
        else:
            self._emit_memory_to_register(arg_loc, target_reg)

    def _emit_b_register_argument(self, arg, arg_loc):
        """Emit B register argument (special handling)."""
        if isinstance(arg.value, MIRImmediate):
            self._emit_load_immediate('A', arg.value.value)
            self.parent._store_to_b_from_a()
        elif arg_loc.kind == LocationKind.HARDWARE:
            if arg_loc.hw_register != 'A':
                self.parent._emit_register_transfer(arg_loc.hw_register, 'A')
            self.parent._store_to_b_from_a()
        else:
            self.parent._emit_load('LDA', arg_loc)
            self.parent._store_to_b_from_a()

    def _emit_immediate_to_register(self, value: int, target_reg: str):
        """Load immediate value into target register."""
        self._emit_load_immediate(target_reg, value)

    def _emit_memory_to_register(self, arg_loc, target_reg: str):
        """Load from memory into target register."""
        # Handle stack-relative addressing: LDX/LDY don't support sr,S mode
        if target_reg in ('X', 'Y') and arg_loc.kind == LocationKind.STACK:
            self.parent._emit_load('LDA', arg_loc)
            if target_reg == 'X':
                self._emit_implied(Opcode.TAX, "Transfer to X (no LDX sr,S)")
            else:
                self._emit_implied(Opcode.TAY, "Transfer to Y (no LDY sr,S)")
        else:
            mnemonic = {'A': 'LDA', 'X': 'LDX', 'Y': 'LDY'}.get(target_reg)
            if mnemonic:
                self.parent._emit_load(mnemonic, arg_loc)

    def _emit_hw_to_register(self, src_reg: str, target_reg: str):
        """Transfer from one hardware register to another."""
        if src_reg == target_reg:
            return  # Nothing to do

        # Use appropriate transfer instruction
        if src_reg == 'A' and target_reg == 'X':
            self._emit_implied(Opcode.TAX, f"Transfer A to X")
        elif src_reg == 'A' and target_reg == 'Y':
            self._emit_implied(Opcode.TAY, f"Transfer A to Y")
        elif src_reg == 'X' and target_reg == 'A':
            self._emit_implied(Opcode.TXA, f"Transfer X to A")
        elif src_reg == 'X' and target_reg == 'Y':
            self._emit_implied(Opcode.TXY, f"Transfer X to Y")
        elif src_reg == 'Y' and target_reg == 'A':
            self._emit_implied(Opcode.TYA, f"Transfer Y to A")
        elif src_reg == 'Y' and target_reg == 'X':
            self._emit_implied(Opcode.TYX, f"Transfer Y to X")
        else:
            raise InstructionSelectionError(
                f"Cannot transfer from {src_reg} to {target_reg}")

    def _emit_variable_argument(self, arg, arg_loc):
        """Emit variable-bound argument (store to memory location)."""
        # Load into A
        if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
            pass  # Already in A
        elif arg_loc.kind == LocationKind.HARDWARE:
            if arg_loc.hw_register == 'X':
                self._emit_transfer('X', 'A')
            elif arg_loc.hw_register == 'Y':
                self._emit_transfer('Y', 'A')
        elif isinstance(arg.value, MIRImmediate):
            self._emit_load_immediate('A', arg.value.value)
        else:
            self.parent._emit_load('LDA', arg_loc)

        # Store to variable location
        var_loc = self.parent._get_operand_location(arg.location)
        self.parent._emit_store('STA', var_loc)

    # ========================================================================
    # DBR Management
    # ========================================================================

    def _emit_caller_dbr_setup(self, instr: Call) -> bool:
        """
        Handle caller-managed DBR setup if needed.

        Args:
            instr: Call instruction

        Returns:
            True if DBR restore needed after call
        """
        if not (instr.is_far and instr.mode_attr and instr.bank_attr):
            return False

        from r65.compiler.hir.attributes import DataBankMode

        if instr.mode_attr.databank != DataBankMode.CALLER:
            return False

        # Caller manages DBR: save, set, call, restore
        self._emit_push('B', "Save current data bank (caller)")
        self._emit_load_immediate('A', instr.bank_attr.bank_number, "Load callee's bank number")
        self._emit_push('A', "Push bank number")
        self._emit_pull('B', "Set data bank for callee")
        return True

    # ========================================================================
    # Cross-Mode Call Handling
    # ========================================================================

    def _emit_entry_mode_switch(self, instr: Call):
        """
        Switch to callee's entry mode before making the call.

        The callee expects to receive arguments in its entry mode.
        Default caller mode is m8, so we only need to switch if callee expects m16.

        Args:
            instr: Call instruction with callee mode info
        """
        from r65.compiler.typeck.processor_mode import ModeState

        callee_entry = instr.callee_entry_m_mode
        if callee_entry is None:
            return  # No mode info (indirect call or unknown), assume compatible

        # Get current mode from emitter
        current_mode_bits = self.emitter.get_accu_mode()
        current_is_m16 = (current_mode_bits == 16)
        callee_wants_m16 = (callee_entry == ModeState.M16)

        if not current_is_m16 and callee_wants_m16:
            # Caller is m8, callee wants m16 - switch to m16
            self._emit_immediate(Opcode.REP_IMMEDIATE, 0x20, "Switch to m16 for callee")
            self.emitter.emit_accu_mode(16)
        elif current_is_m16 and not callee_wants_m16:
            # Caller is m16, callee wants m8 - switch to m8
            self._emit_immediate(Opcode.SEP_IMMEDIATE, 0x20, "Switch to m8 for callee")
            self.emitter.emit_accu_mode(8)
        # Otherwise modes match, no switch needed

    def _emit_exit_mode_restore(self, instr: Call):
        """
        Restore mode after receiving return value from callee.

        After the call, the caller is in callee's exit mode.
        If callee exited in m16 (u16 return), switch back to m8 (default mode).

        This ensures the caller continues in the expected default mode.

        Args:
            instr: Call instruction with callee mode info
        """
        from r65.compiler.typeck.processor_mode import ModeState

        callee_exit = instr.callee_exit_m_mode
        if callee_exit is None:
            return  # No mode info (indirect call or unknown), assume m8

        if callee_exit == ModeState.M16:
            # Callee returned in m16 mode, switch back to m8
            self._emit_immediate(Opcode.SEP_IMMEDIATE, 0x20, "Restore m8 after u16 return")
            self.emitter.emit_accu_mode(8)
        # If callee exited in m8, we're already in the right mode

    def _update_mode_after_call(self, instr: Call):
        """
        Update emitter's mode tracking to reflect callee's exit mode.

        After the call returns, the CPU is in the callee's exit mode (not entry mode).
        This updates the emitter's tracking without emitting any instructions, so that
        subsequent code (like return value collection) uses the correct mode.

        Args:
            instr: Call instruction with callee mode info
        """
        from r65.compiler.typeck.processor_mode import ModeState

        callee_exit = instr.callee_exit_m_mode
        if callee_exit is None:
            return  # No mode info, assume unchanged

        if callee_exit == ModeState.M16:
            self.emitter.emit_accu_mode(16)
        else:
            self.emitter.emit_accu_mode(8)

    # ========================================================================
    # Call Emission
    # ========================================================================

    def _emit_call_instruction(self, instr: Call):
        """
        Emit the actual call instruction.

        Handles both direct and indirect calls.

        Args:
            instr: Call instruction
        """
        if isinstance(instr.function, VirtualRegister):
            # Indirect call through function pointer
            self._emit_indirect_call_trampoline(instr.function, instr.is_far)
        elif isinstance(instr.function, str):
            # Direct call
            if instr.is_far:
                self._emit_address(Opcode.JSL, instr.function)
            else:
                self._emit_address(Opcode.JSR, instr.function)
        else:
            raise InstructionSelectionError(f"Unknown function type in Call: {type(instr.function)}")

    def _emit_indirect_call_trampoline(self, func_ptr_vreg: VirtualRegister, is_far: bool):
        """
        Generate trampoline for indirect function call through function pointer.

        Near trampoline (16-bit address):
            LDA func_ptr+1  ; High byte
            PHA
            LDA func_ptr    ; Low byte
            PHA
            RTS             ; Jumps to address on stack

        Far trampoline (24-bit address):
            LDA func_ptr+2  ; Bank byte
            PHA
            LDA func_ptr+1  ; High byte
            PHA
            LDA func_ptr    ; Low byte
            PHA
            RTL             ; Long return

        Note: For stack-relative locations, each PHA changes the stack pointer,
        so subsequent loads need their offsets adjusted by +1 for each previous PHA.

        Args:
            func_ptr_vreg: VirtualRegister holding the function pointer
            is_far: True for far call (24-bit), False for near call (16-bit)
        """
        ptr_loc = self.parent._get_operand_location(func_ptr_vreg)
        is_stack = ptr_loc.kind == LocationKind.STACK

        if is_far:
            # Far call trampoline (24-bit address)
            # Each PHA shifts subsequent stack-relative offsets by +1
            bank_loc = self.parent._offset_location(ptr_loc, 2)
            self.parent._emit_load('LDA', bank_loc, "Load bank byte")
            self._emit_push('A', "Push bank")

            # After 1 PHA, stack offsets need +1 adjustment
            if is_stack:
                high_loc = self.parent._offset_location(ptr_loc, 1 + 1)  # +1 for PHA
            else:
                high_loc = self.parent._offset_location(ptr_loc, 1)
            self.parent._emit_load('LDA', high_loc, "Load high byte")
            self._emit_push('A', "Push high")

            # After 2 PHAs, stack offsets need +2 adjustment
            if is_stack:
                low_loc = self.parent._offset_location(ptr_loc, 0 + 2)  # +2 for 2 PHAs
            else:
                low_loc = ptr_loc
            self.parent._emit_load('LDA', low_loc, "Load low byte")
            self._emit_push('A', "Push low")

            self._emit_implied(Opcode.RTL, "Indirect far call via trampoline")
        else:
            # Near call trampoline (16-bit address)
            high_loc = self.parent._offset_location(ptr_loc, 1)
            self.parent._emit_load('LDA', high_loc, "Load high byte")
            self._emit_push('A', "Push high")

            # After 1 PHA, stack offsets need +1 adjustment
            if is_stack:
                low_loc = self.parent._offset_location(ptr_loc, 0 + 1)  # +1 for PHA
            else:
                low_loc = ptr_loc
            self.parent._emit_load('LDA', low_loc, "Load low byte")
            self._emit_push('A', "Push low")

            self._emit_implied(Opcode.RTS, "Indirect near call via trampoline")

    # ========================================================================
    # Return Value Collection
    # ========================================================================

    def _emit_return_value_collection(self, instr: Call):
        """
        Collect return values from registers.

        Return values come back in A, X, Y (in order).
        """
        if not instr.returns:
            return

        return_registers = ['A', 'X', 'Y']

        for i, return_vreg in enumerate(instr.returns):
            if i >= len(return_registers):
                raise InstructionSelectionError(f"Too many return values (max {len(return_registers)})")

            source_reg = return_registers[i]
            dest_loc = self.parent._get_operand_location(return_vreg)

            if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == source_reg:
                pass  # Already in correct location
            elif dest_loc.kind == LocationKind.HARDWARE:
                self._emit_return_register_transfer(source_reg, dest_loc.hw_register)
            else:
                self._emit_return_store(source_reg, dest_loc)

    def _emit_return_register_transfer(self, source_reg: str, dest_reg: str):
        """Transfer return value between hardware registers."""
        self._emit_transfer(source_reg, dest_reg)

    def _emit_return_store(self, source_reg: str, dest_loc):
        """Store return value from register to memory."""
        mnemonic = STORE_MNEMONICS.get(source_reg)
        if mnemonic:
            self.parent._emit_store(mnemonic, dest_loc)

    # ========================================================================
    # Built-in Function Calls
    # ========================================================================

    def _emit_builtin_call(self, instr: Call):
        """
        Emit code for built-in function call.

        Built-in categories:
        - Processor control: wai(), stp(), xba(), NOP([count])
        - Mode control: SEP(flags), REP(flags)
        - Block moves: mvn(src_bank, dst_bank), mvp(src_bank, dst_bank)
        - Arithmetic: mul(a, b), div(a, b), mod(a, b) - call runtime library
        - Shifts: shl(a, n), shr(a, n) - call runtime library

        Args:
            instr: Call instruction with builtin_name set
        """
        from r65.compiler.builtins import BuiltinRegistry, BuiltinKind

        builtin = BuiltinRegistry.get_builtin(instr.builtin_name)
        if not builtin:
            raise unknown_value("built-in function", instr.builtin_name)

        if builtin.kind == BuiltinKind.PROCESSOR_CONTROL:
            self._emit_processor_control_builtin(instr, builtin)
        elif builtin.kind == BuiltinKind.SOFTWARE_INTERRUPT:
            self._emit_software_interrupt_builtin(instr, builtin)
        elif builtin.kind == BuiltinKind.BLOCK_MOVE:
            self._emit_block_move_builtin(instr, builtin)
        elif builtin.kind in (BuiltinKind.ARITHMETIC, BuiltinKind.SHIFT):
            self._emit_runtime_builtin(instr, builtin)

    def _emit_processor_control_builtin(self, instr: Call, builtin):
        """Emit processor control built-in (wai, stp, xba, NOP)."""
        opcode = BUILTIN_OPCODES.get(builtin.instruction)
        if not opcode:
            raise unknown_value("processor control builtin", builtin.instruction)

        if instr.builtin_name == 'NOP':
            count = 1  # Default
            if len(instr.args) == 1:
                arg = instr.args[0]
                if isinstance(arg.value, MIRImmediate):
                    count = arg.value.value
                else:
                    raise InstructionSelectionError("NOP() count must be a constant immediate value")

            for _ in range(count):
                self._emit_implied(opcode)
        else:
            self._emit_implied(opcode)

    def _emit_software_interrupt_builtin(self, instr: Call, builtin):
        """Emit software interrupt built-in (cop)."""
        if len(instr.args) != 1:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 1 argument, got {len(instr.args)}")

        opcode = BUILTIN_OPCODES.get(builtin.instruction)
        if not opcode:
            raise unknown_value("software interrupt builtin", builtin.instruction)

        arg = instr.args[0]

        # COP requires an immediate signature byte
        if isinstance(arg.value, MIRImmediate):
            self._emit_immediate(opcode, arg.value.value)
        else:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() requires a constant signature byte")

    def _emit_block_move_builtin(self, instr: Call, builtin):
        """Emit block move built-in (mvn, mvp)."""
        if len(instr.args) != 2:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 2 arguments, got {len(instr.args)}")

        src_bank_arg = instr.args[0]
        dst_bank_arg = instr.args[1]

        if not isinstance(src_bank_arg.value, MIRImmediate) or not isinstance(dst_bank_arg.value, MIRImmediate):
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects immediate bank numbers")

        src_bank = src_bank_arg.value.value
        dst_bank = dst_bank_arg.value.value

        opcode = BUILTIN_OPCODES.get(builtin.instruction)
        if not opcode:
            raise unknown_value("block move builtin", builtin.instruction)

        self.emitter.emit_instr(opcode, BlockMove(src_bank, dst_bank))

    def _emit_runtime_builtin(self, instr: Call, builtin):
        """Emit runtime library built-in (mul, div, mod, shl, shr)."""
        if len(instr.args) != 2:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 2 arguments, got {len(instr.args)}")

        # Load first argument into A
        arg0 = instr.args[0]
        arg0_loc = self.parent._get_operand_location(arg0.value)
        if isinstance(arg0.value, MIRImmediate):
            self._emit_load_immediate('A', arg0.value.value)
        elif arg0_loc.kind == LocationKind.HARDWARE and arg0_loc.hw_register == 'A':
            pass  # Already in A
        else:
            self.parent._emit_load('LDA', arg0_loc)

        # Load second argument into X
        arg1 = instr.args[1]
        arg1_loc = self.parent._get_operand_location(arg1.value)
        if isinstance(arg1.value, MIRImmediate):
            self._emit_load_immediate('X', arg1.value.value)
        elif arg1_loc.kind == LocationKind.HARDWARE and arg1_loc.hw_register == 'X':
            pass  # Already in X
        else:
            self.parent._emit_load('LDX', arg1_loc)

        # Call runtime library function
        runtime_func_name = f"__builtin_{instr.builtin_name}"
        self._emit_address(Opcode.JSR, runtime_func_name)

        # Store result if needed
        if instr.returns:
            return_vreg = instr.returns[0]
            dest_loc = self.parent._get_operand_location(return_vreg)

            if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
                pass  # Already in A
            else:
                self.parent._emit_store('STA', dest_loc)

    # ========================================================================
    # D Register Management for Far Pointer Parameters
    # ========================================================================

    def _emit_d_restore_before_call(self):
        """
        Restore D register to its original value before making a call.

        When D = S is set up for far pointer parameters, D must be restored
        before calling other functions that may use zeropage/DP addressing.

        The original D value was saved by PHD in the prologue. We use PLD to
        pop it from the stack and restore it. After the call returns, if we
        need D = S again, we'll push it back with PHD.
        """
        self._emit_instr(Opcode.PLD, comment="Restore D from stack before call")

    def _emit_d_push_only(self):
        """
        Push D back onto the stack after a call.

        Since we used PLD before the call, we must push D back so the
        epilogue's PLD will pop the correct value. This is used when there
        are no more far pointer dereferences after this call.
        """
        self._emit_instr(Opcode.PHD, comment="Save D back to stack for epilogue")

    def _emit_d_equals_s_restore(self):
        """
        Re-establish D = S after a call for continued far pointer access.

        After a call returns, if there are more far pointer dereferences,
        we need to:
        1. Push D back onto the stack (since PLD popped it before the call)
        2. Set D = S for continued far pointer access
        """
        self._emit_instr(Opcode.PHD, comment="Save D back to stack")
        self._emit_instr(Opcode.TSC, comment="Transfer S to A")
        self._emit_instr(Opcode.TCD, comment="Set D = S for far pointer access")

    def _has_far_ptr_derefs_after_call(self, call_instr: Call) -> bool:
        """
        Check if there are far pointer dereferences after this call.

        Uses simple scan approach: checks all remaining instructions in the
        function for LoadIndirect/StoreIndirect with is_far=True.

        Args:
            call_instr: The current Call instruction

        Returns:
            True if far pointer dereferences exist after this call
        """
        from r65.compiler.mir.nodes import LoadIndirect, StoreIndirect

        if not self.parent.current_function:
            return False

        # Find the call instruction's position and scan forward
        found_call = False

        for block in self.parent.current_function.blocks.values():
            for instr in block.instructions:
                if instr is call_instr:
                    found_call = True
                    continue

                if found_call:
                    # Check for far pointer indirect operations
                    if isinstance(instr, (LoadIndirect, StoreIndirect)):
                        if instr.is_far:
                            return True

        return False
