# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
from r65.compiler.mir.nodes import Call, TraitDispatch, VirtualRegister, ArgumentMechanism, Immediate as MIRImmediate

if TYPE_CHECKING:
    from r65.compiler.mir.liveness import ClobberRegion
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.codegen.abi_model import ABIKind
from r65.compiler.codegen.opcodes import (
    Opcode, TRANSFER_OPCODES, PUSH_OPCODES, PULL_OPCODES,
    LOAD_IMMEDIATE_OPCODES, STORE_MNEMONICS, BUILTIN_OPCODES
)
from r65.compiler.codegen.asm_nodes import BlockMove
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.errors import unknown_value
from r65.compiler.codegen.base_selector import BaseSelector
from r65.compiler.codegen.abi import StackStateTracker


class SpillInfo(NamedTuple):
    """Information about a hardware register that needs spilling."""
    vreg: Optional[VirtualRegister]  # The virtual register allocated to the hw reg, or None for direct hw reg usage
    hw_reg: str            # Hardware register name ('A', 'X', 'Y')
    spill_mode: Optional[int] = None  # For A register: mode at spill time (8 or 16), None for X/Y
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
        # Stack state tracker - tracks bytes pushed onto stack for spilling
        # Used to adjust stack-relative accesses while spills are active
        self.stack_tracker: StackStateTracker = StackStateTracker()
        # Track pending A spill info for reload (need to restore mode)
        self.pending_a_spill: Optional[SpillInfo] = None
        # Track pending per-call X/Y spills for reload
        # When X/Y are spilled via per-call fallback (no region), we need to
        # emit PLX/PLY after the call. Maps hw_reg -> SpillInfo.
        self.pending_xy_spills: Dict[str, SpillInfo] = {}

    def set_block_regions(self, block_id: int, regions: Dict[str, List['ClobberRegion']]):
        """Set pre-computed regions for a block."""
        self.block_regions[block_id] = regions
        self.current_block_id = block_id
        # Clear active regions when entering new block
        self.active_regions.clear()
        # Clear any pending per-call X/Y spills
        self.pending_xy_spills.clear()
        # Reset stack tracker for new block
        self.stack_tracker.reset()

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
        self.region_state = ActiveRegionState()
        # Pre-computed regions for current function: {block_id: {hw_reg: [ClobberRegion]}}
        self._function_regions: Optional[Dict[int, Dict[str, List]]] = None
        # Whether A is bound to a vreg (uses region-based spilling if True)
        self._a_bound_to_vreg: bool = False

    def initialize_regions_for_function(self):
        """
        Pre-compute clobber regions for all blocks in the current function.

        Should be called once per function before processing any blocks.
        """
        reg_alloc = self.parent.reg_alloc
        if not reg_alloc or not reg_alloc.instr_liveness:
            self._function_regions = None
            self._a_bound_to_vreg = False
            return

        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        # Build preserves map from function signatures in the current function
        preserves_map = self._build_preserves_map()

        # Check if A is bound to a vreg (e.g., function parameter @ A)
        # If so, include A in region-based spilling
        a_alloc = reg_alloc.get_hw_alloc('A')
        self._a_bound_to_vreg = a_alloc.allocated_vreg is not None

        analyzer = ClobberRegionAnalyzer(reg_alloc.instr_liveness)
        self._function_regions = analyzer.analyze_function(
            preserves_map,
            include_a=self._a_bound_to_vreg
        )

    def initialize_regions_for_block(self, block_id: int):
        """
        Initialize region tracking state for a specific block.

        Should be called when starting to process a new block.

        Args:
            block_id: The block ID being processed
        """
        if self._function_regions is None:
            # No pre-computed regions, will fall back to per-call spilling
            self.region_state = ActiveRegionState()
            return

        default_regions = {'A': [], 'X': [], 'Y': []} if self._a_bound_to_vreg else {'X': [], 'Y': []}
        regions = self._function_regions.get(block_id, default_regions)
        self.region_state.set_block_regions(block_id, regions)

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

    def _store_multibyte_outgoing(self, arg, arg_loc, source_size: int, target_size: int, outgoing_offset: int):
        """
        Store a multi-byte value to outgoing area byte by byte.

        Handles zero-extension when source is smaller than target.
        Stores in little-endian order (low byte at lowest offset).
        Must be called in m8 mode.

        Args:
            arg: Argument being stored
            arg_loc: Source location
            source_size: Size of source value in bytes
            target_size: Size of target parameter in bytes
            outgoing_offset: Base stack-relative offset for storage
        """
        from r65.compiler.codegen.asm_nodes import StackOffset

        for byte_idx in range(target_size):
            if byte_idx >= source_size:
                # Zero-extend: store 0 for bytes beyond source size
                byte_name = {2: "bank", 1: "high", 0: "low"}.get(byte_idx, f"byte{byte_idx}")
                self._emit_load_immediate('A', 0, f"Zero-extend {byte_name} byte")
            else:
                byte_name = {2: "bank", 1: "high", 0: "low"}.get(byte_idx, f"byte{byte_idx}")
                if byte_idx > 0:
                    loc = self.parent._offset_location(arg_loc, byte_idx)
                else:
                    loc = arg_loc
                self.parent._emit_load('LDA', loc, f"Load {byte_name} byte")
            self.emitter.emit_instr(
                Opcode.STA_STACK, StackOffset(outgoing_offset + byte_idx),
                f"Store {byte_name} byte to outgoing"
            )

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

        # Check if we need D/DBR management for far pointer params
        from r65.compiler.mir.nodes import FarPtrStrategy
        needs_d_management = (
            self.parent.current_function and
            self.parent.current_function.has_far_ptr_stack_params and
            self.parent.current_function.far_ptr_strategy != FarPtrStrategy.SET_DBR
        )
        needs_dbr_management = (
            self.parent.current_function and
            self.parent.current_function.far_ptr_strategy == FarPtrStrategy.SET_DBR
        )

        # Step 0: Compute hardware register spills needed
        # Spill hw registers that are live across this call and not preserved by callee
        spills = self._compute_hw_spills(instr)

        # Step 0.5: Emit spills BEFORE argument setup (to avoid clobbering args)
        self._emit_hw_spills(spills)

        # Step 0.7: Restore D register BEFORE argument setup (for far pointer functions)
        # CRITICAL: PLD must happen before pushing arguments. If PLD happens after,
        # it pops 2 bytes from the top of the stack, consuming part of the just-pushed
        # arguments instead of the saved D value.
        # The stack tracker records the PLD displacement so that all subsequent
        # stack-relative accesses (argument loading, return value storing) are
        # automatically adjusted.
        if needs_d_management:
            self._emit_d_restore_before_call()

        # Step 0.7b: SET_DBR: save ptr-bank DBR and set DBR to code bank for callee
        if needs_dbr_management:
            self._emit_dbr_save_before_call()

        # Step 1: Set up arguments
        stack_bytes_pushed = self._emit_argument_setup(instr)

        # Step 2: Handle caller-managed DBR (databank=caller)
        needs_dbr_restore = self._emit_caller_dbr_setup(instr)

        # Step 2.6: Switch to callee's entry mode if needed
        # Callee's prologue expects to be in entry_m_mode
        self._emit_entry_mode_switch(instr)

        # Step 3: Make the call
        self._emit_call_instruction(instr, stack_bytes_pushed)

        # Invalidate XBA state after call (function may have modified A/B)
        self.parent._invalidate_xba_state()

        # Step 3.5: Update emitter mode to reflect callee's exit mode
        # This is critical - the callee may exit in a different mode than entry
        self._update_mode_after_call(instr)

        # Step 4: Restore DBR if caller-managed
        if needs_dbr_restore:
            self._emit_pull('B', "Restore data bank (caller)")

        # Step 5: Collect return values (in callee's exit mode)
        pascal_result_bytes = getattr(instr, 'pascal_result_bytes', 0)
        if pascal_result_bytes > 0:
            # Pascal: callee cleaned params, result space is at TOS.
            # PLA the result from the stack instead of reading from registers.
            self._emit_pascal_return_value_collection(instr, pascal_result_bytes)
            # Callee cleaned params, we PLA'd result — all pushed bytes are gone.
            self.region_state.stack_tracker.pop(stack_bytes_pushed)
            stack_bytes_pushed = 0
        elif pascal_result_bytes == 0 and stack_bytes_pushed > 0 and self.parent.abi_model.kind == ABIKind.PASCAL:
            # Pascal void call: callee cleaned all params, nothing remains.
            self.region_state.stack_tracker.pop(stack_bytes_pushed)
            stack_bytes_pushed = 0
        else:
            self._emit_return_value_collection(instr)

        # Step 5.5: Caller cleanup of PHA-pushed args (if spill fallback was used)
        # Done after return value collection so A/X/Y return values are safely
        # stored in their virtual register destinations before we clobber A.
        if stack_bytes_pushed > 0:
            returns_in_x = self._call_returns_in_x(instr)
            returns_in_a = self._call_returns_in_a(instr)
            self._emit_caller_arg_cleanup(stack_bytes_pushed, returns_in_x=returns_in_x,
                                          returns_in_a=returns_in_a)

        # Step 6: Restore mode after receiving return value
        # If callee exited in m16 (u16 return), switch back to m8
        self._emit_exit_mode_restore(instr)

        # Step 6.5: Reload spilled hardware registers (region-based)
        # Only reload registers where this is the last call in the region
        reloads = self._compute_hw_reloads(instr)
        self._emit_hw_reloads(reloads)

        # Step 8: Restore D to stack and optionally re-establish D = S
        if needs_d_management:
            # We used PLD before the call, so we must push D back
            # If there are more far pointer dereferences, also set D = S
            if self._has_far_ptr_derefs_after_call(instr):
                live_regs = self._compute_live_regs_after_call(instr, reloads)
                self._emit_d_equals_s_restore(live_regs)  # PHD + TSC + TCD
            else:
                self._emit_d_push_only()  # Just PHD (for epilogue)

        # Step 8b: SET_DBR: restore ptr-bank DBR after call
        if needs_dbr_management:
            self._emit_dbr_restore_after_call()

    # ========================================================================
    # Trait Dispatch
    # ========================================================================

    def select_trait_dispatch(self, instr: TraitDispatch):
        """
        Generate code for TraitDispatch instruction.

        Emits argument setup, then JSR to the dispatch wrapper function.
        The dispatch wrapper (generated separately) loads TypeId and jumps
        to the concrete implementation via a jump table.

        For far self (self_is_far=True), the caller must:
        1. PHB to save caller's DBR
        2. Load bank byte from far pointer → PHA; PLB to set DBR
        3. Load 16-bit address into Y
        4. JSR/JSL dispatch wrapper
        5. PLB to restore caller's DBR

        Args:
            instr: TraitDispatch instruction
        """
        # Trait dispatch is treated like a regular call to the dispatch wrapper.
        # The wrapper function is generated in codegen.py and handles the
        # TypeId lookup and jump table dispatch.
        dispatch_name = f"{instr.trait_name}__{instr.method_name}__dispatch"

        # Compute hw register spills (no preserves_attr for trait dispatch)
        spills = self._compute_trait_dispatch_spills(instr)
        self._emit_hw_spills(spills)

        # For far self, save caller's DBR before arg setup sets it to object's bank
        if instr.self_is_far:
            self._emit_push('B', "Save caller's DBR for far self dispatch")
            self.region_state.stack_tracker.push(1)  # PHB = 1 byte

        # Set up arguments (same mechanism as regular calls)
        stack_bytes_pushed = self._emit_trait_dispatch_args(instr)

        # Emit the call to the dispatch wrapper
        if instr.is_far:
            self._emit_address(Opcode.JSL, dispatch_name)
        else:
            self._emit_address(Opcode.JSR, dispatch_name)

        # Invalidate XBA state after dispatch
        self.parent._invalidate_xba_state()

        # Collect return values
        self._emit_return_value_collection(instr)

        # Caller cleanup of PHA-pushed args (if spill fallback was used)
        if stack_bytes_pushed > 0:
            returns_in_x = self._call_returns_in_x(instr)
            returns_in_a = self._call_returns_in_a(instr)
            self._emit_caller_arg_cleanup(stack_bytes_pushed, returns_in_x=returns_in_x,
                                          returns_in_a=returns_in_a)

        # Restore caller's DBR after far self dispatch
        if instr.self_is_far:
            self._emit_pull('B', "Restore caller's DBR after far self dispatch")
            self.region_state.stack_tracker.pop(1)  # PLB = 1 byte

        # Reload spilled registers
        reloads = self._compute_trait_dispatch_reloads(instr)
        self._emit_hw_reloads(reloads)

    def _compute_trait_dispatch_spills(self, instr: TraitDispatch) -> List[SpillInfo]:
        """Compute hw register spills for trait dispatch (no preserves)."""
        spills: List[SpillInfo] = []
        reg_alloc = self.parent.reg_alloc
        if not reg_alloc:
            return spills

        instr_idx = None
        if reg_alloc.instr_liveness:
            pos = reg_alloc.instr_liveness.get_instruction_position(instr)
            if pos:
                _, instr_idx = pos

        # Determine return registers
        return_regs = self._get_callee_return_registers(instr)
        callee_return_type = getattr(instr, 'callee_return_type', None)
        if callee_return_type is not None:
            from r65.compiler.hir.types import TupleTypeInfo
            if isinstance(callee_return_type, TupleTypeInfo):
                num_returns = len(callee_return_type.element_types)
            else:
                num_returns = 1
        else:
            num_returns = len(instr.returns)
        call_return_set = set(return_regs[:num_returns])

        for reg_name in ['A', 'X', 'Y']:
            if reg_name in call_return_set:
                continue

            use_region_based = (
                instr_idx is not None and
                self._function_regions is not None and
                (reg_name in ('X', 'Y') or (reg_name == 'A' and self._a_bound_to_vreg))
            )

            if use_region_based:
                region = self.region_state.get_region_for_call(reg_name, instr_idx)
                if region is not None:
                    if self.region_state.is_first_call_in_region(reg_name, instr_idx):
                        spills.append(SpillInfo(vreg=None, hw_reg=reg_name))
                        self.region_state.mark_region_active(reg_name, region)
                    continue

            hw_alloc = reg_alloc.get_hw_alloc(reg_name)
            vreg = hw_alloc.allocated_vreg

            if vreg is not None:
                if reg_alloc.instr_liveness:
                    pos = reg_alloc.instr_liveness.get_instruction_position(instr)
                    if pos:
                        block_id, idx = pos
                        if reg_alloc.instr_liveness.is_live_after(vreg, block_id, idx):
                            spills.append(SpillInfo(vreg=vreg, hw_reg=reg_name))
                else:
                    if hw_alloc.is_bound:
                        spills.append(SpillInfo(vreg=vreg, hw_reg=reg_name))
                continue

            if reg_name in ('X', 'Y') and reg_alloc.instr_liveness and self._function_regions is None:
                pos = reg_alloc.instr_liveness.get_instruction_position(instr)
                if pos:
                    block_id, idx = pos
                    if reg_alloc.instr_liveness.is_hw_reg_live_after(reg_name, block_id, idx):
                        spills.append(SpillInfo(vreg=None, hw_reg=reg_name))

        return spills

    def _compute_trait_dispatch_reloads(self, instr: TraitDispatch) -> List[SpillInfo]:
        """Compute hw register reloads after trait dispatch."""
        reloads: List[SpillInfo] = []
        reg_alloc = self.parent.reg_alloc
        if not reg_alloc:
            return reloads

        # Skip return registers — they hold the call result, not the spilled value
        return_regs = self._get_callee_return_registers(instr)
        callee_return_type = getattr(instr, 'callee_return_type', None)
        if callee_return_type is not None:
            from r65.compiler.hir.types import TupleTypeInfo
            if isinstance(callee_return_type, TupleTypeInfo):
                num_returns = len(callee_return_type.element_types)
            else:
                num_returns = 1
        else:
            num_returns = len(instr.returns)
        call_return_set = set(return_regs[:num_returns])

        instr_idx = None
        if reg_alloc.instr_liveness:
            pos = reg_alloc.instr_liveness.get_instruction_position(instr)
            if pos:
                _, instr_idx = pos

        for reg_name in ['A', 'X', 'Y']:
            if reg_name in call_return_set:
                continue

            use_region_based = (
                instr_idx is not None and
                self._function_regions is not None and
                (reg_name in ('X', 'Y') or (reg_name == 'A' and self._a_bound_to_vreg))
            )

            if use_region_based:
                region = self.region_state.get_region_for_call(reg_name, instr_idx)
                if region is not None:
                    if self.region_state.is_last_call_in_region(reg_name, instr_idx):
                        reloads.append(SpillInfo(vreg=None, hw_reg=reg_name))
                        self.region_state.mark_region_inactive(reg_name)
                    continue

            hw_alloc = reg_alloc.get_hw_alloc(reg_name)
            vreg = hw_alloc.allocated_vreg
            if vreg is not None:
                if reg_alloc.instr_liveness:
                    pos = reg_alloc.instr_liveness.get_instruction_position(instr)
                    if pos:
                        block_id, idx = pos
                        if reg_alloc.instr_liveness.is_live_after(vreg, block_id, idx):
                            reloads.append(SpillInfo(vreg=vreg, hw_reg=reg_name))
                else:
                    if hw_alloc.is_bound:
                        reloads.append(SpillInfo(vreg=vreg, hw_reg=reg_name))
                continue

            if reg_name in ('X', 'Y') and reg_alloc.instr_liveness and self._function_regions is None:
                pos = reg_alloc.instr_liveness.get_instruction_position(instr)
                if pos:
                    block_id, idx = pos
                    if reg_alloc.instr_liveness.is_hw_reg_live_after(reg_name, block_id, idx):
                        reloads.append(SpillInfo(vreg=None, hw_reg=reg_name))

        return reloads

    def _emit_trait_dispatch_args(self, instr: TraitDispatch) -> int:
        """Emit trait dispatch arguments. Delegates to ABIModel.emit_trait_dispatch_args."""
        return self.parent.abi_model.emit_trait_dispatch_args(self, instr)

    def load_y_with_self(self, arg: 'Argument', stack_bytes_pushed: int):
        """Load Y register with the self pointer address for trait dispatch.

        Handles different source locations:
        - Scratch/DP: LDY dp_addr
        - Stack: REP #$20; LDA d,S; TAY; SEP #$20 (no LDY d,S on 65816)
        - Hardware (X): TXY
        - Immediate: LDY #imm
        - Memory: LDY abs_addr
        """
        from r65.compiler.codegen.constants import M_FLAG

        arg_loc = self.parent._get_operand_location(arg.value)

        # Adjust for stack args that were pushed
        if arg_loc.kind == LocationKind.STACK and stack_bytes_pushed > 0:
            arg_loc = self.parent._offset_location(arg_loc, stack_bytes_pushed)

        if isinstance(arg.value, MIRImmediate):
            # Immediate address
            self.parent._emit_immediate(Opcode.LDY_IMMEDIATE, arg.value.value, "Load self ptr into Y")
        elif arg_loc.is_hw():
            if arg_loc.hw_register == 'Y':
                pass  # Already in Y
            elif arg_loc.hw_register == 'X':
                self.parent.emitter.emit_raw("    TXY")
            elif arg_loc.hw_register == 'A':
                self.parent.emitter.emit_raw("    TAY")
            else:
                raise InstructionSelectionError(f"Cannot load Y from hardware register {arg_loc.hw_register}", source_loc=self.parent._current_source_loc)
        elif arg_loc.kind == LocationKind.SCRATCH:
            # Scratch (direct page) location — LDY dp
            self.parent._emit_load('LDY', arg_loc, "Load self ptr into Y")
        elif arg_loc.kind == LocationKind.STACK:
            # Stack-relative: no LDY d,S exists on 65816
            # Use: REP #$20; LDA d,S; TAY; SEP #$20
            self.parent._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for stack self load")
            self.parent.emitter.emit_accu_mode(16)
            self.parent._emit_load('LDA', arg_loc, "Load self ptr from stack")
            self.parent.emitter.emit_raw("    TAY")
            self.parent._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
        elif arg_loc.kind == LocationKind.MEMORY:
            # Absolute memory location
            self.parent._emit_load('LDY', arg_loc, "Load self ptr into Y")
        else:
            raise InstructionSelectionError(f"Cannot load Y from location kind {arg_loc.kind}", source_loc=self.parent._current_source_loc)

    def load_y_with_far_self(self, arg: 'Argument', stack_bytes_pushed: int):
        """Load Y register and set DBR from a 3-byte far self pointer for trait dispatch.

        For a far self pointer (3 bytes: addr_low, addr_high, bank):
        1. Load bank byte (offset +2) into A
        2. PHA; PLB — set DBR to object's bank (net stack change = 0)
        3. Load 16-bit address (offset +0) into Y

        Handles different source locations:
        - Scratch/DP: LDA dp+2 for bank, LDY dp for addr
        - Stack: LDA sp+2,S for bank; REP; LDA sp,S; TAY; SEP for addr
        - Memory: LDA abs+2 for bank, LDY abs for addr
        - Immediate: extract statically
        """
        from r65.compiler.codegen.constants import M_FLAG

        arg_loc = self.parent._get_operand_location(arg.value)

        # Adjust for stack args that were pushed
        if arg_loc.kind == LocationKind.STACK and stack_bytes_pushed > 0:
            arg_loc = self.parent._offset_location(arg_loc, stack_bytes_pushed)

        if isinstance(arg.value, MIRImmediate):
            # Immediate far address: extract bank and addr statically
            val = arg.value.value
            bank = (val >> 16) & 0xFF
            addr = val & 0xFFFF
            # Set DBR to bank
            self.parent._emit_immediate(Opcode.LDA_IMMEDIATE, bank, f"Bank byte ${bank:02X}")
            self._emit_push('A', "Push bank byte")
            self._emit_pull('B', "Set DBR to object's bank")
            # Load 16-bit address into Y
            self.parent._emit_immediate(Opcode.LDY_IMMEDIATE, addr, f"Load self addr ${addr:04X} into Y")
        elif arg_loc.kind == LocationKind.SCRATCH:
            # Scratch (direct page) location
            # Load bank byte from dp+2
            bank_loc = self.parent._offset_location(arg_loc, 2)
            self.parent._emit_load('LDA', bank_loc, "Load bank byte from far self ptr")
            self._emit_push('A', "Push bank byte")
            self._emit_pull('B', "Set DBR to object's bank")
            # Load 16-bit address into Y from dp
            self.parent._emit_load('LDY', arg_loc, "Load self addr into Y")
        elif arg_loc.kind == LocationKind.STACK:
            # Stack-relative: load bank byte (offset +2)
            bank_loc = self.parent._offset_location(arg_loc, 2)
            self.parent._emit_load('LDA', bank_loc, "Load bank byte from far self ptr")
            self._emit_push('A', "Push bank byte")
            self._emit_pull('B', "Set DBR to object's bank")
            # Load 16-bit address: no LDY d,S on 65816, use LDA d,S; TAY
            self.parent._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for stack self addr load")
            self.parent.emitter.emit_accu_mode(16)
            self.parent._emit_load('LDA', arg_loc, "Load self addr from stack")
            self.parent.emitter.emit_raw("    TAY")
            self.parent._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
        elif arg_loc.kind == LocationKind.MEMORY:
            # Absolute memory location
            bank_loc = self.parent._offset_location(arg_loc, 2)
            self.parent._emit_load('LDA', bank_loc, "Load bank byte from far self ptr")
            self._emit_push('A', "Push bank byte")
            self._emit_pull('B', "Set DBR to object's bank")
            self.parent._emit_load('LDY', arg_loc, "Load self addr into Y")
        else:
            raise InstructionSelectionError(
                f"Cannot load far self from location kind {arg_loc.kind}",
                source_loc=self.parent._current_source_loc
            )

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

        # Determine which registers receive return values from this call.
        # A call that defines a register via return should never be spilled —
        # spilling would save garbage and PLX/PLY would clobber the return value.
        # Note: instr.returns may be empty (return values captured by separate Move
        # instructions), so we determine the count from callee_return_type.
        return_regs = self._get_callee_return_registers(instr)
        callee_return_type = getattr(instr, 'callee_return_type', None)
        if callee_return_type is not None:
            from r65.compiler.hir.types import TupleTypeInfo
            if isinstance(callee_return_type, TupleTypeInfo):
                num_returns = len(callee_return_type.element_types)
            else:
                num_returns = 1
        else:
            num_returns = len(instr.returns)
        call_return_set = set(return_regs[:num_returns])

        for reg_name in ['A', 'X', 'Y']:
            # Skip if callee preserves this register
            if reg_name in preserved:
                continue

            # Skip if this call returns a value in this register
            if reg_name in call_return_set:
                continue

            # Check if region-based spilling is available for this register
            # - X/Y always use region-based when available
            # - A uses region-based only when bound to a vreg
            use_region_based = (
                instr_idx is not None and
                self._function_regions is not None and
                (reg_name in ('X', 'Y') or (reg_name == 'A' and self._a_bound_to_vreg))
            )

            if use_region_based:
                # Check if this call is in a region for this register
                region = self.region_state.get_region_for_call(reg_name, instr_idx)

                if region is not None:
                    # This call is in a clobber region
                    if self.region_state.is_first_call_in_region(reg_name, instr_idx):
                        # First call in region - need to spill
                        spills.append(SpillInfo(vreg=None, hw_reg=reg_name))
                        # Mark region as active
                        self.region_state.mark_region_active(reg_name, region)
                    # Else: already in active region, no spill needed
                    continue

            # Fall back to per-call analysis for A (when not bound) or when no region
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

        Uses region-based analysis for X/Y: only returns registers where this call
        is the LAST clobbering call in the region.

        For A register: uses per-call spilling - reload if it was spilled for this call.

        Args:
            instr: The Call instruction

        Returns:
            List of SpillInfo for registers that need reloading after THIS call
        """
        reloads: List[SpillInfo] = []

        reg_alloc = self.parent.reg_alloc
        if not reg_alloc or not reg_alloc.instr_liveness:
            # If A was spilled, still need to reload it
            if self.region_state.pending_a_spill is not None:
                reloads.append(self.region_state.pending_a_spill)
            return reloads

        # Get instruction position
        pos = reg_alloc.instr_liveness.get_instruction_position(instr)
        if not pos:
            if self.region_state.pending_a_spill is not None:
                reloads.append(self.region_state.pending_a_spill)
            return reloads

        _, instr_idx = pos

        # Check each active region for X/Y and A (when bound)
        regs_to_check = ('A', 'X', 'Y') if self._a_bound_to_vreg else ('X', 'Y')
        for hw_reg in regs_to_check:
            if self.region_state.is_region_active(hw_reg):
                if self.region_state.is_last_call_in_region(hw_reg, instr_idx):
                    # Last call in region - need to reload
                    reloads.append(SpillInfo(vreg=None, hw_reg=hw_reg))
                    # Clear active region
                    self.region_state.clear_active_region(hw_reg)

        # A register per-call spilling (only when NOT using region-based)
        # This handles the case where A is used but not bound to a vreg
        if not self._a_bound_to_vreg and self.region_state.pending_a_spill is not None:
            reloads.append(self.region_state.pending_a_spill)

        # X/Y per-call spilling: when a vreg allocated to X/Y is live across
        # a call but no clobber region was created (because the liveness analysis
        # only tracks direct HardwareRegister usage, not vreg-to-hw mappings),
        # the per-call fallback emits PHY/PHX but _compute_hw_reloads won't find
        # an active region. Check pending per-call spills and emit PLY/PLX.
        for hw_reg in ('X', 'Y'):
            if hw_reg in self.region_state.pending_xy_spills:
                reloads.append(self.region_state.pending_xy_spills[hw_reg])
                del self.region_state.pending_xy_spills[hw_reg]

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
            if spill.hw_reg == 'A':
                # A register size depends on current mode
                current_mode = self.parent.emitter.get_accu_mode()
                # Create spill info with mode recorded
                spill_with_mode = SpillInfo(
                    vreg=spill.vreg,
                    hw_reg=spill.hw_reg,
                    spill_mode=current_mode
                )
                self._emit_push('A', f"Spill A (m{current_mode})")
                # Track stack growth based on actual mode
                self.region_state.stack_tracker.push(2 if current_mode == 16 else 1)
                # Save spill info for reload
                self.region_state.pending_a_spill = spill_with_mode
            else:
                # X/Y are always 16-bit (2 bytes)
                self._emit_push(spill.hw_reg, f"Spill {spill.hw_reg} (region start)")
                self.region_state.stack_tracker.push(2)
                # Track per-call spill for X/Y when no region is active.
                # Region-based spills have mark_region_active() called before
                # _emit_hw_spills(), so is_region_active() distinguishes the paths.
                if not self.region_state.is_region_active(spill.hw_reg):
                    self.region_state.pending_xy_spills[spill.hw_reg] = spill

    def _emit_hw_reloads(self, spills: List[SpillInfo]):
        """
        Emit pull instructions to reload spilled hardware registers.

        Must be called in reverse order of spills due to stack behavior (LIFO).
        Updates the spill offset to track stack shrinkage.

        For region-based spilling, this is only called at region end.

        For A register reloads, ensures we're in the same mode used when spilling,
        since PHA/PLA push/pull different amounts based on m8 vs m16 mode.

        Args:
            spills: List of registers to reload (will be processed in reverse)
        """
        from r65.compiler.codegen.constants import M_FLAG

        for spill in reversed(spills):
            if spill.hw_reg == 'A':
                # Get the mode A was spilled in
                pending = self.region_state.pending_a_spill
                if pending and pending.spill_mode is not None:
                    spill_mode = pending.spill_mode
                    current_mode = self.parent.emitter.get_accu_mode()

                    # Switch to spill mode if different
                    if current_mode != spill_mode:
                        if spill_mode == 16:
                            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                                 "Switch to m16 for A reload")
                            self.parent.emitter.emit_accu_mode(16)
                        else:
                            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG,
                                                 "Switch to m8 for A reload")
                            self.parent.emitter.emit_accu_mode(8)

                    self._emit_pull('A', f"Reload A (m{spill_mode})")

                    # Track stack shrinkage based on spill mode
                    self.region_state.stack_tracker.pop(2 if spill_mode == 16 else 1)

                    # Clear pending A spill
                    self.region_state.pending_a_spill = None
                else:
                    # Fallback: no mode info, assume m8
                    self._emit_pull('A', "Reload A (region end)")
                    self.region_state.stack_tracker.pop(1)
            else:
                # X/Y are always 16-bit (2 bytes)
                self._emit_pull(spill.hw_reg, f"Reload {spill.hw_reg} (region end)")
                self.region_state.stack_tracker.pop(2)

    def get_current_spill_offset(self) -> int:
        """
        Get the current spill offset for adjusting stack-relative accesses.

        Returns:
            Number of bytes currently pushed for spilling
        """
        return self.region_state.stack_tracker.displacement

    # ========================================================================
    # Argument Setup
    # ========================================================================

    def _emit_argument_setup(self, instr: Call, pre_arg_stack_adj: int = 0) -> int:
        """Set up call arguments. Delegates to ABIModel.emit_call_args."""
        return self.parent.abi_model.emit_call_args(self, instr, pre_arg_stack_adj)

    def arg_sort_key(self, arg):
        """Sort key for argument processing order."""
        if arg.mechanism == ArgumentMechanism.STACK:
            return 0  # Stack first
        elif arg.mechanism == ArgumentMechanism.SCRATCH_PARAM:
            return 1  # Scratch params second (STA to DP, clobbers A)
        elif arg.mechanism == ArgumentMechanism.VARIABLE:
            return 2  # Variable-bound third
        elif arg.mechanism == ArgumentMechanism.REGISTER:
            target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)
            if target_reg == 'B':
                return 3  # B fourth (clobbers A)
            elif target_reg in ['X', 'Y']:
                return 4  # X, Y fifth
            elif target_reg == 'A':
                return 5  # A last (to avoid being clobbered)
        return 6

    def emit_outgoing_stack_argument(self, arg, arg_loc, outgoing_offset: int):
        """Emit stack argument via STA d,S into caller's outgoing area.

        Writes the argument value to a fixed stack offset without moving SP.
        Replaces PHA-based pushing with caller-owned outgoing args convention.

        IMPORTANT: When source value is smaller than parameter type, we must
        zero-extend. For example, an 8-bit loop variable passed to a u16
        parameter needs its high byte set to 0, not loaded from garbage memory.

        Args:
            arg: Argument being emitted
            arg_loc: Physical location of the source value
            outgoing_offset: Stack-relative offset in the outgoing area ($01 for first param)
        """
        from r65.compiler.codegen.type_utils import get_type_size
        from r65.compiler.codegen.constants import M_FLAG
        from r65.compiler.codegen.asm_nodes import StackOffset

        # Adjust outgoing offset for current spill offset.
        # When region spills are active (PHY/PHX pushed before call),
        # SP has moved down and the outgoing area is shifted by the same amount.
        # Source locations are auto-adjusted via _emit_load -> _get_opcode_for_location,
        # but STA destinations use StackOffset directly and need manual adjustment.
        spill_offset = self.get_current_spill_offset()
        outgoing_offset += spill_offset

        # Determine PARAMETER size (what we need to store)
        param_size = 1
        if arg.param_type is not None:
            param_size = get_type_size(arg.param_type)
        elif hasattr(arg.value, 'type_info') and arg.value.type_info:
            param_size = get_type_size(arg.value.type_info)
        elif isinstance(arg.value, MIRImmediate):
            value = arg.value.value
            if value > 0xFFFF or value < -32768:
                param_size = 3
            elif value > 0xFF or value < -128:
                param_size = 2

        # Determine SOURCE size for zero-extension
        source_size = param_size
        if hasattr(arg.value, 'type_info') and arg.value.type_info:
            source_size = get_type_size(arg.value.type_info)
        elif isinstance(arg.value, MIRImmediate):
            source_size = param_size

        if param_size == 3:
            # 24-bit (far pointer): store byte by byte in m8
            self._ensure_m8_mode("8-bit A for byte store")
            if arg_loc.is_hw('A'):
                raise InstructionSelectionError("Cannot store 24-bit value from A register", source_loc=self.parent._current_source_loc)
            self._store_multibyte_outgoing(arg, arg_loc, source_size, 3, outgoing_offset)

        elif param_size == 2:
            # 16-bit: prefer single m16 STA d,S when possible
            if source_size < 2:
                # Source smaller than param — zero-extend byte by byte
                self._ensure_m8_mode("8-bit A for zero-ext store")
                self._store_multibyte_outgoing(arg, arg_loc, source_size, 2, outgoing_offset)
            elif arg_loc.is_hw('A'):
                # Value in A — ensure m16 for 16-bit store
                current_mode = self.parent.emitter.get_accu_mode()
                if current_mode != 16:
                    self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for u16 outgoing arg")
                    self.parent.emitter.emit_accu_mode(16)
                self.emitter.emit_instr(Opcode.STA_STACK, StackOffset(outgoing_offset), "Outgoing u16 arg")
            elif isinstance(arg.value, MIRImmediate):
                self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for u16 outgoing arg")
                self.parent.emitter.emit_accu_mode(16)
                self._emit_load_immediate('A', arg.value.value)
                self.emitter.emit_instr(Opcode.STA_STACK, StackOffset(outgoing_offset), "Outgoing u16 arg")
            else:
                # Memory/stack/scratch source: load in m16, store
                self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for u16 outgoing arg")
                self.parent.emitter.emit_accu_mode(16)
                self.parent._emit_load('LDA', arg_loc)
                self.emitter.emit_instr(Opcode.STA_STACK, StackOffset(outgoing_offset), "Outgoing u16 arg")

        else:
            # 8-bit: ensure m8, load value, store to outgoing offset
            self._ensure_m8_mode("8-bit A for u8 outgoing arg")
            if arg_loc.is_hw('A'):
                pass  # Already in A
            elif arg_loc.is_hw():
                if arg_loc.hw_register == 'X':
                    self._emit_transfer('X', 'A')
                elif arg_loc.hw_register == 'Y':
                    self._emit_transfer('Y', 'A')
            elif isinstance(arg.value, MIRImmediate):
                self._emit_load_immediate('A', arg.value.value)
            else:
                self.parent._emit_load('LDA', arg_loc)
            self.emitter.emit_instr(Opcode.STA_STACK, StackOffset(outgoing_offset), "Outgoing u8 arg")

    def emit_pha_stack_argument(self, arg, arg_loc, param_size: int):
        """Emit stack argument via PHA (fallback when region spills are active).

        Pushes the argument value onto the stack. Used when region-based spills
        (PHY/PHX) have pushed bytes between the frame and SP, making the outgoing
        area inaccessible to the callee at fixed offsets.

        Source locations are auto-adjusted for current stack displacement by
        _emit_load -> _get_opcode_for_location. The caller updates the tracker
        AFTER this method returns (by the full param_size). For multi-byte pushes
        where intermediate PHAs shift SP, we temporarily adjust the tracker
        between byte pushes to keep source reads correct.

        Args:
            arg: Argument being emitted
            arg_loc: Physical location of the source value
            param_size: Size of the parameter in bytes (1, 2, or 3)
        """
        from r65.compiler.codegen.constants import M_FLAG

        if param_size == 1:
            # 8-bit: load into A, PHA
            self._ensure_m8_mode("8-bit A for u8 stack arg")
            if arg_loc.is_hw('A'):
                pass
            elif arg_loc.is_hw():
                if arg_loc.hw_register == 'X':
                    self._emit_transfer('X', 'A')
                elif arg_loc.hw_register == 'Y':
                    self._emit_transfer('Y', 'A')
            elif isinstance(arg.value, MIRImmediate):
                self._emit_load_immediate('A', arg.value.value)
            else:
                self.parent._emit_load('LDA', arg_loc)
            self._emit_push('A', "Push u8 arg")

        elif param_size == 2:
            # 16-bit: load into A (m16), PHA
            source_size = param_size
            if hasattr(arg.value, 'type_info') and arg.value.type_info:
                from r65.compiler.codegen.type_utils import get_type_size
                source_size = get_type_size(arg.value.type_info)

            if source_size < 2:
                # Zero-extend: push high byte (0) first, then low byte
                self._ensure_m8_mode("8-bit A for zero-ext push")
                self._emit_load_immediate('A', 0, "Zero high byte")
                self._emit_push('A', "Push high byte (zero)")
                # First PHA shifted SP by 1; adjust tracker so source reads are correct
                self.region_state.stack_tracker.push(1)
                if arg_loc.is_hw('A'):
                    pass
                elif arg_loc.is_hw('X'):
                    self._emit_transfer('X', 'A')
                elif arg_loc.is_hw('Y'):
                    self._emit_transfer('Y', 'A')
                elif isinstance(arg.value, MIRImmediate):
                    self._emit_load_immediate('A', arg.value.value & 0xFF)
                else:
                    self.parent._emit_load('LDA', arg_loc)
                self._emit_push('A', "Push low byte")
                # Undo temporary adjustment (caller adds full param_size after return)
                self.region_state.stack_tracker.pop(1)
            else:
                # Use PEA for immediate u16 values — pushes 16-bit regardless
                # of accumulator mode, no REP/SEP needed.
                if isinstance(arg.value, MIRImmediate):
                    self._emit_immediate(Opcode.PEA, arg.value.value,
                                         "Push u16 immediate")
                else:
                    self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for u16 stack arg")
                    self.parent.emitter.emit_accu_mode(16)
                    if arg_loc.is_hw('A'):
                        pass
                    elif arg_loc.is_hw('X'):
                        self._emit_transfer('X', 'A')
                    elif arg_loc.is_hw('Y'):
                        self._emit_transfer('Y', 'A')
                    else:
                        self.parent._emit_load('LDA', arg_loc)
                    self._emit_push('A', "Push u16 arg")

        elif param_size == 3:
            # 24-bit (far pointer): push byte by byte in m8, high byte first
            source_size = param_size
            if hasattr(arg.value, 'type_info') and arg.value.type_info:
                from r65.compiler.codegen.type_utils import get_type_size
                source_size = get_type_size(arg.value.type_info)

            self._ensure_m8_mode("8-bit A for far ptr push")
            bytes_pushed_so_far = 0
            for byte_idx in range(2, -1, -1):  # bank, high, low
                byte_name = {2: "bank", 1: "high", 0: "low"}[byte_idx]
                if byte_idx >= source_size:
                    self._emit_load_immediate('A', 0, f"Zero-extend {byte_name} byte")
                else:
                    if byte_idx > 0:
                        loc = self.parent._offset_location(arg_loc, byte_idx)
                    else:
                        loc = arg_loc
                    self.parent._emit_load('LDA', loc, f"Load {byte_name} byte")
                self._emit_push('A', f"Push {byte_name} byte")
                bytes_pushed_so_far += 1
                # Temporarily adjust tracker for intermediate PHAs
                if byte_idx > 0:  # Not the last byte
                    self.region_state.stack_tracker.push(1)
            # Undo all temporary adjustments (caller adds full param_size after return)
            self.region_state.stack_tracker.pop(bytes_pushed_so_far - 1)

    def _emit_caller_arg_cleanup(self, stack_bytes_pushed: int, returns_in_x: bool = False,
                                  returns_in_a: bool = True):
        """Emit caller-side cleanup of PHA-pushed arguments after call returns.

        Uses PLX to pop 2 bytes at a time, preserving the return value in A.
        X is safe to clobber here: it's either dead (callee clobbered it) or
        its pre-call value is saved by region spilling and will be reloaded later.

        When returns_in_x is True (multi-register return via A+X), uses PLY
        instead to avoid clobbering the X return value.

        For odd remaining bytes, saves A in a free register, pops with PLA,
        and restores A — unless the call returns void (returns_in_a=False),
        in which case just PLA directly.

        Also updates the stack tracker to reflect the removed bytes.

        Args:
            stack_bytes_pushed: Number of bytes that were PHA-pushed for args
            returns_in_x: True if the call returns a value in X that must be preserved
            returns_in_a: True if the call returns a value in A that must be preserved
        """
        if stack_bytes_pushed == 0:
            return

        remaining = stack_bytes_pushed

        if returns_in_x:
            # X holds a return value — use PLY (Y is 16-bit, pops 2 bytes)
            while remaining >= 2:
                self._emit_pull('Y', "Pop pushed arg bytes (preserve X)")
                remaining -= 2
            if remaining == 1:
                self._ensure_m8_mode("8-bit A for 1-byte pop")
                if returns_in_a:
                    # Save A in Y, pop 1 byte, restore A (X untouched)
                    self.parent.emitter.emit_instr(Opcode.TAY, comment="Save return A")
                    self._emit_pull('A', "Pop 1 pushed arg byte")
                    self.parent.emitter.emit_instr(Opcode.TYA, comment="Restore return A")
                else:
                    self._emit_pull('A', "Pop 1 pushed arg byte")
        else:
            # Normal path: PLX to pop 2 bytes at a time
            while remaining >= 2:
                self._emit_pull('X', "Pop pushed arg bytes")
                remaining -= 2
            if remaining == 1:
                self._ensure_m8_mode("8-bit A for 1-byte pop")
                if returns_in_a:
                    # Save A return value in X, pop 1 byte with PLA, restore A
                    self.parent.emitter.emit_instr(Opcode.TAX, comment="Save return A")
                    self._emit_pull('A', "Pop 1 pushed arg byte")
                    self.parent.emitter.emit_instr(Opcode.TXA, comment="Restore return A")
                else:
                    # Void call — no return value in A to preserve
                    self._emit_pull('A', "Pop 1 pushed arg byte")

        # Reduce stack tracker to undo the PHA tracking
        self.region_state.stack_tracker.pop(stack_bytes_pushed)

    def emit_register_argument(self, arg, arg_loc):
        """Emit register argument (move to specified register)."""
        from r65.compiler.codegen.type_utils import get_type_size

        target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)

        if arg_loc.is_hw(target_reg):
            return  # Already in correct register

        # For A register with 16-bit param type, switch to m16 before loading.
        # X/Y are always 16-bit regardless of M flag, but A depends on it.
        if target_reg == 'A' and arg.param_type is not None:
            if get_type_size(arg.param_type) == 2:
                current_mode = self.emitter.get_accu_mode()
                if current_mode == 8:
                    self._emit_immediate(Opcode.REP_IMMEDIATE, 0x20,
                                         "Switch to m16 for u16 @ A parameter")
                    self.emitter.emit_accu_mode(16)

        if target_reg == 'B':
            self._emit_b_register_argument(arg, arg_loc)
        elif isinstance(arg.value, MIRImmediate):
            self._emit_immediate_to_register(arg.value.value, target_reg)
        elif arg_loc.is_hw():
            # Source is a hardware register - emit transfer
            self._emit_transfer(arg_loc.hw_register, target_reg)
        else:
            self._emit_memory_to_register(arg_loc, target_reg)

    def _emit_b_register_argument(self, arg, arg_loc):
        """Emit B register argument (special handling)."""
        if isinstance(arg.value, MIRImmediate):
            self._emit_load_immediate('A', arg.value.value)
            self.parent._store_to_b_from_a()
        elif arg_loc.is_hw():
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

    def emit_variable_argument(self, arg, arg_loc):
        """Emit variable-bound argument (store to memory location)."""
        # Load into A
        if arg_loc.is_hw('A'):
            pass  # Already in A
        elif arg_loc.is_hw():
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

    def emit_scratch_param_argument(self, arg, arg_loc):
        """
        Emit scratch parameter argument (store to zero-page scratch address).

        Similar to variable-bound, but stores to a specific DP address
        assigned by the scratch parameter analysis pass.
        """
        from r65.compiler.codegen.type_utils import get_type_size
        from r65.compiler.codegen.constants import M_FLAG
        from r65.compiler.codegen.asm_nodes import Address

        scratch_addr = arg.scratch_addr
        param_size = 1
        if arg.param_type is not None:
            param_size = get_type_size(arg.param_type)
        elif hasattr(arg.value, 'type_info') and arg.value.type_info:
            param_size = get_type_size(arg.value.type_info)

        # Skip if value is already at the target scratch address (forwarding optimization)
        if (arg_loc.kind == LocationKind.SCRATCH and
                arg_loc.scratch_addr == scratch_addr and
                arg_loc.size >= param_size):
            return

        if param_size == 3:
            # 24-bit far pointer: store low word in m16, bank byte in m8
            self._ensure_m8_mode("8-bit mode before far ptr scratch")

            # Store low 16 bits
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for far ptr low word")
            self.parent.emitter.emit_accu_mode(16)
            if isinstance(arg.value, MIRImmediate):
                self._emit_load_immediate('A', arg.value.value & 0xFFFF)
            else:
                self.parent._emit_load('LDA', arg_loc)
            self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr),
                                   f"Scratch far ptr ${scratch_addr:02X} (low word)")
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "8-bit A for bank byte")
            self.parent.emitter.emit_accu_mode(8)

            # Store bank byte
            if isinstance(arg.value, MIRImmediate):
                self._emit_load_immediate('A', (arg.value.value >> 16) & 0xFF)
            else:
                byte2_loc = self.parent._offset_location(arg_loc, 2)
                self.parent._emit_load('LDA', byte2_loc)
            self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr + 2),
                                   f"Scratch far ptr ${scratch_addr + 2:02X} (bank)")
        elif param_size == 2:
            # 16-bit value: need m16 mode for single STA
            if arg_loc.is_hw('A'):
                # Already in A, check mode
                current_mode = self.parent.emitter.get_accu_mode()
                if current_mode != 16:
                    self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for scratch param")
                    self.parent.emitter.emit_accu_mode(16)
                self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr),
                                       f"Scratch param ${scratch_addr:02X}")
                self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
                self.parent.emitter.emit_accu_mode(8)
            elif isinstance(arg.value, MIRImmediate):
                self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for scratch param")
                self.parent.emitter.emit_accu_mode(16)
                self._emit_load_immediate('A', arg.value.value)
                self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr),
                                       f"Scratch param ${scratch_addr:02X}")
                self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
                self.parent.emitter.emit_accu_mode(8)
            else:
                self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for scratch param")
                self.parent.emitter.emit_accu_mode(16)
                self.parent._emit_load('LDA', arg_loc)
                self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr),
                                       f"Scratch param ${scratch_addr:02X}")
                self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
                self.parent.emitter.emit_accu_mode(8)
        else:
            # 8-bit value: standard LDA/STA in m8
            self._ensure_m8_mode("8-bit A for scratch param")

            if arg_loc.is_hw('A'):
                pass  # Already in A
            elif arg_loc.is_hw():
                if arg_loc.hw_register == 'X':
                    self._emit_transfer('X', 'A')
                elif arg_loc.hw_register == 'Y':
                    self._emit_transfer('Y', 'A')
            elif isinstance(arg.value, MIRImmediate):
                self._emit_load_immediate('A', arg.value.value)
            else:
                self.parent._emit_load('LDA', arg_loc)

            self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr),
                                   f"Scratch param ${scratch_addr:02X}")

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

    def _emit_call_instruction(self, instr: Call, stack_bytes_pushed: int = 0):
        """
        Emit the actual call instruction.

        Handles both direct and indirect calls.

        Args:
            instr: Call instruction
            stack_bytes_pushed: Bytes pushed for stack arguments (shifts fn ptr location)
        """
        if isinstance(instr.function, VirtualRegister):
            # Indirect call through function pointer
            self._emit_indirect_call_trampoline(instr.function, instr.is_far, stack_bytes_pushed)
        elif isinstance(instr.function, str):
            # Direct call
            if instr.is_far:
                self._emit_address(Opcode.JSL, instr.function)
            else:
                self._emit_address(Opcode.JSR, instr.function)
        else:
            raise InstructionSelectionError(f"Unknown function type in Call: {type(instr.function)}", source_loc=self.parent._current_source_loc)

    def _emit_indirect_call_trampoline(self, func_ptr_vreg: VirtualRegister, is_far: bool,
                                       stack_bytes_pushed: int = 0):
        """
        Generate trampoline for indirect function call through function pointer.

        The 65816 RTS/RTL instructions pop an address and add 1 before jumping.
        For a proper call, we need TWO addresses on the stack:
        1. The return address-1 (so the callee's RTS/RTL returns here)
        2. The target address-1 (so our RTS/RTL jumps to the callee)

        Near trampoline (16-bit address):
            ; Push return address (compile-time label, already -1)
            LDA #>(__ret_label - 1)
            PHA
            LDA #<(__ret_label - 1)
            PHA
            ; Push target address from function pointer
            LDA func_ptr+1  ; High byte
            PHA
            LDA func_ptr    ; Low byte
            PHA
            ; Subtract 1 from target (RTS adds 1)
            SEC / SBC chain on $01,S and $02,S
            RTS             ; → callee, callee's RTS → __ret_label
        __ret_label:

        Far trampoline (24-bit address):
            ; Push return address (compile-time, 3 bytes, already -1)
            LDA #:(__ret_label - 1)
            PHA
            LDA #>(__ret_label - 1)
            PHA
            LDA #<(__ret_label - 1)
            PHA
            ; Push target address (3 bytes)
            LDA func_ptr+2, +1, +0 with PHA drift
            ; Subtract 1 from target
            SEC / SBC chain on $01,S..$03,S
            RTL             ; → callee, callee's RTL → __ret_label
        __ret_label:

        Note: For stack-relative locations, each PHA changes the stack pointer,
        so subsequent loads need their offsets adjusted by +1 for each previous PHA.

        Args:
            func_ptr_vreg: VirtualRegister holding the function pointer
            is_far: True for far call (24-bit), False for near call (16-bit)
            stack_bytes_pushed: Bytes pushed for stack arguments before trampoline
        """
        from r65.compiler.codegen.asm_nodes import StackOffset, Immediate, Address
        from r65.compiler.codegen.register_alloc import PhysicalLocation

        ret_label = self.parent._get_unique_label()

        # Trampoline must run in m8 mode: each PHA pushes exactly 1 byte,
        # and RTS/RTL pops a fixed 2/3 byte address regardless of m flag.
        self._ensure_m8_mode("8-bit A for trampoline")

        ptr_loc = self.parent._get_operand_location(func_ptr_vreg)
        is_stack = ptr_loc.kind == LocationKind.STACK

        # Handle hardware register fn pointers: spill to scratch before trampoline
        # since we need byte-level access (offset) which hw registers don't support
        if ptr_loc.is_hw() and ptr_loc.hw_register in ('X', 'Y'):
            needed = 3 if is_far else 2
            scratch_addr = None
            param_scratch_addrs = set()
            cur_func = getattr(self.parent, 'current_function', None)
            if cur_func and hasattr(cur_func, 'scratch_param_addrs'):
                param_scratch_addrs = set(cur_func.scratch_param_addrs.values())
            if hasattr(self.parent.reg_alloc, 'scratch_pool') and self.parent.reg_alloc.scratch_pool:
                for scratch in self.parent.reg_alloc.scratch_pool.scratches:
                    if scratch.size >= needed and scratch.address not in param_scratch_addrs:
                        scratch_addr = scratch.address
                        break
            if scratch_addr is not None:
                # Store hw reg directly to scratch DP (STX/STY in x16 stores 2 bytes)
                store_op = Opcode.STX_DP if ptr_loc.hw_register == 'X' else Opcode.STY_DP
                self.emitter.emit_instr(store_op, Address(scratch_addr),
                                       f"Spill {ptr_loc.hw_register} fn ptr to scratch ${scratch_addr:02X}")
                ptr_loc = PhysicalLocation(
                    kind=LocationKind.SCRATCH,
                    scratch_addr=scratch_addr,
                    size=needed
                )
                is_stack = False
            else:
                raise InstructionSelectionError(
                    f"Cannot spill hardware register fn pointer for indirect call trampoline",
                    source_loc=self.parent._current_source_loc
                )

        # Adjust ptr_loc for stack arguments pushed before the trampoline
        if is_stack and stack_bytes_pushed > 0:
            ptr_loc = self.parent._offset_location(ptr_loc, stack_bytes_pushed)

        # Track how many bytes we've pushed (for stack-relative drift)
        pha_count = 0

        if is_far:
            # === Far call trampoline (24-bit address) ===

            # Step 1: Push return address - 1 (3 bytes: bank, high, low)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f":({ret_label} - 1)"),
                             "Return address bank")
            self._emit_push('A', "Push return bank")
            pha_count += 1
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">({ret_label} - 1)"),
                             "Return address high")
            self._emit_push('A', "Push return high")
            pha_count += 1
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<({ret_label} - 1)"),
                             "Return address low")
            self._emit_push('A', "Push return low")
            pha_count += 1

            # Step 2: Push target address (3 bytes from function pointer)
            bank_offset = 2 + (pha_count if is_stack else 0)
            bank_loc = self.parent._offset_location(ptr_loc, bank_offset)
            self.parent._emit_load('LDA', bank_loc, "Load target bank byte")
            self._emit_push('A', "Push target bank")
            pha_count += 1

            high_offset = 1 + (pha_count if is_stack else 0)
            high_loc = self.parent._offset_location(ptr_loc, high_offset)
            self.parent._emit_load('LDA', high_loc, "Load target high byte")
            self._emit_push('A', "Push target high")
            pha_count += 1

            low_offset = 0 + (pha_count if is_stack else 0)
            if low_offset > 0:
                low_loc = self.parent._offset_location(ptr_loc, low_offset)
            else:
                low_loc = ptr_loc
            self.parent._emit_load('LDA', low_loc, "Load target low byte")
            self._emit_push('A', "Push target low")

            # Step 3: Subtract 1 from target address (top 3 bytes on stack)
            self._emit_trampoline_address_adjust(3)

            self._emit_implied(Opcode.RTL, "Indirect far call via trampoline")

            # Step 4: Return label (callee's RTL returns here)
            self.emitter.emit_label(ret_label)
        else:
            # === Near call trampoline (16-bit address) ===

            # Step 1: Push return address - 1 (2 bytes: high, low)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">({ret_label} - 1)"),
                             "Return address high")
            self._emit_push('A', "Push return high")
            pha_count += 1
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<({ret_label} - 1)"),
                             "Return address low")
            self._emit_push('A', "Push return low")
            pha_count += 1

            # Step 2: Push target address (2 bytes from function pointer)
            high_offset = 1 + (pha_count if is_stack else 0)
            high_loc = self.parent._offset_location(ptr_loc, high_offset)
            self.parent._emit_load('LDA', high_loc, "Load target high byte")
            self._emit_push('A', "Push target high")
            pha_count += 1

            low_offset = 0 + (pha_count if is_stack else 0)
            if low_offset > 0:
                low_loc = self.parent._offset_location(ptr_loc, low_offset)
            else:
                low_loc = ptr_loc
            self.parent._emit_load('LDA', low_loc, "Load target low byte")
            self._emit_push('A', "Push target low")

            # Step 3: Subtract 1 from target address (top 2 bytes on stack)
            self._emit_trampoline_address_adjust(2)

            self._emit_implied(Opcode.RTS, "Indirect near call via trampoline")

            # Step 4: Return label (callee's RTS returns here)
            self.emitter.emit_label(ret_label)

    def _emit_trampoline_address_adjust(self, byte_count: int):
        """
        Subtract 1 from address bytes pushed on stack for RTS/RTL trampoline.

        RTS pops 2 bytes and adds 1 before jumping. RTL pops 3 bytes and adds 1.
        So we must push (target_address - 1) for the CPU to jump to the right place.

        Uses SEC/SBC chain with carry propagation to handle byte-boundary wrapping.
        Stack layout after PHAs: [$01,S=low, $02,S=high, $03,S=bank (if far)]

        Args:
            byte_count: Number of address bytes on stack (2 for near, 3 for far)
        """
        from r65.compiler.codegen.asm_nodes import StackOffset

        self._emit_implied(Opcode.SEC, "Subtract 1 from trampoline address")

        for i in range(byte_count):
            stack_pos = i + 1  # $01,S for low, $02,S for high, $03,S for bank
            subtract_val = 1 if i == 0 else 0  # Only subtract 1 from low byte; propagate borrow

            self._emit_instr(Opcode.LDA_STACK, StackOffset(stack_pos))
            self._emit_immediate(Opcode.SBC_IMMEDIATE, subtract_val)
            self._emit_instr(Opcode.STA_STACK, StackOffset(stack_pos))

    # ========================================================================
    # Return Value Collection
    # ========================================================================

    def _get_callee_return_registers(self, instr: Call):
        """
        Get the return register order for a callee based on its return type.

        Args:
            instr: Call instruction with callee info

        Returns:
            List of register names in order
        """
        from r65.compiler.codegen.constants import get_return_registers

        # Try to get callee's return type and mode from the Call instruction
        callee_return_type = getattr(instr, 'callee_return_type', None)
        callee_entry_mode = instr.callee_entry_m_mode if hasattr(instr, 'callee_entry_m_mode') else None

        if callee_return_type is not None:
            return get_return_registers(callee_return_type, callee_entry_mode)
        return ['A', 'X', 'Y']

    def _call_returns_in_a(self, instr) -> bool:
        """Check if a call returns a value in A (i.e., is not void)."""
        callee_return_type = getattr(instr, 'callee_return_type', None)
        if callee_return_type is None:
            # No return type info — conservatively assume it returns in A
            return len(getattr(instr, 'returns', [])) > 0
        from r65.compiler.hir.types import BasicTypeInfo
        if callee_return_type == BasicTypeInfo('void'):
            return False
        return True

    def _call_returns_in_x(self, instr) -> bool:
        """Check if a call returns a value in the X register.

        Used by _emit_caller_arg_cleanup to determine whether PLX would clobber
        a return value. When True, cleanup uses PLY instead of PLX.
        """
        return_regs = self._get_callee_return_registers(instr)
        callee_return_type = getattr(instr, 'callee_return_type', None)
        if callee_return_type is not None:
            from r65.compiler.hir.types import TupleTypeInfo
            if isinstance(callee_return_type, TupleTypeInfo):
                num_returns = len(callee_return_type.element_types)
            else:
                num_returns = 1
        else:
            num_returns = len(getattr(instr, 'returns', []))
        active_regs = return_regs[:num_returns]
        return 'X' in active_regs

    def _emit_return_value_collection(self, instr: Call):
        """
        Collect return values from registers.

        Return values come back in A, B, X, Y (in order) depending on
        the callee's return register ordering.
        """
        if not instr.returns:
            return

        return_registers = self._get_callee_return_registers(instr)

        for i, return_vreg in enumerate(instr.returns):
            if i >= len(return_registers):
                raise InstructionSelectionError(f"Too many return values (max {len(return_registers)})", source_loc=self.parent._current_source_loc)

            source_reg = return_registers[i]
            dest_loc = self.parent._get_operand_location(return_vreg)

            if dest_loc.is_hw(source_reg):
                pass  # Already in correct location
            elif source_reg == 'B':
                # B return: XBA to access B value in A, store, then XBA back
                self.parent._access_b_value_in_a()
                if dest_loc.is_hw():
                    if dest_loc.hw_register != 'A':
                        self._emit_return_register_transfer('A', dest_loc.hw_register)
                        self.parent._ensure_xba_state_normal()
                    # If dest is A, the value is already there after XBA - just
                    # need to NOT swap back since we want A to hold the value
                else:
                    self.parent._emit_store('STA', dest_loc)
                    self.parent._ensure_xba_state_normal()
            elif dest_loc.is_hw():
                self._emit_return_register_transfer(source_reg, dest_loc.hw_register)
            else:
                self._emit_return_store(source_reg, dest_loc)

    def _emit_pascal_return_value_collection(self, instr: Call, result_bytes: int):
        """Pull return value from stack result space (Pascal convention).

        After a Pascal call, the callee has cleaned up params and written
        the return value to the result space. The result space is now at TOS.
        We PLA the result into the destination.

        Args:
            instr: Call instruction
            result_bytes: Size of result space in bytes (1 or 2)
        """
        if not instr.returns:
            # Void function but somehow has result bytes — just pop them
            remaining = result_bytes
            while remaining >= 2:
                self._emit_pull('X', "Discard Pascal result space")
                remaining -= 2
            if remaining == 1:
                self._ensure_m8_mode("8-bit for discard")
                self._emit_pull('A', "Discard Pascal result space")
            return

        return_vreg = instr.returns[0]
        dest_loc = self.parent._get_operand_location(return_vreg)

        if result_bytes == 1:
            self._ensure_m8_mode("8-bit for Pascal result pull")
            self._emit_pull('A', "Pull Pascal result (u8)")
            if dest_loc.is_hw('A'):
                pass  # Already in A
            elif dest_loc.is_hw():
                if dest_loc.hw_register == 'X':
                    self._emit_transfer('A', 'X')
                elif dest_loc.hw_register == 'Y':
                    self._emit_transfer('A', 'Y')
            else:
                self.parent._emit_store('STA', dest_loc)
        elif result_bytes == 2:
            from r65.compiler.codegen.constants import M_FLAG
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit for Pascal result pull")
            self.parent.emitter.emit_accu_mode(16)
            self._emit_pull('A', "Pull Pascal result (u16)")
            if dest_loc.is_hw('A'):
                pass  # Already in A
            elif dest_loc.is_hw():
                if dest_loc.hw_register == 'X':
                    self._emit_transfer('A', 'X')
                elif dest_loc.hw_register == 'Y':
                    self._emit_transfer('A', 'Y')
            else:
                self.parent._emit_store('STA', dest_loc)
        elif result_bytes == 3:
            # 24-bit result (far pointer): pull byte by byte in m8
            self._ensure_m8_mode("8-bit for Pascal 3-byte result pull")
            # Stack has: low, high, bank (low at TOS)
            self._emit_pull('A', "Pull Pascal result low byte")
            self.parent._emit_store('STA', dest_loc)
            loc_hi = self.parent._offset_location(dest_loc, 1)
            self._emit_pull('A', "Pull Pascal result high byte")
            self.parent._emit_store('STA', loc_hi)
            loc_bank = self.parent._offset_location(dest_loc, 2)
            self._emit_pull('A', "Pull Pascal result bank byte")
            self.parent._emit_store('STA', loc_bank)

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
            raise unknown_value("built-in function", instr.builtin_name, source_loc=self.parent._current_source_loc)

        if builtin.kind == BuiltinKind.PROCESSOR_CONTROL:
            self._emit_processor_control_builtin(instr, builtin)
        elif builtin.kind == BuiltinKind.SOFTWARE_INTERRUPT:
            self._emit_software_interrupt_builtin(instr, builtin)
        elif builtin.kind == BuiltinKind.BLOCK_MOVE:
            self._emit_block_move_builtin(instr, builtin)
        elif builtin.kind in (BuiltinKind.ARITHMETIC, BuiltinKind.SHIFT):
            self._emit_runtime_builtin(instr, builtin)
        elif builtin.kind == BuiltinKind.CONST_MATH:
            from r65.compiler.codegen.errors import CodegenError
            raise CodegenError(
                f"const math builtin '{instr.builtin_name}' must be resolved at compile time "
                f"(only usable in const fn or with constant arguments)",
                source_loc=self.parent._current_source_loc
            )

    def _emit_processor_control_builtin(self, instr: Call, builtin):
        """Emit processor control built-in (wai, stp, xba, NOP)."""
        opcode = BUILTIN_OPCODES.get(builtin.instruction)
        if not opcode:
            raise unknown_value("processor control builtin", builtin.instruction, source_loc=self.parent._current_source_loc)

        if instr.builtin_name == 'NOP':
            count = 1  # Default
            if len(instr.args) == 1:
                arg = instr.args[0]
                if isinstance(arg.value, MIRImmediate):
                    count = arg.value.value
                else:
                    raise InstructionSelectionError("NOP() count must be a constant immediate value", source_loc=self.parent._current_source_loc)

            for _ in range(count):
                self._emit_implied(opcode)
        else:
            self._emit_implied(opcode)

    def _emit_software_interrupt_builtin(self, instr: Call, builtin):
        """Emit software interrupt built-in (cop)."""
        if len(instr.args) != 1:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 1 argument, got {len(instr.args)}", source_loc=self.parent._current_source_loc)

        opcode = BUILTIN_OPCODES.get(builtin.instruction)
        if not opcode:
            raise unknown_value("software interrupt builtin", builtin.instruction, source_loc=self.parent._current_source_loc)

        arg = instr.args[0]

        # COP requires an immediate signature byte
        if isinstance(arg.value, MIRImmediate):
            self._emit_immediate(opcode, arg.value.value)
        else:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() requires a constant signature byte", source_loc=self.parent._current_source_loc)

    def _emit_block_move_builtin(self, instr: Call, builtin):
        """Emit block move built-in (mvn, mvp)."""
        if len(instr.args) != 2:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 2 arguments, got {len(instr.args)}", source_loc=self.parent._current_source_loc)

        src_bank_arg = instr.args[0]
        dst_bank_arg = instr.args[1]

        if not isinstance(src_bank_arg.value, MIRImmediate) or not isinstance(dst_bank_arg.value, MIRImmediate):
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects immediate bank numbers", source_loc=self.parent._current_source_loc)

        src_bank = src_bank_arg.value.value
        dst_bank = dst_bank_arg.value.value

        opcode = BUILTIN_OPCODES.get(builtin.instruction)
        if not opcode:
            raise unknown_value("block move builtin", builtin.instruction, source_loc=self.parent._current_source_loc)

        self.emitter.emit_instr(opcode, BlockMove(src_bank, dst_bank))

    def _emit_runtime_builtin(self, instr: Call, builtin):
        """Emit runtime library built-in (mul, div, mod, shl, shr)."""
        if len(instr.args) != 2:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 2 arguments, got {len(instr.args)}", source_loc=self.parent._current_source_loc)

        # Load first argument into A
        arg0 = instr.args[0]
        arg0_loc = self.parent._get_operand_location(arg0.value)
        if isinstance(arg0.value, MIRImmediate):
            self._emit_load_immediate('A', arg0.value.value)
        elif arg0_loc.is_hw('A'):
            pass  # Already in A
        elif arg0_loc.is_hw('X'):
            self._emit_implied(Opcode.TXA)  # Transfer X to A
        elif arg0_loc.is_hw('Y'):
            self._emit_implied(Opcode.TYA)  # Transfer Y to A
        else:
            self.parent._emit_load('LDA', arg0_loc)

        # Load second argument into X
        arg1 = instr.args[1]
        arg1_loc = self.parent._get_operand_location(arg1.value)
        if isinstance(arg1.value, MIRImmediate):
            self._emit_load_immediate('X', arg1.value.value)
        elif arg1_loc.is_hw('X'):
            pass  # Already in X
        elif arg1_loc.is_hw('A'):
            self._emit_implied(Opcode.TAX)  # Transfer A to X
        elif arg1_loc.is_hw('Y'):
            # Y to X requires going through A (no direct TYX on 65816)
            # Save A, TYA, TAX, restore A
            self._emit_implied(Opcode.PHA)
            self._emit_implied(Opcode.TYA)
            self._emit_implied(Opcode.TAX)
            self._emit_implied(Opcode.PLA)
        else:
            self.parent._emit_load('LDX', arg1_loc)

        # Call runtime library function
        runtime_func_name = f"__builtin_{instr.builtin_name}"
        self._emit_address(Opcode.JSR, runtime_func_name)

        # Store result if needed
        if instr.returns:
            return_vreg = instr.returns[0]
            dest_loc = self.parent._get_operand_location(return_vreg)

            if dest_loc.is_hw('A'):
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

        Updates the stack tracker because PLD pops 2 bytes from the frame,
        shifting SP up. Without this, stack-relative offsets for storing
        the call's return value would be off by 2 bytes.
        """
        self._emit_instr(Opcode.PLD, comment="Restore D from stack before call")
        self.region_state.stack_tracker.pop(2)

    def _emit_d_push_only(self):
        """
        Push D back onto the stack after a call.

        Since we used PLD before the call, we must push D back so the
        epilogue's PLD will pop the correct value. This is used when there
        are no more far pointer dereferences after this call.

        Updates the stack tracker to undo the displacement from PLD.
        """
        self._emit_instr(Opcode.PHD, comment="Save D back to stack for epilogue")
        self.region_state.stack_tracker.push(2)

    def _compute_live_regs_after_call(self, instr: Call, reloads: List[SpillInfo]) -> set:
        """
        Determine which hardware registers (A, X, Y) are live after return
        value collection and spill reloads (steps 5 through 6.5).

        A register is live if:
        - A return value landed in A and was hw-coalesced (stayed in A)
        - A was reloaded from a spill

        X/Y are live if:
        - A return value was hw-coalesced to X or Y
        - X or Y was reloaded from a spill

        Args:
            instr: The Call instruction
            reloads: Spill reloads that were emitted in step 6.5

        Returns:
            Set of live hardware register names (e.g. {'A'}, {'A', 'X'})
        """
        live = set()

        # Check spill reloads — any reloaded register is live
        for spill in reloads:
            live.add(spill.hw_reg)

        # Check return values hw-coalesced to registers
        if instr.returns:
            return_registers = self._get_callee_return_registers(instr)
            for i, return_vreg in enumerate(instr.returns):
                if i >= len(return_registers):
                    break
                source_reg = return_registers[i]
                dest_loc = self.parent._get_operand_location(return_vreg)
                if dest_loc.is_hw(source_reg):
                    # Return value stayed in its source register (hw-coalesced)
                    live.add(source_reg)

        return live

    def _emit_d_equals_s_restore(self, live_regs: set):
        """
        Re-establish D = S after a call for continued far pointer access.

        After a call returns, if there are more far pointer dereferences,
        we need to:
        1. Push D back onto the stack (since PLD popped it before the call)
        2. Set D = S for continued far pointer access

        TSC clobbers A, so we use a tiered strategy based on which registers
        are live at this point:

        Tier 1 - A is dead (common: void calls, returns stored to stack):
          PHD          ; push D (2 bytes)
          TSC          ; A = SP (clobbers dead A)
          TCD          ; D = SP = frame_base

        Tier 2 - A is live, X or Y is dead:
          PHD          ; push D
          TAX/TAY      ; save A in dead register
          TSC          ; A = SP
          TCD          ; D = SP = frame_base
          TXA/TYA      ; restore A

        Tier 3 - A, X, Y all live (extremely rare):
          PHD          ; push D (2 bytes), SP = frame_base
          PHA          ; save A (1 byte), SP = frame_base - 1
          REP #$20     ; 16-bit for correct INC across page boundary
          TSC          ; A(16) = SP = frame_base - 1
          INC A        ; A(16) = frame_base
          TCD          ; D = frame_base
          SEP #$20     ; back to 8-bit
          PLA          ; restore A, SP = frame_base, D = SP

        Updates the stack tracker to undo the displacement from PLD.
        """
        from r65.compiler.codegen.constants import M_FLAG

        a_live = 'A' in live_regs

        if not a_live:
            # Tier 1: A is dead — fast path (3 instructions, ~8 cycles)
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            self.region_state.stack_tracker.push(2)
            self._emit_instr(Opcode.TSC, comment="A = SP (A is dead)")
            self._emit_instr(Opcode.TCD, comment="D = SP = frame_base (D = S)")
        elif 'X' not in live_regs:
            # Tier 2a: A live, X dead — use TAX/TXA (5 instructions, ~12 cycles)
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            self.region_state.stack_tracker.push(2)
            self._emit_instr(Opcode.TAX, comment="Save A in X")
            self._emit_instr(Opcode.TSC, comment="A = SP")
            self._emit_instr(Opcode.TCD, comment="D = SP = frame_base (D = S)")
            self._emit_instr(Opcode.TXA, comment="Restore A from X")
        elif 'Y' not in live_regs:
            # Tier 2b: A live, Y dead — use TAY/TYA (5 instructions, ~12 cycles)
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            self.region_state.stack_tracker.push(2)
            self._emit_instr(Opcode.TAY, comment="Save A in Y")
            self._emit_instr(Opcode.TSC, comment="A = SP")
            self._emit_instr(Opcode.TCD, comment="D = SP = frame_base (D = S)")
            self._emit_instr(Opcode.TYA, comment="Restore A from Y")
        else:
            # Tier 3: All registers live — PHA fallback (8 instructions)
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            self.region_state.stack_tracker.push(2)
            self._emit_instr(Opcode.PHA, comment="Save A before D=S setup")
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for D=S setup")
            self.parent.emitter.emit_accu_mode(16)
            self._emit_instr(Opcode.TSC, comment="A(16) = SP")
            self._emit_instr(Opcode.INC, comment="A(16) = SP + 1 (frame_base)")
            self._emit_instr(Opcode.TCD, comment="D = frame_base (D = S)")
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
            self._emit_instr(Opcode.PLA, comment="Restore A after D=S setup")

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

    # ========================================================================
    # DBR Management for SET_DBR Strategy
    # ========================================================================

    def _emit_dbr_save_before_call(self):
        """Save ptr-bank DBR and set DBR to code bank before a call.

        Under SET_DBR strategy, DBR is set to the far pointer's bank.
        Callees expect DBR to be the code bank (bank 0 for LoROM), so we
        save DBR (PHB), push code bank (PHK), and pop it into DBR (PLB).
        """
        self._emit_instr(Opcode.PHB, comment="Save ptr-bank DBR before call")
        self.region_state.stack_tracker.push(1)
        self._emit_instr(Opcode.PHK, comment="Push code bank")
        self._emit_instr(Opcode.PLB, comment="Set DBR to code bank for callee")
        # PHK pushes 1, PLB pops 1 — net zero for stack tracker

    def _emit_dbr_restore_after_call(self):
        """Restore ptr-bank DBR after a call returns."""
        self._emit_instr(Opcode.PLB, comment="Restore ptr-bank DBR after call")
        self.region_state.stack_tracker.pop(1)
