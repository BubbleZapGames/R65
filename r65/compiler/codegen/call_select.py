# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Call instruction selector: Function calls and built-ins.

Handles function call generation including argument setup, call emission,
return value collection, and built-in function expansion.

Includes hardware register spill/reload around calls based on liveness
and callee's #[preserves()] attribute.

Region-based spilling:
Instead of spilling around each call, we identify "clobber regions" -
maximal sequences of calls where a register is live but not used. We save
once at the start of the region and restore once at the end, reducing
redundant push/pull operations.
"""

from typing import List, Set, Dict, FrozenSet, NamedTuple, Optional, TYPE_CHECKING
from r65.compiler.mir.nodes import (
    Call, CallBase, MemoryLocation, TraitDispatch, VirtualRegister, ArgumentMechanism,
    Immediate as MIRImmediate, ChainRole, iter_operands,
)

if TYPE_CHECKING:
    from r65.compiler.mir.liveness import ClobberRegion
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.codegen.abi_model import ABIKind, emit_d_equals_s_keeping_c
from r65.compiler.codegen.opcodes import (
    Opcode, TRANSFER_OPCODES, PUSH_OPCODES, PULL_OPCODES,
    LOAD_IMMEDIATE_OPCODES, STORE_MNEMONICS,
)
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.errors import unknown_value
from r65.compiler.codegen.base_selector import BaseSelector
from r65.compiler.codegen.type_utils import arg_param_size, scalar_width
from r65.compiler.codegen.arg_order import arg_sort_key
from r65.compiler.codegen.clobber_summary import code_writes
from r65.compiler.mir.call_effects import INDEX_REGS, CallEffects, call_effects


class SpillInfo(NamedTuple):
    """Information about a hardware register that needs spilling."""
    vreg: Optional[VirtualRegister]  # The virtual register allocated to the hw reg, or None for direct hw reg usage
    hw_reg: str            # Hardware register name ('A', 'X', 'Y')
    spill_mode: Optional[int] = None  # For A register: mode at spill time (8 or 16), None for X/Y
    # Stack spill location will be managed via push/pull


class ActiveRegionState:
    """
    Tracks active clobber regions during code generation.

    For X and Y, tracks whether we're currently inside a clobber region (have
    saved but not yet restored).

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
        # Track pending A spill info for reload (need to restore mode)
        self.pending_a_spill: Optional[SpillInfo] = None
        # Track pending per-call X/Y spills for reload
        # When X/Y are spilled via per-call fallback (no region), we need to
        # emit PLX/PLY after the call. Maps hw_reg -> SpillInfo.
        self.pending_xy_spills: Dict[str, SpillInfo] = {}
        # Registers on the stack, in push order. X and Y regions are found
        # one register at a time, so they need not nest: a reload pulls
        # everything above its register first (_emit_hw_reloads).
        self.spill_stack: List[str] = []
        # Registers pulled early that way while their region goes on: their
        # region's next call pushes them again.
        self.respill: Set[str] = set()

    def set_block_regions(self, block_id: int, regions: Dict[str, List['ClobberRegion']]):
        """Set pre-computed regions for a block."""
        self.block_regions[block_id] = regions
        self.current_block_id = block_id
        # Clear active regions when entering new block
        self.active_regions.clear()
        # Clear any pending per-call X/Y spills
        self.pending_xy_spills.clear()
        self.spill_stack.clear()
        self.respill.clear()

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

    Uses region-based spilling: instead of spilling around each call,
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
        # The vregs allocation placed in A (spilled per call, not by region)
        self._a_vregs: List[VirtualRegister] = []
        # call_effects per call of the current function, by id(instr)
        self._effects: Dict[int, CallEffects] = {}
        # Where each block's code starts, and where each dispatch had put
        # self in Y, by node index (_self_still_in_y)
        self._block_starts: Dict[int, int] = {}
        self._self_in_y: Dict[int, int] = {}

    def initialize_regions_for_function(self):
        """
        Pre-compute clobber regions for all blocks in the current function.

        Should be called once per function before processing any blocks.
        """
        self._effects = {}
        self._block_starts = {}
        self._self_in_y = {}
        reg_alloc = self.parent.reg_alloc
        if not reg_alloc or not reg_alloc.instr_liveness:
            self._function_regions = None
            return

        from r65.compiler.mir.liveness import ClobberRegionAnalyzer

        hw_of = {vid: loc.hw_register for vid, loc in reg_alloc.allocations.items()
                 if loc.kind == LocationKind.HARDWARE}
        analyzer = ClobberRegionAnalyzer(reg_alloc.instr_liveness, hw_of)
        self._function_regions = analyzer.analyze_function(self.effects)
        self._region_analyzer = analyzer
        a_ids = {vid for vid, reg in hw_of.items() if reg == 'A'}
        self._a_vregs = list({v.id: v for block in self.parent.current_function.blocks.values()
                              for instr in block.instructions
                              for _, v in iter_operands(instr)
                              if type(v) is VirtualRegister and v.id in a_ids}.values())

    def initialize_regions_for_block(self, block_id: int):
        """
        Initialize region tracking state for a specific block.

        Should be called when starting to process a new block.

        Args:
            block_id: The block ID being processed
        """
        self._block_starts[block_id] = len(self.parent.emitter.nodes)
        if self._function_regions is None:
            # No pre-computed regions, will fall back to per-call spilling
            self.region_state = ActiveRegionState()
            return

        # Only X and Y get regions; A takes the per-call path.
        regions = self._function_regions.get(block_id, {'X': [], 'Y': []})
        self.region_state.set_block_regions(block_id, regions)

    def effects(self, instr) -> CallEffects:
        """The register effects of *instr* (mir/call_effects), with the
        callee's clobber summary when it has been generated."""
        effects = self._effects.get(id(instr))
        if effects is None:
            summaries = getattr(self.parent.func_gen, 'register_summaries', None)
            effects = self._effects[id(instr)] = call_effects(instr, summaries)
        return effects

    def callee_preserves(self, instr) -> Set[str]:
        """The registers a call leaves as they were: the callee's
        `#[preserves(...)]` plus the index registers its code never writes."""
        return set(self.effects(instr).preserved)

    def _clobbered_saves(self, instr) -> List[str]:
        """The D/DBR this call saves around itself, from the callee's
        `#[clobbers(...)]`: none for a callee that never returns, and no DBR
        when the caller's databank=caller bracket restores it already."""
        from r65.compiler.hir.attributes import DataBankMode
        from r65.compiler.hir.types import NeverTypeInfo
        saves = self.effects(instr).caller_saves
        if not saves or isinstance(instr.callee_return_type, NeverTypeInfo):
            return []
        if (instr.is_far and instr.bank_attr and instr.mode_attr
                and instr.mode_attr.databank == DataBankMode.CALLER):
            saves = saves - {'DBR'}
        if saves and getattr(instr, 'pascal_result_bytes', 0) > 0:
            raise InstructionSelectionError(
                f"call to '{instr.function}', which clobbers "
                f"{', '.join(sorted(saves))}, cannot return a value under --abi pascal",
                source_loc=self.parent._current_source_loc)
        return [reg for reg in ('D', 'DBR') if reg in saves]

    def _check_inferred_preserves(self, instr, start: int, inferred: Set[str]):
        """A call kept an index register unsaved on its callee's summary (or
        an extern's declaration): the code emitted around the JSR must not
        write it either - except to land the call's own result there."""
        from r65.compiler.codegen.asm_nodes import Address, Instruction
        rest = [n for n in self.parent.emitter.nodes[start:]
                if not (isinstance(n, Instruction) and n.opcode in (Opcode.JSR, Opcode.JSL)
                        and n.operand == Address(instr.function))]
        clobbered = ((code_writes(rest) & inferred)
                     - self._call_result_registers(instr))
        if clobbered:
            raise InstructionSelectionError(
                f"internal: call to '{instr.function}' writes {sorted(clobbered)} "
                f"while its callee summary keeps it unsaved",
                source_loc=self.parent._current_source_loc)

    # ========================================================================
    # Call-Specific Emission Helpers
    # ========================================================================

    def _emit_transfer(self, source: str, dest: str):
        """Emit a register transfer instruction."""
        opcode = TRANSFER_OPCODES.get((source, dest))
        if opcode:
            # TAX/TAY in m8/x16 transfer B:A into the 16-bit index register.
            # B may hold stale data from earlier 16-bit operations, so force
            # m16 to transfer the full 16-bit accumulator (which is what we
            # mean when targeting an x16 index register).
            if source == 'A' and dest in ('X', 'Y'):
                self.parent._ensure_m16_mode()
            self._emit_implied(opcode)

    def _emit_push(self, reg: str, comment: str = None):
        """Emit a push of *reg*, a PUSH_OPCODES key ('DBR' for PHB; B has no push)."""
        self._emit_implied(PUSH_OPCODES[reg], comment)

    def _emit_pull(self, reg: str, comment: str = None):
        """Emit a pull into *reg*, a PULL_OPCODES key ('DBR' for PLB; B has no pull)."""
        self._emit_implied(PULL_OPCODES[reg], comment)

    def _emit_load_immediate(self, reg: str, value: int, comment: str = None):
        """Emit a load immediate instruction."""
        opcode = LOAD_IMMEDIATE_OPCODES.get(reg)
        if opcode:
            self._emit_immediate(opcode, value, comment)

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

        call_start = len(self.parent.emitter.nodes)
        effects = self.effects(instr)
        # The index registers kept on the callee's word alone - its summary,
        # or an extern's #[clobbers(...)] - which nothing here saves
        inferred = effects.inferred - effects.declared
        if instr.callee_is_extern:
            inferred |= effects.declared & INDEX_REGS

        # Step 0: Compute hardware register spills needed
        # Spill hw registers that are live across this call and not preserved by callee
        spills = self._compute_hw_spills(instr)

        # Step 0.5: Restore D register BEFORE anything is pushed (for far pointer
        # functions). CRITICAL: the D the prologue saved is on top of the frame,
        # so PLD must come before the spills and the argument pushes - after
        # them it would pop a spilled register or part of an argument instead.
        # The stack tracker records the PLD displacement so that all subsequent
        # stack-relative accesses (argument loading, return value storing) are
        # automatically adjusted.
        if needs_d_management:
            self._emit_d_restore_before_call()

        # Step 0.7: Emit spills BEFORE argument setup (to avoid clobbering args)
        self._emit_hw_spills(spills)

        # Step 0.7b: SET_DBR: save ptr-bank DBR and set DBR to the data bank for callee
        if needs_dbr_management:
            self._emit_dbr_save_before_call()

        # Step 0.8: Save the D/DBR the callee's #[clobbers(...)] lists
        caller_saves = self._clobbered_saves(instr)
        for reg in caller_saves:
            self._emit_push(reg, f"Save {reg} (callee clobbers it)")

        # Step 1: Set up arguments
        stack_bytes_pushed = self._emit_argument_setup(instr)

        # Step 2: Handle caller-managed DBR (databank=caller)
        needs_dbr_restore = self._emit_caller_dbr_setup(instr)

        # Step 2.6: Switch to callee's entry mode if needed
        # Callee's prologue expects to be in entry_m_mode
        self._emit_entry_mode_switch(instr)

        # Step 3: Make the call
        self._emit_call_instruction(instr, stack_bytes_pushed)
        pascal_result_bytes = getattr(instr, 'pascal_result_bytes', 0)
        if self.parent.abi_model.kind == ABIKind.PASCAL and stack_bytes_pushed > 0:
            # A Pascal callee removes its parameters before it returns, so by
            # here only the result space is left of what was pushed.
            self.emitter.note_stack(-(stack_bytes_pushed - pascal_result_bytes))

        # Invalidate XBA state after call (function may have modified A/B)
        self.parent._invalidate_xba_state()

        # Step 3.5: Update emitter mode to reflect callee's exit mode
        # This is critical - the callee may exit in a different mode than entry
        self._update_mode_after_call(instr)

        # Step 4: Restore DBR if caller-managed
        if needs_dbr_restore:
            self._emit_pull('DBR', "Restore data bank (caller)")

        # Step 4.5: Restore the saved D/DBR before collecting the results,
        # whose stores may go through either. The pushed arguments are above
        # them, so those come off first, into no register holding a result.
        if caller_saves:
            if stack_bytes_pushed > 0 and self.parent.abi_model.kind != ABIKind.PASCAL:
                self._emit_caller_arg_cleanup(
                    stack_bytes_pushed,
                    set(effects.results) | (self.callee_preserves(instr) & INDEX_REGS))
                stack_bytes_pushed = 0
            for reg in reversed(caller_saves):
                self._emit_pull(reg, f"Restore {reg} (callee clobbers it)")

        # Step 5: Collect return values (in callee's exit mode)
        if pascal_result_bytes > 0:
            # Pascal: callee cleaned params, result space is at TOS.
            # PLA the result from the stack instead of reading from registers.
            self._emit_pascal_return_value_collection(instr, pascal_result_bytes)
            stack_bytes_pushed = 0
        elif stack_bytes_pushed > 0 and self.parent.abi_model.kind == ABIKind.PASCAL:
            # Pascal void call: callee cleaned all params, nothing remains.
            stack_bytes_pushed = 0
        else:
            self._emit_return_value_collection(instr)

        # Step 5.5: Caller cleanup of PHA-pushed stack args
        # Done after return value collection, so the cleanup knows which
        # registers still hold a result and pulls into none of them. Nor
        # into one the callee preserves: it was not spilled, so it may hold
        # a live value.
        if stack_bytes_pushed > 0:
            self._emit_caller_arg_cleanup(
                stack_bytes_pushed,
                self._call_live_after_collection(instr)
                | (self.callee_preserves(instr) & INDEX_REGS))

        # Step 6: Restore mode after receiving return value
        # If callee exited in m16 (u16 return), switch back to m8
        self._emit_exit_mode_restore(instr)

        # Step 6.2: SET_DBR: restore ptr-bank DBR after call. Its PHB went on
        # above the spills, so it comes off before they are reloaded.
        if needs_dbr_management:
            self._emit_dbr_restore_after_call()

        # Step 6.5: Reload spilled hardware registers (region-based)
        # Only reload registers where this is the last call in the region
        reloads = self._compute_hw_reloads(instr)
        if needs_d_management:
            reloads = self._reload_all_spills(reloads)
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

        if inferred:
            self._check_inferred_preserves(instr, call_start, inferred)

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

        # In a far-pointer function D (D=S) or DBR (SET_DBR) is the caller's
        # own; the method expects the program's, as any callee does
        # (select_call). Nested the same way: PLD before the spills, PHB
        # after them.
        from r65.compiler.mir.nodes import FarPtrStrategy
        func = self.parent.current_function
        needs_d_management = (func is not None and func.has_far_ptr_stack_params
                              and func.far_ptr_strategy != FarPtrStrategy.SET_DBR)
        needs_dbr_management = (func is not None
                                and func.far_ptr_strategy == FarPtrStrategy.SET_DBR)

        # A dispatch preserves nothing (call_effects), so this spills every
        # register live across it
        spills = self._compute_hw_spills(instr)
        if needs_d_management:
            self._emit_d_restore_before_call()
        self._emit_hw_spills(spills)
        if needs_dbr_management:
            self._emit_dbr_save_before_call()

        # Chain coalescing: SOLO and START dispatches push PHB; SOLO and END
        # dispatches pull PLB. MIDDLE dispatches do neither — DBR is held
        # at self's bank across the whole run. See
        # analysis/trait_chain_coalesce.analyze_trait_dispatch_chains.
        # A MIDDLE/END member of either kind skips its Y load when Y still
        # holds self from the member before (_self_still_in_y). A held DBR
        # cannot nest with the per-call D/DBR above, so those dispatches
        # each stand alone.
        role = getattr(instr, 'self_chain_role', ChainRole.SOLO)
        if needs_d_management or needs_dbr_management:
            role = ChainRole.SOLO
        emit_phb = instr.self_is_far and role in (ChainRole.SOLO, ChainRole.START)
        emit_dbr_set = instr.self_is_far and role in (ChainRole.SOLO, ChainRole.START)
        emit_plb = instr.self_is_far and role in (ChainRole.SOLO, ChainRole.END)

        if emit_phb:
            self._emit_push('DBR', "Save caller's DBR for far self dispatch")

        # Set up arguments (same mechanism as regular calls). The ABI's
        # emit_trait_dispatch_args defers self loading to us via
        # _pending_self_y_arg so we can choose to skip the DBR-set step
        # for MIDDLE/END chain members.
        self._pending_self_y_arg = None
        self._pending_self_y_stack_bytes = 0
        stack_bytes_pushed = self._emit_trait_dispatch_args(instr)

        self_y_arg = self._pending_self_y_arg
        self_y_stack_bytes = self._pending_self_y_stack_bytes
        if self_y_arg is not None:
            if instr.self_is_far:
                if emit_dbr_set:
                    self._set_dbr_from_far_self(self_y_arg, self_y_stack_bytes)
                if not self._self_still_in_y(instr):
                    self._load_y_addr_from_far_self(self_y_arg, self_y_stack_bytes)
            elif not self._self_still_in_y(instr):
                self.load_y_with_self(self_y_arg, self_y_stack_bytes)
            self._self_in_y[id(instr)] = len(self.parent.emitter.nodes)

        # Emit the call to the dispatch wrapper
        if instr.is_far:
            self._emit_address(Opcode.JSL, dispatch_name)
        else:
            self._emit_address(Opcode.JSR, dispatch_name)

        # Invalidate XBA state after dispatch
        self.parent._invalidate_xba_state()

        # The method returns in its exit mode, not the width the caller
        # reached the wrapper in
        self._update_mode_after_call(instr)

        # Collect return values
        self._emit_return_value_collection(instr)

        # Caller cleanup of PHA-pushed stack args
        if stack_bytes_pushed > 0:
            self._emit_caller_arg_cleanup(stack_bytes_pushed, self._call_live_after_collection(instr))

        # Restore caller's DBR after far self dispatch
        if emit_plb:
            self._emit_pull('DBR', "Restore caller's DBR after far self dispatch")
        if needs_dbr_management:
            self._emit_dbr_restore_after_call()

        # Reload spilled registers
        reloads = self._compute_hw_reloads(instr)
        if needs_d_management:
            reloads = self._reload_all_spills(reloads)
        self._emit_hw_reloads(reloads)

        if needs_d_management:
            if self._has_far_ptr_derefs_after_call(instr):
                self._emit_d_equals_s_restore(self._compute_live_regs_after_call(instr, reloads))
            else:
                self._emit_d_push_only()

    def _self_still_in_y(self, instr: TraitDispatch) -> bool:
        """Whether Y still holds self from the chain member before *instr*.

        Read off the code emitted since that member put self in Y - its
        dispatch, by the implementors' summaries, and everything after -
        which is all the code that ran in between if every block on the
        path between the two (`self_y_path`) was emitted inside that run.
        """
        start = self._self_in_y.get(instr.self_y_after)
        summaries = self._dispatch_summaries()
        if start is None or summaries is None:
            return False
        now = len(self.parent.emitter.nodes)
        if not all(start < self._block_starts.get(b, -1) <= now
                   for b in instr.self_y_path[1:]):
            return False
        return 'Y' not in code_writes(self.parent.emitter.nodes[start:now], summaries)

    def _dispatch_summaries(self) -> Optional[Dict[str, FrozenSet[str]]]:
        """The clobber summaries, with each dispatch wrapper's: what the
        wrapper writes and every implementor it may reach does. None at
        -O0, where there are none."""
        func_gen = self.parent.func_gen
        summaries = getattr(func_gen, 'register_summaries', None)
        graph = getattr(func_gen, 'call_graph', None)
        if not summaries or graph is None:
            return None
        from r65.compiler.codegen.codegen import DISPATCH_WRAPPER_WRITES
        summaries = dict(summaries)
        for (trait, method), impls in graph.trait_impls.items():
            if impls and all(i in summaries for i in impls):
                summaries[f"{trait}__{method}__dispatch"] = DISPATCH_WRAPPER_WRITES.union(
                    *(summaries[i] for i in impls))
        return summaries

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

    def _far_self_arg_loc(self, arg: 'Argument', stack_bytes_pushed: int):
        """Resolve the location of a far-self pointer argument, adjusted for
        any caller-pushed bytes on the stack."""
        arg_loc = self.parent._get_operand_location(arg.value)
        if arg_loc.kind == LocationKind.STACK and stack_bytes_pushed > 0:
            arg_loc = self.parent._offset_location(arg_loc, stack_bytes_pushed)
        return arg_loc

    def _set_dbr_from_far_self(self, arg: 'Argument', stack_bytes_pushed: int):
        """Load bank byte from a 3-byte far self pointer and set DBR.

        Emits: LDA <bank-byte-source>; PHA; PLB.
        """
        arg_loc = self._far_self_arg_loc(arg, stack_bytes_pushed)

        if isinstance(arg.value, MIRImmediate):
            val = arg.value.value
            bank = (val >> 16) & 0xFF
            self.parent._emit_immediate(Opcode.LDA_IMMEDIATE, bank, f"Bank byte ${bank:02X}")
        elif arg_loc.kind in (LocationKind.SCRATCH, LocationKind.STACK, LocationKind.MEMORY):
            bank_loc = self.parent._offset_location(arg_loc, 2)
            self.parent._emit_load('LDA', bank_loc, "Load bank byte from far self ptr")
        else:
            raise InstructionSelectionError(
                f"Cannot load far self bank from location kind {arg_loc.kind}",
                source_loc=self.parent._current_source_loc
            )
        self._emit_push('A', "Push bank byte")
        self._emit_pull('DBR', "Set DBR to object's bank")

    def _load_y_addr_from_far_self(self, arg: 'Argument', stack_bytes_pushed: int):
        """Load the 16-bit address of a 3-byte far self pointer into Y."""
        from r65.compiler.codegen.constants import M_FLAG

        arg_loc = self._far_self_arg_loc(arg, stack_bytes_pushed)

        if isinstance(arg.value, MIRImmediate):
            val = arg.value.value
            addr = val & 0xFFFF
            self.parent._emit_immediate(Opcode.LDY_IMMEDIATE, addr, f"Load self addr ${addr:04X} into Y")
        elif arg_loc.kind == LocationKind.SCRATCH:
            self.parent._emit_load('LDY', arg_loc, "Load self addr into Y")
        elif arg_loc.kind == LocationKind.STACK:
            # No LDY d,S on 65816 — use LDA d,S; TAY
            self.parent._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for stack self addr load")
            self.parent.emitter.emit_accu_mode(16)
            self.parent._emit_load('LDA', arg_loc, "Load self addr from stack")
            self.parent.emitter.emit_raw("    TAY")
            self.parent._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
        elif arg_loc.kind == LocationKind.MEMORY:
            self.parent._emit_load('LDY', arg_loc, "Load self addr into Y")
        else:
            raise InstructionSelectionError(
                f"Cannot load far self addr from location kind {arg_loc.kind}",
                source_loc=self.parent._current_source_loc
            )

    def load_y_with_far_self(self, arg: 'Argument', stack_bytes_pushed: int):
        """Load Y register and set DBR from a 3-byte far self pointer for trait dispatch.

        For a far self pointer (3 bytes: addr_low, addr_high, bank):
        1. Load bank byte (offset +2) into A
        2. PHA; PLB — set DBR to object's bank (net stack change = 0)
        3. Load 16-bit address (offset +0) into Y

        Equivalent to ``_set_dbr_from_far_self`` followed by
        ``_load_y_addr_from_far_self``. Used for direct (non-dispatch) calls
        with a far self (``ABIModel._emit_self_y``); dispatch calls the two
        helpers separately to skip the DBR set on MIDDLE/END chain members.
        """
        self._set_dbr_from_far_self(arg, stack_bytes_pushed)
        self._load_y_addr_from_far_self(arg, stack_bytes_pushed)

    # ========================================================================
    # Hardware Register Spill/Reload (Region-Based)
    # ========================================================================

    def _compute_hw_spills(self, instr: CallBase) -> List[SpillInfo]:
        """
        Compute which hardware registers need spilling for this call.

        X and Y spill by region: only at the call that starts one (or that
        resumes one pulled early, see _emit_hw_reloads). A region covers a
        direct use of the register and every vreg allocation placed in it.
        A spills per call, when a vreg placed in A is live after it.

        Args:
            instr: The Call instruction

        Returns:
            List of SpillInfo for registers that need spilling at THIS call
        """
        reg_alloc = self.parent.reg_alloc
        if not reg_alloc:
            return []
        pos = (reg_alloc.instr_liveness.get_instruction_position(instr)
               if reg_alloc.instr_liveness else None)
        if self._function_regions is None or pos is None:
            return self._spills_without_liveness(instr)
        block_id, instr_idx = pos

        preserved = self.callee_preserves(instr)
        # A register the call defines is never spilled: spilling would save
        # garbage and the matching PLX/PLY would clobber the result.
        call_return_set = self._call_result_registers(instr)

        spills: List[SpillInfo] = []
        for reg_name in ('A', 'X', 'Y'):
            if reg_name in preserved or reg_name in call_return_set:
                continue
            if reg_name == 'A':
                il = reg_alloc.instr_liveness
                if any(il.is_live_after(v, block_id, instr_idx) for v in self._a_vregs):
                    spills.append(SpillInfo(vreg=None, hw_reg='A'))
                continue
            region = self.region_state.get_region_for_call(reg_name, instr_idx)
            if region is None:
                continue
            if self.region_state.is_first_call_in_region(reg_name, instr_idx):
                spills.append(SpillInfo(vreg=None, hw_reg=reg_name))
                self.region_state.mark_region_active(reg_name, region)
            elif reg_name in self.region_state.respill:
                # Pulled early to reach a register below it
                spills.append(SpillInfo(vreg=None, hw_reg=reg_name))
                self.region_state.respill.discard(reg_name)

        # Push what is restored last first, so the regions starting here nest:
        # A, reloaded after this call, goes on top; an X/Y region by its last
        # call, latest deepest.
        def ends(spill):
            region = self.region_state.get_region_for_call(spill.hw_reg, instr_idx)
            return region.clobbering_calls[-1] if region is not None else -1
        spills.sort(key=ends, reverse=True)
        return spills

    def _spills_without_liveness(self, instr: CallBase) -> List[SpillInfo]:
        """Per-call spills with no liveness to consult: every register a vreg
        is bound to, unless the call keeps it or returns in it."""
        reg_alloc = self.parent.reg_alloc
        skip = self.callee_preserves(instr) | self._call_result_registers(instr)
        spills = []
        for reg_name in ('A', 'X', 'Y'):
            hw_alloc = reg_alloc.get_hw_alloc(reg_name)
            if reg_name not in skip and hw_alloc.allocated_vreg is not None and hw_alloc.is_bound:
                spills.append(SpillInfo(vreg=hw_alloc.allocated_vreg, hw_reg=reg_name))
        return spills

    def _compute_hw_reloads(self, instr: CallBase) -> List[SpillInfo]:
        """
        Compute which hardware registers need reloading after this call.

        Region-based for X/Y: only returns registers where this call is the
        LAST clobbering call in the region. A is per-call: reload it if it was
        spilled for this call.

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

        # Reloading a register the call defines would pull the caller's old
        # value back over the result.
        call_return_set = self._call_result_registers(instr)

        # Check each active X/Y region
        for hw_reg in ('X', 'Y'):
            if hw_reg in call_return_set:
                # Don't reload — value just placed there is the return value.
                # The matching spill was also skipped, so the stack stays
                # balanced. The region analysis opened this region because it
                # sees only the ABI result registers (call_effects), not the
                # register a result is collected into (_call_result_registers).
                if self.region_state.is_region_active(hw_reg):
                    self.region_state.clear_active_region(hw_reg)
                continue
            if self.region_state.is_region_active(hw_reg):
                if self.region_state.is_last_call_in_region(hw_reg, instr_idx):
                    # Last call in region - need to reload
                    reloads.append(SpillInfo(vreg=None, hw_reg=hw_reg))
                    # Clear active region
                    self.region_state.clear_active_region(hw_reg)

        # A is always spilled per call: reload it if it was spilled for this
        # call, unless the call defines A (its spill was skipped).
        if self.region_state.pending_a_spill is not None:
            # Only add if not already handled by region reload above
            if 'A' not in call_return_set and not any(r.hw_reg == 'A' for r in reloads):
                reloads.append(self.region_state.pending_a_spill)
            elif 'A' in call_return_set:
                # Spill was skipped at the call site, clear the pending marker
                self.region_state.pending_a_spill = None

        # X/Y spilled per call (_spills_without_liveness): pulled right after.
        for hw_reg in ('X', 'Y'):
            if hw_reg in self.region_state.pending_xy_spills:
                if hw_reg in call_return_set:
                    # Spill was skipped, drop the pending marker too.
                    del self.region_state.pending_xy_spills[hw_reg]
                    continue
                reloads.append(self.region_state.pending_xy_spills[hw_reg])
                del self.region_state.pending_xy_spills[hw_reg]

        return reloads

    def _reload_all_spills(self, reloads: List[SpillInfo]) -> List[SpillInfo]:
        """*reloads* plus every register still spilled, for a D=S function.

        The PHD after the call must land where the prologue's did, and the
        D=S it re-establishes must be the frame base the [dp] offsets assume,
        so nothing may stay pushed across it. A region with calls still to
        come is pushed again at its next one (`respill`).
        """
        wanted = {spill.hw_reg for spill in reloads}
        extra = []
        for reg in self.region_state.spill_stack:
            if reg not in wanted:
                extra.append(SpillInfo(vreg=None, hw_reg=reg))
                self.region_state.respill.add(reg)
        return reloads + extra

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
                self.region_state.spill_stack.append('A')
                # Save spill info for reload
                self.region_state.pending_a_spill = spill_with_mode
            else:
                # X/Y are always 16-bit (2 bytes)
                self._emit_push(spill.hw_reg, f"Spill {spill.hw_reg} (region start)")
                self.region_state.spill_stack.append(spill.hw_reg)
                # Track per-call spill for X/Y when no region is active.
                # Region-based spills have mark_region_active() called before
                # _emit_hw_spills(), so is_region_active() distinguishes the paths.
                if not self.region_state.is_region_active(spill.hw_reg):
                    self.region_state.pending_xy_spills[spill.hw_reg] = spill

    def _emit_hw_reloads(self, spills: List[SpillInfo]):
        """
        Emit pull instructions to reload spilled hardware registers.

        Pulls in the reverse of the push order, whatever order `spills` is
        in. A register pushed after one of these (a Y region opened inside an
        X region that ends first) is pulled too, to reach it, and pushed again
        at the next call of its region.
        Updates the spill offset to track stack shrinkage.

        For region-based spilling, this is only called at region end.

        For A register reloads, ensures we're in the same mode used when spilling,
        since PHA/PLA push/pull different amounts based on m8 vs m16 mode.

        Args:
            spills: List of registers to reload (will be processed in reverse)
        """
        from r65.compiler.codegen.constants import M_FLAG

        stack = self.region_state.spill_stack
        wanted = {spill.hw_reg: spill for spill in spills}
        on_stack = [reg for reg in wanted if reg in stack]
        if on_stack:
            depth = min(stack.index(reg) for reg in on_stack)
            for reg in stack[depth:]:
                if reg not in wanted:
                    if reg not in self.region_state.active_regions:
                        raise InstructionSelectionError(
                            f"internal: {reg} spilled above a reloaded register "
                            f"outside any clobber region",
                            source_loc=self.parent._current_source_loc)
                    wanted[reg] = SpillInfo(vreg=None, hw_reg=reg)
                    self.region_state.respill.add(reg)
            ordered = ([wanted[reg] for reg in reversed(stack[depth:])]
                       + [s for s in reversed(spills) if s.hw_reg not in stack])
            del stack[depth:]
        else:
            ordered = list(reversed(spills))

        for spill in ordered:
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

                    # Clear pending A spill
                    self.region_state.pending_a_spill = None
                else:
                    # Fallback: no mode info, assume m8
                    self._emit_pull('A', "Reload A (region end)")
            else:
                # X/Y are always 16-bit (2 bytes)
                self._emit_pull(spill.hw_reg, f"Reload {spill.hw_reg} (region end)")

    def get_current_spill_offset(self) -> int:
        """
        Get the current spill offset for adjusting stack-relative accesses.

        Returns:
            Number of bytes currently pushed for spilling
        """
        return self.parent.emitter.displacement

    # ========================================================================
    # Argument Setup
    # ========================================================================

    def _emit_argument_setup(self, instr: Call, pre_arg_stack_adj: int = 0) -> int:
        """Set up call arguments. Delegates to ABIModel.emit_call_args."""
        return self.parent.abi_model.emit_call_args(self, instr, pre_arg_stack_adj)

    arg_sort_key = staticmethod(arg_sort_key)

    def emit_pha_stack_argument(self, arg, arg_loc, param_size: int):
        """Push one stack argument via PHA (Default and Pascal pass every stack arg this way).

        Source locations are placed past everything pushed so far, the bytes
        of this argument included, by _emit_load -> _get_opcode_for_location
        reading the emitter's displacement.

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
                # Zero-extend u8 → u16. Stack order requires high byte pushed
                # first (deeper), low byte pushed last (top). When the source
                # itself is in A, we can't do `LDA #0; PHA` first without
                # destroying it — switch to m16 and mask instead so the single
                # 16-bit PHA pushes high=0, low=source atomically.
                if arg_loc.is_hw('A'):
                    self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG,
                                         "16-bit A for zero-ext push (source in A)")
                    self.parent.emitter.emit_accu_mode(16)
                    self._emit_immediate(Opcode.AND_IMMEDIATE, 0x00FF,
                                         "Mask high byte for zero-ext")
                    self._emit_push('A', "Push u16 (high=0, low=source)")
                else:
                    self._ensure_m8_mode("8-bit A for zero-ext push")
                    self._emit_load_immediate('A', 0, "Zero high byte")
                    self._emit_push('A', "Push high byte (zero)")
                    if arg_loc.is_hw('X'):
                        self._emit_transfer('X', 'A')
                    elif arg_loc.is_hw('Y'):
                        self._emit_transfer('Y', 'A')
                    elif isinstance(arg.value, MIRImmediate):
                        self._emit_load_immediate('A', arg.value.value & 0xFF)
                    else:
                        self.parent._emit_load('LDA', arg_loc)
                    self._emit_push('A', "Push low byte")
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
            if isinstance(arg.value, MIRImmediate) and source_size == 3:
                # A constant address: its bank byte, then PEA pushes the
                # word high byte first, so it lands low byte on top.
                from r65.compiler.codegen.asm_nodes import Immediate
                bank_ref, word_ref = self._far_constant_refs(arg.value)
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(bank_ref), "Bank byte")
                self._emit_push('A', "Push bank byte")
                self._emit_instr(Opcode.PEA, Immediate(word_ref), "Push address")
                return
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

    def _far_constant_refs(self, value: 'MIRImmediate'):
        """`(bank_ref, word_ref)` operands for a constant far address: a
        number, or a static's address plus an offset."""
        if value.symbol is None:
            return f"${(value.value >> 16) & 0xFF:02X}", f"${value.value & 0xFFFF:04X}"
        _, bank_ref, word_ref = self.parent.move_selector.variable_address_refs(
            value.symbol, value.symbol_offset)
        return bank_ref, word_ref

    def _emit_caller_arg_cleanup(self, stack_bytes_pushed: int, live_regs: Set[str]):
        """Pop the PHA-pushed arguments after the call returns.

        `live_regs` (from _call_live_after_collection) are the registers that
        hold a result at this point; the cleanup writes none of them. Byte
        pairs are pulled into X, or into Y when X is live; an odd byte goes
        through PLA, with A parked in that index register when A is live.
        When X and Y are both live, C (A alone for a single byte) is stored
        over the deepest argument bytes, SP steps past the rest, and C is
        pulled back last, so every register survives.

        Also updates the stack tracker to reflect the removed bytes.
        """
        from r65.compiler.codegen.asm_nodes import StackOffset
        if stack_bytes_pushed == 0:
            return

        remaining = stack_bytes_pushed
        puller = next((r for r in ('X', 'Y') if r not in live_regs), None)

        if puller is None:
            if remaining == 1:
                self._ensure_m8_mode("8-bit A for 1-byte pop")
                self.parent.emitter.emit_instr(Opcode.STA_STACK, StackOffset(1),
                                               "Park A over the pushed arg byte")
            else:
                self._ensure_m16_mode("16-bit A to park C")
                self.parent.emitter.emit_instr(Opcode.STA_STACK, StackOffset(remaining - 1),
                                               "Park C over the top pushed arg bytes")
                if remaining > 2:
                    self._emit_implied(Opcode.TSC)
                    self._emit_implied(Opcode.CLC)
                    self._emit_immediate(Opcode.ADC_IMMEDIATE, remaining - 2,
                                         "Drop the arg bytes below it")
                    self._emit_implied(Opcode.TCS)
                    self.parent.emitter.note_stack(-(remaining - 2))
            self._emit_pull('A', "Pull the parked value back")
        else:
            while remaining >= 2:
                self._emit_pull(puller, "Pop pushed arg bytes")
                remaining -= 2
            if remaining == 1:
                self._ensure_m8_mode("8-bit A for 1-byte pop")
                if 'A' in live_regs:
                    to_idx, from_idx = ((Opcode.TAX, Opcode.TXA) if puller == 'X'
                                        else (Opcode.TAY, Opcode.TYA))
                    self.parent.emitter.emit_instr(to_idx, comment="Save return A")
                    self._emit_pull('A', "Pop 1 pushed arg byte")
                    self.parent.emitter.emit_instr(from_idx, comment="Restore return A")
                else:
                    self._emit_pull('A', "Pop 1 pushed arg byte")


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

        # An 8-bit A or B argument is set up in m8: in m16 a load into A
        # writes B too, which may already hold the `@ B` argument.
        if target_reg == 'B' or (target_reg == 'A' and (
                arg.param_type is None or get_type_size(arg.param_type) == 1)):
            self.parent._ensure_m8_mode()

        if target_reg == 'B':
            self._emit_b_register_argument(arg, arg_loc)
        elif isinstance(arg.value, MIRImmediate):
            self._emit_immediate_to_register(arg.value.value, target_reg)
        elif arg_loc.is_hw():
            # Source is a hardware register - emit transfer
            self._emit_transfer(arg_loc.hw_register, target_reg)
        else:
            self._emit_memory_to_register(arg_loc, target_reg)

    def emit_argument_setup(self, arg, arg_loc):
        """Emit what puts one non-stack argument in place.

        A VARIABLE argument needs nothing: CallLowerer's MIR Store, sized by
        the parameter type, wrote the variable before the Call. A SELF_Y
        argument is loaded after every other one (`_emit_self_y`).
        """
        if arg.mechanism == ArgumentMechanism.REGISTER:
            self.emit_register_argument(arg, arg_loc)
        elif arg.mechanism == ArgumentMechanism.SCRATCH_PARAM:
            self.emit_scratch_param_argument(arg, arg_loc)

    def arg_setup_clobbers_a(self, arg, arg_loc) -> bool:
        """Does emitting this argument destroy whatever is currently in A?

        Answered by emitting the setup on trial and reading what the emitted
        instructions write, so it cannot disagree with the emission. `XBA`
        counts: it is a swap, so even an A-resident source is destroyed when
        the old B lands in A.
        """
        with self.parent.trial_emission() as trial:
            self.emit_argument_setup(arg, arg_loc)
            return 'A' in code_writes(trial.nodes)

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
            # X/Y are 16-bit (x16). LDA + TAX/TAY must run in m16, otherwise
            # LDA loads only the low byte and TAX in m8/x16 transfers B:A
            # (with stale B) into the 16-bit index — corrupting the high byte.
            self.parent._ensure_m16_mode()
            self.parent._emit_load('LDA', arg_loc)
            if target_reg == 'X':
                self._emit_implied(Opcode.TAX, "Transfer to X (no LDX sr,S)")
            else:
                self._emit_implied(Opcode.TAY, "Transfer to Y (no LDY sr,S)")
        else:
            mnemonic = {'A': 'LDA', 'X': 'LDX', 'Y': 'LDY'}.get(target_reg)
            if mnemonic:
                self.parent._emit_load(mnemonic, arg_loc)

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
        param_size = arg_param_size(arg)

        # Determine source size for zero-extension check
        source_size = param_size
        if hasattr(arg.value, 'type_info') and arg.value.type_info:
            source_size = get_type_size(arg.value.type_info)

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

            # Store bank byte. When the source vreg is narrower than the far
            # pointer (e.g. a near pointer literal being promoted to far at
            # the call site), reading byte+2 would fall outside the allocated
            # location and pick up garbage from the adjacent slot. Zero-extend
            # the bank byte in that case — for static refs whose bank is not
            # bank 0, callers must use an explicit near→far conversion before
            # the call.
            if isinstance(arg.value, MIRImmediate):
                self._emit_load_immediate('A', (arg.value.value >> 16) & 0xFF)
            elif source_size < 3:
                self._emit_load_immediate('A', 0)
            else:
                byte2_loc = self.parent._offset_location(arg_loc, 2)
                self.parent._emit_load('LDA', byte2_loc)
            self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr + 2),
                                   f"Scratch far ptr ${scratch_addr + 2:02X} (bank)")
        elif param_size == 2:
            needs_zero_ext = source_size < 2

            if needs_zero_ext:
                # Source is u8 but param is u16: store byte-by-byte in m8 to avoid
                # garbage in the high byte of A when switching to m16.
                self._ensure_m8_mode("8-bit A for zero-ext scratch param")
                if arg_loc.is_hw('A'):
                    pass  # Already in A low byte
                elif arg_loc.is_hw('X'):
                    self._emit_transfer('X', 'A')
                elif arg_loc.is_hw('Y'):
                    self._emit_transfer('Y', 'A')
                elif isinstance(arg.value, MIRImmediate):
                    self._emit_load_immediate('A', arg.value.value & 0xFF)
                else:
                    self.parent._emit_load('LDA', arg_loc)
                self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr),
                                       f"Scratch param ${scratch_addr:02X} (low byte)")
                self._emit_load_immediate('A', 0)
                self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr + 1),
                                       f"Scratch param ${scratch_addr + 1:02X} (high byte, zero-ext)")
            else:
                # 16-bit value: need m16 mode for single STA
                # A single 16-bit STA. The mode is left at m16: what follows
                # switches only if it needs to (the callee's entry mode is put
                # in place before the call), so adjacent u16 scratch arguments
                # share one REP.
                if self.parent.emitter.get_accu_mode() != 16:
                    self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for scratch param")
                    self.parent.emitter.emit_accu_mode(16)
                if arg_loc.is_hw('A'):
                    pass
                elif arg_loc.is_hw('X') or arg_loc.is_hw('Y'):
                    # In m16 TXA/TYA transfer the whole 16-bit register
                    self._emit_transfer('X' if arg_loc.is_hw('X') else 'Y', 'A')
                elif isinstance(arg.value, MIRImmediate):
                    self._emit_load_immediate('A', arg.value.value)
                else:
                    self.parent._emit_load('LDA', arg_loc)
                self.emitter.emit_instr(Opcode.STA_DP, Address(scratch_addr),
                                       f"Scratch param ${scratch_addr:02X}")
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

        # Caller manages DBR: save, set, call, restore.
        # bank_attr.bank_number is a WLA bank *index*; the 65816 bank *byte*
        # adds the ROM base (.BASE $80 for LoROM FastROM, $C0 for HiROM).
        callee_bank = (getattr(self.emitter, 'rom_base', 0)
                       + instr.bank_attr.bank_number)
        self._emit_push('DBR', "Save current data bank (caller)")
        self._emit_load_immediate('A', callee_bank, "Load callee's bank number")
        self._emit_push('A', "Push bank number")
        self._emit_pull('DBR', "Set data bank for callee")
        return True

    # ========================================================================
    # Cross-Mode Call Handling
    # ========================================================================

    def _emit_entry_mode_switch(self, instr: Call):
        """
        Switch to callee's entry mode before making the call.

        The callee expects to receive arguments in its entry mode. Pin the
        accumulator to the callee's entry width, in either direction.

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
            # Callee returned in m16: stay there (the tracked mode already
            # says so). What follows switches if it needs m8; the u16 result
            # is usually stored straight away, in m16.
            return
        elif instr.callee_entry_m_mode == ModeState.M16:
            # Caller switched to m16 to pass a u16 @ A argument; the callee
            # returns in m8 (already m8 at runtime via its own epilogue), but the
            # caller's asm-tracked mode is still m16 — the entry REP is
            # propagated across the JSR by the asm-mode dataflow, which can't see
            # the callee's exit mode. Emit a SEP so the dataflow and the WLA
            # `.ACCU` directive agree with the m8 runtime mode. Without it a
            # following m8 `LDA #imm` is assembled 3 bytes wide and the trailing
            # $00 byte executes as BRK. Runtime no-op (mode is already m8).
            self._emit_immediate(Opcode.SEP_IMMEDIATE, 0x20, "Resync m8 after u16-arg call")
            self.emitter.emit_accu_mode(8)
        # If callee exited in m8 with an m8 entry, we're already in m8

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
        elif isinstance(instr.function, MemoryLocation):
            # Far call through the pointer a bank-0 static holds, read in
            # place by JML [abs] (optimize/memory_operand.py folds the load).
            loc = self.parent._get_operand_location(instr.function)
            addr = loc.memory_addr if loc.memory_addr is not None else loc.memory_label
            self._emit_dp_indirect_far_call(addr, comment=str(instr.function))
        else:
            raise InstructionSelectionError(f"Unknown function type in Call: {type(instr.function)}", source_loc=self.parent._current_source_loc)

    def _dp_offset_for_indirect_call(self, func_ptr_vreg: VirtualRegister):
        """Return the scratch address for a JML [abs] far-indirect-call fast
        path, or None if the fast path doesn't apply.

        Fires only for SCRATCH locations: the fn pointer lives in zeropage
        scratch slots (DP=$0000 by default), so JML [<scratch_addr>] reads
        the 3-byte target directly from zeropage. No D-state coupling.

        Only SCRATCH qualifies: JML [abs] reads its pointer from bank 0 at an
        absolute address, never D-relative, so a stack slot cannot be
        addressed this way. See docs/future-work.md.

        Hardware-resident pointers and MEMORY locations also return None.
        """
        ptr_loc = self.parent._get_operand_location(func_ptr_vreg)

        if ptr_loc.kind == LocationKind.SCRATCH:
            return ptr_loc.scratch_addr

        return None

    def _emit_dp_indirect_far_call(self, dp_offset, comment: str = ""):
        """Emit the JML [abs] fast-path sequence for a far indirect call.

        Sequence (14 cycles total):

            PHK                ; push current PBR              (3 cyc)
            PEA <ret-1         ; push 16-bit return-1          (5 cyc)
            JML [<dp_offset>]  ; absolute indirect long        (6 cyc)
        ret:

        JML [abs] reads the pointer from $00:abs: a zeropage scratch slot
        while D=0, or a low-RAM or direct-page static. *dp_offset* is that
        address, or a label for it.

        The callee's RTL pops 3 bytes (lo, hi, pbr) and increments — control
        returns to ``ret`` with the original PBR restored. Same calling
        convention as a real JSL.

        Stack-tracker bookkeeping: PHK + PEA push 3 bytes; the callee's RTL
        pops 3 bytes. Net change at ``ret`` is 0. Mirrors the existing
        trampoline, which also nets zero on the stack tracker.

        Caller has already verified preconditions:
          - is_far == True
          - the pointer is in a scratch slot or a bank-0 static
        """
        from r65.compiler.codegen.asm_nodes import Immediate, Address

        ret_label = self.parent._get_unique_label()

        # PHK and PEA push 1 and 2 bytes whatever M is, but the callee is
        # entered in m8: an indirect call has no callee mode to follow, and
        # the trampoline it stands in for always enters in m8.
        self._ensure_m8_mode("m8 for indirect callee")
        phk_comment = (
            f"Push PBR for far indirect call ({comment})"
            if comment else "Push PBR for far indirect call"
        )
        self._emit_implied(Opcode.PHK, phk_comment)
        # PEA expression: WLA-DX accepts `PEA label-1` directly. We avoid
        # outer parens because PEA's syntax can be ambiguous with indirect
        # forms, and the binary-minus has higher precedence than what
        # follows in this emitter's token stream.
        self._emit_instr(
            Opcode.PEA,
            Immediate(f"{ret_label}-1"),
            "Push return address - 1",
        )
        self._emit_instr(
            Opcode.JMP_INDIRECT_LONG,
            Address(dp_offset),
            f"Far indirect call via JML [{_format_jml_operand(dp_offset)}]",
        )
        self.emitter.emit_label(ret_label)
        # The callee's RTL took back the three bytes PHK and PEA pushed.
        self.emitter.note_stack(-3)

    def _emit_indirect_call_trampoline(self, func_ptr_vreg: VirtualRegister, is_far: bool,
                                       stack_bytes_pushed: int = 0):
        """
        Generate trampoline for indirect function call through function pointer.

        Far indirect calls have a fast path: when the pointer is in a scratch
        slot, the call lowers to PHK / PEA / JML [abs] (14 cycles, ~7 bytes). Otherwise, fall through to
        the generic RTS/RTL trampoline below (~78 cycles, ~30 bytes).
        Near indirect calls always use the generic trampoline (no JML [abs] for
        16-bit pointers; JMP_INDIRECT only reads 2 bytes from the
        current bank, which would not synthesize a proper near JSR).

        See docs/register_memory_config.md §2.5 for the fast-path rationale.

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

        Note: each PHA moves the stack pointer; the emitter counts it, so a
        stack-relative pointer slot loaded after it is placed past it.

        Args:
            func_ptr_vreg: VirtualRegister holding the function pointer
            is_far: True for far call (24-bit), False for near call (16-bit)
            stack_bytes_pushed: Bytes pushed for stack arguments before trampoline
        """
        from r65.compiler.codegen.asm_nodes import StackOffset, Immediate, Address
        from r65.compiler.codegen.register_alloc import PhysicalLocation

        # Fast path: PHK / PEA / JML [abs] for far indirect calls when the
        # pointer is in a scratch slot. Saves ~64 cycles vs the trampoline.
        if is_far:
            dp_offset = self._dp_offset_for_indirect_call(func_ptr_vreg)
            if dp_offset is not None:
                self._emit_dp_indirect_far_call(
                    dp_offset, comment=f"vreg {func_ptr_vreg}"
                )
                return

        ret_label = self.parent._get_unique_label()

        # Trampoline must run in m8 mode: each PHA pushes exactly 1 byte,
        # and RTS/RTL pops a fixed 2/3 byte address regardless of m flag.
        self._ensure_m8_mode("8-bit A for trampoline")

        ptr_loc = self.parent._get_operand_location(func_ptr_vreg)

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
            else:
                raise InstructionSelectionError(
                    f"Cannot spill hardware register fn pointer for indirect call trampoline",
                    source_loc=self.parent._current_source_loc
                )

        # The pointer's slot is frame-relative: the resolver adds every byte
        # pushed since (stack arguments and the trampoline's own pushes).


        if is_far:
            # === Far call trampoline (24-bit address) ===

            # Step 1: Push return address - 1 (3 bytes: bank, high, low)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f":({ret_label} - 1)"),
                             "Return address bank")
            self._emit_push('A', "Push return bank")
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">({ret_label} - 1)"),
                             "Return address high")
            self._emit_push('A', "Push return high")
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<({ret_label} - 1)"),
                             "Return address low")
            self._emit_push('A', "Push return low")

            # Step 2: Push target address (3 bytes from function pointer)
            bank_offset = 2
            bank_loc = self.parent._offset_location(ptr_loc, bank_offset)
            self.parent._emit_load('LDA', bank_loc, "Load target bank byte")
            self._emit_push('A', "Push target bank")

            high_offset = 1
            high_loc = self.parent._offset_location(ptr_loc, high_offset)
            self.parent._emit_load('LDA', high_loc, "Load target high byte")
            self._emit_push('A', "Push target high")

            low_offset = 0
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
            # Our RTL took the target and the callee's the return address:
            # six bytes pushed that no pull in this function took back.
            self.emitter.note_stack(-6)
        else:
            # === Near call trampoline (16-bit address) ===

            # Step 1: Push return address - 1 (2 bytes: high, low)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">({ret_label} - 1)"),
                             "Return address high")
            self._emit_push('A', "Push return high")
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<({ret_label} - 1)"),
                             "Return address low")
            self._emit_push('A', "Push return low")

            # Step 2: Push target address (2 bytes from function pointer)
            high_offset = 1
            high_loc = self.parent._offset_location(ptr_loc, high_offset)
            self.parent._emit_load('LDA', high_loc, "Load target high byte")
            self._emit_push('A', "Push target high")

            low_offset = 0
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
            # Our RTS took the target and the callee's the return address.
            self.emitter.note_stack(-4)

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

    def _call_result_registers(self, instr) -> Set[str]:
        """Registers a call defines: what must not be spilled, reloaded or
        pulled into around it.

        The callee's return registers, plus the hw-allocated destination of
        each return vreg (_emit_return_value_collection's TAX/TAY lands the
        value there). `instr.returns` may be empty when the values are
        captured by separate Moves, so the count comes from
        callee_return_type.
        """
        regs = set(self.effects(instr).results)
        for return_vreg in instr.returns:
            dest_loc = self.parent._get_operand_location(return_vreg)
            if dest_loc.is_hw() and dest_loc.hw_register in ('A', 'X', 'Y'):
                regs.add(dest_loc.hw_register)
        return regs

    def _call_live_after_collection(self, instr) -> Set[str]:
        """Registers holding a result once the return values are collected.

        The hw-allocated destination of each return vreg, plus the callee's
        return register of each value no return vreg collects (separate
        Moves capture those later, so they stay in place). A result stored
        to memory leaves its register dead; a void call leaves nothing.
        """
        results = self.effects(instr).results
        if not results:
            return set()
        regs = set(results[len(instr.returns):])
        for return_vreg in instr.returns:
            dest_loc = self.parent._get_operand_location(return_vreg)
            if dest_loc.is_hw() and dest_loc.hw_register in ('A', 'X', 'Y'):
                regs.add(dest_loc.hw_register)
        return regs

    def _emit_return_value_collection(self, instr: Call):
        """
        Collect return values from registers.

        Return values come back in A, B, X, Y (in order) depending on
        the callee's return register ordering.
        """
        if not instr.returns:
            return

        return_registers = self.effects(instr).results
        # Per-return-value size table (in bytes), parallel to instr.returns.
        # u8 returns need zero-extension when transferred A→X/Y because the
        # B register (high byte of A in m16) holds whatever stale value it
        # had before the call — without an explicit AND #$00FF, TAX would
        # place B-junk into the X high byte.
        return_sizes = self._get_return_value_sizes(instr)

        for i, return_vreg in enumerate(instr.returns):
            if i >= len(return_registers):
                raise InstructionSelectionError(f"Too many return values (max {len(return_registers)})", source_loc=self.parent._current_source_loc)

            source_reg = return_registers[i]
            dest_loc = self.parent._get_operand_location(return_vreg)
            return_size = return_sizes[i] if i < len(return_sizes) else 1

            if dest_loc.is_hw(source_reg):
                pass  # Already in correct location
            elif source_reg == 'B':
                # B return: XBA to access B value in A, store, then XBA back
                self.parent._access_b_value_in_a()
                if dest_loc.is_hw():
                    if dest_loc.hw_register != 'A':
                        self._emit_return_register_transfer('A', dest_loc.hw_register, return_size)
                        self.parent._ensure_xba_state_normal()
                    # If dest is A, the value is already there after XBA - just
                    # need to NOT swap back since we want A to hold the value
                else:
                    self.parent._emit_store('STA', dest_loc)
                    self.parent._ensure_xba_state_normal()
            elif dest_loc.is_hw():
                self._emit_return_register_transfer(source_reg, dest_loc.hw_register, return_size)
            else:
                self._emit_return_store(source_reg, dest_loc, return_size)

    def _get_return_value_sizes(self, instr: Call) -> List[int]:
        """Return per-return-value byte sizes (parallel to instr.returns).

        Falls back to [1] * len(returns) when the type info isn't available.
        """
        callee_return_type = instr.callee_return_type
        if callee_return_type is None:
            return [1] * len(instr.returns)
        from r65.compiler.hir.types import MultiReturnTypeInfo
        types = (callee_return_type.element_types
                 if isinstance(callee_return_type, MultiReturnTypeInfo)
                 else [callee_return_type])
        return [scalar_width(t) or 2 for t in types]

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

    def _emit_return_register_transfer(self, source_reg: str, dest_reg: str, source_size: int = 2):
        """Transfer return value between hardware registers.

        For u8 (source_size == 1) source from A to X/Y, AND #$00FF in m16
        before TAX/TAY so the B register's stale contents don't leak into
        the high byte of the index register. Without this, an x16 TAX of
        a u8 return value produces X = (B << 8) | A, where B is whatever
        the callee happened to leave it as — often non-zero.
        """
        if source_reg == 'A' and dest_reg in ('X', 'Y') and source_size == 1:
            self.parent._ensure_m16_mode()
            self._emit_immediate(Opcode.AND_IMMEDIATE, 0x00FF,
                                 "Zero-extend u8 return for transfer to x16")
            self._emit_transfer(source_reg, dest_reg)
        else:
            self._emit_transfer(source_reg, dest_reg)

    def _emit_return_store(self, source_reg: str, dest_loc, return_size: int = 1):
        """Store return value from register to memory.

        HIR rejects a 1-byte return value bound to X or Y (`hir/builder.py`:
        "does not fit return register X, which holds 2 byte(s)"), so the width
        hazard is unreachable; routing through the shared helper keeps it so.
        """
        if source_reg in STORE_MNEMONICS:
            self._emit_store_from_reg(source_reg, dest_loc,
                                      return_size >= 2, return_size)

    # ========================================================================
    # Built-in Function Calls
    # ========================================================================

    def _emit_builtin_call(self, instr: Call):
        """
        Emit code for built-in function call.

        Built-in categories:
        - Processor control: NOP([count])
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
        """Emit processor control built-in (NOP)."""
        opcode = Opcode.__members__.get(builtin.instruction)
        if not opcode:
            raise unknown_value("processor control builtin", builtin.instruction, source_loc=self.parent._current_source_loc)

        count = 1  # Default
        if len(instr.args) == 1:
            arg = instr.args[0]
            if isinstance(arg.value, MIRImmediate):
                count = arg.value.value
            else:
                raise InstructionSelectionError("NOP() count must be a constant immediate value", source_loc=self.parent._current_source_loc)

        for _ in range(count):
            self._emit_implied(opcode)

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
            # TYX copies all 16 bits and leaves A (the first argument) alone.
            self._emit_implied(Opcode.TYX)
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

    def _emit_d_push_only(self):
        """
        Push D back onto the stack after a call.

        Since we used PLD before the call, we must push D back so the
        epilogue's PLD will pop the correct value. This is used when there
        are no more far pointer dereferences after this call.

        Updates the stack tracker to undo the displacement from PLD.
        """
        self._emit_instr(Opcode.PHD, comment="Save D back to stack for epilogue")

    def _compute_live_regs_after_call(self, instr: Call, reloads: List[SpillInfo]) -> set:
        """Hardware registers live after return value collection and spill
        reloads (steps 5 through 6.5): every register holding a result
        (_call_live_after_collection) plus every reloaded register, and the
        index registers the call keeps, which may hold a value never saved.
        """
        live = self._call_live_after_collection(instr)
        live.update(spill.hw_reg for spill in reloads)
        live.update(self.callee_preserves(instr) & INDEX_REGS)
        return live

    def _emit_d_equals_s_restore(self, live_regs: set):
        """
        Re-establish D = S after a call for continued far pointer access.

        After a call returns, if there are more far pointer dereferences,
        we need to:
        1. Push D back onto the stack (since PLD popped it before the call)
        2. Set D = S for continued far pointer access

        TSC writes all of C (A and B), so we use a tiered strategy based on
        which registers are live at this point:

        Tier 1 - A is dead (common: void calls, returns stored to stack):
          PHD          ; push D (2 bytes)
          TSC          ; A = SP (clobbers dead A)
          TCD          ; D = SP = frame_base

        Tier 2 - A is live, B is dead, X or Y is dead:
          PHD          ; push D
          TAX/TAY      ; save A in dead register
          TSC          ; A = SP
          TCD          ; D = SP = frame_base
          TXA/TYA      ; restore A

        Tier 3 - B is live, or A, X, Y all are (rare; the TXA/TYA of tier 2
        restores only A in m8):
          PHD          ; push D (2 bytes), SP = frame_base
          (emit_d_equals_s_keeping_c: REP #$20 if m8; PHA, two bytes;
           TSC; INC A; INC A; TCD; PLA; SEP #$20 if it was m8)

        Updates the stack tracker to undo the displacement from PLD.
        """
        if 'B' in live_regs or {'A', 'X', 'Y'} <= live_regs:
            # Tier 3: park C on the stack
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            emit_d_equals_s_keeping_c(self.parent.emitter)
        elif 'A' not in live_regs:
            # Tier 1: A is dead — fast path (3 instructions, ~8 cycles)
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            self._emit_instr(Opcode.TSC, comment="A = SP (A is dead)")
            self._emit_instr(Opcode.TCD, comment="D = SP = frame_base (D = S)")
        elif 'X' not in live_regs:
            # Tier 2a: A live, X dead — use TAX/TXA (5 instructions, ~12 cycles)
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            self._emit_instr(Opcode.TAX, comment="Save A in X")
            self._emit_instr(Opcode.TSC, comment="A = SP")
            self._emit_instr(Opcode.TCD, comment="D = SP = frame_base (D = S)")
            self._emit_instr(Opcode.TXA, comment="Restore A from X")
        else:
            # Tier 2b: A live, Y dead — use TAY/TYA (5 instructions, ~12 cycles)
            self._emit_instr(Opcode.PHD, comment="Save D back to stack")
            self._emit_instr(Opcode.TAY, comment="Save A in Y")
            self._emit_instr(Opcode.TSC, comment="A = SP")
            self._emit_instr(Opcode.TCD, comment="D = SP = frame_base (D = S)")
            self._emit_instr(Opcode.TYA, comment="Restore A from Y")

    def _has_far_ptr_derefs_after_call(self, call_instr: Call) -> bool:
        """
        Check whether any far pointer dereference is reachable from this call.

        "Reachable" is a CFG question, not a listing-order one. A loop like

            loop { let c = s[i]; if c == 0 { break; } sink(c); i++; }

        dereferences `s` in the header block, which sits *before* the call's
        block in the block map but *after* it across the back edge, so the
        search follows CFG successors (back edges included), not listing order.

        Args:
            call_instr: The current Call instruction

        Returns:
            True if a far pointer dereference is reachable after this call
        """
        from r65.compiler.mir.nodes import CompareIndirect, LoadIndirect, StoreIndirect

        func = self.parent.current_function
        if not func:
            return False

        def has_far_deref(instructions) -> bool:
            return any(
                isinstance(i, (LoadIndirect, StoreIndirect, CompareIndirect)) and i.is_far
                for i in instructions
            )

        # Locate the call, then check the tail of its own block.
        home_id = None
        for block_id, block in func.blocks.items():
            for idx, instr in enumerate(block.instructions):
                if instr is call_instr:
                    if has_far_deref(block.instructions[idx + 1:]):
                        return True
                    home_id = block_id
                    break
            if home_id is not None:
                break

        if home_id is None:
            return False

        # Then every block reachable from it, back edges included.
        seen = set()
        queue = list(func.blocks[home_id].successors)
        while queue:
            block_id = queue.pop()
            if block_id in seen:
                continue
            seen.add(block_id)
            block = func.blocks.get(block_id)
            if block is None:
                continue
            if has_far_deref(block.instructions):
                return True
            queue.extend(block.successors)

        return False

    # ========================================================================
    # DBR Management for SET_DBR Strategy
    # ========================================================================

    def _emit_dbr_save_before_call(self):
        """Save ptr-bank DBR and set DBR to the data bank before a call.

        Under SET_DBR strategy, DBR is set to the far pointer's bank.
        Callees expect DBR to be the data bank ($00, or $80 under FastROM), so we
        save DBR (PHB) and load the data bank (Emitter.data_bank_load).
        """
        self._emit_instr(Opcode.PHB, comment="Save ptr-bank DBR before call")
        self.emitter.emit_data_bank_load("Set DBR to the data bank for callee",
                                         self.parent._current_source_loc)
        # The load's pushes and pulls balance — net zero for stack tracker

    def _emit_dbr_restore_after_call(self):
        """Restore ptr-bank DBR after a call returns."""
        self._emit_instr(Opcode.PLB, comment="Restore ptr-bank DBR after call")


def _format_jml_operand(addr) -> str:
    """`$1234` for an address, the label itself for a label."""
    if not isinstance(addr, int):
        return str(addr)
    return f"${addr:02X}" if addr <= 0xFF else f"${addr:04X}"
