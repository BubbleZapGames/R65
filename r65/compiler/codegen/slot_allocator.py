"""
Unified stack slot allocator with reuse optimization.

Allocates memory slots for local variables and temporaries, reusing slots
when variable lifetimes don't overlap (determined by liveness analysis).

This module provides a unified approach that handles both stack parameters
(with fixed offsets from caller) and local variables in a single pass.
Parameters are treated as preassigned slots that participate in liveness
analysis but don't contribute to the local frame size.

Handles multi-byte values (e.g., 3-byte far pointers) by allocating
consecutive slots and tracking the full range for interference checking.
"""

from typing import Dict, Set, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from r65.compiler.mir.nodes import (
    VirtualRegister, MIRFunction, Move, Return, HardwareRegister, Call,
    Store, Load, BinaryOp, UnaryOp, TypeConvert, Compare, BitTest, Rotate,
    ToBool, LoadIndirect, StoreIndirect, StatusFlagRead, InlineAsm,
    TraitDispatch, RestoreRegister, SaveRegister,
    CondBranch, JumpTable, LookupTable,
)
from r65.compiler.mir.liveness import LivenessAnalyzer
from r65.compiler.hir.unified_type_utils import get_unified_type_size
from r65.compiler.codegen.type_utils import get_vreg_size

from typing import TYPE_CHECKING as _TYPE_CHECKING
if _TYPE_CHECKING:
    from r65.compiler.codegen.abi import StackFrameLayout

# Frozenset for ALU-type instructions that route through A register
_ALU_TYPES = frozenset({BinaryOp, UnaryOp, TypeConvert, Compare, BitTest, Rotate, ToBool})


@dataclass
class PreassignedSlot:
    """A stack slot with a fixed offset (e.g., stack parameter)."""
    vreg: VirtualRegister
    base_offset: int  # Offset from S before frame allocation
    size: int


@dataclass
class SlotAllocation:
    """
    Result of unified slot allocation.

    Contains both local allocations (contributing to frame_size) and
    preassigned allocations (params) with their final computed offsets.
    """
    # Local variables: vreg -> slot number (offset within frame, starting at 0)
    register_to_slot: Dict[VirtualRegister, int]

    # Local variable sizes
    register_to_size: Dict[VirtualRegister, int]

    # Total frame size (only locals, not params)
    total_slots: int

    # Statistics
    variables_count: int
    slots_saved: int

    # Hardware-coalesceable vregs (don't need stack slots)
    hw_coalesceable: Dict[VirtualRegister, str] = None

    # Return-sinkable vregs: Load from MemoryLocation used only in Return
    # These don't need stack slots; the load is emitted at the return site
    return_sinkable: Dict[VirtualRegister, Any] = None

    # Stack parameters: vreg -> final offset (after frame adjustment)
    param_offsets: Dict[VirtualRegister, int] = None

    # Param sizes
    param_sizes: Dict[VirtualRegister, int] = None

    # Maximum live frame bytes at any call site (for frame-aware stack depth)
    max_live_frame_bytes_at_calls: int = 0

    def __post_init__(self):
        if self.hw_coalesceable is None:
            self.hw_coalesceable = {}
        if self.return_sinkable is None:
            self.return_sinkable = {}
        if self.param_offsets is None:
            self.param_offsets = {}
        if self.param_sizes is None:
            self.param_sizes = {}

    def get_offset(self, vreg: VirtualRegister) -> Optional[int]:
        """Get slot offset for any vreg (local only - params use param_offsets)."""
        return self.register_to_slot.get(vreg)

    def get_param_offset(self, vreg: VirtualRegister) -> Optional[int]:
        """Get final stack offset for a parameter vreg."""
        return self.param_offsets.get(vreg)

    def get_size(self, vreg: VirtualRegister) -> Optional[int]:
        """Get size for any vreg."""
        if vreg in self.register_to_size:
            return self.register_to_size[vreg]
        if vreg in self.param_sizes:
            return self.param_sizes[vreg]
        return None

    def is_param(self, vreg: VirtualRegister) -> bool:
        """Check if vreg is a stack parameter."""
        return vreg in self.param_offsets


class StackSlotAllocator:
    """
    Unified stack slot allocator handling params and locals together.

    Stack layout after frame allocation:

        High addresses
        [arg2]          <- param_offsets[1] = base + prologue + frame_size
        [arg1]          <- param_offsets[0] = base + prologue + frame_size
        [return addr]
        [saved regs]    <- prologue_stack_bytes
        [local N]       <- local_slots[N] = N
        ...
        [local 1]       <- local_slots[0] = 1
        S ->
        Low addresses

    Parameters have fixed positions (above return addr) but their S-relative
    offsets change when we allocate a frame. This allocator computes final
    offsets for everything in one pass.
    """

    def __init__(
        self,
        mir_func: MIRFunction,
        preassigned: Optional[List[PreassignedSlot]] = None,
        prologue_stack_bytes: int = 0,
        instr_liveness: Optional[Any] = None,
        pre_allocated_vregs: Optional[Set[VirtualRegister]] = None,
        outgoing_arg_bytes: int = 0,
        layout: Optional['StackFrameLayout'] = None
    ):
        """
        Initialize unified slot allocator.

        Args:
            mir_func: MIR function to allocate slots for
            preassigned: Stack parameters with their base offsets
            prologue_stack_bytes: Bytes pushed by prologue (return addr + saved regs)
            instr_liveness: Optional InstructionLivenessAnalyzer for call-liveness analysis
            pre_allocated_vregs: Vregs already allocated externally (e.g. scratch params),
                excluded from local slot allocation
            outgoing_arg_bytes: Caller-owned outgoing argument area size
            layout: Optional StackFrameLayout for offset computation
        """
        self.func = mir_func
        self.preassigned = preassigned or []
        self.prologue_stack_bytes = prologue_stack_bytes
        self.liveness_analyzer = LivenessAnalyzer(mir_func)
        self.instr_liveness = instr_liveness
        self.pre_allocated_vregs = pre_allocated_vregs or set()
        self.outgoing_arg_bytes = outgoing_arg_bytes
        self.layout = layout

        # Build vreg lookup for preassigned
        self._preassigned_vregs: Dict[int, PreassignedSlot] = {
            slot.vreg.id: slot for slot in self.preassigned
        }

    def allocate(self) -> SlotAllocation:
        """
        Allocate slots for all virtual registers.

        Returns:
            SlotAllocation with locals and params mapped to final offsets
        """
        # Run liveness analysis (includes all vregs)
        self.liveness_analyzer.analyze()

        # Coalesce vreg-to-vreg Moves when lifetimes don't interfere
        self._coalesce_vreg_moves()

        # Identify vregs that can stay in hardware registers
        hw_coalesceable = self._find_hw_coalesceable_vregs()

        # Identify return-sinkable vregs (Load from MemoryLocation, only used in Return)
        return_sinkable = self._find_return_sinkable_vregs()

        # Collect local vregs (excluding hw-coalesceable, return-sinkable, preassigned params,
        # and externally pre-allocated vregs like scratch params)
        exclude_vregs = set(hw_coalesceable.keys()) | set(return_sinkable.keys()) | self.pre_allocated_vregs
        for slot in self.preassigned:
            exclude_vregs.add(slot.vreg)

        local_vregs = self._collect_virtual_registers(exclude=exclude_vregs)

        # Calculate sizes for locals
        local_sizes: Dict[VirtualRegister, int] = {}
        for vreg in local_vregs:
            local_sizes[vreg] = get_vreg_size(vreg)

        # Determine which locals are live at any call site (for frame partitioning)
        call_spanning_vregs = self._find_call_spanning_vregs(local_vregs) if self.instr_liveness else set()

        # Allocate locals with liveness-based reuse, partitioned by call-liveness
        local_slots, frame_size, slots_saved = self._allocate_locals(
            local_vregs, local_sizes, call_spanning_vregs
        )

        # Compute max live frame bytes at calls
        max_live = self._compute_max_live_frame_bytes_at_calls(
            local_slots, local_sizes
        )

        # Compute final param offsets (adjusted for frame allocation)
        param_offsets: Dict[VirtualRegister, int] = {}
        param_sizes: Dict[VirtualRegister, int] = {}

        for slot in self.preassigned:
            if self.layout is not None:
                # Update layout with computed frame size, then use it
                self.layout.local_frame_size = frame_size
                final_offset = self.layout.param_offset(slot.base_offset)
            else:
                # Fallback: inline formula
                total_frame_size = frame_size + self.outgoing_arg_bytes
                final_offset = slot.base_offset + self.prologue_stack_bytes + total_frame_size
            param_offsets[slot.vreg] = final_offset
            param_sizes[slot.vreg] = slot.size

        return SlotAllocation(
            register_to_slot=local_slots,
            register_to_size=local_sizes,
            total_slots=frame_size,
            variables_count=len(local_vregs),
            slots_saved=slots_saved,
            hw_coalesceable=hw_coalesceable,
            return_sinkable=return_sinkable,
            param_offsets=param_offsets,
            param_sizes=param_sizes,
            max_live_frame_bytes_at_calls=max_live
        )

    def _allocate_locals(
        self,
        local_vregs: List[VirtualRegister],
        local_sizes: Dict[VirtualRegister, int],
        call_spanning_vregs: Optional[Set[VirtualRegister]] = None
    ) -> Tuple[Dict[VirtualRegister, int], int, int]:
        """
        Allocate local variables with liveness-based reuse.

        When call_spanning_vregs is provided, locals are partitioned:
        - Call-spanning locals (live at any call) -> allocated at low slot numbers
          (bottom of frame, closest to SP). These survive partial frame deallocation.
        - Pre-call-dead locals (dead before all calls) -> allocated at high slot numbers
          (top of frame). These can be reclaimed before calls.

        Args:
            local_vregs: Local variables to allocate
            local_sizes: Size of each local
            call_spanning_vregs: Set of vregs live at any call site (optional)

        Returns:
            (slot_mapping, frame_size, slots_saved)
        """
        if not local_vregs:
            return {}, 0, 0

        allocation: Dict[VirtualRegister, int] = {}
        # Track allocated ranges: (start, end, vreg)
        allocated_ranges: List[Tuple[int, int, VirtualRegister]] = []
        next_slot = 0

        # Partition and sort: call-spanning first (low slots), then pre-call-dead (high slots)
        # Within each partition, sort by size descending for better packing
        if call_spanning_vregs:
            spanning = [v for v in local_vregs if v in call_spanning_vregs]
            non_spanning = [v for v in local_vregs if v not in call_spanning_vregs]
            sorted_vregs = (
                sorted(spanning, key=lambda v: -local_sizes[v]) +
                sorted(non_spanning, key=lambda v: -local_sizes[v])
            )
        else:
            # Sort by size descending for better packing
            sorted_vregs = sorted(local_vregs, key=lambda v: -local_sizes[v])

        for vreg in sorted_vregs:
            size = local_sizes[vreg]

            # Enforce minimum 2-byte slot size. The codegen often operates in
            # m16 mode (16-bit accumulator) even for 1-byte variables (bool, u8),
            # causing 16-bit loads/stores that would overflow a 1-byte slot into
            # adjacent memory. Padding to 2 bytes prevents this corruption.
            alloc_size = max(size, 2)

            # Allocate sequentially without reuse to avoid overlap bugs.
            # TODO: re-enable liveness-based reuse after fixing interference analysis
            assigned_slot = next_slot
            next_slot = assigned_slot + alloc_size

            allocation[vreg] = assigned_slot
            allocated_ranges.append((assigned_slot, assigned_slot + size, vreg))

        # Frame size is the total local slots needed
        # Ensure frame_size covers all allocated ranges (safety check)
        max_end = 0
        for (start, end, _) in allocated_ranges:
            if end > max_end:
                max_end = end
        frame_size = max(next_slot, max_end)

        # DEBUG: Print slot allocation for debugging
        import os
        if os.environ.get('R65_DEBUG_SLOTS') and frame_size >= 10:
            import sys
            print(f"SLOT_ALLOC frame_size={frame_size}", file=sys.stderr)
            for vreg, slot in sorted(allocation.items(), key=lambda x: x[1]):
                size = local_sizes.get(vreg, '?')
                print(f"  {vreg} (size={size}) -> slot {slot} (bytes {slot}-{slot+size-1})", file=sys.stderr)

        # Calculate slots saved
        total_without_reuse = sum(local_sizes.values())
        slots_saved = total_without_reuse - frame_size

        return allocation, frame_size, slots_saved

    def _find_call_spanning_vregs(
        self,
        local_vregs: List[VirtualRegister]
    ) -> Set[VirtualRegister]:
        """
        Find local vregs that are live at any Call instruction.

        Args:
            local_vregs: Local variables to check

        Returns:
            Set of vregs that are live at at least one call site
        """
        if not self.instr_liveness:
            return set()

        local_set = set(local_vregs)
        spanning: Set[VirtualRegister] = set()

        for block_id, block in self.func.blocks.items():
            for instr_idx, instr in enumerate(block.instructions):
                if isinstance(instr, (Call, TraitDispatch)):
                    for vreg in local_set:
                        if vreg in spanning:
                            continue
                        # Live "at" the call = live just before it executes
                        # is_live_after at instr_idx-1 means live entering instr_idx
                        if instr_idx > 0:
                            if self.instr_liveness.is_live_after(vreg, block_id, instr_idx - 1):
                                spanning.add(vreg)
                        else:
                            # First instruction in block: check block live_in
                            info = self.instr_liveness.liveness.get(block_id)
                            if info and vreg in info.live_in:
                                spanning.add(vreg)

        return spanning

    def _compute_max_live_frame_bytes_at_calls(
        self,
        local_slots: Dict[VirtualRegister, int],
        local_sizes: Dict[VirtualRegister, int]
    ) -> int:
        """
        Compute the maximum total bytes of locals live at any Call instruction.

        This metric determines how much of the frame must be kept during calls.
        The reclaimable portion is frame_size - max_live_frame_bytes_at_calls.

        Args:
            local_slots: Mapping of vreg to slot offset
            local_sizes: Mapping of vreg to size in bytes

        Returns:
            Maximum live frame bytes at any call site (0 if no calls)
        """
        if not self.instr_liveness or not local_slots:
            return 0

        max_live = 0

        for block_id, block in self.func.blocks.items():
            for instr_idx, instr in enumerate(block.instructions):
                if not isinstance(instr, (Call, TraitDispatch)):
                    continue

                # Sum sizes of locals live at this call
                live_bytes = 0
                for vreg, slot in local_slots.items():
                    size = local_sizes.get(vreg, 1)
                    # Check if live just before the call
                    if instr_idx > 0:
                        if self.instr_liveness.is_live_after(vreg, block_id, instr_idx - 1):
                            live_bytes += size
                    else:
                        info = self.instr_liveness.liveness.get(block_id)
                        if info and vreg in info.live_in:
                            live_bytes += size

                if live_bytes > max_live:
                    max_live = live_bytes

        return max_live

    # ------------------------------------------------------------------
    # Vreg-to-vreg move coalescing
    # ------------------------------------------------------------------

    def _coalesce_vreg_moves(self) -> None:
        """
        Coalesce Move(dest=VReg, src=VReg) instructions where the two vregs
        don't interfere.  Replace all uses of *dest* with *src*, remove the
        Move, and propagate register hints.

        This recovers the zero-cost aliasing that was previously done in
        builder.py (reusing the same vreg for ``let x = y``).  The builder
        now always allocates a fresh vreg so that independent mutations
        don't corrupt each other; this pass merges them back when safe.

        Must run BEFORE hw-coalescence so the hw pass sees simplified MIR.
        """
        changed = True
        while changed:
            changed = False
            # Process blocks in order; within each block process moves in
            # instruction order so that chains (a→b, b→c) collapse correctly.
            for block_id in sorted(self.func.blocks):
                block = self.func.blocks[block_id]
                i = 0
                while i < len(block.instructions):
                    instr = block.instructions[i]
                    if (isinstance(instr, Move)
                            and isinstance(instr.dest, VirtualRegister)
                            and isinstance(instr.source, VirtualRegister)
                            and instr.dest != instr.source):
                        dest = instr.dest
                        src = instr.source
                        if not self.liveness_analyzer.interferes(dest, src):
                            # Propagate register hint from dest to src
                            if dest.register_hint and not src.register_hint:
                                src.register_hint = dest.register_hint
                            # Replace dest with src everywhere in the MIR
                            self._replace_vreg_everywhere(dest, src)
                            # Remove the (now-redundant) Move
                            block.instructions.pop(i)
                            changed = True
                            continue  # re-examine same index
                    i += 1
            if changed:
                # Liveness data is stale after replacement — recompute
                self.liveness_analyzer = LivenessAnalyzer(self.func)
                self.liveness_analyzer.analyze()

    @staticmethod
    def _replace_vreg_in_instr(instr, old: VirtualRegister, new: VirtualRegister):
        """Replace every occurrence of *old* vreg with *new* inside *instr*."""

        def _sub(val):
            return new if isinstance(val, VirtualRegister) and val == old else val

        # --- Memory operations ---
        if isinstance(instr, Load):
            instr.dest = _sub(instr.dest)
        elif isinstance(instr, Store):
            instr.source = _sub(instr.source)
        elif isinstance(instr, LoadIndirect):
            instr.dest = _sub(instr.dest)
            instr.pointer = _sub(instr.pointer)
        elif isinstance(instr, StoreIndirect):
            instr.source = _sub(instr.source)
            instr.pointer = _sub(instr.pointer)

        # --- Moves & conversions ---
        elif isinstance(instr, Move):
            instr.dest = _sub(instr.dest)
            instr.source = _sub(instr.source)
        elif isinstance(instr, TypeConvert):
            instr.dest = _sub(instr.dest)
            instr.source = _sub(instr.source)
        elif isinstance(instr, ToBool):
            instr.dest = _sub(instr.dest)
            instr.source = _sub(instr.source)

        # --- ALU ---
        elif isinstance(instr, BinaryOp):
            instr.dest = _sub(instr.dest)
            instr.left = _sub(instr.left)
            instr.right = _sub(instr.right)
        elif isinstance(instr, UnaryOp):
            instr.dest = _sub(instr.dest)
            instr.operand = _sub(instr.operand)
        elif isinstance(instr, Rotate):
            instr.dest = _sub(instr.dest)
            instr.source = _sub(instr.source)

        # --- Compare / BitTest ---
        elif isinstance(instr, Compare):
            instr.left = _sub(instr.left)
            instr.right = _sub(instr.right)
        elif isinstance(instr, BitTest):
            instr.value = _sub(instr.value)

        # --- Control flow ---
        elif isinstance(instr, CondBranch):
            instr.condition = _sub(instr.condition)
        elif isinstance(instr, JumpTable):
            instr.scrutinee = _sub(instr.scrutinee)
        elif isinstance(instr, LookupTable):
            instr.dest = _sub(instr.dest)
            instr.scrutinee = _sub(instr.scrutinee)

        # --- Return ---
        elif isinstance(instr, Return):
            instr.values = [_sub(v) for v in instr.values]

        # --- Calls ---
        elif isinstance(instr, (Call, TraitDispatch)):
            for arg in instr.args:
                arg.value = _sub(arg.value)
            instr.returns = [_sub(r) for r in instr.returns]
            if isinstance(instr, Call) and isinstance(instr.function, VirtualRegister):
                instr.function = _sub(instr.function)
            if isinstance(instr, TraitDispatch) and isinstance(instr.self_ptr, VirtualRegister):
                instr.self_ptr = _sub(instr.self_ptr)

        # --- Save / Restore ---
        elif isinstance(instr, SaveRegister):
            instr.save_location = _sub(instr.save_location)
        elif isinstance(instr, RestoreRegister):
            instr.save_location = _sub(instr.save_location)

        # --- StatusFlagRead ---
        elif isinstance(instr, StatusFlagRead):
            instr.dest = _sub(instr.dest)

        # Other instructions (Jump, InlineAsm, SetMode, etc.) have no vregs.

    def _replace_vreg_everywhere(self, old: VirtualRegister, new: VirtualRegister):
        """Replace *old* with *new* in every instruction of every block."""
        for block in self.func.blocks.values():
            for instr in block.instructions:
                self._replace_vreg_in_instr(instr, old, new)

    def _find_hw_coalesceable_vregs(self) -> Dict[VirtualRegister, str]:
        """
        Find vregs that can stay in hardware registers without spilling.

        A vreg is hw-coalesceable if:
        1. Its only definition is a Move from a hardware register
        2. All uses are in the same block as the def
        3. The hardware register is not clobbered between def and last use

        For the return-only case, clobber checking is limited to Calls.
        For non-return uses, full clobber analysis checks all instruction types
        that route through the hardware register during codegen.

        Uses a two-pass approach for A/B interdependency:
        - Pass 1: Find coalesceable vregs where no instruction clobbers the register
        - Pass 2: Re-check remaining candidates, treating Pass 1 coalesceable
          Move instructions as no-ops (lets A coalesce when only "clobber" was
          a B param save that is itself coalesceable)

        Returns:
            Dict mapping vreg to the hardware register name it should stay in
        """
        vreg_defs: Dict[int, List] = {}
        vreg_uses: Dict[int, List] = {}

        # Track instruction positions for ordering checks
        # Map: instruction -> (block_id, instr_idx)
        instr_positions: Dict[Any, Tuple[int, int]] = {}

        for block_id, block in self.func.blocks.items():
            for instr_idx, instr in enumerate(block.instructions):
                instr_positions[id(instr)] = (block_id, instr_idx)
                instr_type = type(instr)

                if instr_type is Move:
                    if type(instr.dest) is VirtualRegister:
                        vreg_id = instr.dest.id
                        if type(instr.source) is HardwareRegister:
                            vreg_defs.setdefault(vreg_id, []).append((instr, instr.source.name))
                        else:
                            vreg_defs.setdefault(vreg_id, []).append((instr, None))

                    if type(instr.source) is VirtualRegister:
                        vreg_uses.setdefault(instr.source.id, []).append(instr)

                elif instr_type is BinaryOp or instr_type is UnaryOp:
                    # BinaryOp/UnaryOp implicitly produce their result in A.
                    # Track dest vreg as defined-in-A for coalescence,
                    # but only if no prior def exists. This avoids double-counting
                    # vregs first loaded from A (Move vreg <- A) then modified
                    # in-place (BinaryOp vreg = vreg + 1) — the Move def already
                    # enables coalescence for those.
                    if type(instr.dest) is VirtualRegister:
                        vreg_id = instr.dest.id
                        if vreg_id not in vreg_defs:
                            dest_size = get_vreg_size(instr.dest)
                            # Skip far pointer results (3 bytes) — codegen stores
                            # them byte-by-byte, A holds only a partial value
                            if dest_size <= 2:
                                vreg_defs[vreg_id] = [(instr, 'A')]

                    # Track operand uses
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if type(var) is VirtualRegister:
                            vreg_uses.setdefault(var.id, []).append(instr)

                elif instr_type is LoadIndirect:
                    # LoadIndirect deposits its result in A via LDA [dp],Y.
                    # Track dest vreg as defined-in-A for coalescence (u8 only).
                    if type(instr.dest) is VirtualRegister:
                        vreg_id = instr.dest.id
                        if vreg_id not in vreg_defs:
                            dest_size = get_vreg_size(instr.dest)
                            if dest_size == 1:
                                vreg_defs[vreg_id] = [(instr, 'A')]

                    # Track pointer/index uses
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if type(var) is VirtualRegister:
                            vreg_uses.setdefault(var.id, []).append(instr)

                elif instr_type is Load:
                    # Load deposits its result in A via LDA.
                    # Track dest vreg as defined-in-A for coalescence.
                    if type(instr.dest) is VirtualRegister:
                        vreg_id = instr.dest.id
                        if vreg_id not in vreg_defs:
                            dest_size = get_vreg_size(instr.dest)
                            if dest_size <= 2:
                                vreg_defs[vreg_id] = [(instr, 'A')]

                    # Track operand uses
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if type(var) is VirtualRegister:
                            vreg_uses.setdefault(var.id, []).append(instr)

                elif instr_type is Call or instr_type is TraitDispatch:
                    # Call with exactly 1 return vreg: the return value is in A.
                    # Track as defined-in-A for coalescence (e.g., result = func()).
                    # A Call clobbers all registers, so no other vreg can be live
                    # in A after the Call — no all_uses_consume_a filter needed.
                    if len(instr.returns) == 1:
                        ret_vreg = instr.returns[0]
                        if type(ret_vreg) is VirtualRegister:
                            vreg_id = ret_vreg.id
                            if vreg_id not in vreg_defs:
                                dest_size = get_vreg_size(ret_vreg)
                                if dest_size <= 2:
                                    vreg_defs[vreg_id] = [(instr, 'A')]
                    # Track uses of arguments (fall through to else branch won't
                    # happen since we matched Call, so handle uses here)
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if type(var) is VirtualRegister:
                            vreg_uses.setdefault(var.id, []).append(instr)

                elif instr_type is Return:
                    if instr.values:
                        for val in instr.values:
                            if type(val) is VirtualRegister:
                                vreg_uses.setdefault(val.id, []).append(instr)

                else:
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if type(var) is VirtualRegister:
                            vreg_uses.setdefault(var.id, []).append(instr)

        # Build candidate list: (vreg_id, def_instr, hw_reg, uses)
        # For ALU-def vregs (BinaryOp/UnaryOp/LoadIndirect), restrict to cases
        # where all uses are Return, Move, Compare, Store (as source), or as
        # the LEFT operand of a BinaryOp (chained arithmetic).
        _alu_def_types = frozenset({BinaryOp, UnaryOp, LoadIndirect, Load})
        candidates = []
        for vreg_id, defs in vreg_defs.items():
            if len(defs) != 1:
                continue
            def_instr, hw_reg = defs[0]
            if hw_reg is None:
                continue
            uses = vreg_uses.get(vreg_id, [])

            # ALU-def vregs: only safe if the value is consumed by Move,
            # Return, Compare, Store (as value source), as the LEFT
            # operand of a BinaryOp (chained arithmetic), or as a Call
            # argument (pushed via PHA or stored via STA from A).
            # - Move variants just read A (STA/TAX/TAY preserve A).
            # - Store from our vreg emits STA directly from A (preserves A).
            # - Compare emits CMP which reads A without modifying it.
            # - BinaryOp-left reads A first (no clobber), then overwrites A
            #   with the result — safe because the clobber analysis handles it
            #   (the BinaryOp at max_use_idx is not scanned for clobbers).
            # - Call arg: codegen reads A for PHA/STA; emit_call_args ensures
            #   A-resident args are processed before other args clobber A.
            if type(def_instr) in _alu_def_types:
                all_uses_safe = True
                for use in uses:
                    use_type = type(use)
                    if use_type is Return or use_type is Move or use_type is Compare:
                        continue
                    if (use_type is Store and
                        type(use.source) is VirtualRegister and
                        use.source.id == vreg_id):
                        continue
                    if (use_type is StoreIndirect and
                        type(use.source) is VirtualRegister and
                        use.source.id == vreg_id):
                        continue
                    if (use_type is BinaryOp and
                        type(use.left) is VirtualRegister and
                        use.left.id == vreg_id):
                        continue
                    if (use_type is Call and
                        any(type(a.value) is VirtualRegister and
                            a.value.id == vreg_id
                            for a in use.args)):
                        continue
                    all_uses_safe = False
                    break
                if not all_uses_safe:
                    continue

            candidates.append((vreg_id, def_instr, hw_reg, uses))

        # Two-pass coalescence
        coalesceable: Dict[VirtualRegister, str] = {}
        noop_instrs: Set[int] = set()  # instruction ids treated as no-ops

        for pass_num in range(2):
            remaining = []
            for vreg_id, def_instr, hw_reg, uses in candidates:
                if not uses:
                    # No uses - can be coalesceable (dead value)
                    self._mark_coalesceable(vreg_id, hw_reg, coalesceable)
                    continue

                def_pos = instr_positions.get(id(def_instr))
                if def_pos is None:
                    remaining.append((vreg_id, def_instr, hw_reg, uses))
                    continue

                if self._is_hw_unclobbered_in_range(
                    hw_reg, vreg_id, def_instr, uses, instr_positions, noop_instrs
                ):
                    self._mark_coalesceable(vreg_id, hw_reg, coalesceable)
                    # Only add Move instructions to noop_instrs. The two-pass
                    # mechanism is designed for A/B interdependency where a
                    # coalesceable Move (e.g., XBA for B) becomes a no-op.
                    # Call/TraitDispatch instructions ALWAYS clobber registers
                    # during execution even if their return value is coalesceable,
                    # so they must never be treated as no-ops.
                    if type(def_instr) is Move:
                        noop_instrs.add(id(def_instr))
                else:
                    remaining.append((vreg_id, def_instr, hw_reg, uses))

            candidates = remaining

        return coalesceable

    def _find_return_sinkable_vregs(self) -> Dict[VirtualRegister, 'MemoryLocation']:
        """
        Find vregs whose loads can be sunk to the return site.

        A vreg is return-sinkable if:
        1. Its only definition is a Load from a MemoryLocation
        2. All uses are in Return instructions
        3. All defs and uses are in the same block

        These vregs don't need stack slots; the load is emitted directly
        into the target return register at the return site.

        Returns:
            Dict mapping vreg to the MemoryLocation source of the Load
        """
        from r65.compiler.mir.nodes import MemoryLocation

        # Build def/use maps for Load instructions with vreg destinations
        vreg_load_defs: Dict[int, List] = {}  # vreg_id -> [(Load instr, block_id)]
        vreg_uses: Dict[int, List] = {}  # vreg_id -> [(instr, block_id)]

        for block_id, block in self.func.blocks.items():
            for instr in block.instructions:
                if type(instr) is Load:
                    if type(instr.dest) is VirtualRegister:
                        vreg_load_defs.setdefault(instr.dest.id, []).append((instr, block_id))

                # Track uses in Return instructions
                if type(instr) is Return:
                    if instr.values:
                        for val in instr.values:
                            if type(val) is VirtualRegister:
                                vreg_uses.setdefault(val.id, []).append((instr, block_id))
                else:
                    # Track uses in non-Return instructions
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if type(var) is VirtualRegister:
                            vreg_uses.setdefault(var.id, []).append((instr, block_id))

        sinkable: Dict[VirtualRegister, MemoryLocation] = {}

        for vreg_id, load_defs in vreg_load_defs.items():
            # Must have exactly one Load definition
            if len(load_defs) != 1:
                continue

            load_instr, def_block_id = load_defs[0]
            source = load_instr.source

            # Source must be a MemoryLocation
            if not isinstance(source, MemoryLocation):
                continue

            uses = vreg_uses.get(vreg_id, [])
            if not uses:
                continue  # Dead value - not useful to sink

            # All uses must be Return instructions in the same block
            all_returns_same_block = True
            for use_instr, use_block_id in uses:
                if not isinstance(use_instr, Return):
                    all_returns_same_block = False
                    break
                if use_block_id != def_block_id:
                    all_returns_same_block = False
                    break

            if all_returns_same_block:
                sinkable[load_instr.dest] = source

        return sinkable

    def _mark_coalesceable(
        self, vreg_id: int, hw_reg: str, coalesceable: Dict[VirtualRegister, str]
    ):
        """Mark a vreg as hw-coalesceable by finding its def instruction."""
        _def_types = frozenset({Move, BinaryOp, UnaryOp, LoadIndirect, Load})
        for block in self.func.blocks.values():
            for instr in block.instructions:
                instr_type = type(instr)
                if instr_type is Call or instr_type is TraitDispatch:
                    if instr.returns and type(instr.returns[0]) is VirtualRegister:
                        if instr.returns[0].id == vreg_id:
                            coalesceable[instr.returns[0]] = hw_reg
                            return
                elif instr_type in _def_types and type(instr.dest) is VirtualRegister:
                    if instr.dest.id == vreg_id:
                        coalesceable[instr.dest] = hw_reg
                        return

    def _get_vreg_from_def(self, def_instr: Any, vreg_id: int) -> Optional[VirtualRegister]:
        """Extract VirtualRegister object from its defining instruction."""
        def_type = type(def_instr)
        if def_type is Call or def_type is TraitDispatch:
            if def_instr.returns:
                for r in def_instr.returns:
                    if type(r) is VirtualRegister and r.id == vreg_id:
                        return r
        else:
            dest = getattr(def_instr, 'dest', None)
            if type(dest) is VirtualRegister and dest.id == vreg_id:
                return dest
        return None

    def _is_hw_unclobbered_in_range(
        self,
        hw_reg: str,
        vreg_id: int,
        def_instr: Any,
        uses: List,
        instr_positions: Dict[Any, Tuple[int, int]],
        noop_instrs: Set[int],
    ) -> bool:
        """
        Check if a hardware register is unclobbered between def and all uses.

        Supports both same-block and cross-block uses. For cross-block uses,
        uses liveness analysis to identify which blocks the vreg is live in
        and checks only those blocks for clobbers.

        Args:
            hw_reg: Hardware register name ('A', 'B', 'X', 'Y')
            vreg_id: The vreg ID being checked
            def_instr: The instruction that defines the vreg
            uses: List of instructions that use the vreg
            instr_positions: Map from instruction id to (block_id, instr_idx)
            noop_instrs: Set of instruction ids to skip (from previous pass)

        Returns:
            True if the register is not clobbered in the range
        """
        def_pos = instr_positions.get(id(def_instr))
        if def_pos is None:
            return False

        def_block_id, def_idx = def_pos

        # Classify uses into same-block and cross-block
        same_block_max_idx = def_idx
        cross_block_uses = []  # (block_id, instr_idx)
        for use_instr in uses:
            use_pos = instr_positions.get(id(use_instr))
            if use_pos is None:
                return False
            use_block_id, use_idx = use_pos
            if use_block_id == def_block_id:
                same_block_max_idx = max(same_block_max_idx, use_idx)
            else:
                cross_block_uses.append((use_block_id, use_idx))

        # Determine scan range in def block
        block = self.func.blocks[def_block_id]
        if cross_block_uses:
            # Must survive to block exit — scan all instructions after def
            scan_end = len(block.instructions)
        else:
            scan_end = same_block_max_idx

        # Check def block
        for i in range(def_idx + 1, scan_end):
            if self._instruction_clobbers_register(block.instructions[i], hw_reg,
                                                    vreg_id, noop_instrs):
                return False

        if not cross_block_uses:
            return True

        # Cross-block: get vreg object and its live blocks
        vreg_obj = self._get_vreg_from_def(def_instr, vreg_id)
        if not vreg_obj:
            return False
        live_ranges = self.liveness_analyzer.get_live_ranges()
        live_blocks = live_ranges.get(vreg_obj, set())

        # Build set of use block ids with max use index per block
        use_block_max: Dict[int, int] = {}
        for ub_id, u_idx in cross_block_uses:
            use_block_max[ub_id] = max(use_block_max.get(ub_id, 0), u_idx)

        # Check each live block (excluding def block, already checked)
        for block_id in live_blocks:
            if block_id == def_block_id:
                continue
            blk = self.func.blocks.get(block_id)
            if not blk:
                continue

            if block_id in use_block_max:
                # Use block: check from start to max use index, OR to end of
                # block if the vreg is live-out (used in a successor block).
                # Without this, instructions after the last use (e.g., a Call)
                # would be missed even though the vreg must survive to the
                # block exit.
                vreg_live_out = any(
                    succ_id in live_blocks
                    for succ_id in blk.successors
                )
                scan_end = len(blk.instructions) if vreg_live_out else use_block_max[block_id]
                for i in range(0, scan_end):
                    if self._instruction_clobbers_register(blk.instructions[i], hw_reg,
                                                            vreg_id, noop_instrs):
                        return False
            else:
                # Intermediate block (vreg is live but not used): check all instructions
                for instr in blk.instructions:
                    if self._instruction_clobbers_register(instr, hw_reg,
                                                            vreg_id, noop_instrs):
                        return False
        return True

    def _instruction_clobbers_register(
        self,
        instr: Any,
        hw_reg: str,
        vreg_id: int,
        noop_instrs: Set[int],
    ) -> bool:
        """
        Check if an instruction clobbers a specific hardware register.

        During codegen, most operations route through the A register.
        Store instructions that read from a vreg do NOT clobber A if
        the value being stored comes from the vreg itself (which IS A
        when coalesceable). We check the vreg_id to handle this.

        Args:
            instr: MIR instruction to check
            hw_reg: Hardware register to check ('A', 'B', 'X', 'Y')
            vreg_id: The vreg being checked for coalescence
            noop_instrs: Instructions to skip (coalesceable from previous pass)

        Returns:
            True if the instruction clobbers hw_reg
        """
        # Skip instructions that are themselves coalesceable (no-ops)
        if id(instr) in noop_instrs:
            return False

        instr_type = type(instr)

        # Calls clobber all registers not in preserves
        if instr_type is Call or instr_type is TraitDispatch:
            preserved = set()
            if getattr(instr, 'preserves_attr', None):
                preserved = set(instr.preserves_attr.registers)
            return hw_reg not in preserved

        # InlineAsm: conservatively assume it clobbers everything
        if instr_type is InlineAsm:
            return True

        if hw_reg == 'A':
            return self._clobbers_a(instr, vreg_id)
        elif hw_reg == 'B':
            return self._clobbers_b(instr, vreg_id)
        elif hw_reg in ('X', 'Y'):
            return self._clobbers_xy(instr, hw_reg, vreg_id)

        return False

    def _clobbers_a(self, instr: Any, vreg_id: int) -> bool:
        """
        Check if an instruction clobbers the A register.

        During codegen, most operations route values through A:
        - Move to vreg: LDA source; STA dest (clobbers A)
        - BinaryOp: LDA left; ADC right; STA dest (clobbers A)
        - Store from vreg: LDA vreg_slot; STA dest (clobbers A)
        - Load to vreg: LDA source; STA vreg_slot (clobbers A)
        - Move from B: XBA (clobbers A)

        The key exception: Store from OUR vreg doesn't clobber A,
        because our vreg IS A (it's coalesceable). The codegen will
        emit STA directly without needing to load first.
        """
        instr_type = type(instr)

        if instr_type is Store:
            # Store uses A to transfer values. If the source is our vreg,
            # codegen will use STA directly (A already has the value).
            # If source is a different vreg or hw register, it clobbers A.
            if type(instr.source) is VirtualRegister and instr.source.id == vreg_id:
                return False  # STA from our vreg = STA from A (no clobber)
            if type(instr.source) is HardwareRegister and instr.source.name == 'A':
                return False  # STA from A (already in A)
            return True  # Needs LDA from somewhere else

        if instr_type is Move:
            # Move to vreg (other than ours): LDA + STA clobbers A
            if type(instr.dest) is VirtualRegister and instr.dest.id != vreg_id:
                return True
            # Move to hw register from vreg/immediate: codegen loads through A
            if type(instr.dest) is HardwareRegister:
                if instr.dest.name in ('X', 'Y'):
                    # LDX/LDY don't clobber A (unless source is stack-relative for X/Y)
                    # But Move to X/Y from vreg goes LDA vreg; TAX (clobbers A)
                    # Unless source is our coalesceable vreg — then just TAX/TAY
                    if type(instr.source) is VirtualRegister:
                        if instr.source.id == vreg_id:
                            return False  # TAX/TAY from our vreg — A preserved
                        # Move to X/Y from a vreg with matching register_hint
                        # is a no-op (value already in that register), no LDA needed
                        if instr.source.register_hint == instr.dest.name:
                            return False
                        return True
                    return False  # LDX #imm or LDX addr don't clobber A
                if instr.dest.name == 'B':
                    return True  # XBA clobbers A
                if instr.dest.name == 'A':
                    return True  # Overwrites A
            # Move from B: XBA clobbers A
            if type(instr.source) is HardwareRegister and instr.source.name == 'B':
                return True
            return False

        if instr_type in _ALU_TYPES:
            # ALU ops route through A, clobbering it — UNLESS the dest is our vreg,
            # in which case the op re-defines our vreg and leaves its new value in A
            # (analogous to Store exception: our vreg's value stays in A).
            if ((instr_type is BinaryOp or instr_type is UnaryOp) and
                type(instr.dest) is VirtualRegister and
                instr.dest.id == vreg_id):
                return False
            # CMP preserves A on 65816 — it only sets flags. If the left operand
            # is our vreg (already in A), no load is needed and A survives.
            if instr_type is Compare:
                if type(instr.left) is VirtualRegister and instr.left.id == vreg_id:
                    return False
            return True

        if instr_type is Load:
            # LDA into vreg clobbers A
            return True

        if instr_type is LoadIndirect:
            # LDA [dp],Y into vreg clobbers A
            return True

        if instr_type is StoreIndirect:
            # STA [dp],Y uses A. If source is our vreg, codegen emits STA
            # directly from A (our vreg IS A). Otherwise it needs LDA first.
            if type(instr.source) is VirtualRegister and instr.source.id == vreg_id:
                return False  # STA [dp],Y from our vreg = STA from A (no clobber)
            return True

        if instr_type is StatusFlagRead:
            # PHP; PLA; AND - clobbers A
            return True

        return False

    def _clobbers_b(self, instr: Any, vreg_id: int) -> bool:
        """
        Check if an instruction clobbers the B register.

        B is the high byte of the 16-bit accumulator. It is clobbered by:
        - XBA instruction (swaps A and B) — emitted for Move to/from B
        - Any m16 operation (REP #$20 + LDA writes both A low and B high)

        Since any instruction involving u16 types may trigger m16 mode
        during codegen, B is clobbered by TypeConvert, u16 BinaryOp/UnaryOp,
        u16 Store/Load, u16 Compare, and u16 Moves through memory.
        """
        instr_type = type(instr)

        if instr_type is Move:
            # Move to/from B: XBA swaps, clobbers B
            if type(instr.dest) is HardwareRegister and instr.dest.name == 'B':
                return True
            if type(instr.source) is HardwareRegister and instr.source.name == 'B':
                return True
            # Move of u16 value through A (LDA/STA in m16) clobbers B
            if self._instr_involves_u16(instr):
                return True
            return False

        # TypeConvert always uses m16 (widening or narrowing)
        if instr_type is TypeConvert:
            return True

        # Any u16 arithmetic/logic clobbers B via m16 mode
        if instr_type is BinaryOp or instr_type is UnaryOp or instr_type is Compare:
            return self._instr_involves_u16(instr)

        # u16 Store/Load use m16 LDA/STA
        if instr_type is Store or instr_type is StoreIndirect or instr_type is Load or instr_type is LoadIndirect:
            return self._instr_involves_u16(instr)

        return False

    def _instr_involves_u16(self, instr: Any) -> bool:
        """Check if an instruction involves u16 types (may trigger m16 mode)."""
        # Check type_info on the instruction
        type_info = getattr(instr, 'type_info', None)
        if type_info is not None:
            size = get_unified_type_size(type_info)
            if size >= 2:
                return True
        # Check dest vreg size
        dest = getattr(instr, 'dest', None)
        if type(dest) is VirtualRegister:
            if get_vreg_size(dest) >= 2:
                return True
        return False

    def _clobbers_xy(self, instr: Any, hw_reg: str, vreg_id: int) -> bool:
        """
        Check if an instruction clobbers X or Y register.

        X/Y are only modified by:
        - Move to X/Y (LDX, LDY, TAX, TAY, TXY, TYX)
        - BinaryOp with dest=HardwareRegister X/Y (INX, DEX, INY, DEY)
        - RestoreRegister for X/Y (PLX, PLY)
        - Calls that don't preserve X/Y (handled in caller)
        """
        instr_type = type(instr)

        if instr_type is Move:
            if type(instr.dest) is HardwareRegister and instr.dest.name == hw_reg:
                # Move to our register — unless source is our vreg (no-op)
                if type(instr.source) is VirtualRegister and instr.source.id == vreg_id:
                    return False  # Value already in register
                return True
            return False

        if instr_type is BinaryOp:
            # INX/DEX/INY/DEY patterns: BinaryOp dest=HardwareRegister
            if type(instr.dest) is HardwareRegister and instr.dest.name == hw_reg:
                return True
            return False

        if instr_type is RestoreRegister:
            # PLX/PLY restores clobber the register
            if instr.register.name == hw_reg:
                return True
            return False

        # Other instructions don't modify X/Y
        # (they may use X/Y for indexing but don't modify them)
        return False

    def _collect_virtual_registers(self, exclude: Set[VirtualRegister] = None) -> List[VirtualRegister]:
        """Collect all virtual registers, optionally excluding some."""
        if exclude is None:
            exclude = set()

        vregs: Set[VirtualRegister] = set()

        for block in self.func.blocks.values():
            for instr in block.instructions:
                uses = self.liveness_analyzer._get_uses(instr)
                defs = self.liveness_analyzer._get_defs(instr)

                for var in uses + defs:
                    if isinstance(var, VirtualRegister) and var not in exclude:
                        vregs.add(var)

        return sorted(vregs, key=lambda v: v.id)

    def get_slot_for_register(self, vreg: VirtualRegister, allocation: SlotAllocation) -> Optional[int]:
        """
        Get the slot number for a virtual register.

        Args:
            vreg: Virtual register
            allocation: Slot allocation result

        Returns:
            Slot number, or None if not allocated
        """
        return allocation.register_to_slot.get(vreg)
