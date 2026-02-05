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
)
from r65.compiler.mir.liveness import LivenessAnalyzer
from r65.compiler.hir.unified_type_utils import get_unified_type_size
from r65.compiler.hir import HIRError


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

    # Stack parameters: vreg -> final offset (after frame adjustment)
    param_offsets: Dict[VirtualRegister, int] = None

    # Param sizes
    param_sizes: Dict[VirtualRegister, int] = None

    def __post_init__(self):
        if self.hw_coalesceable is None:
            self.hw_coalesceable = {}
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
        prologue_stack_bytes: int = 0
    ):
        """
        Initialize unified slot allocator.

        Args:
            mir_func: MIR function to allocate slots for
            preassigned: Stack parameters with their base offsets
            prologue_stack_bytes: Bytes pushed by prologue (return addr + saved regs)
        """
        self.func = mir_func
        self.preassigned = preassigned or []
        self.prologue_stack_bytes = prologue_stack_bytes
        self.liveness_analyzer = LivenessAnalyzer(mir_func)

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

        # Identify vregs that can stay in hardware registers
        hw_coalesceable = self._find_hw_coalesceable_vregs()

        # Collect local vregs (excluding hw-coalesceable and preassigned params)
        exclude_vregs = set(hw_coalesceable.keys())
        for slot in self.preassigned:
            exclude_vregs.add(slot.vreg)

        local_vregs = self._collect_virtual_registers(exclude=exclude_vregs)

        # Calculate sizes for locals
        local_sizes: Dict[VirtualRegister, int] = {}
        for vreg in local_vregs:
            local_sizes[vreg] = self._get_vreg_size(vreg)

        # Allocate locals with liveness-based reuse
        local_slots, frame_size, slots_saved = self._allocate_locals(local_vregs, local_sizes)

        # Compute final param offsets (adjusted for frame allocation)
        param_offsets: Dict[VirtualRegister, int] = {}
        param_sizes: Dict[VirtualRegister, int] = {}

        for slot in self.preassigned:
            # Final offset = base_offset + prologue_bytes + frame_size
            final_offset = slot.base_offset + self.prologue_stack_bytes + frame_size
            param_offsets[slot.vreg] = final_offset
            param_sizes[slot.vreg] = slot.size

        return SlotAllocation(
            register_to_slot=local_slots,
            register_to_size=local_sizes,
            total_slots=frame_size,
            variables_count=len(local_vregs),
            slots_saved=slots_saved,
            hw_coalesceable=hw_coalesceable,
            param_offsets=param_offsets,
            param_sizes=param_sizes
        )

    def _allocate_locals(
        self,
        local_vregs: List[VirtualRegister],
        local_sizes: Dict[VirtualRegister, int]
    ) -> Tuple[Dict[VirtualRegister, int], int, int]:
        """
        Allocate local variables with liveness-based reuse.

        Args:
            local_vregs: Local variables to allocate
            local_sizes: Size of each local

        Returns:
            (slot_mapping, frame_size, slots_saved)
        """
        if not local_vregs:
            return {}, 0, 0

        allocation: Dict[VirtualRegister, int] = {}
        # Track allocated ranges: (start, end, vreg)
        allocated_ranges: List[Tuple[int, int, VirtualRegister]] = []
        next_slot = 0

        # Sort by size descending for better packing
        sorted_vregs = sorted(local_vregs, key=lambda v: -local_sizes[v])

        for vreg in sorted_vregs:
            size = local_sizes[vreg]
            assigned_slot = None

            # Try to reuse existing slot
            for start_slot in range(next_slot):
                end_slot = start_slot + size
                can_use = True

                # Check against other locals
                for (other_start, other_end, other_vreg) in allocated_ranges:
                    if start_slot < other_end and end_slot > other_start:
                        if self.liveness_analyzer.interferes(vreg, other_vreg):
                            can_use = False
                            break

                if can_use:
                    assigned_slot = start_slot
                    break

            if assigned_slot is None:
                assigned_slot = next_slot
                next_slot = assigned_slot + size
            else:
                new_end = assigned_slot + size
                if new_end > next_slot:
                    next_slot = new_end

            allocation[vreg] = assigned_slot
            allocated_ranges.append((assigned_slot, assigned_slot + size, vreg))

        # Frame size is the total local slots needed
        # Ensure frame_size covers all allocated ranges (safety check)
        max_end = 0
        for (start, end, _) in allocated_ranges:
            if end > max_end:
                max_end = end
        frame_size = max(next_slot, max_end)

        # Calculate slots saved
        total_without_reuse = sum(local_sizes.values())
        slots_saved = total_without_reuse - frame_size

        return allocation, frame_size, slots_saved

    def _get_vreg_size(self, vreg: VirtualRegister) -> int:
        """Get size of virtual register in bytes."""
        try:
            return get_unified_type_size(vreg.type_info)
        except (HIRError, AttributeError, TypeError):
            return 1

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

                if isinstance(instr, Move):
                    if isinstance(instr.dest, VirtualRegister):
                        vreg_id = instr.dest.id
                        if vreg_id not in vreg_defs:
                            vreg_defs[vreg_id] = []

                        if isinstance(instr.source, HardwareRegister):
                            vreg_defs[vreg_id].append((instr, instr.source.name))
                        else:
                            vreg_defs[vreg_id].append((instr, None))

                    if isinstance(instr.source, VirtualRegister):
                        vreg_id = instr.source.id
                        if vreg_id not in vreg_uses:
                            vreg_uses[vreg_id] = []
                        vreg_uses[vreg_id].append(instr)

                elif isinstance(instr, Return):
                    if instr.values:
                        for val in instr.values:
                            if isinstance(val, VirtualRegister):
                                vreg_id = val.id
                                if vreg_id not in vreg_uses:
                                    vreg_uses[vreg_id] = []
                                vreg_uses[vreg_id].append(instr)

                else:
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if isinstance(var, VirtualRegister):
                            if var.id not in vreg_uses:
                                vreg_uses[var.id] = []
                            vreg_uses[var.id].append(instr)

        # Build candidate list: (vreg_id, def_instr, hw_reg, uses)
        candidates = []
        for vreg_id, defs in vreg_defs.items():
            if len(defs) != 1:
                continue
            def_instr, hw_reg = defs[0]
            if hw_reg is None:
                continue
            uses = vreg_uses.get(vreg_id, [])
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

                all_returns = all(isinstance(use, Return) for use in uses)

                if all_returns:
                    # Original path: only check for clobbering calls
                    if self._is_return_path_safe(hw_reg, def_pos, uses, instr_positions, noop_instrs):
                        self._mark_coalesceable(vreg_id, hw_reg, coalesceable)
                        noop_instrs.add(id(def_instr))
                    else:
                        remaining.append((vreg_id, def_instr, hw_reg, uses))
                else:
                    # Extended path: full clobber analysis
                    if self._is_hw_unclobbered_in_range(
                        hw_reg, vreg_id, def_instr, uses, instr_positions, noop_instrs
                    ):
                        self._mark_coalesceable(vreg_id, hw_reg, coalesceable)
                        noop_instrs.add(id(def_instr))
                    else:
                        remaining.append((vreg_id, def_instr, hw_reg, uses))

            candidates = remaining

        return coalesceable

    def _mark_coalesceable(
        self, vreg_id: int, hw_reg: str, coalesceable: Dict[VirtualRegister, str]
    ):
        """Mark a vreg as hw-coalesceable by finding its def instruction."""
        for block in self.func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Move) and isinstance(instr.dest, VirtualRegister):
                    if instr.dest.id == vreg_id:
                        coalesceable[instr.dest] = hw_reg
                        return

    def _is_return_path_safe(
        self,
        hw_reg: str,
        def_pos: Tuple[int, int],
        uses: List,
        instr_positions: Dict[Any, Tuple[int, int]],
        noop_instrs: Set[int],
    ) -> bool:
        """Check if a return-only coalescence is safe (no clobbering calls)."""
        def_block_id, def_idx = def_pos

        for use_instr in uses:
            use_pos = instr_positions.get(id(use_instr))
            if use_pos is None:
                continue

            use_block_id, use_idx = use_pos

            if use_block_id == def_block_id:
                block = self.func.blocks[def_block_id]
                for i in range(def_idx + 1, use_idx):
                    instr = block.instructions[i]
                    if id(instr) in noop_instrs:
                        continue
                    if isinstance(instr, Call):
                        preserved = set()
                        if instr.preserves_attr:
                            preserved = set(instr.preserves_attr.registers)
                        if hw_reg not in preserved:
                            return False
            else:
                # Cross-block: conservatively check for any clobbering call
                for block in self.func.blocks.values():
                    for instr in block.instructions:
                        if id(instr) in noop_instrs:
                            continue
                        if isinstance(instr, Call):
                            preserved = set()
                            if instr.preserves_attr:
                                preserved = set(instr.preserves_attr.registers)
                            if hw_reg not in preserved:
                                return False
        return True

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

        All uses must be in the same block as the def. Scans instructions
        between def_idx and max(use_idx) for anything that clobbers the
        hardware register.

        Args:
            hw_reg: Hardware register name ('A', 'B', 'X', 'Y')
            vreg_id: The vreg ID being checked
            def_instr: The Move instruction that defines the vreg
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

        # All uses must be in the same block
        max_use_idx = def_idx
        for use_instr in uses:
            use_pos = instr_positions.get(id(use_instr))
            if use_pos is None:
                return False
            use_block_id, use_idx = use_pos
            if use_block_id != def_block_id:
                return False  # Cross-block: bail conservatively
            max_use_idx = max(max_use_idx, use_idx)

        # Scan instructions between def and last use for clobbers
        block = self.func.blocks[def_block_id]
        for i in range(def_idx + 1, max_use_idx):
            instr = block.instructions[i]
            if self._instruction_clobbers_register(instr, hw_reg, vreg_id, noop_instrs):
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

        # Calls clobber all registers not in preserves
        if isinstance(instr, Call):
            preserved = set()
            if instr.preserves_attr:
                preserved = set(instr.preserves_attr.registers)
            return hw_reg not in preserved

        # InlineAsm: conservatively assume it clobbers everything
        if isinstance(instr, InlineAsm):
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
        if isinstance(instr, Store):
            # Store uses A to transfer values. If the source is our vreg,
            # codegen will use STA directly (A already has the value).
            # If source is a different vreg or hw register, it clobbers A.
            if isinstance(instr.source, VirtualRegister) and instr.source.id == vreg_id:
                return False  # STA from our vreg = STA from A (no clobber)
            if isinstance(instr.source, HardwareRegister) and instr.source.name == 'A':
                return False  # STA from A (already in A)
            return True  # Needs LDA from somewhere else

        if isinstance(instr, Move):
            # Move to vreg (other than ours): LDA + STA clobbers A
            if isinstance(instr.dest, VirtualRegister) and instr.dest.id != vreg_id:
                return True
            # Move to hw register from vreg/immediate: codegen loads through A
            if isinstance(instr.dest, HardwareRegister):
                if instr.dest.name in ('X', 'Y'):
                    # LDX/LDY don't clobber A (unless source is stack-relative for X/Y)
                    # But Move to X/Y from vreg goes LDA vreg; TAX (clobbers A)
                    if isinstance(instr.source, VirtualRegister):
                        return True
                    return False  # LDX #imm or LDX addr don't clobber A
                if instr.dest.name == 'B':
                    return True  # XBA clobbers A
                if instr.dest.name == 'A':
                    return True  # Overwrites A
            # Move from B: XBA clobbers A
            if isinstance(instr.source, HardwareRegister) and instr.source.name == 'B':
                return True
            return False

        if isinstance(instr, (BinaryOp, UnaryOp, TypeConvert, Compare, BitTest, Rotate, ToBool)):
            # All arithmetic/comparison ops route through A
            return True

        if isinstance(instr, Load):
            # LDA into vreg clobbers A
            return True

        if isinstance(instr, (LoadIndirect, StoreIndirect)):
            # Indirect operations use A
            return True

        if isinstance(instr, StatusFlagRead):
            # PHP; PLA; AND - clobbers A
            return True

        return False

    def _clobbers_b(self, instr: Any, vreg_id: int) -> bool:
        """
        Check if an instruction clobbers the B register.

        B is the high byte of the 16-bit accumulator. It's only modified by:
        - XBA instruction (swaps A and B)
        - Move to/from B register in MIR
        - Calls that don't preserve B
        """
        if isinstance(instr, Move):
            # Move to B: XBA to store, clobbers B
            if isinstance(instr.dest, HardwareRegister) and instr.dest.name == 'B':
                return True
            # Move from B: XBA to access, but XBA swaps - clobbers B
            if isinstance(instr.source, HardwareRegister) and instr.source.name == 'B':
                return True
            return False

        # Most other instructions don't touch B
        # (BinaryOp, Store, Load etc. only use A, not B)
        return False

    def _clobbers_xy(self, instr: Any, hw_reg: str, vreg_id: int) -> bool:
        """
        Check if an instruction clobbers X or Y register.

        X/Y are only modified by:
        - Move to X/Y (LDX, LDY, TAX, TAY, TXY, TYX)
        - Calls that don't preserve X/Y
        """
        if isinstance(instr, Move):
            if isinstance(instr.dest, HardwareRegister) and instr.dest.name == hw_reg:
                return True
            return False

        # Other instructions don't typically clobber X/Y
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
