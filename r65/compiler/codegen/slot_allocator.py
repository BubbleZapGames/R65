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
from r65.compiler.mir.nodes import VirtualRegister, MIRFunction, Move, Return, HardwareRegister
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
        frame_size = next_slot

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
        2. Its only use is as a Return source (which goes back to a hw register)

        These vregs don't need stack slots - they can stay in the register.

        Returns:
            Dict mapping vreg to the hardware register name it should stay in
        """
        coalesceable: Dict[VirtualRegister, str] = {}

        vreg_defs: Dict[int, List] = {}
        vreg_uses: Dict[int, List] = {}

        for block in self.func.blocks.values():
            for instr in block.instructions:
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

        for vreg_id, defs in vreg_defs.items():
            if len(defs) != 1:
                continue

            instr, hw_reg = defs[0]
            if hw_reg is None:
                continue

            uses = vreg_uses.get(vreg_id, [])
            if not uses:
                for block in self.func.blocks.values():
                    for check_instr in block.instructions:
                        if isinstance(check_instr, Move) and isinstance(check_instr.dest, VirtualRegister):
                            if check_instr.dest.id == vreg_id:
                                coalesceable[check_instr.dest] = hw_reg
                                break
                continue

            all_returns = all(isinstance(use, Return) for use in uses)
            if not all_returns:
                continue

            for block in self.func.blocks.values():
                for check_instr in block.instructions:
                    if isinstance(check_instr, Move) and isinstance(check_instr.dest, VirtualRegister):
                        if check_instr.dest.id == vreg_id:
                            coalesceable[check_instr.dest] = hw_reg
                            break

        return coalesceable

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
