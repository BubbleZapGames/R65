"""
Stack slot allocator with reuse optimization.

Allocates memory slots for local variables and temporaries, reusing slots
when variable lifetimes don't overlap (determined by liveness analysis).

Handles multi-byte values (e.g., 3-byte far pointers) by allocating
consecutive slots and tracking the full range for interference checking.
"""

from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
from r65.compiler.mir.nodes import VirtualRegister, MIRFunction
from r65.compiler.mir.liveness import LivenessAnalyzer
from r65.compiler.hir.unified_type_utils import get_unified_type_size
from r65.compiler.hir import HIRError


@dataclass
class SlotAllocation:
    """
    Result of slot allocation.

    Maps virtual registers to memory slot numbers, with slots reused
    when possible based on liveness analysis.
    """
    # Maps VirtualRegister to slot number (offset from base)
    register_to_slot: Dict[VirtualRegister, int]

    # Maps VirtualRegister to size in bytes (number of consecutive slots)
    register_to_size: Dict[VirtualRegister, int]

    # Total number of slots needed
    total_slots: int

    # Statistics
    variables_count: int
    slots_saved: int  # Number of slots saved by reuse


class StackSlotAllocator:
    """
    Allocates stack/zero-page slots with reuse optimization.

    Uses liveness analysis to determine which variables can share
    the same memory location, reducing memory usage.
    """

    def __init__(self, mir_func: MIRFunction):
        """
        Initialize slot allocator.

        Args:
            mir_func: MIR function to allocate slots for
        """
        self.func = mir_func
        self.liveness_analyzer = LivenessAnalyzer(mir_func)

    def allocate(self) -> SlotAllocation:
        """
        Allocate slots for all virtual registers with reuse.

        Returns:
            SlotAllocation with mapping and statistics
        """
        # Run liveness analysis
        self.liveness_analyzer.analyze()

        # Collect all virtual registers with their sizes
        virtual_regs = self._collect_virtual_registers()

        if not virtual_regs:
            return SlotAllocation(
                register_to_slot={},
                register_to_size={},
                total_slots=0,
                variables_count=0,
                slots_saved=0
            )

        # Calculate sizes for all vregs
        vreg_sizes: Dict[VirtualRegister, int] = {}
        for vreg in virtual_regs:
            vreg_sizes[vreg] = self._get_vreg_size(vreg)

        # Allocate slots using graph coloring approach (with size awareness)
        register_to_slot = self._allocate_with_reuse(virtual_regs, vreg_sizes)

        # Calculate total slots needed (considering multi-byte values)
        total_slots = 0
        for vreg, slot in register_to_slot.items():
            end_slot = slot + vreg_sizes[vreg]
            if end_slot > total_slots:
                total_slots = end_slot

        # Calculate slots that would have been used without reuse
        total_bytes_without_reuse = sum(vreg_sizes.values())
        variables_count = len(virtual_regs)
        slots_saved = total_bytes_without_reuse - total_slots

        return SlotAllocation(
            register_to_slot=register_to_slot,
            register_to_size=vreg_sizes,
            total_slots=total_slots,
            variables_count=variables_count,
            slots_saved=slots_saved
        )

    def _get_vreg_size(self, vreg: VirtualRegister) -> int:
        """
        Get the size in bytes of a virtual register.

        Args:
            vreg: Virtual register

        Returns:
            Size in bytes (1, 2, or 3 typically)
        """
        try:
            return get_unified_type_size(vreg.type_info)
        except (HIRError, AttributeError, TypeError):
            # Default to 1 byte if type info is missing or invalid
            return 1

    def _collect_virtual_registers(self) -> List[VirtualRegister]:
        """
        Collect all virtual registers used in the function.

        Returns:
            List of unique virtual registers
        """
        vregs: Set[VirtualRegister] = set()

        for block_id, block in self.func.blocks.items():
            for instr in block.instructions:
                # Get all virtual registers used/defined by instruction
                uses = self.liveness_analyzer._get_uses(instr)
                defs = self.liveness_analyzer._get_defs(instr)

                for var in uses + defs:
                    if isinstance(var, VirtualRegister):
                        vregs.add(var)

        return sorted(vregs, key=lambda v: v.id)

    def _allocate_with_reuse(
        self,
        virtual_regs: List[VirtualRegister],
        vreg_sizes: Dict[VirtualRegister, int]
    ) -> Dict[VirtualRegister, int]:
        """
        Allocate slots with reuse using greedy graph coloring.

        Algorithm:
        1. For each variable, try to assign it to an existing slot range
        2. A variable can use a slot range if:
           - The slot range doesn't overlap with any interfering variable's range
        3. If no existing slot range works, allocate at the end

        Handles multi-byte values (e.g., 3-byte far pointers) by treating
        them as occupying consecutive slots.

        Args:
            virtual_regs: List of virtual registers to allocate
            vreg_sizes: Dictionary mapping vregs to their sizes in bytes

        Returns:
            Dictionary mapping VirtualRegister to slot number (start of range)
        """
        allocation: Dict[VirtualRegister, int] = {}
        # Track allocated ranges: list of (start_slot, end_slot, vreg)
        allocated_ranges: List[Tuple[int, int, VirtualRegister]] = []
        next_slot = 0

        # Sort by size descending to allocate larger values first (better packing)
        sorted_vregs = sorted(virtual_regs, key=lambda v: -vreg_sizes[v])

        for vreg in sorted_vregs:
            size = vreg_sizes[vreg]

            # Try to find an existing slot range this variable can use
            assigned_slot = None

            # Try each possible starting position
            for start_slot in range(next_slot):
                end_slot = start_slot + size

                # Check if this range overlaps with any interfering variable
                can_use_range = True

                for (other_start, other_end, other_var) in allocated_ranges:
                    # Check if ranges overlap
                    if start_slot < other_end and end_slot > other_start:
                        # Ranges overlap - check if they interfere
                        if self.liveness_analyzer.interferes(vreg, other_var):
                            can_use_range = False
                            break

                if can_use_range:
                    assigned_slot = start_slot
                    break

            # If no existing slot range works, allocate at the end
            if assigned_slot is None:
                assigned_slot = next_slot
                next_slot = assigned_slot + size
            else:
                # Update next_slot if this allocation extended it
                new_end = assigned_slot + size
                if new_end > next_slot:
                    next_slot = new_end

            # Record the allocation
            allocation[vreg] = assigned_slot
            allocated_ranges.append((assigned_slot, assigned_slot + size, vreg))

        return allocation

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
