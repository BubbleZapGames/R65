"""
Stack slot allocator with reuse optimization.

Allocates memory slots for local variables and temporaries, reusing slots
when variable lifetimes don't overlap (determined by liveness analysis).
"""

from typing import Dict, Set, List, Optional
from dataclasses import dataclass
from r65.compiler.mir.nodes import VirtualRegister, MIRFunction
from r65.compiler.mir.liveness import LivenessAnalyzer


@dataclass
class SlotAllocation:
    """
    Result of slot allocation.

    Maps virtual registers to memory slot numbers, with slots reused
    when possible based on liveness analysis.
    """
    # Maps VirtualRegister to slot number (offset from base)
    register_to_slot: Dict[VirtualRegister, int]

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

        # Collect all virtual registers
        virtual_regs = self._collect_virtual_registers()

        if not virtual_regs:
            return SlotAllocation(
                register_to_slot={},
                total_slots=0,
                variables_count=0,
                slots_saved=0
            )

        # Allocate slots using graph coloring approach
        register_to_slot = self._allocate_with_reuse(virtual_regs)

        # Calculate statistics
        total_slots = max(register_to_slot.values()) + 1 if register_to_slot else 0
        variables_count = len(virtual_regs)
        slots_saved = variables_count - total_slots

        return SlotAllocation(
            register_to_slot=register_to_slot,
            total_slots=total_slots,
            variables_count=variables_count,
            slots_saved=slots_saved
        )

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

    def _allocate_with_reuse(self, virtual_regs: List[VirtualRegister]) -> Dict[VirtualRegister, int]:
        """
        Allocate slots with reuse using greedy graph coloring.

        Algorithm:
        1. For each variable, try to assign it to an existing slot
        2. A variable can use a slot if it doesn't interfere with any
           other variable already using that slot
        3. If no existing slot works, allocate a new slot

        Args:
            virtual_regs: List of virtual registers to allocate

        Returns:
            Dictionary mapping VirtualRegister to slot number
        """
        allocation: Dict[VirtualRegister, int] = {}
        slot_to_vars: Dict[int, List[VirtualRegister]] = {}  # Track which vars use each slot
        next_slot = 0

        for vreg in virtual_regs:
            # Try to find an existing slot this variable can use
            assigned_slot = None

            for slot_num in range(next_slot):
                # Check if this variable interferes with any variable using this slot
                can_use_slot = True

                for other_var in slot_to_vars.get(slot_num, []):
                    if self.liveness_analyzer.interferes(vreg, other_var):
                        can_use_slot = False
                        break

                if can_use_slot:
                    assigned_slot = slot_num
                    break

            # If no existing slot works, allocate a new one
            if assigned_slot is None:
                assigned_slot = next_slot
                next_slot += 1

            # Assign the slot
            allocation[vreg] = assigned_slot

            # Track which variables use this slot
            if assigned_slot not in slot_to_vars:
                slot_to_vars[assigned_slot] = []
            slot_to_vars[assigned_slot].append(vreg)

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
