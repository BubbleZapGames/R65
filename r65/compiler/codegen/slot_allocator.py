"""
Stack slot allocator with reuse optimization.

Allocates memory slots for local variables and temporaries, reusing slots
when variable lifetimes don't overlap (determined by liveness analysis).

Handles multi-byte values (e.g., 3-byte far pointers) by allocating
consecutive slots and tracking the full range for interference checking.
"""

from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
from r65.compiler.mir.nodes import VirtualRegister, MIRFunction, Move, Return, HardwareRegister
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

    # Vregs that can stay in hardware registers (no spill needed)
    # Maps vreg to the hardware register name it should stay in
    hw_coalesceable: Dict[VirtualRegister, str] = None

    def __post_init__(self):
        if self.hw_coalesceable is None:
            self.hw_coalesceable = {}


class StackSlotAllocator:
    """
    Allocates stack/zero-page slots with reuse optimization.

    Uses liveness analysis to determine which variables can share
    the same memory location, reducing memory usage.
    """

    def __init__(self, mir_func: MIRFunction, exclude_vreg_ids: Optional[Set[int]] = None):
        """
        Initialize slot allocator.

        Args:
            mir_func: MIR function to allocate slots for
            exclude_vreg_ids: Set of vreg IDs to exclude from allocation (e.g., stack params)
        """
        self.func = mir_func
        self.liveness_analyzer = LivenessAnalyzer(mir_func)
        self.exclude_vreg_ids = exclude_vreg_ids or set()

    def allocate(self) -> SlotAllocation:
        """
        Allocate slots for all virtual registers with reuse.

        Returns:
            SlotAllocation with mapping and statistics
        """
        # Run liveness analysis
        self.liveness_analyzer.analyze()

        # Identify vregs that can stay in hardware registers
        hw_coalesceable = self._find_hw_coalesceable_vregs()

        # Build exclusion set: hw-coalesceable vregs + pre-allocated stack params
        exclude_vregs = set(hw_coalesceable.keys())
        # Also exclude vregs by ID (for stack params that were pre-allocated)
        for vreg in self._get_all_vregs():
            if vreg.id in self.exclude_vreg_ids:
                exclude_vregs.add(vreg)

        # Collect all virtual registers with their sizes, excluding hw-coalesceable and stack params
        virtual_regs = self._collect_virtual_registers(exclude=exclude_vregs)

        if not virtual_regs:
            return SlotAllocation(
                register_to_slot={},
                register_to_size={},
                total_slots=0,
                variables_count=0,
                slots_saved=0,
                hw_coalesceable=hw_coalesceable
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
            slots_saved=slots_saved,
            hw_coalesceable=hw_coalesceable
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

        # Find all definitions and uses for each vreg
        vreg_defs: Dict[int, List] = {}  # vreg.id -> list of (instr, hw_reg or None)
        vreg_uses: Dict[int, List] = {}  # vreg.id -> list of instr

        for block in self.func.blocks.values():
            for instr in block.instructions:
                # Check for Move from HardwareRegister to VirtualRegister
                if isinstance(instr, Move):
                    if isinstance(instr.dest, VirtualRegister):
                        vreg_id = instr.dest.id
                        if vreg_id not in vreg_defs:
                            vreg_defs[vreg_id] = []

                        # Check if source is a hardware register
                        if isinstance(instr.source, HardwareRegister):
                            vreg_defs[vreg_id].append((instr, instr.source.name))
                        else:
                            vreg_defs[vreg_id].append((instr, None))

                    # Check uses in source
                    if isinstance(instr.source, VirtualRegister):
                        vreg_id = instr.source.id
                        if vreg_id not in vreg_uses:
                            vreg_uses[vreg_id] = []
                        vreg_uses[vreg_id].append(instr)

                # Check Return instruction
                elif isinstance(instr, Return):
                    if instr.values:
                        for val in instr.values:
                            if isinstance(val, VirtualRegister):
                                vreg_id = val.id
                                if vreg_id not in vreg_uses:
                                    vreg_uses[vreg_id] = []
                                vreg_uses[vreg_id].append(instr)

                # For other instructions, mark vregs as used (not coalesceable)
                else:
                    uses = self.liveness_analyzer._get_uses(instr)
                    for var in uses:
                        if isinstance(var, VirtualRegister):
                            if var.id not in vreg_uses:
                                vreg_uses[var.id] = []
                            vreg_uses[var.id].append(instr)

        # Find vregs that meet coalescence criteria
        for vreg_id, defs in vreg_defs.items():
            # Must have exactly one definition
            if len(defs) != 1:
                continue

            instr, hw_reg = defs[0]
            # Must be from a hardware register
            if hw_reg is None:
                continue

            # Check uses - must only be used in Return instructions
            uses = vreg_uses.get(vreg_id, [])
            if not uses:
                # No uses - can coalesce (value unused)
                # Find the vreg object
                for block in self.func.blocks.values():
                    for instr in block.instructions:
                        if isinstance(instr, Move) and isinstance(instr.dest, VirtualRegister):
                            if instr.dest.id == vreg_id:
                                coalesceable[instr.dest] = hw_reg
                                break
                continue

            # All uses must be Return instructions
            all_returns = all(isinstance(use, Return) for use in uses)
            if not all_returns:
                continue

            # Find the vreg object and add to coalesceable
            for block in self.func.blocks.values():
                for check_instr in block.instructions:
                    if isinstance(check_instr, Move) and isinstance(check_instr.dest, VirtualRegister):
                        if check_instr.dest.id == vreg_id:
                            coalesceable[check_instr.dest] = hw_reg
                            break

        return coalesceable

    def _collect_virtual_registers(self, exclude: Set[VirtualRegister] = None) -> List[VirtualRegister]:
        """
        Collect all virtual registers used in the function.

        Args:
            exclude: Set of vregs to exclude from collection (e.g., hw-coalesceable)

        Returns:
            List of unique virtual registers
        """
        if exclude is None:
            exclude = set()

        vregs: Set[VirtualRegister] = set()

        for block_id, block in self.func.blocks.items():
            for instr in block.instructions:
                # Get all virtual registers used/defined by instruction
                uses = self.liveness_analyzer._get_uses(instr)
                defs = self.liveness_analyzer._get_defs(instr)

                for var in uses + defs:
                    if isinstance(var, VirtualRegister) and var not in exclude:
                        vregs.add(var)

        return sorted(vregs, key=lambda v: v.id)

    def _get_all_vregs(self) -> List[VirtualRegister]:
        """Get all virtual registers in the function (no exclusions)."""
        return self._collect_virtual_registers(exclude=set())

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
