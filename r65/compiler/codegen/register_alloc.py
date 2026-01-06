"""
Register allocation: maps virtual registers to physical locations.

Maps MIR virtual registers to:
1. Scratch registers (designated zero-page locations)
2. Stack slots (when scratch pool exhausted)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from r65.compiler.mir.nodes import VirtualRegister, HardwareRegister, MIRFunction
from r65.compiler.codegen.slot_allocator import StackSlotAllocator, SlotAllocation
from r65.compiler.codegen.type_utils import get_type_size


class LocationKind(Enum):
    """Type of physical location for a virtual register."""
    HARDWARE = "hardware"      # Hardware register (A, X, Y)
    SCRATCH = "scratch"        # Scratch register (zero-page)
    STACK = "stack"            # Stack slot
    MEMORY = "memory"          # Static memory location
    IMMEDIATE = "immediate"    # Immediate value (constant)


@dataclass
class PhysicalLocation:
    """
    Physical location of a value.

    Represents where a virtual register is actually stored at runtime.
    """
    kind: LocationKind

    # For HARDWARE: hardware register name
    hw_register: Optional[str] = None

    # For SCRATCH: zero-page address
    scratch_addr: Optional[int] = None

    # For STACK: stack offset from frame pointer
    stack_offset: Optional[int] = None

    # For MEMORY: absolute address
    memory_addr: Optional[int] = None

    # For IMMEDIATE: constant value
    immediate_value: Optional[int] = None

    # For indexed addressing: 'X' or 'Y'
    index_register: Optional[str] = None

    # Size in bytes
    size: int = 1

    def __str__(self) -> str:
        """String representation for debugging."""
        if self.kind == LocationKind.HARDWARE:
            return f"{self.hw_register}"
        elif self.kind == LocationKind.SCRATCH:
            return f"${self.scratch_addr:04X}"
        elif self.kind == LocationKind.STACK:
            return f"[SP+{self.stack_offset}]"
        elif self.kind == LocationKind.MEMORY:
            return f"${self.memory_addr:06X}"
        elif self.kind == LocationKind.IMMEDIATE:
            return f"#{self.immediate_value}"
        return "<?>"


@dataclass
class ScratchRegister:
    """A scratch register available for temporary allocation."""
    address: int      # Zero-page address
    size: int         # Size in bytes (1 or 2)
    name: str         # Name (e.g., "SCRATCH0")
    is_free: bool = True


class ScratchRegisterPool:
    """
    Manages pool of scratch registers for temporary allocation.

    Scratch registers are designated zero-page locations marked with
    #[zeropage(addr, register)] attribute.
    """

    def __init__(self):
        """Initialize empty scratch pool."""
        self.scratches: List[ScratchRegister] = []
        self.allocated: Dict[int, VirtualRegister] = {}  # vreg.id → VirtualRegister

    def add_scratch(self, address: int, size: int, name: str):
        """
        Add a scratch register to the pool.

        Args:
            address: Zero-page address
            size: Size in bytes
            name: Scratch register name
        """
        scratch = ScratchRegister(
            address=address,
            size=size,
            name=name,
            is_free=True
        )
        self.scratches.append(scratch)

    def allocate(self, vreg: VirtualRegister) -> Optional[ScratchRegister]:
        """
        Allocate a scratch register for a virtual register.

        Args:
            vreg: Virtual register to allocate

        Returns:
            ScratchRegister if available, None if pool exhausted
        """
        # Check if already allocated
        if vreg.id in self.allocated:
            # Find the scratch
            for scratch in self.scratches:
                if not scratch.is_free and self.allocated.get(vreg.id) == vreg:
                    # Find which scratch was allocated by checking all scratches
                    # This is inefficient but simple for v1
                    pass

        # Find compatible free scratch (must match or exceed size)
        vreg_size = self._get_vreg_size(vreg)
        for scratch in self.scratches:
            if scratch.is_free and scratch.size >= vreg_size:
                scratch.is_free = False
                self.allocated[vreg.id] = vreg
                return scratch

        # No free scratch available
        return None

    def free(self, vreg: VirtualRegister):
        """
        Free scratch register allocated to virtual register.

        Args:
            vreg: Virtual register to free
        """
        if vreg.id not in self.allocated:
            return

        # Find and free the scratch
        for scratch in self.scratches:
            if not scratch.is_free and self.allocated.get(vreg.id) == vreg:
                scratch.is_free = True
                del self.allocated[vreg.id]
                return

    def _get_vreg_size(self, vreg: VirtualRegister) -> int:
        """Get size of virtual register in bytes."""
        return get_type_size(vreg.type_info)


class StackAllocator:
    """
    Allocates stack slots for spilled virtual registers.

    Stack grows downward on 65816. Stack layout:
    - Return address (2 or 3 bytes)
    - Saved registers (if any)
    - Local variables / spilled registers
    """

    def __init__(self, initial_offset: int = 0):
        """
        Initialize stack allocator.

        Args:
            initial_offset: Starting offset (accounts for return address, etc.)
        """
        self.current_offset = initial_offset
        self.allocated: Dict[int, int] = {}  # vreg.id → stack offset

    def allocate(self, vreg: VirtualRegister) -> int:
        """
        Allocate stack slot for virtual register.

        Args:
            vreg: Virtual register to allocate

        Returns:
            Stack offset for the register
        """
        # Check if already allocated
        if vreg.id in self.allocated:
            return self.allocated[vreg.id]

        # Allocate new slot
        offset = self.current_offset
        vreg_size = self._get_vreg_size(vreg)
        self.current_offset += vreg_size
        self.allocated[vreg.id] = offset

        return offset

    def get_frame_size(self) -> int:
        """Get total stack frame size."""
        return self.current_offset

    def _get_vreg_size(self, vreg: VirtualRegister) -> int:
        """Get size of virtual register in bytes."""
        return get_type_size(vreg.type_info)


class RegisterAllocator:
    """
    Main register allocator for virtual registers.

    Allocation priority:
    1. Hardware registers (A, X, Y) - already handled by MIR
    2. Scratch registers (designated zero-page locations)
    3. Stack slots (spill when scratch exhausted, with reuse optimization)
    """

    def __init__(self,
                 scratch_pool: Optional[ScratchRegisterPool] = None,
                 mir_func: Optional[MIRFunction] = None):
        """
        Initialize register allocator.

        Args:
            scratch_pool: Pool of scratch registers (or None for empty pool)
            mir_func: MIR function for liveness analysis (enables slot reuse)
        """
        self.scratch_pool = scratch_pool or ScratchRegisterPool()
        self.mir_func = mir_func
        self.slot_allocator: Optional[StackSlotAllocator] = None
        self.slot_allocation: Optional[SlotAllocation] = None
        self.allocations: Dict[int, PhysicalLocation] = {}  # vreg.id → PhysicalLocation

        # Base offset for stack slots (starts after scratch registers)
        self.stack_base_offset = 0x16  # Start after common scratch locations

    def allocate_vreg(self, vreg: VirtualRegister) -> PhysicalLocation:
        """
        Allocate physical location for virtual register.

        Strategy:
        1. Try scratch register first
        2. Spill to stack if no scratch available (using slot reuse if available)

        Args:
            vreg: Virtual register to allocate

        Returns:
            PhysicalLocation for the register
        """
        # Check if already allocated
        if vreg.id in self.allocations:
            return self.allocations[vreg.id]

        # Try scratch register first
        scratch = self.scratch_pool.allocate(vreg)
        if scratch:
            location = PhysicalLocation(
                kind=LocationKind.SCRATCH,
                scratch_addr=scratch.address,
                size=self._get_vreg_size(vreg)
            )
            self.allocations[vreg.id] = location
            return location

        # Spill to stack using slot allocation if available
        if self.slot_allocation:
            # Get slot number from slot allocation
            slot_num = self.slot_allocation.register_to_slot.get(vreg)
            if slot_num is not None:
                # Convert slot number to stack offset
                stack_offset = self.stack_base_offset + slot_num
                location = PhysicalLocation(
                    kind=LocationKind.STACK,
                    stack_offset=stack_offset,
                    size=self._get_vreg_size(vreg)
                )
                self.allocations[vreg.id] = location
                return location

        # Fallback: sequential allocation (shouldn't happen if allocate_all called)
        # This is for backwards compatibility if slot allocation fails
        stack_offset = self.stack_base_offset + len([
            loc for loc in self.allocations.values()
            if loc.kind == LocationKind.STACK
        ])
        location = PhysicalLocation(
            kind=LocationKind.STACK,
            stack_offset=stack_offset,
            size=self._get_vreg_size(vreg)
        )
        self.allocations[vreg.id] = location
        return location

    def get_location(self, vreg: VirtualRegister) -> PhysicalLocation:
        """
        Get physical location for virtual register.

        Args:
            vreg: Virtual register

        Returns:
            PhysicalLocation (allocates if not already allocated)
        """
        if vreg.id not in self.allocations:
            return self.allocate_vreg(vreg)
        return self.allocations[vreg.id]

    def get_hw_location(self, hw_reg: HardwareRegister) -> PhysicalLocation:
        """
        Get physical location for hardware register.

        Args:
            hw_reg: Hardware register (A, X, Y, etc.)

        Returns:
            PhysicalLocation representing the hardware register
        """
        # Hardware registers are always in their own location
        return PhysicalLocation(
            kind=LocationKind.HARDWARE,
            hw_register=hw_reg.name,
            size=2 if hw_reg.name in ('A', 'X', 'Y') else 1
        )

    def allocate_all(self, vregs: List[VirtualRegister]):
        """
        Allocate all virtual registers at once.

        Performs slot reuse optimization if MIR function is available.

        Args:
            vregs: List of virtual registers to allocate
        """
        # Run slot allocation with liveness analysis if MIR function available
        if self.mir_func:
            self.slot_allocator = StackSlotAllocator(self.mir_func)
            self.slot_allocation = self.slot_allocator.allocate()

            # Print statistics if any slots were saved
            if self.slot_allocation.slots_saved > 0:
                print(f"Stack slot reuse: {self.slot_allocation.slots_saved} slot(s) saved "
                      f"({self.slot_allocation.total_slots}/{self.slot_allocation.variables_count} slots used)")

        # Allocate each virtual register
        for vreg in vregs:
            self.allocate_vreg(vreg)

    def get_stack_frame_size(self) -> int:
        """Get total stack frame size for spilled registers."""
        if self.slot_allocation:
            return self.slot_allocation.total_slots
        # Fallback: count stack allocations
        return len([
            loc for loc in self.allocations.values()
            if loc.kind == LocationKind.STACK
        ])

    def _get_vreg_size(self, vreg: VirtualRegister) -> int:
        """Get size of virtual register in bytes."""
        return get_type_size(vreg.type_info)
