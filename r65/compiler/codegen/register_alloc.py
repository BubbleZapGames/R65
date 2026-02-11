"""
Register allocation: maps virtual registers to physical locations.

Maps MIR virtual registers to:
1. Hardware registers (A, X, Y) - when available and beneficial
2. Scratch registers (designated zero-page locations)
3. Stack slots (when scratch pool exhausted)

Variables live across calls always spill to stack (scratches are not
callee-saved, so any callee could clobber them).
"""

from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from r65.compiler.mir.nodes import VirtualRegister, HardwareRegister, MIRFunction
from r65.compiler.codegen.slot_allocator import StackSlotAllocator, SlotAllocation, PreassignedSlot
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.errors import MemoryAllocationError

if TYPE_CHECKING:
    from r65.compiler.mir.liveness import InstructionLivenessAnalyzer


class LocationKind(Enum):
    """Type of physical location for a virtual register."""
    HARDWARE = "hardware"      # Hardware register (A, X, Y)
    SCRATCH = "scratch"        # Scratch register (zero-page)
    STACK = "stack"            # Stack slot
    MEMORY = "memory"          # Static memory location
    IMMEDIATE = "immediate"    # Immediate value (constant)
    RETURN_SINKABLE = "return_sinkable"  # Load deferred to return site


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

    # For MEMORY: ROM label (used for #[rom] data accessed directly from ROM)
    memory_label: Optional[str] = None

    # For IMMEDIATE: constant value
    immediate_value: Optional[int] = None

    # For indexed addressing: 'X' or 'Y'
    index_register: Optional[str] = None

    # For RETURN_SINKABLE: the MIR MemoryLocation to load from at return site
    source_location: Optional[Any] = None

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


@dataclass
class HardwareRegAllocation:
    """
    Tracks allocation state of a hardware register.

    Hardware registers (A, X, Y) can hold virtual register values
    temporarily. This tracks which vreg is currently allocated to each.
    """
    name: str                                  # Register name ('A', 'X', 'Y')
    size: int                                  # Size in bytes (1 for A in m8, 2 for A in m16 or X/Y)
    allocated_vreg: Optional[VirtualRegister] = None  # Currently allocated vreg, if any
    is_bound: bool = False                     # True if vreg has explicit binding (@ A)


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
        # Track used address ranges: list of (start, end, name) for overlap detection
        self._used_ranges: List[Tuple[int, int, str]] = []

    def add_scratch(self, address: int, size: int, name: str):
        """
        Add a scratch register to the pool.

        Validates that the new scratch register does not overlap with any
        previously registered scratch registers.

        Args:
            address: Zero-page address
            size: Size in bytes
            name: Scratch register name

        Raises:
            MemoryAllocationError: If scratch register overlaps with existing one
        """
        new_start = address
        new_end = address + size

        # Check for overlaps with existing scratch registers
        for existing_start, existing_end, existing_name in self._used_ranges:
            # Overlap occurs if: new_start < existing_end AND existing_start < new_end
            if new_start < existing_end and existing_start < new_end:
                raise MemoryAllocationError(
                    f"scratch register '{name}' at ${address:04X}-${new_end-1:04X} "
                    f"overlaps with existing scratch register '{existing_name}' "
                    f"at ${existing_start:04X}-${existing_end-1:04X}"
                )

        # Register the address range
        self._used_ranges.append((new_start, new_end, name))

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

    def reset(self):
        """
        Reset pool for a new function.

        Marks all scratches as free and clears allocations.
        Called before each function to allow scratch reuse across functions.
        """
        for scratch in self.scratches:
            scratch.is_free = True
        self.allocated.clear()

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
    1. Hardware registers (A, X, Y) - for explicitly bound variables or temporaries
    2. Scratch registers (designated zero-page locations, call-graph aware)
    3. Stack slots (spill when scratch exhausted, with reuse optimization)

    Call-graph aware: Variables that live across calls cannot use scratches
    that callees might use.
    """

    def __init__(self,
                 scratch_pool: Optional[ScratchRegisterPool] = None,
                 mir_func: Optional[MIRFunction] = None,
                 prologue_stack_bytes: int = 0,
                 instr_liveness: Optional['InstructionLivenessAnalyzer'] = None):
        """
        Initialize register allocator.

        Args:
            scratch_pool: Pool of scratch registers (or None for empty pool)
            mir_func: MIR function for liveness analysis (enables slot reuse)
            prologue_stack_bytes: Bytes pushed by prologue (affects stack param offsets)
            instr_liveness: Instruction-level liveness analyzer (for precise liveness)
        """
        self.scratch_pool = scratch_pool or ScratchRegisterPool()
        self.mir_func = mir_func
        self.prologue_stack_bytes = prologue_stack_bytes
        self.instr_liveness = instr_liveness
        self.slot_allocator: Optional[StackSlotAllocator] = None
        self.slot_allocation: Optional[SlotAllocation] = None
        self.allocations: Dict[int, PhysicalLocation] = {}  # vreg.id → PhysicalLocation

        # Hardware register tracking
        self.hw_allocs: Dict[str, HardwareRegAllocation] = {
            'A': HardwareRegAllocation('A', size=1),  # Size depends on mode
            'B': HardwareRegAllocation('B', size=1),  # High byte of accumulator (m8 only)
            'X': HardwareRegAllocation('X', size=2),  # Always 16-bit
            'Y': HardwareRegAllocation('Y', size=2),  # Always 16-bit
        }

        # Track which vregs are bound to hw registers (from @ bindings)
        self.hw_bindings: Dict[int, str] = {}  # vreg.id -> hw reg name

        # Frame allocation info (set by FunctionCodeGenerator after allocation)
        self.frame_size: int = 0
        self.has_frame_allocation: bool = False

        # Calculate base offset for stack temporaries
        # After frame allocation, locals start at S+1
        # prologue_stack_bytes accounts for return address and preserved registers
        # Entry functions have no return address or preserved registers, so offset is just 1
        if mir_func and mir_func.is_entry:
            self.stack_base_offset = 1  # Entry functions have no prologue overhead
        else:
            self.stack_base_offset = self.prologue_stack_bytes + 1

    def _build_preassigned_params(self) -> List[PreassignedSlot]:
        """
        Build preassigned slots from MIR function's stack parameters.

        Returns:
            List of PreassignedSlot for each stack parameter
        """
        preassigned = []

        if not self.mir_func or not self.mir_func.stack_param_offsets:
            return preassigned

        for param_idx, base_offset in self.mir_func.stack_param_offsets.items():
            vreg = self.mir_func.param_to_vreg.get(param_idx)
            if not vreg:
                continue

            preassigned.append(PreassignedSlot(
                vreg=vreg,
                base_offset=base_offset,
                size=self._get_vreg_size(vreg)
            ))

        return preassigned

    def allocate_vreg(self, vreg: VirtualRegister) -> PhysicalLocation:
        """
        Allocate physical location for virtual register.

        Strategy:
        1. Check for explicit hardware register binding
        2. Check for hw-coalesceable (can stay in hardware register)
        3. Try register hint (for loop variables assigned to X/Y)
        4. Try scratch register (respecting call graph constraints)
        5. Spill to stack if no scratch available (using slot reuse if available)

        Args:
            vreg: Virtual register to allocate

        Returns:
            PhysicalLocation for the register
        """
        # Check if already allocated
        if vreg.id in self.allocations:
            return self.allocations[vreg.id]

        # Check if this vreg has an explicit hw binding
        if vreg.id in self.hw_bindings:
            hw_reg = self.hw_bindings[vreg.id]
            location = PhysicalLocation(
                kind=LocationKind.HARDWARE,
                hw_register=hw_reg,
                size=self._get_vreg_size(vreg)
            )
            self.allocations[vreg.id] = location
            self.hw_allocs[hw_reg].allocated_vreg = vreg
            self.hw_allocs[hw_reg].is_bound = True
            return location

        # Check if this vreg is hw-coalesceable (can stay in hardware register)
        if self.slot_allocation and self.slot_allocation.hw_coalesceable:
            hw_reg = self.slot_allocation.hw_coalesceable.get(vreg)
            if hw_reg:
                location = PhysicalLocation(
                    kind=LocationKind.HARDWARE,
                    hw_register=hw_reg,
                    size=self._get_vreg_size(vreg)
                )
                self.allocations[vreg.id] = location
                # Track the allocation so spill logic can detect it
                self.hw_allocs[hw_reg].allocated_vreg = vreg
                self.hw_allocs[hw_reg].is_bound = False
                return location

        # Check if this vreg is return-sinkable (load deferred to return site)
        if self.slot_allocation and self.slot_allocation.return_sinkable:
            mem_source = self.slot_allocation.return_sinkable.get(vreg)
            if mem_source is not None:
                location = PhysicalLocation(
                    kind=LocationKind.RETURN_SINKABLE,
                    source_location=mem_source,
                    size=self._get_vreg_size(vreg)
                )
                self.allocations[vreg.id] = location
                return location

        # Check for register hint (loop variables)
        if vreg.register_hint in ('X', 'Y'):
            hw_reg = vreg.register_hint
            hw_alloc = self.hw_allocs.get(hw_reg)
            if hw_alloc and hw_alloc.allocated_vreg is None:
                # Register is free - allocate vreg to it
                location = PhysicalLocation(
                    kind=LocationKind.HARDWARE,
                    hw_register=hw_reg,
                    size=self._get_vreg_size(vreg)
                )
                self.allocations[vreg.id] = location
                hw_alloc.allocated_vreg = vreg
                hw_alloc.is_bound = False  # Not explicitly bound, just hinted
                return location
            # Hint couldn't be satisfied (register occupied) - fall through

        # Determine if this vreg lives across any call
        live_across_call = False
        if self.instr_liveness:
            live_across_call = self.instr_liveness.is_live_across_any_call(vreg)

        # Try scratch register (not allowed if live across a call)
        if not live_across_call:
            scratch_loc = self._try_scratch(vreg)
            if scratch_loc:
                return scratch_loc

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

    def _try_scratch(self, vreg: VirtualRegister) -> Optional[PhysicalLocation]:
        """
        Try to allocate a scratch register for a vreg.

        Only called for vregs that are NOT live across any call (scratches
        are not callee-saved so any call could clobber them).

        Args:
            vreg: Virtual register to allocate

        Returns:
            PhysicalLocation if scratch allocated, None otherwise
        """
        vreg_size = self._get_vreg_size(vreg)

        for scratch in self.scratch_pool.scratches:
            if not scratch.is_free:
                continue
            if scratch.size < vreg_size:
                continue

            # Allocate this scratch
            scratch.is_free = False
            self.scratch_pool.allocated[vreg.id] = vreg

            location = PhysicalLocation(
                kind=LocationKind.SCRATCH,
                scratch_addr=scratch.address,
                size=vreg_size
            )
            self.allocations[vreg.id] = location
            return location

        return None

    def bind_vreg_to_hw(self, vreg: VirtualRegister, hw_reg: str):
        """
        Bind a virtual register to a hardware register.

        Used for explicit @ bindings like `let x @ A = ...`

        Args:
            vreg: Virtual register
            hw_reg: Hardware register name ('A', 'X', or 'Y')
        """
        self.hw_bindings[vreg.id] = hw_reg

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

    def get_hw_alloc(self, hw_reg: str) -> HardwareRegAllocation:
        """
        Get hardware register allocation info.

        Args:
            hw_reg: Hardware register name ('A', 'X', 'Y')

        Returns:
            HardwareRegAllocation for the register
        """
        return self.hw_allocs.get(hw_reg, HardwareRegAllocation(hw_reg, size=1))

    def free_hw_register(self, hw_reg: str):
        """
        Free a hardware register (mark as available for reuse).

        Called when a bound variable's liveness ends.

        Args:
            hw_reg: Hardware register name to free
        """
        if hw_reg in self.hw_allocs:
            alloc = self.hw_allocs[hw_reg]
            if alloc.allocated_vreg:
                vreg_id = alloc.allocated_vreg.id
                if vreg_id in self.hw_bindings:
                    del self.hw_bindings[vreg_id]
            alloc.allocated_vreg = None
            alloc.is_bound = False

    def is_vreg_in_hw_register(self, vreg: VirtualRegister) -> Optional[str]:
        """
        Check if a vreg is currently allocated to a hardware register.

        Args:
            vreg: Virtual register to check

        Returns:
            Hardware register name if allocated, None otherwise
        """
        if vreg.id in self.allocations:
            loc = self.allocations[vreg.id]
            if loc.kind == LocationKind.HARDWARE:
                return loc.hw_register
        return None

    def allocate_all(self, vregs: List[VirtualRegister]):
        """
        Allocate all virtual registers at once.

        Uses unified slot allocation that handles both stack parameters and
        local variables in a single pass. Parameters get their final offsets
        computed automatically (no post-hoc adjustment needed).

        Args:
            vregs: List of virtual registers to allocate
        """
        # Run unified slot allocation if MIR function available
        if self.mir_func:
            # Build preassigned slots for stack parameters
            preassigned = self._build_preassigned_params()

            # Collect vregs already allocated externally (e.g. scratch params)
            # so the slot allocator doesn't reserve stack frame space for them
            pre_allocated_vregs = set()
            for vreg_id, loc in self.allocations.items():
                if loc.kind == LocationKind.SCRATCH:
                    # Find the VirtualRegister object for this id
                    for v in vregs:
                        if v.id == vreg_id:
                            pre_allocated_vregs.add(v)
                            break

            # Create unified allocator that handles params + locals together
            self.slot_allocator = StackSlotAllocator(
                self.mir_func,
                preassigned=preassigned,
                prologue_stack_bytes=self.prologue_stack_bytes,
                instr_liveness=self.instr_liveness,
                pre_allocated_vregs=pre_allocated_vregs
            )
            self.slot_allocation = self.slot_allocator.allocate()

            # Pre-allocate param vregs with their final computed offsets
            for vreg, final_offset in self.slot_allocation.param_offsets.items():
                location = PhysicalLocation(
                    kind=LocationKind.STACK,
                    stack_offset=final_offset,
                    size=self.slot_allocation.param_sizes.get(vreg, 1)
                )
                self.allocations[vreg.id] = location

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
