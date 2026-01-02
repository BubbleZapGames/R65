## Code Generation Phase 3: Register Allocation

**Status**: ✅ COMPLETE
**Date**: 2026-01-01
**LOC**: ~400 lines

## Overview

Phase 3 implements register allocation for MIR virtual registers. This phase maps unlimited virtual registers from MIR to physical locations (scratch registers or stack slots) following the R65 philosophy of **explicit programmer control** and **predictable code generation**.

## Design Philosophy

R65's register allocation is **NOT** an optimizing allocator. It follows these principles:

1. **Explicit Control**: Programmer specifies register usage via register aliasing
2. **Predictable Output**: Same source always generates same assembly
3. **Minimal Automation**: Compiler uses designated scratch space only
4. **Trust the Programmer**: No clever heuristics or interference

The compiler is a **translator**, not an optimizer.

## Allocation Strategy

### Priority Order

1. **Hardware Registers** (A, X, Y) - handled by MIR register aliasing
2. **Memory Locations** - explicit `#[zeropage]`, `#[ram]` attributes
3. **Scratch Registers** - designated zero-page locations
4. **Stack** - last resort when scratch exhausted

### Virtual Registers

MIR uses **unlimited virtual registers** (`%0, %1, %2, ...`) during lowering. These must be mapped to physical locations for code generation.

## Components

### 1. PhysicalLocation

**Purpose**: Represents where a value lives at runtime.

```python
@dataclass
class PhysicalLocation:
    kind: LocationKind  # HARDWARE, SCRATCH, STACK, MEMORY

    # Location-specific fields
    hw_register: Optional[str]      # For HARDWARE: "A", "X", "Y"
    scratch_addr: Optional[int]     # For SCRATCH: $16, $17, etc.
    stack_offset: Optional[int]     # For STACK: [SP+0], [SP+1], etc.
    memory_addr: Optional[int]      # For MEMORY: $7E0000, etc.

    size: int  # Size in bytes
```

**String Representation**:
- Hardware: `A`, `X`, `Y`
- Scratch: `$0016`, `$0017`
- Stack: `[SP+0]`, `[SP+1]`
- Memory: `$7E0000`

### 2. ScratchRegisterPool

**Purpose**: Manages pool of scratch registers for temporary allocation.

**Scratch Registers** are designated zero-page locations marked with:
```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

#[zeropage(0x18, register)]
static mut SCRATCH2: u16;  // Takes 0x18-0x19
```

**Key Features**:
- Track available scratch locations
- Allocate to virtual registers
- Free and reuse when registers die
- Size-aware allocation (1-byte vs 2-byte)

**Methods**:
```python
def add_scratch(address, size, name):
    """Add scratch register to pool"""

def allocate(vreg) -> Optional[ScratchRegister]:
    """Allocate scratch for virtual register"""

def free(vreg):
    """Free scratch for reuse"""
```

**Example**:
```
Scratch pool:
  $0016: SCRATCH0 (1 byte)
  $0017: SCRATCH1 (1 byte)
  $0018: SCRATCH2 (2 bytes)

Allocations:
  %0 (u8)  → $0016 (SCRATCH0)
  %1 (u8)  → $0017 (SCRATCH1)
  %2 (u16) → $0018 (SCRATCH2)
  %3 (u8)  → FAILED (pool exhausted)
```

### 3. StackAllocator

**Purpose**: Allocates stack slots for spilled virtual registers.

**Stack Layout** (65816 stack grows downward):
```
Low Address
├─ Return address (2 or 3 bytes)
├─ Saved registers (if any)
├─ Local variables / spilled registers
└─ (grows downward)
```

**Key Features**:
- Tracks current stack offset
- Allocates slots for spilled registers
- Calculates total frame size
- Size-aware (1-byte vs 2-byte)

**Methods**:
```python
def allocate(vreg) -> int:
    """Allocate stack slot, returns offset"""

def get_frame_size() -> int:
    """Total stack frame size"""
```

**Example**:
```
Stack allocations:
  %2 (u8)  → [SP+0] (1 byte)
  %3 (u16) → [SP+1] (2 bytes, uses SP+1 and SP+2)

Frame size: 3 bytes
```

### 4. RegisterAllocator (Main Class)

**Purpose**: Coordinates register allocation strategy.

**Allocation Algorithm**:
```python
def allocate_vreg(vreg):
    1. Check if already allocated
    2. Try scratch register first
    3. Spill to stack if no scratch available
    4. Return PhysicalLocation
```

**Key Methods**:
```python
def allocate_vreg(vreg) -> PhysicalLocation:
    """Allocate physical location for virtual register"""

def get_location(vreg) -> PhysicalLocation:
    """Get location (allocates if needed)"""

def get_hw_location(hw_reg) -> PhysicalLocation:
    """Get location for hardware register"""

def allocate_all(vregs):
    """Bulk allocation"""

def get_stack_frame_size() -> int:
    """Total stack frame size"""
```

## Allocation Examples

### Example 1: Simple Allocation

**Input**: 2 virtual registers, 2 scratch slots

```python
pool = ScratchRegisterPool()
pool.add_scratch(0x16, 1, "SCRATCH0")
pool.add_scratch(0x17, 1, "SCRATCH1")

allocator = RegisterAllocator(scratch_pool=pool)

vreg0 = VirtualRegister(id=0, type_info=u8)
vreg1 = VirtualRegister(id=1, type_info=u8)

loc0 = allocator.allocate_vreg(vreg0)  # → $0016 (SCRATCH0)
loc1 = allocator.allocate_vreg(vreg1)  # → $0017 (SCRATCH1)
```

**Result**: Both allocated to scratch registers.

### Example 2: Spilling to Stack

**Input**: 4 virtual registers, 2 scratch slots

```python
pool = ScratchRegisterPool()
pool.add_scratch(0x16, 1, "SCRATCH0")
pool.add_scratch(0x17, 1, "SCRATCH1")

allocator = RegisterAllocator(scratch_pool=pool)

vregs = [
    VirtualRegister(id=0, type_info=u8),  # → $0016 (scratch)
    VirtualRegister(id=1, type_info=u8),  # → $0017 (scratch)
    VirtualRegister(id=2, type_info=u8),  # → [SP+0] (spill)
    VirtualRegister(id=3, type_info=u16), # → [SP+1] (spill, 2 bytes)
]

for vreg in vregs:
    location = allocator.allocate_vreg(vreg)

frame_size = allocator.get_stack_frame_size()  # 3 bytes
```

**Result**:
- First 2 vregs: scratch registers
- Last 2 vregs: stack slots
- Stack frame: 3 bytes

### Example 3: Size-Aware Allocation

**Input**: Mix of u8 and u16, sized scratch slots

```python
pool = ScratchRegisterPool()
pool.add_scratch(0x16, 1, "SCRATCH0")  # 1-byte
pool.add_scratch(0x18, 2, "SCRATCH1")  # 2-byte

vreg_u8  = VirtualRegister(id=0, type_info=u8)   # → $0016 (1-byte scratch)
vreg_u16 = VirtualRegister(id=1, type_info=u16)  # → $0018 (2-byte scratch)
```

**Result**: Each vreg gets scratch of appropriate size.

## Type Size Calculation

The allocator determines size based on type:

```python
def _get_vreg_size(vreg) -> int:
    type_name = vreg.type_info.name

    if type_name in ('u8', 'i8', 'bool'):
        return 1
    elif type_name in ('u16', 'i16'):
        return 2
    else:
        return 1  # Default
```

## Test Coverage

### Test File: `test_register_alloc.py`

**Test 1: Scratch Pool Management**
- Add scratch registers to pool
- Allocate virtual registers to scratches
- Verify pool exhaustion behavior
- Test freeing and reusing scratches
- ✅ PASSED

**Test 2: Register Allocator (Scratch + Stack)**
- Limited scratch pool (2 slots)
- Allocate 4 virtual registers
- Verify first 2 use scratch
- Verify last 2 spill to stack
- Check stack frame size
- ✅ PASSED

**Test 3: Virtual Register Size Detection**
- Test u8 → 1-byte scratch
- Test u16 → 2-byte scratch
- Verify size matching
- ✅ PASSED

**Test 4: Bulk Allocation**
- Allocate 5 vregs with 1 scratch slot
- Verify 1 scratch, 4 stack
- ✅ PASSED

**All Tests**: ✅ PASSED

### Test Results:
```
Scratch Pool Test: ✅ PASSED
Register Allocator Test: ✅ PASSED
Size Detection Test: ✅ PASSED
Bulk Allocation Test: ✅ PASSED

🎉 All tests passed!
```

## Integration Notes

### For Code Generation (Phase 4+)

The RegisterAllocator will be used during instruction selection:

```python
# During code generation
allocator = RegisterAllocator(scratch_pool=pool)

# For each MIR instruction
for instr in block.instructions:
    if isinstance(instr, Load):
        dest_loc = allocator.get_location(instr.dest)  # PhysicalLocation

        if dest_loc.kind == LocationKind.SCRATCH:
            # Generate: LDA source; STA ${dest_loc.scratch_addr}
            pass
        elif dest_loc.kind == LocationKind.STACK:
            # Generate: LDA source; PHA (or stack offset access)
            pass
```

### Scratch Pool Setup

Scratch pools are configured per-function or globally:

```rust
// In R65 code (future feature):
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

#[zeropage(0x17, register)]
static mut SCRATCH1: u8;
```

For now, scratch pools are created programmatically during code generation.

## Key Design Decisions

1. **Simple Linear Allocation**: No live range analysis in v1 - each vreg gets one location for its entire lifetime. Simple but predictable.

2. **Scratch Before Stack**: Always try scratch first, spill to stack only when necessary. Matches programmer's mental model.

3. **Size-Aware Allocation**: Match vreg size to scratch size. A u16 vreg needs a 2-byte scratch.

4. **No Spill Heuristics**: First-come, first-served allocation. Deterministic and predictable.

5. **Reusable Scratches**: Freeing allows reuse, supporting future optimization (live range analysis).

## Known Limitations

1. **No Live Range Analysis**: Each vreg allocated once for entire lifetime. Future enhancement: analyze liveness and reuse scratches.

2. **No Spill Optimization**: Simple FIFO allocation. Future: prioritize frequently-used vregs for scratch.

3. **Fixed Scratch Pool**: Configured statically. Future: dynamic scratch pool based on function needs.

4. **No Register Coloring**: Future enhancement for more efficient scratch usage.

## Files Created/Modified

**Created**:
- `r65/compiler/codegen/register_alloc.py` (~380 lines)
- `test_register_alloc.py` (~360 lines)

**Modified**:
- `r65/compiler/codegen/__init__.py` (added exports)

**Total**: ~400 LOC for Phase 3 implementation

## Next Steps

Phase 4 will implement **Instruction Selection**:
- Convert MIR instructions to 65816 mnemonics
- Use RegisterAllocator to resolve virtual register locations
- Emit proper addressing modes
- Handle 8-bit vs 16-bit mode transitions

---

**Phase 3 Status**: ✅ COMPLETE
**All Tests**: ✅ PASSING
**Ready for Phase 4**: ✅ YES
