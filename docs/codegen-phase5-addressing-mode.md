# Code Generation Phase 5: Addressing Mode Selection

**Status**: ✅ COMPLETE
**Date**: 2026-01-01
**LOC**: ~350 lines

## Overview

Phase 5 implements intelligent addressing mode selection for the 65816 processor. This phase chooses the optimal addressing mode based on operand type, address range, and instruction requirements - prioritizing smaller code size and faster execution.

## Purpose

The AddressingModeSelector optimizes memory access by selecting the most efficient addressing mode:

- **Direct Page** (zero-page): Fastest, smallest (2 bytes, 3 cycles)
- **Absolute**: Standard (3 bytes, 4 cycles)
- **Long**: 24-bit addressing (4 bytes, 5 cycles)
- **Indexed**: With ,X or ,Y registers
- **Indirect**: Pointer dereference through zero-page

## 65816 Addressing Modes

### Supported Modes

| Mode | Example | Size | Cycles* | Use Case |
|------|---------|------|---------|----------|
| Immediate | `LDA #$42` | 2 bytes | 2 | Constants |
| Accumulator | `INC A` | 1 byte | 2 | A register ops |
| Direct Page | `LDA $20` | 2 bytes | 3 | Zero-page ($00-$FF) |
| Absolute | `LDA $2000` | 3 bytes | 4 | Standard memory |
| Long | `LDA $7E0000` | 4 bytes | 5 | 24-bit addresses |
| Direct Page,X | `LDA $20,X` | 2 bytes | 4 | Zero-page arrays |
| Absolute,X | `LDA $2000,X` | 3 bytes | 4-5** | Standard arrays |
| Absolute,Y | `LDA $2000,Y` | 3 bytes | 4-5** | Standard arrays |
| Indirect | `LDA ($42)` | 2 bytes | 5 | Pointers |
| Indirect,Y | `LDA ($42),Y` | 2 bytes | 5-6** | Pointer arrays |
| Long,X | `LDA $7E0000,X` | 4 bytes | 5 | Large arrays |

\* Approximate cycles (varies with processor mode)
\*\* +1 cycle on page boundary crossing

## AddressingModeSelector Class

### Core Selection Method

```python
def select_for_location(location, index_register=None, is_indirect=False):
    """
    Select optimal addressing mode for physical location.

    Returns: (AddressingMode, operand_string)
    """
```

**Selection Logic**:
1. Check if hardware register (accumulator mode)
2. Get effective address from location
3. Determine address range (direct page / absolute / long)
4. Apply modifiers (indexing, indirection)
5. Return mode enum and formatted operand

### Address Range Detection

```python
is_direct_page = (0 <= addr <= 0xFF)      # Zero-page - use direct page
is_absolute = (0x100 <= addr <= 0xFFFF)   # 16-bit - use absolute
is_long = (addr > 0xFFFF)                 # 24-bit - use long
```

**Optimization**: Direct page mode is **fastest** - always prefer for $00-$FF range.

## Selection Examples

### Example 1: Automatic Direct Page Selection

**Input**: Address $20

**Selection**:
```python
loc = PhysicalLocation(kind=SCRATCH, scratch_addr=0x20)
mode, operand = selector.select_for_location(loc)
# Result: (AddressingMode.DIRECT_PAGE, "$20")
```

**Generated**: `LDA $20` (2 bytes, 3 cycles)

### Example 2: Absolute Mode for High Addresses

**Input**: Address $2000

**Selection**:
```python
loc = PhysicalLocation(kind=MEMORY, memory_addr=0x2000)
mode, operand = selector.select_for_location(loc)
# Result: (AddressingMode.ABSOLUTE, "$2000")
```

**Generated**: `LDA $2000` (3 bytes, 4 cycles)

### Example 3: Indexed Addressing

**Input**: Address $20 with X register

**Selection**:
```python
loc = PhysicalLocation(kind=SCRATCH, scratch_addr=0x20)
mode, operand = selector.select_for_location(loc, index_register='X')
# Result: (AddressingMode.DIRECT_PAGE_X, "$20,X")
```

**Generated**: `LDA $20,X` (2 bytes, 4 cycles)

### Example 4: Indirect Addressing

**Input**: Zero-page pointer $42

**Selection**:
```python
loc = PhysicalLocation(kind=SCRATCH, scratch_addr=0x42)
mode, operand = selector.select_for_location(loc, is_indirect=True)
# Result: (AddressingMode.INDIRECT, "($42)")
```

**Generated**: `LDA ($42)` (2 bytes, 5 cycles)

### Example 5: Indirect Indexed

**Input**: Zero-page pointer $50 with Y index

**Selection**:
```python
loc = PhysicalLocation(kind=SCRATCH, scratch_addr=0x50)
mode, operand = selector.select_for_location(loc, index_register='Y', is_indirect=True)
# Result: (AddressingMode.INDIRECT_Y, "($50),Y")
```

**Generated**: `LDA ($50),Y` (2 bytes, 5-6 cycles)

### Example 6: Long Addressing (24-bit)

**Input**: SNES RAM address $7E0000

**Selection**:
```python
loc = PhysicalLocation(kind=MEMORY, memory_addr=0x7E0000)
mode, operand = selector.select_for_location(loc)
# Result: (AddressingMode.LONG, "$7E0000")
```

**Generated**: `LDA $7E0000` (4 bytes, 5 cycles)

## Immediate Mode Selection

### 8-bit Immediate

```python
mode, operand = selector.select_immediate(0x42, is_16bit=False)
# Result: (AddressingMode.IMMEDIATE, "#$42")
```

**Generated**: `LDA #$42`

### 16-bit Immediate

```python
mode, operand = selector.select_immediate(0x1234, is_16bit=True)
# Result: (AddressingMode.IMMEDIATE, "#$1234")
```

**Generated**: `LDA #$1234`

## Optimization Helpers

### Direct Page Detection

```python
def can_use_direct_page(addr) -> bool:
    """Check if address can use fast direct page mode."""
    return 0 <= addr <= 0xFF
```

**Usage**: Verify if variable can benefit from zero-page allocation.

### STZ Optimization

```python
def should_use_stz(value) -> bool:
    """Check if STZ (store zero) can be used."""
    return value == 0
```

**Benefit**: `STZ $20` is more efficient than `LDA #0; STA $20`

**Example**:
```python
if selector.should_use_stz(value):
    emitter.emit_instruction("STZ", "$20")
else:
    emitter.emit_instruction("LDA", f"#{value:02X}")
    emitter.emit_instruction("STA", "$20")
```

### Performance Estimates

#### Cycle Count Estimation

```python
def get_cycle_count(mode, instruction) -> int:
    """Estimate execution cycles for addressing mode."""
```

**Estimates**:
- Immediate: 2 cycles
- Direct Page: 3 cycles
- Absolute: 4 cycles
- Indexed: 4-5 cycles (page crossing adds 1)
- Indirect: 5-6 cycles

#### Instruction Size Calculation

```python
def get_byte_size(mode, instruction) -> int:
    """Calculate instruction size in bytes."""
```

**Sizes**:
- Accumulator: 1 byte (opcode only)
- Immediate: 2 bytes (opcode + immediate)
- Direct Page: 2 bytes (opcode + 8-bit address)
- Absolute: 3 bytes (opcode + 16-bit address)
- Long: 4 bytes (opcode + 24-bit address)

## Format Helpers

### Operand Formatting

```python
def format_operand(location, index_register=None, is_indirect=False) -> str:
    """Format location as assembly operand string."""
```

**Convenience wrapper** - returns just the operand string without mode enum.

**Example**:
```python
operand = selector.format_operand(loc)
emitter.emit_instruction("LDA", operand)
```

### Immediate Formatting

```python
def format_immediate(value, is_16bit=False) -> str:
    """Format immediate value as operand."""
```

**Example**:
```python
operand = selector.format_immediate(0x42)  # "#$42"
emitter.emit_instruction("LDA", operand)
```

## Test Coverage

### Test File: `test_addressing_mode.py`

**Test 1: Direct Page vs Absolute**
- Zero-page addresses ($00-$FF) → Direct Page
- Absolute addresses ($100-$FFFF) → Absolute
- Long addresses ($10000+) → Long
- ✅ PASSED

**Test 2: Indexed Addressing**
- Direct Page,X: `$20,X`
- Direct Page,Y: `$30,Y`
- Absolute,X: `$2000,X`
- Absolute,Y: `$3000,Y`
- Long,X: `$7E0000,X`
- ✅ PASSED

**Test 3: Indirect Addressing**
- Simple indirect: `($42)`
- Indirect indexed: `($50),Y`
- ✅ PASSED

**Test 4: Immediate Mode**
- 8-bit: `#$42`
- 16-bit: `#$1234`
- ✅ PASSED

**Test 5: Optimization Helpers**
- `can_use_direct_page()`
- `should_use_stz()`
- Cycle count estimation
- Byte size calculation
- ✅ PASSED

**Test 6: Format Helpers**
- `format_operand()`
- `format_immediate()`
- ✅ PASSED

**All Tests**: ✅ PASSED

## Integration with InstructionSelector

The AddressingModeSelector is designed to integrate with InstructionSelector (Phase 4). InstructionSelector can use it to optimize addressing mode selection:

```python
class InstructionSelector:
    def __init__(self, emitter, reg_alloc, mem_alloc):
        self.emitter = emitter
        self.reg_alloc = reg_alloc
        self.mem_alloc = mem_alloc
        self.addr_selector = AddressingModeSelector()  # Add this

    def select_move(self, instr):
        dest_loc = self.reg_alloc.get_location(instr.dest)

        # Use AddressingModeSelector
        mode, operand = self.addr_selector.select_for_location(dest_loc)

        # Optimize for STZ if storing zero
        if isinstance(instr.source, Immediate) and self.addr_selector.should_use_stz(instr.source.value):
            self.emitter.emit_instruction("STZ", operand)
        else:
            # Normal path
            self.emitter.emit_instruction("LDA", src_operand)
            self.emitter.emit_instruction("STA", operand)
```

## Design Decisions

1. **Automatic Range Detection**: Address range automatically determines base mode (direct page / absolute / long). No manual specification needed.

2. **Modifier Stacking**: Indexing and indirection are modifiers applied to base mode. Clear and composable.

3. **Cycle/Size Estimation**: Simplified estimates for optimization decisions. Actual cycles vary with processor mode and page crossings.

4. **Format Helpers**: Convenience methods return strings directly for quick emission without mode enum handling.

5. **STZ Optimization**: Explicit helper method encourages using 65816-specific optimizations.

6. **Enum-Based Modes**: AddressingMode enum provides type safety and clarity versus string constants.

## Known Limitations

1. **No Automatic Indexed Selection**: Caller must explicitly request indexed mode. Future: automatic detection when iterating arrays.

2. **No Stack Relative**: 65816 stack relative mode (sr,S) not implemented. Stack access uses absolute addressing.

3. **Simple Cycle Estimates**: Doesn't account for processor mode (8-bit vs 16-bit affects immediate size/cycles). Future: mode-aware estimates.

4. **No Page Crossing Detection**: Can't predict page crossings (requires knowing runtime values). Uses conservative estimates.

5. **No Instruction-Specific Optimization**: Some instructions have special forms (e.g., BIT immediate). Future: instruction-specific mode selection.

## Files Created/Modified

**Created**:
- `r65/compiler/codegen/addressing_mode.py` (~350 lines)
- `test_addressing_mode.py` (~400 lines)

**Modified**:
- `r65/compiler/codegen/__init__.py` (added exports)

**Total**: ~350 LOC for Phase 5 implementation

## Next Steps

**Phase 6 will implement Function Code Generation**:
- Process MIR functions and basic blocks
- Generate function prologues/epilogues
- Emit block labels
- Integrate all previous phases (memory, registers, instructions, addressing)
- Generate complete function bodies

---

**Phase 5 Status**: ✅ COMPLETE
**All Tests**: ✅ PASSING
**Ready for Phase 6**: ✅ YES
