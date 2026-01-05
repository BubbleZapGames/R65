# Struct Array Indexing Optimization

This document describes the challenge of indexing arrays of structs on the 65816 processor and potential optimization strategies.

## The Problem

When accessing `array[index].field`, the compiler must compute:

```
address = array_base + (index * struct_size) + field_offset
```

For **constant indices**, this is trivial - compute the offset at compile time.

For **variable indices**, we need runtime multiplication. The 65816 has no hardware multiply instruction, making `index * struct_size` expensive for non-power-of-2 struct sizes.

### Power-of-2 Struct Sizes

For struct sizes that are powers of 2 (1, 2, 4, 8, 16...), we use left shifts:

```asm
; index * 4 (struct size = 4)
TXA
ASL A        ; ×2
ASL A        ; ×4
```

Cost: ~2 cycles per shift, very efficient.

### Non-Power-of-2 Struct Sizes

For struct sizes like 3, 5, 6, 7, 9, 10, etc., we need a different approach.

## Optimization Strategies

### Strategy 1: Software Multiplication (Current)

Call the `mul()` built-in function for general multiplication.

```asm
; index * 5 (struct size = 5)
TXA
LDX #5
JSR __mul8      ; A = A * X
```

**Pros:**
- Simple to implement
- Works for any struct size
- No additional memory usage

**Cons:**
- Slow (~100-150 cycles for 8-bit multiply)
- Function call overhead

**When to use:** Default fallback, or when other strategies aren't applicable.

### Strategy 2: Shift-and-Add Decomposition

Decompose multiplication into shifts and adds based on binary representation.

```asm
; index * 3 = (index << 1) + index
TXA
ASL A        ; A = index * 2
STA temp
TXA          ; A = index
CLC
ADC temp     ; A = index * 3

; index * 5 = (index << 2) + index
TXA
ASL A
ASL A        ; A = index * 4
STA temp
TXA
CLC
ADC temp     ; A = index * 5

; index * 7 = (index << 3) - index
TXA
ASL A
ASL A
ASL A        ; A = index * 8
SEC
SBC ...      ; A = index * 7 (need original value)
```

**Pros:**
- Much faster than mul() for small multipliers
- No function call overhead
- No memory for tables

**Cons:**
- Code size grows with multiplier complexity
- Needs scratch register to hold intermediate values
- Complex multipliers need many operations

**Cost analysis:**

| Multiplier | Binary | Method | Operations | ~Cycles |
|------------|--------|--------|------------|---------|
| 3 | 11 | (x<<1) + x | 2 | ~12 |
| 5 | 101 | (x<<2) + x | 2 | ~14 |
| 6 | 110 | (x<<2) + (x<<1) | 3 | ~18 |
| 7 | 111 | (x<<3) - x | 2 | ~14 |
| 9 | 1001 | (x<<3) + x | 2 | ~16 |
| 10 | 1010 | (x<<3) + (x<<1) | 3 | ~20 |
| 15 | 1111 | (x<<4) - x | 2 | ~16 |

**When to use:** Struct sizes with low popcount or near powers of 2.

### Strategy 3: Offset Lookup Table (LUT)

Precompute offsets for each possible index value.

```asm
; For struct size = 3, array of 8 elements
offsets_lut:
    .db 0, 3, 6, 9, 12, 15, 18, 21

; Access array[X].field:
    LDA offsets_lut,X    ; A = base offset for element
    CLC
    ADC #field_offset    ; Add field offset (if non-zero)
    TAX
    LDA array_base,X     ; Load the field value
```

**Pros:**
- Very fast (~10-12 cycles)
- Constant time regardless of struct size
- Simple code generation

**Cons:**
- Uses ROM/RAM for table
- Table size = array_size × entry_size
- Requires 16-bit entries if struct_size × max_index > 255

**Memory cost:**

| Array Size | Entry Size | Table Size |
|------------|------------|------------|
| 8 | u8 | 8 bytes |
| 16 | u8 | 16 bytes |
| 64 | u8 | 64 bytes |
| 64 | u16 | 128 bytes |

**When to use:**
- Small to medium arrays with known size
- Performance-critical code
- When ROM space is available

### Strategy 4: Hybrid Approach

Combine strategies based on context:

1. **Constant index:** Compute at compile time (always)
2. **Power-of-2 struct:** Use shifts (always)
3. **Small array + non-power-of-2:** Generate LUT
4. **Large array + simple multiplier:** Use shift-and-add
5. **Large array + complex multiplier:** Use mul()

## Current Implementation

The current implementation uses:

- **Constant indices:** Compile-time offset calculation
- **Power-of-2 structs:** Shift instructions
- **Non-power-of-2 structs:** `mul()` call (Strategy 1)

## Future Improvements

### Phase 1: Shift-and-Add (Low-hanging fruit)
Implement decomposition for common struct sizes (3, 5, 6, 7, 9, 10, 12).
Most structs fall in the 2-16 byte range where this is optimal.

### Phase 2: LUT Generation
For arrays with:
- Known size at compile time
- Non-power-of-2 element size
- Size × max_index ≤ 255 (or 65535 for u16)

Generate offset tables automatically.

### Phase 3: Profile-Guided Selection
Allow programmer hints or profile data to select strategy:

```rust
#[optimize(lut)]       // Force LUT generation
#[optimize(inline)]    // Force shift-and-add
#[optimize(size)]      // Prefer mul() to save ROM
```

## Code Locations

- **MIR lowering (expression):** `r65/compiler/mir/lowerers/expression.py`
  - `_lower_array_field_access()`
  - `_compute_array_field_offset()`

- **MIR lowering (assignment):** `r65/compiler/mir/lowerers/assignment.py`
  - `_lower_array_field_assignment()`
  - `_compute_array_field_offset()`

- **Instruction selection:** `r65/compiler/codegen/instruction_select.py`
  - `_emit_multiply()` - currently only handles power-of-2

## References

- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Hacker's Delight](https://en.wikipedia.org/wiki/Hacker%27s_Delight) - Bit manipulation tricks
