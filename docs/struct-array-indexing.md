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

Call the `mul8()` or `mul16()` built-in function for general multiplication.

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
- Much faster than mul8()/mul16() for small multipliers
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

## Strategy Selection Algorithm

The compiler uses a simple threshold-based approach:

```
1. Constant index?
   → Compute at compile time (0 cycles)

2. Struct size <= 16 bytes?
   a. Power-of-2 (1, 2, 4, 8, 16)
      → Use shifts
   b. Non-power-of-2 (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
      → Use shift-and-add decomposition

3. Struct size > 16 bytes?
   → Use mul8() or mul16()
```

### Rationale

- **Threshold of 16 bytes**: Struct sizes 1-16 cover the vast majority of real-world structs. Shift-and-add is efficient for all these sizes (typically 6-34 cycles vs 100-150 for mul8()/mul16()).
- **Simple implementation**: Easy to understand, maintain, and verify.

### Complete Decomposition Table (sizes 1-16)

| Size | Binary | Method | ~Cycles |
|------|--------|--------|---------|
| 1 | 00001 | (identity) | 2 |
| 2 | 00010 | x<<1 | 6 |
| 3 | 00011 | (x<<1)+x | 18 |
| 4 | 00100 | x<<2 | 8 |
| 5 | 00101 | (x<<2)+x | 20 |
| 6 | 00110 | (x<<2)+(x<<1) | 26 |
| 7 | 00111 | (x<<3)-x | 20 |
| 8 | 01000 | x<<3 | 10 |
| 9 | 01001 | (x<<3)+x | 22 |
| 10 | 01010 | (x<<3)+(x<<1) | 28 |
| 11 | 01011 | (x<<3)+(x<<1)+x | 34 |
| 12 | 01100 | (x<<3)+(x<<2) | 28 |
| 13 | 01101 | (x<<4)-(x<<1)-x | 30 |
| 14 | 01110 | (x<<4)-(x<<1) | 26 |
| 15 | 01111 | (x<<4)-x | 22 |
| 16 | 10000 | x<<4 | 12 |

## Current Implementation

The current implementation uses the Strategy Selection Algorithm above:

- **Constant indices:** Compile-time offset calculation (0 cycles)
- **Struct size 1:** Identity (no multiplication needed)
- **Struct sizes 2-16:** Shift-and-add decomposition (inline, no function call)
- **Struct sizes > 16:** `mul8()`/`mul16()` runtime function call

The shift-and-add implementation is in `r65/compiler/mir/lowerers/multiply.py`, which is shared between expression and assignment lowerers.

## Code Locations

- **Shared multiplication helpers:** `r65/compiler/mir/lowerers/multiply.py`
  - `emit_shift_and_add_multiply()` - shift-and-add for multipliers 1-16
  - `compute_array_field_offset()` - computes `(index * struct_size) + field_offset`

- **MIR lowering (expression):** `r65/compiler/mir/lowerers/expression.py`
  - `_lower_array_field_access()` - handles `array[index].field` reads
  - `_lower_nested_field_access()` - handles `outer.inner.field` reads

- **MIR lowering (assignment):** `r65/compiler/mir/lowerers/assignment.py`
  - `_lower_array_field_assignment()` - handles `array[index].field = value` writes
  - `_lower_nested_field_assignment()` - handles `outer.inner.field = value` writes

- **Nested field chains:** `r65/compiler/mir/builder.py`
  - `peel_field_chain()` - folds `outer.inner.leaf` into `(base, total_offset)`.
    Struct and union fields are laid out inline, so every link but the innermost
    contributes only a compile-time constant; `array[i].inner.leaf` and
    `ptr.inner.leaf` reuse the array/pointer paths with the folded offset.

- **Liveness analysis:** `r65/compiler/mir/liveness.py`
  - `interferes()` - precise per-instruction liveness for stack slot reuse

## References

- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Hacker's Delight](https://en.wikipedia.org/wiki/Hacker%27s_Delight) - Bit manipulation tricks
