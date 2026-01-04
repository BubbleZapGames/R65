# B Register: Hidden Accumulator High Byte

## Overview

The B register is the **hidden high byte** of the 65816's 16-bit accumulator. In `#[mode(m8)]` mode, the accumulator splits into:
- **A** (low byte) - directly accessible
- **B** (high byte) - accessible via XBA (Exchange B and A) instruction

R65 exposes B as a **first-class register** for parameter passing, return values, and register aliasing, matching common patterns found in hand-written 65816 assembly.

---

## Hardware Background

### The 65816 Accumulator

The 65816 accumulator has two modes:

| Mode | Size | A Register | B Register |
|------|------|------------|------------|
| `#[mode(m16)]` | 16-bit | Full 16-bit accumulator | N/A (meaningless) |
| `#[mode(m8)]` | 8-bit | Low byte (bits 0-7) | High byte (bits 8-15) |

### XBA Instruction

**XBA** (Exchange B and A) swaps the two bytes of the accumulator:

```asm
; Before XBA:
; A = $34, B = $12

XBA

; After XBA:
; A = $12, B = $34
```

**Cost**: 3 cycles (no memory access required)

**Use case**: Efficiently access both bytes of 16-bit value without memory operations

---

## Language Integration

### Mode Restriction

**B register is ONLY available in `#[mode(m8)]` mode**

```rust
// OK: m8 mode
#[mode(m8, x8)]
fn process(value @ B: u8) -> u8 {
    return B;
}

// ERROR: m16 mode
#[mode(m16, x16)]
fn process(value @ B: u8) -> u8 {  // Compile error!
    return B;
}
```

**Error message**:
```
error: B register only available in m8 mode
  --> test.r65:2:20
   |
2  | fn process(val @ B: u8) {
   |                    ^ B requires #[mode(m8, ...)]
   |
   = note: function has mode m16 where accumulator is 16-bit
```

**Rationale**: In m16 mode, the accumulator is a single 16-bit register. The concept of "high byte" and "low byte" as separate entities doesn't apply.

---

## Parameter Passing

### B as Function Parameter

B can be used alone or combined with other registers:

```rust
// B only
#[mode(m8, x8)]
fn process_high(value @ B: u8) -> u8 {
    return B & 0xF0;
}

// A and B together
#[mode(m8, x8)]
fn pack_word(low @ A: u8, high @ B: u8) -> u16 {
    return A as u16 | ((B as u16) << 8);
}

// B with X
#[mode(m8, x8)]
fn combine(high @ B: u8, index @ X: u8) -> u8 {
    return B + X;
}

// All registers
#[mode(m8, x8)]
fn use_all(a @ A: u8, b @ B: u8, x @ X: u8, y @ Y: u8) -> u8 {
    return a + b + x + y;
}
```

### Caller Setup

The caller is responsible for setting up the B register before the call:

```rust
#[mode(m8, x8)]
fn caller() {
    let low: u8 = 0x34;
    let high: u8 = 0x12;

    // Setup B register
    A = high;    // Load high byte into A
    xba();       // Exchange: A now has low, B has high
    A = low;     // Restore low byte to A

    // Now: A = 0x34, B = 0x12
    let result = pack_word(A, B);
}
```

**Generated Assembly**:
```asm
caller:
    LDA #$12       ; Load high byte
    XBA            ; B = $12, A = (previous A)
    LDA #$34       ; A = $34, B = $12
    JSR pack_word  ; Call with A=low, B=high
```

---

## Return Values

### Returning B

B can be returned alone or with other registers:

```rust
// Return B only
#[mode(m8, x8)]
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // A is NOT restored!
}

// Return A and B
#[mode(m8, x8)]
fn unpack_word(value: u16) -> (u8, u8) {
    A = value as u8;           // Low byte
    B = (value >> 8) as u8;    // High byte
    return A, B;
}

// Return B first, A second
#[mode(m8, x8)]
fn swap_bytes(low @ A: u8, high @ B: u8) -> (u8, u8) {
    return B, A;  // Swap order
}

// Return B and X
#[mode(m8, x8)]
fn get_high_and_index(value: u16, index: u8) -> (u8, u8) {
    B = (value >> 8) as u8;
    X = index;
    return B, X;  // A not returned - caller must preserve!
}
```

### Return Conventions

| Return Statement | First Return | Second Return | Third Return | A Preserved? |
|-----------------|--------------|---------------|--------------|--------------|
| `return B;` | B | - | - | **NO** |
| `return A;` | A | - | - | Yes |
| `return A, B;` | A | B | - | Yes |
| `return B, A;` | B | A | - | Yes (but as 2nd) |
| `return B, X;` | B | X | - | **NO** |
| `return A, B, X;` | A | B | X | Yes |

### Critical Rule: Caller Must Preserve A

**When a function returns only B (or B without A), the callee does NOT restore A.**

```rust
#[mode(m8, x8)]
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // A is clobbered!
}

// WRONG: A is lost
#[mode(m8, x8)]
fn bad_caller() {
    A = 0x42;  // Important value in A
    let high = get_high_byte(0x1234);
    // A is now undefined! Lost 0x42
}

// CORRECT: Preserve A
#[mode(m8, x8)]
fn good_caller() {
    A = 0x42;  // Important value in A
    let saved_a = A;
    let high = get_high_byte(0x1234);
    A = saved_a;  // Restore A
    // A is 0x42 again, high is in B
}
```

### Reading B Return Value

After calling a function that returns B, the caller must read the B value:

```rust
#[mode(m8, x8)]
fn caller() {
    let high = get_high_byte(0x1234);  // Returns in B

    // Option 1: Read via XBA
    xba();           // Exchange: B → A
    let value = A;   // Read from A
    xba();           // Exchange back

    // Option 2: Assign from B directly
    let value @ B = B;  // Alias B register

    // Option 3: Use B value immediately
    B = B & 0xF0;   // Mask high nibble
}
```

---

## Register Aliasing

B can be aliased just like other registers:

```rust
#[mode(m8, x8)]
fn process() {
    let high_byte @ B = 0x12;   // Alias B register
    high_byte = high_byte & 0xF0;  // Modifies B

    let low_byte @ A = 0x34;    // Alias A register

    // Both A and B are now set
    let combined = (low_byte as u16) | ((high_byte as u16) << 8);
}
```

**Generated Assembly**:
```asm
process:
    LDA #$12
    XBA            ; B = $12
    AND #$F0
    XBA            ; A = result, B = $12 (masked)
    LDA #$34       ; A = $34
    ; ...
```

---

## Preservation Rules

### B NOT Allowed in `#[preserves(...)]`

**B cannot appear in the preserves attribute**:

```rust
// ERROR: B in preserves
#[mode(m8, x8)]
#[preserves(B, X, Y)]  // Compile error!
fn bad_function() { }
```

**Error message**:
```
error: B register not allowed in preserves attribute
  --> test.r65:3:14
   |
3  | #[preserves(B, X, Y)]
   |              ^ B cannot be preserved separately from A
   |
   = note: B is the high byte of the A register
```

**Rationale**: B and A are two halves of the same hardware register. Preserving B without preserving A is meaningless. If you need to preserve the high byte, preserve the entire accumulator state via `#[mode(..., transition=inline)]`.

---

## Code Generation and Optimization

### Minimizing XBA Instructions

The compiler **minimizes XBA instructions** by batching B operations:

```rust
#[mode(m8, x8)]
fn efficient_b_usage() {
    B = 0x12;      // XBA to set B
    B = B + 1;     // No XBA - still working with B
    B = B & 0xF0;  // No XBA - still working with B
    A = 0x34;      // XBA to switch back to A
}
```

**Generated Assembly**:
```asm
efficient_b_usage:
    LDA #$12
    XBA            ; Switch to B (only once)
    INC A          ; Operate on B (via A)
    AND #$F0       ; Still on B
    XBA            ; Switch back to A (only once)
    LDA #$34
    RTS
```

**Optimization**: Multiple consecutive B operations compiled with only **two XBA** instructions (enter B, exit B), not one XBA per operation.

### XBA Emission Rules

The compiler emits XBA when:
1. **Switching from A to B context**
2. **Switching from B to A context**
3. **Function entry/exit** with B parameters/returns
4. **Explicit `xba()` builtin** called by programmer

The compiler does NOT emit XBA:
- Between consecutive B operations
- Between consecutive A operations
- When already in the correct context

---

## The xba() Builtin

R65 provides the `xba()` builtin for explicit exchange:

```rust
#[mode(m8, x8)]
fn manual_exchange() {
    A = 0x34;
    B = 0x12;

    xba();  // Explicit exchange
    // Now: A = 0x12, B = 0x34

    A = A + 1;  // Modify new A value (was B)

    xba();  // Exchange back
    // Now: A = 0x34, B = 0x13
}
```

**Generated Assembly**:
```asm
manual_exchange:
    LDA #$34
    XBA
    LDA #$12
    XBA            ; Explicit XBA from xba()
    INC A
    XBA            ; Explicit XBA from xba()
    RTS
```

**Use case**: When you need direct control over exchange timing, or matching specific hand-written assembly patterns.

---

## Common Patterns

### Pattern 1: Pack Two Bytes into Word

```rust
#[mode(m8, x8)]
fn pack_word(low @ A: u8, high @ B: u8) -> u16 {
    // Combine using type casts and shifts
    return (low as u16) | ((high as u16) << 8);
}

// Alternative: Return both bytes and let caller combine
#[mode(m8, x8)]
fn pack_word_v2(low @ A: u8, high @ B: u8) -> (u8, u8) {
    return A, B;  // Caller assembles into u16
}
```

### Pattern 2: Unpack Word into Two Bytes

```rust
#[mode(m8, x8)]
fn unpack_word(value: u16) -> (u8, u8) {
    A = value as u8;           // Low byte (truncate)
    B = (value >> 8) as u8;    // High byte (shift and truncate)
    return A, B;
}

// Caller:
let (low @ A, high @ B) = unpack_word(0x1234);
// A = 0x34, B = 0x12
```

### Pattern 3: Extract High Byte Only

```rust
#[mode(m8, x8)]
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // Caller must preserve A
}
```

### Pattern 4: Byte Swapping

```rust
#[mode(m8, x8)]
fn swap_bytes(value: u16) -> u16 {
    A = value as u8;
    B = (value >> 8) as u8;
    xba();  // Swap them
    return (B as u16) | ((A as u16) << 8);
}
```

### Pattern 5: BCD Addition with Carry

```rust
#[mode(m8, x8)]
fn bcd_add_16bit(a: u16, b: u16) -> u16 {
    // Split into bytes
    let a_low = a as u8;
    let a_high = (a >> 8) as u8;
    let b_low = b as u8;
    let b_high = (b >> 8) as u8;

    // Use A for low byte, B for high byte
    A = a_low;
    B = a_high;

    // Add low bytes (would use SED for decimal mode)
    A = A + b_low;

    // Add high bytes with carry
    B = B + b_high;  // Carry handled by hardware

    return (A as u16) | ((B as u16) << 8);
}
```

---

## Mode Transitions and B Register

B register works correctly with mode transitions:

```rust
#[mode(m8, x8, transition=inline)]
fn safe_b_function(value @ B: u8) -> u8 {
    B = B & 0xF0;
    return B;
}

// Function will emit:
// 1. PHP (save status)
// 2. SEP #$30 (ensure m8, x8)
// 3. ... B operations ...
// 4. PLP (restore status)
// 5. RTS
```

The mode transition ensures the function operates in m8 mode even if called from m16 mode context.

---

## Performance Characteristics

| Operation | Cycles | Notes |
|-----------|--------|-------|
| XBA | 3 | Exchange A and B |
| B = value | 5 | LDA + XBA + STA |
| A to B copy | 3 | XBA (if A already has value) |
| B to memory | 5 | XBA + STA |
| Memory to B | 5 | LDA + XBA |

**Key insight**: B operations are fast (no memory access) but require XBA overhead. Compiler batching minimizes this.

---

## Summary

### When to Use B Register

**Good use cases**:
- Byte packing/unpacking into 16-bit words
- Extracting high/low bytes of u16 values
- Matching hand-written 65816 assembly patterns
- Efficient byte manipulation without memory access
- Functions that naturally work with both bytes of accumulator

**Avoid**:
- Using B in m16 mode (compile error)
- Mixing B preservation with `#[preserves(...)]` (compile error)
- Excessive XBA instructions (let compiler optimize)

### Design Philosophy

R65's B register support reflects the **hardware-first design principle**:
- Exposes real hardware capabilities (XBA instruction)
- Matches hand-written assembly patterns
- Makes byte manipulation efficient and explicit
- No hidden overhead - XBA cost is visible in language semantics
- Compiler optimizes XBA placement, but programmer controls usage

---

*Last Updated: 2026-01-03*
*Status: Design Complete, Implementation Pending*
