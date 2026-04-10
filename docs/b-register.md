# B Register: Hidden Accumulator High Byte

## Overview

The B register is the **hidden high byte** of the 65816's 16-bit accumulator. In m8 mode (the default), the accumulator splits into:
- **A** (low byte) - directly accessible
- **B** (high byte) - accessible via XBA (Exchange B and A) instruction

R65 exposes B as a **first-class register** for parameter passing, return values, and register aliasing, matching common patterns found in hand-written 65816 assembly.

---

## Hardware Background

### The 65816 Accumulator

The 65816 accumulator has two modes:

| Mode | Size | A Register | B Register |
|------|------|------------|------------|
| m16 (via `@ A: u16` param) | 16-bit | Full 16-bit accumulator | N/A (meaningless) |
| m8 (default) | 8-bit | Low byte (bits 0-7) | High byte (bits 8-15) |

### XBA Instruction

**XBA** (Exchange B and A) swaps the two bytes of the 16-bit accumulator (C register):

```asm
; Before XBA:
; A = $34, B = $12 (C = $1234)

XBA

; After XBA:
; A = $12, B = $34 (C = $3412)
```

**Cost**: 3 cycles (no memory access required)

**Use case**: Efficiently access both bytes of 16-bit value without memory operations

### TAX/TXA Behavior with Mixed Modes

When accumulator is 8-bit (m8) and index registers are 16-bit (x16), transfer instructions behave based on the **destination register size**:

```asm
; m8/x16 mode (8-bit A, 16-bit X/Y)
; C register = $HHLL where A=$LL, B=$HH

TAX     ; Transfers full 16-bit C ($HHLL) to X
        ; X now contains both A and B bytes!

TXA     ; Transfers full 16-bit X to C
        ; Both A and B are overwritten!
```

**Important**: TAX/TXA always transfer 16 bits when X is 16-bit, regardless of the M flag. This means:
- `X = A` copies both A and B (the hidden high byte) to X
- `A = X` overwrites both A and B with X's value

To transfer only the low byte with zero-extension, use explicit casts (`X = A as u8`).

---

## Language Integration

### Automatic Mode Switching

**B register access automatically switches to 8-bit mode** if not already in 8-bit mode.

```rust
fn process(value @ A: u16) {
    BG1VOFS = A;   // 16-bit store
    BG1VOFS = B;   // Auto SEP #$20, then XBA to access B
}
```

The compiler ensures XBA operations occur in 8-bit mode for correct B register semantics.

---

## Parameter Passing

### B as Function Parameter

B can be used alone or combined with other registers (in m8 mode):

```rust
// B only (m8 mode - default)
fn process_high(value @ B: u8) -> u8 {
    return B & 0xF0;
}

// A and B together (m8 mode - default)
fn pack_word(low @ A: u8, high @ B: u8) -> u16 {
    return A as u16 | ((B as u16) << 8);
}

// B with X (m8 mode - default, X/Y always u16)
fn combine(high @ B: u8, index @ X: u16) -> u8 {
    return B + (index as u8);
}

// All registers (m8 mode - default, X/Y always u16)
fn use_all(a @ A: u8, b @ B: u8, x @ X: u16, y @ Y: u16) -> u8 {
    return a + b + (x as u8) + (y as u8);
}
```

### Caller Setup

The caller is responsible for setting up the B register before the call:

```rust
// m8 mode (default)
fn caller() {
    let low: u8 = 0x34;
    let high: u8 = 0x12;

    // Setup B register
    A = high;       // Load high byte into A
    asm!("XBA");    // Exchange: A now has low, B has high
    A = low;        // Restore low byte to A

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

B can be returned alone or with other registers (in m8 mode):

```rust
// Return B only (m8 mode - default)
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // A is NOT restored!
}

// Return A and B (m8 mode - default)
fn unpack_word(value: u16) -> (u8, u8) {
    A = value as u8;           // Low byte
    B = (value >> 8) as u8;    // High byte
    return A, B;
}

// Return B first, A second (m8 mode - default)
fn swap_bytes(low @ A: u8, high @ B: u8) -> (u8, u8) {
    return B, A;  // Swap order
}

// Return B and X (m8 mode - default)
fn get_high_and_index(value: u16, index: u8) -> (u8, u16) {
    B = (value >> 8) as u8;
    X = index as u16;
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
// m8 mode (default)
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // A is clobbered!
}

// WRONG: A is lost
fn bad_caller() {
    A = 0x42;  // Important value in A
    let high = get_high_byte(0x1234);
    // A is now undefined! Lost 0x42
}

// CORRECT: Preserve A
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
// m8 mode (default)
fn caller() {
    let high = get_high_byte(0x1234);  // Returns in B

    // Option 1: Read via XBA
    asm!("XBA");     // Exchange: B → A
    let value = A;   // Read from A
    asm!("XBA");     // Exchange back

    // Option 2: Assign from B directly
    let value @ B = B;  // Alias B register

    // Option 3: Use B value immediately
    B = B & 0xF0;   // Mask high nibble
}
```

---

## Register Aliasing

B can be aliased just like other registers (in m8 mode):

```rust
// m8 mode (default)
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
#[preserves(B, X, Y)]  // Compile error!
fn bad_function() { }
```

**Error message**:
```
error: B register not allowed in preserves attribute
  --> test.r65:2:14
   |
2  | #[preserves(B, X, Y)]
   |              ^ B cannot be preserved separately from A
   |
   = note: B is the high byte of the A register
```

**Rationale**: B and A are two halves of the same hardware register. Preserving B without preserving A is meaningless.

---

## Register Transfers

### Direct Transfers (No Cast)

Direct register-to-register assignments use native transfer instructions:

| Source | Code | Generated Assembly | Notes |
|--------|------|-------------------|-------|
| `X = A` | TAX | `TAX` | Transfers full 16-bit C to X (includes B!) |
| `A = X` | TXA | `TXA` | Overwrites both A and B with X |
| `Y = A` | TAY | `TAY` | Transfers full 16-bit C to Y (includes B!) |
| `A = Y` | TYA | `TYA` | Overwrites both A and B with Y |

**Warning**: In m8/x16 mode, `TAX` transfers BOTH bytes of the accumulator (A and B) to X. The B register value becomes the high byte of X!

### Casted Transfers (Zero-Extended)

When casting to `u8`, the compiler uses the `AND #$00FF` pattern to ensure clean 16-bit values:

```rust
X = A as u8;  // Zero-extend A to 16-bit, put in X
A = X as u8;  // Take low byte of X, zero-extend to A
Y = A as u8;  // Same pattern
```

**Generated Assembly for `X = A as u8`**:
```asm
REP #$20        ; 16-bit mode
AND #$00FF      ; Clear high byte (B) in 16-bit C
TAX             ; Transfer clean 16-bit value to X
SEP #$20        ; Back to 8-bit mode
```

### B Register Transfers

B is always 8-bit, so transfers to/from 16-bit X/Y require zero-extension:

| Transfer | Generated Assembly |
|----------|-------------------|
| `X = B` | `XBA; REP; AND #$00FF; TAX; SEP; XBA` |
| `B = X` | `PHA; REP; TXA; SEP; XBA; PLA` |
| `Y = B` | `XBA; REP; AND #$00FF; TAY; SEP; XBA` |
| `B = Y` | `PHA; REP; TYA; SEP; XBA; PLA` |

**Example `X = B`**:
```asm
XBA             ; Get B value into A position
REP #$20        ; 16-bit mode
AND #$00FF      ; Zero-extend to clean 16-bit
TAX             ; Transfer to X
SEP #$20        ; Back to 8-bit mode
XBA             ; Restore A/B positions
```

---

## Code Generation and Optimization

### Minimizing XBA Instructions

The compiler **minimizes XBA instructions** by batching B operations:

```rust
// m8 mode (default)
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
4. **Explicit `asm!("XBA")`** written by the programmer

The compiler does NOT emit XBA:
- Between consecutive B operations
- Between consecutive A operations
- When already in the correct context

---

## Explicit XBA via asm!()

There is no `xba()` built-in function. When you need direct control over
exchange timing — e.g. matching specific hand-written assembly patterns —
drop into inline assembly:

```rust
// m8 mode (default)
fn manual_exchange() {
    A = 0x34;
    B = 0x12;

    asm!("XBA");  // Explicit exchange
    // Now: A = 0x12, B = 0x34

    A = A + 1;    // Modify new A value (was B)

    asm!("XBA");  // Exchange back
    // Now: A = 0x34, B = 0x13
}
```

**Generated Assembly**:
```asm
manual_exchange:
    LDA #$34
    XBA
    LDA #$12
    XBA            ; Explicit XBA from asm!()
    INC A
    XBA            ; Explicit XBA from asm!()
    RTS
```

---

## Common Patterns

### Pattern 1: Pack Two Bytes into Word

```rust
// m8 mode (default)
fn pack_word(low @ A: u8, high @ B: u8) -> u16 {
    // Combine using type casts and shifts
    return (low as u16) | ((high as u16) << 8);
}

// Alternative: Return both bytes and let caller combine
fn pack_word_v2(low @ A: u8, high @ B: u8) -> (u8, u8) {
    return A, B;  // Caller assembles into u16
}
```

### Pattern 2: Unpack Word into Two Bytes

```rust
// m8 mode (default)
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
// m8 mode (default)
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // Caller must preserve A
}
```

### Pattern 4: Byte Swapping

```rust
// m8 mode (default)
fn swap_bytes(value: u16) -> u16 {
    A = value as u8;
    B = (value >> 8) as u8;
    asm!("XBA");  // Swap them
    return (B as u16) | ((A as u16) << 8);
}
```

### Pattern 5: BCD Addition with Carry

```rust
// m8 mode (default)
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

## Mode and B Register

B register is only available in m8 mode (the default). Functions using B should not have a `@ A: u16` parameter:

```rust
// m8 mode (default) - B is available
fn safe_b_function(value @ B: u8) -> u8 {
    B = B & 0xF0;
    return B;
}

// m16 mode (due to @ A: u16) - B is NOT available
fn wide_function(value @ A: u16) -> u16 {
    // B = 0x12;  // ERROR: B not available in m16 mode
    return value + 1;
}
```

The compiler automatically handles mode transitions when calling between m8 and m16 functions.

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

*Last Updated: 2026-02-01*
*Status: Implemented*
