# Processor Mode System

## Overview

R65 uses a **simplified automatic mode system** where CPU modes are inferred from function parameter types rather than specified via attributes.

### Key Principles

1. **X/Y Always 16-bit**: Index registers X and Y are always in x16 mode (16-bit)
2. **A Mode Inferred**: Accumulator mode (m8/m16) is inferred from `@ A` parameter type
3. **Automatic Transitions**: Compiler inserts REP/SEP instructions as needed
4. **`#[mode]` for DBR only**: The attribute only controls data bank register management

---

## Mode Rules

### Default Mode

All functions start with the default mode:
- **M mode**: m8 (8-bit accumulator)
- **X mode**: x16 (16-bit index registers) - always

### Function Entry Mode

The compiler infers entry mode from function parameters:

```rust
// m8 mode (default) - no u16 @ A parameter
fn process(value @ A: u8) { }
fn helper() { }

// m16 mode - inferred from u16 @ A parameter
fn process16(value @ A: u16) { }
fn wide_ops(data @ A: u16, idx @ X: u16) { }
```

### X/Y Parameter Validation

X and Y register parameters **must** be u16. The compiler rejects u8/i8:

```rust
// ✅ Correct - X/Y are u16
fn indexed(idx @ X: u16) { }
fn both_index(x @ X: u16, y @ Y: u16) { }

// ❌ Compile error - X/Y must be u16
fn invalid(idx @ X: u8) { }  // Error: X/Y registers are always 16-bit
```

---

## Code Generation

### Prologue

For functions with m16 entry mode (u16 @ A parameter):
```asm
function_name:
    .ACCU 16          ; Tell assembler A is 16-bit
    .INDEX 16         ; Tell assembler X/Y are 16-bit
    REP #$20          ; Set 16-bit accumulator mode
    ; ... function body ...
```

For functions with m8 entry mode (default):
```asm
function_name:
    .ACCU 8           ; Tell assembler A is 8-bit
    .INDEX 16         ; Tell assembler X/Y are 16-bit
    ; ... function body (no mode switch needed) ...
```

### Epilogue

For m16 functions, restore m8 before return:
```asm
    ; ... function body ...
    SEP #$20          ; Restore 8-bit accumulator mode
    RTS/RTL
```

### Cross-Mode Calls

When calling a function with different entry mode:
```rust
fn caller() {
    // caller is m8 (default)
    callee_m16(0x1234);  // callee expects m16
}

fn callee_m16(value @ A: u16) {
    // callee runs in m16
}
```

The callee's prologue handles the mode switch (REP #$20), and epilogue restores (SEP #$20).

---

## Data Bank Management

The `#[mode]` attribute is retained **only** for data bank register (DBR) management:

### databank=none (default)

No DBR management. Function uses whatever DBR is set by caller:
```rust
fn local_work() { }  // Uses caller's DBR
```

### databank=inline

Callee saves DBR, sets it to function's bank, restores on exit:
```rust
#[mode(databank=inline)]
#[bank(2)]
far fn graphics_helper() {
    // DBR automatically set to bank 2
    // Original DBR restored before RTL
}
```

Generated code:
```asm
graphics_helper:
    PHB              ; Save caller's DBR
    LDA #$02         ; Load function's bank
    PHA
    PLB              ; Set DBR to bank 2
    ; ... function body ...
    PLB              ; Restore caller's DBR
    RTL
```

### databank=caller

Caller is responsible for setting DBR (useful for batching):
```rust
#[mode(databank=caller)]
#[bank(2)]
far fn helper1() { }

#[mode(databank=caller)]
#[bank(2)]
far fn helper2() { }

fn caller() {
    // Manually manage DBR for multiple calls
    PHB
    LDA #$02
    PHA
    PLB
    helper1();  // DBR already set
    helper2();  // DBR already set
    PLB
}
```

---

## Migration from Old System

### Old Syntax (No Longer Supported)

```rust
// ❌ These no longer work:
#[mode(m8, x8)]
#[mode(m16, x16)]
#[mode(m8, x16, transition=inline)]
#[mode(m16, transition=caller)]
```

### New Syntax

```rust
// ✅ Mode inferred from parameters:
fn process(val @ A: u8) { }       // m8
fn wide(val @ A: u16) { }         // m16 (inferred)
fn indexed(idx @ X: u16) { }      // m8, X/Y always u16

// ✅ Only databank in #[mode]:
#[mode(databank=inline)]
far fn far_func() { }
```

### Compiler Errors

The compiler provides helpful errors for old syntax:
```
error: #[mode(m8)] is no longer supported.
  CPU mode is now automatically inferred from parameter types:
  - u16 @ A parameter -> m16 entry mode
  - otherwise -> m8 entry mode (default)
  - X/Y registers are always u16 (x16 mode)
  Use #[mode(databank=...)] for data bank management only.
```

---

## Interrupt Handlers

Interrupt handlers use automatic mode management:

```rust
#[interrupt(nmi)]
fn vblank_handler() {
    // Interrupts can fire in any mode
    // RTI restores original STATUS (including M/X flags)
}
```

The interrupt prologue saves STATUS via PHP, and RTI restores it. The handler body executes in the default mode (m8, x16).

---

## Design Rationale

### Why Always x16?

1. **Simplicity**: One less thing to track and manage
2. **Performance**: 16-bit index registers are generally more useful for SNES
3. **Safety**: Prevents mode mismatch bugs with X/Y parameters
4. **Compatibility**: Most SNES code uses x16 mode

### Why Infer A Mode?

1. **Ergonomics**: No need for explicit mode annotations
2. **Type Safety**: Mode is tied to actual parameter type
3. **Automatic Transitions**: Compiler handles REP/SEP
4. **Fewer Bugs**: Can't have type/mode mismatch

### Why Keep databank in #[mode]?

1. **Orthogonal Concern**: DBR management is separate from CPU mode
2. **Far Calls**: Only relevant for cross-bank calls
3. **Batching**: Allows `databank=caller` optimization

---

## Summary

| Aspect | Old System | New System |
|--------|-----------|------------|
| M mode | `#[mode(m8/m16)]` | Inferred from `@ A` type |
| X mode | `#[mode(x8/x16)]` | Always x16 |
| Transitions | `transition=none/inline/caller` | Automatic |
| DBR | `databank=none/inline/caller` | Unchanged |
| X/Y params | Any type | Must be u16 |

*Last Updated: 2026-01-19*
