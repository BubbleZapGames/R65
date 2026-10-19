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

### Entry and Exit Widths

Each function has an entry width (m16 if it has a `u16 @ A` parameter,
otherwise m8) and an exit width (m16 if it returns a u16 in A). The
function's `.ACCU` anchor declares its entry width, and every caller
switches to it before the call:

```asm
function_name:
    .ACCU 16          ; entered in m16: the caller already did REP #$20
    .INDEX 16
    ; ... function body ...
    RTS/RTL           ; returns in its exit width
```

### Cross-Mode Calls

```rust
fn caller() {
    // caller is m8 (default)
    callee_m16(0x1234);  // callee expects m16
}

fn callee_m16(value @ A: u16) {
    // callee runs in m16
}
```

The caller emits `REP #$20` before loading a `u16 @ A` argument and calling,
and switches back to the width it needs after the call (`SEP #$20` after a
u16 return). Interrupt handlers and the reset entry have no known entry
width; they set their own (see [interrupt-mode-transition.md](interrupt-mode-transition.md)).

### Switching Inside a Function

Codegen switches lazily: an instruction that needs the other width emits
`REP #$20` or `SEP #$20` first and leaves the CPU there. Each block is
entered in a width chosen from what its code wants
(`analysis/width_demand.py`). `mir/mode_tracker.py` follows explicit
`SetMode` only, so it would call every block of an m8 function m8 and
restore m8 at each one, inside loops too.

- **Demand.** `instruction_width` predicts what codegen does with each
  MIR instruction: the width its first accumulator operation runs in
  (none for index-register work, X/Y compares and moves, and branches)
  and the width it leaves. That is usually the width of the
  instruction's type. A call needs its first argument's width and leaves
  the callee's exit width; inline asm runs in the function's width. The
  prediction runs after register allocation, so it knows a u16 move into
  X goes through A only from a stack slot.
- **Assignment.** `WidthAssignment` picks each block's entry width (the
  entry block keeps the function's) to run the fewest switches, each
  weighted 10× per loop level. A tie keeps the function's width.

Codegen labels each block with its width (`.ACCU`), switches at its top
only if an already emitted predecessor ends in the other width, and
switches before a block's closing branches to the width its successors
are entered in, so a loop latch hands its back edge the width the loop
runs in. A switch of M alone leaves N, Z, C and V, so it may sit between
a compare and its branch. A wrong prediction costs a switch, never
correctness: the emitter still switches before each instruction that
needs a width.

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

### Inferred function modes

```rust
// ✅ Mode inferred from parameters:
fn process(val @ A: u8) { }       // m8
fn wide(val @ A: u16) { }         // m16 (inferred)
fn indexed(idx @ X: u16) { }      // m8, X/Y always u16

// ✅ Only databank in #[mode]:
#[mode(databank=inline)]
far fn far_func() { }
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

The CPU pushes STATUS when the interrupt fires and RTI restores it, so the prologue needs no PHP. The handler body executes in the default mode (m8, x16).

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

*Last Updated: 2026-10-18*
