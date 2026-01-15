# STATUS Register Flag Properties

This document describes the STATUS register property access feature, which provides direct access to individual CPU flags via `STATUS.Carry`, `STATUS.Zero`, etc.

## Overview

The 65816 processor STATUS register (P register) contains 8 flags that control processor behavior and reflect the results of operations. R65 exposes these flags as properties on the `STATUS` register, enabling:

1. **Optimized conditional branching** - `if STATUS.Carry` generates direct `BCS`/`BCC` instructions
2. **Direct flag manipulation** - `STATUS.Carry = true` generates `SEC`
3. **Type-safe access** - Flags are boolean-typed with compile-time validation

## STATUS Register Layout

```
Bit 7  6  5  4  3  2  1  0
    N  V  M  X  D  I  Z  C
    |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  +-- Carry
    |  |  |  |  |  |  +----- Zero
    |  |  |  |  |  +-------- IRQ Disable
    |  |  |  |  +----------- Decimal Mode
    |  |  |  +-------------- Index Register Size (X flag)
    |  |  +----------------- Accumulator Size (M flag)
    |  +-------------------- Overflow
    +----------------------- Negative
```

## Flag Properties

| Property | Bit | Mask | Branch Instructions | Write Instructions | Writable |
|----------|-----|------|---------------------|-------------------|----------|
| `STATUS.Carry` | 0 | 0x01 | BCS / BCC | SEC / CLC | Yes |
| `STATUS.Zero` | 1 | 0x02 | BEQ / BNE | - | No |
| `STATUS.Irq` | 2 | 0x04 | (bit test) | SEI / CLI | Yes |
| `STATUS.Decimal` | 3 | 0x08 | (bit test) | SED / CLD | Yes |
| `STATUS.XY16` | 4 | 0x10 | (bit test) | REP #$10 / SEP #$10 | Yes |
| `STATUS.A16` | 5 | 0x20 | (bit test) | REP #$20 / SEP #$20 | Yes |
| `STATUS.Overflow` | 6 | 0x40 | BVS / BVC | - | No |
| `STATUS.Negative` | 7 | 0x80 | BMI / BPL | - | No |

### Branchable vs Non-Branchable Flags

**Branchable flags** (Carry, Zero, Overflow, Negative) have dedicated branch instructions that directly test the flag. These generate optimal single-instruction branches.

**Non-branchable flags** (Irq, Decimal, XY16, A16) don't have dedicated branch instructions. Testing these flags generates a bit-test sequence: `PHP; PLA; AND #mask; BNE/BEQ`.

### Writable vs Read-Only Flags

**Writable flags** can be set or cleared directly:
- `Carry` - SEC/CLC
- `Irq` - SEI/CLI
- `Decimal` - SED/CLD
- `XY16` - REP #$10 (true=16-bit) / SEP #$10 (false=8-bit)
- `A16` - REP #$20 (true=16-bit) / SEP #$20 (false=8-bit)

**Read-only flags** (Zero, Overflow, Negative) are set by CPU operations and cannot be written directly. Attempting to write to these flags is a compile error.

## Usage Examples

### Conditional Branching

```rust
// Direct branch on branchable flag
if STATUS.Carry {
    // Generates: BCS label
    handle_carry_set();
}

// Negated condition
if !STATUS.Zero {
    // Generates: BNE label
    handle_not_zero();
}

// Non-branchable flag (generates bit test)
if STATUS.Irq {
    // Generates: PHP; PLA; AND #$04; BNE label
    handle_irq_disabled();
}
```

### Flag Manipulation

```rust
// Set flags
STATUS.Carry = true;      // SEC
STATUS.Irq = true;        // SEI (disable interrupts)
STATUS.Decimal = true;    // SED (enable BCD mode)

// Clear flags
STATUS.Carry = false;     // CLC
STATUS.Irq = false;       // CLI (enable interrupts)
STATUS.Decimal = false;   // CLD (disable BCD mode)

// Mode flags (A16/XY16 = true means 16-bit, false means 8-bit)
STATUS.XY16 = true;       // REP #$10 (16-bit index)
STATUS.XY16 = false;      // SEP #$10 (8-bit index)
STATUS.A16 = true;    // REP #$20 (16-bit accumulator)
STATUS.A16 = false;   // SEP #$20 (8-bit accumulator)
```

### Reading Flag Values

```rust
// Read flag into variable (less common)
let carry_was_set = STATUS.Carry;
// Generates: PHP; PLA; AND #$01; STA variable

// For flags other than Carry, result is normalized to 0/1
let is_negative = STATUS.Negative;
// Generates: PHP; PLA; AND #$80; BEQ +; LDA #1; +: STA variable
```

### Compound Conditions

Compound conditions fall back to regular bit operations:

```rust
// Compound condition - each flag tested separately
if STATUS.Carry && STATUS.Zero {
    // Short-circuit evaluation:
    // Test Carry first, if set then test Zero
}
```

## Code Generation

### Branchable Flag Conditions

| Condition | Generated Code |
|-----------|----------------|
| `if STATUS.Carry` | `BCS label` |
| `if !STATUS.Carry` | `BCC label` |
| `if STATUS.Zero` | `BEQ label` |
| `if !STATUS.Zero` | `BNE label` |
| `if STATUS.Overflow` | `BVS label` |
| `if !STATUS.Overflow` | `BVC label` |
| `if STATUS.Negative` | `BMI label` |
| `if !STATUS.Negative` | `BPL label` |

### Non-Branchable Flag Conditions

| Condition | Generated Code |
|-----------|----------------|
| `if STATUS.Irq` | `PHP; PLA; AND #$04; BNE label` |
| `if !STATUS.Irq` | `PHP; PLA; AND #$04; BEQ label` |
| `if STATUS.Decimal` | `PHP; PLA; AND #$08; BNE label` |
| `if STATUS.XY16` | `PHP; PLA; AND #$10; BNE label` |
| `if STATUS.A16` | `PHP; PLA; AND #$20; BNE label` |

### Flag Write Instructions

| Assignment | Generated Code |
|------------|----------------|
| `STATUS.Carry = true` | `SEC` |
| `STATUS.Carry = false` | `CLC` |
| `STATUS.Irq = true` | `SEI` |
| `STATUS.Irq = false` | `CLI` |
| `STATUS.Decimal = true` | `SED` |
| `STATUS.Decimal = false` | `CLD` |
| `STATUS.XY16 = true` | `REP #$10` |
| `STATUS.XY16 = false` | `SEP #$10` |
| `STATUS.A16 = true` | `REP #$20` |
| `STATUS.A16 = false` | `SEP #$20` |

### SEP/REP Combining Optimization

When multiple `STATUS.A16` and `STATUS.XY16` assignments occur sequentially with the same direction (both setting to `true` or both setting to `false`), the compiler automatically combines them into a single SEP or REP instruction:

```rust
// Sequential assignments to 16-bit mode (true = 16-bit)
STATUS.A16 = true;
STATUS.XY16 = true;
// Generates: REP #$30 (not REP #$20; REP #$10)

// Sequential assignments to 8-bit mode (false = 8-bit)
STATUS.A16 = false;
STATUS.XY16 = false;
// Generates: SEP #$30 (not SEP #$20; SEP #$10)
```

This optimization only applies when:
1. Both assignments are to `A16` or `XY16` flags (the mode flags that use SEP/REP)
2. Both assignments have the same boolean value (both `true` or both `false`)
3. There are no intervening instructions between the assignments

Mixed directions are handled separately:

```rust
STATUS.A16 = true;   // 16-bit accumulator
STATUS.XY16 = false; // 8-bit index
// Generates: REP #$20; SEP #$10 (cannot be combined)
```

## Function Return Optimization

Functions that directly return a STATUS flag enable optimized branch generation at call sites:

```rust
fn is_carry_set() -> bool {
    return STATUS.Carry;
}

// At call site, compiler can use direct branch
if is_carry_set() {
    // Can generate: JSR is_carry_set; BCS label
    // Instead of: JSR is_carry_set; CMP #0; BNE label
}
```

The compiler tracks functions with `return STATUS.Flag` patterns and propagates this information to enable the optimization.

## Error Handling

### Non-Writable Flag Error

```rust
STATUS.Zero = true;  // Compile error!
```

Error message:
```
type error: Cannot write to STATUS.Zero
  This flag is set by CPU operations, not directly writable
  Writable flags: Carry, Irq, Decimal, XY16, A16
```

### Invalid Flag Name Error

```rust
STATUS.Invalid = true;  // Compile error!
```

Error message:
```
HIR error: Unknown STATUS flag 'Invalid'. Valid flags: Carry, Zero, Irq, Decimal, XY16, A16, Overflow, Negative
```

## Implementation Notes

### HIR Representation

STATUS flag access is represented by `HIRStatusFlagAccess` node containing:
- `flag_name`: String name of the flag
- `bit_position`: Bit position (0-7)
- `bit_mask`: Bit mask value

### MIR Instructions

Three MIR instruction types handle STATUS flags:
- `StatusFlagTest` - Test flag for conditional branching
- `StatusFlagSet` - Set or clear a flag
- `StatusFlagRead` - Read flag value into virtual register

### Type System

STATUS flag properties have type `bool`. Assignments require boolean values.

## See Also

- [Type System](type-system.md) - Type checking rules
- [Control Flow](control-flow.md) - Conditional branching
- [Code Generation](code-generation.md) - Assembly output details
