# Operators Design

## Overview

R65 provides operators and functions that clearly distinguish between hardware-supported operations (fast) and software-implemented operations (slow).

**Design Philosophy**:
- **Operators (`+`, `-`, `*`, `/`, etc.)** = Hardware instructions or simple instruction sequences (2-10 cycles)
- **Functions (`mul8()`/`mul16()`, `div8()`/`div16()`, `mod8()`, `shl8()`/`shl16()`, `shr8()`/`shr16()`)** = Software subroutines (20-200+ cycles)

All operations are **unchecked** - overflow, underflow, and division by zero are undefined behavior (matching hardware philosophy of no runtime checks).

---

## String Concatenation

### Addition with Strings: `+`

When the `+` operator is used with string operands, it performs compile-time string concatenation instead of arithmetic addition.

**Syntax**: `"string1" + "string2"`

**Type Rules**:
- If either operand is a string literal, performs string concatenation
- If both operands are non-strings, performs arithmetic addition  
- Non-string operands are converted to strings and concatenated

**Behavior**:
- Concatenates strings at compile time
- Supports mixing strings with integers (integer converted to string)
- Supports nested concatenation

**Examples**:
```rust
// Basic concatenation
static mut HELLO: [u8; 16] = "Hello, " + "World";
// Becomes: "Hello, World"

// Mixed concatenation
static mut COUNT: [u8; 16] = "Count: " + 42;
// Becomes: "Count: 42"

// Nested concatenation
static mut COMPLEX: [u8; 16] = "A" + "B" + "C";
// Becomes: "ABC"
```

**Performance**:
- Zero runtime cost - concatenation happens at compile time
- Same as using a single string literal

**Restrictions**:
- Only works in static initializers (constant expressions)
- Operands must be compile-time constants (string literals, integers, etc.)

---

## Cost Model at a Glance

### Fast Operations (use operators)
```rust
let x = a + b;       // Addition: CLC, ADC (2-4 cycles)
let x = a - b;       // Subtraction: SEC, SBC (2-4 cycles)
let x = a * 2;       // Multiply by 2: ASL (2 cycles)
let x = a / 4;       // Divide by 4: LSR, LSR (4 cycles)
let x = a << 3;      // Constant shift: ASL, ASL, ASL (6 cycles)
let x = a & 0x0F;    // Bitwise AND (2-4 cycles)
```

### Slow Operations (use functions)
```rust
let x = mul8(a, b);    // General multiply: JSR __mul (20-100+ cycles)
let x = div8(a, b);    // General divide: JSR __div (50-200+ cycles)
let x = mod8(a, b);    // Modulo: JSR __mod (50-200+ cycles)
let x = shl8(a, n);    // Variable shift: loop (6-50+ cycles)
let x = shr8(a, n);    // Variable shift: loop (6-50+ cycles)
```

---

## Arithmetic Operators

### Addition: `+`

**Syntax**: `a + b`

**Type Rules**:
- Both operands should be the same type
- Mixed-size integer operands trigger implicit promotion (u8 + u16 → u16)
- Result type matches the wider operand type

**Behavior**:
- Wrapping on overflow (no checks)
- Sets carry flag on overflow
- Examples:
  - `0xFF + 0x01` = `0x00` (wraps)
  - `0xFFFF + 0x0001` = `0x0000` (wraps)

**Assembly Mapping**:
```rust
// Register optimization
let result @ A = a + b;
// CLC
// ADC operand

// Memory operations
let c: u8 = VAR1 + VAR2;
// LDA VAR1
// CLC
// ADC VAR2
// STA c
```

**Optimizations**:
- `a + 1` may use INC instruction
- Constant folding: `5 + 3` = `8` at compile time

**Performance**: 2-4 cycles

---

### Subtraction: `-`

**Syntax**: `a - b`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type

**Behavior**:
- Wrapping on underflow (no checks)
- Sets carry flag (borrow) on underflow
- Examples:
  - `0x00 - 0x01` = `0xFF` (wraps)
  - `0x0000 - 0x0001` = `0xFFFF` (wraps)

**Assembly Mapping**:
```rust
let c = a - b;
// SEC
// SBC operand
```

**Optimizations**:
- `a - 1` may use DEC instruction
- Constant folding applies

**Performance**: 2-4 cycles

---

### Multiplication: `*` (Restricted)

**Syntax**: `a * n` where `n` is a power-of-2 constant from 1 to 256

**Type Rules**:
- Left operand: any integer type
- Right operand: **must be compile-time constant** power of 2 (1, 2, 4, 8, 16, 32, 64, 128, 256)
- Result type matches left operand type (truncated)

**Behavior**:
- Compiler error if multiplier is not a power of 2 from 1 to 256
- Implemented using shifts (very fast)
- Result truncated to operand size

**Assembly Mapping**:
```rust
let x = a * 1;   // No-op (compiler may optimize away)
let x = a * 2;   // ASL (shift left 1)
let x = a * 4;   // ASL, ASL (shift left 2)
let x = a * 8;   // ASL, ASL, ASL (shift left 3)
let x = a * 64;  // 6x ASL (shift left 6)

// These are ERRORS:
let x = a * 3;   // ERROR: Use mul8(a, 3)
let x = a * b;   // ERROR: Use mul8(a, b) for variable multiply
```

**Performance**: 2-6 cycles

---

### Division: `/` (Restricted)

**Syntax**: `a / n` where `n` is a power-of-2 constant from 1 to 256

**Type Rules**:
- Left operand: any integer type
- Right operand: **must be compile-time constant** power of 2 (1, 2, 4, 8, 16, 32, 64, 128, 256)
- Result type matches left operand type

**Behavior**:
- Compiler error if divisor is not a power of 2 from 1 to 256
- Implemented using logical shifts (LSR) for both signed and unsigned
- **Note**: Signed division does not currently round toward zero — it uses the same LSR as unsigned

**Assembly Mapping**:
```rust
let x: u8 = a / 1;   // No-op
let x: u8 = a / 2;   // LSR (shift right 1)
let x: u8 = a / 4;   // LSR, LSR (shift right 2)
let x: u8 = a / 8;   // LSR, LSR, LSR (shift right 3)
let x: u8 = a / 64;  // 6x LSR (shift right 6)

// These are ERRORS:
let x = a / 3;   // ERROR: Use div8(a, 3)
let x = a / b;   // ERROR: Use div8(a, b) for variable division
```

**Performance**: 2-6 cycles

---

## Multiplication Functions: `mul8()` / `mul16()`

**Syntax**: `mul8(a, b)` or `mul16(a, b)`

**Type Rules**:
- Both operands must be the same integer type
- Result type matches operand type (truncated)

**Behavior**:
- Full multiplication with subroutine call
- Result truncated to operand size (high byte/word discarded)
- Wrapping on overflow (no checks)
- With `--cfg snes`, 8-bit multiplication uses the SNES hardware multiplier (faster)

**Assembly Mapping**:
```rust
let x: u8 = mul8(a, b);
// LDA a
// LDX b
// JSR __mul_u8
// (result in A)

let x: u16 = mul16(a, b);
// [Load a, b into appropriate registers/memory]
// JSR __mul_u16
// (result in A or memory)
```

**Performance**: 20-100+ cycles (varies by implementation; hardware mul faster on SNES)

**Examples**:
```rust
let area: u8 = mul8(width, height);
let scaled: u8 = mul8(value, 3);       // Not a power of 2
let offset: u16 = mul16(y as u16, 256 as u16);
```

---

## Division Functions: `div8()` / `div16()`

**Syntax**: `div8(a, b)` or `div16(a, b)`

**Type Rules**:
- Both operands must be the same integer type
- Result type matches operand type

**Behavior**:
- Full division with subroutine call
- Division by zero is **undefined behavior** (no check)

**Assembly Mapping**:
```rust
let x: u8 = div8(a, b);
// LDA a
// LDX b
// JSR __div_u8
// (result in A)
```

**Performance**: 50-200+ cycles

**Examples**:
```rust
let avg: u8 = div8(sum, count);
let tiles: u8 = div8(pixels, 7);       // Not a power of 2
```

---

## Modulo Function: `mod8()`

**Syntax**: `mod8(a, b)`

**Type Rules**:
- Both operands must be the same integer type
- Result type matches operand type

**Behavior**:
- Returns remainder after division
- Modulo by zero is **undefined behavior** (no check)

**Assembly Mapping**:
```rust
let x: u8 = mod8(a, b);
// LDA a
// LDX b
// JSR __mod_u8
// (result in A)
```

**Performance**: 50-200+ cycles (often implemented with division)

**Examples**:
```rust
let remainder: u8 = mod8(distance, tile_size);

// Prefer this for power of 2:
let wrapped = index & 0xFF;  // Same as mod8(index, 256)
```

**Note**: The `%` operator is parsed but **not supported at code generation** — use the `mod8()` function instead.

---

## Bitwise Operators

### Bitwise AND: `&`

**Syntax**: `a & b`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type

**Assembly Mapping**:
```rust
let c = a & b;
// LDA a
// AND b
```

**Common Use Cases**:
```rust
let low_nibble = value & 0x0F;     // Mask low 4 bits
let is_set = flags & 0x80;         // Test bit 7
let wrapped = index & 0xFF;        // Modulo 256 (for u8)
let aligned = addr & 0xFFF0;       // Align to 16 bytes
```

**Performance**: 2-4 cycles

---

### Bitwise OR: `|`

**Syntax**: `a | b`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type

**Assembly Mapping**:
```rust
let c = a | b;
// LDA a
// ORA b
```

**Common Use Cases**:
```rust
let flags = flags | 0x01;          // Set bit 0
let combined = high | low;         // Combine bit fields
```

**Performance**: 2-4 cycles

---

### Bitwise XOR: `^`

**Syntax**: `a ^ b`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type

**Assembly Mapping**:
```rust
let c = a ^ b;
// LDA a
// EOR b
```

**Common Use Cases**:
```rust
let inverted = value ^ 0xFF;       // Flip all bits (u8)
let toggled = flags ^ 0x80;        // Toggle bit 7
let zero = x ^ x;                  // Always zero (identity)
```

**Performance**: 2-4 cycles

---

### Left Shift: `<<` (Constant Only)

**Syntax**: `a << n` where `n` is a compile-time constant

**Type Rules**:
- Left operand: any integer type
- Right operand: **must be compile-time constant**
- Result type matches left operand

**Behavior**:
- Shifts left, fills with zeros
- Shift by ≥ bit width is **undefined behavior**
- Each shift multiplies by 2 (with wrapping)

**Assembly Mapping**:
```rust
let x = a << 1;   // ASL
let x = a << 2;   // ASL, ASL
let x = a << 3;   // ASL, ASL, ASL
let x = a << 4;   // ASL, ASL, ASL, ASL

// ERROR: variable shift
let x = a << n;   // ERROR: Use shl8(a, n) for variable shifts
```

**Performance**: 2 cycles per shift

---

### Right Shift: `>>` (Constant Only)

**Syntax**: `a >> n` where `n` is a compile-time constant

**Type Rules**:
- Left operand: any integer type
- Right operand: **must be compile-time constant**
- Result type matches left operand

**Behavior**:
- Uses logical shift (LSR) for both signed and unsigned types (fills with 0)
- **Note**: Arithmetic right shift (sign-preserving) is not currently implemented
- Shift by ≥ bit width is **undefined behavior**

**Assembly Mapping**:
```rust
let x: u8 = a >> 1;   // LSR
let x: u8 = a >> 2;   // LSR, LSR

// ERROR: variable shift
let x = a >> n;   // ERROR: Use shr8(a, n) for variable shifts
```

**Performance**: 2 cycles per shift

---

## Shift Functions: `shl8()` / `shl16()` and `shr8()` / `shr16()`

### Left Shift Function: `shl8()` / `shl16()`

**Syntax**: `shl8(n, shift_amount)` or `shl16(n, shift_amount)`

**Type Rules**:
- First operand: any integer type
- Second operand: `u8` (can be variable or constant)
- Result type matches first operand

**Behavior**:
- Shifts left by variable amount
- Implemented as loop (expensive)
- Shift by ≥ bit width is **undefined behavior**

**Assembly Mapping**:
```rust
let x = shl8(a, count);
// LDA a
// LDX count
// BEQ done
// loop:
// ASL A
// DEX
// BNE loop
// done:
```

**Performance**: ~8 cycles + (6 cycles × shift_amount)

**Examples**:
```rust
let shifted = shl8(value, 3);        // OK but prefer: value << 3
let dynamic = shl8(base, bit_pos);   // Variable shift - required
```

---

### Right Shift Function: `shr8()` / `shr16()`

**Syntax**: `shr8(n, shift_amount)` or `shr16(n, shift_amount)`

**Type Rules**:
- First operand: any integer type
- Second operand: `u8` (can be variable or constant)
- Result type matches first operand

**Behavior**:
- **Unsigned types**: Logical shift (fill with 0)
- **Signed types**: Arithmetic shift (fill with sign bit)
- Implemented as loop (expensive)
- Shift by ≥ bit width is **undefined behavior**

**Assembly Mapping**:
```rust
// Unsigned
let x: u8 = shr8(a, count);
// LDA a
// LDX count
// BEQ done
// loop:
// LSR A
// DEX
// BNE loop
// done:

// Signed (more complex)
let x: i8 = shri8(a, count);
// [Sign-preserving loop]
```

**Performance**: ~8 cycles + (6-8 cycles × shift_amount)

**Examples**:
```rust
let shifted = shr8(value, 2);        // OK but prefer: value >> 2
let dynamic = shr8(bits, offset);    // Variable shift - required
```

---

## Comparison Operators

### Equality: `==`

**Syntax**: `a == b`

**Type Rules**:
- Both operands must be the same type
- Result type: `bool`

**Assembly Mapping**:
```rust
if a == b { }
// LDA a
// CMP b
// BNE skip_block
// [block code]
// skip_block:
```

**Performance**: 4-6 cycles

---

### Inequality: `!=`

**Syntax**: `a != b`

**Type Rules**:
- Both operands must be the same type
- Result type: `bool`

**Assembly Mapping**:
```rust
if a != b { }
// LDA a
// CMP b
// BEQ skip_block
// [block code]
// skip_block:
```

**Performance**: 4-6 cycles

---

### Less Than: `<`

**Syntax**: `a < b`

**Type Rules**:
- Both operands must be the same type
- Signed vs unsigned matters: `i8 < i8` vs `u8 < u8`
- Result type: `bool`

**Assembly Mapping**:
```rust
// Unsigned comparison (simple)
if a < b { }
// LDA a
// CMP b
// BCS skip_block  // Branch if A >= b (carry set)
// [block code]
// skip_block:

// Signed comparison (complex - requires overflow handling)
if (a: i8) < (b: i8) { }
// [More complex sequence to handle sign]
// OR: JSR __cmp_signed_less
```

**Performance**:
- Unsigned: 4-6 cycles
- Signed: 8-15 cycles or subroutine call

---

### Less Than or Equal: `<=`

**Syntax**: `a <= b`

**Assembly Mapping**:
- Implemented as `!(a > b)` or `(a < b) || (a == b)`

**Performance**: Similar to `<`

---

### Greater Than: `>`

**Syntax**: `a > b`

**Assembly Mapping**:
- Implemented as `b < a` (swap operands)

**Performance**: Similar to `<`

---

### Greater Than or Equal: `>=`

**Syntax**: `a >= b`

**Assembly Mapping**:
```rust
// Unsigned (simple)
if a >= b { }
// LDA a
// CMP b
// BCC skip_block  // Branch if A < b (carry clear)
// [block code]
// skip_block:
```

**Performance**: Similar to `<`

---

## Logical Operators

### Logical AND: `&&`

**Syntax**: `a && b`

**Type Rules**:
- Both operands must be `bool`
- Result type: `bool`
- **Short-circuit evaluation**: If `a` is false, `b` is not evaluated

**Assembly Mapping**:
```rust
if a && b { }
// LDA a
// BEQ skip_block    // If a is false, skip
// LDA b
// BEQ skip_block    // If b is false, skip
// [block code]
// skip_block:
```

**Performance**: 4-8 cycles (depending on short-circuit)

---

### Logical OR: `||`

**Syntax**: `a || b`

**Type Rules**:
- Both operands must be `bool`
- Result type: `bool`
- **Short-circuit evaluation**: If `a` is true, `b` is not evaluated

**Assembly Mapping**:
```rust
if a || b { }
// LDA a
// BNE execute_block  // If a is true, execute
// LDA b
// BEQ skip_block     // If b is false, skip
// execute_block:
// [block code]
// skip_block:
```

**Performance**: 4-8 cycles (depending on short-circuit)

---

### Logical NOT: `!`

**Syntax**: `!a`

**Type Rules**:
- Operand must be `bool`
- Result type: `bool`

**Assembly Mapping**:
```rust
let x = !a;
// LDA a
// EOR #$01  // Flip bit 0 (assuming normalized bool: 0 or 1)
```

**Performance**: 2-4 cycles

---

## Unary Operators

### Unary Minus: `-`

**Syntax**: `-a`

**Type Rules**:
- Operand: any integer type
- Result type matches operand type

**Behavior**:
- Two's complement negation
- `-0` = `0`
- Signed overflow: `-128i8` = `-128i8` (wraps)

**Assembly Mapping**:
```rust
let x = -a;
// Method 1: EOR + INC
// LDA a
// EOR #$FF  // Invert bits (or #$FFFF for u16)
// INC A     // Add 1

// Method 2: SEC + SBC (alternative)
// SEC
// LDA #$00
// SBC a
```

**Performance**: 4-6 cycles

---

### Bitwise NOT: `~`

**Syntax**: `~a`

**Type Rules**:
- Operand: any integer type
- Result type matches operand type

**Assembly Mapping**:
```rust
let x = ~a;
// LDA a
// EOR #$FF    // Flip all bits (u8)
// EOR #$FFFF  // Flip all bits (u16)
```

**Performance**: 2-4 cycles

---

## Compound Assignment Operators

All arithmetic and bitwise operators support compound assignment:

```rust
a += b;   // a = a + b
a -= b;   // a = a - b
a &= b;   // a = a & b
a |= b;   // a = a | b
a ^= b;   // a = a ^ b
a <<= n;  // a = a << n (n must be constant)
a >>= n;  // a = a >> n (n must be constant)
```

**Restricted** (inherit base operator restrictions):
```rust
a *= 2;   // OK: desugars to a = a * 2 (power of 2)
a *= b;   // ERROR: desugars to a = a * b (variable multiply not allowed)
a /= 4;   // OK: desugars to a = a / 4 (power of 2)
a /= b;   // ERROR: desugars to a = a / b (variable divide not allowed)
a %= b;   // ERROR: % operator not supported at codegen — use a = mod8(a, b)
```

**Optimization**: Direct memory operations when beneficial:
```rust
A += 1;           // INC A (2 cycles)
COUNTER += 1;     // INC COUNTER (5-6 cycles)
FLAGS |= 0x80;    // LDA FLAGS, ORA #$80, STA FLAGS
```

---

## Type Compatibility

### Implicit Integer Promotion

When binary operators have mixed-size integer operands, the smaller type is implicitly promoted to match the larger:

```rust
let a: u8 = 10;
let b: u16 = 20;
let c: u16 = a + b;  // OK: a implicitly promoted to u16
```

Explicit casts can also be used:
```rust
let c: u16 = (a as u16) + b;     // Explicit cast (same result)
```

### Same-Size Wrapping

When both operands are the same size, results wrap within that size:

```rust
let x: u8 = 250;
let y: u8 = 10;
let z = x + y;      // z: u8 = 4 (wraps), NOT u16 = 260
let z = mul8(x, y);  // z: u8 = 196 (2500 truncated to u8)
```

---

## Overflow and Underflow Behavior

### Philosophy: No Runtime Checks

All operations **wrap on overflow/underflow with no checks**.

```rust
let x: u8 = 255;
let y = x + 1;         // y = 0 (wraps)

let x: i8 = 127;
let y = x + 1;         // y = -128 (wraps)

let x: u8 = 0;
let y = x - 1;         // y = 255 (wraps)

let x: u8 = 200;
let y = mul8(x, 2);     // y = 144 (400 wraps to 144)

let x: u8 = 255;
let y = x * 2;         // y = 254 (510 truncated to u8)
```

### Division by Zero

Division by zero is **undefined behavior** (no runtime check):

```rust
let x = div8(10, 0);    // UNDEFINED BEHAVIOR
let x = 10 / 0;        // ERROR: constant division by zero (compile-time)
```

---

## Operator Precedence

Standard C/Rust precedence (highest to lowest):

1. **Unary**: `!`, `~`, `-` (unary)
2. **Multiplicative**: `*` (restricted), `/` (restricted), function calls: `mul8()`, `div8()`, `mod8()`
3. **Additive**: `+`, `-`
4. **Shift**: `<<`, `>>` (constant only), `shl8()`, `shr8()`
5. **Comparison**: `<`, `<=`, `>`, `>=`
6. **Equality**: `==`, `!=`
7. **Bitwise AND**: `&`
8. **Bitwise XOR**: `^`
9. **Bitwise OR**: `|`
10. **Logical AND**: `&&`
11. **Logical OR**: `||`
12. **Assignment**: `=`, `+=`, `-=`, etc.

**Parentheses** override precedence: `(a + b) * c`

**Note**: Function calls (`mul8()`, `div8()`, `mod8()`, etc.) have high precedence like all function calls.

---

## Performance Summary

### Fast Operations (2-10 cycles)
```rust
a + b, a - b              // 2-4 cycles
a & b, a | b, a ^ b       // 2-4 cycles
a << n, a >> n            // 2-8 cycles (n constant)
a * 2, a * 4, a * 128     // 2-14 cycles (power of 2, up to 256)
a / 2, a / 4, a / 128     // 2-14 cycles (unsigned power of 2, up to 256)
a == b, a != b            // 4-6 cycles
a < b, a >= b (unsigned)  // 4-6 cycles
-a, ~a, !a                // 2-6 cycles
a += 1, a -= 1            // 2-6 cycles (INC/DEC)
```

### Slow Operations (20-200+ cycles)
```rust
mul8(a, b)                  // 20-100+ cycles
div8(a, b)                  // 50-200+ cycles
mod8(a, b)                  // 50-200+ cycles
shl8(a, n), shr8(a, n)     // 8 + (6 × n) cycles
rotate_left(a, n)           // Similar to shifts
rotate_right(a, n)          // Similar to shifts
a < b (signed)              // 8-15 cycles or subroutine
```

---

## Optimization Guidelines

1. **Use operators for constants**: `a * 8` instead of `mul8(a, 8)`
2. **Prefer shifts**: `a << 3` is equivalent to `a * 8`
3. **Use AND for power-of-2 modulo**: `a & 0xFF` instead of `mod8(a, 256)`
4. **Avoid division in loops**: Pre-compute or use lookup tables
5. **Favor 8-bit operations**: Faster than 16-bit in m8 mode
6. **Use INC/DEC for ±1**: `a += 1` optimizes to INC
7. **Constant folding**: Let compiler optimize `5 + 3` to `8`
8. **Strength reduction**: Compiler may optimize `mul8(a, 2)` to `a << 1` (use `a * 2` for this directly)

---

## Examples

### Sprite Position Update
```rust
#[zeropage]
static mut SPRITE_X: u8;
#[zeropage]
static mut SPRITE_Y: u8;

fn update_sprite(velocity: u8) {
    // Fast: increment
    SPRITE_X += velocity;

    // Fast: power-of-2 division (tile index)
    let tile_x = SPRITE_X / 8;  // Compiles to LSR, LSR, LSR

    // Fast: wrapping (u8 naturally wraps at 256)
    SPRITE_X = SPRITE_X + 1;  // Auto-wraps to 0 after 255
}
```

### Tilemap Offset Calculation
```rust
fn get_tile_offset(x: u8, y: u8) -> u16 {
    // Multiply y by 32 (tiles per row)
    // 32 = power of 2, so * 32 or << 5 both work
    let row_offset: u16 = (y as u16) * 32;  // 5x ASL = multiply by 32

    // Add x coordinate
    let offset = row_offset + (x as u16);
    return offset;
}
```

### Bit Manipulation
```rust
fn process_flags(flags @ A: u8) -> u8 {
    // Clear bit 0, set bit 7
    let result @ A = (flags & ~0x01) | 0x80;
    return result;
}
```

### Division with Non-Power-of-2
```rust
fn calculate_average(sum: u16, count: u8) -> u8 {
    // count might not be power of 2, must use div8()
    let avg = div8(sum as u8, count);  // Expensive!
    return avg;
}
```

---

## Runtime Library Functions

The compiler provides these runtime functions:

### Multiplication
- `__mul_u8(a: u8, b: u8) -> u8`
- `__mul_i8(a: i8, b: i8) -> i8`
- `__mul_u16(a: u16, b: u16) -> u16`
- `__mul_i16(a: i16, b: i16) -> i16`

### Division
- `__div_u8(a: u8, b: u8) -> u8`
- `__div_i8(a: i8, b: i8) -> i8`
- `__div_u16(a: u16, b: u16) -> u16`
- `__div_i16(a: i16, b: i16) -> i16`

### Modulo
- `__mod_u8(a: u8, b: u8) -> u8`
- `__mod_i8(a: i8, b: i8) -> i8`
- `__mod_u16(a: u16, b: u16) -> u16`
- `__mod_i16(a: i16, b: i16) -> i16`

### Variable Shifts
- `__shl_u8(value: u8, shift: u8) -> u8`
- `__shl_u16(value: u16, shift: u8) -> u16`
- `__shr_u8(value: u8, shift: u8) -> u8`
- `__shr_i8(value: i8, shift: u8) -> i8`
- `__shr_u16(value: u16, shift: u8) -> u16`
- `__shr_i16(value: i16, shift: u8) -> i16`

---

## Compiler Error Messages

### Restricted Operators
```rust
let x = a * 3;
// ERROR: Multiply operator (*) requires a power-of-2 constant operand (1 to 256)
// HELP: Use mul8(a, 3) for general multiplication

let x = a / 5;
// ERROR: Divide operator (/) requires a power-of-2 constant divisor (1 to 256)
// HELP: Use div8(a, 5) for general division

let x = a << count;
// ERROR: Left shift operator (<<) requires constant shift amount
// HELP: Use shl8(a, count) for variable shifts

let x = a * b;
// ERROR: Multiply operator (*) requires constant right operand
// HELP: Use mul8(a, b) for variable multiplication
```

### Type Mismatches
```rust
let x = mul8(a: u8, b: u16);
// ERROR: mul8() requires both operands to be the same type
// HELP: Cast to same type: mul16(a as u16, b)
```

---

## Increment/Decrement Operators

R65 supports increment (`++`) and decrement (`--`) operators as statement-only syntax:

```rust
x++;        // Desugars to: x += 1 (then to: x = x + 1)
counter--;  // Desugars to: counter -= 1 (then to: counter = counter - 1)
```

**Design decisions:**
- **Statement-only**: Not expressions (no return value)
- **Postfix form only**: `x++`, not `++x`
- **Works with all lvalue types**: variables, registers, arrays, struct fields
- **Zero overhead**: Desugars to compound assignment in parser

**Hardware register optimization:**

When incrementing or decrementing hardware registers (A, X, Y), the compiler generates optimal single-cycle instructions:

```rust
X++;   // Generates: INX     (2 cycles)
Y++;   // Generates: INY     (2 cycles)
A++;   // Generates: INC A   (2 cycles)

X--;   // Generates: DEX     (2 cycles)
Y--;   // Generates: DEY     (2 cycles)
A--;   // Generates: DEC A   (2 cycles)
```

Without optimization, `X++` would generate the verbose sequence `TXA; CLC; ADC #1; TAX` (8+ cycles). The compiler automatically detects the pattern `reg = reg ± 1` and emits the efficient instruction.

**Examples:**
```rust
// Variables
counter++;

// Registers (optimized)
A++;
X--;

// Array elements
buffer[i]++;

// Struct fields
player.health--;
```

**Implementation:** `x++` desugars to `x += 1` in the parser (which then desugars to `x = x + 1` in HIR). Generates same code as manual increment.

### Not Included

**Logical compound assignments** (`&&=`, `||=`):
- Rarely useful in systems programming
- Can be added if use case emerges

**Prefix increment/decrement** (`++x`, `--x`):
- Only postfix form currently supported
- Prefix form unnecessary since operators don't return values

---

## Future Enhancements

### Checked Operations
```rust
// Possible future syntax:
let (result, overflow) = checked_mul(a, b);
let result = saturating_add(a, b);  // Clamp to max
```

### Wide Multiply/Divide
```rust
// Return full result:
let result: u16 = mul_wide(a: u8, b: u8);  // Full 16-bit result
let (quotient, remainder) = divmod(a, b);  // Both results
```

### Intrinsic STATUS Flag Access
```rust
// Read processor status flags:
let carry = STATUS & CARRY_FLAG;
let overflow = STATUS & OVERFLOW_FLAG;
```

---

**STATUS**: Implemented
**Last Updated**: 2026-02-10
**Not Yet Implemented**: `%` operator at codegen (use `mod8()` function), signed arithmetic right shift (uses logical shift for both)
