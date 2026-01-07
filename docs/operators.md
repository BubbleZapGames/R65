# Operators Design

## Overview

R65 provides operators and functions that clearly distinguish between hardware-supported operations (fast) and software-implemented operations (slow).

**Design Philosophy**:
- **Operators (`+`, `-`, `*`, `/`, etc.)** = Hardware instructions or simple instruction sequences (2-10 cycles)
- **Functions (`mul()`, `div()`, `shl()`, etc.)** = Software subroutines (20-200+ cycles)

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
let x = mul(a, b);   // General multiply: JSR __mul (20-100+ cycles)
let x = div(a, b);   // General divide: JSR __div (50-200+ cycles)
let x = mod(a, b);   // Modulo: JSR __mod (50-200+ cycles)
let x = shl(a, n);   // Variable shift: loop (6-50+ cycles)
```

---

## Arithmetic Operators

### Addition: `+`

**Syntax**: `a + b`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type
- No implicit type promotion

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

**Syntax**: `a * n` where `n` is a constant `1`, `2`, `4`, or `8`

**Type Rules**:
- Left operand: any integer type
- Right operand: **must be compile-time constant** 1, 2, 4, or 8
- Result type matches left operand type (truncated)

**Behavior**:
- Compiler error if multiplier is not 1, 2, 4, or 8
- Implemented using shifts (very fast)
- Result truncated to operand size

**Assembly Mapping**:
```rust
let x = a * 1;   // No-op (compiler may optimize away)
let x = a * 2;   // ASL (shift left 1)
let x = a * 4;   // ASL, ASL (shift left 2)
let x = a * 8;   // ASL, ASL, ASL (shift left 3)

// These are ERRORS:
let x = a * 3;   // ERROR: Use mul(a, 3)
let x = a * 16;  // ERROR: Use mul(a, 16) or a << 4
let x = a * b;   // ERROR: Use mul(a, b) for variable multiply
```

**Performance**: 2-6 cycles

---

### Division: `/` (Restricted)

**Syntax**: `a / n` where `n` is a constant `1`, `2`, `4`, or `8`

**Type Rules**:
- Left operand: any integer type
- Right operand: **must be compile-time constant** 1, 2, 4, or 8
- Result type matches left operand type
- Signed vs unsigned matters: `i8 / 2` vs `u8 / 2`

**Behavior**:
- Compiler error if divisor is not 1, 2, 4, or 8
- Unsigned: implemented using logical shifts (LSR)
- Signed: more complex (check sign, adjust, shift)
- Truncates toward zero for signed integers

**Assembly Mapping**:
```rust
// Unsigned division
let x: u8 = a / 1;   // No-op
let x: u8 = a / 2;   // LSR (shift right 1)
let x: u8 = a / 4;   // LSR, LSR (shift right 2)
let x: u8 = a / 8;   // LSR, LSR, LSR (shift right 3)

// Signed division (more complex)
let x: i8 = a / 2;
// Check sign bit
// CMP #$80
// BCC positive
// INC A          // Adjust for negative (round toward zero)
// positive:
// LSR            // Shift right

// These are ERRORS:
let x = a / 3;   // ERROR: Use div(a, 3)
let x = a / 16;  // ERROR: Use div(a, 16) or a >> 4
let x = a / b;   // ERROR: Use div(a, b) for variable division
```

**Performance**: 2-8 cycles (unsigned faster than signed)

---

## Multiplication Function: `mul()`

**Syntax**: `mul(a, b)`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type (truncated)
- Available for: `u8`, `i8`, `u16`, `i16`

**Behavior**:
- Full multiplication with subroutine call
- Result truncated to operand size:
  - `mul(a: u8, b: u8)` → `u8` (high byte discarded)
  - `mul(a: u16, b: u16)` → `u16` (high word discarded)
- Wrapping on overflow (no checks)

**Assembly Mapping**:
```rust
let x: u8 = mul(a, b);
// LDA a
// LDX b
// JSR __mul_u8
// (result in A)

let x: u16 = mul(a, b);
// [Load a, b into appropriate registers/memory]
// JSR __mul_u16
// (result in A or memory)
```

**Optimizations**:
- Compiler may detect constant powers of 2 and use shifts
- `mul(a, 8)` → `a << 3` (but explicit `a * 8` is preferred)
- Small constants may use repeated addition or lookup tables

**Performance**: 20-100+ cycles (varies by implementation)

**Examples**:
```rust
let area = mul(width, height);
let scaled = mul(value, 3);      // Not a power of 2
let offset = mul(y, 256);        // Could optimize to shift
```

---

## Division Function: `div()`

**Syntax**: `div(a, b)`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type
- Signed vs unsigned matters: `div(i8, i8)` vs `div(u8, u8)`
- Available for: `u8`, `i8`, `u16`, `i16`

**Behavior**:
- Full division with subroutine call
- Division by zero is **undefined behavior** (no check)
- Truncates toward zero for signed integers

**Assembly Mapping**:
```rust
let x: u8 = div(a, b);
// LDA a
// LDX b
// JSR __div_u8
// (result in A)

let x: i8 = div(a, b);
// [Handle sign complexities]
// JSR __div_i8
```

**Optimizations**:
- Compiler may detect constant powers of 2 and suggest `a / n` instead
- Warning: `div(a, 8)` → "Use a / 8 for better performance"

**Performance**: 50-200+ cycles

**Examples**:
```rust
let avg = div(sum, count);
let tiles = div(pixels, 7);      // Not a power of 2
```

---

## Modulo Function: `mod()`

**Syntax**: `mod(a, b)`

**Type Rules**:
- Both operands must be the same type
- Result type matches operand type
- Signed vs unsigned matters
- Available for: `u8`, `i8`, `u16`, `i16`

**Behavior**:
- Returns remainder after division
- Modulo by zero is **undefined behavior** (no check)
- Sign of result matches dividend for signed types

**Assembly Mapping**:
```rust
let x: u8 = mod(a, b);
// LDA a
// LDX b
// JSR __mod_u8
// (result in A)
```

**Optimizations**:
- Powers of 2: `mod(a, 256)` → `a & 0xFF`
- Compiler suggests: "Use a & 0x07 instead of mod(a, 8)"

**Performance**: 50-200+ cycles (often implemented with division)

**Examples**:
```rust
let remainder = mod(distance, tile_size);
let wrapped = mod(index, buffer_size);

// Prefer this for power of 2:
let wrapped = index & 0xFF;  // Same as mod(index, 256)
```

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
let x = a << n;   // ERROR: Use shl(a, n) for variable shifts
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
- **Unsigned types** (`u8`, `u16`): Logical shift (fill with 0)
- **Signed types** (`i8`, `i16`): Arithmetic shift (fill with sign bit)
- Shift by ≥ bit width is **undefined behavior**

**Assembly Mapping**:
```rust
// Unsigned (logical shift)
let x: u8 = a >> 1;   // LSR
let x: u8 = a >> 2;   // LSR, LSR

// Signed (arithmetic shift - more complex)
let x: i8 = a >> 1;
// CMP #$80        // Test sign
// ROR             // Rotate with carry

// ERROR: variable shift
let x = a >> n;   // ERROR: Use shr(a, n) for variable shifts
```

**Performance**: 2-4 cycles per shift (signed slightly slower)

---

## Shift Functions: `shl()` and `shr()`

### Left Shift Function: `shl()`

**Syntax**: `shl(n, shift_amount)`

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
let x = shl(a, count);
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
let shifted = shl(value, 3);        // OK but prefer: value << 3
let dynamic = shl(base, bit_pos);   // Variable shift - required
```

---

### Right Shift Function: `shr()`

**Syntax**: `shr(n, shift_amount)`

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
let x: u8 = shr(a, count);
// LDA a
// LDX count
// BEQ done
// loop:
// LSR A
// DEX
// BNE loop
// done:

// Signed (more complex)
let x: i8 = shr(a, count);
// [Sign-preserving loop]
```

**Performance**: ~8 cycles + (6-8 cycles × shift_amount)

**Examples**:
```rust
let shifted = shr(value, 2);        // OK but prefer: value >> 2
let dynamic = shr(bits, offset);    // Variable shift - required
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

**Not allowed** (use explicit assignment):
```rust
a *= b;   // ERROR: Use a = mul(a, b)
a /= b;   // ERROR: Use a = div(a, b)
a %= b;   // ERROR: Use a = mod(a, b)
```

**Optimization**: Direct memory operations when beneficial:
```rust
A += 1;           // INC A (2 cycles)
COUNTER += 1;     // INC COUNTER (5-6 cycles)
FLAGS |= 0x80;    // LDA FLAGS, ORA #$80, STA FLAGS
```

---

## Type Compatibility

### Same-Type Requirement

**All binary operators require both operands to be the same type.**

```rust
let a: u8 = 10;
let b: u16 = 20;
let c = a + b;        // ERROR: type mismatch
let c = mul(a, b);    // ERROR: type mismatch
```

### Explicit Casting Required

```rust
let a: u8 = 10;
let b: u16 = 20;
let c: u16 = (a as u16) + b;     // OK
let c: u16 = mul(a as u16, b);   // OK
```

### No Integer Promotion

Unlike C, there is **no automatic integer promotion**:

```rust
let x: u8 = 250;
let y: u8 = 10;
let z = x + y;      // z: u8 = 4 (wraps), NOT u16 = 260
let z = mul(x, y);  // z: u8 = 196 (2500 truncated to u8)
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
let y = mul(x, 2);     // y = 144 (400 wraps to 144)

let x: u8 = 255;
let y = x * 2;         // y = 254 (510 truncated to u8)
```

### Division by Zero

Division by zero is **undefined behavior** (no runtime check):

```rust
let x = div(10, 0);    // UNDEFINED BEHAVIOR
let x = 10 / 0;        // ERROR: constant division by zero (compile-time)
```

---

## Operator Precedence

Standard C/Rust precedence (highest to lowest):

1. **Unary**: `!`, `~`, `-` (unary)
2. **Multiplicative**: `*` (restricted), function calls: `mul()`, `div()`, `mod()`
3. **Additive**: `+`, `-`
4. **Shift**: `<<`, `>>` (constant only), `shl()`, `shr()`
5. **Comparison**: `<`, `<=`, `>`, `>=`
6. **Equality**: `==`, `!=`
7. **Bitwise AND**: `&`
8. **Bitwise XOR**: `^`
9. **Bitwise OR**: `|`
10. **Logical AND**: `&&`
11. **Logical OR**: `||`
12. **Assignment**: `=`, `+=`, `-=`, etc.

**Parentheses** override precedence: `(a + b) * c`

**Note**: Function calls (`mul()`, `div()`, etc.) have high precedence like all function calls.

---

## Performance Summary

### Fast Operations (2-10 cycles)
```rust
a + b, a - b              // 2-4 cycles
a & b, a | b, a ^ b       // 2-4 cycles
a << n, a >> n            // 2-8 cycles (n constant)
a * 2, a * 4, a * 8       // 2-6 cycles (power of 2)
a / 2, a / 4, a / 8       // 2-8 cycles (unsigned power of 2)
a == b, a != b            // 4-6 cycles
a < b, a >= b (unsigned)  // 4-6 cycles
-a, ~a, !a                // 2-6 cycles
a += 1, a -= 1            // 2-6 cycles (INC/DEC)
```

### Slow Operations (20-200+ cycles)
```rust
mul(a, b)                 // 20-100+ cycles
div(a, b)                 // 50-200+ cycles
mod(a, b)                 // 50-200+ cycles
shl(a, n), shr(a, n)      // 8 + (6 × n) cycles
a < b (signed)            // 8-15 cycles or subroutine
```

---

## Optimization Guidelines

1. **Use operators for constants**: `a * 8` instead of `mul(a, 8)`
2. **Prefer shifts**: `a << 3` faster than `mul(a, 8)`
3. **Use AND for power-of-2 modulo**: `a & 0xFF` instead of `mod(a, 256)`
4. **Avoid division in loops**: Pre-compute or use lookup tables
5. **Favor 8-bit operations**: Faster than 16-bit in m8 mode
6. **Use INC/DEC for ±1**: `a += 1` optimizes to INC
7. **Constant folding**: Let compiler optimize `5 + 3` to `8`
8. **Strength reduction**: Compiler may optimize `mul(a, 2)` to `a << 1`

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
    // 32 = power of 2, but not 1/2/4/8, so use mul() or shift
    let row_offset: u16 = (y as u16) << 5;  // Shift by 5 = multiply by 32

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
    // count might not be power of 2, must use div()
    let avg = div(sum as u8, count);  // Expensive!
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
// ERROR: Multiply operator (*) only supports constants 1, 2, 4, or 8
// HELP: Use mul(a, 3) for general multiplication

let x = a / 16;
// ERROR: Divide operator (/) only supports constants 1, 2, 4, or 8
// HELP: Use div(a, 16) or a >> 4 for division by 16

let x = a << count;
// ERROR: Left shift operator (<<) requires constant shift amount
// HELP: Use shl(a, count) for variable shifts

let x = a * b;
// ERROR: Multiply operator (*) requires constant right operand
// HELP: Use mul(a, b) for variable multiplication
```

### Type Mismatches
```rust
let x = (a: u8) + (b: u16);
// ERROR: Cannot add u8 and u16
// HELP: Cast to same type: (a as u16) + b

let x = mul(a: u8, b: u16);
// ERROR: mul() requires both operands to be the same type
// HELP: Cast to same type: mul(a as u16, b)
```

---

## Compound Assignment Operators

### Overview

R65 supports compound assignment operators that combine a binary operation with assignment. These are **syntactic sugar** that desugar during HIR lowering:

```rust
x += 5;  // Desugars to: x = x + 5
```

**No performance difference** - compound assignments compile to the same code as manual expansion.

### Supported Operators

| Operator | Example | Desugars To | Category |
|----------|---------|-------------|----------|
| `+=` | `x += 5` | `x = x + 5` | Arithmetic |
| `-=` | `x -= 3` | `x = x - 3` | Arithmetic |
| `*=` | `x *= 2` | `x = x * 2` | Arithmetic |
| `/=` | `x /= 4` | `x = x / 4` | Arithmetic |
| `%=` | `x %= 8` | `x = x % 8` | Arithmetic |
| `&=` | `x &= 0x0F` | `x = x & 0x0F` | Bitwise |
| `\|=` | `x \|= 0x80` | `x = x \| 0x80` | Bitwise |
| `^=` | `x ^= 0xFF` | `x = x ^ 0xFF` | Bitwise |
| `<<=` | `x <<= 2` | `x = x << 2` | Shift |
| `>>=` | `x >>= 1` | `x = x >> 1` | Shift |

### Semantics

**Left-hand side evaluated once:**
```rust
array[get_index()] += 5;
// get_index() called only ONCE
// Equivalent to:
let temp_idx = get_index();
array[temp_idx] = array[temp_idx] + 5;
```

**Type checking:** Same rules as the underlying binary operation
- Left and right types must be compatible
- Result type must match left-hand side type
- Same restrictions (e.g., `*=` requires constant power-of-2 or uses `mul()`)

### Examples

#### Variables
```rust
#[zeropage]
static mut COUNTER: u8 = 0;

fn increment() {
    COUNTER += 1;  // Clearer than COUNTER = COUNTER + 1
}
```

#### Registers
```rust
fn process_value(input @ A: u8) {
    A += 10;        // Add 10 to accumulator
    A &= 0x0F;      // Mask low nibble
    A <<= 2;        // Shift left 2 bits
}
```

#### Array Elements
```rust
#[ram]
static mut BUFFER: [u8; 256] = [0; 256];

fn update_buffer(index: u8, delta: u8) {
    BUFFER[index] += delta;
}
```

#### Struct Fields
```rust
struct Player {
    health: u8,
    score: u16,
}

fn take_damage(p: near<Player>, damage: u8) {
    (*p).health -= damage;
}

fn add_score(p: near<Player>, points: u16) {
    (*p).score += points;
}
```

### Assembly Output

Compound assignments compile to the same code as manual expansion:

```rust
// Source
A += 5;

// Assembly (same as A = A + 5)
CLC
ADC #$05
```

```rust
// Source
COUNTER += 1;

// Assembly (same as COUNTER = COUNTER + 1)
LDA COUNTER
CLC
ADC #$01
STA COUNTER
```

### Operator Restrictions Apply

Compound assignments inherit restrictions from their underlying operators:

```rust
// OK: Constant power-of-2 multiplication
x *= 8;  // Uses shifts

// ERROR: Variable multiplication requires mul()
x *= y;  // Compiler error

// OK: Constant shift
x <<= 3;

// ERROR: Variable shift requires shl()
x <<= n;  // Compiler error
```

### Increment/Decrement Operators

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

**STATUS**: Design Complete
**Last Updated**: 2026-01-02
**Next Steps**: Implement in compiler frontend (lexer/parser) and MIR
