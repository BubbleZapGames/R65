# Type System and Mode Tracking Design

## Overview

R65's type system is designed around the 65816's unique characteristic: register sizes change based on processor mode. The type system must track processor modes statically and validate register usage at compile time.

**Design Principles**:
- Static mode tracking - know mode at every program point
- Mode-dependent register types - A/X/Y change size based on mode
- Explicit mode management - programmer controls mode transitions
- Compile-time mode validation - catch mode errors early

---

## Basic Types

### Primitive Integer Types

```rust
u8   // Unsigned 8-bit  (0 to 255)
i8   // Signed 8-bit    (-128 to 127)
u16  // Unsigned 16-bit (0 to 65535)
i16  // Signed 16-bit   (-32768 to 32767)
bool // Boolean (0 or not 0, stored as u8)
```

**Size and Alignment**:
```
u8, i8, bool:  1 byte, no alignment requirement
u16, i16:      2 bytes, no alignment requirement
```

**Range Checking**: None - all operations wrap on overflow

---

### Hardware Register Types

Hardware registers have **mode-dependent types**:

```rust
// In m8 mode (8-bit accumulator):
A: u8
B: u8  // High byte of accumulator (m8 mode only)

// In m16 mode (16-bit accumulator):
A: u16
// B: not available (compile error if used)

// X/Y registers (always x16 mode):
X: u16
Y: u16

// Mode-independent registers:
STATUS: u8   // Always 8-bit
D: u16       // Always 16-bit (Direct Page)
DBR: u8      // Always 8-bit (Data Bank Register)
PBR: u8      // Always 8-bit (Program Bank Register, read-only)
S: u16       // Always 16-bit (Stack Pointer)
```

**Type at Compile Time**: Determined by current mode annotation

**Special Case: B Register**
- **Only available in m8 mode** (default) - compiler error if used in m16 mode (when function has `@ A: u16` parameter)
- B is the **hidden high byte** of the 16-bit accumulator
- Accessed via XBA (Exchange B and A) instruction
- Cannot appear in `#[preserves(...)]` attribute (compile error)
- See [docs/b-register.md](b-register.md) for complete details

---

### Composite Types

#### Arrays

```rust
[T; N]  // Fixed-size array of N elements of type T
```

**Size**: `N * sizeof(T)` bytes
**Alignment**: No requirement (packed)
**Indexing**: No bounds checking

#### Structs

```rust
struct Name {
    field1: T1,
    field2: T2,
    // ...
}
```

**Layout**: Packed, fields in declaration order, no padding
**Size**: Sum of field sizes
**Alignment**: No requirement

#### Enums

```rust
enum Name {
    Variant1 = value1,
    Variant2,
    // ...
}
```

**Representation**: Smallest integer type that fits all values
**Size**: Inferred (u8 if all values fit in 0-255, otherwise u16)

#### Pointers

```rust
*ptr: T      // 16-bit near pointer (current bank)
far *ptr: T  // 24-bit far pointer (includes bank)
```

**Size**:
- Near pointers (`*ptr: T`): 2 bytes
- Far pointers (`far *ptr: T`): 3 bytes

**Metadata**: None (raw addresses)

#### Function Pointers

```rust
fn(params) -> return_type        // Near function (JSR/RTS)
far fn(params) -> return_type    // Far function (JSL/RTL)
```

**Size**:
- `fn()`: 2 bytes (16-bit address)
- `far fn()`: 3 bytes (24-bit address)

**Calling Convention**: Encoded in type

---

## Processor Modes

### Mode Bits

The 65816 STATUS register has two mode bits:

- **M bit (bit 5)**: Memory/Accumulator size
  - M=1: 8-bit mode (m8)
  - M=0: 16-bit mode (m16)

- **X bit (bit 4)**: Index register size
  - X=1: 8-bit mode (x8)
  - X=0: 16-bit mode (x16)

**Note**: While the 65816 hardware supports x8 mode, **R65 always uses x16 mode**. X and Y registers are always 16-bit (`u16`). Attempting to use `@ X: u8` or `@ Y: u8` is a compile error.


---

### Static Mode Analysis

The compiler tracks the current mode at **every program point**. Mode is inferred from function parameters:

```rust
// Default mode: m8 (8-bit A), x16 (16-bit X/Y)
fn example(val @ A: u8) {
    // Mode here: m8, x16
    // A: u8, X: u16, Y: u16

    A = 10;  // OK: A is u8
    X = 1000;  // OK: X is always u16
}

// m16 mode inferred from @ A: u16 parameter
fn example16(val @ A: u16) {
    // Mode here: m16, x16
    // A: u16, X: u16, Y: u16

    A = 1000;  // OK: A is u16
    X = 1000;  // OK: X is always u16
}
```

## Type Checking

### Assignment Type Checking

**Rule**: Both sides must have identical types (no implicit conversions)

```rust
let a: u8 = 10;
let b: u16 = 20;

a = b;  // ERROR: type mismatch (u8 vs u16)
b = a;  // ERROR: type mismatch (u16 vs u8)

b = a as u16;  // OK: explicit cast
```

### Register Alias Type Checking

**Rule**: Alias type must match current register type

```rust
fn example() {
    let value @ A: u8 = 10;   // OK: A is u8 in m8 mode
    let value @ A: u16 = 10;  // ERROR: A is u8, not u16

    REP(0x20);  // Switch to m16
    let value @ A: u16 = 1000;  // OK: A is now u16
}
```

### Function Call Type Checking

**Rule**: Argument types must match parameter types exactly

```rust
fn add(a: u8, b: u8) -> u8 { }

let x: u8 = 10;
let y: u16 = 20;

add(x, x);  // OK
add(x, y);  // ERROR: y is u16, parameter expects u8
add(x, y as u8);  // OK: explicit cast
```

### Pointer Type Checking

**Rule**: Pointer types must match exactly

```rust
let *p1: u8 = 0x2000;
let *p2: u16 = 0x3000;
let far *p3: u8 = 0x01_2000;

p1 = p2;  // ERROR: *u8 vs *u16
p1 = p3;  // ERROR: *u8 vs far *u8

p1 = p2 as *u8;      // OK: explicit cast
p1 = p3 as *u8;      // OK: explicit cast (drops bank)
```

---

## Type Inference

### Limited Type Inference

R65 has **very limited** type inference:

**Inferred**:
- Numeric literal types from context
- Register alias types from register mode

**Not Inferred**:
- Variable types (must be explicit or obvious from initializer)
- Function return types (must be explicit)
- Generic types (no generics yet)

### Integer Literal Inference

```rust
let x: u8 = 10;    // 10 inferred as u8
let y: u16 = 1000; // 1000 inferred as u16

let z = 10;  // ERROR: cannot infer type
```

### Register Alias Inference

```rust
// Default m8 mode
fn example() {
    let value @ A = 10;  // Inferred: value is u8 (A's current type in m8 mode)
}

// m16 mode (inferred from @ A: u16 parameter)
fn example16(input @ A: u16) {
    let value @ A = 1000;  // Inferred: value is u16 (A's type in m16 mode)
}
```

---

### Type Checking Pass

1. **Symbol Table**: Build with mode-aware register types
2. **Expression Typing**: Check types bottom-up
3. **Assignment**: Verify type compatibility
4. **Function Calls**: Check parameter and return types
5. **Casts**: Validate explicit casts

---

**STATUS**: Design Complete
**Last Updated**: 2025-12-31
**Next Steps**: Implement in type checker and mode analyzer
