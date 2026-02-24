# Type System and Mode Tracking Design

## Overview

R65's type system is designed around the 65816's unique characteristic: register sizes change based on processor mode. The type system must track processor modes statically and validate register usage at compile time.

**Design Principles**:
- Static mode tracking - know mode at every program point
- Mode-dependent register types - A changes size based on mode
- Automatic mode inference - mode determined from function parameters
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
- **Only meaningful in m8 mode** (default) — in m16 mode, B is part of the 16-bit A register
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
ptr: *T      // 16-bit near pointer (current bank)
ptr: far *T  // 24-bit far pointer (includes bank)
```

**Size**:
- Near pointers (`*T`): 2 bytes
- Far pointers (`far *T`): 3 bytes

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

#### Tuples (Multiple Return Values)

```rust
(u8, u8)       // Two u8 values (returned in A, B in m8 mode)
(u8, u16)      // Mixed sizes (returned in A, X)
```

Tuple types are used for multiple return values. They cannot be stored in variables — only destructured at call sites:
```rust
let (a, b) = get_pair();
```

#### Never Type

```rust
-> !           // Function never returns
```

Used for functions that loop forever or halt the processor. The compiler omits `RTS`/`RTL` and emits `WAI` as a safety fallback.

#### Pointer to Array

```rust
*[T; N]        // Pointer to fixed-size array (coerces to *T)
```

A `*[T; N]` provides compile-time bounds checking on constant indices and implicitly coerces to `*T`.

#### Type Aliases

```rust
type Word = u16;
type Callback = fn(u8) -> u8;
```

Type aliases create alternate names for existing types. They are fully transparent to the type checker.

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

**Rule**: Integer types are compatible with each other for assignment. Explicit casts are recommended but not required for integer-to-integer assignment:

```rust
let a: u8 = 10;
let b: u16 = 20;

b = a;          // OK: integer types are compatible
a = b;          // OK: truncates (programmer's responsibility)
b = a as u16;   // Preferred: explicit cast makes intent clear
```

**Non-integer types** require exact type matches (pointers, structs, enums).

### Implicit Integer Promotion in Expressions

When binary operations have mixed-size integer operands, the compiler automatically inserts a widening cast:

```rust
let a: u8 = 10;
let b: u16 = 1000;
let c: u16 = a + b;  // a is implicitly promoted to u16
```

This applies to arithmetic, bitwise, and comparison operators.

### Register Alias Type Checking

**Rule**: Alias type must match current register type (determined by function's mode)

```rust
// m8 mode (default)
fn example() {
    let value @ A: u8 = 10;   // OK: A is u8 in m8 mode
    let value @ A: u16 = 10;  // ERROR: A is u8, not u16
}

// m16 mode (inferred from parameter)
fn example16(input @ A: u16) {
    let value @ A: u16 = 1000;  // OK: A is u16 in m16 mode
}
```

**Note**: Mode is fixed per function based on parameter types. Manual `REP`/`SEP` instructions (via `asm!()`) do not change the compiler's mode tracking.

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
let p1: *u8 = 0x2000;
let p2: *u16 = 0x3000;
let p3: far *u8 = 0x01_2000;

p1 = p2;  // ERROR: *u8 vs *u16
p1 = p3;  // ERROR: *u8 vs far *u8

p1 = p2 as *u8;      // OK: explicit cast
p1 = p3 as *u8;      // OK: explicit cast (drops bank)
```

### Explicit Cast Rules (`as`)

The `as` keyword performs explicit type conversions:

| Cast | Behavior |
|------|----------|
| `u8 as u16` | Zero-extend |
| `i8 as i16` | Sign-extend |
| `u16 as u8` | Truncate (keep low byte) |
| `u8 as i8` | Reinterpret bits |
| `bool as u8` | `false` → 0, `true` → 1 |
| `u8 as bool` | 0 → `false`, non-zero → `true` |
| `*T as *U` | Pointer reinterpret |
| `*T as u16` | Pointer to integer |
| `u16 as *T` | Integer to pointer |
| `far *T as u16` | Far pointer truncated to 16-bit |
| `Enum as u8/u16` | Enum to underlying integer |

### Pointer Auto-Dereference

Struct field access through pointers is automatically dereferenced:

```rust
#[zeropage]
static mut PTR: *Player;

PTR.x = 10;    // Auto-dereferences: (*PTR).x = 10
let v = PTR.y; // Auto-dereferences: let v = (*PTR).y
```

No explicit `(*ptr).field` syntax is required.

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

**STATUS**: Implemented
**Last Updated**: 2026-02-10
