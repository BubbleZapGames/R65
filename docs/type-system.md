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

A struct with a single positional field is a [newtype](#newtypes), which is a
different thing entirely — a scalar, not an aggregate. Two or more positional
fields are rejected:

```rust
struct Point(u8, u16);
// error: a newtype wraps exactly one field
//   hint: give each field a name: 'struct Point { field0: u8, field1: u16 }'
```

#### Unions

```rust
union Name {
    field1: T1,
    field2: T2,
    // ...
}
```

**Layout**: All fields at offset 0 — they share the same bytes
**Size**: Largest field size
**Alignment**: No requirement
**Tag**: None — no record of which field was last written

See [unions.md](unions.md) for initialization rules and restrictions.

#### Newtypes

```rust
struct TileId(u8);
struct Q10(i16);
```

A newtype gives a scalar its own name in the type system while staying that
scalar at runtime. It is **not an aggregate**: a `TileId` is one byte in a
register, passed and returned by value, and its methods take `self` by value.

**Layout**: Identical to the payload — a newtype adds nothing
**Size**: The payload's size (1 or 2 bytes)
**Payload**: `u8` `i8` `bool` `u16` `i16`, any enum, or any near pointer

This is the opposite trade-off from a [type alias](#type-aliases), which is
transparent and adds no safety, and from a single-field struct, which is nominal
but must be passed by reference.

```rust
type q10 = i16;              // transparent: q10 + i16 type-checks
struct Q10v { value: i16 }   // nominal, but passed as *Q10v
struct Q10(i16);             // nominal AND free
```

**Rules**:

- Payload must be a scalar of at most 2 bytes, so it fits in a register.
- Values of the payload type flow **in** implicitly; a newtype never flows
  **out** without an explicit `.0` or `as`.
- Operators are inherited from the payload, and the result stays nominal.
- Two different newtypes never mix, even with the same payload.
- `.0` reads the payload and is read-only — assign the whole value instead.
- `Newtype(x)` is checked like an assignment into the payload, so it rejects
  exactly what `let t: Newtype = x;` rejects. `x as Newtype` is the cast, and
  the only spelling that truncates.
- Methods take `self` by value; see [abi-models.md](abi-models.md).
- A newtype may implement a trait for static dispatch, but cannot be a `*dyn`
  target, and cannot implement `Clone` (see below).

```rust
let t: TileId = 5;        // OK: payload flows in
let t = TileId(5);        // OK: same thing written explicitly
let u: TileId = t + 1;    // OK: `+` inherited, result is TileId
let b: bool = t < u;      // OK: comparisons yield bool

let n: u8 = t;
// error: type mismatch in let binding: expected u8, found TileId

let n: u8 = t.0;          // OK: explicit unwrap
let n: u8 = t as u8;      // OK: same, 0 cycles
```

Construction and casting differ only in strictness — both compile to nothing:

```rust
let t = TileId(300);
// error: integer literal 300 does not fit in type u8 (valid range: 0 to 255)

let t = 300 as TileId;    // OK: `as` truncates, as it does for any type
```

Distinct newtypes are incompatible even when their payloads match:

```rust
struct Q10(i16);
struct Ticks(i16);

let a: Q10 = 5;
let b: Ticks = 6;
let c = a + b;
// error: operator '+' has mismatched types 'Q10' and 'Ticks'
```

**Traits, statically dispatched.** A newtype may implement a trait, and the
receiver form follows the implementing type rather than the trait declaration —
so a newtype implements a `*self`-declared trait with bare `self`.

```rust
trait Drawable { fn draw(*self); }

impl Drawable for TileId {
    fn draw(self) { }    // bare self: one self form per type
}
```

What it cannot be is a **`*dyn` target**. Dynamic dispatch reads a TypeId byte
at offset 0 of the pointee, and every byte a newtype has is payload — the same
conflict that stops unions implementing traits, except that here it bites only
at the cast, since that byte is injected only for traits actually used with
`*dyn`.

```rust
let d: far *dyn Drawable = &T as far *dyn Drawable;
// error: cannot form a '*dyn Drawable' over newtype 'TileId'
```

`Clone` is rejected for a different reason: a newtype is a scalar, so plain
assignment already copies it.

```rust
impl Clone for TileId { }
// error: newtype 'TileId' cannot implement Clone

let b: TileId = a;   // copying needs no impl
```

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

#### Multiple Return Values

```rust
fn get_pair() -> u8, u8 { return A, B; }   // Two u8 values in A, B (m8 mode)
fn get_word() -> u8, u16 { return A, X; }  // u8 + u16 in A, X
```

The return type lists the value types; the compiler assigns registers in hardware order. The first value goes to `A` (one byte in m8, two in m16), the second to `B` when it is 8-bit in m8 mode (otherwise `X`), then `X` and `Y` (both 16-bit). Register names are not used in the return type.

Caller captures return values with multi-binding:
```rust
let a, b = get_pair();    // declare and bind
a, b = get_pair();        // assign to existing variables
```

#### Never Type

```rust
-> !           // Function never returns
```

Used for functions that loop forever or halt the processor. The compiler omits `RTS`/`RTL` and emits a branch-to-self infinite loop (`BRA`) as a safety fallback.

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

Type aliases create alternate names for existing types. They are fully transparent to the type checker — `Word` and `u16` are the same type, and mixing them is not an error. For a name the type checker keeps *distinct*, use a [newtype](#newtypes).

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

### Array Literal Element Checking

**Rule**: Initializing from an array literal is assignment, so every element is
checked against the element type with the assignment rule

```rust
struct Q10(i16);

let a: [Q10; 2] = [1, 2];         // OK: payload flows in, as in a `let`
let b: [u8; 2]  = [x_i8, 2];      // OK: same-size signed/unsigned still mix
let c: [i16; 2] = [q, 2];         // ERROR: element 1 has type Q10, expected i16
let d: [u8; 2]  = [300, 2];       // ERROR: 300 does not fit in u8
```

When the element type comes from context, the first element is checked like any
other. Only when there is *no* context does the first element instead supply the
inferred element type, and there is then nothing to check it against:

```rust
let e = [x_i8, y_u8];             // element type inferred as i8 from element 1
```

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

### Return Type Checking

**Rule**: `return` is assignment-shaped — each returned value must be assignable
to the declared return type, and that type is the literal context

```rust
struct Q10(i16);

fn a() -> i16 { let q: Q10 = 5; return q; }   // ERROR: returning 'Q10' from '-> i16'
fn b() -> i16 { let q: Q10 = 5; return q.0; } // OK: explicit field access
fn c() -> Q10 { let n: i16 = 5; return n; }   // OK: transparent in
fn d() -> i8  { return 0xFF; }                // ERROR: 255 does not fit in i8
fn e() -> i8  { return -1; }                  // OK
```

The same rule as assignment, so a newtype cannot launder itself back into its
payload on the way out of a function, and an out-of-range literal is caught the
way `let x: i8 = 0xFF;` already catches it.

A multi-return signature checks each position against its own declared type. A
`-> !` function and a bare `return;` have nothing to compare, and are unchecked.

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
| `Newtype as T` | Newtype to its payload (0 cycles) |
| `T as Newtype` | Payload to newtype, truncating if narrower |

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
