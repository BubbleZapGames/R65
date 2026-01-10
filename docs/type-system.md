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
bool // Boolean (0 or 1, stored as u8)
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

// In x8 mode (8-bit index):
X: u8
Y: u8

// In x16 mode (16-bit index):
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
- **Only available in `#[mode(m8)]`** - compiler error if used in m16 mode
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

### Mode States

Four possible mode combinations:

```rust
#[mode(m8, x8)]   // M=1, X=1: A=u8,  X=u8,  Y=u8
#[mode(m8, x16)]  // M=1, X=0: A=u8,  X=u16, Y=u16
#[mode(m16, x8)]  // M=0, X=1: A=u16, X=u8,  Y=u8
#[mode(m16, x16)] // M=0, X=0: A=u16, X=u16, Y=u16
```

### Partial Mode Annotations

Functions can specify only one mode dimension:

```rust
#[mode(m8)]   // Only specify accumulator mode (X mode unconstrained)
#[mode(x16)]  // Only specify index mode (M mode unconstrained)
```

**Meaning**: Function works correctly regardless of unspecified mode

---

## Mode Tracking

### Static Mode Analysis

The compiler tracks the current mode at **every program point**:

```rust
#[mode(m8, x8)]
fn example() {
    // Mode here: m8, x8
    // A: u8, X: u8, Y: u8

    A = 10;  // OK: A is u8

    REP(0x20);  // Clear M bit → m16 mode
    // Mode here: m16, x8
    // A: u16, X: u8, Y: u8

    A = 1000;  // OK: A is now u16
    X = 1000;  // ERROR: X is still u8

    REP(0x10);  // Clear X bit → x16 mode
    // Mode here: m16, x16
    // A: u16, X: u16, Y: u16

    X = 1000;  // OK: X is now u16
}
```

### Mode at Function Boundaries

**Entry Mode**: Specified by `#[mode(...)]` attribute

```rust
#[mode(m8, x8)]
fn process() {
    // Caller must ensure we enter in m8, x8 mode
    // A: u8, X: u8, Y: u8
}
```

**Exit Mode**: Must match entry mode (unless `transition` specified)

```rust
#[mode(m8, x8)]
fn example() {
    REP(0x30);  // Switch to m16, x16
    // ...
    SEP(0x30);  // Must restore to m8, x8 before return
}

// Compiler error if mode at exit doesn't match entry
```

---

## Mode Transitions

### Manual Mode Changes

Programmer uses `SEP()` and `REP()` functions:

```rust
SEP(0x20);  // Set M bit (8-bit accumulator)
REP(0x20);  // Clear M bit (16-bit accumulator)
SEP(0x10);  // Set X bit (8-bit index)
REP(0x10);  // Clear X bit (16-bit index)
SEP(0x30);  // Set both (8-bit both)
REP(0x30);  // Clear both (16-bit both)
```

**Compiler Tracking**: Compiler analyzes SEP/REP calls to update mode

```rust
#[mode(m8, x8)]
fn example() {
    // Mode: m8, x8

    REP(0x30);  // Compiler knows: now m16, x16
    // Mode: m16, x16

    SEP(0x20);  // Compiler knows: now m8, x16
    // Mode: m8, x16
}
```

### Automatic Transitions

Functions can specify `transition=inline` for automatic mode management:

```rust
#[mode(m16, x16, transition=inline)]
fn needs_16bit() {
    // Compiler generates:
    // PHP          ; Save status (including mode bits)
    // REP #$30     ; Switch to m16, x16
    // ... function body ...
    // PLP          ; Restore original mode
    // RTS
}
```

**Safety**: Caller can be in any mode - callee handles transition

### Caller-Side Transitions

Functions can specify `transition=caller`:

```rust
#[mode(m16, x16, transition=caller)]
fn batch_candidate() {
    // Caller generates mode transition if needed
}

fn caller() {
    // Compiler can batch multiple calls:
    REP(0x30);
    batch_candidate();
    another_16bit_function();
    yet_another();
    SEP(0x30);
}
```

**Optimization**: Enables batching when calling multiple same-mode functions

---

## Mode Checking Rules

### Rule 1: Register Usage Must Match Mode

```rust
#[mode(m8, x8)]
fn example() {
    A = 255;     // OK: A is u8
    A = 256;     // ERROR: 256 doesn't fit in u8

    let x: u16 = A;  // ERROR: A is u8, cannot assign to u16
    let x: u8 = A;   // OK: types match
}
```

### Rule 2: Mode Transitions Must Be Explicit

```rust
#[mode(m8, x8)]
fn example() {
    let value: u16 = 1000;
    A = value;  // ERROR: A is u8, value is u16

    // Must change mode first:
    REP(0x20);  // Switch to m16
    A = value;  // OK: A is now u16
}
```

### Rule 3: Function Entry Mode Must Be Satisfied

```rust
#[mode(m16, x16)]
fn needs_16bit() { }

#[mode(m8, x8)]
fn caller() {
    needs_16bit();  // ERROR: mode mismatch (unless transition specified)

    // Must change mode first:
    REP(0x30);
    needs_16bit();  // OK
    SEP(0x30);
}

// OR use transition:
#[mode(m16, x16, transition=inline)]
fn needs_16bit_safe() { }

fn caller() {
    needs_16bit_safe();  // OK: callee handles transition
}
```

### Rule 4: Function Exit Mode Must Match Entry

```rust
#[mode(m8, x8)]
fn bad_example() {
    REP(0x30);  // Switch to m16, x16
    // ERROR: exit mode (m16, x16) doesn't match entry (m8, x8)
}

#[mode(m8, x8)]
fn good_example() {
    REP(0x30);  // Switch to m16, x16
    // ... do work ...
    SEP(0x30);  // Restore to m8, x8
    // OK: exit mode matches entry
}
```

### Rule 5: Control Flow Must Have Consistent Modes

```rust
#[mode(m8, x8)]
fn example(flag: bool) {
    if flag {
        REP(0x20);  // m16, x8
        // ...
        SEP(0x20);  // Back to m8, x8
    } else {
        REP(0x30);  // m16, x16
        // ...
        SEP(0x30);  // Back to m8, x8
    }
    // Both branches must converge to same mode (m8, x8)

    A = 10;  // OK: mode is m8, x8 on all paths
}
```

---

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
#[mode(m8, x8)]
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
#[mode(m8, x8)]
fn example() {
    let value @ A = 10;  // Inferred: value is u8 (A's current type)

    REP(0x20);
    let value @ A = 1000;  // Inferred: value is u16 (A's new type)
}
```

---

## Mode Tracking Algorithm

### Control Flow Graph (CFG)

The compiler builds a CFG and tracks mode at each basic block:

```
Basic Block 1 (Entry):
  Mode: m8, x8
  ↓
Basic Block 2:
  Mode: m8, x8
  REP(0x30)
  Mode: m16, x16
  ↓
Basic Block 3:
  Mode: m16, x16
```

### Mode Join Points

When control flow merges, modes must match:

```rust
#[mode(m8, x8)]
fn example(flag: bool) {
    // Block 1: m8, x8

    if flag {
        // Block 2: m8, x8
        REP(0x20);
        // Block 2 exit: m16, x8
    } else {
        // Block 3: m8, x8
        REP(0x30);
        // Block 3 exit: m16, x16  ← MISMATCH!
    }

    // Block 4: ???
    // ERROR: mode mismatch at join point (m16,x8 vs m16,x16)
}
```

**Fix**: All paths must reach same mode at join:

```rust
#[mode(m8, x8)]
fn example(flag: bool) {
    if flag {
        REP(0x20);
        // ... work in m16, x8 ...
        SEP(0x20);  // Back to m8, x8
    } else {
        REP(0x30);
        // ... work in m16, x16 ...
        SEP(0x30);  // Back to m8, x8
    }
    // Both paths converge at m8, x8 ✓
}
```

### Loop Mode Consistency

Loop entry and exit modes must match:

```rust
#[mode(m8, x8)]
fn example() {
    loop {
        // Entry: m8, x8

        REP(0x30);
        // ... work ...
        SEP(0x30);

        // Exit: must be m8, x8 (to match entry)
    }
}
```

---

## Mode Tracking Examples

### Example 1: Simple Mode Change

```rust
#[mode(m8, x8)]
fn process_u16(value: u16) -> u16 {
    // Entry: m8, x8 (A: u8, X: u8, Y: u8)

    REP(0x20);  // Clear M bit
    // Now: m16, x8 (A: u16, X: u8, Y: u8)

    A = value;  // OK: A is u16, value is u16
    A = A + 1;  // OK: u16 + u16

    SEP(0x20);  // Set M bit
    // Now: m8, x8 (A: u8, X: u8, Y: u8)

    return A;  // ERROR: A is u8, but return type is u16
}

// Fixed version:
#[mode(m8, x8)]
fn process_u16_fixed(value: u16) -> u16 {
    REP(0x20);
    A = value;
    A = A + 1;
    let result @ A: u16 = A;  // Capture while still u16
    SEP(0x20);
    return result;  // OK: returning u16 variable
}
```

### Example 2: Conditional Mode Changes

```rust
#[mode(m8, x8)]
fn conditional(use_16bit: bool, data: u16) {
    if use_16bit {
        REP(0x30);  // m16, x16
        A = data;
        SEP(0x30);  // Back to m8, x8
    } else {
        // Stay in m8, x8
        A = data as u8;
    }
    // Both paths: m8, x8 ✓

    A = A + 1;  // OK: A is u8 on all paths
}
```

### Example 3: Cross-Function Mode Handling

```rust
#[mode(m16, x16, transition=caller)]
fn worker() {
    A = 1000;  // OK: A is u16
}

#[mode(m8, x8)]
fn caller() {
    // Current mode: m8, x8

    REP(0x30);  // Compiler knows: now m16, x16
    worker();   // OK: mode matches
    SEP(0x30);  // Back to m8, x8

    // Current mode: m8, x8
}
```

### Example 4: Mode-Polymorphic Function

```rust
// Function works in any X mode (doesn't use X/Y)
#[mode(m16)]
fn process_a() {
    A = 1000;  // OK: A is u16 (m16 specified)
    // X, Y not used - works regardless of x8 or x16
}

#[mode(m16, x8)]
fn caller1() {
    process_a();  // OK: m16 matches, x8 compatible
}

#[mode(m16, x16)]
fn caller2() {
    process_a();  // OK: m16 matches, x16 compatible
}
```

---

## Type Errors and Diagnostics

### Type Mismatch

```rust
let a: u8 = 10;
let b: u16 = a;

// ERROR: type mismatch
// expected: u16
// found: u8
// help: use explicit cast: `a as u16`
```

### Mode Mismatch in Function Call

```rust
#[mode(m16, x16)]
fn needs_16bit() { }

#[mode(m8, x8)]
fn caller() {
    needs_16bit();
}

// ERROR: mode mismatch in function call
// function expects: m16, x16
// current mode: m8, x8
// help: add mode transition:
//   REP(0x30);
//   needs_16bit();
//   SEP(0x30);
// or: add `transition=inline` to `needs_16bit`
```

### Mode Exit Mismatch

```rust
#[mode(m8, x8)]
fn bad() {
    REP(0x30);
    // ... forget to restore ...
}

// ERROR: function exit mode doesn't match entry mode
// entry mode: m8, x8
// exit mode: m16, x16
// note: mode changed at line X with REP(0x30)
// help: restore mode before return: SEP(0x30)
```

### Mode Join Mismatch

```rust
#[mode(m8, x8)]
fn branches(flag: bool) {
    if flag {
        REP(0x20);
    }
    A = 10;
}

// ERROR: mode mismatch at control flow join
// path 1 mode: m16, x8  (from if-branch)
// path 2 mode: m8, x8   (from else-branch)
// note: modes must match at line X
// help: restore mode in if-branch: SEP(0x20)
```

### Register Size Mismatch

```rust
#[mode(m8, x8)]
fn example() {
    A = 1000;
}

// ERROR: value out of range for type
// value: 1000
// type: u8 (max 255)
// note: A is u8 in m8 mode
// help: switch to m16 mode:
//   REP(0x20);
//   A = 1000;
//   SEP(0x20);
```

---

## Implementation Notes

### Mode Tracking Data Structure

```python
class Mode:
    m_mode: bool  # True = m8, False = m16
    x_mode: bool  # True = x8, False = x16

    def a_type(self) -> Type:
        return U8 if self.m_mode else U16

    def x_type(self) -> Type:
        return U8 if self.x_mode else U16

    def y_type(self) -> Type:
        return U8 if self.x_mode else U16

class BasicBlock:
    entry_mode: Mode
    exit_mode: Mode
    instructions: List[Instruction]
```

### Mode Analysis Pass

1. **Initialize**: Set entry mode from function annotation
2. **Propagate**: Track mode through each instruction
3. **SEP/REP**: Update mode state
4. **Validate**: Check register usage against current mode
5. **Join**: Verify modes match at control flow joins
6. **Exit**: Verify exit mode matches entry mode

### Type Checking Pass

1. **Symbol Table**: Build with mode-aware register types
2. **Expression Typing**: Check types bottom-up
3. **Assignment**: Verify type compatibility
4. **Function Calls**: Check parameter and return types
5. **Casts**: Validate explicit casts

---

## Future Enhancements

### Mode Inference

Currently: Programmer must manually track modes
Future: Compiler could infer minimal mode changes:

```rust
fn auto_mode() {
    let x: u16 = 1000;
    A = x;  // Compiler auto-inserts REP(0x20)
}
```

### Partial Mode Functions

Allow functions that work in multiple modes:

```rust
#[mode(m8 | m16, x8)]  // Works in m8 or m16, requires x8
fn flexible() { }
```

### Mode Assertions

Runtime checks for debugging:

```rust
debug_assert_mode!(m8, x8);  // Panic if not in expected mode
```

---

**STATUS**: Design Complete
**Last Updated**: 2025-12-31
**Next Steps**: Implement in type checker and mode analyzer
