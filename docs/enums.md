# Enums

**Status: implemented** — fieldless enums with methods, and tagged enums with
struct payloads, index-`self` methods and payload-binding patterns.

R65 has two kinds of enum. A **fieldless** enum is a named integer constant, the
C-style enum the language has always had. A **tagged** enum carries data: each
variant may name a struct, and the value is a discriminant plus that struct.

Both take `impl` blocks. What differs is the receiver, and that follows from the
layout.

---

## Fieldless Enums

Every variant is a bare name, with an optional explicit discriminant. Values
auto-increment from the last one given:

```rust
enum Dir { North = 0, East, South, West }

let d: Dir = Dir::North;
let n: u8 = d as u8;
```

The enum is one byte wide, or two once a variant exceeds 255. That width is
inferred from the largest variant, and an enum binds a register of the matching
size:

```rust
enum Dir  { North, East }          // 1 byte
enum Wide { A = 0, B = 256 }       // 2 bytes

fn turn(d @ A: Dir)  { }           // A is 8-bit
fn seek(w @ X: Wide) { }           // X is 16-bit
```

### Methods

An `impl` block gives a fieldless enum methods. `self` is the discriminant and
rides in **A**, so a method call costs nothing beyond the call itself — the same
convention a newtype uses, for the same reason: there is nothing to point at.

```rust
impl Dir {
    fn opposite(self) -> Dir {
        match self {
            Dir::North => Dir::South,
            Dir::East  => Dir::West,
            Dir::South => Dir::North,
            Dir::West  => Dir::East,
        }
    }

    fn start() -> Dir { Dir::North }    // associated fn — no self
}

let back = Dir::East.opposite();
```

`self` occupies A, so no parameter may bind `A` or `B` (B is A's high byte).
Bind `X` or `Y`, or pass on the stack. `*self` is an error: an enum is a value.

An all-constant `match` like `opposite` above compiles to a **lookup table** —
`LDA table,X` after a bounds check, not a dispatch.

---

## Tagged Enums

A variant may carry a payload, which is always a previously declared **struct**:

```rust
struct Circle { r: u8 }
struct Rect   { w: u8, h: u8 }

enum Shape {
    Empty,          // unit variant
    C(Circle),
    R(Rect),
}
```

There are no inline field lists and no positional tuples — a payload names a
struct, so it has fields to reach through and can carry its own `impl` block.

Discriminants are assigned densely from 0 and `= n` is rejected on a tagged
enum: a dense tag is what keeps the jump-table path eligible.

### Layout: two planes

A tagged enum is stored the way an `#[soa]` struct is — split into planes, one
static per type:

```rust
#[lowram]
static mut POOL: [Shape; 16];
//  ->  POOL__tag:  [u8; 16]             stride 1
//  ->  POOL__data: [Shape__Payload; 16] stride 2
```

`Shape__Payload` is a synthetic union with one field per payload variant, all at
offset 0, so its size is the largest payload. The parent `POOL` is allocated no
bytes; the planes hold the data.

The point of the split is the tag. In an array-of-structs layout, reading
element `i`'s tag means scaling `i` by the element size first. Here the tag
plane strides by 1, so the raw index addresses it:

```
LDA POOL__tag,X        ; 4 cycles, no multiply
```

Only a payload access pays for scaling, and only by the payload's own stride —
a power of two where that costs at most one byte per element (3 → 4, 7 → 8), so
it scales with `ASL` rather than a shift-and-add. This is the same rounding
`#[soa]` applies to a field plane.

A single instance is a pool of one:

```rust
#[lowram]
static mut SHAPE: Shape;      // SHAPE__tag: [u8; 1], SHAPE__data: [_; 1]
```

### No contiguous element

An element's bytes are split across two arrays, so there is nothing to copy,
address, or hold. These are all rejected:

- a local or parameter of the enum type
- a field of the enum type in another aggregate
- `&POOL[i]`
- `impl Clone`
- `impl Trait for` — dispatch stores a TypeId byte at offset 0, which is where
  the discriminant lives. This is the union restriction, for the same reason.

Pass the element **index** instead — that is what a method receives.

### Construction

A variant is written directly into the planes. There is no temporary to
assemble and no block copy: one store for the tag, one per payload field.

```rust
POOL[i] = Shape::C(Circle { r: 5 });
// LDA #$01   STA POOL__tag,X       (tag)
// LDA #$05   STA POOL__data,Y      (payload)

POOL[i] = Shape::Empty;
// STZ POOL__tag,X                  (tag only)
```

### Matching

`match` on an element dispatches on its tag. A payload pattern binds the payload
by name:

```rust
fn area(i: u16) -> u8 {
    return match POOL[i] {
        Shape::C(c) => c.r,
        Shape::R(r) => r.w,
        Shape::Empty => 0,
    };
}
```

The binding is a **view of the pool, not a copy**. `c.r` is one absolute-indexed
load off the payload plane, and writing through it writes the pool:

```rust
match POOL[i] {
    Shape::C(c) => { c.r = c.r + 1; }   // increments POOL[i]'s payload
    _ => {}
}
```

A match must cover every variant or carry a `_`. Coverage is checked against the
variants, not against the tag's integer range, and an arm naming a different
enum is rejected.

A payload binding cannot appear inside an or-pattern. Which variant matched is
what decides how the payload plane's bytes are typed, and no two payload
variants share a type, so a binding spanning alternatives would read the wrong
struct through the arm it was not written for:

```rust
match POOL[i] {
    Shape::C(c) | Shape::R(r) => c.r,   // error: binds a payload inside an or-pattern
    Shape::Empty => 0,
}

Shape::C | Shape::Empty => 1,           // fine — no binding
```

### Methods

`self` is the element **index**, and rides in **X** — where indexed addressing
wants it. This is the `#[soa]` receiver, for the same reason: the element has no
address.

```rust
impl Shape {
    fn area(self) -> u8 {
        return match self {
            Shape::C(c) => c.r,
            Shape::R(r) => r.w,
            Shape::Empty => 0,
        };
    }
}

let a = POOL[2].area();
```

Because a method hard-codes its plane addresses, a tagged enum backs **exactly
one** static — declare a second and the compiler refuses, since its methods
would have nowhere to point.

---

## Codegen

`Shape::area` above, as emitted by r65c:

```
main:
    LDX #$02                     ; the receiver: an element index
    JSR Shape__area

Shape__area:
    PHX
    LDA $0200,X                  ; tag plane, raw index — no scaling
    CMP #$01
    BNE Shape__area__L3
Shape__area__L2:                 ; Shape::C
    REP #$20
    TXA
    ASL A                        ; payload stride 2
    TAX
    SEP #$20
    LDA $0210,X                  ; POOL__data[i].C.r
    ...
```

Passing the receiver is one instruction. The tag read is one instruction. Only
the payload access scales the index, and only by the payload's own stride.

Match strategy selection is unchanged: the scrutinee is an ordinary byte by the
time the lowerer sees it, so a tagged enum gets a lookup table, a jump table or
a compare chain on the same cost model as any other match (see
[docs/control-flow.md](control-flow.md)).

---

## Known limits

- **No initializers.** A tagged enum static cannot carry one — transposing an
  array literal into two planes is not implemented. SNES RAM is unpredictable at
  power-on, so **write every element before matching on it**. An uninitialized
  tag is a byte of garbage: the jump-table path bounds-checks it, and a compare
  chain is first-match-wins so it falls through to the last arm, but neither is
  a substitute for initializing.
- **One static per tagged enum type**, as for `#[soa]`.
- **A payload must be a declared struct** — not a union, newtype, enum, `#[soa]`
  struct, primitive, array or pointer.
- **No `= n` on a tagged enum's variants.**
- A payload cannot be copied in from an existing struct
  (`POOL[i] = Shape::C(TEMPLATE)`); write the fields.
- **No payload binding inside an or-pattern** — give the variant its own arm.

---

## References

- [SoA Struct Arrays](soa.md) — the plane layout and index-`self` convention
  this reuses
- [Unions](unions.md) — the payload union's layout rule, and the offset-0
  argument for rejecting traits
- [Type System](type-system.md)
- [Control Flow](control-flow.md) — match lowering and its cost model
- [ABI Models](abi-models.md) — receiver conventions
