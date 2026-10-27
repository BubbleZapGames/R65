# Enums

**Status: implemented** — fieldless enums with methods, and tagged enums with
struct payloads, index-`self` methods, payload-binding patterns, and
[`#[soa]` payloads](#soa-payloads) whose field planes overlay the data region.

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
- a field of the enum type in a **plain** struct — see below for the `#[soa]`
  case, which is allowed
- `&POOL[i]`
- `impl Clone`
- `impl Trait for` — dispatch stores a TypeId byte at offset 0, which is where
  the discriminant lives. This is the union restriction, for the same reason.

Pass the element **index** instead — that is what a method receives.

### A field of an `#[soa]` struct

A tagged enum *may* be a field of an `#[soa]` struct. The outer static splits
into planes anyway, so the enum contributes its own two rather than needing a
contiguous form. A field that is an *array* of them is rejected: it would need a
pair of planes per element.

```rust
struct Body { x: u16, hp: u8 }

#[soa]
struct Thing { body: Body, shape: Shape }

#[lowram]
static mut things: [Thing; 8];
//  -> things__body:        [Body; 8]
//  -> things__shape__tag:  [u8; 8]              stride 1
//  -> things__shape__data: [Shape__Payload; 8]
```

Every property survives: the tag read is still `LDA things__shape__tag,X` off a
raw index, `match things[i].shape` and `things[i].shape = Shape::C(..)` reach
those planes, `things[i].shape as u8` is the discriminant, and a method takes
the *outer* array's index as its receiver — `things[i].shape.area()`.

What this buys is the shape both Rust and R65 push you toward when every variant
would otherwise carry the same fields: put the shared state **beside** the
union, not inside it. `things[i].body.hp` is then a plain plane access needing
no match, where a `body` inside each variant would need one to reach.

A plain (non-`#[soa]`) struct still rejects the field. There the enum would have
to be contiguous — a tag byte with the payload inline — which puts the element
stride multiply back on every tag read, the cost the plane split exists to
remove.

The enum still backs exactly one storage: either a static of its own or one
`#[soa]` field, not both.

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

Every operand is read before the first store, and the tag is written last.
That is what makes rewriting an element as another variant built from its old
payload safe — the new fields land on the bytes the old ones are read from:

```rust
match POOL[i] {
    Shape::C(c) => { POOL[i] = Shape::R(Rect { w: 1, h: c.r }); }   // reads c.r first
    _ => {}
};
```

The last non-constant operand is stored straight from its evaluation; any
earlier ones go through a temporary, so the common one-operand rewrite costs
nothing extra.

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

The view is of the element that was **matched**. Its index is read once: an
index that could change — a static, an expression, a local the arms assign —
is copied into a temporary first, so a side effect in it runs once and
`i = i + 1` inside the arm does not move `c` to another element. (A plain
local or a literal is read where it is used, at no cost.) A construction
`POOL[f()] = Shape::C(..)` likewise evaluates `f()` once for the tag and all
the payload stores.

Rewriting the matched element ends the binding. The new variant's planes
overlay the old one's, so a later read of `c` would see the new bytes as
`Circle` fields:

```rust
match POOL[i] {
    Shape::C(c) => {
        POOL[i] = Shape::R(Rect { w: c.r, h: 1 });   // fine: c read before the store
        OUT = c.r;                                   // error: used after the rewrite
    }
    _ => {}
}
```

```
error: binding 'c' is used after its element was rewritten (line 3)
hint: the rewrite overlays the element's payload planes, so 'c' no longer names
      its old fields; read what you need into a local before the rewrite, or use
      it in the new variant's literal
```

The check follows source order and is conservative across branches: a rewrite
inside an `if` ends the binding for everything after the `if`.

To test a single variant, `if let` is the same match with a `_ => {}` arm, and
binds the same view; `while let` and `let ... else` follow suit (see
[docs/control-flow.md](control-flow.md#if-let-while-let-and-let--else)):

```rust
if let Shape::C(c) = POOL[i] { c.r = c.r + 1; }

let Shape::R(r) = POOL[i] else { return 0; };
OUT = r.w;
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

### The discriminant as a value

An element has no contiguous value, so a cast of one is not a conversion — it
is a read of the tag plane, and that is all it compiles to:

```rust
let k: u16 = POOL[i] as u16;    // LDA POOL__tag,X  (+ AND #$FF to widen)
let j: u8  = POOL[i] as u8;     // LDA POOL__tag,X
```

This is what makes a **per-kind table** the natural way to hold constants that
vary by variant. One row per variant, addressed by the element's own tag:

```rust
struct Row { speed: u16, hp: u8, pad: u8 }      // 4 bytes: index scales by two ASLs

static ROWS: [Row; 3] = [
    Row { speed:   0, hp:  0, pad: 0 },         // Shape::Empty
    Row { speed: 128, hp: 25, pad: 0 },         // Shape::C
    Row { speed: 192, hp: 12, pad: 0 },         // Shape::R
];

let speed = ROWS[POOL[i] as u16].speed;
```

The row order is the variant order, and nothing checks that for you — a test
that asserts one column per variant is what catches a mis-ordered table.

The alternative, a `match self` per column, generates a lookup table *per
method* and pays a call to reach it. One table of rows costs a shift and an
indexed load, and reads as a table.

Only an integer target is a tag read. `POOL[i] as *u8` is still refused, as is
a bare `POOL[i]`, and the diagnostic points at both spellings that do work:

```
error: cannot use a whole element of tagged enum array 'POOL'
hint: a tagged enum element is a tag plus a payload in two separate planes,
      with no contiguous value; `match POOL[i]` to reach the payload, or
      `POOL[i] as u8` for the discriminant
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

## `#[soa]` Payloads

A payload struct may itself be `#[soa]`. The enum's data plane then stops being
an array of union elements and becomes a **region** that every variant overlays
with one plane per field — so an arm body, and the payload struct's own
methods, index by the raw pool index with no scaling at all. This section gives
the motivation, the layout, and the rules.

### Motivation

Take the shape the rest of this document builds toward — an `#[soa]` entity
pool whose `kind` is a tagged enum, with per-kind state in the payloads and
per-kind behaviour on the payload structs — written with *plain* payloads:

```rust
struct Fireball { heat: u8, phase: u8 }
enum Kind { None, Barrel(Barrel), Fireball(Fireball), Spring, Boss(Boss) }

#[soa]
struct Enemy { kind: Kind, x: u8, y: u8, vx: i8, vy: i8, hp: u8 }
#[lowram] static mut POOL: [Enemy; 16];

impl Hazard for Fireball {
    fn burn(*self) -> u8 { self.heat = self.heat + 1; return self.heat; }
}

impl Enemy {
    fn update(self) {
        match self.kind {
            Kind::Fireball(f) => { f.phase = f.phase + 1; self.vy = -1; self.x = f.burn(); }
            ...
        };
    }
}
```

Everything on the shared planes is one indexed instruction. The payload is
where the cost hides, in two places. Both listings are real `r65c` output.

**The arm recomputes X on every switch between a payload field and a shared
field.** The payload plane strides by 2 (the largest payload) while every
shared plane strides by 1, so the single-slot X cache thrashes:

```asm
REP #$20            ; f.phase = f.phase + 1
LDA $01,S           ;   reload self
ASL A               ;   scale to payload stride
TAX
SEP #$20
LDA $0211,X
INC A
STA $0211,X
REP #$20            ; self.vy = -1
LDA $01,S           ;   reload self again
TAX
SEP #$20
LDA #$FF
STA $0260,X
```

Thirteen instructions for two byte updates.

**A method on the payload struct needs a pointer.** `f` is a view of the pool,
but `Fireball::burn` takes `*self`, so the call site has to materialize an
address — twelve instructions, repeated for every call in the arm, and the
callee then reaches its fields through `DBR:Y`:

```asm
REP #$20
LDA #$0230          ; #POOL__kind__data
STA $03,S
LDA $01,S           ; self
ASL A
STA $05,S
LDA $03,S
CLC
ADC $05,S
TAY
SEP #$20
JSR Fireball__burn  ; body: LDA $0000,Y / INC A / STA $0000,Y
```

This is the pointer-`self` cost that made traits unusable in classickong's
enemy loop, arriving through the back door. `#[soa]` on the payload structs
removes both.

### Layout: every variant's planes overlay one region

With `#[soa]` payloads the data plane is no longer an array of union elements.
Each variant lays its fields out by the ordinary `#[soa]` formula — one plane
per field, each padded 3→4 / 7→8 — and the variants overlay one region.

Planes overlay only planes of their **own stride**. The region is cut into one
lane per stride, narrowest first; a variant's k-th field of stride s sits k
planes into lane s, and each lane is as wide as the variant with the most
fields of that stride needs:

```
lane(s)                  = Σ_{t<s} t · max over V of #(V's fields of stride t)
addr(variant V, field f) = data_base + N · (lane(stride f) + k · stride f)
                           (f is V's k-th field of that stride)
region size              = N · Σ_s s · max over V of #(V's fields of stride s)
```

The lanes are what keep each element's bytes its own. A plane puts element i at
`i · stride` from its base, so two planes of different strides laid from the
same base would not agree on which element a byte belongs to: a `u16` plane's
element `i` would cover the bytes a `u8` plane uses for elements `2i` and
`2i + 1`, and writing one element's payload would rewrite two others'. Two
planes of the same stride at the same offset agree on every byte.

The region is at least as large as the largest variant, and larger when
variants spread their fields over different strides: `{u8, u8}` and
`{u16, u8}` need 2 + 2 bytes per element where either alone needs 3 or fewer.
It can also exceed `N × max(packed payload)` because a `far *T` field pays one
pad byte per element for its plane, exactly as an `#[soa]` struct would.

Keeping the region small:

- Variants share bytes only within a width, so widths spread across variants
  each cost a lane of their own.
- A wide value most variants carry belongs in the parent `#[soa]` struct beside
  the enum (see [A field of an `#[soa]` struct](#a-field-of-an-soa-struct)),
  not repeated in each payload.
- When sharing matters more than raw-index access, use plain payloads: their
  data plane is an ordinary union array, where variants of any width share
  bytes, at the cost of scaling the index for every payload access.

For the enum above with `N = 16`:

| Symbol | Bytes | Stride |
|---|---|---|
| `POOL__kind__tag` | `$0200` + 16 | 1 |
| `POOL__kind__data` | `$0210` + 32 (region) | — |
|   `POOL__kind__Barrel__rolling` | `$0210` + 16 | 1 |
|   `POOL__kind__Fireball__heat` | `$0210` + 16 | 1 |
|   `POOL__kind__Fireball__phase` | `$0220` + 16 | 1 |
|   `POOL__kind__Boss__stage` | `$0210` + 16 | 1 |
|   `POOL__kind__Boss__cooldown` | `$0220` + 16 | 1 |

`Barrel::rolling`, `Fireball::heat` and `Boss::stage` are the same sixteen
bytes. That is the union, transposed: element `i`'s payload bytes are still
private to element `i`, they are just no longer adjacent. Reading a plane that
belongs to a variant other than the one in the tag yields whatever the last
variant left there — exactly the guarantee a union gives now, no weaker.

Unit variants contribute no planes. An enum whose payload variants are all
`#[soa]` and all single-`u8` degenerates to one shared plane, which is what a
hand-written parallel array would be.

In the assembly the region is `POOL__kind__data` (allocated as before) and
each variant plane is a bare `.DEFINE` at region + offset, under an "Overlay
Planes" header — even in main RAM, where the region is a `.RAMSECTION` entry,
since a section cannot give two names to the same bytes.

### Access

A payload binding still names a view; only what the view rewrites to changes.
`f.phase` for an `#[soa]` variant rewrites to `POOL__kind__Fireball__phase[i]`
— the same base-rewrite `impl Enemy` uses to turn `self.hp` into `POOL[self].hp`,
applied one level down. So the arm above becomes:

```asm
INC POOL__kind__Fireball__phase,X     ; f.phase = f.phase + 1
LDA #$FF
STA POOL__vy,X                        ; self.vy = -1
```

X holds `self` for the whole arm, because every plane in reach — payload and
shared — strides by 1. This is consequence 1 of the
[X-reuse rule](soa.md#index-reuse-in-x--read-this-before-choosing-field-types):
an all-`u8` element is a strict win.

Construction is unchanged in shape — one tag store plus one store per field —
but each field store goes to its own plane with the raw index:

```rust
POOL[i].kind = Kind::Fireball(Fireball { heat: 9, phase: 0 });
// LDA #$02  STA POOL__kind__tag,X
// LDA #$09  STA POOL__kind__Fireball__heat,X
// STZ       POOL__kind__Fireball__phase,X
```

The payload must be a struct literal. `Kind::Fireball(TEMPLATE)` is rejected
outright — there is no contiguous value to copy from — where a plain payload
merely fails later as an aggregate assignment.

`POOL[i].kind as u8`, `match`, exhaustiveness, the or-pattern rule and the
per-kind table idiom are all untouched: they only ever read the tag.

### Methods: the payload struct takes the pool index

An `#[soa]` struct's methods already take `self` as an element index in X
([soa.md](soa.md#methods-self-is-an-element-index)). For a payload struct that
index is the **outer pool's** — the same `self` an `impl Kind` or `impl Enemy`
method receives, because the payload planes are indexed by it:

```rust
#[soa] struct Fireball { heat: u8, phase: u8 }

impl Hazard for Fireball {
    fn burn(self) -> u8 { self.heat = self.heat + 1; return self.heat; }
}

Kind::Fireball(f) => { self.x = f.burn(); }
```

```asm
JSR Fireball__burn          ; X already holds self — nothing to set up

Fireball__burn:
    INC POOL__kind__Fireball__heat,X
    LDA POOL__kind__Fireball__heat,X
    RTS
```

The twelve-instruction pointer build is gone, and so is the `DBR:Y` body.
Calling a trait method through a payload binding now costs exactly what
calling an inherent method on the enum costs: one `JSR` (methods are far
by default, but a same-bank call is demoted to `JSR`/`RTS` at `-O1`; see
[ABI Models](abi-models.md#why-far-is-the-default)).

Static dispatch only, as for any `#[soa]` implementor. `*dyn` over an SoA
element is unchanged — designed, not built (see [soa.md](soa.md#known-limits)).

### A payload's `self` is not a number

The index travels in X, but in the source a payload method's `self` is an
element of its variant and nothing else. A payload has no back-reference to
the struct it sits in — as in Rust, where `impl Fireball { fn burn(&mut self) }`
sees the `Fireball` and cannot name the `Enemy` around it — so the method may
use `self` in exactly three ways:

```rust
self.heat           // its own field
self.burn()         // its own method
self as u16         // the pool index, asked for explicitly
```

Any other use is rejected: `POOL[self].x`, `helper(self)`, `self == 0`,
`let i = self`.

```
error: 'self' in a method on #[soa] payload struct 'Fireball' is not a value
hint: it is an element of 'Kind::Fireball', not an index: reach its own fields
      as `self.field`; code that needs the parent's fields belongs in an impl
      on the parent; write `self as u16` if the pool index itself is needed
```

So a kind's behaviour splits in two. The payload's own methods keep what
touches only its own state; code that mixes the shared fields with the
payload belongs on the **parent**. There `self.kind` is matched like any
element and the binding is the payload — the R65 spelling of Rust's
`let Kind::Fireball(f) = &mut self.kind else { return };` — so the parent can
call the payload's methods and reach both kinds of field:

```rust
impl Fireball {
    // Own state only: heat up, report the flicker phase.
    fn process(self) -> u8 {
        self.heat = self.heat + 1;
        return self.phase;
    }
}

impl Enemy {
    fn burn(self) {
        match self.kind {
            Kind::Fireball(f) => {
                self.x = self.x + f.process();  // shared plane, from the payload's tick
                if f.heat > 9 {                 // the parent may read the binding too
                    self.kind = Kind::Spring;   // change of kind
                }
            }
            _ => {}
        };
    }
}
```

Every access is still one `abs,X` instruction at the element index, and the
payload call passes X straight through; the price of the shape is one tag test
per call when the caller has already matched.

So when the caller *has* already matched — a per-frame loop dispatching on the
tag — put the mixing code in that arm, with the binding, instead of calling a
parent method that matches again:

```rust
for i in 0..16 {
    match POOL[i].kind {
        Kind::Fireball(f) => {
            POOL[i].x = POOL[i].x + f.process();
            if f.heat > 9 { POOL[i].kind = Kind::Spring; }
        }
        ...
    };
}
```

One tag test per element, no call; the payload's own methods still hold its
own state. classickong's `enemy_process` is this shape.

### Rules

**All or nothing.** Within one tagged enum, either every payload struct is
`#[soa]` or none is. Mixing is an error:

```rust
#[soa] struct Fireball { heat: u8, phase: u8 }
       struct Barrel   { rolling: u8 }

enum Kind { Barrel(Barrel), Fireball(Fireball) }
```

```
error: enum 'Kind' mixes #[soa] and plain payloads: 'Kind::Fireball' is #[soa],
       'Kind::Barrel' is not
hint: a tagged enum's data plane has one layout — either an array of union
      elements or a set of overlaid field planes; mark every payload struct
      #[soa], or none of them
```

The two layouts could coexist in the region — a plain payload is one
element-strided slot, an `#[soa]` one a set of planes, and the region would
simply be the larger — but then the enum's data plane has two stride
disciplines, arm bodies pay a different index cost depending on which variant
they matched, and a struct's `#[soa]`-ness is decided by whether it happens to
be used elsewhere. One layout per enum keeps the cost model readable from the
declaration. If one payload has to stay pointer-addressable (it also lives
outside the pool), keep them all plain and accept the pointer path.

**One storage, chained.** A `#[soa]` struct backs exactly one static. For a
payload struct that static is *the enum's*: `Fireball` → payload of `Kind` →
field of `Enemy` → `POOL`. Every link may appear once: a static of `Fireball`
alongside the enum, or `Fireball` as the payload of a second variant or a
second enum, is refused:

```
error: #[soa] struct 'Fireball' is the payload of 'Kind::Fireball' but is also
       stored by static 'FIRE'
error: #[soa] struct 'Fireball' is already the payload of 'Kind::Fireball'
```

**Every `#[soa]` restriction applies to the payload struct.** No locals, no
parameters, no `&`, no `*self`, no whole-element copy, no `impl Clone`, no
initializer on the pool. `POOL[i].kind = Kind::Fireball(TEMPLATE)` stays
rejected — write the fields.

**The receiver is the outer index.** `Fireball::burn` above only makes sense
called on an element of `POOL`; there is no other Fireball. That coupling is
what `#[soa]` means, and it is the same coupling `impl Kind` already has. It
is an implementation fact, not a handle: in the source `self` is the element,
and reaching the parent from it is rejected (see
[A payload's `self` is not a number](#a-payloads-self-is-not-a-number)).

### Known limits

- **Mixed-width payload fields recompute X on width switches**, as for any
  `#[soa]` struct. All-`u8` payloads — the common case for entity state — are
  the intended target.
- **A payload struct with an array or nested-aggregate field** inherits the
  `#[soa]` limits for those (`POOL[i].coords[j]` unsupported; 5- and 6-byte
  fields keep the shift-and-add).
- **Region size is the max variant, not the sum**, so a variant with many
  narrow fields and a variant with one wide field cost the same as they do
  today. There is no per-variant opt-out of the overlay.

### Implementation

- **Validation** — `HIRBuilder._classify_enum_payloads` (`hir/builder.py`),
  run from `_setup_tagged_enums` once every variant is resolved: the
  all-or-nothing check, then the payload-struct registry
  (`_payload_structs`) and the one-storage checks; the result is the value of
  `_tagged_enums[name]`, whose key pass 1 already added.
- **Planes** — `_declare_enum_planes` declares the per-variant field-plane
  symbols in pass 1.5, so bodies can name them; `_pad_enum_payloads` pads the
  `Enum__Payload` union to the region size; `_enum_plane_pair` appends the
  overlay entries, which `_expand_soa_statics` synthesizes with
  `soa_overlay = (region decl, byte offset)` and keeps *out* of the parent's
  `soa_planes` so nothing counts them twice.
- **Allocation** — `MemoryAllocator._allocate_overlay` (pass 3 of
  `allocate_all`) gives each overlay its region's address plus the offset and
  reserves nothing; `SymbolGenerator.emit_overlay_definitions` emits them as
  `.DEFINE`s. `MIRBuilder.plane_stride` makes every static array-index site
  honour a plane's padded `soa_stride` when the plane is reached by name.
- **Rewrite** — `ExpressionBuilder._soa_payload_access` turns `f.field` and
  `self.field` into `PLANE[i]` before the type checker; `build_enum_construction`
  stores per plane; a payload struct's method body runs with `payload_self`
  set and its `self` bound to the enum's pool.

Tests: `r65/tests/language/types/test_tagged_enums.py` (`TestSoaPayloadPlanes`,
`TestSoaPayloadRewrite`, and the rejections in `TestDeclaration`) and
`r65/tests/e2e/test_tagged_enums.py` (`TestSoaPayloadLayout`,
`TestSoaPayloadAccess`).

---

## Known limits

- **No initializers.** A tagged enum static cannot carry one — transposing an
  array literal into two planes is not implemented. SNES RAM is unpredictable at
  power-on, so **write every element before matching on it**. An uninitialized
  tag is a byte of garbage: the jump-table path bounds-checks it, and a compare
  chain is first-match-wins so it falls through to the last arm, but neither is
  a substitute for initializing.
- **One static per tagged enum type**, as for `#[soa]`.
- **A payload must be a declared struct** — plain or `#[soa]`, but not mixed
  within one enum (see [`#[soa]` Payloads](#soa-payloads)) — not a union,
  newtype, enum, primitive, array or pointer.
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
