# Unions

## Overview

A `union` is a named aggregate whose fields all occupy the **same bytes**. Every
field sits at offset 0, and the union's size is that of its largest field:

```rust
union Pixel {
    raw: u16,      // offset 0, 2 bytes
    bytes: [u8; 2] // offset 0, 2 bytes
}                  // size_of(Pixel) == 2
```

Writing one field and reading another reinterprets the same memory. There is no
tag, no discriminant, and no record of which field was written last — a union is
a compile-time view over a region of bytes, and it costs nothing at runtime.

Unions exist for two jobs on a memory-starved machine:

1. **RAM overlays** — several subsystems need scratch space but only one is live
   at a time. A union lets them share one allocation instead of each holding its
   own permanently.
2. **Type punning** — reading a `u16` as `[u8; 2]` for DMA setup, pointer
   splitting, or fixed-point work, without shifts and masks.

---

## Declaration

Same field syntax as a struct:

```rust
union Scene {
    battle_hp: u16,
    menu: [u8; 2],
}

#[zeropage(0x40)]
static mut SCENE: Scene;
```

**Rules**:
- At least one field is required — an empty union has no layout
- Fields may be any sized type: primitives, arrays, structs, other unions, pointers
- Packed, like structs — no alignment or padding
- `union` is a contextual keyword, exactly like `struct` and `enum`: declaring a
  union does not stop you naming a variable `union`

---

## Initialization

A union literal names **exactly one** field. The remaining bytes are zero-filled
to the union's size:

```rust
#[ram]
static mut P: Pixel = Pixel { raw: 0x1234 };   // OK

#[ram]
static mut Q: Pixel = Pixel { raw: 0, bytes: [1, 1] };
// error: union literal for 'Pixel' must initialize exactly one field, found 2
```

Naming two fields would be contradictory — they are the same bytes. A struct
literal, by contrast, still requires every field.

```rust
union State { flag: u8, buffer: [u8; 4] }

#[ram]
static mut S: State = State { flag: 0xAB };
// ROM data: $AB, $00, $00, $00
```

---

## Field Access

Identical in syntax and cost to struct field access — a union field read or write
compiles to exactly the instructions a struct field at offset 0 would:

```rust
P.raw = 0x1234;
let lo: u8 = P.bytes[0];   // 0x34
let hi: u8 = P.bytes[1];   // 0x12
P.bytes[1] = 0xFF;
let back: u16 = P.raw;     // 0xFF34
```

Rust requires `unsafe` to read a union field, because reading a field that was
never written is undefined there. R65 has no `unsafe` keyword and takes the same
stance it takes on array bounds: the hardware will happily read those bytes, and
it is the programmer's job to know what is in them. Reading a field you never
wrote yields whatever bytes are there — on the SNES, at power-on, that is
unpredictable.

**Nested access** works: a union inside a struct (or vice versa) is reached with
`OUTER.inner.field`. The chain folds to a single constant offset at compile time,
so it costs the same as a one-level access:

```rust
union Word { raw: u16, bytes: [u8; 2] }
struct Holder { tag: u8, w: Word }

H.w.raw = 0x1234;
let hi: u8 = H.w.bytes[1];   // 0x12
```

---

## Unions and Pointers

Pointers to unions work like pointers to structs, near and far:

```rust
fn hi_of(p: *Pixel) -> u8 { return p.bytes[1]; }
far fn far_hi(p: far *Pixel) -> u8 { return p.bytes[1]; }
```

Arrays of unions stride by the union's size:

```rust
#[ram]
static mut PIXELS: [Pixel; 64];   // 128 bytes
PIXELS[10].raw = 0x1234;
```

---

## Methods

Inherent `impl` blocks work:

```rust
impl Pixel {
    fn lo(*self) -> u8 { return self.bytes[0]; }
    fn hi(*self) -> u8 { return self.bytes[1]; }
}
```

**Unions cannot implement traits.** Trait dispatch stores a `__type_id: u8` at
offset 0 to identify the concrete type, and in a union offset 0 is live field
data — the TypeId and the fields would alias and corrupt each other:

```rust
impl Drawable for Pixel { ... }
// error: union 'Pixel' cannot implement trait 'Drawable'
```

The one exception is `Clone`, which is a bitwise copy of `size_of(Pixel)` bytes.
It reads no fields and adds no layout, so it is allowed:

```rust
impl Clone for Pixel { }        // OK
DEST.clone_from(&SOURCE);
```

The operator traits (`AddAssign`, `PartialEq`, ...) are rejected along with the
rest: arithmetic on an untagged overlay has no well-defined meaning.

---

## Layout Reference

| Property | Struct | Union |
|----------|--------|-------|
| Field offsets | Packed, running total | All 0 |
| Size | Sum of field sizes | Largest field size |
| Literal | Must name all fields | Must name exactly one |
| Traits | Yes | No (except `Clone`) |
| `*dyn` dispatch | Yes | No |

Both layouts are computed by `layout_fields` in
`r65/compiler/hir/unified_type_utils.py` — the single definition of aggregate
layout in the compiler.

---

## Example: Splitting a Far Pointer

```rust
union FarAddr {
    addr: far *u8,   // 3 bytes
    parts: [u8; 3],
}

#[zeropage(0x30)]
static mut TARGET: FarAddr;

fn setup_dma() {
    TARGET.addr = SPRITE_DATA;
    DMA_SRC_LO   = TARGET.parts[0];
    DMA_SRC_HI   = TARGET.parts[1];
    DMA_SRC_BANK = TARGET.parts[2];
}
```

Without a union this needs shifts, masks, and a bank-byte extraction; with one it
is three loads.
