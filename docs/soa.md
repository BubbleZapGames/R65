# SoA Struct Arrays — `#[soa]`

**Status: implemented** — layout, methods and nested fields all work and
are covered by emulator tests. Initializer transposition and `*dyn` over an
SoA element are not supported, and one language-wide codegen bug affects
`#[soa]` alongside everything else; see [Known limits](#known-limits).

`#[soa]` marks a **struct**. Every array of that struct is laid out as one
array per field — a *plane* — so `arr[i].field` addresses `arr__field[i]`
and the index scales by the *field* size instead of the *element* size.
The struct's methods take `self` as an element **index**, not a pointer,
so `self.hp` is a single indexed access.

```rust
#[soa]
struct Enemy { kind: u8, hp: u8, x: u8, y: u8 }

impl Enemy {
    fn hurt(self) { self.hp = self.hp - 1; }
}

#[lowram]
static mut POOL: [Enemy; 16];

POOL[i].hurt();          // -> Enemy__hurt(i), index in X
POOL[i].hp = 100;        // -> STA POOL__hp,X — stride 1, no multiply
```

A `#[soa]` struct backs **exactly one** static. That is what lets every
method hard-code its plane addresses instead of carrying a base pointer.

---

## Motivation

`arr[i].field` scales the index by `sizeof(T)`. `compute_scaled_index`
([`mir/lowerers/multiply.py:273`](../r65/compiler/mir/lowerers/multiply.py))
picks the strategy:

| `sizeof(T)` | Lowering | Cost |
|---|---|---|
| 1 | index used as-is | free |
| 2, 4, 8, 16 | one `<<` | ~4 cycles |
| 3, 5, 6, 9, 10, 12 | shift + add | ~12–16 cycles |
| 7, 15 | shift − subtract | ~12–16 cycles |
| 11, 13, 14 | two shifts + two ops | ~20 cycles |
| **> 16** | **`JSL mul16`** | **~57 cycles** |

The decomposition table is
[`multiply.py:102-114`](../r65/compiler/mir/lowerers/multiply.py); the
`mul16` call is at `multiply.py:338`. The ~57-cycle figure is measured, not
estimated — see *Strength-reduce a multiply by a constant* in
[`outstanding-optimization-issues.md`](../outstanding-optimization-issues.md),
which clocks classickong's per-frame `mul8(i, 3)` at ~57 cycles against ~16
for the inline `(i<<1)+i`.

Above 16 bytes the cost is not just cycles. The call requires
`include!("lib/math.r65")` (`_check_runtime_mul_available`,
`multiply.py:39`), clobbers registers, and — per the module docstring at
`multiply.py:8-24` — silently truncates the index to 0–255.

### What this costs in practice

No shipped R65 game uses an array of structs for its entity pool. Every
one hand-writes parallel arrays, and several say so in a header comment:

| Site | Hand-written SoA |
|---|---|
| `classickong/src/enemy.r65:48-90` | **15** parallel `[_; 16]` arrays — as a struct this is ~18 bytes, i.e. the `mul16` path |
| `classickong/src/items.r65:55-104` | 3 + 6 arrays (items, particles) |
| `classickong/src/kong.r65:76-88, 119-122` | 5 arrays × 22 (ladders), 2 × 8 (fireballs) |
| `rotor_raiders/src/muni.r65:49-58` | 8 parallel `[_; 16]` arrays |
| `rotor_raiders/src/unit.r65:146-156` | 8 parallel `[_; 16]` arrays |
| `astroids/src/game.r65:61-68` | `B_X`/`B_Y`, `A_X`/`A_Y` |
| `pacman/src/game.r65:629-690` | OAM flattened to `[u8; 544]` with a manual `i << 2` |

`rotor_raiders/src/muni.r65` states the reason outright:

> Struct-of-arrays with a rotating free cursor… `*` only accepts
> power-of-two constants and mul8 costs ~57 cycles against ~16 for a
> shift, so every index arithmetic here stays in shifts and masks.

And where an array of structs *is* used, the struct is hand-padded to a
power of two:

- `pacman/src/game.r65:100` — `struct Ghost` carries an explicit
  `pad: u8` to reach 8 bytes, across 58 access sites.
- `textadv/src/world.r65:1-27` — three structs padded to 8/8/4, with a
  header comment explaining the compiler constraint.

`#[soa]` is the language-level answer: keep the struct, lose the stride.

---

## Syntax

`#[soa]` attaches to a **struct declaration**:

```rust
#[soa]
struct Enemy { kind: u8, hp: u8 }
```

Every static of that struct is then SoA, whatever its storage class:

```rust
#[lowram]         static mut POOL: [Enemy; 16];   // lowram planes
#[zeropage(0x40)] static mut POOL: [Enemy; 8];    // explicit, consecutive
                  static mut POOL: [Enemy; 16];   // auto RAM
                  static mut PLAYER: Player;      // pool of one
```

`#[soa]` composes with `#[zeropage]`, `#[lowram]`, `#[ram]` and
`#[bank(n)]` on the static. It does **not** compose with `#[hw]` or
`extern static`, and a `#[soa]` struct may back only one static.

A **single instance** desugars to a one-element array, so `PLAYER.hp` is
`PLAYER[0].hp` and `PLAYER.heal()` passes index 0. Nothing downstream
needs a singleton special case.

Access is unchanged from the array-of-structs form:

```rust
POOL[i].hp = 100;
let k: u8 = POOL[i].kind;
POOL[i].hp += 1;              // compound assignment works
POOL[i].pos.y = 8;            // nested inline aggregates work
```

---

## Methods: `self` is an element index

An `#[soa]` struct's methods take `self` **by value**, and that value is
the element index. It arrives in **X**, which is where indexed addressing
wants it, so a field access inside a method is a single instruction:

```rust
impl Enemy {
    fn step(self) { self.x = self.x + 1; self.y = self.y + 2; }
}
```

```asm
Enemy__step:
    LDA POOL__x,X
    INC A
    STA POOL__x,X
    LDA POOL__y,X
    CLC / ADC #$02
    STA POOL__y,X
    RTS
```

No pointer, no stack frame, no setup. `self` is an ordinary `u16`, so
`if self == 0`, `other_fn(self)` and `POOL[self].hp` all work. That holds for
an `#[soa]` struct's own methods; a struct that is a tagged enum's payload is
stricter — its `self` is an element, not a number (see
[enums.md](enums.md#a-payloads-self-is-not-a-number)).

Writing `*self` on an `#[soa]` method is an error — there is no address to
take. This follows the same "one self form per type" rule as newtypes: a
trait declared with `*self` is still implemented by an `#[soa]` struct with
bare `self`.

```rust
trait Tick { fn tick(*self); }
impl Tick for Enemy { fn tick(self) { self.hp = 0; } }   // index self
```

Register budget: `self` occupies X for the whole body, so an `#[soa]`
method may not bind a parameter to `X`.

### Inlined on a constant element

A method shared by a small fixed set of elements - a player and an enemy,
`POOL[0]` and `POOL[1]` - is usually called on a constant. Marked
`#[inline(always)]`, such a call compiles to the code a direct `POOL[0].f`
does: the inliner substitutes the constant for `self`, and the
constant-index pass (`optimize/const_index.py`, after inlining at `-O1`)
rewrites every `POOL__f,X` whose X the block set to a known constant into
the absolute `POOL__f+k`, then drops the X loads nothing reads any more.

```rust
impl Air {
    #[inline(always)]
    fn step(self) { if self.landed == 0 { self.x = self.x + 1; } }
}
P[1].step();          // LDA P__landed+1 / BNE / INC P__x+2 - no LDX, no ,X
P[i].step();          // a variable element: the call, or POOL__f,X inlined
```

Not inlined, the method pays what an index method pays per call - the
`JSR`, X reloaded at each switch between field widths (see
[Index reuse in X](#index-reuse-in-x--read-this-before-choosing-field-types)),
a frame for its locals - which on a per-frame routine of a game on its frame
budget is measurable: rotor_raiders' shared airframe draw cost 14 ticks a
minute of its stress run that way, and inlined it ran 21 faster than the two
hand-written copies it replaced. The fold is block-local and conservative:
it trusts X only across loads, stores, ALU ops, compares and branches, and
deletes an X load only when a CFG liveness pass - in which a call, inline
asm, a block move or a trait dispatch counts as reading X - finds it dead.

---

## Layout

For `#[soa] struct Enemy { kind: u8, x: u16, hp: u8 }` and
`static mut POOL: [Enemy; 16];`, the compiler synthesizes one ordinary
static per top-level field, in declaration order:

| Synthesized static | Field type | Stride | Bytes |
|---|---|---|---|
| `POOL__kind` | `u8` | 1 | 16 |
| `POOL__x` | `u16` | 2 | 32 |
| `POOL__hp` | `u8` | 1 | 16 |

Each inherits the parent's storage attribute, bank, and mutability. **The
parent occupies zero bytes** — it survives only so that `POOL[i].hp`
type-checks.

### Stride: rounded up when it costs at most one byte

A field plane's **stride** is normally the field's size, but a field of 3
or 7 bytes is padded to 4 or 8 — one byte of waste per element buys a
power-of-two scale, so the index becomes a shift instead of a
shift-and-add:

```
stride(field) = 4  if sizeof(field) == 3      (far *T, far fn(), 3-byte aggregates)
              = 8  if sizeof(field) == 7
              = sizeof(field)  otherwise
```

The rule stops at one byte. A 5-byte field would need 3 bytes of padding
to reach 8 (60% waste) and a 6-byte field 2 bytes (33%); neither is worth
it for one `CLC`/`ADC` pair, so those keep the shift-and-add. See
[Padded fields](#stride-3-vs-stride-4-the-padding-rule) for the measured difference and
[Codegen](#stride-3-vs-stride-4-the-padding-rule) for the emitted code.

So for `#[soa] struct Area { name: far *u8, exits: u8 }` and
`static AREAS: [Area; 32];`, the layout is:

| Synthesized static | Field type | Stride | Bytes |
|---|---|---|---|
| `AREAS__name` | `far *u8` (3) | **4** | 128 |
| `AREAS__exits` | `u8` | 1 | 32 |

160 bytes against the AoS form's 128 — the 32 pad bytes are the price of
`ASL A / ASL A` instead of a spill, a shift, a `CLC`, and an `ADC`.

Note this makes the SoA form *larger* than a hand-padded AoS struct in one
specific case: `struct Area { name: far *u8, exits: u8 }` is already 4
bytes, so AoS already gets a power-of-two stride for free. SoA wins where
the element has several fields, not where padding one field already fixed
the whole struct.

RAM emission is ordinary `.RAMSECTION`:

```asm
.RAMSECTION "lowram" BANK 0 SLOT 1
    POOL__kind dsb 16
    POOL__x    dsb 32
    POOL__hp   dsb 16
.ENDS
```

**Auto-allocated** children are placed by the normal allocator and may not
be adjacent. **Explicitly addressed** children are laid out consecutively
from the parent address:

```
addr(field_k) = parent_address + Σ_{j<k}  N · stride(field_j)
```

So `#[zeropage(0x40)] static mut POOL: [Enemy; 8];` over a `#[soa]`
`Enemy` puts
`POOL__kind` at `$40`, `POOL__x` at `$48`, `POOL__hp` at `$58`.
This keeps the memory map writable by hand and gives the hand-written
parallel arrays a mechanical migration path.

The `__` separator is not spellable in R65 source, so `POOL__kind`
cannot collide with a user identifier by accident. A deliberate collision
is caught by the existing `Redefinition of '...'` error in
`SymbolTable.declare`.

---

## Restrictions

An SoA element has no contiguous base address, so everything needing one is
rejected — with a diagnostic naming the construct the author wrote, never a
late allocation failure naming a synthesized plane.

| Construct | Why it cannot work | Write instead |
|---|---|---|
| `&POOL[i]` | The element's bytes are in one array per field | Pass the index `i` |
| `&POOL` | Same, for the whole pool | Name a field plane |
| `&POOL[i].field` | A plane's next byte is the *next element's* same field, so the pointer walks the plane, not the element | Read/write `POOL[i].field` directly, or copy it to a local |
| `let e: Enemy;` | No contiguous value to hold | Hold a `u16` index |
| `fn f(e: Enemy)` / `fn f(e: *Enemy)` | Nothing to pass or point at | Take the index |
| `struct W { lead: Enemy }` | Would need a plane inside an aggregate | Store the index |
| — | *but* a field may be a **tagged enum**: it contributes its own tag and payload planes (`POOL__kind__tag`, `POOL__kind__data`) — see [enums.md](enums.md) | — |
| — | *and* an `#[soa]` struct may be a **tagged enum's payload**, provided every payload of that enum is `#[soa]`: its fields become planes over the enum's data region and its methods take the pool index — see [enums.md, `#[soa]` Payloads](enums.md#soa-payloads) | — |
| `E = POOL[i]` (whole element) | Aggregate by value | Copy the fields |
| `POOL.clone()` | Needs one contiguous block | Copy per field |
| `impl Clone for Enemy {}` | Clone means a contiguous byte copy | Copy the fields |
| a second static of the type | Methods hard-code one pool's addresses | Merge the pools |
| `#[soa] union U { .. }` | Fields overlay at offset 0 | Keep it a union |
| `#[soa] struct T(u8);` | A newtype is a scalar, not an aggregate | — |
| `#[soa]` with `#[hw]` | Hardware registers are not placed by the compiler | — |
| `#[soa]` with `extern static` | Placement is owned by the asm file | — |
| `fn m(*self)` on an SoA struct | No address to point at | `fn m(self)` |
| `[E { .. }; N]` / `[0; N]` fill on the static | No per-plane form yet | Write the array literal out, or assign at runtime |
| `static POOL: [E; N];` with no initializer | ROM with nothing in it | Add `mut`, or an array literal |

Three things that are **allowed**, because the obvious guess is wrong:

- **Trait impls.** `impl Tick for Enemy { fn tick(self) }` works, with the
  index receiver. The receiver form follows the implementing type, not the
  trait declaration — the same rule newtypes use. Only `impl Clone` is
  rejected, and only because Clone specifically means a contiguous copy.
- **An explicit address**, per the layout rule above.
- **Methods that take arguments**, and methods that call other methods on
  `self` — `self.mark()` inside `self.kill()` passes the index straight
  through.

`&POOL[i].field` is rejected deliberately. The address is well-defined —
one byte of one plane — but it is not the element's: `p + 1` is the same field
of element `i + 1`, which is the opposite of what a field pointer means for an
array of structs. The same check covers `&self.field` in an `#[soa]` method and
a payload binding's field (`&f.heat`), which are the same thing by the time
the type checker sees them.

### Migration note

An array of plain structs supports `&ARR[i]`, and `textadv` depends on it:

```rust
let a: far *Area = &AREAS[CUR as u16] as far *Area;   // not #[soa]
```

Adding `#[soa]` to `Area` is therefore a breaking change at that call site,
by design. The rewrite is to index each field directly rather than take a
pointer once and dereference it repeatedly — which is also what removes the
stride multiply, so the two go together.

---

## Initializers

An array literal on an `#[soa]` static is transposed into one literal per
plane, and each plane then takes the ordinary array-literal path. On an
immutable static that is one ROM table per field — the natural shape for a
per-kind constant table indexed by a tag:

```rust
#[soa] struct Area { id: u8, exits: u16, flags: u8 }
static AREAS: [Area; 3] = [
    Area { id: 0, exits: 3, flags: 0x01 },
    Area { id: 1, exits: 1, flags: 0x00 },
    Area { id: 2, exits: 2, flags: 0x04 },
];

let n: u16 = AREAS[i].exits;     // ASL A / TAX / LDA.l __AREAS__exits_data,X
```

```asm
__AREAS__id_data:     .db $00, $01, $02
__AREAS__exits_data:  .db $03, $00, $01, $00, $02, $00
__AREAS__flags_data:  .db $01, $00, $04
```

Each column is read with the raw index scaled by the *field* width — a
`u8` column with no scaling at all — where an array of plain structs
scales by the element size and adds the field offset. A padded plane (a
3-byte `far *T` at stride 4) is padded in ROM too, so the index and the
table agree.

On a `static mut` the same transposition runs and each plane is
block-copied from its ROM column by `__init_start`, exactly as a plain
array literal would be.

The literal is checked at transposition — one element per slot, each a
literal of the element struct naming every field once — because the
struct-literal validator runs after the static is built and would only see
the columns. Field values are type-checked against their plane's element
type as usual. A fill `[.. ; N]` is still rejected: a struct fill does not
parse, and a scalar fill has no per-plane meaning.

A single instance takes its own struct literal - `static mut PLAYER:
Player = Player { .. };` - which is transposed as a pool of one; an array
literal there is rejected, the declared type not being an array.

A field may itself be an aggregate - an array, a struct, a byte string -
and its plane is filled from the literal's nested values, padded to the
plane's stride. An array field is indexed as usual - `POOL[i].arr[j]`,
or `self.arr[j]` in a method - and X takes `i * stride + j`, the stride
being the plane's (padded) one.

A tagged enum static (docs/enums.md) has no literal form and still takes
no initializer.

---

## Codegen

All four listings below are real `r65c` output for
`ARR[i].health = 1;` against a `#[lowram]` array with a `#[zeropage]`
index, trimmed of the `.ACCU`/`.INDEX` directives. The SoA column is the
hand-written parallel-array equivalent — i.e. exactly what `#[soa]` will
emit.

### The target: stride 1

```asm
; SoA — POOL__health[i] = 1
REP #$20
LDA $00
TAX
SEP #$20
LDA #$01
STA $0200,X
```

### Stride 4 — two `ASL`s saved

`struct Ghost { px: u8, py: u8, dir: u8, state: u8 }`:

```asm
REP #$20
LDA $00
ASL A            ; <-- gone under SoA
ASL A            ; <--
TAX
SEP #$20
LDA #$01
STA $0202,X
```

Real but small. This is why the corpus hand-pads to a power of two rather
than restructuring — `pacman`'s `struct Ghost` carries a `pad: u8` for
exactly this.

### Stride 9 — a stack frame appears

A 9-byte element (the shape in `r65/tests/language/test_acid.py:71`)
lowers to `(i<<3) + i`, and the `+ i` needs the original index back after
three shifts have destroyed it. There is no free register, so the function
**allocates a 2-byte stack frame** it would not otherwise need:

```asm
PHX              ; allocate frame (2 bytes)   <-- gone under SoA
REP #$20
LDA $00
STA $01,S        ; spill index                <-- gone
ASL A
ASL A
ASL A
CLC
ADC $01,S        ; reload index               <-- gone
TAX
SEP #$20
LDA #$01
STA $0208,X
PLA              ; deallocate frame           <-- gone
PLA              ;                            <-- gone
```

The cost is not the three shifts — it is the frame, the spill, and the
reload, all of which vanish.

### Stride 18 — the call

Past the decomposition table (>16 bytes), which is where classickong's
~18-byte enemy record lands:

```asm
REP #$20
LDA $00
REP #$20
PHA              ; push u16 arg
LDA #$12
JSR mul16        ; ~57 cycles
PLX              ; pop arg
TAX
SEP #$20
LDA #$01
STA $0211,X
```

Plus the obligation to `include!("lib/math.r65")`, the register clobber,
and the silent truncation of the index to 0–255 (`multiply.py:8-24`).
This is the case that makes the feature worth having, and the reason
`classickong/src/enemy.r65` is 15 hand-written arrays.

### Stride 3 vs stride 4 — the padding rule

Both listings read a `far *u8` field out of an array element. The only
difference is one pad byte per element.

```asm
; stride 3 — sizeof(far *u8)      ; stride 4 — padded
REP #$20                          REP #$20
LDA $00                           LDA $00
STA $01,S      ; spill index      ASL A
ASL A                             ASL A
CLC                               TAX
ADC $01,S      ; reload index     SEP #$20
TAX                               LDA $0200,X
SEP #$20
LDA $0200,X
```

`(i<<1) + i` needs the original index back after the shift has destroyed
it, and there is no free register — so it round-trips through the stack.
Padding to 4 removes the spill, the reload, the `CLC` and the `ADC`, at a
cost of one byte per element.

This is also why the rule exists at all rather than being left to the
programmer: an SoA field plane is a synthesized static with no name in the
source, so there is nowhere to write a `_pad` field by hand.

<a name="padded-fields"></a>

### Measured against hand-written parallel arrays

`rotor_raiders/src/muni.r65` keeps its munitions pool as eight parallel
arrays, written that way specifically to dodge the stride multiply. Porting
it to one `#[soa] struct` with a method is not a wash — it comes out ahead:

| | index recomputations | instructions |
|---|---|---|
| eight hand-written arrays | 4 | 31 |
| one `#[soa]` struct + method | 2 | 25 |

Same body, same storage class, same four field updates. The difference is
the X-reuse cache: `#[soa]` shares one scaled index across consecutive
fields of equal width (`x_px` and `y` are both stride 2; `life` and `kind`
are both stride 1), where the hand-written form recomputes for every array
because each is a separate symbol with its own base.

So the hand-written idiom is not a performance floor the feature has to
match — it is a ceiling the feature clears.

### Index reuse in X — read this before choosing field types

For AoS, X holds `i * sizeof(T)`, which is the same value for **every**
field of an element, so consecutive accesses to one element share one X:

```rust
POOL[i].hp = 0;      // computes X
POOL[i].kind = 0;    // reuses X
POOL[i].y = 0;       // reuses X
```

For SoA, X holds `i * sizeof(field)`. It is shared only among fields of
**the same width**, and the reuse cache (`MIRBuilder._x_index_cache`,
`mir/builder.py:2516`) is a **single slot**. So an interleaved
mixed-width sequence recomputes X on every switch:

```rust
POOL[i].hp   = 0;    // u8  - X = i
POOL[i].x    = 0;    // u16 - X = i<<1, recompute
POOL[i].kind = 0;    // u8  - X = i, recompute again
```

Consequences, in order of importance:

1. **For an all-`u8` or all-`u16` element, SoA is a strict win** — X is
   computed once per element and shared across every field, exactly as in
   AoS, but with a cheaper (often free) scale. The corpus pools are
   overwhelmingly all-`u8`, which is the intended target.
2. **For a mixed-width element, SoA can be a loss.** Trading one `mul16`
   for several shift+`TAX` pairs is not automatically favourable. Group
   same-width field accesses together, or keep the array AoS.
3. Widening the cache to one slot per scale would remove the thrash. Not
   specified here; it is an optimization, not a correctness requirement.

---

## Known limits

- **Padding costs memory.** A 3- or 7-byte field wastes one byte per
  element (see [Stride](#stride-rounded-up-when-it-costs-at-most-one-byte)).
  For a `[Area; 32]` with one `far *u8` field that is 32 bytes. There is
  no opt-out; if the waste matters, keep the array AoS.
- **5- and 6-byte fields keep the shift-and-add.** Reaching 8 would cost 3
  and 2 bytes per element respectively, past the one-byte rule. Such a
  field is a nested aggregate, since no scalar is 5 or 6 bytes.
- **Byte-plane splitting is not specified.** Splitting a `far *T` field
  into three parallel `[u8; N]` planes would make every access stride-1
  with no padding at all, but costs the field its contiguous address and
  needs three loads to reassemble a pointer. The padding rule gets most of
  the benefit for one byte, so this is left out.
- **`POOL[i].coords[j]`** — an array field inside an SoA element — is
  unsupported. It is already unsupported for AoS
  (`mir/builder.py:2030`), so this is not a new restriction.
- **`asm!` cannot see the layout.** A bare `LDA POOL,X` inside an
  `asm!` body is passed through verbatim (`hir/builder.py:2567` only
  substitutes const-evaluated `{name}` placeholders) and fails at WLA link
  with an undefined-symbol error. Name the field array instead.
- **`size_of(Enemy)`** still returns the packed AoS sum. That number is no
  longer a stride, and using it as one is a bug.
- **`*dyn Trait` over an SoA element is rejected.** The intended design is
  a packed `u16` handle (high byte TypeId, low byte index) plus a `TAX`
  trampoline per implementor, since `JMP (table,X)` needs X for the table
  and the callee needs X for the index. Static-dispatch traits work today
  and cost nothing; this only affects `*dyn`.
- **One pool per type.** Two entity pools that would naturally share a
  struct must either be merged or declared as two structs. An `#[soa]` struct
  that is a tagged enum's payload has no pool of its own at all — the enum's
  is its one storage.

---

## Implementation status

| Landed | |
|---|---|
| `#[soa]` on the struct declaration | grammar, AST, parser, pass-1 validation |
| One storage per type | binding discovered in pass 1 |
| Single-instance desugar | `T` -> `[T; 1]`, index 0 |
| Rejections | locals, params, fields, `&`, whole element, clone, `*self`, `#[hw]`, `extern` |
| Index-`self` methods | `SelfForm.INDEX_IN_X`, receiver passes the index |
| Plane synthesis | one static per field, parent allocated no bytes |
| Funnel rewrite | plane substitution + stride scaling + X reuse key |
| Nested chains | `POOL[i].pos.y`, including inside methods |
| Lint | three per-rule guards (dead-static, naming, zeropage budget) |

| Not yet | |
|---|---|
| `*dyn` over an SoA element | packed handle + trampoline, designed not built |

### Notes for whoever picks this up

**The base rewrite is the load-bearing trick.** `self.hp` and `PLAYER.hp`
are rewritten to `POOL[self].hp` and `PLAYER[0].hp` in
`hir/expression_builder.py`, *before* the type checker runs. That is why
`self` can stay a plain `u16` with no new `TypeInfo`, and why all three
receiver spellings collapse into one arm at the call site.

**Substitute the plane memloc first.** In the MIR funnel the parent static
is allocated no bytes, and `get_memory_location` does not fail on it — it
hands back a location naming the parent, which reaches the assembler as an
undefined symbol the author never wrote. The substitution must happen
before anything reads `base_memloc`.

**The X reuse key is correctness, not optimization.** X holds
`index * stride(field)`, so `POOL[i].hp` (u8) followed by `POOL[i].pos`
(u16) must not reuse the cached value. The stride is folded into the key;
position `[1]` stays the index symbol so cache invalidation still works.

**The X round trip did not materialize.** An earlier draft of this plan
worried that an X-bound `self` would spill to a vreg at entry and reload
per access. It does not: a straight-line method emits `LDA POOL__x,X` with
zero setup. A spill appears only across a real `JSR`, where it is
necessary because the callee may clobber X.

---

## References

- [Struct Array Indexing](struct-array-indexing.md) — the AoS stride strategy
- [Operators and Cost Model](operators.md) — why `*` is restricted
- [Pointers and Memory Model](pointers-memory.md) — storage classes
- [Unions](unions.md) — why a union element cannot be split
- [Traits](traits.md) — the TypeId byte at offset 0
