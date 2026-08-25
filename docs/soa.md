# SoA Struct Arrays — `#[soa]`

**Status: specified, not implemented.** This document is the design; see
[Implementation Plan](#implementation-plan) for the slice ordering.

`#[soa]` lays a `static [T; N]` out as parallel per-field arrays. The
source keeps struct syntax and a single `N`; the compiler emits one array
per field, so `arr[i].field` addresses `arr__field[i]` and the index is
scaled by the *field* size instead of the *element* size.

```rust
struct Enemy { kind: u8, x: u8, y: u8, hp: u8 }

#[lowram]
#[soa]
static mut ENEMIES: [Enemy; 16];

fn hurt(i @ X: u16) {
    ENEMIES[i].hp = ENEMIES[i].hp - 1;   // stride 1, no multiply
}
```

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

`#[soa]` attaches to a `static` or `static mut` whose type is `[Struct; N]`.

```rust
#[soa]                       static mut POOL: [Enemy; 16];      // auto RAM
#[lowram] #[soa]             static mut POOL: [Enemy; 16];      // lowram
#[zeropage(0x40)] #[soa]     static mut POOL: [Enemy; 8];       // explicit
#[soa]                       static TABLE: [Area; 32] = [ ... ]; // ROM
```

Attribute order does not matter. `#[soa]` composes with `#[zeropage]`,
`#[lowram]`, `#[ram]`, and `#[bank(n)]`. It does **not** compose with
`#[hw]` or `extern static`.

Access is unchanged from the AoS form:

```rust
ENEMIES[i].hp = 100;
let k: u8 = ENEMIES[i].kind;
ENEMIES[i].hp += 1;           // compound assignment works
ENEMIES[i].pos.y = 8;         // nested inline aggregates work
```

---

## Layout

For `#[soa] static mut ENEMIES: [Enemy; 16]` over
`struct Enemy { kind: u8, x: u16, hp: u8 }`, the compiler synthesizes one
ordinary static per top-level field, in declaration order:

| Synthesized static | Field type | Stride | Bytes |
|---|---|---|---|
| `ENEMIES__kind` | `u8` | 1 | 16 |
| `ENEMIES__x` | `u16` | 2 | 32 |
| `ENEMIES__hp` | `u8` | 1 | 16 |

Each inherits the parent's storage attribute, bank, and mutability. **The
parent occupies zero bytes** — it survives only so that `ENEMIES[i].hp`
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
[Padded fields](#padded-fields) for the measured difference and
[Codegen](#stride-3-vs-stride-4-the-padding-rule) for the emitted code.

So for `struct Area { name: far *u8, exits: u8 }`, `#[soa] static AREAS:
[Area; 32]` lays out:

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
    ENEMIES__kind dsb 16
    ENEMIES__x    dsb 32
    ENEMIES__hp   dsb 16
.ENDS
```

**Auto-allocated** children are placed by the normal allocator and may not
be adjacent. **Explicitly addressed** children are laid out consecutively
from the parent address:

```
addr(field_k) = parent_address + Σ_{j<k}  N · stride(field_j)
```

So `#[zeropage(0x40)] #[soa] static mut ENEMIES: [Enemy; 8]` puts
`ENEMIES__kind` at `$40`, `ENEMIES__x` at `$48`, `ENEMIES__hp` at `$58`.
This keeps the memory map writable by hand and gives the hand-written
parallel arrays a mechanical migration path.

The `__` separator is not spellable in R65 source, so `ENEMIES__kind`
cannot collide with a user identifier by accident. A deliberate collision
is caught by the existing `Redefinition of '...'` error in
`SymbolTable.declare`.

---

## Restrictions

An SoA element has no contiguous base address. Everything that needs one
is rejected — at the **type checker**, with a diagnostic naming the
source construct, never as a late allocation failure naming a synthesized
symbol the user never wrote.

| Construct | Why it cannot work | Write instead |
|---|---|---|
| `&ENEMIES[i]` | The element's bytes are in 3 different arrays | Pass `i` |
| `&ENEMIES` | Same, for the whole array | Name a field array |
| `ENEMIES[i].method()` | Desugars to `&ENEMIES[i]` (`typeck/call_validator.py:358`) | Free fn taking `i` |
| `let e = ENEMIES[i];` | Aggregate by value | Copy fields |
| `ENEMIES.clone()` | Needs one contiguous block | Per-field copy |
| element `impl`s a `*dyn` trait | Dispatch reads a TypeId byte at the element's offset 0 (`codegen/codegen.py:475`) | Keep AoS, or a type-switch |
| element is a `union` | Fields overlay at offset 0; splitting them destroys the aliasing | Keep AoS |
| `#[soa]` + `#[hw]` | Hardware registers are not an array the compiler places | — |
| `#[soa]` + `extern static` | Placement is owned by the asm file | — |

Two things that are **allowed** and worth stating explicitly, because the
obvious guess is wrong:

- **`impl Clone for Enemy {}` is fine.** Lang-item impls route through
  `_declare_lang_item_impl` (`hir/builder.py:1705`) and never populate
  `_struct_type_ids`, so they inject no TypeId byte and do not change
  layout. The restriction is on `*dyn`-used traits only —
  `struct_name in _struct_type_ids` (`hir/builder.py:1540`) is the exact
  test. `Enemy.clone()` on a *local* `Enemy` is unaffected; only
  `ENEMIES.clone()` on the SoA array itself is rejected.
- **An explicit address is fine**, per the layout rule above.

`&ENEMIES[i].field` — the address of a single *field*, which is
well-defined under SoA — is **also rejected**, but only because it does
not work for AoS either today (`mir/lowerers/expression.py:1124` requires
an identifier base). Supporting it is a natural follow-up.

### Migration note

`&ARR[i]` **compiles today** for a non-SoA array, and `textadv` depends on
it:

```rust
let a: far *Area = &AREAS[CUR as u16] as far *Area;   // AoS only
```

Adding `#[soa]` to such an array is therefore a breaking change at that
call site, by design. The rewrite is to index each field directly rather
than take a pointer once and dereference it repeatedly — which is also
what removes the stride multiply, so the two go together.

---

## Initializers

A ROM SoA table is written as an ordinary array literal and transposed at
compile time:

```rust
struct Area { id: u8, exits: u8, flags: u8 }

#[soa]
static AREAS: [Area; 3] = [
    Area { id: 0, exits: 3, flags: 0x01 },
    Area { id: 1, exits: 1, flags: 0x00 },
    Area { id: 2, exits: 2, flags: 0x04 },
];
```

emits three tables rather than one interleaved one:

```asm
__AREAS__id_data:     .db $00, $01, $02
__AREAS__exits_data:  .db $03, $01, $02
__AREAS__flags_data:  .db $01, $00, $04
```

For a `static mut` the same three tables are emitted and block-copied to
RAM at `__init_start()`, one `BlockCopy` per field.

A padded field emits its pad byte in the table, so the `.db` rows stay a
power of two wide:

```asm
; static AREAS: [Area; 3] where Area { name: far *u8, exits: u8 }
__AREAS__name_data:   .db $00,$80,$C0,$00,  $10,$80,$C0,$00,  $2A,$80,$C0,$00
__AREAS__exits_data:  .db $03, $01, $02
```

**Supported**: an array literal whose every element is a struct literal of
the element type, with every field present. Field order inside each
literal does not matter.

**Not supported** in this design:

- **A non-scalar top-level field.** If a field is itself a struct or
  array, its per-field literal has aggregate elements, which
  `_extract_initializer_bytes` (`mir/lowerers/static_init.py`) does not
  handle. Rejected with a diagnostic.
- **The fill form `[Enemy { .. }; N]`.** This is a *parse* error today,
  independent of SoA — the grammar does not accept a struct literal as the
  repeated value. Nothing to transpose until that is fixed.

---

## Codegen

All four listings below are real `r65c` output for
`ARR[i].health = 1;` against a `#[lowram]` array with a `#[zeropage]`
index, trimmed of the `.ACCU`/`.INDEX` directives. The SoA column is the
hand-written parallel-array equivalent — i.e. exactly what `#[soa]` will
emit.

### The target: stride 1

```asm
; SoA — ENEMIES__health[i] = 1
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

### Index reuse in X — read this before choosing field types

For AoS, X holds `i * sizeof(T)`, which is the same value for **every**
field of an element, so consecutive accesses to one element share one X:

```rust
ENEMIES[i].hp = 0;      // computes X
ENEMIES[i].kind = 0;    // reuses X
ENEMIES[i].y = 0;       // reuses X
```

For SoA, X holds `i * sizeof(field)`. It is shared only among fields of
**the same width**, and the reuse cache (`MIRBuilder._x_index_cache`,
`mir/builder.py:2516`) is a **single slot**. So an interleaved
mixed-width sequence recomputes X on every switch:

```rust
ENEMIES[i].hp   = 0;    // u8  - X = i
ENEMIES[i].x    = 0;    // u16 - X = i<<1, recompute
ENEMIES[i].kind = 0;    // u8  - X = i, recompute again
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
- **`ENEMIES[i].coords[j]`** — an array field inside an SoA element — is
  unsupported. It is already unsupported for AoS
  (`mir/builder.py:2030`), so this is not a new restriction.
- **`asm!` cannot see the layout.** A bare `LDA ENEMIES,X` inside an
  `asm!` body is passed through verbatim (`hir/builder.py:2567` only
  substitutes const-evaluated `{name}` placeholders) and fails at WLA link
  with an undefined-symbol error. Name the field array instead.
- **`size_of(Enemy)`** still returns the packed AoS sum. That number is no
  longer a stride, and using it as one is a bug.

---

## Implementation plan

Six slices; each compiles and is independently testable.

### S1 — attribute and restriction checks, no layout change

`#[soa]` parses today — the grammar's `attr_name: IDENT`
(`frontend/grammar.lark:248-256`) is already generic, and it dies in
exactly one place: the `else` branch of
`AttributeProcessor.process_attributes` (`hir/attributes.py:250`).

- Add `SoaAttribute` beside `StorageAttribute` (`hir/attributes.py:73`)
  and a `_process_soa` branch to the dispatch at `hir/attributes.py:218`,
  guarding `context == 'static'` the way `_process_storage:355` does.
- Add `is_soa` and `is_synthetic` fields to `HIRStaticDecl`
  (`hir/nodes.py:222`).
- Validate: `[Struct; N]` shape, non-union element, element not in
  `_struct_type_ids`, not `#[hw]`, not `extern`.

`#[soa]` is otherwise inert; programs still compile AoS.

*Tests*: new `r65/tests/language/types/test_soa.py` — one accept case plus
one assertion per rejection.

### S2 — type-checker rejections for the non-funnel uses

Still AoS layout. This slice is what makes S3 safe.

`check_array_index` (`typeck/type_checker.py:1976`) is on the *good* path
too: `check_field_access` (`:2051`) type-checks its own base, so a blanket
rejection there kills the only supported form. Use the existing one-shot
flag idiom — `_clone_sugar_allowed` (`type_checker.py:1098, 2236`) is the
model — set around the base check in `check_field_access` and consulted in
`check_array_index`.

Rejections land in `check_array_index`, `pointer_validator.check_addressof`
(`:44`), the receiver `HIRAddressOf` synthesis
(`call_validator.py:358-375`), and `_classify_clone`
(`call_validator.py:214`).

*Tests*: extend `test_soa.py`.

### S3 — desugar, funnel rewrite, X key. **Atomic.**

**Synthesis runs as a third pass** over `hir_decls`, after the pass-2 loop
closes (`hir/builder.py:246`) — not inside `_build_static`. Children go
into `HIRProgram.declarations`; `.statics` is a read-only `@property`
(`hir/nodes.py:102`).

The reason is *not* declaration order. A static cannot forward-reference
its struct at all — pass 1 resolves the static's type eagerly, so
`static mut POOL: [Enemy; 4];` above `struct Enemy` fails with
"Undefined type: Enemy" with or without `#[soa]`. By the time
`_build_static` runs, the struct's `HIRStructDecl` therefore always
exists. The third pass earns its place for a different reason: it is the
only point where the whole program is in hand, which matters for the
checks that consult program-wide state, and it keeps layout work out of a
function whose job is building one declaration.

S1's restriction checks, by contrast, live in `_build_static`
(`_validate_soa_static`) — every one of them is answerable without the
element's field layout, so none touches `size_bytes`.

Declaring symbols late is safe: nothing iterates the symbol table
(`SymbolTable.declare` is a dict insert; every consumer uses `lookup`).

- **Allocator**: skip `is_soa` parents in `allocate_all`
  (`codegen/memory_alloc.py:496`), exactly as `is_extern` is skipped.
- **Lint**: three false positives fire otherwise, since
  `linter._walk_decl:138` visits every `HIRStaticDecl` —
  **L005 `dead_static_mut`** (children are declared but only the parent is
  ever read, so every field warns as dead),
  **`naming_convention`** (`^[A-Z][A-Z0-9_]*$` vs `ENEMIES__kind`), and
  **`zeropage_budget`** (parent and children both counted, so the budget
  double-counts). Skip `is_synthetic` decls in `_walk_decl`, and skip
  `is_soa` parents in `zeropage_budget._type_size:108`.
- **Stride override**: `ArrayTypeInfo.size_bytes` (`hir/types.py:58-60`) is
  `element_size * size`, which under-reports a padded plane (3·N, not
  4·N). Carry the stride on the synthesized decl as `soa_stride` and read
  it in exactly two places — the allocator's size computation
  (`memory_alloc.get_type_size:73`) and the funnel's scale. Do **not**
  change the child's declared element type: the load must stay a 3-byte
  far-pointer load at offset 0 within a 4-byte slot.
- **Funnel**: `emit_static_array_field_access` (`mir/builder.py:2248`) is
  verifiably the sole entry point for `arr[i].field` reads and writes —
  four callers (`expression.py:745, 776`; `assignment.py:330, 361`), and
  compound assignment reuses the same target node
  (`expression_builder.py:455`), so `+=` is covered for free. Substitute
  the per-field memloc for `base_memloc` and pass `field_size` to
  `compute_scaled_index`.
- **X reuse key** — a **correctness prerequisite, not an optimization**.
  With today's key `(base_key, id(index_sym))` (`mir/builder.py:2537`),
  `ENEMIES[i].hp` (u8, X = `i`) followed by `ENEMIES[i].x` (u16, needs
  `2i`) hits the cache and reads the wrong element. Fold the scale into
  position `[0]` as `('soa', parent_key, field_size)`; position `[1]` must
  stay the index symbol so `x_index_cache_invalidate_symbol` keeps working.

*Tests*: a padded-stride case — a `far *u8` field, asserting the plane is
`4*N` bytes and the scale is `ASL A / ASL A` — plus SoA mirrors of
`TestArrayOfStructs::test_array_of_structs_write_read`
(`r65/tests/e2e/test_structs_and_pointers.py:128`) and
`test_large_struct_variable_index` (`:160`, the 18-byte case, where the
`mul16` must vanish); new `r65/tests/compiler/mir/test_soa_x_reuse.py`
asserting one `Move(X, …)` for consecutive same-width fields and two for
mixed-width; lint regressions for all three rules above.

### S4 — nested chains

`peel_field_chain` hands the funnel only a *summed* offset, losing which
top-level field the chain started at. Recover it by locating the field
whose `[offset, offset + size)` half-open range contains the sum; the
residual is the offset within the per-field element. Sound because
`layout_fields` (`hir/unified_type_utils.py:120-129`) is strictly packed
with no zero-sized types — and it correctly lets a *union field inside*
the element alias, which is why the union restriction applies to the
element type only.

Both sides: `_lower_nested_field_access` (`expression.py:745`) and
`_lower_nested_field_assignment` (`assignment.py:330`).

*Tests*: extend `r65/tests/e2e/test_nested_field_access.py`.

### S5 — initializer transposition

Per top-level field, build an `HIRArrayLiteralExpr` of that field's
initializers and let `_emit_array_literal_init`
(`mir/lowerers/static_init.py:299`) run unchanged per child. The transpose
itself is small; the cost is validation, since
`struct_validator.check_struct_literal` runs *after* `_build_static` —
another reason synthesis lives in a late pass. Scalar top-level fields
only; reject nested aggregates.

*Tests*: extend `test_soa.py` and the S3 e2e file.

### S6 — port and measure

Rewrite `rotor_raiders/src/muni.r65:49-58` as one `#[soa] static mut
MUNI: [Muni; 16]` and diff the emitted assembly against the hand-written
parallel arrays. This is the only test that proves the feature is worth
having, and it is where mixed-width X thrashing would show up. If the
diff is not a wash or better, S3's single-slot cache needs widening before
the feature is worth recommending.

---

## References

- [Struct Array Indexing](struct-array-indexing.md) — the AoS stride strategy
- [Operators and Cost Model](operators.md) — why `*` is restricted
- [Pointers and Memory Model](pointers-memory.md) — storage classes
- [Unions](unions.md) — why a union element cannot be split
- [Traits](traits.md) — the TypeId byte at offset 0
