# Operator Overloading Design

## Overview

Operator overloading lets user-defined types (`struct`, `enum`) give meaning to
built-in operators. In R65 this is **desugaring**: the compiler rewrites an
operator expression on a user type into an ordinary method call, then lets the
existing static-dispatch call path do the rest. There is no new runtime
machinery — an overloaded operator compiles to the same `JSR`/`JSL` you would
write by hand.

This makes the multi-byte stdlib types ergonomic. Today you write:

```rust
I32__add_assign(&score, &bonus);   // or score.add(&bonus)
```

With overloading you write:

```rust
score += bonus;                    // identical generated code
```

**Scope of this design (Tiers A + B):** compound-assignment operators
(`+=`, `-=`, ...) and comparison operators (`==`, `<`, ...). Value-producing
binary operators (`let c = a + b`) are deliberately **out of scope** — see
[Why no `a + b`](#why-no-value-producing-a--b) and [future-work.md](future-work.md).

---

## Design Philosophy

R65 structs and enums are **pass-by-reference only** — they cannot be passed,
returned, or assigned by value (`typeck/type_utils.py: is_aggregate_type`). This
single constraint dictates the whole design:

1. **Overload the operators that mutate in place or return a primitive.**
   `a += b` mutates `a` through `*self` and returns nothing. `a < b` returns a
   `bool`. Neither needs to return an aggregate by value, so both fit the
   language as-is.

2. **Do not overload operators that must produce a fresh aggregate value.**
   `let c = a + b` would have to materialize a new `c` and copy into it — the one
   thing the by-reference model forbids. We leave it out rather than bolt on a
   destination-passing mechanism that fights the language.

3. **Newtypes sit outside all of this.** A
   [newtype](type-system.md#newtypes) is not an aggregate, so it never reaches
   the overload machinery: it inherits its payload's operators directly and
   re-wraps the result (`TileId + 1` is a `TileId`), by value, with no impl and
   no call. It also cannot implement these traits — they all take `*self`, and a
   newtype has no pointer self.

   The consequence worth noting is that a register-sized newtype *does* make
   value-producing operators expressible — `let c = a * b` returns two bytes in
   a register, which is exactly what Tier C was deferred for. Overloading them
   for newtypes (so `Q10`'s multiply can mean `(a*b)>>6` rather than the
   inherited integer one) is still future work; see
   [future-work.md](future-work.md).

4. **Zero cost.** Resolution is fully static (the operand types are known at
   compile time), so every overloaded operator lowers to a direct call. No jump
   tables, no `TypeId`, no layout change.

This is not arbitrary: the stdlib already converged on exactly these shapes.
`I32`/`U32`/`F32` arithmetic is mutate-in-place (`far fn add(far *self, other: far *I32)`
== `self += other`) and comparison is `far fn cmp(far *self, other) -> i8`. Tier A+B
overloading is just sugar over the API the stdlib already exposes.

---

## The Operator Traits

Overloading is **opt-in via a trait impl**, mirroring Rust's `core::ops`. The
trait names below are **known to the compiler** (lang items) — you do not declare
them, you only implement them.

### Tier A — Compound Assignment (in-place `*self`)

| Operator | Trait           | Method            | Restriction lifted vs. primitive |
|----------|-----------------|-------------------|----------------------------------|
| `+=`     | `AddAssign`     | `add_assign`      | —                                |
| `-=`     | `SubAssign`     | `sub_assign`      | —                                |
| `*=`     | `MulAssign`     | `mul_assign`      | no power-of-2 restriction         |
| `/=`     | `DivAssign`     | `div_assign`      | no power-of-2 restriction         |
| `%=`     | `RemAssign`     | `rem_assign`      | `%` allowed (calls a method)      |
| `&=`     | `BitAndAssign`  | `bitand_assign`   | —                                |
| `\|=`    | `BitOrAssign`   | `bitor_assign`    | —                                |
| `^=`     | `BitXorAssign`  | `bitxor_assign`   | —                                |
| `<<=`    | `ShlAssign`     | `shl_assign`      | shift amount need not be constant |
| `>>=`    | `ShrAssign`     | `shr_assign`      | shift amount need not be constant |

Every Tier A method takes `*self` plus exactly one right-hand operand and returns
nothing:

```rust
impl AddAssign for I32 {
    far fn add_assign(far *self, other: far *I32) {
        // self += other   (mutates the receiver)
    }
}
```

### Tier B — Comparison

| Operator(s)            | Trait        | Method | Signature                          |
|------------------------|--------------|--------|------------------------------------|
| `==`, `!=`             | `PartialEq`  | `eq`   | `eq(*self, rhs) -> bool`           |
| `<`, `<=`, `>`, `>=`   | `PartialOrd` | `cmp`  | `cmp(*self, rhs) -> i8`            |

`PartialOrd::cmp` is a three-way compare returning the **sign of `self - rhs`**:

| Return | Meaning      |
|--------|--------------|
| `< 0`  | `self < rhs` |
| `0`    | `self == rhs`|
| `> 0`  | `self > rhs` |

This matches the existing `I32::cmp`/`U32::cmp` convention exactly, so the stdlib
needs only a one-line `impl PartialOrd for I32 { ... }` that forwards to its
current `cmp`.

```rust
impl PartialEq for I32 {
    far fn eq(far *self, other: far *I32) -> bool { /* ... */ }
}
impl PartialOrd for I32 {
    far fn cmp(far *self, other: far *I32) -> i8 { /* ... */ }
}
```

---

## How It Works

### Compound assignment: `a += b`

`a` must be a mutable place (local `struct`/`enum`, static, field, or deref) whose
type implements the matching trait. The compiler takes the address of each
operand and emits a direct call:

```rust
score += bonus;
```

desugars to

```rust
I32__add_assign(&score, &bonus);   // JSL I32__add_assign
```

```asm
; load &score, &bonus per the method's calling convention, then:
JSL I32__add_assign        ; far method (self is far *self)
```

`a` and `b` are each evaluated **once**. (Note: for a struct LHS this is *not*
the primitive desugar `a = a + b` — structs can't be assigned by value, and that
desugar would also double-evaluate the place.)

### Equality: `a == b`, `a != b`

```rust
if a == b { ... }        // -> if I32__eq(&a, &b)        { ... }
if a != b { ... }        // -> if !I32__eq(&a, &b)       { ... }
```

`==` is the bare `eq` call; `!=` wraps it in a logical NOT. Result type is `bool`,
so this composes with `&&`, `||`, `if`, `while`, etc. like any other boolean.

### Ordering: `a < b`, `a <= b`, `a > b`, `a >= b`

All four lower to one `cmp` call followed by a primitive `i8` comparison against
zero:

```rust
a <  b     // -> I32__cmp(&a, &b) <  0
a <= b     // -> I32__cmp(&a, &b) <= 0
a >  b     // -> I32__cmp(&a, &b) >  0
a >= b     // -> I32__cmp(&a, &b) >= 0
```

The trailing `<0` / `<=0` / ... is an ordinary signed `i8` compare and reuses the
existing comparison codegen (`compare_select.py`) — no special handling.

---

## Operator Traits Are Compiler Lang Items

The operator traits differ from ordinary user traits in four deliberate ways.
This is what lets them sidestep R65's lack of generics, associated types, and a
`Self` type.

1. **No declaration.** You never write `trait AddAssign { ... }`. The names are
   reserved and recognized by the compiler. `impl AddAssign for I32 { ... }` is
   accepted even though no `trait AddAssign` exists in source.

2. **The rhs type comes from the impl, not a trait signature.** Ordinary traits
   require an *exact* signature match against the declaration. Operator traits
   have no declaration to match, so the impl is free to choose the right-hand
   operand type. `add_assign(far *self, other: far *I32)` and a hypothetical
   `add_assign(*self, value @ A: u16)` are both well-formed — the chosen rhs type
   becomes the type the operator expects on its right side.

3. **Static dispatch only — never `*dyn`.** Operator traits cannot be used as
   trait-object pointers. `*dyn AddAssign` is a compile error. Because they are
   never used dynamically, they are **never added to `_dyn_used_traits`**, so the
   `TypeId` byte is **not** inserted (`hir/builder.py` only assigns a TypeId for
   traits used with `*dyn`). An `I32` with operator impls stays exactly 4 bytes.

4. **One impl per operator per type.** With no generics there is one
   `impl AddAssign for I32`, and its mangled method `I32__add_assign` is unique.
   Heterogeneous right-hand types (e.g. `score += 1i16`) are *not* expressible
   through the operator; keep the existing named methods (`I32::add_i16`) for
   those and call them directly.

Method mangling is unchanged: `Type__method` (`I32__add_assign`, `I32__eq`,
`I32__cmp`). These names don't collide with the stdlib's existing `add`/`cmp`
methods.

---

## Resolution Algorithm

Operand types are only known after type checking, so resolution happens in
**typeck**, and the result is recorded on the node for MIR lowering to consume
(the same pattern as the existing `method_call_info`).

For a compound assignment `a OP= b`:

1. Type-check `a` and `b`.
2. If `typeof(a)` is a primitive → existing primitive path (`a = a OP b`),
   including the power-of-2 / constant-shift validation. **Done.**
3. If `typeof(a)` is a `struct`/`enum`:
   a. Look up the trait for `OP=` (e.g. `+=` → `AddAssign`).
   b. Find `impl AddAssign for <typeof(a)>`. If absent → error (E-OVL-001).
   c. Check `typeof(b)` matches the impl method's rhs parameter type
      (else E-OVL-002).
   d. Require `a` to be a mutable place (else the usual immutability error).
   e. Rewrite to `Type__add_assign(&a, &b)` and annotate the node.

For a comparison `a OP b` (handled in `check_binary_op`):

1. Type-check both operands.
2. If both primitive → existing comparison path. **Done.**
3. If either is a `struct`/`enum`:
   a. `==`/`!=` need `impl PartialEq`; `<`/`<=`/`>`/`>=` need `impl PartialOrd`
      (else E-OVL-001).
   b. Operand types must match the impl's `self`/rhs types (else E-OVL-002).
   c. Rewrite to the `eq`/`cmp` call form above; result type is `bool`.

---

## Semantics & Rules

- **Single evaluation.** Each operand expression is evaluated exactly once.
- **Mutability.** `a OP= b` requires `a` to be a writable place. Compound
  assignment to an immutable binding is the usual immutability error.
- **No const-evaluation.** Overloaded operators always emit a call; they are
  never folded at compile time (there are no compile-time aggregate values).
  Const-eval (`ast_const_eval.py`) is unchanged and still folds primitive
  operators.
- **Primitive operators are untouched.** Overloading only triggers when an
  operand is a user-defined type. `u8`/`u16`/`i8`/`i16`/`bool`/pointer operators
  behave exactly as before, including the `*`/`/` power-of-2 rule and
  constant-shift rule (`operator_validator.py`). That validator is **skipped**
  when an operand is an aggregate, so `score *= factor` on an `I32` is legal even
  though `factor` is not a power of two.
- **Far methods.** Operator methods may be `far fn` (as the stdlib's are). The
  desugared call follows normal near/far call rules; `&var` on a RAM/banked
  object yields the appropriate far pointer.
- **`!=` and ordering are derived.** Only `eq` and `cmp` are implemented; `!=`,
  `<=`, `>`, `>=` are synthesized from them. You cannot override `!=`
  independently of `==`.
- **Enums are eligible** (C-style, `u8`-sized) but rarely need overloading since
  they already compare as integers; structs are the primary use case.

---

## Worked Examples

### Multi-byte integers (`I32`)

```rust
// stdlib/I32.r65 — add these impls (bodies forward to existing methods):
impl AddAssign  for I32 { far fn add_assign(far *self, other: far *I32) { self.add(other); } }
impl SubAssign  for I32 { far fn sub_assign(far *self, other: far *I32) { self.sub(other); } }
impl MulAssign  for I32 { far fn mul_assign(far *self, other: far *I32) { self.mul(other); } }
impl PartialEq  for I32 { far fn eq(far *self, other: far *I32) -> bool { return self.cmp(other) == 0; } }
impl PartialOrd for I32 { far fn cmp(far *self, other: far *I32) -> i8  { /* existing cmp */ } }
```

```rust
// user code — before:
I32__add_assign(&score, &bonus);
if I32__cmp(&score, &target) >= 0 { win(); }

// after:
score += bonus;
if score >= target { win(); }
```

### A user struct

```rust
struct Vec2 { x: i16, y: i16 }

impl AddAssign for Vec2 {
    fn add_assign(*self, other: *Vec2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl PartialEq for Vec2 {
    fn eq(*self, other: *Vec2) -> bool {
        return self.x == other.x && self.y == other.y;
    }
}

fn step(pos: *Vec2, vel: *Vec2) {
    *pos += *vel;                 // Vec2__add_assign(pos, vel)
    if *pos == ORIGIN { reset(); }
}
```

---

## What's Not Included

### Why no value-producing `a + b`

`let c = a + b` (and unary `-a`) for aggregate types requires the compiler to
allocate a fresh aggregate temporary and copy a representation into it — exactly
the by-value behavior R65 forbids at the source level. Supporting it means
introducing destination-passing for expression temporaries (allocate `c`, thread
`&c` through the call as a hidden out-parameter). That is a real feature, not
sugar, and is deferred to Tier C.

The in-place form covers the common cases without it: `c = a; c += b;` expresses
"`c = a + b`" once a copy primitive exists, and accumulation loops
(`total += item`) never needed a temporary at all.

### Also excluded

- **Heterogeneous rhs** (`Add<Rhs>` in Rust) — no generics, so one rhs type per
  operator per type. Use named methods for other rhs types.
- **`&&`, `||`, `!`** — short-circuit logical operators are `bool`-only and not
  overloadable.
- **Index/deref operators** (`Index`, `Deref`) — not in scope.
- **Default method bodies** — consistent with the existing trait system.

---

## Implementation Plan

Localized to the frontend, HIR, and typeck. No new MIR nodes, no codegen
selectors, no changes to dispatch/ABI machinery — desugared calls ride the
existing static-dispatch `Call` path.

| # | Area | File | Change |
|---|------|------|--------|
| 1 | Lang items | new small registry (or constant in `hir/builder.py`) | Reserve operator trait names + `operator → (trait, method)` map. |
| 2 | Impl processing | `hir/builder.py` (~`_process_impl_decl`, near line 1784) | Accept operator-trait impls without a declared trait; **do not** add to `_dyn_used_traits` (no TypeId); record `(struct_name, operator) → mangled_method`, rhs type, return type. |
| 3 | Reject `*dyn` | `hir/builder.py` / typeck | Error if an operator trait is used as a trait-object pointer. |
| 4 | Compound assign | `hir/expression_builder.py:390` (`CompoundAssignment`) | Stop eagerly desugaring to `target = target op value`. Emit a dedicated node (e.g. `HIRCompoundAssign`) so typeck can choose primitive vs. overloaded lowering once types are known. |
| 5 | Resolve compound | `typeck/type_checker.py` | For aggregate LHS, run the [resolution algorithm](#resolution-algorithm); reuse the static method lookup in `typeck/call_validator.py: _try_method_call` (~180-316). |
| 6 | Resolve comparison | `typeck/type_checker.py: check_binary_op` (~1272) | For aggregate operands, resolve `PartialEq`/`PartialOrd` and rewrite to the `eq`/`cmp` call form; result type `bool`. |
| 7 | Skip validator | `typeck/operator_validator.py: validate_binary_op` | No power-of-2 / constant-shift check when an operand is an aggregate. |
| 8 | MIR lowering | `mir/lowerers/expression.py: lower_binary_op` | When the node carries an operator resolution, emit a `Call` (existing path in `mir/lowerers/call.py`) instead of a `BinaryOp`. |
| 9 | Tests | `tests/language/operators/`, `tests/language/impl/`, `tests/e2e/` | Unit tests for each operator's desugaring + resolution errors; e2e ROM test exercising `I32`/`Vec2` `+=` and comparisons (validate via Mesen-GDB). |

---

## Compiler Error Messages

```rust
score += bonus;
// E-OVL-001: type 'I32' does not implement '+=' (trait AddAssign)
// HELP: add `impl AddAssign for I32 { fn add_assign(*self, other: *I32) { ... } }`

let c = a + b;   // a, b: I32
// E-OVL-003: '+' is not overloadable for aggregate types
// HELP: use in-place `+=`; value-producing operators on structs are not supported

score += small;  // small: u16, but impl rhs is *I32
// E-OVL-002: '+=' for 'I32' expects right operand of type '*I32', found 'u16'
// HELP: call the method directly, e.g. score.add_i16(small)

let p: *dyn AddAssign = &score;
// E-OVL-004: operator trait 'AddAssign' cannot be used as a trait object (*dyn)
```

---

## Interactions

- **Traits / TypeId.** Operator impls never trigger TypeId insertion, so they
  don't change struct layout or interfere with `*dyn` dispatch on *other*
  (ordinary) traits the type implements.
- **`F32.add_to!` macro.** The existing three-operand `dest = a + b` macro stays
  the workaround for value-producing math until Tier C lands.
- **Multi-return.** Unaffected; comparison overloads return a single `bool`/`i8`.
- **Doc cross-refs.** [Operators](operators.md) (primitive semantics, precedence —
  unchanged), [Traits](traits.md) (impl mangling, static dispatch).

---

# Appendix: `Clone` — the aggregate-copy primitive

This is the keystone the [Tier C](#future-work-tier-c) gap depends on. R65
aggregates cannot be returned or assigned by value, so there is currently **no
way to copy a struct or array**. `Clone` adds one — destination-passing, the only
shape the by-reference model allows. Once you can copy *and* add-in-place,
value-producing `let c = a + b` becomes expressible as `c.clone_from(&a); c += b`,
and the compiler can later auto-desugar to exactly that.

## The constraint forces destination-passing

Rust's `Clone::clone(&self) -> Self` returns the new value by value — impossible
here (`typeck/type_checker.py: _check_no_aggregate_type` rejects aggregate
returns). So clone must write into a caller-supplied location. The machinery for
that already exists: the compiler destination-passes struct literals into a
let-binding's storage via a `BlockCopy` (MVN block move), and MVN works RAM→RAM.
A clone is essentially that copy with a RAM source.

## Design decisions

1. **Explicit `Clone` only.** Copies happen solely through explicit calls. Bare
   `=` / `let x = y` on an aggregate stays an error — copy cost remains visible,
   consistent with R65's "explicit control" philosophy.
2. **Auto-bitwise, overridable.** `impl Clone for T {}` with an **empty** body
   means the compiler generates the byte copy; a **non-empty** `clone_from` body
   overrides with custom logic. Bitwise is almost always correct (no heap, no
   `Drop`, no ownership), but the hook stays for the rare case.
3. **Two surfaces, structs + arrays.** In-place `dest.clone_from(&src)` and the
   sugar `let c = a.clone();`.

`Clone` is a **compiler lang-item trait** — same mechanism as the operator traits
above: known to the compiler, not declared in source, implemented via
`impl Clone for T`, resolved through existing **static** method dispatch. No
`TypeId`, no layout change, `*dyn Clone` forbidden.

```rust
impl Clone for Player {}                      // empty = auto bitwise copy
impl Clone for Enemy  {                        // custom override
    fn clone_from(*self, src: *Enemy) { /* ... */ }
}

let c = a.clone();                             // sugar: compiler routes &c as dest
dst.clone_from(&src);                          // in-place primitive
let arr2 = data.clone();                       // arrays: built-in, no impl needed
```

## Semantics & rules

- **`.clone()` is legal only as a direct initializer** of a `let` or an assignment
  to an aggregate place — the one position where the compiler has a destination.
  Anywhere else (call arg, return, sub-expression) it is an error.
- **`clone_from(*self, src: *T)`** mutates `*self` in place; `self` must be a
  writable place. Its arguments are pointers, so the by-value aggregate gate never
  fires.
- **Arrays are a built-in intrinsic.** You cannot write `impl Clone for [u8; N]`
  (impl targets must be structs), so array clone is always available, no impl
  needed; `count = N * sizeof(element)`. `[Player; N]` is fine — structs are POD.
- **No layout impact.** Operator/Clone lang-item impls never enter the `*dyn`
  set, so no `TypeId` byte is inserted; a cloned struct stays its raw size.

## Lowering

| Case | Lowering |
|------|----------|
| auto struct / array, **both ends static addresses** | new `AggregateCopy` MIR node → unrolled `LDA/STA` (small) or **MVN** (large) |
| auto, **either end a runtime/far pointer** | `Call memcpy(dest, src, #count)` — `count` is compile-time known |
| manual `impl` body | ordinary static `Call Type__clone_from(&dest, &src)` |
| `let c = a.clone()` | reuse the aggregate let-binding allocation (decompose/promote), swap the init source from ROM literal to the source place; decomposed small structs copy field-by-field (vreg→vreg `Move`) |

**The MVN bank problem.** MVN's bank bytes are assemble-time immediates, but
`clone_from` may receive runtime/far pointers. Resolution: decide static-vs-runtime
**in MIR** — emit `AggregateCopy` (→ MVN/unrolled) only when both ends are static;
otherwise emit a `Call memcpy(#count)` (proven, handles arbitrary banks). An inline
`[dp],Y` loop is a later optimization.

## Implementation plan

Phase 0 (lang-item plumbing) is **shared with the operator traits above** — build
it once.

| # | Area | File | Change |
|---|------|------|--------|
| 0 | Lang items *(shared)* | `hir/lang_items.py` *(new)* | `LANG_ITEM_TRAITS` (Clone + operator traits), `CLONE_METHOD = "clone_from"`. |
| 0 | Impl acceptance *(shared)* | `hir/builder.py: _declare_impl` (~:1569) | Branch lang-item traits to `_declare_lang_item_impl`: accept without a source `trait`, record `_clone_impls[T] = {auto, mangled}`, register `T__clone_from` only for non-empty bodies, **skip** `_dyn_used_traits`/`TypeId`. |
| 0 | Reject `*dyn` *(shared)* | `hir/builder.py: _collect_dyn_traits` (~:268) | Error on `*dyn <lang-item>`. |
| 1 | Parse | `hir/expression_builder.py` (~:162-230) | Recognize `.clone()`/`.clone_from(x)` → `HIRMethodCall`. |
| 2 | Resolve | `typeck/call_validator.py: check_method_call` (~:581) | Classify intrinsic (array / auto struct) vs call (manual); reuse `_try_method_call` (~:180). |
| 2 | Initializer rule | `typeck/type_checker.py` | Allow aggregate `clone()` only in `check_let_statement` (~:915) and the assignment gate (~:1799); else E-CLONE-002. |
| 3 | Copy node | `mir/nodes.py` (~:289) | New `AggregateCopy` (memory/pointer src+dest, `count`). |
| 3 | Lowering | `mir/builder.py` | Clone-init routing at the aggregate let branch (~:686) and `_emit_aggregate_init` (~:821); `_emit_clone_decomposed`; `memcpy` for runtime pointers. |
| 4 | Codegen | `codegen/instruction_select.py` | `select_aggregate_copy` (template: `select_block_copy` ~:2013). |

**Reuse:** `BlockCopy`/`select_block_copy` (MVN template),
`_decompose_struct_local`/`_promote_aggregate_local`/`_emit_aggregate_init`
(storage + init), `_try_method_call` (manual dispatch), `lower_addressof`,
stdlib `memcpy` (`stdlib/string.r65:31`), `_emit_pointer_mem_copy`
(`codegen/base_selector.py:189`).

## Error messages

```rust
let c = a.clone();   // a: Player, no impl Clone for Player
// E-CLONE-001: type 'Player' does not implement Clone
// HELP: add `impl Clone for Player {}` (empty body = bitwise copy)

f(a.clone());        // or: return a.clone();
// E-CLONE-002: `.clone()` is only allowed as the direct initializer of a let or
//              assignment to an aggregate place

let p: *dyn Clone = &a;
// E-CLONE-004: 'Clone' cannot be used as a trait object (*dyn)

src.clone_from(&a);  // src is immutable
// E-CLONE-005: cannot clone into immutable place 'src'
```

## Verification

- **Unit** (`tests/compiler/`, `tests/language/impl/`, via `compile_string`):
  `impl Clone for T {}` accepted without a `trait Clone`; auto impl emits
  `AggregateCopy` and **no** `T__clone_from` symbol; manual impl emits a `Call`;
  array clone needs no impl; struct-with-`impl Clone` stays its raw size (no
  TypeId); error cases E-CLONE-001/002/004/005; regression — bare `c = a` on an
  aggregate still errors.
- **E2E** (`tests/e2e/test_clone.py`, model on `test_local_aggregates.py`, verify
  with Mesen-GDB `ExpectedState(memory=...)`): struct auto-clone is independent of
  its source; array clone copies all bytes; `clone_from` into a `#[ram]` static;
  a manual override (e.g. zeroing a field) proves the call path ran; a far/runtime
  pointer `clone_from` exercises the `memcpy` tier.

---

# Implementation Status

**Implemented** (lang-item foundation + Clone + operator Tiers A/B):

- **Lang-item foundation** — `impl <LangItem> for T` is accepted with no source
  `trait` declaration, registers via the existing static-dispatch path, injects no
  `TypeId` (layout unchanged), and rejects `*dyn`. Shared by Clone and the
  operator traits. (`hir/lang_items.py`, `hir/builder.py`)
- **Clone** — auto bitwise copy from an empty `impl Clone for T {}`, overridable
  with a `clone_from` body; `dst.clone_from(&src)` and the `let c = a.clone()`
  sugar; arrays via a built-in (no impl). Intrinsic copies lower to an
  `AggregateCopy` MIR node → `MVN` (DBR preserved with PHB/PLB).
- **Tier A** — all ten compound-assignment operators desugar to in-place
  `*_assign` methods; the power-of-2 / constant-shift restrictions are lifted for
  aggregate operands; primitive compound-assign is unchanged.
- **Tier B** — `==`/`!=` → `PartialEq::eq` (bool); `<`/`<=`/`>`/`>=` →
  `PartialOrd::cmp` (i8, signed compare vs 0).

Tests: `tests/language/impl/test_lang_item_traits.py`,
`tests/language/impl/test_clone.py`,
`tests/language/operators/test_operator_overload.py`,
`tests/e2e/test_clone.py`, `tests/e2e/test_operator_overload.py`.

**Deferred / known limitations:**

- Clone through a runtime pointer (`p.clone_from(q)` where `p`/`q` are `*T`)
  raises a clear error — place/static operands only for now (the planned
  `memcpy`/`[dp],Y` tier is a follow-up).
- `let c = a.clone()` for a type with a **manual** `clone_from` body is not yet
  supported (call `c.clone_from(&a)` explicitly); auto/array sugar works.
- Value-producing operators (`let c = a + b`, unary `-a`) — Tier C, out of scope.
  Newtypes get these for free from their payload, but cannot *override* them
  (a newtype has no `*self`); see [future-work.md](future-work.md).
- One operator impl per type (no heterogeneous rhs) — by design (no generics).

---

**STATUS**: Implemented — Tiers A + B and the `Clone` primitive (see Implementation Status); Tier C (value-producing ops) deferred
**Scope**: Operator overloading Tiers A (compound assignment) + B (comparison); `Clone` aggregate-copy primitive
**Last Updated**: 2026-06-09
