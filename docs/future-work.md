# Future Work

Items deliberately scoped out of prior implementations. Each entry records *why* it was deferred so a future contributor can judge whether the constraints still apply.

---

## Conventions

When adding to this file:

- Keep entries scoped to *concrete* deferred items, not vague aspirations.
- For each, capture: what it does, why it was deferred, sketch of the approach if revisited.
- If an item lands, delete its entry (don't strike it through — git history preserves the past).
- Group related items under a heading describing the area of work.

---

## Trait Dispatch Chain Coalescing — Further Extensions

The chain-coalescing pass landed in stages: v1 (far-self DBR bracket elision, straight-line paths), v1.5 (cross-method on same trait), v1.6 (cross-trait via cast-transparency), v2 (CFG diamonds, near-self chaining, Y-reload elision).

The pass lives in `r65/compiler/analysis/far_ptr_strategy.py` (entry: `analyze_trait_dispatch_chains`). Codegen consults `TraitDispatch.self_chain_role` and `TraitDispatch.self_y_preloaded` in `r65/compiler/codegen/call_select.py::emit_trait_dispatch`. Soundness predicates are `_function_is_dbr_independent` and `_function_preserves_y`, both memoized on `MIRFunction` and recursive via the trait-resolved call graph.

### Loop-aware chaining (back-edges)

Today the CFG walker rejects any back-edge — a chain cannot extend across or around a loop. Loops break the analysis because back-edges feed mutated state (Y, DBR, possibly the self vreg itself) into the chain head from a non-linear predecessor.

**Why deferred**: requires fixpoint analysis over the loop body to prove all loop-invariant chain preconditions hold across every iteration. The straight-line + diamond cases cover the common patterns; loops are rare in trait-dispatch hot paths.

**Approach if revisited**: SCC analysis to identify loop bodies, then a separate predicate `_loop_body_preserves_chain_state(body, root_vreg)` that walks every block in the SCC and checks all the existing instruction-level disqualifiers. Special care for loop-carried Y/DBR mutations.

### Multi-arm switch joins (3+ way branches)

The diamond bridger in `_try_bridge_diamond` only accepts CFG patterns with exactly two successor arms rejoining at a common block. N-way branches (jump tables, switch-style dispatch with three or more arms) are rejected.

**Why deferred**: the predicate generalizes naturally — every arm must be DBR/Y-independent and rejoin at a common block — but the test surface grows quickly and we have no real-world workload that exercises it.

**Approach if revisited**: replace the strict-2-arm check with a loop over all successors; require all arms have exactly one successor pointing at the same rejoin block, no nested branching, no back-edges, and pass the per-instr predicate. Add tests for 3-way, 4-way, and one-bad-arm cases.

### Nested diamonds

Each arm in a diamond bridger is required to be a single straight-line block. An arm that itself contains an inner if/else (i.e. a nested diamond) is rejected because the arm has more than one successor.

**Why deferred**: composing the diamond bridger recursively is mostly a code-organization problem. The motivating workloads don't have deeply nested control flow on hot dispatch paths.

**Approach if revisited**: extract the "is this region single-entry, single-exit, and chain-preserving?" check into a helper that recurses on inner diamonds. The outer `_try_bridge_diamond` then uses this helper instead of the current single-block-arm check.

### Post-allocation Y-preservation analysis

`_function_preserves_y` runs *before* register allocation, so it relies on `register_hint='Y'`, `self_y_vreg`, and explicit HW Y operands to detect Y writes. A vreg without `register_hint='Y'` that the allocator happens to put in Y would not be flagged — meaning the predicate could mark a function Y-preserving when the allocator's eventual placement makes that false.

In practice this is rare (`register_hint='Y'` is the only mechanism the loop register promoter uses to request Y, and trait method impls pre-allocate `self_y_vreg` to HW Y), but the gap is real.

**Why deferred**: tighter analysis would require running the chain pass *after* register allocation, which is structurally inverted from how it's wired today.

**Approach if revisited**: either (a) run a second chain-coalescing pass post-alloc that revokes optimistic Y-preserved decisions when the allocator made them unsafe, or (b) move the entire chain pass after allocation. (a) is less invasive; (b) is cleaner but a larger refactor.

### Tail-call conversion of trait dispatches

A trait dispatch in tail position (`return obj.method()`) could be converted to a tail call (`JMP` instead of `JSR/JSL` + `RTS/RTL`), saving 6+ cycles.

**Why deferred**: orthogonal to chain coalescing — a separate optimization pass with its own soundness rules (return-value forwarding, stack frame teardown timing). Worth doing but not on the chain pass's path.

**Approach if revisited**: detect tail-position TraitDispatch in MIR (last non-Return instruction in a block whose only successor is the function exit), generate a tail-dispatch wrapper variant that issues `JMP` to the impl instead of returning through the dispatch wrapper.

### TypeId-based devirtualization

When a TraitDispatch's self_ptr is preceded by a `type_id() == X::TYPE_ID` comparison that gates the dispatch, the call could be lowered to a direct `JSR <X::method>` instead of going through the jump table — eliminating the dispatch wrapper overhead entirely.

**Why deferred**: requires either (a) a control-flow-sensitive analysis tracking what TypeIds reach each dispatch point, or (b) explicit programmer intent (e.g., a `match obj.type_id()` construct that the front-end lowers to direct calls). Both are significant.

**Approach if revisited**: option (b) is probably the right path for R65's simplicity-first philosophy — give programmers a direct way to express "I've checked the type, dispatch directly" and lower it accordingly.

---

## Indirect Call Lowering — Further Extensions

The `JML [d]` fast path for far indirect calls landed in
`r65/compiler/codegen/call_select.py::_emit_dp_indirect_far_call`. It
fires only for SCRATCH-resident far function pointers today. Several
adjacent cases remain on the trampoline.

### STACK-resident far fn pointer under D=S

The brief that introduced the fast path described a STACK + D=S variant
where the param's stack offset doubles as a DP offset (because D=S).
That branch is currently deferred.

**Why deferred**: the call sequence in
`call_select::_emit_call_instruction` emits `PLD` *before* every call
in a D=S function (via `_emit_d_restore_before_call`) so the callee
sees a sane D. By the time `_emit_indirect_call_trampoline` runs, D
no longer equals S, and a stack-offset `JML [d]` would read from the
wrong address. Re-establishing D=S immediately before the JML is
possible but trades some of the savings, and worse, forces the callee
to receive D=S — which is incorrect for callees that perform DP-
relative loads.

**Approach if revisited**: thread a flag through
`_emit_call_instruction` indicating "this call will be lowered via the
JML [d] fast path, skip the PLD". Then re-establish caller's D
immediately after the call returns at `ret_label:`. The callee still
sees D=S; this is acceptable only if we constrain the fast path to
target functions known to either save D (`PHD`) or not touch DP. That
constraint requires call-graph analysis at callsites with vreg-typed
function pointers — non-trivial. An alternative: spill the stack-
resident pointer to an ad-hoc scratch slot, then JML [scratch]. Costs
~10 extra cycles vs the brief's ideal but is unconditionally safe and
still beats the trampoline.

When implemented, also resurrect the cost model in
`analysis/far_ptr_strategy._decide_fn_ptr_only_strategy` (Case II,
fn-pointer-only functions) — today that function always returns None
because no benefit accrues without the STACK fast path.

### Near indirect calls

The `JML [d]` opcode is 24-bit. There is no analogous `JSR (d)` / `JMP
(d)` form that synthesizes a 16-bit indirect *call* — only `JMP (addr)`
which is a jump, not a subroutine call. Near indirect calls keep the
trampoline.

**Why deferred**: orthogonal — would require either a different
synthesis (e.g. PEA <ret-1> ; JMP (d)`, but the relative cycle count
needs measurement) or accepting that near indirect through a fn
pointer is just rare and not worth optimizing.

### JML [abs] for absolute-resident pointers

Function pointers stored in static memory (e.g. `#[ram] static mut H:
far fn();`) are not DP-addressable. The 65816 has a `JMP [addr]` form
(opcode 0xDC, the same byte used for `JML [d]` — distinguished only
by addressing-mode width), but R65 currently only emits the DP form.

**Why deferred**: the SCRATCH path covers the common case where the
caller has loaded the pointer into a scratch slot for performance. A
direct memory-resident pointer is rarer and the SBC trampoline cost
isn't dominant in those callsites.

**Approach if revisited**: extend `_dp_offset_for_indirect_call` (or
add a sibling helper) to recognize `MEMORY` locations whose
`memory_addr` resolves to a label/absolute address. Emit
`JMP [<absolute_addr>]`. The emitter already special-cases
`Opcode.JMP_INDIRECT_LONG` to render as `JML [...]`; the same opcode
in `INDIRECT_LONG` mode (vs `DP_INDIRECT_LONG`) would handle this. The
opcode byte is the same; only the operand width differs.

### Tail-call conversion for indirect calls

A far indirect call in tail position could lower to plain `JML [d]`
(no PHK/PEA, no synthetic return) saving 8 more cycles.

**Why deferred**: orthogonal to the fast path — needs tail-call
analysis machinery shared with direct calls. See the trait-dispatch
tail-call entry above for the structural similarity.

---

## Function Inlining — Far Functions with Far-Pointer Stack Params

The inliner's `_far_body_is_bank_safe` (`r65/compiler/optimize/inline.py`)
admits far callees whose bodies produce identical bytes in any
caller's bank — far indirects, zero-page, WRAM long-addressing. It
rejects far callees with `has_far_ptr_stack_params=True`. Concretely
this blocks `put_str` / `put_num` / similar utilities in
classickong.r65, even at -O2, even though their bodies are otherwise
ideal inline candidates (small, hot, called 20–30× each).

The dependency the rejection guards against is real. A far function
with a far-pointer stack parameter relies on its prologue to set DBR
to the parameter's bank — either via `PHB / LDA bank,S / PHA / PLB`
(SET_DBR strategy) or via `PHD / TSC / TCD` (D=S strategy). The body
then uses `(d,S),Y` or `[dp],Y` indirect derefs whose effective bank
is determined by that prologue setup. Inlining splices the body in
but drops the prologue; the caller's DBR is whatever it happens to
be, so the indirect dereferences read the wrong bank.

**Why deferred**: the fix requires emitting equivalent MIR-level DBR
(and possibly DP) management at the inline boundary — Push DBR /
load bank byte / Pull DBR around the inlined body, plus careful
coordination with `analyze_far_ptr_strategy` which currently picks
per-function strategies and would need a per-call-site mode. The
bracket needs to nest correctly with the caller's own DBR state, the
caller's `far_ptr_strategy`, and any other inlined far callees in
the same caller.

**Approach if revisited**: extend `_inline_call` to detect far
callees with far-pointer stack params and emit a MIR prologue/
epilogue bracket around the cloned body. For SET_DBR strategy:
`Push HW_DBR` / load bank byte from `arg.value` / `Move HW_DBR <-
bank_byte` at entry, `Pull HW_DBR` at every exit. For D=S strategy:
`Push HW_D` / `Move HW_D <- HW_S` at entry, `Pull HW_D` at exit. The
choice of strategy should match the caller's `far_ptr_strategy` to
avoid clashing prologues; if the caller doesn't yet have one, the
analysis would need to run again post-inline. Tests should cover:
(a) inlining a far-ptr-param callee into a SET_DBR caller, (b) into
a D=S caller, (c) into a caller with no strategy yet, and (d)
nested inlining where the inlined body itself contains another
far-ptr-param call.

The eligibility predicate would change from `if
func.has_far_ptr_stack_params: return False` to a soundness check
that the body uses only the indirect addressing modes the chosen
bracket supports.

---

## Match LookupTable — 3-byte (far pointer) result support

`_analyze_for_lut` in `r65/compiler/mir/lowerers/match.py` accepts a
match expression as a LookupTable candidate only when the result
type is u8 (1 byte) or u16 (2 bytes); any other size returns `None`
and falls back to JumpTable. Likewise `select_lookup_table` in
`r65/compiler/codegen/control_flow_select.py` only emits `.DB` /
`.DW` table entries — there is no 3-byte path.

This means a match like
```rust
let frame: far *u16 = match anim_idx {
    KONG_ANIM_FACE1 => &kong_face1 as far *u16,
    ...
};
```
where every arm body is a compile-time constant far-pointer cannot
use the inline ROM table; it always lowers to the heavier
JumpTable, which dispatches into per-arm code blocks that each
materialise the same constant pointer into stack-resident bytes.

**Why deferred**: existing `select_lookup_table` builds a single
indexed `LDA table,X` (m8 for u8 or m16 for u16). 3-byte entries
require either:
- three indexed reads at `table_low,X` / `table_mid,X` /
  `table_high,X` (three separate `.DB` tables), or
- one m16 read for the low 16 bits plus a separate `.DB` table for
  the bank byte.
Plus a destination wider than A: the result has to land in a 3-byte
stack slot (or scratch DP) rather than the accumulator, and the
post-load `STA dest` and `BRA merge` shape changes. Also the
`base_value` adjustment, the default path's immediate load, and the
mode bookkeeping all need to be revisited for the 24-bit case.

**Approach if revisited**: extend `_analyze_for_lut` to accept
`result_size == 3` when the result type is a pointer (`*T` or
`far *T`) and every arm body lowers to a constant address via
`try_eval_const_addr` (a new helper alongside `try_eval_const_int`
that recognises `&label as far *T`, `&label[const] as far *T`, and
const-cast variants). In `LookupTable.values` store tuples
`(low, high, bank)`. In `select_lookup_table` emit two indexed
loads + a stack store sequence, mirroring the per-byte spill
pattern used elsewhere for far pointers; the merge BRA stays.

When this lands the kong-frame dispatch in `classickong.r65`'s
`game_show_kong` shrinks from a ~21-arm JumpTable + 21 inline
constant-pointer blocks (~600 bytes) to one indexed LUT + a single
3-byte store (~150 bytes), at the cost of one extra indexed read
per dispatch.

---

## SoA Struct Arrays — `#[soa]` Attribute

Lay a `static [T; N]` out as parallel per-field arrays so `arr[i].field`
desugars to `arr_field[i]` — no per-access software `mul8` for
non-power-of-2 struct sizes, while keeping struct syntax and a single
`N` at the source level. Two intentional restrictions on `#[soa]`
element types: **no `impl Trait`** (no contiguous `self` to point at)
and **no `&arr[i]`** (the element has no single base address).
Callers pass the index `i` instead of a reference.

---

## Far Pointer Return Values (`-> far *T`)

A function returning a `far *T` is currently a type error (`type_checker.py::_check_return_fits_a_register`).

The return ABI is register-count based — each returned value rides back in A, then B or X, then Y — so a value has at most 2 bytes to travel in. A far pointer is 3. Before the check landed, the callee materialised the pointer in its stack frame, kept only the low byte in A, and deallocated the frame; the caller read the remaining two bytes out of dead stack. That worked or not depending on whether an unrelated later call had overwritten the region, which is exactly how it presented during the DERELICT text-adventure evaluation (see `docs/r65-evaluation.md`).

**Why deferred**: a correct fix is an ABI change, not a codegen patch. It needs a defined register triple (e.g. A = low 16 bits, X = bank), caller-side reassembly, and interaction with `#[preserves(...)]`, frame teardown ordering, and the existing multi-return register assignment (`_get_return_register_order`). The hard error is sound in the meantime, and the workaround — index the table at the use site, or write through an output parameter — costs nothing at runtime.

**Approach if revisited**: extend `_get_return_register_count` to return a byte budget rather than a register count, define the triple in `abi_model.py`, and emit reassembly in `_emit_return_value_collection`. Tests should cover the near/far callee cross product and a returned pointer that is used after a subsequent call — the shape that exposed the original bug.

---

## Stdlib Macro Coverage

`write_color!`, `enable_nmi!`, and `enable_autojoy!` in `stdlib/sneslib.r65` had never compiled — a missing semicolon and two `u8`-vs-`Interrupt` type errors respectively. Macro bodies are only parsed when expanded, so a macro nothing calls is never validated.

**Why deferred**: fixed in place, but the class of problem remains — any unused stdlib macro is unverified.

**Approach if revisited**: a test source that expands every stdlib macro once, compiled as part of the suite. Cheap to write, and it turns "unused" into "at least syntactically live". Related: `enable_nmi!`/`disable_nmi!`/`enable_autojoy!` read-modify-write `NMITIMEN` ($4200), which is write-only — they fold open bus into the value and need a shadow variable to be correct.

---

## Newtypes

Deferred work on `struct TileId(u8);` — see [type-system.md](type-system.md#newtypes).

### Operator Overloading (Tier C)

A newtype inherits its payload's operators (`TileId + 1` is a `TileId`) but cannot *override* them. `struct Q10(i16)` therefore gets integer multiply, when its multiply is semantically `(a * b) >> 6` — and `*` on a primitive is restricted to power-of-2 constants, so the correct operation cannot even be written as an operator today.

**Why deferred**: the existing operator traits all take `*self` (`docs/operator-overloading.md`), which a newtype does not have. Supporting them means a by-value variant of the trait shapes — `fn mul(self, rhs: Q10) -> Q10` — plus a resolution rule where a user `impl` wins over the inherited primitive operator, and lifting the power-of-2 `*`/`/` restriction for the overloaded type.

Worth noting that the *hard* part of Tier C is already gone. Value-producing operators were deferred because `let c = a + b` had to materialise an aggregate and copy into it, which needed destination-passing. A register-sized newtype returns in a register, so nothing needs materialising — what remains is the trait plumbing, not an ABI change.

**Approach if revisited**: add by-value operator lang items alongside the `*_assign` set in `hir/lang_items.py`; resolve them in `type_checker.py::_newtype_binop_result` *before* falling through to the inherited primitive path; reuse the existing static-dispatch call lowering, which already handles a register-bound `self` (`hir/builder.py::_impl_self_is_by_value`). Tests should cover a `Q10` multiply against a hand-written `q10_mul` for cycle parity, and confirm the inherited operator still applies to a newtype with no impl.

### Far-Pointer Payloads

`struct Handle(far *Sprite);` is rejected — the payload is 3 bytes and a newtype must fit in a register.

**Why deferred**: the same return-register budget as *Far Pointer Return Values* above. A newtype is only worth having if it is free, and a 3-byte payload is not.

**Approach if revisited**: lands with the far-pointer return ABI. Relax the size check in `hir/builder.py::_validate_newtype_inner` once a 3-byte value has a defined register triple.

### Register Bindings for Bare Enums

`_validate_register_binding_type` (`hir/builder.py`) maps a near pointer to `u16` so it can bind `A`/`X`/`Y`, but rejects a bare enum with "must have a primitive type". Newtypes over enums are allowed through as `u8` because a newtype is a value type by design; a plain `enum Direction` still cannot bind a register, which is why every test passes enums as `dir @ A: u8` and casts at the call site.

**Why deferred**: relaxing it for bare enums is a language-wide change to the register-binding rules, not a newtype one, and nothing forced the question.

**Approach if revisited**: extend the allowed-type table to accept `EnumTypeInfo` as `u8` unconditionally, and drop the newtype special case next to it.

### Trait Implementations

A newtype **may** implement a trait, for static dispatch only. Forming a `*dyn`
over one is rejected at the cast: dynamic dispatch reads a TypeId byte at offset
0 of the pointee, and a newtype is all payload.

The earlier blanket rejection cited that same byte, but it is only injected for
traits actually used with `*dyn` — a statically dispatched trait does not change
layout, so the objection never applied there. In practice the two shapes rarely
meet: a trait whose methods take `self` by value can only be implemented by a
newtype, and one taking `*self` only by a struct.

`impl Clone` stays rejected — redundant rather than impossible, since a newtype
copies with a plain assignment.

### Deliberate Restrictions

Two small deliberate restrictions, each easy to lift but each needing a decision first:

- **Nested newtypes** (`struct Celsius(Temp);`) are rejected. `size_bytes` already recurses, so the mechanics are free — what needs deciding is whether the transparent-in rule reaches through the nesting (does `Celsius` accept a bare `i16`, or only a `Temp`?).
- **`.0` is read-only.** `V.0 = 5` would be a natural spelling, but struct field writes deliberately do not require `mut`, so allowing it would let `let t: TileId = ...; t.0 = 9;` mutate an immutable binding. Revisit only with a story for that.

### `#[cfg(...)]` on impl blocks

Individual methods honour `#[cfg(...)]`, but an `impl` block as a whole cannot be gated — `impl_decl` takes no `attribute*` in the grammar, so a block whose every method is hardware-dependent needs the attribute repeated on each one.

**Why deferred**: per-method filtering covers the cases that came up (`Q10::lerp`, `q10_mul`), so what is left is ergonomics rather than capability.

**Approach if revisited**: allow `attribute*` on `impl_decl` and fold the block's cfg into each method's, so an outer disable wins over an inner enable.
