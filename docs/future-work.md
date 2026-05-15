# Future Work

Items deliberately scoped out of prior implementations. Each entry records *why* it was deferred so a future contributor can judge whether the constraints still apply.

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

## Conventions

When adding to this file:

- Keep entries scoped to *concrete* deferred items, not vague aspirations.
- For each, capture: what it does, why it was deferred, sketch of the approach if revisited.
- If an item lands, delete its entry (don't strike it through — git history preserves the past).
- Group related items under a heading describing the area of work.
