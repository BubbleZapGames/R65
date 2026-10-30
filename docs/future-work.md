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

The pass lives in `r65/compiler/analysis/trait_chain_coalesce.py` (entry: `analyze_trait_dispatch_chains`). Codegen consults `TraitDispatch.self_chain_role` for the DBR bracket and decides each Y reload from the code it emitted (`call_select._self_still_in_y`). The DBR soundness predicate is `_function_is_dbr_independent`, memoized on `MIRFunction` and recursive via the trait-resolved call graph.

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

A far indirect call lowers to `PHK / PEA ret-1 / JML [abs]`
(`r65/compiler/codegen/call_select.py::_emit_dp_indirect_far_call`) when
the pointer is in a scratch slot or in a low-RAM or direct-page static.
`JML [abs]` reads its target from bank 0 at an absolute address, never
D-relative. Other cases remain on the trampoline.

### Stack-resident and `#[ram]` far fn pointers

A pointer in a stack slot (a far fn parameter with no scratch to promote
it into) has no absolute address, and setting D=S does not help: `JML
[abs]` ignores D. A `#[ram]` static sits in bank $7E, which `JML [abs]`
cannot reach.

**Approach if revisited**: copy the pointer into a free scratch slot, then
`JML [scratch]`. Costs ~10 cycles over the in-place form and still beats
the trampoline by ~50, but needs a scratch slot free at the call.

### Near indirect calls

`JML [abs]` jumps through a 24-bit pointer. A 16-bit pointer has only
`JMP (addr)`, but `PEA ret-1 / JMP (addr)` would make the call, since
the callee's RTS pops the pushed return address. Near indirect calls
keep the trampoline.

**Why deferred**: orthogonal — would require either a different
synthesis (`PEA ret-1 / JMP (addr)`, but the relative cycle count
needs measurement) or accepting that near indirect through a fn
pointer is just rare and not worth optimizing.

### Tail-call conversion for indirect calls

A far indirect call in tail position could lower to plain `JML [abs]`
(no PHK/PEA, no synthetic return) saving 8 more cycles.

**Why deferred**: orthogonal to the fast path — needs tail-call
analysis machinery shared with direct calls. See the trait-dispatch
tail-call entry above for the structural similarity.

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

The motivating case, a ~21-arm far-pointer match in classickong's
`game_show_kong`, has since been rewritten to index a `kong_frames`
ROM table, which is the workaround today: put the pointers in a
`static` array and index it.

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

**Approach if revisited**: a test source that expands every stdlib macro once, compiled as part of the suite. Cheap to write, and it turns "unused" into "at least syntactically live".

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

### Deliberate Restrictions

Two small deliberate restrictions, each easy to lift but each needing a decision first:

- **Nested newtypes** (`struct Celsius(Temp);`) are rejected. `size_bytes` already recurses, so the mechanics are free — what needs deciding is whether the transparent-in rule reaches through the nesting (does `Celsius` accept a bare `i16`, or only a `Temp`?).
- **`.0` is read-only.** `V.0 = 5` would be a natural spelling, but struct field writes deliberately do not require `mut`, so allowing it would let `let t: TileId = ...; t.0 = 9;` mutate an immutable binding. Revisit only with a story for that.

### `#[cfg(...)]` on impl blocks

Individual methods honour `#[cfg(...)]`, but an `impl` block as a whole cannot be gated — `impl_decl` takes no `attribute*` in the grammar, so a block whose every method is hardware-dependent needs the attribute repeated on each one.

**Why deferred**: per-method filtering covers the cases that came up (`Q10::lerp`, `q10_mul`), so what is left is ergonomics rather than capability.

**Approach if revisited**: allow `attribute*` on `impl_decl` and fold the block's cfg into each method's, so an outer disable wins over an inner enable.

---

## Clone

Deferred parts of `impl Clone` — see [operator-overloading.md](operator-overloading.md).

### Clone through a runtime pointer

`p.clone_from(...)` or a clone whose receiver is a pointer is rejected (`typeck/call_validator.py`, "clone through a runtime pointer is not yet supported"). The intrinsic copy needs a static address at both ends: `mir/builder.py::_clone_place_memloc` resolves each operand to a `MemoryLocation`, and returns `None` for anything else.

**Why deferred**: the place-based copy covers named variables and statics, which is where clones came up. A pointer end has no address known at compile time, so the copy becomes a different lowering, not an extension of this one.

**Approach if revisited**: lower a pointer end to an indexed copy loop through the pointer (`LDA [src],Y` / `STA [dst],Y`), or call `stdlib/string.r65::memcpy` above some size; the design table in operator-overloading.md already names `memcpy` for this case. Accept the pointer form in `call_validator.py` once the lowering exists, and test near and far pointers at each end.

### `let x = a.clone()` with a custom `Clone` impl

The `.clone()` sugar works only for the compiler-generated bitwise clone (`impl Clone for T {}`). A type with a hand-written `clone_from` body is rejected, with a hint to write `x.clone_from(&a)`.

**Why deferred**: the sugar routes the source into the `let`'s initializer, which may decompose `x` into vregs that have no address. A user `clone_from` needs `&x` to pass as `self`.

**Approach if revisited**: when the impl is manual, force `x` addressable (as clone operands already are), then emit the ordinary static call `x.clone_from(&a)` as the initializer.

---

## Inlining

### Far functions that are not bank-safe

Inlining a far function into a caller in another bank needs a body that means the same thing from either bank (`optimize/inline.py::_far_body_is_bank_safe`). Two cases are refused "for now":

- **`#[mode(databank=inline|caller)]` functions.** The inlined copy would need its own PHB / set DBR / PLB bracket.
- **Bodies containing a `TraitDispatch`.** Whether a dispatch is bank-explicit depends on the trait's near/far calls and on `self_is_far`, so the predicate refuses all of them.

**Why deferred**: failing closed is the right default for a cross-bank correctness check. For the databank case, the bracket costs about as much as the JSL/RTL the inline saves, so it pays only when inlining also folds constant arguments.

**Approach if revisited**: for `TraitDispatch`, accept a dispatch whose trait methods are far and whose `self` is a far pointer, since that is bank-explicit on both axes; keep refusing the rest. For the databank modes, emit the bracket around the inlined body and gate it on a cost check, not on the attribute alone.

---

## Indexing an Element Reached Through a Pointer

Indexing twice through a pointer is a MIR error ("indexing into an element reached through a pointer is not supported", `mir/builder.py`):

```rust
#[zeropage]
static mut P: *[[u8; 4]; 4];
let v: u8 = P[1][2];    // error
```

A static base works (`GRID[i][j]`), and so does an array field through a pointer (`R.cells[j]` with `R: *Row`).

**Why deferred**: nested indexing landed for static bases, where the outer index folds into the base `MemoryLocation` (`resolve_array_base_memloc`). A pointer base has no static address to fold into. The full offset (`i * stride + j`) has to be computed into Y for `LDA (ptr),Y`, and a run-time `i` needs a stride multiply.

**Approach if revisited**: when the base resolves to a pointer (`try_pointer_deref_array_base`), sum the index terms into one offset and lower through the existing pointer-indexed path. **Workaround today**: point at a flat array (`*[u8; 16]`) and write `P[i * 4 + j]`.

---

## Packaging Hazards

Pre-existing global-state assumptions that a package system makes reachable. Each is fine in a single-author program and gets worse as package count grows. See [packaging-system.md](packaging-system.md) for the system these block.

### Statics Are Never Dead-Code Eliminated

`DeadFunctionEliminator` (`optimize/dead_function_elim.py`) prunes unreachable functions, but `MemoryAllocator.allocate_all` (`codegen/memory_alloc.py`) allocates every static in `mir_program.statics` regardless of reachability, and `symbol_gen.emit_all_definitions` emits every allocation. An unused static pulled in from a package still burns zeropage, RAM or ROM — so "pay for what you use" holds for code but not data.

**Why deferred**: harmless when every static in the build was written by the author of the build. Packaging is what makes it cost real bytes, and the lint L005 `dead_static_mut` only warns for *mutable* statics that are written and never read.

**Approach if revisited**: reachability over static reads/writes in MIR, seeded from the same entry points `_find_entry_points` already computes, then filter `mir_program.statics` before `allocate_all`. Care needed for `#[hw]` (never prune), `#[soa]` planes (prune as a group), and `include_bytes!` blobs referenced only from inline asm.

### Explicit Addresses Are Not Collision-Checked

`allocate_zeropage` takes an explicit `#[zeropage(0x40)]` address as-is with no overlap check (the comment at the site says so outright). Two packages hardcoding the same address overlap silently; only a stack-overlap warning (W006) exists.

**Why deferred**: explicit addresses are usually deliberate hardware or ABI choices, and the auto-allocator already avoids reserved bytes. The single-author case rarely collides.

**Approach if revisited**: record explicit ranges in the first allocation pass and error on overlap, with an opt-out attribute for the deliberate-aliasing case (unions over fixed addresses). Packages should prefer auto-allocation; `zeropage_budget` is the only current backstop.

### Program-Wide Singletons

Two constructs assume there is exactly one of them in the program, so a package cannot ship either for general use:

- **`#[soa]` structs and tagged enums** back exactly one static program-wide (`hir/builder.py::_bind_planed_storage`) — that is what lets their methods hard-code plane addresses.
- **`*dyn` TypeIds** come from a global counter assigned in declaration order (`hir/builder.py`), so adding a package shifts the TypeIds of everything declared after it.

**Why deferred**: both are load-bearing for the codegen they enable — the SoA singleton is the reason a field access inside a method is one instruction, and stable TypeIds would need a mangling scheme the flat namespace does not have.

**Approach if revisited**: for SoA, a per-static method instantiation (one copy of each method per backing static) trades ROM for the restriction. For TypeIds, hash the type name instead of counting, and check for collisions at build time.

---

## Parser

### Two places a statement-position `match` still needs `;`

A `match` in statement position ends its statement (`match_stmt` in `grammar.lark`), and loop and `if` bodies may end in one. Two shapes still fail:

```rust
let v: u8 = { match t { .. } G };   // a match, then a non-match tail value
{ G = 1; match t { .. } } G = 2;    // a bare block ending in a match, then more
```

**Why deferred**: both meet the decision `match_stmt` avoids. A `match_stmt` is never a block's last item, which keeps a match before `}` an expression, the block's tail value. The first shape needs a match_stmt followed by a tail expression; the parser can only tell which follows after it has committed. In the second, the bare block is ambiguous between a block statement and a `block_expr` until the token after its `}`; the match tail decides for `block_expr`, which then wants a `;`. Neither came up in real code.

**Approach if revisited**: Rust's model in full. Give `block` and `block_expr` one syntax, `"{" _stmt_item* expr? "}"`, prefer the statement reading at statement start (rule priority settles the reduce/reduce collisions, as it does for `match_stmt.1`), and recover a block's value after the fact from its last block-like statement (`ASTBuilder._if_forms` already does this for an `if` whose then arm ends in a match). The `if` and block forms would then need the same treatment, so `if_stmt`/`if_expr` and `block`/`block_expr` could merge.

---

## Macros

### More than one repeated parameter

A macro arm may have only one repetition group. A second `$(...),*` in the same pattern is an error ("has multiple repeated parameters (not supported)", `frontend/macros.py`).

**Why deferred**: the matcher splits an invocation into the simple parameters plus one repeated tail. Two groups need a rule for where the first one ends, and the macros written so far haven't needed one.

**Approach if revisited**: follow Rust's `macro_rules!`, where a group ends at the separator or literal token that follows it in the pattern. Reject patterns where that boundary is ambiguous, e.g. two adjacent groups with no token between them.
