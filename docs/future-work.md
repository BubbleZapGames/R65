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

## Conventions

When adding to this file:

- Keep entries scoped to *concrete* deferred items, not vague aspirations.
- For each, capture: what it does, why it was deferred, sketch of the approach if revisited.
- If an item lands, delete its entry (don't strike it through — git history preserves the past).
- Group related items under a heading describing the area of work.
