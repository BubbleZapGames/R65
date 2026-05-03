# Register/Memory Configuration: DBR:Y vs D=S

This document describes the two strategies the R65 compiler uses to access
memory through stack-resident pointers, the cases that pick each one, and the
prologue/epilogue/access shapes the codegen emits. It is aimed at compiler
contributors — language users normally don't need to think about it.

There are two largely independent decision points:

1. **Trait method self-pointer access** — how a trait method reaches `*self`.
   For near self this is trivial (Y holds the address). For `far *self` the
   compiler picks per-method between **DBR:Y** (fast, leaf-only) and **D=S**
   (full-featured, more expensive).
2. **Far pointer stack parameters in any function** — when a function takes
   `far *T` as a stack parameter (trait method or otherwise), the compiler
   picks per-function between **SET_DBR** and **D_EQUALS_S** based on a cost
   model.

The two decisions interact: a far-self trait method that needs D=S is also a
function with a far-pointer stack param, so it goes through the same far-ptr
strategy machinery — but with `self_far_uses_d_equals_s = True` it is forced
onto D=S regardless of cost.

---

## 1. Trait method self pointer

Trait methods always receive `self` in `Y`. Field access then uses `LDA d,Y` or
`LDA [d],Y` depending on whether `self` is near or far.

### 1.1 Near `*self` (default)

- Caller: loads 16-bit pointer into `Y`, leaves DBR alone, `JSR` (near trait)
  or `JSL` (far trait) to the dispatch wrapper.
- Dispatch wrapper: `LDA $0000,Y` reads TypeId at offset 0, indexes the jump
  table, branches to the impl.
- Callee: `self_y_vreg` is pre-allocated to `PhysicalLocation(HARDWARE, 'Y')`
  in `function_gen.py` and added to `hw_allocs`. Field access `LDA $offset,Y`
  costs 5 cycles. Self lives in Y for the entire method.
- DBR is the caller's DBR. Field reads use absolute-indexed addressing relative
  to DBR, which is fine because the object lives in DBR's bank by definition
  for a near pointer.

This is the cheapest mode. No prologue/epilogue overhead.

### 1.2 Far `*self`

A far trait pointer is 24-bit. The dispatch caller has to deliver three things
to the impl: the 16-bit address (in Y), the bank (in DBR), and the TypeId
(read by the dispatch wrapper via `LDA [ptr]` since DBR isn't set yet at that
point — see `call_select.py` and the trampoline emission in `codegen.py`).

The **caller side** is the same in both far-self submodes (see
`call_select.emit_trait_dispatch`, ~line 437):

```
PHB                       ; save caller's DBR
<load_y_with_far_self>    ; bank byte → A → PHA → PLB; addr → Y
JSR/JSL <dispatch>        ; near or far trait
<return value collection>
PLB                       ; restore caller's DBR
```

The **callee side** picks one of two paths per method, decided by
`function_gen._analyze_far_self_trait_method` (line 1426). The flag is
`MIRFunction.self_far_uses_d_equals_s`.

#### 1.2.1 DBR:Y path (leaf methods)

Used when the method body does **not**:

- Make any `Call` or `TraitDispatch`
- Read or write any `MemoryLocation` with `storage_type` of `'rom'` or `'hw'`

In this case DBR is already set to the object's bank by the caller, the address
is in Y, and field access is `LDA $offset,Y` exactly like the near case (5
cycles). The method cannot call other functions or touch ROM/HW because doing
so would assume DBR is the caller's DBR — but it isn't, it's the object's
bank.

Codegen treatment:

- `self_y_vreg` pre-allocated to `HARDWARE Y`, identical to near self.
- No prologue or epilogue overhead specific to far self.
- `_has_y_self()` returns True; field access selectors emit `LDA d,Y`.

#### 1.2.2 D=S path (non-leaf methods)

Used when the method has any call (regular or trait), or any ROM/HW access. In
that case DBR must stay at the caller's DBR throughout the body (so calls and
ROM/HW reads work normally), which means we cannot leave self in a "DBR:Y
implies bank" arrangement. Instead we push the full 24-bit far pointer onto
the stack and use `[dp],Y` indirect-long addressing.

`_analyze_far_self_trait_method` sets:

```python
mir_func.self_far_uses_d_equals_s = True
mir_func.has_far_ptr_stack_params = True
```

That second flag routes the function through the general far-ptr-strategy
machinery (Section 2). For far-self D=S the strategy is forced to
`D_EQUALS_S` because `_is_set_dbr_safe` returns False when
`self_far_uses_d_equals_s` is set.

Prologue (after standard preserves and after A-param save):

```
PHB                       ; push bank byte of self (1)
PHY                       ; push 16-bit address of self (2)
PHD                       ; save Direct Page (2)
TSC                       ; A ← S
TCD                       ; D ← S, enabling [dp],Y addressing
```

`ABIInfo.prologue_stack_bytes` includes 3 bytes for `PHB+PHY` when
`self_far_uses_d_equals_s`. The far self pointer now lives on the stack as a
24-bit value at a fixed offset; `self_y_vreg` is pre-allocated to `STACK` at
`prologue_bytes - 3 + frame_size + 1` (function_gen.py line ~250). The
incoming `Move(dest=self_y_vreg, source=HW_Y)` becomes a no-op (move_select.py
skips it because the stack copy is already in place via PHY).

Field access uses the existing D=S indirect-long codegen: `LDA [dp_offset],Y`
where `dp_offset` is the slot holding the 24-bit self pointer and `Y` is the
field offset.

Epilogue:

```
PLD                       ; restore DP (also pops the TSC/TCD setup byte)
                          ; ... preserved register restores ...
```

The 3 bytes of `PHB+PHY` are not directly popped — they are part of the stack
frame and are reclaimed by `_emit_stack_param_cleanup`, which adds 3 to
`cleanup_frame_size` for D=S far-self trait methods.

### 1.3 Decision summary (trait method self)

| `self` kind | Body has call/ROM/HW? | Mode | Self storage | Prologue cost |
|-------------|----------------------|------|--------------|---------------|
| `*self` (near) | any | DBR:Y near | Y (entire body) | 0 |
| `far *self` | no | DBR:Y far (leaf) | Y (entire body) | 0 |
| `far *self` | yes | D=S | stack (3 bytes) | PHB+PHY+PHD+TSC+TCD ≈ 13 cyc |

### 1.4 Self-pointer reuse (chain coalescing)

When the *caller* makes back-to-back `TraitDispatch` calls on the same
self vreg, each dispatch redundantly re-emits the per-call setup. For
**far-self** chains the redundant work is the
`PHB / LDA bank,S / PHA / PLB ... PLB` DBR bracket plus a Y reload;
for **near-self** chains it is just the Y reload. The
`analyze_trait_dispatch_chains` pass (folded into
`analysis/far_ptr_strategy.py`) detects runs of dispatches and elides
the redundant work.

Each `TraitDispatch` MIR instruction carries two independent fields
set by the chain pass:

- `self_chain_role: ChainRole` (default `SOLO`) — controls the DBR
  bracket position for far-self dispatches AND the LDY position for
  near-self dispatches.
- `self_y_preloaded: bool` (default `False`, v2(C)) — set on
  `MIDDLE`/`END` members of a far-self chain when every method's
  impls and every gap instruction also pass the Y-preservation
  predicate. The codegen then skips the Y reload on those members.

| Self | Role | PHB at start | DBR-set | Y reload | PLB at end |
|------|------|-------------|---------|----------|-----------|
| far  | `SOLO`   | yes | yes | yes | yes |
| far  | `START`  | yes | yes | yes | no  |
| far  | `MIDDLE` | no  | no  | yes¹ | no  |
| far  | `END`    | no  | no  | yes¹ | yes |
| near | `SOLO`   | n/a | n/a | yes | n/a |
| near | `START`  | n/a | n/a | yes | n/a |
| near | `MIDDLE` | n/a | n/a | no  | n/a |
| near | `END`    | n/a | n/a | no  | n/a |

¹ Skipped when `self_y_preloaded` is True (v2 commit C).

Soundness conditions for fire:

- Same self — compared via *cast-transparent root* (v1.6, see below).
- **v1.5**: methods may differ between dispatches. The bracket
  itself is method-independent — only the indirect JSR target (the
  dispatch wrapper) changes between calls. The chain predicate runs
  over the union of impl sets across every distinct
  `(trait, method)` pair reached in the chain.
- **v1.6**: traits may also differ between dispatches, as long as the
  self pointers resolve to the same root through trivial trait-
  pointer casts. The `_chain_self_root` walker walks back through
  `Move` (pure SSA rename) and `TypeConvert` nodes whose source and
  target are both `PointerTypeInfo` with matching `is_far` and
  matching `size_bytes`. This covers `*Trait → *OtherTrait` and
  `*Struct → *Trait` casts (zero-cost, address-preserving) and
  rejects far↔near casts and pointer arithmetic / field-offset
  shifts. Cached per-function via `MIRFunction._chain_self_roots`.
- **v2 commit A**: the chain pass extends across simple if/else CFG
  diamonds when both arms are gap-safe and don't redefine the chain
  root self vreg. Loops, multi-arm switches, and joins with more
  complex topology remain unsupported. Diamond bridging is performed
  by `_extended_path` / `_try_bridge_diamond` and applies to both far
  and near chains.
- **v2 commit B**: near-self chains are coalesced by a parallel pass
  using the Y-preservation predicate `_function_preserves_y` (memoized
  on `MIRFunction._chain_preserves_y`). Conservative rejections:
  inline asm, recursion, calls to non-Y-preserving callees, vreg
  with `register_hint='Y'` (loop register promotion target), explicit
  Move-to-`Y`. The trait-method entry-point
  `Move(dest=self_y_vreg, source=HW_Y)` is treated as a no-op
  (Y already holds self at function entry).
- **v2 commit C**: far-self chains additionally elide the Y reload
  for MIDDLE/END members when every method's impls AND every gap
  instruction pass the Y-preservation predicate. The flag and the
  DBR bracket role are *orthogonal*: a chain may have DBR coalescing
  only, Y elision only (rare; would imply near), or both.

Predicates and their disqualifiers:

| Predicate | Disqualifiers (gap instruction or function body) |
|-----------|--------------------------------------------------|
| DBR-independent (`_instr_is_dbr_independent` / `_function_is_dbr_independent`) | non-LONG RAM/lowram access; near pointer deref; inline `MemoryLocation` operands on Compare/BitTest/BinaryOp/UnaryOp pointing at RAM/lowram; nested `TraitDispatch`; call to non-DBR-independent function; functions that use `SET_DBR` / `self_far_uses_d_equals_s` / are in a call-graph cycle |
| Y-preserving (`_instr_preserves_y` / `_function_preserves_y`) | inline asm; nested `TraitDispatch` (dispatch wrapper clobbers Y); call to non-Y-preserving function; vreg with `register_hint='Y'`; explicit Move/etc. with HW-Y dest; recursion |

In addition every gap walker rejects redefinition of the chain's root
self vreg (the address-holder behind any cast-transparency aliases).

The chain pass runs after `analyze_far_ptr_strategy` so each impl's
`far_ptr_strategy` is already known. It only mutates
`self_chain_role` and `self_y_preloaded` on `TraitDispatch` nodes.

The codegen side lives in `call_select.emit_trait_dispatch`, which
inspects both fields and skips the appropriate parts of the bracket
or Y reload. The DBR-bracket helpers were split into
`_set_dbr_from_far_self` and `_load_y_addr_from_far_self`; the
single-shot `load_y_with_far_self` calls both.

Stack accounting for chain MIDDLE/END members: the START member's
`PHB` byte stays on the stack across the run, so MIDDLE/END dispatches
that read self from the stack must offset by +1. This is handled
automatically because `region_state.stack_tracker.push(1)` is called
at START (and `pop(1)` at END), and the resolver adjusts every
stack-relative load via `spill_offset` already.

---

## 2. Far pointer stack parameters

Any function (trait method or not) that takes `far *T` as a *stack* parameter
sets `MIRFunction.has_far_ptr_stack_params = True`. The compiler then picks
between two strategies in `analysis/far_ptr_strategy.py`:

- `FarPtrStrategy.D_EQUALS_S`
- `FarPtrStrategy.SET_DBR`

The strategy is stored on `MIRFunction.far_ptr_strategy` and consulted by
codegen (function_gen.py prologue/epilogue, memory_select.py and
instruction_select.py for addressing-mode selection, call_select.py for
per-call wrappers).

### 2.1 D_EQUALS_S

Sets `D = S` so that `[dp],Y` indirect-long addressing reaches the 24-bit
pointer on the stack.

Prologue:
```
PHD ; TSC ; TCD             ; ~13 cycles
```

Epilogue:
```
PLD                         ; ~5 cycles
```

Per-access:
```
LDA [dp_offset],Y           ; one instruction, ~7 cycles
```

Per-call: D is no longer pointing at zeropage, so any call that wants to use
zeropage scratches has to deal with it. The current implementation pays a 13
cycle per-call wrapper cost in the cost model.

**Trade-offs**:

- ✅ Single per-access instruction; no extra setup for each dereference.
- ✅ Works with multiple far-pointer params (different banks per access).
- ❌ DP no longer points at zeropage → **scratch registers and DP-relative
  zeropage access stop working** for the duration of the function. That's
  why `far_ptr_strategy._choose_strategy` forces `SET_DBR` whenever
  `func.scratch_param_addrs` is non-empty (and SET_DBR is otherwise safe).
- ❌ ROM/HW absolute reads still work (DBR unchanged), but require LONG
  addressing if not in DBR's bank — already the default assumption.

### 2.2 SET_DBR

Sets `DBR` to the bank byte of the (single) far-pointer stack param. Keeps DP
intact so zeropage and scratch registers continue to work.

Prologue (function_gen.py ~line 1027):
```
PHB                         ; save caller DBR
LDA <bank_offset>,S         ; load bank byte from far ptr arg
PHA ; PLB                   ; DBR = arg's bank
                            ; ~19 cycles total
```

Epilogue:
```
PLB                         ; restore caller DBR
```

Per-access:
```
LDA (<addr_offset>,S),Y     ; bare stack-relative-indexed-Y
                            ; ~8 cycles
```

Per-call wrapper (the function's body is in the arg's bank, but callees expect
their own DBR conventions, so each call site rebrackets DBR):
```
PHB ; PHK ; PLB             ; DBR ← PBR before call
JSR/JSL <callee>
PLB                         ; restore arg's bank
```

That extra PHB/PHK/PLB/PLB bracket is the 14*N_calls term in the cost model.

**Trade-offs**:

- ✅ DP intact → zeropage scratch registers and DP-resident statics keep
  working.
- ✅ ROM/HW addressing in the function's body benefits if the bank happens to
  match — but in the general case ROM/HW still uses LONG.
- ❌ Only safe with a **single** far-pointer stack param (can't set DBR to two
  banks at once).
- ❌ Near pointer derefs in the same function become unsafe — changing DBR
  changes which bank the near pointer points into. `_is_set_dbr_safe`
  rejects functions with non-trait-self near `LoadIndirect`/`StoreIndirect`
  via VirtualRegister pointers.
- ❌ Far-pointer derefs through *non-param* far pointers also unsafe (their
  bank could differ from the param's bank).
- ❌ Each call site pays ~14 extra cycles for the PHB/PHK/PLB/PLB bracket.
- ❌ RAM stores at `$7E` need their bank stripped for ABSOLUTE-mode emission
  rather than getting LONG forced — handled by emitter.py.

### 2.3 Choosing between them

`_choose_strategy` (far_ptr_strategy.py:33):

```python
if func.scratch_param_addrs and _is_set_dbr_safe(func):
    return FarPtrStrategy.SET_DBR        # forced — D=S would break scratches

if not _is_set_dbr_safe(func):
    return FarPtrStrategy.D_EQUALS_S     # forced — SET_DBR not safe

# cost-driven choice
d_equals_s_cost = 13 + n_zp + 13*n_calls
set_dbr_cost    = 19 + n_rom + n_hw + n_ram + 14*n_calls
return SET_DBR if set_dbr_cost < d_equals_s_cost else D_EQUALS_S
```

Where the access counts come from `_count_accesses` walking every `Load` /
`Store` / `Call` / `TraitDispatch` in the function's MIR. Zeropage accesses
are cheap under D=S (DP is somewhere else, but they go through ZP scratch
mechanisms separately) and expensive under SET_DBR — that's the dominant
term. Calls are slightly more expensive under SET_DBR because of the
PHB/PHK/PLB/PLB bracket.

`_is_set_dbr_safe` rejects:

1. More than one far-pointer stack param (`len(func.far_ptr_param_indices) > 1`).
2. `func.self_far_uses_d_equals_s` is True (D=S has already been forced).
3. Any `LoadIndirect`/`StoreIndirect` through a non-far vreg pointer (near
   pointer deref).
4. Any far indirect through a vreg that isn't one of the far-ptr params.

### 2.4 Storage-type-aware addressing

The strategy decision is also threaded through addressing-mode selection on
`MemoryLocation`s via `PhysicalLocation.storage_type`:

- **D_EQUALS_S**: ROM uses default LONG (`.l` suffix), HW uses LONG, RAM uses
  default DBR-relative ABSOLUTE.
- **SET_DBR**: ROM uses LONG (`.l` suffix), HW uses LONG, RAM at `$7E` strips
  the bank for ABSOLUTE form (the callee's RAM is in `$7E`, but DBR is the
  param's bank, so we have to override). See emitter.py for the per-storage
  decision and memory_select.py for `is_long_under_set_dbr` (the STZ-LONG
  guard — STZ has no LONG mode, so STZ is rewritten to LDA #$00 / STA when
  SET_DBR forces LONG).

---

## 3. Interaction matrix

For a single function, the combinations actually emitted:

| trait method | self kind | has far ptr stack params? | strategy | notes |
|--------------|-----------|---------------------------|----------|-------|
| no | — | no | (none) | regular function |
| no | — | yes | cost-driven SET_DBR or D=S | classic far-ptr params |
| yes | near | no | (none) | self in Y |
| yes | far, leaf | no | (none) | DBR:Y far self, self in Y |
| yes | far, non-leaf | yes (forced) | D=S (forced via `_is_set_dbr_safe`) | self on stack at fixed offset, accessed via `[dp],Y` |
| yes | far, leaf, but body also takes another `far *T` stack param | yes | cost-driven, but `_is_set_dbr_safe` may reject | rare |

---

## 4. Files to grep when changing this

- `r65/compiler/analysis/far_ptr_strategy.py` — the cost model and safety check.
- `r65/compiler/mir/nodes.py` — `FarPtrStrategy` enum, `MIRFunction` flags
  (`has_far_ptr_stack_params`, `far_ptr_strategy`, `self_far_uses_d_equals_s`,
  `self_y_vreg`, `is_trait_method`, `far_ptr_param_indices`).
- `r65/compiler/codegen/function_gen.py` —
  - `_analyze_far_self_trait_method` (line 1426): DBR:Y vs D=S decision for
    far-self trait methods.
  - `emit_prologue` (around line 990–1040): mode setup, far-ptr prologue
    emission for both strategies.
  - `emit_epilogue` (line 1483): PLB / PLD restore.
  - Pre-allocation of `self_y_vreg` (line 205+): HW Y for DBR:Y, STACK for
    D=S.
- `r65/compiler/codegen/call_select.py` — `emit_trait_dispatch` (line ~430),
  `load_y_with_self` (631), `load_y_with_far_self` (679); per-call SET_DBR
  wrappers.
- `r65/compiler/codegen/memory_select.py` — `[dp],Y` lowering for D=S; LONG
  vs ABSOLUTE selection under SET_DBR; `is_long_under_set_dbr` STZ guard.
- `r65/compiler/codegen/instruction_select.py` — addressing-mode selection
  consulting `PhysicalLocation.storage_type`.
- `r65/compiler/codegen/emitter.py` — RAM bank stripping under SET_DBR; `.l`
  emission for ROM/HW.
- `r65/compiler/codegen/abi.py` — `ABIInfo.prologue_stack_bytes` accounting
  including the +3 for `PHB+PHY` on D=S far-self trait methods.
- `r65/compiler/codegen/control_flow_select.py` — `_emit_stack_param_cleanup`
  adds the +3 to `cleanup_frame_size`; `_get_stack_param_bytes` skips self
  for trait methods.
- `r65/compiler/codegen/register_alloc.py` — local offset shifting by
  `outgoing_arg_bytes`; `hw_allocs` tracking.

---

## 5. Known gotchas

- **Multi-byte slot reservation**: a u16 scratch param at `$00` occupies both
  `$00` and `$01`. Both `function_gen` and `fixedstack_params` mark all
  overlapping scratch addresses as occupied; a missed overlap causes silent
  scratch-collision corruption (see fixed-bugs.md).
- **D=S kills DP**: any code that wants to read zeropage statics or scratch
  registers inside a D=S function fails silently because DP no longer points
  at zeropage. The strategy chooser forces SET_DBR whenever
  `scratch_param_addrs` is non-empty (and SET_DBR is otherwise safe).
- **STZ has no LONG**: under SET_DBR, ROM/HW labels are forced to LONG.
  `STZ.l` doesn't exist; memory_select.py rewrites STZ to LDA #$00 / STA in
  that path (`is_long_under_set_dbr`).
- **PHA region spills break callee param offsets**: when caller-owned outgoing
  arg bytes are active, region-spill PHA/PHX between frame and JSR shifts the
  callee's view of params. The hybrid STA/PHA path in call_select.py adjusts
  for this (`_region_state.spill_offset`).
- **Move(self_y_vreg, HW_Y) is a no-op**: only when D=S far-self path has
  pre-allocated self_y_vreg to STACK and the PHY has already pushed the
  value. move_select.py must skip emitting any code for that Move.

---

*Last updated: 2026-05-02*
