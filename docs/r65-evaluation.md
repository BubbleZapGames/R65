# R65 Language Evaluation — Building a 32-Area Text Adventure

*July 2026*

An evaluation of R65 carried out by writing a complete SNES game in it:
**DERELICT**, a multiple-choice text adventure with 32 areas, 8 items,
requirement-gated choices, and a word-wrapped prose renderer. The project
lives in `textadv/`.

The goal was not the game — it was to find out what R65 is like to build a
real, non-trivial program in. A text adventure was chosen deliberately: it is
data-heavy rather than compute-heavy, so it leans on the parts of the language
that a sprite demo never touches — ROM tables, string handling, struct arrays,
far pointers, and cross-function data flow.

## Verdict

R65 delivers on its central promise. The hardware model is honest, the type
system catches real mistakes, and the generated code is tight. A working
32-area game is **408 lines of engine plus 395 lines of generated data**,
compiling to 5,293 lines of assembly and 17 KB of ROM. Nothing about the
language got in the way of expressing the design.

What it is *not* yet is well-trodden. Every bug found below sits on a path
that is obvious the moment you write a data-driven program — a ROM table of
string pointers, a helper returning a pointer into it, a loop that prints
characters — and every one of them was broken. The core is sound; the edges
have not been walked.

Four compiler/stdlib defects were found and fixed. Three were silent
miscompiles.

---

## What worked well

**The data model mapped cleanly onto the hardware.** The whole world is two
packed ROM tables and no per-area code:

```rust
struct Area   { name: far *u8, desc: far *u8, nchoices: u8, _pad: u8 }
struct Choice { text: far *u8, dest: u8, req: u8, give: u8, _pad: u16 }
```

`req`/`give` are inventory bit indices, so gating and pickups are pure data.
Adding an area is a table row. This is exactly the kind of structure the
"hardware transparency" philosophy should make easy, and it did.

**Packed structs with no padding** meant struct layout was predictable enough
to reason about stride and ROM cost directly. Knowing `Area` is 8 bytes,
not "8 bytes if the compiler feels like it", is genuinely useful at this scale.

**The type system caught real errors.** Enum-vs-`u8` mismatches, missing casts,
and mutability violations all surfaced at compile time. The errors point at the
right line with useful context.

**Codegen quality is good.** The optimiser reported dead-function elimination,
far-to-near conversion, loop register promotion, and stack-slot reuse on every
build. A full screen redraw — clearing 840 tilemap cells and laying out ~230
glyphs through the stdlib `Console` API — costs ~321,000 cycles, about 0.9 of
an NTSC frame. That is comfortably interactive.

**Toolchain honesty.** The compiler exits non-zero and writes no output file on
error, so `make` cannot silently assemble a stale `.asm`. I briefly suspected
otherwise and was wrong — worth stating, because it is the failure mode that
would have made everything else untrustworthy.

---

## Bugs found and fixed

### 1. Address-of in a static initializer silently emitted zero

The single most damaging bug. A ROM table of string pointers — the canonical
shape for this entire genre of program — linked cleanly with every pointer
field set to `$00,$00,$00`:

```rust
static AREAS: [Area; 2] = [
    Area { text: &AREA0_TEXT as far *u8, exit_n: 1 },
    ...
];
```
```
__AREAS_data:
.db $00, $00, $00, $01, $00, $00, $00, $00, $00, $01
```

The `u8` fields were correct; only the pointers were null. `_extract_struct_literal_bytes`
and `_emit_array_literal_init` ran every element through the const evaluator and
substituted `0` whenever it returned `None`. An address-of is not a compile-time
integer, so it became zero — no error, no warning.

**Fix**: lower address-of to WLA-DX `<label` / `>label` / `:label` bytes via a
new `SymbolByte` MIR node, and make any *other* non-constant initializer a hard
error instead of a silent zero. Taking the address of a mutable (RAM) static is
rejected explicitly, since its address is not assigned until after MIR lowering.

`r65/compiler/mir/nodes.py`, `mir/lowerers/static_init.py`, `codegen/codegen.py`.
Tests: `tests/compiler/codegen/test_static_initializer_address_of.py`.

### 2. `D = S` not re-established after a call inside a far-pointer loop

The most insidious bug, because the failure looked like data corruption. This
loop — walk a string, do something with each character — printed one character
and stopped:

```rust
loop {
    let c: u8 = s[i];
    if c == 0 { break; }
    CON.putc(c);
    i++;
}
```

A function that dereferences a `far *T` parameter sets `D = S` so it can use
`[dp],Y`. Around an inner call it emits `PLD`, then afterwards `PHD` — plus
`TSC`/`TCD` to put `D` back at the frame, *if* more far dereferences follow.
That decision came from `_has_far_ptr_derefs_after_call`, which iterated
`blocks.values()` linearly and treated "later in the block map" as "after".

In a loop the dereference lives in the header block, which precedes the call's
block in the map but follows it across the back edge. The scan concluded "no
further dereferences", `D` stayed at the caller's value, and iteration 2 read
direct page `$0009` instead of the frame slot.

**Fix**: replace the linear scan with a CFG reachability query that follows
successors, back edges included.

`r65/compiler/codegen/call_select.py`.
Test: `tests/compiler/codegen/test_far_ptr_d_restore_loop.py`.

### 3. Returning a `far *T` silently truncated to one byte

```rust
far fn area_ptr(id: u8) -> far *Area {
    return &AREAS[id as u16] as far *Area;
}
```

The callee builds the 3-byte pointer in its stack frame, keeps only the low
byte in `A`, and tears the frame down:

```
LDA $01,S
TAX  ; Save return value A in X     <-- one byte of three
TSC / ADC #$08 / TCS                <-- frame, and the other two bytes, gone
```

The caller then read all three bytes back out of the deallocated frame. This
*appeared to work* until an unrelated later call happened to overwrite that
stack region — which is how it presented: adding a second use of the returned
pointer broke the first one.

The return ABI is register-count based (A, then B/X, then Y), so each value has
at most 2 bytes to travel in. Nothing checked that the value fits, and nothing
in the stdlib or test suite returns a far pointer, so it had never been
exercised.

**Fix**: reject return values wider than a register at type check, with a
message pointing at the alternatives. A proper 3-byte return convention is a
real ABI change (register triple, caller reassembly, interaction with
`#[preserves]`, frames, and multi-return ordering) and was out of scope here —
but a hard error is strictly better than a wild pointer.

`r65/compiler/typeck/type_checker.py`.
Test: `tests/compiler/typeck/test_oversized_return.py`.

### 4. Three sneslib macros had never compiled

`write_color!` was missing a semicolon; `enable_nmi!` and `enable_autojoy!`
both had `u8`-vs-`Interrupt` type errors. All three fail at expansion, so any
use is an instant compile error — meaning nothing in the repo has ever called
them.

```rust
macro_rules! write_color($color:expr) {
    A = $color as u16;
    CGDATA = A as u8      // <-- no semicolon
    CGDATA = B
}
```

**Fix**: corrected in `stdlib/sneslib.r65`. Macro bodies are only syntax-checked
at expansion, so an unused macro is never validated — worth a lint or a
smoke-test file that expands every stdlib macro once.

---

## Rough edges (not bugs, but friction)

**`static mut` in RAM is not zero-initialised, and `snes_init()` enables NMI
before `main()` runs.** My NMI handler tested an uninitialised `DIRTY` flag,
fired a DMA mid-`upload_font`, reset `VMADD`, and sent half the font to the
wrong VRAM address. The glyphs for ASCII ≥ 72 were simply missing. This is a
legitimate consequence of the documented memory model, but the template hands
you an NMI handler and an enabled NMI with no hint that the window between
`snes_init()` and your setup code is live.

**`NMITIMEN` ($4200) is write-only, but `enable_nmi!`/`disable_nmi!` do a
read-modify-write on it**, folding open bus into the value. `disable_nmi!()`
happened to work; the pair did not. These macros should use a shadow variable.

**Non-power-of-2 array strides can't be runtime-indexed.** `[far *u8; N]` — an
array of far pointers, the most natural spelling of a string table — fails with
*"Multiply operator only supports powers of 2 (got 3)"*. The workaround is to
pad structs to 8 bytes, which is what the game does and is arguably the right
call on this hardware. But the error surfaces at codegen, deep into the build,
rather than at the declaration.

**No lint rule ever ran inside an `impl` method.** C001 ("PPU register write
unsafe during active rendering") missed the `$2118` writes inside
`Console::print` reached from the main loop — precisely the bug it exists to
catch — while flagging the identical write in a free function.

The rule itself is fine and is already interprocedural. Both `linter.py` and
`lint/call_graph.py` filtered `program.declarations` with `isinstance(decl,
HIRFunctionDecl)`, but impl methods are desugared to `HIRFunctionDecl`s named
`Struct__method` that live *inside* their `HIRImplDecl`. So method bodies were
never walked by any rule, and were absent from the call graph. Compounding it,
`_record_call` resolved callees only via `call.func`, which is `None` for a
method call — the type checker had already put the target in
`method_call_info['mangled_name']`, unused — so every method call degraded to
an indirect edge, and `reachable_from` ignores those.

Fixed: iterate impl methods in both walkers, and resolve the mangled name
before treating a call as indirect. (`lint/linter.py`, `lint/call_graph.py`;
test `tests/compiler/lint/test_impl_method_reachability.py`.)

The remaining noise is *not* a rule defect. C001 is deliberately syntactic — it
does not model forced blank, because blanking is global mutable state that any
function can change. Suppressing writes that merely *look* safe would trade a
sound rule for an unsound one.

Two mechanisms handle it instead. `exempt_on_write_addrs = [0x2100]` skips any
function that writes `INIDISP`: touching it means the function is managing
forced blank deliberately, so its PPU writes are the programmer's call. That is
a heuristic rather than an analysis, and it is whole-function — writes textually
before the guard write count too. It alone takes C001 from 24 warnings to 8,
with no per-project function names to maintain.

The exemption is anchored to the init prologue: it covers only accesses
*before the first loop* in the function, and ends there. Forced blank is
established once during init in straight-line code; the first loop is the game
loop, running with the screen on. So a `main` that blanks during its own init
cannot silence a PPU write down in its game loop — nor one placed after the
loop, nor anything that fades brightness from the loop. This is a strict
tightening (it can only add warnings), and it is position-sensitive in the way
the guard actually needs: OK to write at the very top, not once the loop
begins. The cost is that an init routine which uploads a palette *in a loop*
while blanked now warns; that is the safe direction to err, and
`exclude_subtrees` covers it.

The exemption stops at the function boundary, so helpers called during forced
blank (here `Console::upload_font`) still need `exclude_subtrees`. With
`setup_video` listed there as well, C001 reports 3 — and those 3 are the genuine
active-display writes.

*(Update, Aug 2026)* Three follow-ups tightened this further, all in
`lint/rule_kinds/reachability_forbidden_access.py` and `lint/call_graph.py`:

* **The exemption is now position-sensitive** as described above — it ends at
  the first loop rather than covering all of a guarded function's straight-line
  code. Previously a PPU write after the game loop began was silenced by an
  `INIDISP` write up in init.
* **Function-pointer calls are now traced.** The HIR call graph collected
  `address_taken` but `reachable_from` never followed it, so any forbidden
  access reached only through a function pointer — including a ROM jump table,
  `static TABLE: [fn(); N] = [...]`, the canonical SNES dispatch shape — was a
  blind spot. A caller that makes an indirect call is now widened to the whole
  address-taken set (a sound over-approximation mirroring the MIR call graph),
  and function references in static initializers are recognised as
  address-taken. Trait/method dispatch remains unresolved — a separate case.
  Tests: `tests/compiler/lint/test_fnptr_reachability.py`,
  `test_exempt_on_write.py::test_exemption_ends_at_the_first_loop`.
* **Init-reachable functions are exempted automatically.** The
  "before the first loop = init" boundary now applies across the call graph,
  not just inside one function: a function reachable from an entry point *only*
  through calls that precede the entry's first loop runs entirely during init,
  so its straight-line PPU writes are exempt without an `exclude_subtrees`
  entry. A boot-time video-setup helper no longer needs listing. The loop
  carve-out carries over — a write inside a loop in such a function is still
  flagged — and a helper also reachable through the game loop (e.g. a
  level-load routine called both at boot and mid-game) is not init-only and
  stays checked. On the Pac-Man clone this drops `boot_load` from the config;
  the surviving warnings are the genuinely game-loop-reachable vblank worker
  and level-load helpers, which still assert their safety via `exclude_subtrees`.
  Test: `tests/compiler/lint/test_init_reachable_exemption.py`.
* **Per-item suppression via `#[allow(...)]`.** A Rust-style attribute now
  suppresses named lint codes on an item: `#[allow(C001)] fn vblank_tasks()`,
  `#[allow(L001, C001)]`, or `#[allow(all)]`. It is lexically scoped — the
  codes are silenced for diagnostics *inside* the annotated function or static,
  not transitively through the functions it calls — which makes it the
  complement of `exclude_subtrees` (whole call tree, in config): use `#[allow]`
  when a function's *own* writes are deliberate, `exclude_subtrees` when a whole
  helper tree runs in a safe window. Suppression is applied as a post-pass over
  the collected diagnostics keyed on each item's source span, so it works
  regardless of whether a rule reports during setup, the walk, or finalize —
  not just for C001. Attribute plumbing in `hir/attributes.py`, `hir/nodes.py`,
  `hir/builder.py`; suppression in `lint/linter.py`. Test:
  `tests/compiler/lint/test_allow_attribute.py`.

**The project scaffold ships 3 of 13 stdlib files.** `r65x init` copies
`sneslib`, `math`, and `65816`. A text-oriented project needs `console`,
`string`, `default_font`, and `rand`, all of which must be copied by hand.

**`Console` has no word wrap**, so the game implements its own `print_wrapped`.
That is reasonable for a minimal stdlib, but it is the first thing any text-
heavy program needs.

**`Console::print` reads `self.buffer` unconditionally**, including in VRAM mode
where it is unused, producing uninitialised-memory warnings in the emulator.

---

## Cost model observations

The stdlib `Console` API costs roughly 1,400 cycles per glyph (321k cycles for
a clear plus ~230 glyphs). A hand-written tilemap blit would be closer to 20.
The gap is not the language — it is that every `putc` is a far method call that
re-reads a dozen struct fields through a far pointer. R65's "zero abstraction
cost" claim holds for language constructs; it does not automatically hold for
stdlib *design*. For this game it is irrelevant (under one frame per redraw);
for a per-frame text renderer it would matter.

---

## Testing story

This turned out to be a strength. Mesen-GDB's batch mode plus synthesised movie
files made the game genuinely testable from CI:

- `tools/mkmovie.py` builds a Mesen `.mmo` from a compact action script
  (`down,down,a`). The format is a zip with a plain-text per-frame input log.
- The ROM carries a `snapshot_point()` hook that fires after three seconds of
  no input, so a movie can drive the game into any state and the harness breaks
  on a settled frame.
- `tools/solve.py` BFSes the world graph over `(area, cursor, items)` and emits
  the winning button sequence — which doubles as proof the world is completable.
- `make verify` chains these: solve the graph, replay the route on the ROM,
  assert the player reached the ending area.

Being able to assert `[$0200] = $1F` after a 38-button scripted playthrough is
a strong position for a retro toolchain to be in.

---

## Recommendations

Roughly in priority order:

1. **Test the data-table path.** Bugs 1–3 are all one program shape: build a
   ROM table, get a pointer into it, walk it. A single end-to-end test of that
   shape would have caught all three.
2. **Audit for other silent-zero fallbacks.** Bug 1 was a `if value is None:
   value = 0`. That pattern should be an error everywhere it appears.
3. **Decide on far-pointer returns.** Either implement a 3-byte return
   convention or keep the type error. The error is in place now; the language
   reference should say so, since `-> far *T` reads as obviously legal.
4. **Add a stdlib macro smoke test** that expands every macro once. Three dead
   macros in one file suggests more elsewhere.
5. **Ship the whole stdlib in `r65x init`**, or let the template declare which
   files it needs.
6. **Give `Console` word wrap**, and skip the unused `self.buffer` read in
   VRAM mode.
7. **Grow the lint test suite.** It had 3 tests, which is how "no rule ever
   runs inside a method" survived. Every rule kind wants at least one test that
   exercises it through a method call.
8. **Fix `Console`'s VRAM-mode writes**, or document that VRAM mode is
   force-blank-only. The 3 surviving C001 warnings are real: any program that
   calls `Console::print` in VRAM mode from its main loop corrupts VRAM, which
   is exactly what happened here.

---

# Second exercise: a Pac-Man clone (July 2026)

A second evaluation, this time compute-heavy where DERELICT was data-heavy:
**PACMAN** (in `pacman/`), a complete Pac-Man clone — 28x25 maze, four ghosts
with the arcade targeting personalities, scatter/chase scheduling, frightened
mode with the eaten-ghost/eyes/house-re-entry cycle, death animation, levels,
lives, hi-score, attract mode. All graphics generated by a Python tool into a
`.r65` data file. ~900 lines of R65 game code + ~550 lines of generated data;
15 KB of ROM used. Every mechanic above was verified frame-by-frame in
Mesen-GDB with a synthesized input movie (`pacman/play.sh`).

This exercise leaned on the parts a text adventure never touches: per-frame
sprite/OAM traffic, zeropage entity state, struct arrays, u8 wraparound
arithmetic, speed patterns, and a real-time main loop under NMI.

## What worked well

- **The 8-bit idioms are expressible and fast.** Grid movement as
  `(px & 7) == 4` alignment checks, two's-complement direction deltas in u8
  ROM tables, BCD score digits, per-frame speed-pattern tables — all of it
  wrote naturally and compiled tight. Zeropage "current entity" globals
  (`E_PX/E_PY/E_DIR`) as the mover's working set is exactly the hand-assembly
  pattern, and R65 made it a readable function instead of a routine.
- **Struct arrays** (`GHOSTS: [Ghost; 4]`, 8-byte struct) indexed in loops
  worked correctly and the power-of-2 size kept indexing cheap.
- **The generated-assets workflow** (Python emits `static` byte arrays +
  consts) is pleasant; `include!` + whole-program compilation means zero
  integration friction.
- **`mul8` for ghost-AI distance** and shift-only arithmetic everywhere else:
  the operator restrictions never blocked the design; they just made the cost
  model visible.

## Bugs found and fixed

1. **Compiler (codegen crash): loop register promotion vs widening casts.**
   `let mut p: u16 = pos as u16;` where `p` is then used as a loop counter got
   `p` promoted into X, but the widening `TypeConvert` writes its result with
   `STA`, which cannot target a hardware register — "Cannot resolve hardware
   register X as memory operand", with no source location. Fixed in
   `loop_register_promotion.py`: a vreg that is the *dest* of a
   TypeConvert/ToBool/Rotate is now ineligible for promotion (the source case
   was already handled). Regression test added in `test_loop_promotion.py`.

2. **stdlib doc bug: `set_bg1_map!` example teaches the wrong units.** The
   macro comment read "Usage: `set_bg1_map!(0x38, 0); // $7000 word addr`" —
   but the argument is in $400-word units, so 0x38 overflows VRAM and wraps
   the BG1 map base to word $6000. Following the example produced a screen of
   garbage (the sprite tile data rendered as a tilemap). Correct value for
   $7000 is 0x1C. Fixed the comment in `stdlib/sneslib.r65`. This one cost
   real debugging time precisely because *everything else was right* — VRAM,
   CGRAM, OAM and register writes all verified clean before the unit mismatch
   surfaced.

3. **Mesen-GDB: headless screenshots showed stale frames.** At unlimited
   emulation speed the SNES core renders at most one frame per 10 ms of
   wall-clock (`_skipRender`), so `--screenshot` at a breakpoint captured
   whichever frame last happened to render — for this ROM, deterministically
   an early-boot garbage frame, which masqueraded as a PPU bug for a long
   detour. Fixed: headless mode now sets `DisableFrameSkipping`, making
   breakpoint screenshots reflect the current PPU state.

## Friction notes

- The codegen error in bug 1 had **no source location**, and the offending
  function was only findable by monkeypatching the compiler to print the
  failing MIR instruction. Codegen errors should carry the enclosing function
  name at minimum.
- The scaffolded Makefile only tracked `src/main.r65`, so edits to included
  files didn't rebuild; the template should depend on `src/**/*.r65`.
- No `#[ram]`-array initializers were used (the known `__init_start` MVN/DBR
  bug), which keeps init code manual but was easy to live with.

## Verdict addendum

The real-time path holds up: a 5-sprite, 60 fps game with per-frame DMA and
a nontrivial AI fits comfortably in frame budget with straightforward code.
As with DERELICT, the language itself never blocked the design — the defects
were all in the surrounding surfaces (an optimizer edge case, a doc comment,
a debugging tool), and two of the three were the kind that silently burn
hours rather than fail loudly.
