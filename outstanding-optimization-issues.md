# Outstanding codegen optimization issues

From Myself086's 2026-08-20 review of the disassembled `classickong.r65` ROM.
All six items he raised are now fixed. What remains below is one smaller
sub-issue found alongside them, plus a follow-up that measurement turned up.

The `>> 8` narrowing item landed in two halves. When the word is in memory,
`(mem16 >> 8) as u8` folds to a load of the high byte
(`r65/compiler/optimize/byte_extract.py`). When it is already in a register the
fix was elsewhere entirely: `TypeConvert` was never registered as an
A-coalescence def type in `codegen/slot_allocator.py`, so *every* narrowing cast
spilled twice and allocated a frame. Registering it removed the spills, and with
them gone the existing `AND`-before-`SEP` peephole could drop the mask.

The hardware-multiply item landed in `stdlib/math.r65`: `mul8` now writes both
`$4202` and `$4203` with a single 16-bit `STA` via inline asm, since R65 cannot
name the full accumulator from m8 code. 30 cycles to 21. Verified on Mesen
rather than the bundled emulator, which computes the product on the `$4203`
write and so cannot see a too-short delay -- confirmed sensitive by checking
that a no-delay build fails there.

**Note for whoever picks these up:** six projects (classickong.r65, pacman,
bubblebobble, kungfu.r65, astroids, textadv) vendor their own `src/lib/math.r65`
rather than including `stdlib/`. None of them pick up the `mul8` change until
their copy is synced.

Entries are deleted rather than struck through; git history has the old text.

---

## Dead `XBA` before a 16-bit redefinition of A

An `XBA` is dead when the next definition of A is 16 bits wide and nothing in
between reads A or B -- the 16-bit write lands on both halves, so the swap that
preceded it cannot be observed.

**Measured** (the earlier note said this had not been): **12 sites** in
classickong, 1 in pacman. 36 cycles and 12 bytes.

The catch is that it needs `compute_modes` **seeded** with the label-anchored
`.ACCU` entry modes. Unseeded -- which is how every peephole pass calls it today
-- a function body reachable only by JSL is bottom, so the pass can never prove
the 16-bit width and the rule finds nothing. Measuring this unseeded first
reported 0 sites, which is what the number in the earlier draft would have been.

So this is really two pieces of work: teach the peephole passes to seed entry
modes, then add the rule. The seeding is the larger and riskier half -- it would
also make several existing passes fire inside function bodies where they
currently cannot, which is a win but a broad change.

---

## Strength-reduce a multiply by a constant

Found while measuring the multiply item. classickong's only hot `mul8` call is
`mul8(i, 3)` inside `game_show_kong`, a per-frame draw routine, four times per
call:

```asm
LDA #$03 / XBA / LDA $03,S / JSL mul8 / STA $09,S      ~57 cycles
```

`i * 3` is `(i << 1) + i` -- roughly 16 cycles inline, no call at all. R65's `*`
accepts only power-of-two constants, which is exactly why the author reached for
`mul8` here.

**~41 cycles per call against the ~9 the `$4202` fold saved**, so this is the
larger win at the same call site. Either fold `mul8`/`mul16` with a literal
operand into shifts and adds, or relax `*` to accept small non-power-of-two
constants.

---

## Hoisting and store-zero, seen through `clear_vram`

Two general codegen misses, found trying to widen `clear_vram`'s inner loop.
`sneslib.r65` writes `VMDATAL = 0; VMDATAH = 0;` 32768 times; `VMDATA` is
declared as a `u16` at `$2118`, so `VMDATA = 0` ought to halve the stores.
It is **slower** — ~24 cycles an iteration against ~19:

```asm
clear_vram_wide__L1:
    SEP #$20        ; 3   <- not hoisted out of the loop
    CPY #$8000
    BCS ...
clear_vram_wide__L2:
    REP #$20        ; 3   <- not hoisted out of the loop
    LDA #$00        ; 3   <- STZ needs no accumulator
    STZ $2118       ; 5
    INY
    BRA clear_vram_wide__L1
```

1. **A spurious load before a store-zero.** `STZ` writes zero without touching A,
   so the `LDA #$00` is three wasted cycles.
2. **The loop's mode switches are not hoisted.** Nothing else in that body
   depends on the M flag — `CPY`/`INY` are index ops — so the `REP`/`SEP` pair
   belongs outside the loop. `_hoist_loop_mode_switches` exists and does not
   fire here.

Fixing either makes the wide store a win and takes roughly 100k cycles off a
VRAM clear. Until then, leave `clear_vram` writing bytes.

Checked and rejected while looking for more of these, so the analysis is not
repeated: `div8`'s `WRDIVL`/`WRDIVH` pair is already optimal (`STA` + `STZ`);
`clear_oam` and `init_ppu` write their address pairs once; the `WRMPYA`/`WRMPYB`
*writes* in Q8/Q10/F32/mul16 take their operands from locals rather than from A
and B, so `mul8`'s single-store trick does not transfer.
