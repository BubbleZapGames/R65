# Outstanding codegen optimization issues

From Myself086's 2026-08-20 review of the disassembled `classickong.r65`
ROM. Of the six items raised, five are fixed; what follows is the one
that is still open, plus a smaller sub-issue found alongside it.

The `>> 8` narrowing item that used to head this file **has landed**, in two
halves. When the word is in memory, `(mem16 >> 8) as u8` folds to a load of the
high byte (`r65/compiler/optimize/byte_extract.py`). When it is already in a
register, the fix turned out to be elsewhere entirely: `TypeConvert` was never
registered as an A-coalescence def type in `codegen/slot_allocator.py`, so *every*
narrowing cast spilled twice and allocated a frame. Registering it removed the
spills, and with them gone the existing `AND`-before-`SEP` peephole could finally
drop the mask. Entries are deleted rather than struck through; git history has
the old text.

---

## Hardware multiply writes `$4202`/`$4203` a byte at a time

**Occurrences:** `stdlib/math.r65` `mul8` and `mul16`, plus the same
pattern hand-written in `stdlib/Q8.r65`, `stdlib/Q10.r65`,
`stdlib/F32.r65`. One `mul8` call site and nine `mul16` sites in
classickong.

### What we emit

```asm
mul8:
    STA $4202
    XBA             ; Access B value in A
    STA $4203
    XBA             ; Restore A register   <- dead
    NOP
    NOP
    NOP
    NOP
    REP #$20  ; 16-bit A
    LDA $4216
    RTL
```

30 cycles.

### What it should be

`WRMPYA` and `WRMPYB` are adjacent, so a single 16-bit `STA $4202`
writes both. And `mul8`'s ABI is
`fn mul8(multA @ A: u8, multB @ B: u8)` — the arguments arrive in A and
B, which means the 16-bit accumulator **already holds exactly the word
we want to store**. No shuffling is required at all:

```asm
mul8:
    REP #$20
    STA $4202       ; writes multA -> $4202, multB -> $4203
    NOP
    NOP
    NOP
    NOP
    LDA $4216
    RTL
```

21 cycles conservatively (keeping the full 8-cycle wait), and the two
`XBA`s and one store disappear. Myself086 puts the achievable floor
lower still — the read only needs ~8 cycles after the `$4203` write, and
`LDA $4216` spends 3 cycles on fetch before its first read cycle, so the
padding can likely shrink. That part should be validated in Mesen, not
in `r65/emulator/` — the emulator computes the product immediately on
the `$4203` write, so it cannot catch a too-short delay.

### Why the stdlib can't just be rewritten

R65 has no way to name the full 16-bit accumulator while in m8 mode.
`A` is `u8` and `B` is its high byte; `A as u16` zero-extends (killing
B), which is the opposite of what's wanted. So `WRMPY = <the 16-bit A>`
is currently inexpressible, even with a `#[hw(0x4202)] static mut
WRMPY: u16;` declaration in `sneslib.r65`.

### Decision needed

Pick one:

- **Intrinsic.** Make `mul8`/`mul16` compiler builtins that emit the
  tuned sequence directly. Keeps the language surface unchanged; moves
  hand-tuned assembly into the compiler.
- **Language-level 16-bit-A spelling.** Some way to refer to `B:A` as a
  single u16 value, which would let the stdlib express this (and other
  B-register tricks) in R65 source.
- **`asm!()` in the stdlib.** Cheapest, but it hides the sequence from
  the optimizer and from `#[cfg]`-driven retargeting.

### Separate, smaller sub-issue

The second `XBA ; Restore A register` is dead: the following
`LDA $4216` executes in m16 and overwrites both halves of the
accumulator. A general peephole rule — *an `XBA` is dead if the next
definition of A is 16-bit wide and nothing in between reads A or B* — is
sound and uses the existing `compute_modes` oracle to establish the
width. classickong emits 118 `XBA`s; how many the rule would catch has
not been measured.
