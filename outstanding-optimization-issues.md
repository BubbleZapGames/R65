# Outstanding codegen optimization issues

Two items from Myself086's 2026-08-20 review of the disassembled
`classickong.r65` ROM are still open. Both are real, both are confirmed
against generated output — but neither is a peephole rewrite. Each needs
a decision about *where* the fix belongs before it can be written.

The other four items from that review are fixed; see the peephole passes
`_eliminate_dead_mode_changes` and `_fold_carry_setup_into_rep`, and the
interrupt prologue in `codegen/function_gen.py`.

---

## 1. `>> 8` on a u16 does not narrow to a high-byte load

**Occurrences:** 65 `>> 8` / `>>= 8` sites across `stdlib/` and
`classickong.r65/src/`.

### What we emit

For `HI = (W >> 8) as u8;` where `W: u16` lives at `$20`:

```asm
    REP #$20  ; 16-bit A
    LDA $20
    STA $01,S       ; spill W
    XBA             ; Swap bytes (shift right 8)
    AND #$FF        ; Clear high byte
    STA $03,S       ; spill the shifted u16 temp
    SEP #$20  ; 8-bit A
    STA $05,S       ; spill the truncated u8 temp
    STA $22         ; HI = ...
```

### What it should be

```asm
    LDA $21         ; the high byte, directly
    STA $22
```

Two instructions instead of nine, and no mode switch at all when the
surrounding block is already m8.

### Why a peephole won't do it

Two independent causes, and neither is visible at the asm level:

1. **The shift can't see its source.** `_emit_shift_right`
   (`codegen/instruction_select.py:1113`) is handed only the shift
   count — it operates on whatever is in A and has no idea the value
   came from a known address, so it cannot rewrite the load into
   `LDA addr+1`. Recognising `Load(mem16) → Shr 8 → Trunc u8` has to
   happen where all three nodes are visible, i.e. in MIR.

2. **The `as u8` materializes a temp.** The 16-bit shifted value is
   stored to a stack slot, then the low byte is stored to a *second*
   slot, then to the destination. That is a separate lowering problem
   from the shift and would still cost several instructions even if the
   load were narrowed.

An asm-level rewrite of `LDA <mem> / XBA / AND #$00FF` → `LDA <mem+1>`
also has to prove B (the accumulator's hidden high byte) is dead across
the rewrite, because `AND #$00FF` zeroes B and a narrowed load leaves it
untouched. There is no B-liveness analysis at the peephole layer today.

### Decision needed

Fix in MIR lowering (pattern-match the shift-then-truncate chain and
lower it to a byte load at `addr+1`), or add B-liveness to the peephole
layer and do it there. MIR is the better home — it also gets at the temp
spilling, which the peephole cannot touch.

---

## 2. Hardware multiply writes `$4202`/`$4203` a byte at a time

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
