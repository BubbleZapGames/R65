# R65 in 240 Seconds — Video Script

A fast, dense walkthrough of R65 for game developers and general-audience
programmers — modeled on the pacing and voice of "X in 100 Seconds": cold-open
definition, breathless clause-chaining, dry wit, second person, present tense.
Eleven sections, ~795 words of voiceover at ~3.3 words/sec — lands at 4:00.

---

## 1. Hook — 0:00–0:20
**On screen:** Raw 65816 `.asm` scrolling — `SEP #$20`, `REP #$20`, `PHB`, bank labels, branch math. Red tint. Title **R65** punches in over the last line.

**VO:** "R65 — a Rust-inspired language that targets the 65816 processor for the Super Nintendo: modern syntax, zero abstraction cost. Coding the SNES by hand means juggling 8- and 16-bit CPU modes, swapping memory banks, counting cycles, and fixing branch distances by hand — the stuff every modern language dropped decades ago. R65 gives you all that control back — and lets the compiler do the dirty work."

## 2. First program — 0:20–0:44
**On screen:** A pillars card fades in (one line at a time):
- Hardware-transparent programming for the SNES & 65816
- Type-safe variable management
- Reverse engineering
- Efficient assembly matching hand-written techniques
- Modern tooling for SNES development flows

Then the entry point types out, then a terminal: `r65x init --platform snes my_game` → `make` → a `.smc` icon.
```rust
#[entry]
fn main() -> ! {
    INIDISP = 0x0F;        // screen on, full brightness
    loop {
        wait_vblank();
        update_game();
    }
}
```
**VO:** "And this is a whole program — the entire thing, on one screen. `#[entry]` marks main; that arrow-bang means it never returns, so the compiler doesn't even emit a return. That's the whole pitch — the control of assembly, with the safety and readability of a modern language. Scaffold it with `r65x init`, hit `make`, and you've got a real SNES ROM that boots on any emulator."

## 3. Tight asm / zero cost — 0:44–1:06
**On screen:** Split view. Left: `if HVBJOY & 0x80 != 0 { break; }`. Right: `LDA $4212 / AND #$80 / BNE`. Then a counter graphic: "~60,000 cycles / frame".

**VO:** "This compiles to assembly as tight as hand-written. Take this line — wait until the vblank flag is set. On the right is what R65 generates: load the register, mask the bit, branch. No runtime, no garbage collector, no hidden calls. And that matters, because the SNES gives you about sixty thousand CPU cycles a frame — every one counts. R65's whole philosophy: high-level code costs exactly what the equivalent assembly costs, and nothing more."

## 4. Hardware is first-class — 1:06–1:38
**On screen:** Cuts:
```rust
let health @ A = PLAYER.health;     // A *is* health
#[zeropage] static mut FRAME: u8;   // fast 256 bytes
#[ram]      static mut MAP: [u8; 2048];
#[hw(0x4212)] static mut HVBJOY: u8;  // auto-volatile
```
Small SNES memory-map diagram highlighting zero-page / RAM / hardware.

**VO:** "The hardware isn't hidden behind an abstraction — it *is* the language. Every CPU register is a global: A, X, Y, the status flags, the stack pointer. Bind one to a readable name at zero cost — here, `health` literally *is* the accumulator. Memory's just as explicit: you pick where every variable lives. Zero-page is the fast 256 bytes for hot loop counters; ram for big buffers; and the hardware attribute maps a variable straight onto an I/O register — automatically volatile, so every read and write hits the metal. Immutable data with no attribute just lands in ROM. Your memory map is part of the type signature."

## 5. Functions & the ABI — 1:38–2:07
**On screen:**
```rust
fn fill(buffer @ X: *u8, value @ A: u8, count @ Y: u16) { ... }

fn divide(n @ A: u8, d: u8) -> u8, u8 { return A, B; }
let quotient, remainder = divide(100, 7);

#[preserves(X, Y)]
fn safe(input @ A: u8) -> u8 { ... }
```
**VO:** "Functions give you three ways to pass arguments, ranked by speed: stack parameters like any language, variable-bound parameters, or the fast path — straight through CPU registers. This `fill` takes its pointer in X, value in A, count in Y — zero stack overhead. You can return multiple values in multiple registers — quotient in A, remainder in B — and destructure them like a tuple. And `#[preserves]` auto-saves and restores registers around a call. This is also why R65 is great for reverse engineering: you can match almost any calling convention from a disassembled ROM, function by function."

## 6. The compiler does the miserable parts — 2:07–2:17
**On screen:** Checkmarks fire in sequence: "Auto 8/16-bit mode ✓", "Bank + far calls ✓", "Branch fixup ✓", "10+ optimization passes ✓". Then a red `error: bank overflow` in the editor.

**VO:** "And R65 handles the miserable parts: it auto-manages 8- and 16-bit mode switching, cross-bank calls, and branch fixups, runs a dozen optimization passes, and catches mode and bank mistakes at compile time."

## 7. The modern stuff — 2:17–2:47
**On screen:** Rapid but readable:
```rust
match tile_id {
    0..=15 => 1,
    16 | 32 => 2,
    _ => 0,
}
const PLAYER_TILE: u16 = tile_offset(5, 3);  // folds to a literal
#[interrupt(nmi)] fn vblank() { FRAME++; }
```
Keyword glow: `struct enum trait const fn macro_rules! format!`.

**VO:** "You still get the modern conveniences. Structs and enums, packed with no padding. Pattern matching with range and or-patterns, where the compiler picks the best strategy — jump table, lookup table, or branch chain. `const fn` runs real computation at compile time, so a tile offset becomes a literal baked into your ROM. Traits give you dynamic dispatch when you want polymorphism — a list of objects that each draw themselves. There's a `macro_rules!` system, and a `format!` macro for printf-style text. And interrupt handlers are just functions — tag one `#[interrupt(nmi)]` and R65 writes all the save, restore, and return-from-interrupt boilerplate."

## 8. The standard library — 2:47–3:22
**On screen:** Three quick beats.

*math* — a `--cfg snes` toggle flips a badge from "software" to "hardware multiplier":
```rust
let damage: u16 = mul8(base, multiplier);   // 8×8 → 16
let speed:  u16 = div16(distance, time);
```
*format!* — the rendered result `Score: 00420  HP: 12` appears under it:
```rust
format!(buf, "Score: {u16:05d}  HP: {u8}", score, hp);
```
*Console* — cut to text landing on a SNES screen:
```rust
#[ram] static mut HUD: Console;
HUD.init(0);
HUD.set_area(1, 1, 20, 4);
HUD.print!("SCORE {u16:06d}", score);
```
**VO:** "And all of this ships with a standard library — written in R65 itself, so it's all readable. The chip has no general multiply or divide, so the stdlib gives you `mul8`, `mul16`, `div16`, `mod8`, plus variable shifts. Compile with `--cfg snes` and the exact same calls use the console's hardware multiply and divide units — same code, faster math. For text there's `format!` — printf-style formatting into a byte buffer, with decimal, hex, zero-padding, even chars and strings. And the Console ties it together: a struct with methods that draws text to background tiles, VRAM, or sprites, handling the cursor, alignment, and text areas for you. Call `print!` on it, and your formatted string lands on screen."

## 9. Debugging in Mesen — 3:22–3:42
**On screen:** `r65c game.r65 --dbg` in terminal. Cut to Mesen: stepping through the **original `.r65` source** with a breakpoint on a function name, CPU register + memory panels updating live.

**VO:** "And when something breaks — because it will — you're not staring at raw assembly. Compile with `--dbg` and R65 emits source-level debug symbols. Open the ROM in Mesen and step through your original R65 source, line by line. Set a breakpoint on a function name. Watch the registers and memory update live as it runs. You debug the language you wrote — not the assembly it compiled to."

## 10. It's real: Classic Kong — 3:42–3:53
**On screen:** **Classic Kong** gameplay on a CRT / Mesen. GitHub repo page flashes by.

**VO:** "And this isn't a toy. This is Classic Kong — a complete, playable game, written entirely in R65 and running as a real Super Nintendo ROM. The whole thing's open source, so you can read every line."

## 11. Close — 3:53–4:00
**On screen:** Logo, URL, "MIT · Alpha". Tagline.

**VO:** "That's R65 — open source, MIT licensed, and usable today. Links are below. Go write some modern code, and ship a retro game."

---

## Clean voiceover read-through (~795 words ≈ 4:00 at ~3.3 words/sec)

> R65 — a Rust-inspired language that targets the 65816 processor for the Super Nintendo: modern syntax, zero abstraction cost. Coding the SNES by hand means juggling 8- and 16-bit CPU modes, swapping memory banks, counting cycles, and fixing branch distances by hand — the stuff every modern language dropped decades ago. R65 gives you all that control back — and lets the compiler do the dirty work.
>
> And this is a whole program — the entire thing, on one screen. `#[entry]` marks main; that arrow-bang means it never returns, so the compiler doesn't even emit a return. That's the whole pitch — the control of assembly, with the safety and readability of a modern language. Scaffold it with `r65x init`, hit `make`, and you've got a real SNES ROM that boots on any emulator.
>
> This compiles to assembly as tight as hand-written. Take this line — wait until the vblank flag is set. On the right is what R65 generates: load the register, mask the bit, branch. No runtime, no garbage collector, no hidden calls. And that matters, because the SNES gives you about sixty thousand CPU cycles a frame — every one counts. R65's whole philosophy: high-level code costs exactly what the equivalent assembly costs, and nothing more.
>
> The hardware isn't hidden behind an abstraction — it *is* the language. Every CPU register is a global: A, X, Y, the status flags, the stack pointer. Bind one to a readable name at zero cost — here, `health` literally *is* the accumulator. Memory's just as explicit: you pick where every variable lives. Zero-page is the fast 256 bytes for hot loop counters; ram for big buffers; and the hardware attribute maps a variable straight onto an I/O register — automatically volatile, so every read and write hits the metal. Immutable data with no attribute just lands in ROM. Your memory map is part of the type signature.
>
> Functions give you three ways to pass arguments, ranked by speed: stack parameters like any language, variable-bound parameters, or the fast path — straight through CPU registers. This `fill` takes its pointer in X, value in A, count in Y — zero stack overhead. You can return multiple values in multiple registers — quotient in A, remainder in B — and destructure them like a tuple. And `#[preserves]` auto-saves and restores registers around a call. This is also why R65 is great for reverse engineering: you can match almost any calling convention from a disassembled ROM, function by function.
>
> And R65 handles the miserable parts: it auto-manages 8- and 16-bit mode switching, cross-bank calls, and branch fixups, runs a dozen optimization passes, and catches mode and bank mistakes at compile time.
>
> You still get the modern conveniences. Structs and enums, packed with no padding. Pattern matching with range and or-patterns, where the compiler picks the best strategy — jump table, lookup table, or branch chain. `const fn` runs real computation at compile time, so a tile offset becomes a literal baked into your ROM. Traits give you dynamic dispatch when you want polymorphism — a list of objects that each draw themselves. There's a `macro_rules!` system, and a `format!` macro for printf-style text. And interrupt handlers are just functions — tag one `#[interrupt(nmi)]` and R65 writes all the save, restore, and return-from-interrupt boilerplate.
>
> And all of this ships with a standard library — written in R65 itself, so it's all readable. The chip has no general multiply or divide, so the stdlib gives you `mul8`, `mul16`, `div16`, `mod8`, plus variable shifts. Compile with `--cfg snes` and the exact same calls use the console's hardware multiply and divide units — same code, faster math. For text there's `format!` — printf-style formatting into a byte buffer, with decimal, hex, zero-padding, even chars and strings. And the Console ties it together: a struct with methods that draws text to background tiles, VRAM, or sprites, handling the cursor, alignment, and text areas for you. Call `print!` on it, and your formatted string lands on screen.
>
> And when something breaks — because it will — you're not staring at raw assembly. Compile with `--dbg` and R65 emits source-level debug symbols. Open the ROM in Mesen and step through your original R65 source, line by line. Set a breakpoint on a function name. Watch the registers and memory update live as it runs. You debug the language you wrote — not the assembly it compiled to.
>
> And this isn't a toy. This is Classic Kong — a complete, playable game, written entirely in R65 and running as a real Super Nintendo ROM. The whole thing's open source, so you can read every line.
>
> That's R65 — open source, MIT licensed, and usable today. Links are below. Go write some modern code, and ship a retro game.
