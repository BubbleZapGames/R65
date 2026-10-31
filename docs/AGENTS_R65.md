# Writing R65: a primer for coding agents

Read this before writing R65 code. It covers the language model, a complete
working program, the Rust habits that fail, the standard library's real API,
and how to build and test a ROM. Full reference: <https://r65.dev/docs>.

Every `rust` block below is compiled by the R65 test suite
(`r65/tests/compiler/test_agents_primer.py`). If you copy it, it builds.

## 1. The mental model

R65 is typed 65816 assembly with Rust syntax. There is no runtime, no heap and
no hidden code. Every construct maps to instructions you could write by hand,
and costs what those instructions cost. The CPU registers `A`, `X`, `Y`, `B`,
`STATUS`, `D`, `DBR`, `S` are global variables. `A` is 8-bit unless a function
takes `@ A: u16`, and `X`/`Y` are always 16-bit. The compiler rejects work the
hardware cannot do cheaply (a multiply by 3, a variable shift) and makes you call
a library routine instead, so the cost stays visible.

Target: SNES (LoROM), assembled by WLA-DX. Source files end in `.r65`.

## 2. A complete program

A sprite that moves with the d-pad. Start from this shape: the `#[entry]` reset
handler, an NMI handler, and a main loop that syncs to VBlank.

```rust
#[snesrom(name="DEMO")]
#[stack(0x0100, 0x01FF)]

include!("stdlib/65816.r65")
include!("stdlib/sneslib.r65")
include!("stdlib/math.r65")

#[bank(0)]

// One 8x8 4bpp tile in colour 1: rows of planes 0/1 interleaved, then planes 2/3.
static SPRITE_TILE: [u8; 32] = [
    0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
];
static SPRITE_PAL: [u16; 2] = [0x0000, 0x7FFF];   // colour 0 transparent, colour 1 white

#[lowram]
static mut OAM: OamBuffer;      // 544-byte shadow, DMA'd every frame
#[zeropage]
static mut PX: u16;             // sprite X is 9 bits
#[zeropage]
static mut PY: u8;
#[zeropage]
static mut PAD: u16;

#[entry]
fn init_reset() -> ! {
    snes_init();                // forced blank; clears VRAM/CGRAM/OAM; NMI + auto-joypad on
    main();
}

#[interrupt(nmi)]
fn nmi_handler() {
    clear_nmi!()
}

fn main() -> ! {
    // The screen is still off, so bulk VRAM/CGRAM loads are safe.
    dma_copy_vram(0, &SPRITE_TILE as far *u8, 0x0000, 32);
    dma_copy_cgram(0, &SPRITE_PAL as far *u8, 128, 4);   // OBJ palette 0 = CGRAM 128; size in bytes
    set_obj_sel!(ObjSize::T8x8_16x16, 0, 0);
    set_main_layers!(Screen::Obj);
    OAM.clear();
    PX = 120;
    PY = 104;
    INIDISP = 0x0F;             // screen on, full brightness

    loop {
        wait_nmi!();
        OAM.dma_copy!(0);       // PPU traffic first, while still in VBlank
        wait_autojoy!();
        PAD = ((JOY1H as u16) << 8) | (JOY1L as u16);
        if PAD & JOY_LEFT != 0  { PX = PX - 1; }
        if PAD & JOY_RIGHT != 0 { PX = PX + 1; }
        if PAD & JOY_UP != 0    { PY = PY - 1; }
        if PAD & JOY_DOWN != 0  { PY = PY + 1; }
        OAM.set_full!(0, PX, PY, 0x3000);   // tile 0, priority 3 (attr in the high byte)
    }
}
```

Things this shows that are easy to get wrong:

- **`#[entry]` is the root of the program.** Dead-function elimination keeps only
  what `#[entry]`, interrupt handlers and static initialisers reach. A program
  with a plain `fn main` and no `#[entry]` compiles cleanly to an empty ROM.
- `#[entry]` and `#[interrupt]` functions must be in bank 0. A `#[bank(n)]` line
  applies to every declaration after it.
- `include!(...)` takes no trailing `;`. A file is included once per program.
- Macros end in `!`: `wait_nmi!()`, `clear_nmi!()`, `OAM.set_full!(...)`.
  `snes_init()` and `dma_copy_vram(...)` are functions.

## 3. Rust habits that fail

| You write (Rust habit) | R65 | What happens |
|---|---|---|
| `x * 3`, `x / 10` | `mul8(x, 3)`, `div8(x, 10)`, or `(x << 1) + x` | Compile error. `*` and `/` take only power-of-2 constants (1–256). |
| `x << n` with a variable `n` | `shl8(x, n)`, `shr8`, `shl16`, `shr16` | Compile error. Shift amounts must be constants. (The error hint says `shl()`, which does not exist.) |
| `x % 16` | `x & 15`, or `mod8(x, n)` | Compile error at runtime; `%` exists only in a `const fn` or between constants. |
| `v >> 1` on `i8`/`i16` | fine as written | Arithmetic shift (sign-filling, floors); signed `/` rounds toward zero. Each signed bit costs 4 cycles, not 2. |
| `let x = 0; x = 1;` | `let mut x` | Compile error. Locals and statics are immutable by default. |
| `fn f(p: Player)`, `a = b` on structs | `fn f(p: *Player)`; copy fields | Compile error. Structs, unions and arrays are never passed or assigned by value. |
| `fn f(i @ X: u8)` | `i @ X: u16` | Compile error. X and Y are always 16-bit. |
| `Option<T>`, `Result`, `?`, `panic!` | sentinel values or `-> u8, u8` (value, status) | Not in the language. |
| generics, closures, traits with generics, `mod`, `pub`, `#[derive]` | concrete types; one global namespace; write `impl` by hand | Not in the language. `impl Clone for T {}` is the one generated impl. |
| `Vec`, `String`, `&str` | fixed `[u8; N]` statics; `far *u8` to text | No heap. Strings are byte arrays; add `\0` yourself. |
| `(a, b)` tuple return | `fn f() -> u8, u16 { return a, b; }`, `let a, b = f();` | Multi-return in registers (A, then B or X, then Y). |
| `struct Pair(u8, u8)` | named fields | Only single-field tuple structs (newtypes) exist. |
| a `-> u8` fn with a path that falls off the end | return on every path | **Compiles.** It returns whatever is in A. |
| `let w: u16 = ...; let b: u8 = w;` | `w as u8` to show intent | **Compiles.** Narrowing truncates silently; widening zero/sign-extends. |
| `x++` inside an expression | `x++;` as a statement, or `x += 1` | `++`/`--` are postfix statements only. |

Integer conversions are implicit wherever a value flows into a place (`let`,
assignment, `return`, arguments). Pointers, enums and newtypes need `as`.

## 4. Choosing a design that is cheap

**Storage class** (only `static mut` takes one; `static` is ROM):

| Attribute | Range | Use for |
|---|---|---|
| `#[zeropage]` | `$0000-$00FF` | hot scalars, pointers you index through |
| `#[lowram]` | `$0000-$1FFF` | entity pools, OAM shadow, anything indexed per frame |
| `#[ram]` (also the default) | `$7E2000+` | big buffers; `&` of it is a **far** pointer |
| `#[hw(addr)]` | I/O | hardware registers (volatile); already declared in `sneslib.r65` |

Power-on RAM is garbage. Initialise everything you read before you write it.

**Strides and indexing.** `arr[i]` multiplies `i` by the element size. A power of
2 is a shift; anything else is a ~57-cycle multiply. Pad structs to 2/4/8/16
bytes, or use `#[soa]`.

**Entity pools.** Use an `#[soa]` struct (one array per field, no multiply) and
a tagged enum for kinds, not `*dyn` trait dispatch:

```rust
#[soa]
struct Enemy { x: u16, y: u8, hp: u8 }

impl Enemy {
    fn hurt(self) { self.hp = self.hp - 1; }   // self is an element index in X
}

#[lowram]
static mut ENEMIES: [Enemy; 16];

fn hit_all() {
    for i in 0..16 {
        if ENEMIES[i].hp != 0 {
            ENEMIES[i].hurt();
        }
    }
}
```

**Parameters.** Register parameters (`v @ A: u8`, `i @ X: u16`) are cheapest.
Plain parameters (`v: u8`) go on the stack and must come first. Return in `A` by
default. A `u16` parameter in `A` switches the whole function to 16-bit A (m16);
every other function runs with 8-bit A (m8).

```rust
#[zeropage]
static mut SCORE: u16;
#[zeropage]
static mut CURSOR_X: u16;

fn add_points(n @ A: u8) {
    SCORE = SCORE + (n as u16);
}

fn clamp_x(x @ X: u16) -> u16 {
    if x > 248 { return 248; }
    return x;
}

// A u16 in A would put the function in m16 mode, where A cannot return a u8.
// Take v on the stack instead, so it runs in m8 and returns lo in A, hi in B.
fn split(v: u16) -> u8, u8 {
    return (v & 0xFF) as u8, (v >> 8) as u8;
}

fn use_them() {
    add_points(10);
    let lo, hi = split(SCORE);
    CURSOR_X = clamp_x((lo as u16) + (hi as u16));
}
```

**Near and far.** Every `fn` is far by default (JSL/RTL) and callable from any
bank: free functions, methods, `extern fn` and `fn()` pointer types alike.
`near fn` opts out (JSR/RTS, its own bank only). `-O1`+ demotes far functions
whose callers share their bank, so you rarely need `near`. Write plain `fn`,
not `far fn`: `far fn` pins the function far and blocks that demotion. Pointers: `*T` is
16-bit (current data bank), `far *T` is 24-bit.

**Check the cost.** `r65c src/main.r65 -I src --cfg snes` prints the assembly to
stdout. Look at the function you just wrote. If you see `JSL mul8`, an `XBA` loop
or a stack frame you did not expect, restructure.

## 5. The standard library

The standard library is a package. A project from `r65x init` depends on it in
`r65.toml`, `make` fetches it (`r65pkg fetch`), and every file below is
included by package name: `include!("stdlib/string.r65")`. Nothing is copied
into the project.

| File | What's in it |
|---|---|
| `sneslib.r65` | Every PPU/CPU register as an `#[hw]` static (`INIDISP`, `BGMODE`, `VMADD`, `JOY1L`, ...). `snes_init()`, `clear_vram()`, `clear_oam()`, `delay(frames)`, `is_pal()`. Sync: `wait_nmi!()`, `wait_autojoy!()`, `clear_nmi!()`, `wait_vblank_sync!(FRAME, LAST)`. Joypad masks `JOY_A`, `JOY_B`, `JOY_UP`, ... (u16, compose `JOY1H:JOY1L`). DMA: `dma_copy_vram(ch, src, vram_word, bytes)`, `dma_copy_cgram(ch, src, colour, bytes)`, `dma_copy_oam(ch, src, bytes)` (functions, run-time far pointer), and `dma_copy_vram_const!`/`dma_copy_cgram_const!`/`dma_fill_vram!`/`dma_set_hdma!` (macros, compile-time labels). PPU setup macros: `set_bgmode!`, `set_bg1_map!`..`set_bg4_map!`, `set_bg12_chr!`, `set_bg1_hscroll!`.., `set_main_layers!`, `set_obj_sel!`, `set_brightness!`, `set_screen_off!`, `set_fixed_color!`, Mode 7 and window macros. `OamBuffer` (`.set_full!`, `.set_size!`, `.hide!`, `.dma_copy!`, `.clear()`). `Color` newtype (`Color::rgb!(r, g, b)`). Enums `Screen`, `ObjSize`, `BG`, `Interrupt`, `VmainTrigger`. |
| `math.r65` | `mul8(a: u8, b: u8) -> u16`, `mul16(a: u16, b: u16) -> u16`, `div8(a: u8, d: u16) -> u8`, `div16(a: u16, d: u8) -> u16`, `mod8(a: u8, d: u16) -> u8`, `mod16(a: u16, d: u8) -> u16`, `shl8`/`shr8`/`shl16`/`shr16(value, amount)`, `shri8` (arithmetic shift right). `--cfg snes` uses the hardware multiplier/divider. Decimal digits without `string.r65`: `q = div16(v, 10)`, digit `= v - ((q << 3) + (q << 1))`. |
| `65816.r65` | `block_move!` (MVN), `disable_irq!`/`enable_irq!`, `without_irq!`, `stack_guard_check!`. |
| `string.r65` | `strlen`, `memcpy`, `memset`, `memcmp`, `strcpy`, `strcmp`, `u8_to_dec`, `u16_to_dec`, `u8_to_hex`, `format!(buf, "fmt", args)`. Pointers are `far *u8`. |
| `console.r65` | `Console` text output to a BG tilemap buffer, VRAM or OAM: `init`, `set_area`, `set_cursor`, `print`, `println`, `print!("fmt", ...)`, `clear`, `upload_font`. Pulls in `string.r65`. |
| `rand.r65` | `rand() -> u16`, `rand_update!(entropy)`. |
| `U32.r65`, `I32.r65` | 32-bit integers as structs with `+=`, `-=`, `*=`, `/=`, compare, `to_string`. |
| `Q8.r65`, `Q10.r65` | Fixed point: `Q8` unsigned 8.8, `Q10` signed 10.6, as newtypes over `u16`/`i16`. |
| `F32.r65` | Software 32-bit float. Slow; prefer `Q10`. |
| `msu1.r65` | MSU-1 audio streaming. |

Binary assets go in ROM with `include_bytes!`:
`static TILES: [u8; 2048] = include_bytes!("../assets/tiles.chr");`. `r65x
bmp2chr`, `r65x fontgen`, `r65x brr` and `r65x packer` convert graphics, fonts,
audio and compressed data.

## 6. Build and test

A project from `r65x init my_game` has a Makefile. `init` asks
for the ABI model (`--abi Default|FixedStack` skips the prompt); the Makefile
records it as `ABI =` and passes `--abi $(ABI)` to r65c. A FixedStack project
also includes `stdlib/scratch_regs.r65`, the direct-page scratch slots its
parameters live in, from `main.r65`.

```bash
make            # src/main.r65 -> build/main.asm -> ROM (r65c -v --cfg snes --dbg -I src)
make run        # build and launch the emulator
make tests      # run tests/*.script with mesen-gdb (skipped if not installed)
make syntax     # compile to assembly only
make clean
```

- **Lint.** `r65-lint.toml` (project root) configures `r65c --lint`. Its `C001`
  rule flags PPU registers written where they are unsafe during rendering.
- **Debug symbols.** `--dbg` writes a `.dbg` file for source-level debugging in Mesen.
- **Test scripts.** `tests/*.script` are mesen-gdb command files with `assert`
  checks (`until main 60`, `input 1 right`, `frames 10`, `assert PX:u16 == 130`).
  `make tests` runs each one headless.
- **Automated ROM tests.** [Mesen-GDB](https://github.com/BubbleZapGames/Mesen-GDB)
  runs a ROM headless with assertions:
  `mesen-gdb build/GAME.smc --batch --headless --break main --check-mem 0x10=0xFF --timeout 5000`
  (exit 0 pass, 1 fail, 2 timeout). Its Python package `mesen_gdb.rom_driver`
  drives the pad and reads memory from pytest:
  `s.wait(30).right(20); assert s.u16(0x0000) == 140`.
- `#[zeropage]`/`#[lowram]` statics are `.DEFINE`s, not symbols. Read them by the
  address in the `.DEFINE` line of `build/main.asm`. `#[ram]` statics are in the `.sym` file.
  A static the program never touches is removed; mark one `#[used]` (or give it an
  address) to keep it for a test to read.
- Trust VRAM/OAM/memory dumps over screenshots when checking what is on screen.
  A loop that DMAs OAM before updating it (as in section 2) leaves PPU OAM one
  frame behind the RAM shadow.
- `asm!("BRK")` is a breakpoint; `#[used]` keeps a function that nothing calls
  (for a debugger `jump`), or a static nothing refers to.

## 7. SNES and R65 pitfalls

- **`OAM.set_full!(idx, x: u16, y: u8, chr: u16)`**: `chr` is `tile | (attr << 8)`.
  The attribute byte is `vhppcccN` (flip, priority, palette, name-table bit), so
  `0x3000 | tile` is priority 3, palette 0. The tile number counts in 16-word
  (4bpp) steps from the OBJ base set by `set_obj_sel!`: with base 0, tile 1 sits
  at VRAM word `0x0010`.
- **`OamBuffer.set!` writes an 8-bit X and leaves the X-high bit alone.** A sprite
  that crossed x=256 stays 256px off-screen. Use `set_full!` (u16 X) unless the
  sprite can never leave the 0–255 range.
- **A register alias does not survive other work.** `let lo @ A = JOY1L; let hi =
  JOY1H;` is rejected, because loading `JOY1H` overwrites A. Compose through plain
  values instead:

```rust
// error: 'lo' lives in A, and reading JOY1H overwrites A
#[zeropage]
static mut PAD_BAD: u16;

fn read_pad_bad() {
    let lo @ A = JOY1L;
    let hi = JOY1H;
    PAD_BAD = ((hi as u16) << 8) | (lo as u16);
}
```

  The working form is the one in section 2: `((JOY1H as u16) << 8) | (JOY1L as u16)`.
- **`set_bgN_map!` takes the tilemap address in $400-word units.** Word address
  `$7000` is `set_bg1_map!(0x1C, 0)`, not `0x7000`. `set_bg12_chr!` uses 4K-word units.
- **VRAM, CGRAM and OAM are writable only in VBlank or forced blank.** Do bulk
  loads before `INIDISP = 0x0F`. In the loop, do DMA right after `wait_nmi!()`,
  before game logic. Queue tile updates and drain them there.
- **`dma_copy_cgram` takes a size in bytes** (colours × 2). Sprite palettes start
  at CGRAM colour 128.
- **The PPU expands 5-bit colour to 8-bit by bit replication**, `(v << 3) | (v >> 2)`.
  Use this when matching screenshot pixels against palette colours.
- **Read the joypad after `wait_autojoy!()`.** Auto-read finishes a few scanlines
  into VBlank; `JOY1L`/`JOY1H` are stale before that.
- **Counters wrap.** `PY - 1` at 0 is 255, and the u16 `PX` in section 2 wraps
  to 65535 and leaves the screen. Bounds-check before you step.
- **No bounds checking.** An out-of-range index reads or writes whatever is there.
- **`asm!` is a black box.** The compiler assumes it clobbers all registers and
  does not track mode changes inside it.

## 8. Where to read more

Reference: <https://r65.dev/docs>

| Question | Page |
|---|---|
| Types, casts, newtypes, pointers | `language/types`, `language/new_types` |
| Operators, costs, the math functions | `language/expressions`, `platform/math` |
| Structs, `#[soa]`, unions | `language/structs`, `language/unions` |
| Enums, tagged enums, `match`, `if let` | `language/enums`, `language/match-expressions`, `language/control-flow` |
| Functions, register params, multi-return, near/far | `language/functions`, `platform/abi-models` |
| Every attribute | `language/attributes` |
| Macros (`macro_rules!`) | `language/macros` |
| Traits, operator overloading | `language/traits`, `language/operator-overloading` |
| Registers, `STATUS`, `B` | `hardware/registers`, `hardware/processor-modes` |
| Memory map, storage classes | `hardware/memory-model` |
| Interrupt handlers | `hardware/interrupts` |
| Stdlib built-ins, `asm!` | `platform/builtin_library`, `platform/assembly-code` |
| `r65c` flags, `r65x` tools, ROM header | `platform/compiler`, `platform/tools`, `platform/snes-rom-header` |

Prefix each page with `https://r65.dev/docs/`. In the R65 repo, `docs/*.md`
holds the design documents behind the language.
