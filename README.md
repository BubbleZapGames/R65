# R65 - Modern Language for Retro Hardware

R65 is a Rust-inspired programming language for the 6502/65816 processor family, designed for SNES development. It gives you the control of assembly with the safety and readability of a modern language.

Website and documentation: [r65.dev](https://r65.dev/)

```rust
#[entry]
fn main() -> ! {
    INIDISP = 0x0F;  // Screen on, full brightness
    loop {
        wait_nmi!();
        update_game();
    }
}
```

## Why R65?

- **Write readable code** that compiles to efficient, hand-written-quality assembly
- **Catch bugs at compile time** - mode mismatches, bank overflow, type errors
- **Direct hardware access** - registers, memory-mapped I/O, and processor modes are first-class
- **Match existing patterns** - replicate any calling convention from disassembled ROMs

## Quick Start

### Installation

Requires Python 3.10+ (dependencies: `lark`, plus `tomli` on Python 3.10).

```bash
git clone https://github.com/BubbleZapGames/R65.git
cd R65
pip install -e .
```

### Install WLA-DX

WLA-DX is the target assembler/linker. Install via Homebrew or download binaries:

```bash
brew install wla-dx
```

Or download pre-built binaries from https://github.com/vhelin/wla-dx/releases and place `wla-65816` and `wlalink` on your PATH.

### Create a Project and Build

```bash
r65x init --platform snes my_game
cd my_game
make                          # Compile, assemble and link build/MY_GAME.smc
make run                      # Open it in an emulator (Mesen, higan or bsnes)
make tests                    # Run tests/*.script with Mesen-GDB, if installed
```

The generated `src/main.r65` already boots the console and runs an empty game
loop; start there. ROM tests use [Mesen-GDB](https://github.com/BubbleZapGames/Mesen-GDB),
a scriptable SNES debugger.

### Using an AI Coding Agent

Point the agent at [docs/AGENTS_R65.md](docs/AGENTS_R65.md) before it writes
R65. It is a short primer: a complete program, the Rust habits that don't
carry over, the standard library's API, and how to build and test a ROM.

---

# R65 Language Guide

## Basic Types

R65 has five basic types that map directly to hardware:

| Type | Size | Range |
|------|------|-------|
| `u8` | 1 byte | 0 to 255 |
| `i8` | 1 byte | -128 to 127 |
| `u16` | 2 bytes | 0 to 65535 |
| `i16` | 2 bytes | -32768 to 32767 |
| `bool` | 1 byte | `true` or `false` |

```rust
let lives: u8 = 3;
let score: u16 = 0;
let game_over: bool = false;
```

## Variables and Memory

### RAM Variables

A `static mut` with no attribute is auto-allocated in main RAM; `#[ram]` is the explicit spelling. Use `#[lowram]` for the first 8K (`$0000-$1FFF`), which is faster and reachable with near pointers:

```rust
static mut SCORE: u16 = 0;       // auto-allocated RAM (same as #[ram])

#[ram]
static mut HIGH_SCORE: u16 = 0;

#[lowram]
static mut ENEMIES: [Enemy; 8];  // Array of 8 enemies
```

### Zero-Page Variables (Fast!)

Zero-page (`$00-$FF`) is the fastest memory on the 65816. Use it for frequently-accessed variables:

```rust
#[zeropage]
static mut FRAME_COUNT: u8 = 0;

#[zeropage(0x20)]  // Explicit address
static mut PLAYER_X: u8 = 128;
```

### Hardware Registers

Map hardware registers with `#[hw]`:

```rust
#[hw(0x2100)]
static mut INIDISP: u8;  // Screen brightness

#[hw(0x4212)]
static mut HVBJOY: u8;   // VBlank flag

// Hardware registers are automatically volatile
// Every read/write goes directly to hardware
loop {
    if HVBJOY & 0x80 != 0 { break; }  // Wait for VBlank
}
```

### ROM Data

Immutable statics are automatically stored in ROM:

```rust
static SINE_TABLE: [u8; 256] = [0, 3, 6, 9, /* ... */];

// Include raw binary data from a file (type inferred from the file size)
static SPRITE_DATA = include_bytes!("sprites.bin");
```

## CPU Registers

All 65816 registers are available as global variables:

```rust
A = 0x42;             // Accumulator (u8 default, u16 with @ A: u16 parameter)
X = 10;               // X index (always u16)
Y = 20;               // Y index (always u16)
let hi: u8 = B;       // Accumulator high byte (m8 mode only, via XBA)
STATUS.Carry = false; // Flags are accessed individually (CLC)
D = 0x0000;           // Direct page base
A = 0x7E; DBR = A;    // Data bank (set through A)
S = 0x01FF;           // Stack pointer
// PBR             -- Program bank (read-only)
```

### Register Aliasing

Give registers meaningful names with zero overhead:

```rust
fn update_player() {
    let mut health @ A = PLAYER.health;   // Load into A, call it 'health'
    health = health - 1;                   // Decrement (actually modifies A)
    PLAYER.health = health;                // Store back
}
```

## Functions

### Basic Functions

```rust
fn add(a: u8, b: u8) -> u8 {
    return a + b;
}

let result = add(10, 20);  // result = 30
```

### Register Parameters (Fast!)

Pass values directly in registers for maximum speed:

```rust
#[lowram]
static mut BUFFER: [u8; 256];

fn fill(buffer @ X: *u8, value @ A: u8, count @ Y: u16) {
    // buffer ptr in X, value in A, count in Y
    // No stack overhead!
    while count > 0 {
        count--;
        buffer[count] = value;
    }
}

// Call - value goes straight into registers
fill(&BUFFER, 0, 256);
```

### Multiple Return Values

Functions can return values in multiple registers. The return type is a type list; the compiler assigns registers in order (A, then B or X, then Y) from the types and the mode:

```rust
fn divmod(dividend @ A: u8, divisor: u8) -> u8, u16 {
    // quotient in A, remainder in X
    return A, X;
}

let quotient, remainder = divmod(100, 7);
```

Two 8-bit values (`-> u8, u8`) return in A and B; use `u16` for the second value if you want it in X.

### Never-Return Functions

Use `-> !` for functions that never return (e.g., main game loop):

```rust
#[entry]
fn main() -> ! {
    loop { update(); }  // Compiler omits RTS/RTL
}
```

### Register Preservation

Declare which registers a function preserves:

```rust
#[preserves(X, Y)]
fn safe_function(input @ A: u8) -> u8 {
    // Compiler auto-saves X and Y at entry, restores at exit
    X = 100;  // Safe to modify
    Y = 200;
    return A;
}
```

`#[preserves(A)]` is rejected on a function that returns a value, since the result is passed in A.

## Processor Modes

The 65816 has 8-bit and 16-bit accumulator modes. R65 infers the mode automatically from parameter types:

```rust
// 8-bit accumulator (default)
fn process_byte(value @ A: u8) -> u8 {
    return value + 1;
}

// 16-bit accumulator (inferred from u16 @ A)
fn process_word(value @ A: u16) -> u16 {
    return value + 1;
}
```

X and Y index registers are always 16-bit in R65.

### Data Bank Management

For far functions that access data in their own bank:

```rust
#[bank(1)]
#[mode(databank=inline)]
far fn graphics_routine() {
    // Compiler saves/restores DBR and sets it to this function's bank
}

#[mode(databank=caller)]
far fn batch_function() {
    // Caller is responsible for setting DBR
}
```

## Control Flow

### If/Else

```rust
if health == 0 {
    game_over();
} else if health < 20 {
    flash_warning();
}
```

### Loops

```rust
// Infinite loop
loop {
    update();
    if done { break; }
}

// While loop
while enemies_remaining > 0 {
    spawn_enemy();
    enemies_remaining--;
}

// For loop (0 to 9)
for i in 0..10 {
    BUFFER[i] = 0;
}

// Labeled loops for nested break/continue
'outer: loop {
    'inner: loop {
        if condition { break 'outer; }
    }
}
```

### Match

`match` is an expression over integers, bools and enums. Arms take literals, inclusive/exclusive ranges, or-patterns and `_`; every match must be exhaustive:

```rust
let category: u8 = match tile_id {
    0..=15 => 1,        // inclusive range
    16..32 => 2,        // exclusive range
    _ => 0,
};

let bonus: u8 = match input {
    0 | 1 | 2 => 10,    // or-pattern
    _ => 0,
};
```

See [docs/control-flow.md](docs/control-flow.md).

## Structs

Structs are packed (no padding) and passed by reference:

```rust
struct Player {
    x: u8,
    y: u8,
    health: u16,
    sprite: u8,
}

#[zeropage]
static mut PLAYER: Player;

PLAYER.x = 128;
PLAYER.health = 100;

// Pass by pointer - fields auto-dereference
fn damage_player(player: *Player, amount @ A: u8) {
    player.health = player.health - amount as u16;
}

damage_player(&PLAYER, 10);
```

Nested aggregates are accessed with `outer.inner.field`; the chain folds to one constant offset.

### Methods

`impl` blocks add methods, associated functions and constants to a type. Methods in an `impl` (or `trait`) are **far** (JSL/RTL) by default so they can be called from any bank; `near fn` opts back to JSR/RTS. Free functions stay near.

```rust
impl Player {
    const MAX_HP: u16 = 100;

    fn heal(*self, amount @ A: u8) {      // far (JSL/RTL) by default
        self.health = self.health + amount as u16;
    }
    near fn tick(*self) {                 // near (JSR/RTS)
        self.x = self.x + 1;
    }
    fn reset(p: *Player) {                // associated fn: no self
        p.health = Player::MAX_HP;
    }
}

PLAYER.heal(5);
PLAYER.tick();
Player::reset(&PLAYER);
```

At `-O1` and above the compiler demotes a far method back to JSR/RTS when every caller is in its bank. See [docs/abi-models.md](docs/abi-models.md).

### Newtypes

A single-field tuple struct wraps a register-sized scalar (`u8`, `i8`, `bool`, `u16`, `i16`, an enum or a near pointer). It is a distinct type at compile time and just the payload at runtime, so it passes and returns by value:

```rust
struct TileId(u8);

impl TileId {
    fn zero() -> TileId       { TileId(0) }       // associated fn, no receiver
    fn raw(self) -> u8        { self.0 }          // self arrives in A
    fn bumped(self) -> TileId { TileId(self.0 + 1) }
}

let t = TileId::zero();
let t2: TileId = 5;      // payload flows in implicitly
let u: TileId = t2 + 1;  // operators inherited, result stays TileId
let n: u8 = u.0;         // out only via '.0' or 'as' - 0 cycles
```

Two different newtypes never mix, and `let n: u8 = t;` is an error. The stdlib's `Q10`, `Q8` and `Color` are newtypes. See [docs/type-system.md](docs/type-system.md).

### Unions

All fields of a `union` share offset 0; the size is that of the largest field. There is no tag - reading a field you never wrote yields whatever bytes are there:

```rust
union Pixel { raw: u16, bytes: [u8; 2] }     // 2 bytes total

#[ram]
static mut PIXEL: Pixel = Pixel { raw: 0x1234 };  // literal names exactly one field

PIXEL.raw = 0x1234;
let hi: u8 = PIXEL.bytes[1];                  // 0x12 - same memory
```

Unions take inherent `impl` blocks but cannot implement traits (other than `Clone`). See [docs/unions.md](docs/unions.md).

## Arrays

```rust
#[ram]
static mut BUFFER: [u8; 256] = [0; 256];

BUFFER[0] = 42;
BUFFER[X] = value;  // Index with register

// String literals initialize byte arrays
#[ram]
static mut MESSAGE: [u8; 16] = "Hello, SNES!\0";
```

**Note:** No bounds checking - you're responsible for valid indices.

### SoA Struct Arrays

`#[soa]` on a struct lays an array of it out as one array per field, so `arr[i].field` indexes by the field size with no stride multiply. Methods take `self` as the element **index** in X:

```rust
#[soa]
struct Enemy { kind: u8, hp: u8 }

impl Enemy {
    fn hurt(self) { self.hp = self.hp - 1; }   // LDA POOL__hp,X ...
}

#[lowram]
static mut POOL: [Enemy; 16];   // one [_; 16] plane per field

POOL[i].hp = 100;               // stride 1, no multiply
POOL[i].hurt();                 // passes the index
```

A `#[soa]` struct backs exactly one `static mut`; elements have no contiguous base, so `&POOL[i]`, locals and whole-element copies are rejected. See [docs/soa.md](docs/soa.md).

## Enums

C-style enums are named integer constants with explicit or auto-increment values. They take `impl` blocks, with `self` as the discriminant in A:

```rust
enum Direction { North = 0, East, South, West }

impl Direction {
    fn opposite(self) -> Direction {
        match self {
            Direction::North => Direction::South,
            Direction::East  => Direction::West,
            Direction::South => Direction::North,
            Direction::West  => Direction::East,
        }
    }
}

let dir = Direction::North;
let value: u8 = dir as u8;
let back = dir.opposite();
```

### Tagged Enums

A variant may carry a declared struct. The pool is stored as a tag plane plus a payload plane, so `self` is an element index in X and a pattern binding views the pool rather than copying it:

```rust
struct Circle { r: u8 }
struct Rect   { w: u8, h: u8 }

enum Shape { Empty, C(Circle), R(Rect) }

#[lowram]
static mut POOL: [Shape; 16];      // POOL__tag: [u8; 16] + POOL__data: [_; 16]

impl Shape {
    fn width(self) -> u8 {          // self is the element index, in X
        return match self {
            Shape::C(c) => c.r,     // c views POOL__data[self], no copy
            Shape::R(r) => r.w,
            Shape::Empty => 0,
        };
    }
}

POOL[i] = Shape::C(Circle { r: 5 });   // tag store + payload stores
let w = POOL[i].width();
let kind: u8 = POOL[i] as u8;           // reads the tag
```

A tagged enum backs exactly one `static mut` and takes no initializer. Payload structs may themselves be `#[soa]`, in which case each variant's fields become planes indexed by the raw pool index. See [docs/enums.md](docs/enums.md).

## Pointers

```rust
// Near pointer (16-bit, current bank)
let ptr: *u8 = 0x2000 as *u8;
*ptr = 42;           // Write through pointer
let val = ptr[Y];    // Indexed access

// Far pointer (24-bit, any bank)
let rom_ptr: far *u8 = 0x01_8000 as far *u8;

// Zero-page pointer (fastest indirect access)
#[lowram]
static mut BUFFER: [u8; 256];

#[zeropage(0x40)]
static mut DATA_PTR: *u8;

DATA_PTR = &BUFFER[0];
DATA_PTR[Y] = value;  // LDA/STA ($40),Y - very fast!
```

`&` of a `#[ram]` static is a far pointer (3 bytes); assign it to a `far *u8` or place the data in `#[lowram]`/`#[zeropage]` for a near pointer.

## Operators

R65 uses hardware-aware operators. Fast operations use operators; slow operations use functions:

```rust
// Fast (use operators)
let sum = a + b;
let diff = a - b;
let masked = value & 0x0F;
let shifted = value << 2;     // Constant shift only

// Slow (use functions from math.r65 - makes cost visible)
include!("lib/math.r65")

let product: u16 = mul8(a, b);       // 8x8 -> 16-bit multiply
let quotient: u8 = div8(a, b);       // 8-bit divide
let dynamic_shift = shl8(value, amount);  // Variable shift
```

`*` and `/` accept only power-of-2 constants; shifts need a constant amount. `div16`, `mod8` and `mod16` use the SNES hardware divider and require `--cfg snes`.

### Operator Overloading

A struct gives meaning to the compound-assignment and comparison operators by implementing the compiler-known traits `AddAssign`, `SubAssign`, `MulAssign`, `DivAssign`, `RemAssign`, `BitAndAssign`, `BitOrAssign`, `BitXorAssign`, `ShlAssign`, `ShrAssign`, `PartialEq` (`==`, `!=`) and `PartialOrd` (`<`, `<=`, `>`, `>=`):

```rust
struct Vec2 { x: i16, y: i16 }

impl AddAssign for Vec2 {
    fn add_assign(*self, other: *Vec2) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl PartialEq for Vec2 {
    fn eq(*self, other: *Vec2) -> bool {
        if self.x != other.x { return false; }
        return self.y == other.y;
    }
}

POS += VEL;                    // Vec2__add_assign(&POS, &VEL)
if POS == ORIGIN { reset(); }  // Vec2__eq(&POS, &ORIGIN)
```

Dispatch is static; there is no value-producing `a + b` for aggregates. The stdlib's `I32`, `U32` and `F32` use this. See [docs/operator-overloading.md](docs/operator-overloading.md).

### Clone

Structs and arrays are never copied implicitly (`let c = src;` on an aggregate is an error). Implement `Clone` to copy explicitly:

```rust
impl Clone for Player {}          // empty body = compiler-generated bitwise copy

let c = PLAYER.clone();           // copy into a new local
SAVED = PLAYER.clone();           // or into an existing aggregate
SAVED.clone_from(&PLAYER);        // in-place form
let arr2 = DATA.clone();          // arrays clone without an impl
```

Give `Clone` a body (`fn clone_from(*self, src: *Player) { ... }`) to customize it.

## Interrupts

```rust
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNT++;
    // Compiler auto-generates: PHP, save regs, body, restore regs, PLP, RTI
}

#[interrupt(irq, preserve=false)]  // Manual preservation
fn fast_irq() {
    // You handle register saving
    asm!("RTI");
}
```

## Cross-Bank Calls

For code that spans multiple ROM banks:

```rust
#[bank(0)]
fn main_loop() {
    sound_update();  // JSL to bank 1
}

#[bank(1)]
far fn sound_update() {
    // In bank 1, uses JSL/RTL
}

#[bank(1)]
#[mode(databank=inline)]
far fn graphics_routine() {
    // Compiler manages DBR automatically
}
```

`#[bank(n)]` sets the bank for everything that follows it; `#[bank(auto)]` lets the compiler place functions in the banks above the last explicit one.

## Inline Assembly

When you need direct hardware control:

```rust
fn wait_for_interrupt() {
    asm!("WAI");
}

fn disable_interrupts() {
    asm!("SEI");
}
```

### Assembly Interop

Link against hand-written assembly with `include_asm!` and body-less `extern` declarations:

```rust
include_asm!("sound.s");                        // pulled into the current bank

extern fn sound_tick(a @ A: u8) -> u8;          // JSR
extern far fn sound_play_song(id @ A: u8);      // JSL
extern static SONG_TABLE: [u8; 64];             // placed by the asm file

sound_play_song(SONG_TABLE[0]);
```

An `extern fn` is assumed to clobber everything unless it carries `#[preserves(...)]`. See [docs/reserved-keywords.md](docs/reserved-keywords.md#assembly-interop).

## Macros

`macro_rules!` with six fragment types (`expr`, `ident`, `literal`, `ty`, `reg`, `tt`), repetition (`$(...),*`) and multiple arms. The first arm whose arity and fragment types match wins:

```rust
macro_rules! inc_twice($reg:reg) {
    $reg++;
    $reg++;
}

macro_rules! set_var {
    ($name:ident)             => { $name = 0; };
    ($name:ident, $v:literal) => { $name = $v; };
}

inc_twice!(X);      // Expands to: X++; X++;
set_var!(SCORE);    // arm 1 -> SCORE = 0;
set_var!(SCORE, 5); // arm 2 -> SCORE = 5;
```

### Method Macros

A macro inside an `impl` block is invoked on a receiver, which replaces `self` textually. `Type::name!()` invokes one without a receiver:

```rust
impl Sprite {
    macro_rules! move_by($dx:expr, $dy:expr) {
        self.x = self.x + $dx;
        self.y = self.y + $dy;
    }
}

impl Color {
    macro_rules! rgb($r:expr, $g:expr, $b:expr) {
        { Color((($b & 0x1F) << 10) | (($g & 0x1F) << 5) | ($r & 0x1F)) }
    }
}

PLAYER.move_by!(1, 2);           // self is replaced by PLAYER
let red = Color::rgb!(31, 0, 0); // associated invocation, no receiver
```

See [docs/macros.md](docs/macros.md).

### format! Macro

Format values into a byte buffer (printf-style):

```rust
#[ram]
static mut BUF: [u8; 64] = [0; 64];

format!(BUF, "Score: {u16:05d} HP: {u8}", score, health);
format!(BUF, "Speed: {s}", speed);     // any type with to_string()
```

Supported specifiers: `{u8}`, `{u16}`, `{i8}`, `{i16}` (decimal), `{u8:x}`, `{u16:x}` (hex), `{bool}`, `{c}` (char), `{s}` (string pointer, or any value with a `to_string` method - `U32`, `I32`, `F32`, `Q10`, `Q8`). Width and zero-padding: `{u8:3d}`, `{u16:05d}`.

## Const Evaluation

Compile-time evaluation of constant expressions:

```rust
const TILE_SIZE: u8 = 8;
const MASK: u8 = 0x80 | 0x40;
static BUFFER: [u8; TILE_SIZE * 2] = [0; TILE_SIZE * 2];
```

### Const Functions

`const fn` enables user-defined compile-time computation. Calls with all-const arguments are folded to literals; calls with runtime arguments emit a normal function call.

```rust
const fn tile_offset(x: u8, y: u8) -> u16 {
    return (y as u16) * 32 + (x as u16);
}
const PLAYER_TILE: u16 = tile_offset(5, 3);
```

## Traits

Traits provide TypeId-based dynamic dispatch for polymorphism. Trait methods take stack parameters (no `@ X` bindings), and a trait pointer is spelled `*dyn Trait`:

```rust
trait Drawable {
    fn draw(*self, x: u16, y: u16);
    fn get_width(*self) -> u8;
}

impl Drawable for Player {
    fn draw(*self, x: u16, y: u16) { /* ... */ }
    fn get_width(*self) -> u8 { return 16; }
}

// Heterogeneous collection via trait pointers
#[lowram]
static mut OBJECTS: [*dyn Drawable; 2];

fn draw_all() {
    for i in 0..2 {
        OBJECTS[i].draw(0, 0);
    }
}
```

Trait methods are far by default; declare hot dispatched traits with `near fn` (far dispatch costs roughly twice as much).

### Default Methods and Supertraits

A trait method may supply a body that implementors inherit, and a trait may require supertraits:

```rust
trait Shape {
    fn get_width(*self) -> u8;
    fn get_height(*self) -> u8;

    // Default body: inherited unless the impl overrides it
    fn get_perimeter(*self) -> u16 {
        return ((self.get_width() as u16) + (self.get_height() as u16)) << 1;
    }
}

trait Position { fn px(*self) -> u8; }
trait Sprite   { fn tile(*self) -> u8; }

trait Drawable: Position + Sprite {     // supertraits
    fn draw(*self);
}

let d: *dyn Drawable = &PLAYER;
A = d.px();                   // inherited, dispatched via Position
let p: *dyn Position = d;     // upcast is free
```

The built-in `ToString` trait (`fn to_string(far *self, buf: far *u8) -> u16`) plugs a type into `format!`'s `{s}`. See [docs/traits.md](docs/traits.md) for full trait documentation.

## Conditional Compilation

Use `#[cfg]` to conditionally compile for different targets:

```rust
#[cfg(snes)]
far fn mul16(a @ A: u8, b: u16) -> u16 {
    // Hardware multiplier implementation
}

#[cfg(not(snes))]
far fn mul16(a @ A: u8, b: u16) -> u16 {
    // Software fallback
}
```

## Lints

`r65c --lint` runs the linter after type checking. Built-in rules (`L001 unused_mut` ... `L009 xy16_mode`) and custom rules (forbidden calls, reachability, storage-class enforcement, naming conventions, zeropage budgets, ...) are enabled in an `r65-lint.toml` next to your sources; `r65x init` ships one that also enables the SNES rules `C001` (PPU writes outside VBlank) and `C002` (DMA into PPU ports). Suppress per item:

```rust
#[allow(C001)]
fn vblank_tasks() { /* ... */ }
```

---

# Project Structure

```
my_game/
├── src/
│   ├── main.r65       # Reset / NMI handlers + game loop
│   ├── player.r65
│   ├── enemies.r65
│   └── lib/           # sneslib.r65, math.r65, 65816.r65 copied by r65x init
├── build/             # .asm, .smc, .dbg output
├── r65-lint.toml      # Lint configuration
└── Makefile           # Build script
```

Use `include!()` to combine files:

```rust
// main.r65
include!("player.r65")
include!("enemies.r65")

#[entry]
fn main() -> ! { /* ... */ }
```

---

# Reference

## Compiler Commands

```bash
r65c source.r65 -o output.asm   # Compile to assembly (stdout if -o omitted)
r65c source.r65 -O2             # -O0 none, -O1 default, -O2 adds inlining + loop unrolling
r65c source.r65 -v              # Verbose output (-q for errors only)
r65c source.r65 --dbg           # Mesen-compatible .dbg debug file
r65c source.r65 --cfg snes      # Compile with SNES cfg flag
r65c source.r65 -I src/lib      # Add an include search path
r65c source.r65 --abi FixedStack  # ABI model: Default, FixedStack or Pascal
r65c source.r65 --lint          # Run lints (--lint-only, --allow CODE, --deny CODE)
r65c source.r65 --dump-mir      # Dump tokens/ast/hir/mir; --stop-after PHASE

r65x init --platform snes game  # Create new project
r65x fontgen --font f.ttf > font.r65    # Font tiles from a TrueType font (R65 array literal)
r65x bmp2chr sprite.bmp -o sprite.chr   # Indexed bitmap to CHR data
r65x packer pack data.bin -o data.lz -x lz4   # Compress (aplib, lz*, rle*, hal)
r65x brr encode sample.wav -o sample.brr      # SNES audio samples

r65-emu game.smc --trace        # Built-in 65816 emulator: --breakpoint, --disasm, --profile
```

## Standard Library

| Module | Description |
|--------|-------------|
| `stdlib/sneslib.r65` | SNES hardware registers, PPU/DMA/joypad macros (`wait_nmi!`, `JOY_*`), `OamBuffer`, `Color` newtype |
| `stdlib/math.r65` | `mul8`/`mul16`, `div8`/`div16`, `mod8`/`mod16`, variable shifts |
| `stdlib/string.r65` | `memcpy`/`memset`/`strcpy`/`strcmp`..., character classes, number-to-decimal/hex, `format!` |
| `stdlib/console.r65` | Text console rendering to a nametable buffer, VRAM directly, or OAM sprites |
| `stdlib/default_font.r65` | 2bpp 8x8 font tileset for console |
| `stdlib/I32.r65` | 32-bit signed integer (operators, `Clone`, `ToString`) |
| `stdlib/U32.r65` | 32-bit unsigned integer (operators, `Clone`, `ToString`) |
| `stdlib/F32.r65` | 32-bit floating point (Woz format; operators, `ToString`) |
| `stdlib/Q10.r65` | Q10.6 fixed-point newtype (signed, 1/64) with `from_int`, `mul`, `lerp`, `clamp!` |
| `stdlib/Q8.r65` | Q8.8 fixed-point newtype (unsigned, 1/256) |
| `stdlib/rand.r65` | 16-bit xorshift pseudo-random generator |
| `stdlib/msu1.r65` | MSU-1 streaming audio and data access |
| `stdlib/65816.r65` | STATUS flag constants, `without_irq!`, `block_move!`, `stack_guard_check!` |
| `stdlib/scratch_regs.r65` | Scratch register definitions for compiler |

## Documentation

| Document | Description |
|----------|-------------|
| [r65.dev](https://r65.dev/) | Language reference and guides |
| [docs/AGENTS_R65.md](docs/AGENTS_R65.md) | Primer for AI coding agents writing R65 |
| [CLAUDE.md](CLAUDE.md) | Complete language specification |
| [docs/abi-models.md](docs/abi-models.md) | Function calls, parameters, methods, ABI |
| [docs/pointers-memory.md](docs/pointers-memory.md) | Memory model, addressing modes |
| [docs/type-system.md](docs/type-system.md) | Types, modes, conversions, newtypes |
| [docs/unions.md](docs/unions.md) | Unions |
| [docs/enums.md](docs/enums.md) | Enums, enum methods, tagged enums |
| [docs/soa.md](docs/soa.md) | `#[soa]` struct-of-arrays layout |
| [docs/operators.md](docs/operators.md) | Operator semantics and cost |
| [docs/operator-overloading.md](docs/operator-overloading.md) | Operator traits and `Clone` |
| [docs/control-flow.md](docs/control-flow.md) | Loops, branches, labels, match |
| [docs/code-generation.md](docs/code-generation.md) | How R65 compiles to assembly |
| [docs/macros.md](docs/macros.md) | Macro system |
| [docs/traits.md](docs/traits.md) | Traits and dynamic dispatch |
| [docs/struct-array-indexing.md](docs/struct-array-indexing.md) | Struct array indexing strategies |
| [docs/register-allocation.md](docs/register-allocation.md) | Register allocation strategy |
| [docs/register_memory_config.md](docs/register_memory_config.md) | DBR:Y vs D=S far-pointer strategies |
| [docs/b-register.md](docs/b-register.md) | B register (accumulator high byte) |
| [docs/status-flags.md](docs/status-flags.md) | STATUS register flags |
| [docs/snes-rom-header.md](docs/snes-rom-header.md) | ROM header configuration, FastROM |
| [docs/mode-transition-analysis.md](docs/mode-transition-analysis.md) | Processor mode transitions |
| [docs/interrupt-mode-transition.md](docs/interrupt-mode-transition.md) | Interrupt handler modes |
| [docs/array-bounds-checking.md](docs/array-bounds-checking.md) | Array bounds design rationale |
| [docs/reserved-keywords.md](docs/reserved-keywords.md) | Reserved keywords, assembly interop |
| [docs/packaging-system.md](docs/packaging-system.md) | Package system design (`r65.toml`, `r65pkg`) - not yet implemented |
| [docs/future-work.md](docs/future-work.md) | Deferred features and why |

## External References

- [WLA-DX Assembler](https://wla-dx.readthedocs.io/) - Target assembler
- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Wiki](https://wiki.superfamicom.org/) - SNES hardware reference

---

# Status

R65 is under active development. The core compiler pipeline is functional:

- Lexer, parser, AST
- HIR with type resolution and const evaluation
- Type checking with mode tracking
- MIR with CFG-based optimization (`-O0`/`-O1`/`-O2`) and three ABI models
- Code generation targeting WLA-DX assembly, LoROM/HiROM and FastROM
- Linter with built-in and configurable rules
- Standard library (sneslib, math, string, console, 32-bit and fixed/floating-point types)
- Asset tools (`r65x fontgen`, `bmp2chr`, `packer`, `brr`) and a built-in 65816 emulator (`r65-emu`)
- ~3400 tests, including an end-to-end suite validated on the built-in emulator

---

**R65** - Write modern code, ship retro games.
