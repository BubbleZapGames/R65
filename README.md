# R65 - Modern Language for Retro Hardware

R65 is a Rust-inspired programming language for the 6502/65816 processor family, designed for SNES development. It gives you the control of assembly with the safety and readability of a modern language.

```rust
#[entry]
fn main() -> ! {
    INIDISP = 0x0F;  // Screen on, full brightness
    loop {
        wait_vblank();
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

```bash
git clone https://github.com/yourusername/R65.git
cd R65
pip install -e .
```

### Create a Project and Build

```bash
r65x init --platform snes my_game
cd my_game
make                          # Assemble and link (requires WLA-DX)
```
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

Use `#[ram]` for general-purpose variables:

```rust
#[ram]
static mut SCORE: u16 = 0;

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
static SPRITE_DATA: [u8; 256] = include_bytes!("sprites.bin");

static SINE_TABLE: [u8; 256] = [0, 3, 6, 9, /* ... */];
```

## CPU Registers

All 65816 registers are available as global variables:

```rust
A = 0x42;        // Accumulator (u8 default, u16 with @ A: u16 parameter)
X = 10;          // X index (always u16)
Y = 20;          // Y index (always u16)
STATUS;          // Processor flags
D = 0x0000;      // Direct page base
DBR = 0x7E;      // Data bank
PBR              // Program bank (read-only)
S                // Stack pointer
```

### Register Aliasing

Give registers meaningful names with zero overhead:

```rust
fn update_player() {
    let health @ A = PLAYER.health;   // Load into A, call it 'health'
    health = health - 1;               // Decrement (actually modifies A)
    PLAYER.health = health;            // Store back
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
fn fill(*buffer @ X: u8, value @ A: u8, count @ Y: u16) {
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

// For loop (countdown)
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

// Pass by pointer
fn damage_player(*player: Player, amount @ A: u8) {
    (*player).health = (*player).health - amount as u16;
}

damage_player(&PLAYER, 10);
```

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

## Enums

C-style enums with explicit or auto-increment values:

```rust
enum Direction { North = 0, East, South, West }

let dir = Direction::North;
let value: u8 = dir as u8;
```

## Pointers

```rust
// Near pointer (16-bit, current bank)
let ptr: *u8 = 0x2000;
*ptr = 42;           // Write through pointer
let val = ptr[Y];    // Indexed access

// Far pointer (24-bit, any bank)
let rom_ptr: far *u8 = 0x01_8000;

// Zero-page pointer (fastest indirect access)
#[zeropage(0x40)]
static mut DATA_PTR: *u8;

DATA_PTR = &BUFFER[0];
DATA_PTR[Y] = value;  // LDA/STA ($40),Y - very fast!
```

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

let product = mul8(a, b);     // 8-bit multiply
let quotient = div16(a, b);   // 16-bit divide
let dynamic_shift = shl8(value, amount);  // Variable shift
```

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

## Macros

Simplified `macro_rules!` with repetition support:

```rust
macro_rules! inc_twice($reg:reg) {
    $reg++;
    $reg++;
}

inc_twice!(X);  // Expands to: X++; X++;
```

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

---

# Project Structure

```
my_game/
├── main.r65           # Entry point
├── player.r65         # Player code
├── enemies.r65        # Enemy code
├── Makefile           # Build script
└── assets/
    ├── sprites.bin
    └── tiles.bin
```

Use `include!()` to combine files:

```rust
// main.r65
include!("player.r65")
include!("enemies.r65")

#[entry]
fn main() { /* ... */ }
```

---

# Reference

## Compiler Commands

```bash
r65c source.r65 -o output.asm   # Compile to assembly
r65c source.r65 -v              # Verbose output
r65c source.r65 --dbg           # Output debugging file 
r65c source.r65 --cfg snes      # Compile with SNES cfg flag
r65x init --platform snes game  # Create new project
```

## Standard Library

| Module | Description |
|--------|-------------|
| `stdlib/sneslib.r65` | SNES hardware registers and DMA macros |
| `stdlib/math.r65` | Multiply, divide, modulo, variable shift |

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Complete language specification |
| [docs/calling-convention.md](docs/calling-convention.md) | Function calls, parameters, ABI |
| [docs/pointers-memory.md](docs/pointers-memory.md) | Memory model, addressing modes |
| [docs/type-system.md](docs/type-system.md) | Types, modes, conversions |
| [docs/operators.md](docs/operators.md) | Operator semantics and cost |
| [docs/control-flow.md](docs/control-flow.md) | Loops, branches, labels |
| [docs/code-generation.md](docs/code-generation.md) | How R65 compiles to assembly |
| [docs/macros.md](docs/macros.md) | Macro system |
| [docs/struct-array-indexing.md](docs/struct-array-indexing.md) | Struct array indexing strategies |

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
- MIR with CFG-based optimization
- Code generation targeting WLA-DX assembly
- Standard library (sneslib, math, extended types)
- End-to-end test suite with 65816 emulator validation

---

**R65** - Write modern code, ship retro games.
