# R65 - Modern Language for Retro Hardware

R65 is a Rust-inspired programming language for the 6502/65816 processor family, designed for SNES development. It gives you the control of assembly with the safety and readability of a modern language.

```rust
#[entry]
#[mode(m8, x8)]
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

### Create a Project

```bash
r65x init --platform snes my_game
cd my_game
```

### Compile and Build

```bash
r65c main.r65 -o main.asm    # Compile R65 to assembly
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

#[ram]
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
A = 0x42;        // Accumulator
X = 10;          // X index
Y = 20;          // Y index
STATUS;          // Processor flags (read-only access to flags)
D = 0x0000;      // Direct page base
DBR = 0x7E;      // Data bank
// PBR            // Program bank (read-only)
// S              // Stack pointer
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
fn multiply(value @ A: u8, count @ X: u8) -> u8 {
    // value is already in A, count in X
    // No stack overhead!
    let result @ A = 0;
    loop {
        if count == 0 { break; }
        result = result + value;
        count--;
    }
    return result;  // Returns A
}

// Call with values already in registers - zero overhead!
let x @ A = 5;
let y @ X = 3;
let product = multiply(x, y);
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

The 65816 has 8-bit and 16-bit modes. R65 tracks these at compile time:

```rust
#[mode(m8, x8)]   // 8-bit accumulator and index registers
fn process_byte(value @ A: u8) -> u8 {
    return value + 1;
}

#[mode(m16, x16)] // 16-bit mode
fn process_word(value @ A: u16) -> u16 {
    return value + 1;
}
```

### Mode Transitions

```rust
#[mode(m16, x16, transition=inline)]
fn safe_16bit() {
    // Callable from any mode - compiler handles transition
}

#[mode(m8, x8)]
fn caller() {
    safe_16bit();  // Works! Compiler inserts PHP/REP.../PLP
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

## Pointers

```rust
// Near pointer (16-bit, current bank)
let *ptr: u8 = 0x2000;
*ptr = 42;           // Write through pointer
let val = ptr[Y];    // Indexed access

// Far pointer (24-bit, any bank)
let far *rom_ptr: u8 = 0x01_8000;

// Zero-page pointer (fastest indirect access)
#[zeropage(0x40)]
static mut *DATA_PTR: u8;

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

// Slow (use functions - makes cost visible)
let product = mul8(a, b);     // 8-bit multiply
let quotient = div16(a, b);   // 16-bit divide
let dynamic_shift = shl(value, amount);  // Variable shift
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
fn main() -> ! { /* ... */ }
```

---

# Reference

## Compiler Commands

```bash
r65c source.r65 -o output.asm   # Compile to assembly
r65c source.r65 -v              # Verbose output
r65c source.r65 --dump-ast      # Debug: show AST
r65c source.r65 --dump-mir      # Debug: show MIR
```

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

## External References

- [WLA-DX Assembler](https://wla-dx.readthedocs.io/) - Target assembler
- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Wiki](https://wiki.superfamicom.org/) - SNES hardware reference

---

# Status

R65 is under active development. Core compiler passes are implemented:

- ✅ Lexer, Parser, AST
- ✅ Type System with mode tracking
- ✅ MIR and code generation
- 🚧 Full pipeline integration
- ⏳ Standard library

---

**R65** - Write modern code, ship retro games.
