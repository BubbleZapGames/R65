# R65 Compiler

A Rust-inspired compiler for 6502/65816 processors targeting WLA-DX assembly syntax, designed specifically for SNES ROM development and reverse engineering.

## Project Philosophy

**Hardware-First Design**: This language embraces the limitations of 8-bit/16-bit architecture rather than abstracting them away. The goal is to create a programming language that closely mirrors the hardware capabilities of the 6502/65816 processor while providing modern type safety and clean syntax.

## Core Design Principles

1. **Hardware Transparency**: CPU registers (A, X, Y), bank boundaries, and processor modes are first-class language concepts
2. **Type Safety**: Catch bank overflow, mode mismatches, and size errors at compile time
3. **Explicit Control**: No `unsafe` keyword - all code has direct hardware access
4. **Zero Abstraction Cost**: High-level constructs compile to efficient assembly matching hand-written code
5. **Simplicity First**: Omit complex Rust features that don't map well to hardware

## Language Features

### Comments

Standard Rust/C-style comments: `//` for line comments, `/* */` for block comments (non-nesting). No doc comments.

### Global Hardware Registers

All 65816 processor registers are exposed as global variables:

```rust
A: u8       // Accumulator (u16 when u16 @ A parameter)
X: u16      // X index register (always 16-bit in R65)
Y: u16      // Y index register (always 16-bit in R65)
B: u8       // Accumulator high byte (m8 mode only)
STATUS: u8  // Processor status flags (NVMXDIZC)
D: u16      // Direct Page register
DBR: u8     // Data Bank Register
PBR: u8     // Program Bank Register (read-only)
S: u16      // Stack Pointer
```

**Rules**: All mutable except `PBR` (read-only). `A` type depends on function parameter (u8 default, u16 if `@ A: u16`). `X` and `Y` are always u16 (x16 mode). `B` only available in m8 mode. Modifying `D`, `DBR`, `S` without restoration causes bugs.

*(See [docs/b-register.md](docs/b-register.md) and [docs/status-flags.md](docs/status-flags.md) for details)*

### Arrays and Strings

Fixed-size arrays with no bounds checking:

```rust
#[ram]
static mut BUFFER: [u8; 256] = [0; 256];
#[ram]
static mut MESSAGE: [u8; 16] = "Hello, World!";  // String literal, zero-padded
```

**Rules**: No runtime bounds checking (too expensive). Out-of-bounds is UB. String literals only in static array initializers; extended ASCII allowed, UTF-8 rejected. Add explicit `\0` for null termination.

**Pass-by-reference only**: Arrays cannot be passed by value. Use pointer parameters.

*(See [docs/array-bounds-checking.md](docs/array-bounds-checking.md) for design rationale)*

### Error Handling

No built-in error handling. Use return codes, global error flags, or sentinel values. No `Result`, `Option`, or `panic!()`.

### Type Conversions

All conversions require explicit `as` keyword - no implicit conversions.

| Conversion | Behavior | Cost |
|------------|----------|------|
| `(x: u8) as u16` | Zero-extend | 2-4 cycles |
| `(x: i8) as i16` | Sign-extend | 4-8 cycles |
| `(x: u16) as u8` | Truncate | 0-2 cycles |
| `(x: u8) as i8` | Reinterpret | 0 cycles |

*(See [docs/type-system.md](docs/type-system.md) for type system rules)*

### File Inclusion and Inline Assembly

```rust
include!("hardware.r65")  // Textual inclusion (C-style #include)
asm!("WAI");              // Embed raw 65816 assembly
asm!("PHP","WAI");        // Multiple instructions
```

**Rules**: `include!()` path relative to including file; global namespace (no modules). `asm!()` treats as black box, assumes all registers clobbered.

### Macros

Simplified `macro_rules!` syntax with 6 fragment types (`expr`, `ident`, `literal`, `ty`, `reg`, `tt`) and repetition (`$(...),*`). Single pattern per macro, no `=>` syntax, no hygiene.

```rust
macro_rules! inc_twice($reg:reg) { $reg++; $reg++; }
inc_twice!(X);  // Expands to: X++; X++;
```

*(See [docs/macros.md](docs/macros.md) for complete syntax)*

### Const Evaluation

Compile-time evaluation of constant expressions (arithmetic, bitwise, logical, casts). Usable in array sizes and attribute parameters. **No const functions**.

```rust
const TILE_SIZE: u8 = 8;
const MASK: u8 = 0x80 | 0x40;
static BUFFER: [u8; TILE_SIZE * 2] = [0; TILE_SIZE * 2];
```

### Enums and Structs

**Enums**: C-style with explicit or auto-increment values. No data-carrying variants.

```rust
enum Direction { North = 0, East, South, West }
let dir = Direction::North;
let value: u8 = dir as u8;
```

**Structs**: Packed (no padding), fields in declaration order. No methods (use free functions).

```rust
struct Player { x: u8, y: u8, health: u16 }  // 4 bytes total
PLAYER.x = 10;
let p = Player { x: 10, y: 20, health: 100 };
```

**Pass-by-reference only**: Structs cannot be passed by value or directly assigned. Use pointers.

### Volatile Semantics

`#[hw]` variables are automatically volatile - every access goes to hardware. No caching, elimination, or reordering.

```rust
#[hw(0x4212)]
static mut HVBJOY: u8;
loop { if HVBJOY & 0x01 != 0 { break; } }  // Always reads hardware
```

### Memory Storage Classes

| Storage | Range | Speed | Attribute |
|---------|-------|-------|-----------|
| Direct Page | `$0000-$00FF` | 2-3 cycles | `#[zeropage]` |
| Low RAM | `$0000-$1FFF` | 3-4 cycles | `#[lowram]` |
| Main RAM | `$7E2000-$7FFFFF` | 4-5 cycles | `#[ram]` |
| ROM | Various | 4-5 cycles | *(implicit)* |
| Hardware | I/O addresses | 4-6 cycles | `#[hw(addr)]` |

**Storage class is determined by mutability:**
- `static` (immutable) → automatically ROM, no attribute needed
- `static mut` → requires explicit storage attribute (`#[zeropage]`, `#[lowram]`, `#[ram]`, or `#[hw]`)

```rust
static MESSAGE: [u8; 12] = "Hello";     // Immutable = ROM (no attribute)
#[zeropage(0x42)]
static mut TEMP: u8;                     // Explicit zeropage address
#[zeropage]
static mut AUTO_ZP: u8;                  // Auto-allocated zeropage
#[ram]
static mut BUFFER: [u8; 256];            // Main RAM
#[stack(0x1F00, 0x1FFF)]                 // Reserve stack region
```

Auto-allocation finds next available address. Zeropage and lowram share physical memory.

**Scratch Registers**: Define with `register` flag for compiler temporaries:
```rust
#[zeropage(0x10, register)]
static mut SCRATCH0: u8;
```

*(See [docs/pointers-memory.md](docs/pointers-memory.md) for complete memory model and [docs/snes-rom-header.md](docs/snes-rom-header.md) for ROM header configuration)*

### Processor Mode (Automatic)

**Simplified mode system**: CPU mode is automatically inferred from parameter types.

```rust
fn process(value @ A: u8) { }    // m8 mode (8-bit A) - default
fn process16(value @ A: u16) { } // m16 mode (16-bit A) - inferred from u16 @ A
fn indexed(idx @ X: u16) { }     // X/Y always u16 (x16 mode)
```

**Rules**:
- **Default mode**: m8 (8-bit A), x16 (16-bit X/Y)
- **Function entry**: m16 if function has `@ A: u16` parameter, otherwise m8
- **X/Y registers**: Always u16 - compiler error if `@ X: u8` or `@ Y: u8`
- **Auto REP/SEP**: Compiler inserts mode switches around 16-bit A operations

**`#[mode]` attribute**: Only for data bank management, not CPU mode:

```rust
#[mode(databank=inline)]  // Callee manages DBR (saves/restores)
#[mode(databank=caller)]  // Caller manages DBR
far fn graphics_code() { }
```

*(See [docs/mode-transition-analysis.md](docs/mode-transition-analysis.md) for details)*

### Register Aliasing and Preservation

**Aliasing**: Named references to hardware registers with zero runtime cost.

```rust
let hitpoints @ A = PLAYER.health;  // A holds hitpoints
hitpoints = hitpoints - 1;           // Modifies A
```

**Preservation**: Compiler generates save/restore code for declared registers.

```rust
#[preserves(X, Y)]
fn preserves_xy(value @ A: u8) -> u8 {
    X = 10; Y = 20;  // Saved at entry, restored at exit
    return A;
}
```

**Valid**: `A`, `X`, `Y`, `STATUS`, `D`, `DBR`. **Invalid**: `B`, `PBR`, `S`.

*(See [docs/register-allocation.md](docs/register-allocation.md) and [docs/calling-convention.md](docs/calling-convention.md))*

### Function Parameters and Returns

Three parameter-passing mechanisms:

| Type | Syntax | Speed |
|------|--------|-------|
| Register | `param @ A: u8` | 0-3 cycles |
| Variable-bound | `param @ VAR: u8` | 3-6 cycles |
| Stack | `param: u8` | 5-10 cycles |

```rust
fn add(left @ A: u8, right @ X: u16) -> u8 { }    // Register (X/Y must be u16)
fn process(temp @ TEMP: u8) -> u8 { }              // Variable-bound
fn calculate(a: u8, b: u8) -> u8 { }               // Stack (must come first)
```

**Note**: X/Y register parameters must be u16 (always 16-bit in R65).

**Returns**: Implicit A return, explicit `return X`, multiple `return (A, X)`, or via zero-page variables. All return paths must have identical signatures.

**Function pointers**: `fn()` (near JSR/RTS), `far fn()` (cross-bank JSL/RTL).

*(See [docs/calling-convention.md](docs/calling-convention.md) for complete ABI)*

### Cross-Bank Function Calls

```rust
fn local_function() { }              // JSR/RTS, bank 0

#[bank(1)]
far fn sound_engine() { }            // JSL/RTL, bank 1

#[mode(databank=inline)]
far fn graphics_code() { }           // Callee manages DBR (PHB/PLB)
```

**Bank directive**: `#[bank(n)]` sets bank context, `#[bank(auto)]` for automatic placement. Immutable statics (ROM) inherit bank.

**Call rules**: Near functions can only call near functions in same bank. Far functions callable from anywhere.

**Data bank modes** (via `#[mode(databank=...)]`):
- `none` (default): No DBR management
- `inline`: Callee saves/restores DBR and sets it to function's bank
- `caller`: Caller manages DBR (for batching multiple far calls)

*(See [docs/calling-convention.md](docs/calling-convention.md) for cross-bank details)*

### Interrupt Handlers

```rust
#[interrupt(nmi)]
fn vblank_handler() { }  // Auto PHP, saves, body, restores, PLP, RTI

#[interrupt(irq, preserve=false)]
fn fast_handler() { }    // Manual control
```

Vectors: `nmi`, `irq`, `brk`, `cop`, `abort`. Default `preserve=true`.

*(See [docs/interrupt-mode-transition.md](docs/interrupt-mode-transition.md))*

### Pointer Types

Pointers use `*` prefix with optional `far`/`near`:

```rust
let *ptr: u8 = 0x2000;            // Near (16-bit, current DBR)
let far *far_ptr: u8 = addr;      // Far (24-bit)
*ptr = 5;                          // Dereference
ptr[Y] = 5;                        // Indexed

#[zeropage(0x42)]
static mut *PTR: u8;               // Zero-page pointer (fastest)
```

*(See [docs/pointers-memory.md](docs/pointers-memory.md) for complete pointer documentation)*

### Variable Initialization

Compiler generates `__init_start()` for static initializers. **SNES RAM is unpredictable at power-on** - always initialize variables that need known values.

### Operators

Hardware-aware operators with restrictions for expensive operations:

| Category | Operator | Restriction |
|----------|----------|-------------|
| Arithmetic | `+`, `-` | None |
| Multiply/Divide | `*`, `/` | Constants 1,2,4,8 only |
| Shift | `<<`, `>>` | Constant amounts only |
| Bitwise | `&`, `\|`, `^`, `~` | None |
| Compare | `==`, `!=`, `<`, `>`, `<=`, `>=` | None |
| Logical | `&&`, `\|\|`, `!` | Short-circuit |

**Function alternatives**: `mul8()`/`mul16()`, `div8()`/`div16()`, `mod8()`/`mod16()`, `shl()`, `shr()` for variable amounts.

**Also**: Compound assignments (`+=`, etc.), increment/decrement (`x++`, `x--` - postfix, statement-only).

*(See [docs/operators.md](docs/operators.md) for complete semantics)*

## What's Omitted

- ❌ Lifetimes and borrowing
- ❌ Traits and generics
- ❌ Error handling types (`Result`, `Option`, `panic!()`)
- ❌ Advanced enums (data-carrying variants)
- ❌ Closures, async/await
- ❌ Procedural macros
- ❌ Pattern matching
- ❌ String types, dynamic collections
- ❌ Module system (`mod`, `pub`)
- ❌ Methods and `impl` blocks
- ❌ `unsafe` keyword
- ❌ Bounds checking

## Compiler Architecture

```
Source (.r65) → Lexer → Parser → AST → HIR → Type Checking → MIR →
Code Generation → WLA-DX Assembly (.asm)
```

**Passes**: Lexer → Parser → HIR (desugar, resolve) → Type Checking → MIR (CFG) → Optimization → Code Generation

*(See [docs/type-system.md](docs/type-system.md) and [docs/code-generation.md](docs/code-generation.md))*

## Using the Compiler

```bash
r65c game.r65 -o game.asm    # Compile to WLA-DX assembly
r65c game.r65                 # Compile to stdout
r65c game.r65 -o game.asm -v  # Verbose output
r65x init --platform snes my_project  # Create test project
```

## Directory Structure

```
/home/nathan/R65/
├── r65/compiler/           # Compiler source
│   ├── frontend/           # Lexer, parser, AST
│   ├── hir/                # High-level IR
│   ├── typeck/             # Type checking
│   ├── mir/                # Mid-level IR
│   ├── codegen/            # Code generation
│   └── builtins/           # Built-in functions
├── stdlib/core65/hw/       # Hardware register definitions
└── docs/                   # Documentation
```

## Target Platform

- **CPU**: 65816 (16-bit extension of 6502)
- **Primary Target**: Super Nintendo Entertainment System (SNES)
- **Assembler**: WLA-DX
- **ROM Format**: LoROM or HiROM

## Key Technical Decisions

1. **No `unsafe`**: All code has direct hardware access
2. **All registers exposed**: A, X, Y, STATUS, D, DBR, S mutable; PBR read-only; B in m8 mode only
3. **Hardware-aware operators**: `*`, `/`, `<<`, `>>` restricted; expensive ops use functions
4. **Automatic volatile**: `#[hw]` variables always access hardware
5. **No bounds checking**: Programmer responsible for index safety
6. **Packed structs**: No alignment padding
7. **Hybrid parameters**: Register, variable-bound, or stack with explicit syntax
8. **Automatic preservation**: `#[preserves(...)]` generates save/restore
9. **Implicit A return**: No explicit return = A returned
10. **Automatic mode inference**: X/Y always x16; A mode inferred from `@ A: u16` parameter

## Detailed Design Documents

### Language Features
- [Operators and Cost Model](docs/operators.md)
- [Control Flow Structures](docs/control-flow.md)
- [Pointers and Memory Model](docs/pointers-memory.md)
- [Type System](docs/type-system.md)
- [B Register](docs/b-register.md)
- [STATUS Flags](docs/status-flags.md)
- [Array Bounds Checking](docs/array-bounds-checking.md)
- [Calling Convention](docs/calling-convention.md)
- [Mode Transitions](docs/mode-transition-analysis.md)
- [Interrupt Handling](docs/interrupt-mode-transition.md)
- [Register Allocation](docs/register-allocation.md)
- [Reserved Keywords](docs/reserved-keywords.md)
- [Macros](docs/macros.md)
- [SNES ROM Header](docs/snes-rom-header.md)

### Code Generation
- [Code Generation](docs/code-generation.md)
- [Struct Array Indexing](docs/struct-array-indexing.md)

## References

- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)

*Last Updated: 2026-01-19*
*STATUS: Design Complete, Implementation In Progress*
