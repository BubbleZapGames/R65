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

Standard Rust/C-style comments:

```rust
// Single-line comment

/*
   Multi-line
   block comment
*/

fn example() {
    let x = 10;  // Inline comment

    /* Block comment
       can span multiple lines */
    let y = 20;
}
```

**Comment rules:**
- `//` starts a line comment (until end of line)
- `/* */` for block comments (can be multi-line)
- Block comments do not nest
- Comments are ignored by compiler (no doc comments)

### Global Hardware Registers

All 65816 processor registers are exposed as global variables:

```rust
// Built-in global registers (always in scope)
A: u8       // Accumulator (u16 in m16 mode)
X: u8       // X index register (u16 in x16 mode)
Y: u8       // Y index register (u16 in x16 mode)
B: u8       // Accumulator high byte (only in m8 mode, hidden register)
STATUS: u8  // Processor status flags (NVMXDIZC)
D: u16      // Direct Page register (zero-page base)
DBR: u8     // Data Bank Register (default data bank)
PBR: u8     // Program Bank Register (read-only)
S: u16      // Stack Pointer

// Basic usage
A = 0x0F;
X = A;
B = 0x42;                // Only in m8 mode - see B Register section
D = 0x2000;              // Change zero-page base
DBR = 0x7E;              // Set data bank
let bank = PBR;          // Read current bank (write = error)
```

**Rules:**
- All registers mutable except `PBR` (read-only, write is compile error)
- `A`, `X`, `Y` types change with processor mode (u8/u16)
- `B` only available in `#[mode(m8)]` (compile error in m16 mode)
- `D`, `S` always u16; `STATUS`, `DBR`, `PBR`, `B` always u8
- All usable in aliasing (`let name @ D = expr`) and `#[preserves(...)]` (except `PBR` and `B`)
- **Safety**: Modifying `D`, `DBR`, `S` without restoration causes bugs/crashes

*(See [docs/b-register.md](docs/b-register.md) for complete details, code generation tables, and optimization patterns)*

### Arrays

Fixed-size arrays with no bounds checking:

```rust
#[ram]
static mut BUFFER: [u8; 256] = [0; 256];

fn process() {
    BUFFER[0] = 42;      // OK
    BUFFER[255] = 99;    // OK
    BUFFER[300] = 1;     // No bounds check - undefined behavior!

    let index = 10;
    BUFFER[index] = 5;   // No runtime bounds check
}
```

**Array indexing rules:**
- No compile-time bounds checking (even for constant indices)
- No runtime bounds checking (too expensive on 6502/65816)
- Programmer is responsible for ensuring valid indices
- Out-of-bounds access is undefined behavior
- Matches hardware-level programming expectations

*(See [docs/array-bounds-checking.md](docs/array-bounds-checking.md) for design rationale)*

### String Literals for Byte Arrays

String literals can be used to initialize `[u8; N]` arrays in static declarations:

```rust
#[ram]
static mut MESSAGE: [u8; 16] = "Hello, World!";  // Zero-padded to 16 bytes

#[ram]
static mut ESCAPED: [u8; 8] = "A\nB\tC\0";  // With escape sequences

#[ram]
static mut HEX_DATA: [u8; 4] = "\xC0\xC1\xFE\xFF";  // Hex escapes for high bytes
```

**String literal rules:**
- Only allowed in static array initializers (no inline string expressions)
- Target type must be `[u8; N]` (compile error otherwise)
- Extended ASCII (0x00-0xFF) allowed; UTF-8 multi-byte characters are rejected
- If string is shorter than array, remaining bytes are zero-padded
- If string is longer than array, compile error

**Supported escape sequences:**
| Escape | Value | Description |
|--------|-------|-------------|
| `\n` | 0x0A | Newline |
| `\t` | 0x09 | Tab |
| `\r` | 0x0D | Carriage return |
| `\0` | 0x00 | Null |
| `\\` | 0x5C | Backslash |
| `\"` | 0x22 | Double quote |
| `\x##` | 0x00-0xFF | Hex byte value |

**No automatic null termination:** Strings are not null-terminated by default. Add explicit `\0` if needed:
```rust
#[ram]
static mut C_STRING: [u8; 8] = "Hello\0";  // Null-terminated
```

### Error Handling

No built-in error handling - programmer defines conventions:

```rust
// Common patterns: return codes, error flags, multiple returns
fn divide(a: u8, b: u8) -> (u8, u8) {
    if b == 0 { return 0, 1; }  // result=0, error=1
    return a / b, 0;             // result, error=0
}
```

**Rules**: No `Result`, `Option`, or `panic!()`; use return codes, global error flags, or sentinel values.

### Type Conversions and Casting

All conversions require explicit `as` keyword - no implicit conversions.

| Conversion | Syntax | Behavior | Cost |
|------------|--------|----------|------|
| Widening (unsigned) | `(x: u8) as u16` | Zero-extend | 2-4 cycles |
| Widening (signed) | `(x: i8) as i16` | Sign-extend | 4-8 cycles |
| Narrowing | `(x: u16) as u8` | Truncate (keep low byte) | 0-2 cycles |
| Reinterpret | `(x: u8) as i8` | Same bits, new type | 0 cycles |
| To boolean | `(x: u8) as bool` | 0=false, non-zero=true | 2 cycles |
| From boolean | `true as u8` | Normalize to 0 or 1 | 2 cycles |

```rust
let wide: u16 = (narrow: u8) as u16;  // Zero-extend
let byte: u8 = (word: u16) as u8;     // Truncate
```

**Code generation**: Compiler chooses between memory-based (LDA/STA) or mode-switching (REP/SEP) strategies; batches mode changes when beneficial.

*(See [docs/type-system.md](docs/type-system.md) for type system rules and mode-aware casting)*

### File Inclusion

C-style textual file inclusion (no module system):

```rust
include!("hardware.r65")  // Insert file contents here
include!("player.r65")

fn main() {
    init_hardware();
    update_player();
}
```

**Rules**: `include!("path")` performs textual inclusion like C `#include`; path relative to including file; all code shares global namespace (no modules, no `mod`/`pub`); circular includes are errors.

### Inline Assembly

Embed raw 65816 assembly:

```rust
fn wait() {
    asm!("WAI");
}

fn multi_instruction() {
    asm!("PHP","WAI");
}
```

**Rules**: `asm!("inst1","inst2" ...)` emits raw assembly verbatim; no variable interpolation; compiler assumes all registers clobbered; no optimization across boundaries; programmer handles register preservation.

### Const Evaluation

Compile-time evaluation of constant expressions:

```rust
const TILE_SIZE: u8 = 8;
const TILES_PER_ROW: u16 = 256 / TILE_SIZE;  // Arithmetic
const BIT_MASK: u8 = 0x80 | 0x40;            // Bitwise

static BUFFER: [u8; TILES_PER_ROW] = [0; TILES_PER_ROW];  // Array size
#[zeropage(TILE_SIZE * 2)]  // Attribute parameter
static mut TEMP: u16;
```

**Rules**: Supports arithmetic, bitwise, logical ops, and type casts; usable in array sizes and attribute parameters; **no const functions** - expressions only.

### Enums

C-style enums with explicit or auto-increment values:

```rust
enum Direction { North = 0, East, South, West }  // Auto-increment after 0
enum State { Idle, Running, Jumping }            // Starts at 0

let dir = Direction::North;
if dir == Direction::North { }
let value: u8 = dir as u8;  // Cast to/from integers
```

**Rules**: No data-carrying variants; auto-increment from 0 or previous + 1; compiler infers smallest type; access with `Enum::Variant`; comparable with `==`/`!=`; cast to/from integers.

### Structs

Packed structs (no alignment padding):

```rust
struct Player {
    x: u8,       // Offset 0
    y: u8,       // Offset 1
    health: u16, // Offset 2-3 (packed, no alignment)
}  // Size: 4 bytes

#[zeropage(0x30)]
static mut PLAYER: Player;

#[ram]
static mut ENEMIES: [Player; 8];

// Field access with . operator
PLAYER.x = 10;
PLAYER.health = 100;
ENEMIES[0].x = 5;  // Array access

// Struct literal initialization (in let/static contexts)
let p = Player { x: 10, y: 20, health: 100 };

#[ram]
static mut DEFAULT_PLAYER: Player = Player { x: 0, y: 0, health: 100 };
```

**Rules**: All structs packed (no padding); fields in declaration order; size = sum of field sizes; use `.` for field access; nested/array access supported; no methods (use free functions). Struct literals are allowed in `let` and `static` initializers.

### Volatile Semantics

`#[hw]` variables are automatically volatile - every access goes to hardware:

```rust
#[hw(0x4212)]
static mut HVBJOY: u8;

#[hw(0x2100)]
static mut INIDISP: u8;

loop {
    let status = HVBJOY;  // Always reads hardware
    if status & 0x01 != 0 { break; }
}

INIDISP = 0x80;  // Write 1
INIDISP = 0x0F;  // Write 2 (not eliminated)
```

**Rules**: Every read/write accesses hardware; no caching, elimination, or reordering of `#[hw]` accesses; critical for memory-mapped I/O and polling. Non-hardware volatiles: use `#[hw]` or `asm!()` barriers.

### Memory Storage Classes

Variables can be placed in different memory regions with explicit attributes:

```rust
// Direct page (zero-page) - fastest (2-3 cycles)
// Range: $0000-$00FF
#[zeropage(0x42)]
static mut TEMP: u8 = 0;

#[zeropage]
static mut AUTO_ZP: u8;  // Auto-allocated to next available address

// Low RAM - shared with zeropage and stack
// Range: $0000-$1FFF (explicit), auto-allocation starts at $0100
#[lowram(0x0200)]
static mut BUFFER: [u8; 256];

#[lowram]
static mut AUTO_LOW: u8;  // Auto-allocated from $0100+ (avoids stack)

// Main RAM - slower (4-5 cycles)
// Range: $7E2000-$7FFFFF
#[ram]
static mut WORK_RAM: [u8; 256] = [0; 256];

// ROM data - read-only
#[rom(0x8000)]
static GRAPHICS: [u8; 4096] = include_bytes!("gfx.bin");

// Hardware registers - memory-mapped I/O
#[hw(0x2100)]
static mut INIDISP: u8;  // Screen brightness register
```

### Stack Reservation

Reserve a region in low RAM for the stack using `#[stack(lower, upper)]`:

```rust
// Reserve $1F00-$1FFF for stack (256 bytes) - global directive
#[stack(0x1F00, 0x1FFF)]

// Low RAM auto-allocation will skip the stack region
#[lowram]
static mut VAR: u8;  // Gets $0100, not $1F00
```

**Default Stack:** If no `#[stack]` attribute is specified, the default stack region is `$0100-$01FF` (256 bytes). The stack pointer is automatically initialized in the `#[entry]` function prologue if the upper bound is not `$01FF`.

**Memory Map:**
| Region | Range | Storage Class | Notes |
|--------|-------|---------------|-------|
| Direct Page | `$0000-$00FF` | `#[zeropage]` | Faster DP addressing |
| Low RAM | `$0000-$1FFF` | `#[lowram]` | Auto starts at `$0100` |
| Stack | (any slice in low RAM) | `#[stack(lo, hi)]` | Default: `$0100-$01FF` |
| Main RAM | `$7E2000-$7FFFFF` | `#[ram]` | SNES work RAM |
| Hardware | I/O addresses | `#[hw(addr)]` | Auto-volatile |

**Auto-Allocation:**
- Variables without explicit addresses are auto-allocated in source order
- Auto-allocation finds the next available address that fits the variable's size
- Explicit addresses are used as-is without collision checking
- Zeropage and lowram share the same physical memory ($0000-$1FFF)

### Scratch Registers (Compiler-Managed Memory)

The compiler uses **scratch registers** for temporary values. Memory management is the programmer's responsibility - define scratch registers with the `register` flag:

```rust
#[zeropage(0x10, register)]
static mut SCRATCH0: u8;

#[zeropage(0x12, register)]
static mut SCRATCH1: u16;

#[ram(0x7E0000, register)]
static mut RAM_SCRATCH: u8;  // Slower but more available
```

**Rules:**
- Compiler **never** auto-allocates scratch space
- Without scratch registers, temporaries use the stack (slower)
- More scratch = better performance, fewer stack operations
- Regular variables (no `register` flag) are never used as scratch

### Processor Mode Annotations

Functions specify register sizes (8-bit or 16-bit) with `#[mode(...)]`:

```rust
#[mode(m8, x8)]   // 8-bit accumulator and index: A=u8, X=u8, Y=u8
#[mode(m16, x16)] // 16-bit mode: A=u16, X=u16, Y=u16
#[mode(x16)]      // Partial mode: only X/Y size specified
```

**Mode Transition Strategies:**

| Option | Behavior | Use Case |
|--------|----------|----------|
| `transition=none` (default) | Programmer handles mode changes | Manual control, performance |
| `transition=inline` | Callee saves/restores mode | Safe, works from any caller mode |
| `transition=caller` | Caller manages mode transition | Enables batching multiple calls |

```rust
#[mode(m16, x16, transition=inline)]
fn safe_16bit() { }  // Callable from any mode

#[mode(m16, x16, transition=caller)]
fn batch_me() { }    // Caller can batch multiple same-mode calls
```

*(See [docs/mode-transition-analysis.md](docs/mode-transition-analysis.md) and [docs/type-system.md](docs/type-system.md) for mode tracking and type system details)*

### Register Aliasing

Register aliases provide named references to hardware registers using `let name @ register = value` syntax:

```rust
fn process_player() {
    let hitpoints @ A = PLAYER[0].hitpoints;  // A now holds hitpoints
    hitpoints = hitpoints - 1;                 // Same as: A = A - 1
    PLAYER[0].hitpoints = hitpoints;
}

fn calculate() {
    let x_coord @ X = entity.x;
    let y_coord @ Y = entity.y;
    x_coord = x_coord + 1;  // Modifies X
    y_coord = y_coord + 1;  // Modifies Y
}
```

**Aliasing rules:** The alias is a true reference to the register (not a copy), zero runtime cost

*(See [docs/register-allocation.md](docs/register-allocation.md) for register allocation strategy)*

### Register Preservation

Functions declare which registers they preserve with `#[preserves(...)]`. The compiler automatically generates save/restore code:

```rust
#[preserves(X, Y)]
fn preserves_xy(value @ A: u8) -> u8 {
    X = 10;      // Compiler saves X at entry, restores at exit
    Y = 20;      // Compiler saves Y at entry, restores at exit
    return A;    // X and Y guaranteed unchanged to caller
}
```

**Generated code:**
```asm
preserves_xy:
    PHX          ; Auto-generated save
    PHY          ; Auto-generated save
    LDX #$0A
    LDY #$14
    PLY          ; Auto-generated restore
    PLX          ; Auto-generated restore
    RTS
```

**Valid registers**: `A`, `X`, `Y`, `STATUS`, `D`, `DBR`
**Invalid registers**: `B` (tied to A), `PBR` (read-only), `S` (stack pointer)

*(See [docs/calling-convention.md](docs/calling-convention.md) for complete ABI and preservation details)*

### Function Parameters

R65 supports three parameter-passing mechanisms for maximum flexibility:

| Type | Syntax | Speed | Use Case |
|------|--------|-------|----------|
| Register alias | `param @ A: u8` | Fastest (0-3 cycles) | Performance-critical, few parameters |
| Variable-bound | `param @ VAR: u8` | Fast (3-6 cycles) | Zero-page communication patterns |
| Stack | `param: u8` | Slower (5-10 cycles) | Many parameters, reentrancy |

```rust
fn add(left @ A: u8, right @ X: u8) -> u8 { }     // Register parameters
fn process(temp @ TEMP: u8) -> u8 { }              // Zero-page parameter
fn calculate(a: u8, b: u8) -> u8 { }               // Stack parameters
```

**Key Rules**: Stack parameters must come first; zero-cost calls when arguments match parameter aliases.

*(See [docs/calling-convention.md](docs/calling-convention.md) for ABI details, stack layout, and calling conventions)*

### Function Pointers

Function pointers encode calling convention in the type:

```rust
type RegCallback = fn(a @ A: u8, b @ X: u8) -> u8;   // Near, register params
type FarCallback = far fn(a @ A: u8) -> u8;          // Far call

#[ram]
static mut HANDLER: fn(input @ A: u8) -> u8;

fn update(input @ A: u8) {
    let result = HANDLER(input);  // Indirect call via trampoline
}
```

**Rules**: Type system enforces matching conventions; compiler generates trampolines for indirect calls; useful for callbacks, state machines, dispatch tables.

### Cross-Bank Function Calls

The `far` keyword indicates JSL/RTL calling convention, while `#[bank]` controls placement:

```rust
fn local_function() { }                 // JSR/RTS (near call, default)

#[bank(1)]
far fn sound_engine() { }               // JSL/RTL, data_bank=none (default)

#[bank(1, data_bank=inline)]
far fn graphics_code() { }              // JSL/RTL, callee manages DBR

#[bank(2, data_bank=caller)]
far fn decompression_routine() { }     // JSL/RTL, caller manages DBR
```

**Calling conventions:**
- `fn()`: Near call using JSR/RTS (16-bit address, same bank)
- `far fn()`: Far call using JSL/RTL (24-bit address, cross-bank)

**Data Bank Register (DBR) options:**
- `data_bank=none` (default): No DBR management - programmer handles manually
- `data_bank=inline`: Callee sets/restores DBR to its program bank
- `data_bank=caller`: Caller sets/restores DBR (enables batching multiple calls)

*(Function parameters, pointers, and cross-bank calls detailed in [docs/calling-convention.md](docs/calling-convention.md))*

### Interrupt Handlers

Interrupt handlers use `#[interrupt(vector)]` where vector is `nmi`, `irq`, `brk`, `cop`, or `abort`.

```rust
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNT = FRAME_COUNT + 1;
    // Auto-generates: PHP, register saves, body, register restores, PLP, RTI
}

#[interrupt(irq, preserve=false)]  // Manual control for performance
fn fast_handler() {
    // Programmer handles all preservation
}
```

**Preservation**: `preserve=true` (default) auto-generates PHP/PLP and register saves/restores; `preserve=false` for manual control; using `#[preserves(...)]` implies `preserve=false`.

*(See [docs/interrupt-mode-transition.md](docs/interrupt-mode-transition.md) for interrupt mode transition details)*

### Function Return Values

Functions implicitly return A unless an explicit `return` statement specifies otherwise:

```rust
fn read_status() {
    A = HVBJOY;      // Implicitly returns A
}

fn get_x_value() -> u8 {
    X = 100;
    return X;        // Return via X register
}

fn divide(dividend: u8, divisor: u8) -> (u8, u8) {
    return A, X;     // Return multiple registers
}

fn calculate() -> u8 {
    let result = 42;
    return result;   // Return local variable via stack
}
```

**Return conventions:** No `return` = A implicitly returned; `return X/Y` = specific register; `return A, X` = multiple registers; `return variable` = stack return

### Built-in Functions for Special Instructions

```rust
// Block move instructions (65816 only)
A = 255;        // count - 1
X = 0x1000;     // src_addr
Y = 0x2000;     // dest_addr
mvn(0x00, 0x7E);  // Move forward from bank 0x00 to bank 0x7E
mvp(0x7E, 0x00);  // Move backward from bank 0x7E to bank 0x00

// Processor mode control
SEP(0x30);  // Set processor status bits (m8, x8 mode)
REP(0x30);  // Reset processor status bits (m16, x16 mode)

// B register access (m8 mode only)
xba();  // Exchange B and A registers (swap high/low bytes)

// Control instructions
wai();  // Wait for interrupt
stp();  // Stop processor
```

### Pointer Types

```rust
near<T>  // 16-bit pointer (current bank)
far<T>   // 24-bit pointer (includes bank byte)

// Example: Fast indirect addressing through zero-page
#[zeropage(0x42)]
static mut PTR: near<u8>;

*PTR = 5;      // Generates: LDA #$05, STA ($42)
PTR[Y] = 5;    // Generates: LDA #$05, STA ($42),Y
```

*(See [docs/pointers-memory.md](docs/pointers-memory.md) for pointer types and memory model)*

### Variable Initialization

The compiler generates an `__init_start()` routine for static variables with initializers:

```rust
#[zeropage]
static mut FLAGS: u8 = 0x80;     // Initialized
static mut LIVES: u8;            // No initializer - undefined value!
```

**Important:** SNES RAM contents at power-on are **unpredictable**. Variables without initializers have undefined values. Always initialize variables that need known starting values.

### Operators and Hardware Cost Model

R65 uses **hardware-aware operators**: syntax indicates performance cost.

| Category | Operator | Restriction | Function Alternative |
|----------|----------|-------------|---------------------|
| Arithmetic | `+`, `-` | None | - |
| Multiply | `*` | Constants 1,2,4,8 only | `mul(a,b)` |
| Divide | `/` | Constants 1,2,4,8 only | `div(a,b)` |
| Bitwise | `&`, `\|`, `^`, `~` | None | - |
| Shift | `<<`, `>>` | Constant amounts only | `shl(a,n)`, `shr(a,n)` |
| Compare | `==`, `!=`, `<`, `>`, `<=`, `>=` | None | - |
| Logical | `&&`, `\|\|`, `!` | Short-circuit | - |

- **Compound assignments**: `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
- **Increment/decrement**: `x++`, `x--` (statement-only, postfix)
- All operations wrap on overflow (no runtime checks)

*(See [docs/operators.md](docs/operators.md) for complete operator semantics and assembly mappings)*

## What's Included (Minimal Feature Set)

- ✅ Basic types: `u8, i8, u16, i16, bool`
- ✅ Fixed-size arrays: `[T; N]`
- ✅ Structs (no methods initially)
- ✅ C-style enums: Explicit or auto-increment values; no data-carrying variants
- ✅ Functions with parameters and return types
- ✅ Register aliasing: `let name @ A = expr` for named register access
- ✅ Hybrid function parameters: register aliases (`param @ A`), variable-bound (`param @ VAR`), or stack values (`param`)
- ✅ Function pointers: `fn()` (near) and `far fn()` (cross-bank) with calling convention encoded in type
- ✅ Register preservation: `#[preserves(A, X, Y, STATUS, D, DBR, S)]` to declare preservation guarantees
- ✅ Interrupt handlers: `#[interrupt(nmi/irq/brk/cop/abort)]` with automatic register preservation and RTI
- ✅ Control flow: `if/else, loop, while, loop-while, break, continue, return, never type (!)` - See [docs/control-flow.md](docs/control-flow.md)
- ✅ Operators with hardware cost model:
  - Arithmetic: `+`, `-`, `*` (constants 1/2/4/8 only), `/` (constants 1/2/4/8 only)
  - Functions for expensive ops: `mul()`, `div()`, `mod()`, `shl()`, `shr()`
  - Bitwise: `&`, `|`, `^`, `~`, `<<` (constant), `>>` (constant)
  - Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
  - Logical: `&&`, `||`, `!` (with short-circuit evaluation)
  - Compound assignment: `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
  - Increment/decrement: `++`, `--` (statement-only, postfix)
- ✅ `let` bindings (immutable by default, `let mut` for mutable)
- ✅ All 65816 processor registers: A, X, Y, STATUS, D, DBR, S (mutable); PBR (read-only); B (m8 mode only)
- ✅ B register support: Hidden accumulator high byte in m8 mode with parameter passing, return values, and XBA context tracking optimization
- ✅ Storage attributes: `#[zeropage]`, `#[lowram]`, `#[ram]`, `#[rom]`, `#[hw]`, `#[stack(lower, upper)]`
- ✅ Mode annotations: `#[mode(m8/m16, x8/x16)]` with optional `transition=none/auto/caller`
- ✅ Built-in mode control: `SEP()`, `REP()`, and `xba()` functions for manual mode and register control
- ✅ Far/near calling conventions: `far fn()` for JSL/RTL cross-bank calls; `fn()` for JSR/RTS near calls
- ✅ Bank management: `#[bank(n)]` for function placement with optional `data_bank=none/auto/caller` for DBR management
- ✅ Const evaluation: Compile-time evaluation of constant expressions (arithmetic, bitwise, logical operations); no const functions
- ✅ Inline assembly: `asm!("instruction")` for embedding raw 65816 assembly; no variable interpolation
- ✅ File inclusion: `include!("file")` for textual inclusion (C-style); no module system

## What's Omitted (Too Complex or Incompatible)

- ❌ Lifetimes and borrowing
- ❌ Traits and generics
- ❌ Error handling types (`Result`, `Option`, `panic!()`)
- ❌ Advanced enums (data-carrying variants)
- ❌ Closures
- ❌ Async/await
- ❌ Procedural macros (simplified declarative macros planned - see [docs/macros.md](docs/macros.md))
- ❌ Pattern matching (initially - can add later)
- ❌ String types (`String`, `&str`)
- ❌ Dynamic collections (`Vec`, `HashMap`)
- ❌ Module system (`mod`, `pub`, visibility, namespacing)
- ❌ Methods and `impl` blocks (initially - use free functions)
- ❌ `unsafe` keyword (all code has direct hardware access)
- ❌ Bounds checking (compile-time or runtime)


## Compiler Architecture

### Pipeline

```
Source (.r65) → Lexer → Parser → AST → HIR → Type Checking → MIR →
Code Generation → WLA-DX Assembly (.asm)
```

### Compiler Passes

1. **Lexer**: Tokenize source code, recognize register keywords
2. **Parser**: Build AST with special nodes for register operations
3. **HIR (High-level IR)**: Desugar syntax, resolve names, process attributes
4. **Type Checking**: Validate types, modes, register usage, bank boundaries - See [docs/type-system.md](docs/type-system.md)
5. **MIR (Mid-level IR)**: CFG construction, virtual registers
6. **Optimization**: Constant propagation, dead code elimination, zero-page allocation
7. **Code Generation**: Memory allocation, register allocation, instruction selection, addressing modes, WLA-DX emission - See [docs/code-generation.md](docs/code-generation.md)

## Using the Compiler

The R65 compiler (`r65c`) provides a simple, user-friendly command-line interface:

### Basic Usage

```bash
# Compile R65 source to WLA-DX assembly
r65c game.r65 -o game.asm

# Compile to stdout
r65c game.r65

# Compile from stdin
cat source.r65 | r65c - -o output.asm

# Verbose output (show compilation phases)
r65c game.r65 -o game.asm -v

# Quiet mode (suppress all non-error output)
r65c game.r65 -o game.asm -q
```

### Development/Debugging Options

For compiler developers and debugging:

```bash
# Dump intermediate representations
r65c game.r65 --dump-ast         # Show parsed AST
r65c game.r65 --dump-hir         # Show High-Level IR
r65c game.r65 --dump-mir         # Show Mid-Level IR
r65c game.r65 --dump-tokens      # Show tokenized output

# Stop at specific compilation phase
r65c game.r65 --stop-after parse      # Stop after parsing
r65c game.r65 --stop-after hir        # Stop after HIR building
r65c game.r65 --stop-after typecheck  # Stop after type checking
r65c game.r65 --stop-after mir        # Stop after MIR building
```

### Installation

After installing via `pip install -e .`, the `r65c` command becomes available system-wide.

## Directory Structure (Planned)

```
/home/nathan/R65/
├── r65/
│   └─ compiler/
│      ├── main.py              # CLI entry point
│      ├── frontend/            # Lexer, parser, AST
│      ├── hir/                 # High-level IR
│      ├── typeck/              # Type checking, mode checking
│      ├── mir/                 # Mid-level IR
│      ├── optimize/            # Optimization passes
│      ├── codegen/             # Code generation
│      ├── builtins/            # Built-in functions (mvn, mvp, etc.)
│      ├── tests/               # Unit and integration tests
│      └── utils/               # Errors, diagnostics
├── stdlib/                  # Standard library
│   └── core65/
│       └── hw/              # Hardware register definitions
├── docs/                    # Documentation
├── setup.py
├── requirements.txt
└── README.md
```

## Memory Hierarchy

Performance characteristics of different storage:

1. **Registers** (A, X, Y): Fastest - in CPU (2 cycles)
2. **Direct Page** ($0000-$00FF): Very fast - special addressing mode (2-3 cycles)
3. **Low RAM** ($0100-$1FFF): Fast - same bank as direct page (3-4 cycles)
4. **Main RAM** ($7E2000-$7FFFFF): Slower - requires bank switching (4-5 cycles)
5. **ROM**: Read-only data (4-5 cycles)
6. **Hardware Registers**: Memory-mapped I/O (4-6 cycles)

## Target Platform

- **CPU**: 65816 (16-bit extension of 6502)
- **Primary Target**: Super Nintendo Entertainment System (SNES)
- **Assembler**: WLA-DX
- **ROM Format**: LoROM or HiROM

## Key Technical Decisions

1. **No `unsafe` keyword**: All code has direct hardware access by default
2. **All processor registers exposed**: A, X, Y, STATUS, D, DBR, S exposed as mutable global variables; PBR exposed as read-only global (write is compile error)
3. **Automatic volatile**: All `#[hw]` variables are automatically volatile; every access goes to hardware, no caching or reordering
4. **Limited bounds checking**: Compile-time bounds checking for constant array indices; no runtime bounds checking; programmer responsible for dynamic index safety
5. **No error handling**: No built-in Result, Option, or panic; programmer defines own error conventions
6. **Hardware-aware operators**: Operators (`*`, `/`, `<<`, `>>`) restricted to cheap operations (constants 1/2/4/8 for multiply/divide, constant shifts); expensive operations use explicit functions (`mul()`, `div()`, `mod()`, `shl()`, `shr()`); syntax immediately reveals performance cost
7. **Context-aware type conversions**: Compiler chooses between memory-based and REP/SEP-based conversions for optimal performance; batches mode changes when beneficial
8. **Const expressions only**: Compile-time evaluation of constant expressions supported; const functions not supported
9. **Inline assembly**: `asm!()` for raw assembly with simple string syntax; compiler treats as black box, assumes all registers clobbered
10. **C-style enums**: Simple enums with explicit or auto-increment values; no explicit underlying type; cast to/from integers
11. **File inclusion only**: `include!()` for textual file inclusion (C-style); no module system, visibility, or namespacing
12. **Packed structs**: All structs are packed by default with no alignment padding; fields laid out in declaration order
13. **Register aliasing**: `let name @ A = expr` creates zero-cost aliases for registers; improves readability without runtime overhead
14. **Hybrid parameters**: Three parameter types: register aliases (`param @ A`), variable-bound (`param @ VAR` for existing static variables), or stack values (`param` with callee cleanup)
15. **Parameter ordering**: Stack parameters must precede aliased parameters; compiler error otherwise
16. **Argument alias optimization**: No setup code generated when call arguments already match parameter aliases; enables zero-cost calling conventions
17. **Automatic preservation**: `#[preserves(...)]` declares register preservation; compiler auto-generates PHA/PLA, PHX/PLX, PHY/PLY at entry/exit
18. **Interrupt preservation**: `#[interrupt(vector)]` defaults to automatic register preservation (`preserve=true`); can be disabled with `preserve=false` or `#[preserves(...)]` for manual control
19. **Implicit A return**: Functions without explicit `return` statements return A register value
20. **Explicit register returns**: `return X`, `return Y`, `return A, X` return via hardware registers; local variables returned via stack
21. **Storage attributes**: Memory location separate from type (`near<T>` can be in zero-page or RAM)
22. **Flexible mode handling**: `#[mode(...)]` with three transition strategies: `none` (convention-based, default), `auto` (callee wrapper), `caller` (caller-side wrapper with batching)
23. **Automatic initialization**: `__init_start()` generated for all static variables with explicit initializers (RAM is not zeroed on SNES power-on)
24. **Consistent far/near**: `far fn()` for both function definitions and pointers indicates JSL/RTL calling convention; `fn()` indicates JSR/RTS; `#[bank(n)]` controls placement with optional `data_bank` parameter

## Use Cases

1. **SNES Game Development**: Write new games with modern syntax and type safety
2. **ROM Reverse Engineering**: Disassemble ROMs into readable source
3. **ROM Hacking**: Modify existing games with better tooling
4. **Education**: Learn 6502/65816 architecture with safer, clearer code

## Future Enhancements

- Declarative macros (simplified `macro_rules!`-style - see [docs/macros.md](docs/macros.md))
- Basic module system
- Methods and `impl` blocks
- Limited generics (monomorphization)

## Detailed Design Documents

### Language Features
- [Operators and Cost Model](docs/operators.md) - Integer operators with hardware-aware design
- [Control Flow Structures](docs/control-flow.md) - If/else, loops, break, continue, return
- [Pointers and Memory Model](docs/pointers-memory.md) - Near/far pointers, addressing modes, memory layout
- [Type System](docs/type-system.md) - Type checking, conversions, and mode-aware types
- [B Register](docs/b-register.md) - Hidden accumulator high byte (m8 mode only)
- [Array Bounds Checking](docs/array-bounds-checking.md) - Design rationale for no bounds checking
- [Calling Convention](docs/calling-convention.md) - ABI, parameter passing, register preservation
- [Mode Transitions](docs/mode-transition-analysis.md) - Mode transition strategies and optimization
- [Interrupt Handling](docs/interrupt-mode-transition.md) - Interrupt handler mode transitions
- [Register Allocation](docs/register-allocation.md) - Register allocation strategy
- [Reserved Keywords](docs/reserved-keywords.md) - Language keyword and register name reference
- [Macros](docs/macros.md) - Simplified Rust-style macro system (planned)

### Code Generation
- [Code Generation](docs/code-generation.md) - Complete code generation reference: memory allocation, register allocation, instruction selection, addressing modes, function generation, and WLA-DX assembly output
- [Struct Array Indexing](docs/struct-array-indexing.md) - Optimization strategies for array[index].field access patterns

## References

- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)
- [Rust Compiler Architecture](https://rustc-dev-guide.rust-lang.org/)


*Last Updated: 2026-01-06*
*STATUS: Design Complete, Implementation Pending*
