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
STATUS: u8  // Processor status flags (NVMXDIZC)
D: u16      // Direct Page register (zero-page base)
DBR: u8     // Data Bank Register (default data bank)
PBR: u8     // Program Bank Register (read-only)
S: u16      // Stack Pointer

// Basic usage
A = 0x0F;
X = A;
D = 0x2000;              // Change zero-page base
DBR = 0x7E;              // Set data bank
let bank = PBR;          // Read current bank (write = error)
```

**Rules:**
- All registers mutable except `PBR` (read-only, write is compile error)
- `A`, `X`, `Y` types change with processor mode (u8/u16)
- `D`, `S` always u16; `STATUS`, `DBR`, `PBR` always u8
- All usable in aliasing (`let name @ D = expr`) and `#[preserves(...)]` (except `PBR`)
- **Safety**: Modifying `D`, `DBR`, `S` without restoration causes bugs/crashes

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

### Error Handling

No built-in error handling patterns - programmer's responsibility:

```rust
// Pattern 1: Return codes
fn read_controller() -> u8 {
    let value @ A = CONTROLLER;
    if value == 0xFF {
        return 0;  // Error indicator
    }
    return value;
}

// Pattern 2: Global error flag
#[zeropage]
static mut ERROR_CODE: u8 = 0;

fn parse_data() -> u8 {
    if /* error condition */ {
        ERROR_CODE = 1;
        return 0;
    }
    ERROR_CODE = 0;
    return result;
}

// Pattern 3: Multiple return values
fn divide(a: u8, b: u8) -> (u8, u8) {
    if b == 0 {
        return 0, 1;  // result=0, error=1
    }
    return a / b, 0;  // result, error=0
}
```

**Error handling rules:**
- No `Result`, `Option`, or exception types
- No built-in `panic!()` or `assert!()` macros
- Programmer defines their own error conventions
- Common patterns: return codes, error flags, sentinel values
- Hardware-level control and responsibility

### Type Conversions and Casting

All conversions require explicit `as` keyword. No implicit conversions.

```rust
// Widening (zero/sign extend)
let y: u16 = (x: u8) as u16;  // 0x00FF (zero-extend)
let y: i16 = (x: i8) as i16;  // 0xFFFF if negative (sign-extend)

// Narrowing (truncate)
let y: u8 = (x: u16) as u8;   // Keep low byte only

// Reinterpret (zero cost)
let y: i8 = (x: u8) as i8;    // Same bits, different type

// Boolean conversions
let x: u8 = true as u8;       // Normalize to 0 or 1
let b: bool = (x: u8) as bool;  // Any non-zero = true (unstrict)
```

**Context-aware code generation**: Compiler chooses between memory-based (LDA/STA) or mode-switching (REP/SEP) strategies based on value location, current mode, and subsequent operations. Batches mode changes when beneficial.

**Rules**: Unsigned widening zero-extends; signed widening sign-extends; narrowing truncates (no overflow check); same-size reinterprets bits (zero cost); no pointer conversions.

*(See [docs/type-system.md](docs/type-system.md) for detailed type system semantics)*

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
```

**Rules**: All structs packed (no padding); fields in declaration order; size = sum of field sizes; use `.` for field access; nested/array access supported; no methods (use free functions).

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
#[zeropage(0x42)]
static mut TEMP: u8 = 0;

// Regular RAM - slower (4-5 cycles)
#[ram]
static mut BUFFER: [u8; 256] = [0; 256];

// ROM data - read-only
#[rom(0x8000)]
static GRAPHICS: [u8; 4096] = include_bytes!("gfx.bin");

// Hardware registers - memory-mapped I/O
#[hw(0x2100)]
static mut INIDISP: u8;  // Screen brightness register
```

### Processor Mode Annotations

Functions specify 8-bit or 16-bit modes with optional transition handling:

```rust
#[mode(m8, x8)]  // 8-bit accumulator and index registers
fn process_byte() { }

#[mode(m16, x16)]  // 16-bit mode
fn process_word() { }

#[mode(x16)]  // 16-bit index mode only
fn mixed_mode() { }

#[mode(m16, x16, transition=auto)]  // Callee wrapper (safe)
fn safe_routine() { }

#[mode(m16, x16, transition=caller)]  // Caller wrapper (allows batching)
fn needs_16bit() { }
```

**Mode transition options:**
- `transition=none` (default): No automatic transitions - programmer handles with `SEP()`/`REP()` or relies on calling convention
- `transition=auto`: Compiler generates PHP/REP/SEP/PLP wrapper inside callee; safe regardless of caller mode
- `transition=caller`: Compiler generates wrapper at call site; enables batching optimization when calling multiple same-mode functions

*(See [docs/mode-transition-analysis.md](docs/mode-transition-analysis.md) for mode transition design and [docs/mode-transition-status.md](docs/mode-transition-status.md) for implementation status)*

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

Functions can declare which registers and processor state they preserve using the `#[preserves(...)]` attribute:

```rust
// Function that only modifies A
#[preserves(X, Y)]
fn careful_add(value @ A: u8) -> u8 {
    // Can freely use A (not in preserves list)
    A = A + 1;

    // X and Y must remain unchanged
    // Compiler error if we modify X or Y without saving/restoring
    return A;
}

// Function that preserves everything except what it returns
#[preserves(X, Y, STATUS)]
fn get_input() -> u8 {
    A = CONTROLLER;
    return A;
}

// Function that preserves processor state registers
#[preserves(A, X, Y, STATUS, D, DBR)]
fn safe_helper() {
    // Must manually save/restore all registers if we need to use them
    let saved_a = A;
    A = HWREG;  // Use A temporarily
    A = saved_a;  // Restore before return
}

// No preserves attribute = no guarantees (any register may be modified)
fn wild_function() {
    A = 1;
    X = 2;
    Y = 3;
    // Caller cannot assume any register is preserved
}
```

**Preservation rules:**
- `#[preserves(...)]` lists registers that **must remain unchanged** by function exit
- Available registers: `A, X, Y, STATUS, D, DBR, S`
- `PBR` cannot be in preserves list (it's read-only and cannot be modified)
- Compiler enforces preservation - error if register modified without save/restore
- Programmer is responsible for manual save/restore (compiler does not auto-generate)
- Registers in the return signature cannot be in the preserves list
- **Compiler warning**: If a register is in `#[preserves(...)]` but never modified in the function body

**Example:**
```rust
#[preserves(X, Y)]
fn good_function() -> u8 {
    A = 42;
    return A;  // OK: X and Y not modified, A returned
}
```

*(See [docs/calling-convention.md](docs/calling-convention.md) for complete calling convention and ABI details)*

### Function Parameters

Three parameter types with different performance tradeoffs:

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

fn add(left @ A: u8, right @ X: u8) -> u8 { }      // Register alias (fastest)
fn process(temp @ TEMP: u8) -> u8 { }               // Variable-bound (fast)
fn calculate(a: u8, b: u8) -> u8 { }                // Stack (slower, reentrant)
fn hybrid(a: u8, reg @ A: u8, var @ TEMP: u8) { }   // Mixed (stack first!)
```

**Rules**: `param @ A/X/Y` = register alias; `param @ VAR` = bound to static variable; `param` = stack with callee cleanup. **Ordering**: Stack parameters must precede aliased parameters (compiler error otherwise). **Optimization**: No setup code when arguments already match parameter aliases (e.g., `process(A, TEMP)` when params are `@ A, @ TEMP`).

### Function Pointers

Function pointers encode calling convention in the type. Near (`fn()`) for same-bank, far (`far fn()`) for cross-bank:

```rust
// Function pointer types
type RegCallback = fn(a @ A: u8, b @ X: u8) -> u8;
type StackCallback = fn(a: u8, b: u8) -> u8;
type FarCallback = far fn(a @ A: u8) -> u8;

// State machine example
type Handler = fn(input @ A: u8) -> u8;

#[ram]
static mut CURRENT_HANDLER: Handler;

fn menu_handler(input @ A: u8) -> u8 { return 0; }
fn game_handler(input @ A: u8) -> u8 { return 1; }

fn init() {
    CURRENT_HANDLER = menu_handler;
}

fn update(input @ A: u8) {
    let action = CURRENT_HANDLER(input);  // Indirect call via trampoline
    if action == 1 {
        CURRENT_HANDLER = game_handler;
    }
}
```

**Rules:**
- Type system enforces matching calling conventions
- Compiler generates trampolines for indirect calls (JMP through pointer)
- Useful for callbacks, state machines, dispatch tables

### Cross-Bank Function Calls

The `far` keyword indicates JSL/RTL calling convention, while `#[bank]` controls placement:

```rust
fn local_function() { }                 // JSR/RTS (near call, default)

#[bank(1)]
far fn sound_engine() { }               // JSL/RTL, data_bank=none (default)

#[bank(1, data_bank=auto)]
far fn graphics_code() { }              // JSL/RTL, callee manages DBR

#[bank(2, data_bank=caller)]
far fn decompression_routine() { }     // JSL/RTL, caller manages DBR
```

**Calling conventions:**
- `fn()`: Near call using JSR/RTS (16-bit address, same bank)
- `far fn()`: Far call using JSL/RTL (24-bit address, cross-bank)

**Data Bank Register (DBR) options:**
- `data_bank=none` (default): No DBR management - programmer handles manually
- `data_bank=auto`: Callee sets/restores DBR to its program bank
- `data_bank=caller`: Caller sets/restores DBR (enables batching multiple calls)

*(Function parameters, pointers, and cross-bank calls detailed in [docs/calling-convention.md](docs/calling-convention.md))*

### Interrupt Handlers

Interrupt handlers use `#[interrupt(vector)]` attribute. Available vectors: `nmi`, `irq`, `brk`, `cop`, `abort`.

```rust
// Automatic preservation (default)
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNT = FRAME_COUNT + 1;
    // Compiler auto-generates: PHP, PHA, PHX, PHY, PHD, PHB at entry
    // and PLB, PLD, PLY, PLX, PLA, PLP, RTI at exit
}

// Manual control
#[interrupt(irq, preserve=false)]
fn manual_handler() {
    // Programmer handles preservation
}

// Using #[preserves] implies preserve=false
#[interrupt(irq)]
#[preserves(X, Y, STATUS, D, DBR)]
fn selective_handler() {
    let saved_x = X;
    A = compute_value();  // Can modify A freely
    X = saved_x;
}
```

**Rules:**
- `preserve=true` (default): Compiler auto-generates all register preservation and RTI
- `preserve=false`: Manual preservation for performance-critical handlers
- If `#[preserves(...)]` present, `preserve` is implicitly `false`
- Compiler populates interrupt vector table automatically

*(See [docs/interrupt-mode-transition.md](docs/interrupt-mode-transition.md) for interrupt handler mode transitions)*

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

The compiler automatically generates an `__init_start()` routine for all static variables with explicit initializers:

```rust
#[zeropage]
static mut FLAGS: u8 = 0x80;     // Explicit initializer (non-zero)

#[zeropage]
static mut COUNTER: u8 = 0;      // Explicit initializer (zero)

static mut LIVES: u8;      // No initializer - undefined value!

fn main() {
    // __init_start() automatically called here
}
```

Generated assembly:
```asm
__init_start:
    LDA #$80
    STA FLAGS
    LDA #$00
    STA COUNTER
    RTS

main:
    JSR __init_start
    ; ... rest of main
```

**Important:** On SNES hardware, RAM contents at power-on are **unpredictable**. Variables without explicit initializers have undefined values. Always initialize variables that need known starting values, even if initializing to zero.

### Operators and Hardware Cost Model

R65 uses a **hardware-aware operator design** where the syntax clearly indicates performance cost:

**Operators = Fast (hardware instructions or simple sequences)**
```rust
let sum = a + b;          // ADC instruction (2-4 cycles)
let diff = a - b;         // SBC instruction (2-4 cycles)
let x = a * 8;            // ASL, ASL, ASL (6 cycles)
let x = a / 4;            // LSR, LSR (4 cycles)
let x = a << 3;           // Constant shift (6 cycles)
let masked = a & 0x0F;    // AND instruction (2-4 cycles)
```

**Functions = Slow (subroutine calls)**
```rust
let product = mul(a, b);     // JSR __mul (20-100+ cycles)
let quotient = div(a, b);    // JSR __div (50-200+ cycles)
let remainder = mod(a, b);   // JSR __mod (50-200+ cycles)
let shifted = shl(a, n);     // Variable shift loop (8+ cycles)
```

**Operator Restrictions:**
- `*` operator: Only allows constants 1, 2, 4, or 8 (optimized to shifts)
- `/` operator: Only allows constants 1, 2, 4, or 8 (optimized to shifts)
- `<<`, `>>` operators: Only allow constant shift amounts
- General multiplication/division requires `mul()`, `div()` functions
- Variable shifts require `shl()`, `shr()` functions

**Examples:**
```rust
// Valid - uses operators (fast)
let x = a * 8;        // Compiles to 3 shift instructions
let y = b / 2;        // Compiles to 1 shift instruction
let z = c << 4;       // Constant shift

// Invalid - must use functions
let x = a * 3;        // ERROR: Use mul(a, 3)
let y = b / 7;        // ERROR: Use div(b, 7)
let z = c << count;   // ERROR: Use shl(c, count)

// Correct usage with functions
let x = mul(a, 3);       // General multiplication
let y = div(b, 7);       // General division
let z = shl(c, count);   // Variable shift
```

This design makes the programmer **immediately aware** when they're using expensive operations, encouraging optimization-conscious coding.

*(See [docs/operators.md](docs/operators.md) for complete operator semantics and implementation)*

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
- ✅ `let` bindings (immutable by default, `let mut` for mutable)
- ✅ All 65816 processor registers: A, X, Y, STATUS, D, DBR, S (mutable); PBR (read-only)
- ✅ Storage attributes: `#[zeropage]`, `#[ram]`, `#[rom]`, `#[hw]`
- ✅ Mode annotations: `#[mode(m8/m16, x8/x16)]` with optional `transition=none/auto/caller`
- ✅ Built-in mode control: `SEP()` and `REP()` functions for manual mode transitions
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
- ❌ Macros (maybe add `include_bytes!()` later)
- ❌ Pattern matching (initially - can add later)
- ❌ String types (`String`, `&str`)
- ❌ Dynamic collections (`Vec`, `HashMap`)
- ❌ Module system (`mod`, `pub`, visibility, namespacing)
- ❌ Methods and `impl` blocks (initially - use free functions)
- ❌ `unsafe` keyword (all code has direct hardware access)
- ❌ Bounds checking (compile-time or runtime)

## Example Program

```rust
// Hardware registers
#[hw(0x2100)] static mut INIDISP: u8;
#[hw(0x4200)] static mut JOYPAD1: u8;

// Direct page variables
#[zeropage(0x20)] static mut FRAME_COUNT: u16 = 0;
#[zeropage(0x22)] static mut VBLANK_FLAG: u8 = 0;

// Interrupt handler with automatic preservation
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNT = FRAME_COUNT + 1;
    VBLANK_FLAG = 1;
}

// Functions with register preservation
#[mode(m8, x8)]
#[preserves(X, Y)]
fn wait_vblank() {
    loop {
        let flag @ A = VBLANK_FLAG;
        if flag != 0 {
            VBLANK_FLAG = 0;
            break;
        }
    }
}

#[mode(m8, x8)]
#[preserves(X, Y)]
fn update_brightness(frame @ A: u16) {
    let brightness @ A = frame as u8;
    INIDISP = brightness;
}

// Entry point
#[entry]
#[mode(m8, x8)]
fn main() -> ! {
    INIDISP = 0x0F;
    loop {
        wait_vblank();
        update_brightness(FRAME_COUNT);
    }
}
```

## Compiler Architecture

### Pipeline

```
Source (.r65) → Lexer → Parser → AST → HIR → Type Checking → MIR →
Code Generation → WLA-DX Assembly (.asm)
```

### Phases

1. **Lexer**: Tokenize source code, recognize register keywords - See [docs/parser-complete.md](docs/parser-complete.md)
2. **Parser**: Build AST with special nodes for register operations - See [docs/parser-complete.md](docs/parser-complete.md)
3. **HIR (High-level IR)**: Desugar syntax, resolve names, process attributes
4. **Type Checking**: Validate types, modes, register usage, bank boundaries - See [docs/type-system.md](docs/type-system.md)
5. **MIR (Mid-level IR)**: CFG construction, virtual registers - See [docs/mir-implementation-status.md](docs/mir-implementation-status.md)
6. **Optimization**: Constant propagation, dead code elimination, zero-page allocation
7. **Code Generation**: Register allocation, addressing mode selection, WLA-DX emission - See [docs/codegen-*.md](docs/) for detailed phases

## Implementation STATUS

**Current**: Planning complete, no code written yet

**Phase 1 (MVP)**: Basic compiler - lexer, parser, simple codegen
**Phase 2**: Type system with mode checking
**Phase 3**: MIR and optimization
**Phase 4**: Full hardware features (banks, DMA, interrupts)
**Phase 5**: Standard library

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
2. **Direct Page**: Very fast - special addressing mode (2-3 cycles)
3. **RAM**: Slower - absolute addressing (4-5 cycles)
4. **ROM**: Read-only data (4-5 cycles)
5. **Hardware Registers**: Memory-mapped I/O (4-6 cycles)

## Target Platform

- **CPU**: 65816 (16-bit extension of 6502)
- **Primary Target**: Super Nintendo Entertainment System (SNES)
- **Assembler**: WLA-DX
- **ROM Format**: LoROM or HiROM

## Key Technical Decisions

1. **No `unsafe` keyword**: All code has direct hardware access by default
2. **All processor registers exposed**: A, X, Y, STATUS, D, DBR, S exposed as mutable global variables; PBR exposed as read-only global (write is compile error)
3. **Automatic volatile**: All `#[hw]` variables are automatically volatile; every access goes to hardware, no caching or reordering
4. **No bounds checking**: Arrays have no compile-time or runtime bounds checking; programmer responsible for safety
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
17. **Explicit preservation**: `#[preserves(...)]` declares register preservation contract; compiler enforces but programmer implements; no automatic save/restore
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

- Tuples for multiple returns
- Pattern matching on integers
- Basic module system
- Methods and `impl` blocks
- Limited generics (monomorphization)

## Detailed Design Documents

### Language Features
- [Operators and Cost Model](docs/operators.md) - Integer operators with hardware-aware design
- [Control Flow Structures](docs/control-flow.md) - If/else, loops, break, continue, return
- [Pointers and Memory Model](docs/pointers-memory.md) - Near/far pointers, addressing modes, memory layout
- [Type System](docs/type-system.md) - Type checking, conversions, and mode-aware types
- [Array Bounds Checking](docs/array-bounds-checking.md) - Design rationale for no bounds checking
- [Calling Convention](docs/calling-convention.md) - ABI, parameter passing, register preservation
- [Mode Transitions](docs/mode-transition-analysis.md) - Mode transition strategies and optimization
- [Interrupt Handling](docs/interrupt-mode-transition.md) - Interrupt handler mode transitions
- [Register Allocation](docs/register-allocation.md) - Register allocation strategy
- [Reserved Keywords](docs/reserved-keywords.md) - Language keyword reference
- [Register Case Sensitivity](docs/register-case-sensitivity.md) - Register naming conventions

### Implementation Status
- [Parser Implementation](docs/parser-complete.md) - Parser status and AST structure
- [Parser Named Attributes](docs/parser-named-attributes.md) - Attribute syntax implementation
- [MIR Implementation](docs/mir-implementation-status.md) - Mid-level IR implementation status
- [Mode Transition Status](docs/mode-transition-status.md) - Mode transition implementation status
- [Documentation Update Status](docs/documentation-update-STATUS.md) - Documentation maintenance tracking

### Code Generation
- [Code Generation Overview](docs/codegen-assembly.md) - Assembly generation strategy
- [Implementation Plan](docs/codegen-implementation-plan.md) - Overall codegen architecture
- [Phase 2: Memory Allocation](docs/codegen-phase2-memory-allocation.md) - Memory layout and allocation
- [Phase 3: Register Allocation](docs/codegen-phase3-register-allocation.md) - Register allocation pass
- [Phase 4: Instruction Selection](docs/codegen-phase4-instruction-selection.md) - Instruction selection
- [Phase 5: Addressing Modes](docs/codegen-phase5-addressing-mode.md) - Addressing mode selection
- [Phase 6: Function Generation](docs/codegen-phase6-function-generation.md) - Function code generation
- [Phase 7: Program Assembly](docs/codegen-phase7-program-assembly.md) - Final program assembly
- [Phase 7 Advanced Features](docs/phase-7-advanced-features.md) - Advanced codegen features

## References

- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)
- [Rust Compiler Architecture](https://rustc-dev-guide.rust-lang.org/)

## License

[To be determined]

## Contributors

[To be determined]

---

*Last Updated: 2025-12-31*
*STATUS: Design Complete, Implementation Pending*
