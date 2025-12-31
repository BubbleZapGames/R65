# 65Rustic Compiler

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

The 6502/65816 hardware registers are exposed as global mutable variables:

```rust
// Built-in global registers (always in scope)
A: u8       // Accumulator (u16 in m16 mode)
X: u8       // X index register (u16 in x16 mode)
Y: u8       // Y index register (u16 in x16 mode)
Status: u8  // Processor status register (NVMXDIZC flags)

// Direct usage
A = 0x0F;
X = A;
Y = X + 1;
```

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

All type conversions require explicit `as` keyword. No implicit conversions.

**Supported types:** `u8`, `i8`, `u16`, `i16`, `bool`

**Conversion strategy:** Context-aware code generation - compiler chooses optimal method based on:
- Value location (register vs memory)
- Current processor mode
- Subsequent operations (stay in converted mode?)
- Batching opportunities

#### Basic Conversion Rules

```rust
// 1. Widening: u8 → u16 (zero-extend)
let x: u8 = 255;
let y: u16 = x as u16;  // 0x00FF

// 2. Widening: i8 → i16 (sign-extend)
let x: i8 = -1;
let y: i16 = x as i16;  // 0xFFFF

// 3. Narrowing: u16 → u8 (truncate)
let x: u16 = 0x1234;
let y: u8 = x as u8;    // 0x34

// 4. Narrowing: i16 → i8 (truncate)
let x: i16 = -300;
let y: i8 = x as i8;    // -44

// 5. Reinterpret: u8 ↔ i8 (same size, zero cost)
let x: u8 = 255;
let y: i8 = x as i8;    // -1

// 6. Reinterpret: u16 ↔ i16 (same size, zero cost)
let x: u16 = 65535;
let y: i16 = x as i16;  // -1

// 7. bool → integer (normalize to 0 or 1)
let flag: bool = true;
let x: u8 = flag as u8; // 1

// 8. integer → bool (unstrict: any non-zero = true)
let x: u8 = 5;
let b: bool = x as bool; // true (stored as 5)
```

#### Context-Aware Code Generation

**Strategy 1: Memory-to-Memory (simple cases)**
```rust
#[zeropage(0x20)]
static mut X: u8;
#[ram]
static mut Y: u16;

fn convert() {
    Y = X as u16;  // Simple memory-to-memory
}
```
```asm
; u8 → u16: Zero-extend
LDA $20         ; Load X
STA Y           ; Store low byte
STZ Y+1         ; Zero high byte
; 3 instructions, ~11 cycles
```

**Strategy 2: REP/SEP (value in A, staying in mode)**
```rust
#[mode(m8, x8)]
fn calculate(input @ A: u8) -> u16 {
    let wide = input as u16;  // A already has value
    let result = wide + 100;  // More 16-bit work
    return result;            // Returning u16
}
```
```asm
; Value in A, doing more 16-bit work
REP #$20        ; Switch to m16 mode
AND #$00FF      ; Zero-extend A
CLC
ADC #$0064      ; Add 100 (16-bit)
; Stay in m16, return via A
; No SEP needed - caller handles mode
; 3 instructions, ~12 cycles
```

**Strategy 3: Batched Mode Changes**
```rust
#[mode(m8, x8)]
fn process() {
    let a: u8 = 10;
    let b: u16 = a as u16;
    let c: u16 = b * 2;
    let d: u16 = c + 50;
    let e: u8 = d as u8;
}
```
```asm
; Batch conversions - single mode switch
.A8
LDA #$0A        ; a = 10
REP #$20        ; Switch to m16 once
AND #$00FF      ; Zero-extend to u16
ASL A           ; Multiply by 2
CLC
ADC #$0032      ; Add 50
; Truncate back to u8
SEP #$20        ; Switch to m8 once
STA result
; 2 mode switches total vs 4 if done individually
```

**Strategy 4: Sign Extension (i8 → i16)**
```rust
let x: i8 = -50;
let y: i16 = x as i16;
```

*Method A: Branch-based (memory)*
```asm
LDA x           ; Load signed byte
STA y           ; Store low byte
BPL positive    ; Test sign bit
LDA #$FF        ; Negative: high = FF
BRA store_hi
positive:
LDA #$00        ; Positive: high = 00
store_hi:
STA y+1
; 6-7 instructions, ~15-20 cycles, variable timing
```

*Method B: REP-based (A register, more work ahead)*
```asm
LDA x           ; Load signed byte
REP #$20        ; Switch to m16
AND #$00FF      ; Clear high byte
BIT #$0080      ; Test sign bit
BEQ positive
ORA #$FF00      ; Extend sign
positive:
; A now sign-extended, stay in m16 for more ops
; 5 instructions, ~17 cycles, consistent timing
```

**Compiler chooses based on:**
- Branch method: Memory-to-memory, single conversion
- REP method: Value in A, subsequent 16-bit operations

#### Conversion Rules Summary

- **No implicit conversions** - always require `as`
- **Unsigned widening**: Zero-extend high byte(s)
- **Signed widening**: Sign-extend high byte(s)
- **Narrowing**: Truncate (take low byte), no overflow check
- **Same-size**: Reinterpret bits (zero cost)
- **bool → integer**: Normalize to 0 or 1
- **integer → bool**: Store as-is (unstrict bool)
- **Pointer conversions**: Not supported

#### Cost Analysis

| Conversion | Memory Method | REP/SEP Method | Best For |
|------------|---------------|----------------|----------|
| u8 → u16 | 6 bytes, ~11 cycles | 6 bytes, ~6 cycles (stay) | REP if in A + more work |
| i8 → i16 | 10-13 bytes, ~15-20 cycles | ~20 bytes, ~17 cycles | Branch for simple, REP for consistent timing |
| u16 → u8 | 4 bytes, ~7 cycles | 6 bytes, ~9 cycles | Memory always |
| Same-size | 0-4 bytes, 0-7 cycles | N/A | Type-only or simple move |
| bool ↔ int | 6-8 bytes, ~10-14 cycles | Similar | Context-dependent |

### File Inclusion

No module system, but textual file inclusion is supported:

```rust
// main.65r
include!("hardware.65r")  // Insert contents of hardware.65r here
include!("player.65r")    // Insert contents of player.65r here

fn main() {
    init_hardware();
    update_player();
}

// hardware.65r
#[hw(0x2100)]
static mut INIDISP: u8;

fn init_hardware() {
    INIDISP = 0x0F;
}

// player.65r
#[zeropage(0x20)]
static mut PLAYER_X: u8;

fn update_player() {
    PLAYER_X = PLAYER_X + 1;
}
```

**File inclusion rules:**
- `include!("path")` performs textual inclusion (like C `#include`)
- Path is relative to the including file
- All included code shares the same namespace (no modules)
- All items are visible to all code (no privacy/visibility)
- Circular includes are an error
- Include happens at compile time (no runtime cost)

**Use cases:**
- Organize hardware register definitions in separate files
- Split large programs across multiple files
- Share common code between projects
- Keep ROM data tables in separate files

```rust
// Typical organization
include!("snes_hardware.65r")  // Hardware register definitions
include!("constants.65r")      // Game constants
include!("player.65r")         // Player code
include!("enemy.65r")          // Enemy code
include!("graphics.65r")       // Graphics routines
```

**No module system:**
- No `mod`, `pub`, or visibility modifiers
- No namespacing (everything in global scope)
- Programmer responsible for avoiding name collisions
- Simple and predictable

### Inline Assembly

Embed raw 65816 assembly instructions directly:

```rust
// Single instruction
fn wait_for_interrupt() {
    asm!("WAI");
}

// Multiple instructions
fn save_and_wait() {
    asm!(
        "PHP"
        "WAI"
    );
}

// Complex assembly blocks
fn custom_dma_transfer() {
    asm!(
        "LDA #$80"
        "STA $2100"
        "LDA #$01"
        "STA $420B"
        "NOP"
        "NOP"
    );
}
```

**Inline assembly rules:**
- Two forms: `asm!("instruction")` for single line, `asm!("line1" "line2" ...)` for multiple
- No commas between instruction strings (string concatenation)
- Raw assembly emitted verbatim to output
- No variable interpolation or references to Rust variables
- No explicit clobber lists - compiler assumes all registers may be modified
- Use for special instructions, precise timing, or hardware-specific operations

**Interaction with compiler:**
- Compiler treats inline asm as black box
- All registers considered potentially clobbered
- No optimization across asm boundaries
- Programmer responsible for preserving registers if needed

```rust
// Example: Manual register preservation
#[preserves(X, Y)]
fn careful_asm() {
    asm!(
        "PHX"           // Save X
        "PHY"           // Save Y
        "LDX #$00"      // Use X
        "LDY #$00"      // Use Y
        "; custom code here"
        "PLY"           // Restore Y
        "PLX"           // Restore X
    );
}
```

### Const Evaluation

Compile-time constant evaluation is supported for expressions:

```rust
// Const variables with computed values
const SCREEN_WIDTH: u16 = 256;
const SCREEN_HEIGHT: u16 = 224;
const TILE_SIZE: u8 = 8;

// Arithmetic operations
const TILES_PER_ROW: u16 = SCREEN_WIDTH / TILE_SIZE;  // 32
const TILES_PER_COL: u16 = SCREEN_HEIGHT / TILE_SIZE; // 28
const TOTAL_TILES: u16 = TILES_PER_ROW * TILES_PER_COL; // 896

// Bitwise and logical operations
const BIT_MASK: u8 = 0x80 | 0x40;  // 0xC0
const FLAG_ENABLED: bool = true;

// Use in array sizes
static TILE_BUFFER: [u8; TOTAL_TILES] = [0; TOTAL_TILES];

// Use in attributes
const PLAYER_ADDR: u16 = 0x7E0000;
#[zeropage(TILE_SIZE * 2)]  // Address 0x10
static mut TEMP_DATA: u16;
```

**Const evaluation rules:**
- Arithmetic, bitwise, and logical operations on constants
- Type casts between numeric types
- Can be used for array sizes and attribute parameters
- **No const functions** - only expressions are evaluated
- All evaluation happens at compile time (zero runtime cost)

```rust
// NOT SUPPORTED - const functions
const fn calculate_offset(x: u8, y: u8) -> u16 {  // Error!
    return (y as u16) * 32 + (x as u16);
}
```

### Enums

C-style enums with explicit or auto-increment values:

```rust
// Explicit values
enum Direction {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

// Auto-increment (starts at 0)
enum State {
    Idle,      // 0
    Running,   // 1
    Jumping,   // 2
}

// Mixed explicit and auto-increment
enum Priority {
    Low = 0,
    Medium,    // 1
    High,      // 2
    Critical = 10,
    Urgent,    // 11
}

// Usage
let dir: Direction = Direction::North;
let state: State = State::Running;

// Comparison
if dir == Direction::North {
    // ...
}

// In structs and arrays
struct Entity {
    direction: Direction,
    state: State,
}

static MOVE_TABLE: [Direction; 4] = [
    Direction::North,
    Direction::East,
    Direction::South,
    Direction::West,
];
```

**Enum rules:**
- C-style enums only (no data-carrying variants)
- Values are compile-time constants
- Auto-increment starts at 0, or from previous value + 1
- No explicit underlying type specification (compiler infers smallest type)
- Access variants with `EnumName::Variant` syntax
- Comparable with `==` and `!=`
- Can cast to/from integers with `as`

```rust
// Casting between enum and integer
let dir: Direction = Direction::East;
let value: u8 = dir as u8;          // 1
let back: Direction = value as Direction;  // Direction::East

// Can be used in const contexts
const DEFAULT_DIR: Direction = Direction::North;
```

### Structs

Structs are packed by default with no alignment padding:

```rust
struct Player {
    x: u8,      // Offset 0
    y: u8,      // Offset 1
    health: u16, // Offset 2-3 (no alignment, packed)
    flags: u8,  // Offset 4
}  // Total size: 5 bytes

// Structs can be placed in different storage classes
#[zeropage(0x30)]
static mut PLAYER: Player;

#[ram]
static mut ENEMIES: [Player; 8];

#[rom(0x8000)]
static PLAYER_INITIAL: Player = Player {
    x: 10,
    y: 20,
    health: 100,
    flags: 0,
};
```

**Struct layout rules:**
- All structs are packed (no padding between fields)
- No alignment requirements
- Fields are laid out in declaration order
- Total size is the sum of all field sizes
- Matches hand-optimized assembly data structures

**Struct field access:**

Use `.` operator to access struct fields:

```rust
struct Point {
    x: u8,
    y: u8,
}

struct Entity {
    pos: Point,
    health: u8,
}

fn example() {
    // Read fields
    let mut p: Point;
    p.x = 10;           // Write field
    p.y = 20;           // Write field
    let x = p.x;        // Read field

    // Nested structs
    let mut e: Entity;
    e.pos.x = 5;        // Nested field access
    e.pos.y = 10;
    e.health = 100;

    // Structs in arrays
    static mut ENEMIES: [Entity; 8];
    ENEMIES[0].pos.x = 50;
    ENEMIES[0].health = 100;
}
```

**Field access rules:**
- Use `.` operator for field access
- Works with nested structs (e.g., `entity.pos.x`)
- Works with arrays of structs (e.g., `array[0].field`)
- Fields can be read or written if variable is mutable
- No methods on structs (use free functions)

### Volatile Semantics

All hardware register variables are automatically volatile - every access goes directly to hardware:

```rust
#[hw(0x4212)]
static mut HVBJOY: u8;  // SNES hardware status register

// Polling loop - each read accesses hardware
loop {
    let status = HVBJOY;  // Always reads from $4212
    if status & 0x01 != 0 { break; }
}

// Every write goes to hardware
#[hw(0x2100)]
static mut INIDISP: u8;

INIDISP = 0x80;  // Write 1: Force blank
INIDISP = 0x0F;  // Write 2: Both writes execute (not optimized away)
```

**Volatile rules for `#[hw]` variables:**
- Every read must access hardware (no caching in registers)
- Every write must go to hardware (no elimination of "redundant" writes)
- Accesses cannot be reordered relative to other `#[hw]` accesses
- Compiler treats each access as having side effects
- Critical for memory-mapped I/O, polling loops, and timing-sensitive code

```rust
// This works correctly (not optimized)
#[hw(0x2100)]
static mut INIDISP: u8;
#[hw(0x2101)]
static mut OBSEL: u8;

fn setup_ppu() {
    INIDISP = 0x80;  // Force blank first
    OBSEL = 0x03;    // Configure sprites
    INIDISP = 0x0F;  // Unblank
    // All three writes execute in order
}
```

**Non-hardware volatile memory:**
Regular variables (`#[zeropage]`, `#[ram]`, `#[rom]`) are not volatile - the compiler may optimize accesses. For volatile non-hardware memory (DMA buffers, shared memory), use `#[hw]` attribute even if not technically a hardware register, or ensure proper barriers with `asm!()`.

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
#[preserves(X, Y, Status)]
fn get_input() -> u8 {
    A = CONTROLLER;
    return A;
}

// Function that preserves processor state registers
#[preserves(A, X, Y, Status, D, DBR)]
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
- Available registers: `A, X, Y, Status, D, DBR`
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

### Function Parameters

Three parameter types support different performance/flexibility tradeoffs:

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

// Register parameters (@ A/X/Y) - fastest, zero-cost
fn add(left @ A: u8, right @ X: u8) -> u8 {
    return left + right;
}

// Variable-bound parameters (@ VAR) - fast, zero-cost
fn process(temp @ TEMP: u8, flags @ FLAGS: u8) -> u8 {
    return temp + flags;
}

// Stack parameters (no @) - slower, reentrant
fn calculate(a: u8, b: u8, c: u8) -> u8 {
    return a + b + c;
}

// Mix all types
fn hybrid(
    status @ A: u8,        // Register (2 cycles)
    temp @ TEMP: u8,       // Variable-bound (3-4 cycles)
    extra: u8              // Stack (7+ cycles)
) -> u8 {
    return status + temp + extra;
}
```

**Parameter types:**
- `param @ A/X/Y`: Register alias (fastest, zero-cost)
- `param @ VAR`: Bound to existing static variable (fast, zero-cost, shows dependencies)
- `param`: Stack parameter with callee cleanup (slower, reentrant)

**Parameter ordering rules:**
- Stack parameters (no `@`) must come before aliased parameters (`@ register` or `@ variable`)
- Mixing parameter types requires stack parameters first, otherwise compiler error

```rust
// VALID: Stack parameters first, then aliased parameters
fn valid(a: u8, b: u8, reg @ A: u8, temp @ TEMP: u8) -> u8 { }

// INVALID: Aliased parameter before stack parameter
fn invalid(reg @ A: u8, a: u8) -> u8 { }  // Compiler error!
```

**Argument alias optimization:**

When calling functions with aliased parameters, the compiler skips parameter setup if the arguments already match the parameter aliases:

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

fn process(input @ A: u8, temp @ TEMP: u8) -> u8 {
    return input + temp;
}

fn caller() {
    // Case 1: Arguments already in correct locations - zero setup cost
    A = 5;
    TEMP = 10;
    let result = process(A, TEMP);  // No code generated for parameter passing

    // Case 2: Arguments in different locations - setup required
    X = 7;
    Y = 3;
    let result = process(X, Y);     // Generates: TXA, STY TEMP, then JSR

    // Case 3: Mixed - partial setup
    A = 5;
    Y = 10;
    let result = process(A, Y);     // Generates: STY TEMP, then JSR
}
```

**Optimization details:**
- If argument expression matches parameter alias exactly, no setup code generated
- If argument is in different register/variable, compiler generates transfer/store
- Works with all alias types: registers (`@ A/X/Y`) and variables (`@ VAR`)
- Enables zero-cost calling conventions when arguments are pre-positioned
- Programmer can manually arrange data for optimal performance

### Function Pointers

Function pointers encode calling convention in the type. Near (`fn()`) for same-bank, far (`far fn()`) for cross-bank:

```rust
// Function pointer types
type RegCallback = fn(a @ A: u8, b @ X: u8) -> u8;
type VarCallback = fn(a @ TEMP1: u8, b @ TEMP2: u8) -> u8;
type StackCallback = fn(a: u8, b: u8) -> u8;
type FarCallback = far fn(a @ A: u8) -> u8;

// Static function pointer (state machine example)
#[ram]
static mut CURRENT_HANDLER: InputHandler;

fn init() {
    CURRENT_HANDLER = menu_handler;
}

fn update() {
    let action = CURRENT_HANDLER(INPUT_STATE);  // Indirect call via trampoline
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
#[preserves(X, Y, Status, D, DBR)]
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
A = count - 1;
X = src_addr;
Y = dest_addr;
mvn(src_bank, dest_bank);  // Move forward
mvp(src_bank, dest_bank);  // Move backward

// Processor mode control
SEP(flags: u8);  // Set processor status bits (e.g., SEP(0x30) for m8,x8)
REP(flags: u8);  // Reset processor status bits (e.g., REP(0x30) for m16,x16)

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

### Variable Initialization

The compiler automatically generates an `__init_start()` routine for non-zero initializers:

```rust
#[zeropage]
static mut FLAGS: u8 = 0x80;     // Non-zero initializer

#[zeropage]
static mut COUNTER: u8 = 0;      // Zero initializer

static mut LIVES: u8;      // Not assigned - no init needed

fn main() {
    // __init_start() automatically called here
}
```

Generated assembly:
```asm
__init_start:
    LDA #$80
    STA FLAGS
    RTS

main:
    JSR __init_start
    ; ... rest of main
```

## What's Included (Minimal Feature Set)

- ✅ Basic types: `u8, i8, u16, i16, bool`
- ✅ Fixed-size arrays: `[T; N]`
- ✅ Structs (no methods initially)
- ✅ C-style enums: Explicit or auto-increment values; no data-carrying variants
- ✅ Functions with parameters and return types
- ✅ Register aliasing: `let name @ A = expr` for named register access
- ✅ Hybrid function parameters: register aliases (`param @ A`), variable-bound (`param @ VAR`), or stack values (`param`)
- ✅ Function pointers: `fn()` (near) and `far fn()` (cross-bank) with calling convention encoded in type
- ✅ Register preservation: `#[preserves(A, X, Y, Status, D, DBR)]` to declare preservation guarantees
- ✅ Interrupt handlers: `#[interrupt(nmi/irq/brk/cop/abort)]` with automatic register preservation and RTI
- ✅ Control flow: `if/else, loop, break, continue, return`
- ✅ All arithmetic, logical, bitwise, and comparison operators
- ✅ `let` bindings (immutable by default, `let mut` for mutable)
- ✅ Global registers A, X, Y, Status
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
Source (.65r) → Lexer → Parser → AST → HIR → Type Checking → MIR →
Code Generation → WLA-DX Assembly (.asm)
```

### Phases

1. **Lexer**: Tokenize source code, recognize register keywords
2. **Parser**: Build AST with special nodes for register operations
3. **HIR (High-level IR)**: Desugar syntax, resolve names, process attributes
4. **Type Checking**: Validate types, modes, register usage, bank boundaries
5. **MIR (Mid-level IR)**: CFG construction, virtual registers
6. **Optimization**: Constant propagation, dead code elimination, zero-page allocation
7. **Code Generation**: Register allocation, addressing mode selection, WLA-DX emission

## Implementation Status

**Current**: Planning complete, no code written yet

**Phase 1 (MVP)**: Basic compiler - lexer, parser, simple codegen
**Phase 2**: Type system with mode checking
**Phase 3**: MIR and optimization
**Phase 4**: Full hardware features (banks, DMA, interrupts)
**Phase 5**: Standard library

## Directory Structure (Planned)

```
/home/nathan/65rustic/
├── compiler/
│   ├── main.py              # CLI entry point
│   ├── frontend/            # Lexer, parser, AST
│   ├── hir/                 # High-level IR
│   ├── typeck/              # Type checking, mode checking
│   ├── mir/                 # Mid-level IR
│   ├── optimize/            # Optimization passes
│   ├── codegen/             # Code generation
│   ├── builtins/            # Built-in functions (mvn, mvp, etc.)
│   └── utils/               # Errors, diagnostics
├── stdlib/                  # Standard library
│   └── core65/
│       └── hw/              # Hardware register definitions
├── tests/                   # Unit and integration tests
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
2. **Global registers**: A, X, Y, Status exposed as built-in global variables
3. **Automatic volatile**: All `#[hw]` variables are automatically volatile; every access goes to hardware, no caching or reordering
4. **No bounds checking**: Arrays have no compile-time or runtime bounds checking; programmer responsible for safety
5. **No error handling**: No built-in Result, Option, or panic; programmer defines own error conventions
6. **Context-aware type conversions**: Compiler chooses between memory-based and REP/SEP-based conversions for optimal performance; batches mode changes when beneficial
7. **Const expressions only**: Compile-time evaluation of constant expressions supported; const functions not supported
8. **Inline assembly**: `asm!()` for raw assembly with simple string syntax; compiler treats as black box, assumes all registers clobbered
9. **C-style enums**: Simple enums with explicit or auto-increment values; no explicit underlying type; cast to/from integers
10. **File inclusion only**: `include!()` for textual file inclusion (C-style); no module system, visibility, or namespacing
11. **Packed structs**: All structs are packed by default with no alignment padding; fields laid out in declaration order
12. **Register aliasing**: `let name @ A = expr` creates zero-cost aliases for registers; improves readability without runtime overhead
13. **Hybrid parameters**: Three parameter types: register aliases (`param @ A`), variable-bound (`param @ VAR` for existing static variables), or stack values (`param` with callee cleanup)
14. **Parameter ordering**: Stack parameters must precede aliased parameters; compiler error otherwise
15. **Argument alias optimization**: No setup code generated when call arguments already match parameter aliases; enables zero-cost calling conventions
16. **Explicit preservation**: `#[preserves(...)]` declares register preservation contract; compiler enforces but programmer implements; no automatic save/restore
17. **Interrupt preservation**: `#[interrupt(vector)]` defaults to automatic register preservation (`preserve=true`); can be disabled with `preserve=false` or `#[preserves(...)]` for manual control
18. **Implicit A return**: Functions without explicit `return` statements return A register value
19. **Explicit register returns**: `return X`, `return Y`, `return A, X` return via hardware registers; local variables returned via stack
20. **Storage attributes**: Memory location separate from type (`near<T>` can be in zero-page or RAM)
21. **Flexible mode handling**: `#[mode(...)]` with three transition strategies: `none` (convention-based, default), `auto` (callee wrapper), `caller` (caller-side wrapper with batching)
22. **Automatic initialization**: `__init_start()` generated for non-zero static initializers
23. **Consistent far/near**: `far fn()` for both function definitions and pointers indicates JSL/RTL calling convention; `fn()` indicates JSR/RTS; `#[bank(n)]` controls placement with optional `data_bank` parameter

## Use Cases

1. **SNES Game Development**: Write new games with modern syntax and type safety
2. **ROM Reverse Engineering**: Disassemble ROMs into readable source
3. **ROM Hacking**: Modify existing games with better tooling
4. **Education**: Learn 6502/65816 architecture with safer, clearer code

## Future Enhancements

- Simple C-style enums
- Tuples for multiple returns
- Pattern matching on integers
- Basic module system
- Methods and `impl` blocks
- Limited generics (monomorphization)
- Inline assembly support

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

*Last Updated: 2025-12-29*
*Status: Design Complete, Implementation Pending*
