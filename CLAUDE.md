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
// Basic mode specification (transition=none by default)
#[mode(m8, x8)]  // 8-bit accumulator and index registers
fn process_byte() {
    // Compiler knows A, X, Y are all u8 here
    // No automatic mode transitions - for disassembled/convention-based code
}

#[mode(m16, x16)]  // 16-bit mode
fn process_word() {
    // Compiler knows A, X, Y are all u16 here
}

#[mode(x16)]  // 16-bit index mode, accumulator mode is undetermined
fn process_word() {
    // Compiler knows X and Y are both u16 here
}

// Automatic callee wrapper (for new safe code)
#[mode(m16, x16, transition=auto)]
fn safe_routine() {
    // Compiler generates wrapper in callee: PHP, REP #$30, ..., PLP
    // Safe regardless of caller's mode
}

// Caller-side mode transitions
#[mode(m16, x16, transition=caller)]
fn needs_16bit() {
    // Compiler generates mode switch at call site (in caller)
    // Callee assumes correct mode, no wrapper
    A = 0x1234;
}

// No mode annotation - mode unknown
fn legacy_routine() {
    // Compiler doesn't know A, X, Y sizes
    // Some code
}

// Mode changes are mixed, indeterminate modes, or modes not defined at beginning of routine then no [mode] attribute generated.
fn process_mixed() {
    // Some mixed code
}
```

**Mode transition options:**
- `transition=none` (default): No automatic transitions - assumes correct mode or programmer handles transitions manually with `SEP()` and `REP()` built-in functions
- `transition=auto`: Compiler generates PHP, REP/SEP, ..., PLP wrapper inside callee (callee-side); safe regardless of caller mode
- `transition=caller`: Compiler generates PHP, REP/SEP, ..., PLP at call site (caller-side); callee assumes correct mode; enables batching optimization

**Caller vs Callee transitions - Generated code comparison:**

```rust
// Example functions
#[mode(m16, x16, transition=auto)]
fn callee_side() {
    A = 0x1234;
}

#[mode(m16, x16, transition=caller)]
fn caller_side() {
    A = 0x5678;
}

#[mode(m8, x8)]
fn main() {
    callee_side();  // Callee handles transition
    caller_side();  // Caller handles transition
}
```

**Generated assembly:**

```asm
main:
    ; Call callee_side - no wrapper needed at call site
    JSR callee_side

    ; Call caller_side - compiler generates wrapper at call site
    PHP             ; Save mode
    REP #$30        ; Switch to m16,x16
    JSR caller_side
    PLP             ; Restore mode
    RTS

callee_side:
    ; Callee-side wrapper
    PHP             ; Save mode
    REP #$30        ; Switch to m16,x16
    LDA #$1234      ; Function body
    PLP             ; Restore mode
    RTS

caller_side:
    ; No wrapper - assumes caller set mode
    LDA #$5678      ; Function body
    RTS
```

**When to use each:**
- `transition=none`: Convention-based code, disassembled ROMs, or manual mode handling with `SEP()`/`REP()`; maximum performance
- `transition=auto`: Function called from many different places/modes; self-contained and safe
- `transition=caller`: Multiple calls to same-mode functions; enables batching optimization for performance

**Batching optimization with `transition=caller`:**

```rust
// Multiple 16-bit functions
#[mode(m16, x16, transition=caller)]
fn process_a() { A = A + 1; }

#[mode(m16, x16, transition=caller)]
fn process_b() { A = A * 2; }

#[mode(m16, x16, transition=caller)]
fn process_c() { A = A - 5; }

#[mode(m8, x8)]
fn optimized_caller() {
    // Compiler can batch the mode transition for all three calls
    PHP();          // Save once
    REP(0x30);      // Switch to m16 once

    process_a();    // All three assume m16
    process_b();
    process_c();

    PLP();          // Restore once
}
```

**Generated assembly (optimized):**
```asm
optimized_caller:
    PHP             ; Save mode once
    REP #$30        ; Switch to m16,x16 once

    JSR process_a   ; Simple calls
    JSR process_b
    JSR process_c

    PLP             ; Restore once
    RTS

process_a:
    INC A           ; No wrapper
    RTS

process_b:
    ASL A           ; No wrapper
    RTS

process_c:
    SEC             ; No wrapper
    SBC #$0005
    RTS
```

**Overhead comparison:**
- `transition=auto` for 3 calls: 30 cycles (3× PHP+REP+PLP)
- `transition=caller` batched: 10 cycles (1× PHP+REP+PLP)
- Savings: 20 cycles

```

### Register Aliasing

Register aliases provide named references to hardware registers for better code readability while maintaining zero-cost abstraction:

```rust
// Local variable aliasing
fn process_player() {
    // Create alias 'hitpoints' for A register
    let hitpoints @ A = PLAYER[0].hitpoints;
    // A now contains the hitpoints value
    // 'hitpoints' is just another name for A

    hitpoints = hitpoints - 1;  // Same as: A = A - 1

    PLAYER[0].hitpoints = hitpoints;  // Write back from A
}

// Multiple aliases
fn calculate() {
    let x_coord @ X = entity.x;  // X register
    let y_coord @ Y = entity.y;  // Y register

    x_coord = x_coord + 1;  // Modifies X
    y_coord = y_coord + 1;  // Modifies Y
}

// Aliases are true register references, not copies
fn alias_demo() {
    let value @ A = 10;
    A = 20;
    // 'value' is now 20 (same register)

    value = 30;
    // A is now 30 (same register)
}
```

**Aliasing rules:**
- Syntax: `let name @ register = expression`
- The alias is a true reference to the register, not a copy
- Reading the alias reads the register
- Writing to the alias writes the register
- Multiple aliases to the same register refer to the same value
- Aliases are compile-time only - zero runtime cost

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
- No attribute means no preservation guarantees (all registers may be modified)
- Programmer is responsible for manual save/restore (compiler does not auto-generate)
- Registers in the return signature cannot be in the preserves list
- **Compiler warning**: If a register is in `#[preserves(...)]` but never used/modified in the function body, warn about redundant preservation declaration

**Interaction with returns:**
```rust
// ERROR: Cannot preserve A when returning it
#[preserves(A)]
fn bad_function() -> u8 {
    return A;  // Compiler error: A is modified by return
}

// OK: Preserve non-returned registers
#[preserves(X, Y)]
fn good_function() -> u8 {
    A = 42;
    return A;
}

// OK: Multiple return registers not in preserves
#[preserves(Y)]
fn divide(dividend @ A: u8, divisor @ X: u8) -> (u8, u8) {
    // Calculate quotient and remainder...
    return A, X;  // Y must be preserved
}
```

**Warnings for redundant preservation:**
```rust
// WARNING: X is in preserves list but never used
#[preserves(X, Y)]
fn only_uses_a() -> u8 {
    A = 42;
    return A;
    // Compiler: "Warning: X in #[preserves] but never modified - remove from list"
}

// OK: No warning - Y is actually used
#[preserves(X, Y)]
fn uses_y_temporarily() -> u8 {
    let saved_y = Y;
    Y = 100;
    A = some_calculation();
    Y = saved_y;  // Restored
    return A;
}

// Suggested fix: Remove unused registers from preserves
#[preserves(Y)]  // Only list what's actually needed
fn only_uses_a_fixed() -> u8 {
    A = 42;
    return A;
}
```

### Function Parameters and Register Aliasing

Function parameters can be bound to registers using the same aliasing syntax:

```rust
// Parameters with register aliases
fn add(left @ A: u8, right @ X: u8) -> u8 {
    // 'left' is an alias for A register
    // 'right' is an alias for X register
    return left + right;  // Same as: return A + X
}

// Parameters without aliases are stack parameters
fn process(value: u8, count: u8) {
    // 'value' and 'count' are on the stack
    A = value;  // Load from stack to A
    X = count;  // Load from stack to X
}

// Mix register and stack parameters
fn hybrid(dividend @ A: u8, divisor: u8) -> u8 {
    // 'dividend' is in A register (zero-cost)
    // 'divisor' is on stack (requires load)
    X = divisor;
    return dividend / X;
}

// Calling functions
fn caller() {
    // Register parameters - direct pass
    A = 10;
    X = 20;
    let sum = add(A, X);

    // Or with expressions (compiler loads into registers)
    let sum = add(5, 15);  // Compiler: A=5, X=15, then call

    // Stack parameters
    let result = process(100, 200);  // Pushed to stack
}
```

**Parameter rules:**
- Parameters with `@ register` syntax are register aliases (zero-cost)
- Parameters with `@ VARIABLE` syntax are bound to existing static variables (zero-cost)
- Parameters without `@` are stack parameters (storage cost)
- Register parameters provide named access to hardware registers
- Variable-bound parameters provide named access to existing global/static variables
- Stack parameters use callee cleanup convention (callee removes from stack before return)
- Mix all types based on performance needs

**Variable-bound parameters:**

Parameters can be bound to existing static variables (zero-page, RAM, or hardware registers):

```rust
// Declare static variables
#[zeropage(0x12)]
static mut COUNTER: u8 = 0;

#[zeropage(0x13)]
static mut FLAGS: u8 = 0;

#[ram]
static mut BUFFER_SIZE: u16;

#[hw(0x2100)]
static mut INIDISP: u8;

// Bind parameters to existing variables
fn process(
    counter @ COUNTER: u8,      // Bound to zero-page variable
    flags @ FLAGS: u8,          // Bound to zero-page variable
    size @ BUFFER_SIZE: u16,    // Bound to RAM variable
    brightness @ INIDISP: u8    // Bound to hardware register
) -> u8 {
    // 'counter' is an alias for COUNTER
    // 'flags' is an alias for FLAGS
    // etc.
    return counter + flags;
}

fn caller() {
    // Caller writes to the variables
    COUNTER = 10;
    FLAGS = 20;
    BUFFER_SIZE = 256;
    INIDISP = 0x0F;

    // Call function - variables already set
    let result = process(COUNTER, FLAGS, BUFFER_SIZE, INIDISP);
}
```

**Generated assembly:**
```asm
; Variables already defined
COUNTER = $12           ; Zero-page
FLAGS = $13             ; Zero-page
BUFFER_SIZE = $7E0100   ; RAM
INIDISP = $2100         ; Hardware register

caller:
    LDA #10
    STA COUNTER         ; Write to zero-page (3 cycles)
    LDA #20
    STA FLAGS

    LDA #$00
    STA BUFFER_SIZE     ; Write to RAM (4 cycles)
    LDA #$01
    STA BUFFER_SIZE+1

    LDA #$0F
    STA INIDISP         ; Write to hardware register

    JSR process
    ; Result in A

process:
    LDA COUNTER         ; Read from zero-page (3 cycles)
    CLC
    ADC FLAGS
    RTS                 ; No cleanup needed - variables are persistent
```

**Benefits of variable-bound parameters:**
- **Zero-cost**: No stack manipulation, no parameter passing overhead
- **Fast**: Direct memory access (especially zero-page: 3 cycles)
- **Clear**: Function signature shows exactly which globals are used
- **Flexible**: Works with zero-page, RAM, and hardware registers
- **No cleanup**: Variables persist, no stack cleanup needed

**Stack parameter calling convention:**

```rust
// Function with stack parameters
fn process(a: u8, b: u8, c: u8) -> u8 {
    // a, b, c are stack parameters
    let result = a + b + c;
    return result;
}

fn caller() {
    let val = process(10, 20, 30);
}
```

**Generated assembly (callee cleanup):**
```asm
caller:
    ; Caller pushes parameters (right-to-left)
    LDA #30
    PHA         ; Push c
    LDA #20
    PHA         ; Push b
    LDA #10
    PHA         ; Push a

    JSR process ; Call - no cleanup needed

    ; Result in A, stack already cleaned by callee
    STA val

process:
    ; Stack layout after JSR:
    ; SP+1, SP+2: Return address (pushed by JSR)
    ; SP+3: Parameter a
    ; SP+4: Parameter b
    ; SP+5: Parameter c

    TSX
    LDA $0104,X ; Load a (SP+3 in absolute addressing)
    STA temp_a
    LDA $0105,X ; Load b
    STA temp_b
    LDA $0106,X ; Load c

    ; ... function body ...

    ; Callee cleanup: remove 3 parameters from stack
    PLA         ; Remove a
    PLA         ; Remove b
    PLA         ; Remove c

    RTS         ; Return with stack cleaned
```

**Why callee cleanup:**
- Cleanup code appears only once (in callee)
- Smaller code size (important for ROM)
- More efficient than caller cleanup
- Standard pattern for 6502/65816 assembly

**Complete parameter example - mixing all types:**

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

#[ram]
static mut GLOBAL_COUNT: u16;

// Function using all parameter types
fn complex_function(
    // Register parameters (fastest - 2 cycles)
    status @ A: u8,
    index @ X: u8,

    // Variable-bound parameters (very fast - 3-4 cycles)
    temp @ TEMP: u8,
    count @ GLOBAL_COUNT: u16,

    // Stack parameters (slower - 7+ cycles, but reentrant)
    extra1: u8,
    extra2: u8
) -> u8 {
    // Use all parameters
    let result = status + index + temp + (count as u8) + extra1 + extra2;
    return result;
}

fn caller() {
    // Set up variable-bound parameters
    TEMP = 5;
    GLOBAL_COUNT = 100;

    // Call with mixed parameter types
    A = 10;                    // Register param
    X = 20;                    // Register param
    let result = complex_function(A, X, TEMP, GLOBAL_COUNT, 30, 40);
}
```

**Parameter type decision guide:**

| Parameter Type | When to Use | Speed | Reentrant | Cleanup |
|---------------|-------------|-------|-----------|---------|
| Register `@ A/X/Y` | Critical path, up to 3 params | Fastest (2 cycles) | Yes | None |
| Variable `@ VAR` | Frequently used, non-reentrant OK | Very fast (3-4 cycles) | No | None |
| Stack `param` | Many parameters, needs reentrancy | Slow (7+ cycles) | Yes | Callee |

### Cross-Bank Function Calls

The `#[bank]` attribute controls function placement and calling convention:

```rust
// Normal function - uses JSR/RTS (same bank)
#[bank(0)]
fn local_function() {
    // Called with JSR, returns with RTS
}

// Long call function - uses JSL/RTL (cross-bank capable)
#[bank(1, long_call)]  // Default: data_bank=none
fn sound_engine() {
    // Called with JSL, returns with RTL
    // DBR not changed - programmer handles if needed
}

// Long call with automatic DBR management
#[bank(1, long_call, data_bank=auto)]
fn graphics_code() {
    // Called with JSL
    // Callee automatically sets DBR to bank 1
    // Callee restores DBR before returning

    // Can safely access data in bank 1
    let data = GRAPHICS_BUFFER;
}

// Long call with caller-managed DBR
#[bank(2, long_call, data_bank=caller)]
fn decompression_routine() {
    // Called with JSL
    // Caller sets DBR before call
    // Caller restores DBR after call
}

fn caller() {
    local_function();           // JSR (same bank)
    sound_engine();             // JSL (no DBR change)
    graphics_code();            // JSL (callee manages DBR)
    decompression_routine();    // JSL (caller manages DBR)
}
```

**Generated assembly:**

```asm
caller:
    ; Local function - same bank
    JSR local_function    ; JSR/RTS

    ; Long call - no DBR management
    JSL sound_engine      ; JSL/RTL, DBR unchanged

    ; Long call - callee manages DBR (data_bank=auto)
    JSL graphics_code     ; JSL/RTL, function handles DBR

    ; Long call - caller manages DBR (data_bank=caller)
    PHB                   ; Save current DBR
    LDA #$02              ; Bank 2
    PHA
    PLB                   ; Set DBR to bank 2
    JSL decompression_routine
    PLB                   ; Restore DBR

    RTS

local_function:
    ; ... code ...
    RTS                   ; Normal return

sound_engine:
    ; ... code ...
    RTL                   ; Long return, DBR unchanged

graphics_code:
    ; Auto DBR management
    PHB                   ; Save DBR
    PHK                   ; Push Program Bank (K register)
    PLB                   ; Set DBR = Program Bank (bank 1)

    ; ... code that accesses bank 1 data ...

    PLB                   ; Restore original DBR
    RTL                   ; Long return

decompression_routine:
    ; Caller manages DBR
    ; Assumes DBR already set to bank 2
    ; ... code ...
    RTL                   ; Long return
```

**Data Bank Register options:**

- `data_bank=none` (default): No DBR management, programmer handles manually if needed
- `data_bank=auto`: Callee automatically sets DBR to its program bank and restores before return
- `data_bank=caller`: Caller sets DBR before call and restores after call

**When to use each:**

| data_bank | Use Case | Overhead |
|-----------|----------|----------|
| `none` | Function doesn't access data in its bank, or uses long addressing | Zero |
| `auto` | Function accesses data in its own bank, called from multiple places | 6 cycles (PHB, PHK, PLB, ..., PLB) |
| `caller` | Caller knows required bank, multiple calls to same bank | Varies (one-time per batch) |

**Bank attribute syntax:**

```rust
#[bank(n)]                                    // long_call=false (default), uses JSR/RTS
#[bank(n, long_call)]                         // long_call=true, uses JSL/RTL, data_bank=none (default)
#[bank(n, long_call, data_bank=auto)]         // JSL/RTL, callee manages DBR
#[bank(n, long_call, data_bank=caller)]       // JSL/RTL, caller manages DBR
```

**Defaults:**
- `long_call`: `false` (uses JSR/RTS)
- `data_bank`: `none` (only applies when `long_call=true`)

**Batching optimization with `data_bank=caller`:**

```rust
#[bank(1, long_call, data_bank=caller)]
fn sound_init() { }

#[bank(1, long_call, data_bank=caller)]
fn sound_update() { }

fn game_init() {
    // Set DBR once for multiple calls to bank 1
    set_data_bank(1);
    sound_init();
    sound_update();
    restore_data_bank();
}
```

### Interrupt Handlers

Interrupt handlers are special functions called by hardware events. They use the `#[interrupt(vector)]` attribute:

```rust
// V-Blank interrupt (fires every frame on SNES)
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNT = FRAME_COUNT + 1;
    // Compiler auto-generates register preservation and RTI
}

// IRQ interrupt (timers, H-blank, etc.)
#[interrupt(irq)]
fn irq_handler() {
    // Handle IRQ event
}

// Software interrupt (BRK instruction)
#[interrupt(brk)]
fn break_handler() {
    // Handle BRK
}

// Coprocessor interrupt (COP instruction)
#[interrupt(cop)]
fn cop_handler() {
    // Handle COP
}

// Abort interrupt
#[interrupt(abort)]
fn abort_handler() {
    // Handle ABORT
}
```

**Available interrupt vectors:**
- `nmi` - Non-Maskable Interrupt (V-Blank on SNES) - Vector at `$FFEA`
- `irq` - Interrupt Request (maskable) - Vector at `$FFEE`
- `brk` - Software interrupt - Vector at `$FFE6`
- `cop` - Coprocessor instruction - Vector at `$FFE4`
- `abort` - Abort interrupt - Vector at `$FFE8`

**Interrupt handler rules:**
- Must use `#[interrupt(vector)]` or `#[interrupt(vector, preserve=true/false)]` attribute
- `preserve` parameter controls automatic register preservation (defaults to `true`)
- If `#[preserves(...)]` attribute is present, `preserve` is implicitly set to `false`
- Cannot have parameters (called by hardware, not by code)
- Cannot have return values (use RTI, not RTS)
- When `preserve=true`: Compiler auto-generates `PHP, PHA, PHX, PHY, PHD, PHB` at entry and `PLB, PLD, PLY, PLX, PLA, PLP` before RTI
- When `preserve=false`: No automatic preservation - programmer must handle it manually
- Compiler always generates `RTI` (Return from Interrupt) instead of `RTS`
- Compiler automatically populates interrupt vector table with handler address
- Only one handler allowed per interrupt vector (compiler error if duplicate)

**Generated assembly example:**
```rust
#[interrupt(nmi)]
fn vblank() {
    A = A + 1;
}
```

Generates:
```asm
vblank:
    PHP             ; Auto-generated: Push Status
    PHA             ; Auto-generated: Push A
    PHX             ; Auto-generated: Push X
    PHY             ; Auto-generated: Push Y
    PHD             ; Auto-generated: Push Direct Page
    PHB             ; Auto-generated: Push Data Bank

    INC A           ; Function body

    PLB             ; Auto-generated: Restore Data Bank
    PLD             ; Auto-generated: Restore Direct Page
    PLY             ; Auto-generated: Restore Y
    PLX             ; Auto-generated: Restore X
    PLA             ; Auto-generated: Restore A
    PLP             ; Auto-generated: Restore Status
    RTI             ; Auto-generated: Return from Interrupt

; Vector table (auto-generated)
.org $FFEA
.dw vblank
```

**Automatic preservation options:**

```rust
// Default: Automatic preservation (preserve=true)
#[interrupt(nmi)]
fn auto_handler() {
    A = A + 1;
    // Compiler auto-generates all preservation
}

// Explicit automatic preservation
#[interrupt(nmi, preserve=true)]
fn explicit_auto() {
    A = A + 1;
    // Same as default - compiler auto-generates preservation
}

// Manual preservation - advanced usage
#[interrupt(nmi, preserve=false)]
fn manual_handler() {
    // Programmer must manually preserve what's needed
    // Use this for performance-critical handlers
    A = A + 1;
}

// Using #[preserves] implies preserve=false
#[interrupt(irq)]  // preserve=false is implicit
#[preserves(X, Y, Status, D, DBR)]  // Only preserve these
fn selective_handler() {
    // Programmer controls preservation via #[preserves]
    let saved_x = X;
    let saved_y = Y;

    A = compute_value();  // Can modify A freely

    Y = saved_y;
    X = saved_x;
}
```

**Why automatic preservation is the default:**

Interrupt handlers default to automatic preservation (`preserve=true`) because:

1. Interrupts can fire at **any time**, interrupting any code
2. **Must** preserve all processor state (requirement for correctness)
3. Preservation code is identical boilerplate for every interrupt
4. Missing preservation is **catastrophic** (corrupts interrupted code state)
5. Safe default for beginners - advanced users can opt out with `preserve=false`

### Function Return Values

**Functions implicitly return the A register value** unless an explicit `return` statement specifies otherwise:

```rust
// No explicit return - implicitly returns A
#[mode(m8, x8)]
fn read_status() {
    A = HVBJOY;
    // Implicitly: return A
}

// Explicit register returns
#[mode(m8, x8)]
fn get_value() -> u8 {
    A = 42;
    return A;  // Return via A register
}

#[mode(m8, x8)]
fn get_x_value() -> u8 {
    X = 100;
    return X;  // Return via X register
}

// Multiple register return (tuple)
#[mode(m8, x8)]
fn divide(dividend: u8, divisor: u8) -> (u8, u8) {
    A = quotient;
    X = remainder;
    return A, X;  // Return both A and X
}

// Local variable return - uses stack
#[mode(m8, x8)]
fn calculate() -> u8 {
    let result = 42;
    return result;  // Returned via stack (not a register)
}

// Caller usage
fn caller() {
    let status = read_status();      // Gets A
    let val = get_x_value();         // Gets X
    let (q, r) = divide(10, 3);      // Gets A and X
    A, X = divide(10, 3);            // Signature matches. Caller does not have to handle returned values
    X, A = divide(10, 3);            // Signature register set matches but register order does not. Throw compilation error.
    let calc = calculate();          // Gets value from stack. Caller is responsible for decrementing stack pointer when it is done with value
}
```

**Return conventions:**
- No `return` statement: A register value returned implicitly
- `return A`, `return X`, `return Y`: Return via specific hardware register
- `return A, X`: Return multiple registers as tuple
- `return variable`: Return local variable via stack
- Never type (`!`): Function never returns (infinite loops, handlers)

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
- ✅ Functions with parameters and return types
- ✅ Register aliasing: `let name @ A = expr` for named register access
- ✅ Hybrid function parameters: register aliases (`param @ A`), variable-bound (`param @ VAR`), or stack values (`param`)
- ✅ Register preservation: `#[preserves(A, X, Y, Status, D, DBR)]` to declare preservation guarantees
- ✅ Interrupt handlers: `#[interrupt(nmi/irq/brk/cop/abort)]` with automatic register preservation and RTI
- ✅ Control flow: `if/else, loop, break, continue, return`
- ✅ All arithmetic, logical, bitwise, and comparison operators
- ✅ `let` bindings (immutable by default, `let mut` for mutable)
- ✅ Global registers A, X, Y, Status
- ✅ Storage attributes: `#[zeropage]`, `#[ram]`, `#[rom]`, `#[hw]`
- ✅ Mode annotations: `#[mode(m8/m16, x8/x16)]` with optional `transition=none/auto/caller`
- ✅ Built-in mode control: `SEP()` and `REP()` functions for manual mode transitions
- ✅ Bank management: `#[bank(n, long_call, data_bank=none/auto/caller)]` for function placement and cross-bank calls with JSL/RTL

## What's Omitted (Too Complex or Incompatible)

- ❌ Lifetimes and borrowing
- ❌ Traits and generics
- ❌ Advanced enums (keep simple C-style enums only)
- ❌ Closures
- ❌ Async/await
- ❌ Macros (maybe add `include_bytes!()` later)
- ❌ Pattern matching (initially - can add later)
- ❌ String types (`String`, `&str`)
- ❌ Dynamic collections (`Vec`, `HashMap`)
- ❌ Modules (initially - single file programs)
- ❌ Methods and `impl` blocks (initially - use free functions)
- ❌ `unsafe` keyword (all code has direct hardware access)

## Example Program

```rust
// Hardware register declarations
#[hw(0x2100)]
static mut INIDISP: u8;

#[hw(0x4212)]
static HVBJOY: u8;

// Direct page variable
#[zeropage(0x20)]
static mut FRAME_COUNT: u16 = 0;

#[zeropage(0x22)]
static mut VBLANK_FLAG: u8 = 0;

// V-Blank interrupt handler
#[interrupt(nmi)]
#[mode(m8, x8)]
fn vblank_handler() {
    FRAME_COUNT = FRAME_COUNT + 1;
    VBLANK_FLAG = 1;
    // Compiler auto-generates register preservation and RTI
}

#[mode(m8, x8)]
#[preserves(X, Y)]  // Only modifies A
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
#[preserves(X, Y)]  // Only modifies A
fn update_brightness(frame @ A: u16) {
    // 'frame' is an alias for A register (parameter)
    let brightness @ A = frame as u8;
    INIDISP = brightness;
}

#[entry]
#[mode(m8, x8)]
fn main() -> ! {
    INIDISP = 0x0F;  // Set screen brightness

    loop {
        wait_vblank();

        // Pass via A register
        update_brightness(FRAME_COUNT);
    }
}
```

## Example: Reverse Engineering ROM with Mixed Conventions

```rust
// Disassembled SNES ROM with various calling conventions

// ============= Main Game Loop (Convention: m8,x8) =============

#[mode(m8, x8)]
fn game_main_loop() {
    init_ppu();          // Calls m8,x8 function
    init_controller();   // Calls m8,x8 function

    loop {
        read_controller();
        update_game_logic();
        copy_sprites_to_oam();  // Calls m16,x16 function - compiler warns
        wait_vblank();
    }
}

#[mode(m8, x8)]
#[preserves(X, Y)]
fn read_controller() {
    A = JOYPAD1;
    CONTROLLER_STATE = A;
}

// ============= DMA Routines (Manual Mode Switching) =============

#[mode(m16, x16, transition=manual)]
fn copy_sprites_to_oam() {
    // Original disassembled code had manual mode switching
    PHP();           // Save processor status
    REP(0x30);       // Switch to 16-bit A and X

    A = 0x0220;      // 544 bytes to copy
    X = SPRITE_BUFFER_ADDR;
    Y = 0x0000;      // OAM starts at $0000

    mvn(0x7E, 0x00); // Block move from WRAM to OAM

    PLP();           // Restore processor status
}

// ============= PPU Initialization (Convention: m8,x8) =============

#[mode(m8, x8)]
fn init_ppu() {
    A = 0x80;
    INIDISP = A;     // Force blank

    A = 0x00;
    OBSEL = A;       // 8x8 and 16x16 sprites, CHR at $0000

    // ... more initialization ...
}

// ============= New Feature You're Adding (Safe Auto Mode) =============

#[mode(m8, x8, transition=auto)]
fn my_new_feature() {
    // New code with automatic safety
    // Compiler generates PHP/REP/SEP/PLP wrapper
    // Safe to call from anywhere

    check_collision();
    update_score();
}

#[mode(m8, x8)]
#[preserves(X, Y)]
fn check_collision() -> u8 {
    // Your new game logic
    return A;
}

// ============= Interrupt Handlers =============

#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNTER = FRAME_COUNTER + 1;
    VBLANK_FLAG = 1;
}
```

**Compiling this ROM hack:**
```bash
65rustic build rom_hack.65r -o rom_hack.asm
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
3. **Register aliasing**: `let name @ A = expr` creates zero-cost aliases for registers; improves readability without runtime overhead
4. **Hybrid parameters**: Three parameter types: register aliases (`param @ A`), variable-bound (`param @ VAR` for existing static variables), or stack values (`param` with callee cleanup)
5. **Explicit preservation**: `#[preserves(...)]` declares register preservation contract; compiler enforces but programmer implements; no automatic save/restore
6. **Interrupt preservation**: `#[interrupt(vector)]` defaults to automatic register preservation (`preserve=true`); can be disabled with `preserve=false` or `#[preserves(...)]` for manual control
7. **Implicit A return**: Functions without explicit `return` statements return A register value
8. **Explicit register returns**: `return X`, `return Y`, `return A, X` return via hardware registers; local variables returned via stack
9. **Storage attributes**: Memory location separate from type (`near<T>` can be in zero-page or RAM)
10. **Flexible mode handling**: `#[mode(...)]` with three transition strategies: `none` (convention-based, default), `auto` (callee wrapper), `caller` (caller-side wrapper with batching)
11. **Automatic initialization**: `__init_start()` generated for non-zero static initializers
12. **Cross-bank calls**: `#[bank(n, long_call)]` uses JSL/RTL; optional `data_bank` parameter controls Data Bank Register management (none/auto/caller)

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
