# Calling Convention and ABI Design

## Overview

R65's ABI (Application Binary Interface) is designed to **match hand-written assembly patterns** from existing SNES code while providing type safety and mode tracking. The calling convention is **explicit and flexible**, allowing programmers to replicate any calling pattern found in disassembled ROMs.

**Design Principles**:
- Explicit parameter passing - programmer specifies mechanism
- Match hand-written patterns - replicate common assembly idioms
- Flexible conventions - support multiple parameter styles
- Predictable stack layout - deterministic frame structure
- No hidden overhead - visible costs

---

## Function Signature Types

### Three Parameter Types

```rust
fn example(
    stack_param: u8,           // Stack parameter (callee cleanup)
    reg_param @ A: u8,         // Register alias parameter
    var_param @ TEMP: u8       // Variable-bound parameter
) -> u8 { }
```

**Three distinct mechanisms**:
1. **Stack parameters**: Passed on stack, callee cleans up
2. **Register alias parameters**: Passed in hardware registers (A, X, Y)
3. **Variable-bound parameters**: Passed via specific memory locations

---

## Parameter Passing Conventions

### 1. Stack Parameters

**Syntax**: `param: Type`

**Mechanism**: Pushed by caller, popped by callee

```rust
fn add(a: u8, b: u8) -> u8 {
    return a + b;
}

// Caller:
let result = add(10, 20);
```

**Generated Assembly**:

**Caller**:
```asm
LDA #20        ; Second parameter (pushed first)
PHA
LDA #10        ; First parameter
PHA
JSR add
; Callee cleaned up stack
```

**Callee**:
```asm
add:
    TSC            ; Get stack pointer
    CLC
    ADC #1         ; Skip return address (2 bytes)
    TAX

    LDA $01,X      ; Load 'a' (first parameter)
    STA temp_a
    LDA $03,X      ; Load 'b' (second parameter)
    STA temp_b

    ; ... function body ...

    ; Cleanup: adjust stack pointer
    TSC
    CLC
    ADC #2         ; Remove 2 parameters
    TCS
    RTS
```

**Characteristics**:
- Slowest (stack access is slow on 65816)
- Reentrant (supports recursion)
- Unlimited parameters
- Order: **Right to left** (last parameter pushed first)

**Use when**: Many parameters, need reentrancy, or matching specific stack-based calling convention

---

### 2. Register Alias Parameters

**Syntax**: `param @ Register: Type`

**Mechanism**: Caller places value in specified register

```rust
fn add(a @ A: u8, b @ X: u8) -> u8 {
    let sum @ A = a + b;
    return sum;
}

// Caller:
let value1 @ A = 10;
let value2 @ X = 20;
let result = add(value1, value2);
```

**Generated Assembly**:

**Caller**:
```asm
LDA #10        ; a @ A
LDX #20        ; b @ X
JSR add
; Result in A
```

**Callee**:
```asm
add:
    ; a already in A
    ; b already in X
    STX temp       ; Save X
    LDA temp       ; Load into A (if needed for addition)
    CLC
    ADC ...
    RTS
```

**Characteristics**:
- Fastest (no stack access)
- Limited to 3 registers (A, X, Y) in most modes
- **In m8 mode (default): 4 registers available (A, X, Y, B)**
- Zero overhead if values already in registers
- Order: Any order (explicit)
- **X/Y must be u16** (always x16 mode)

**Use when**: Performance critical, few parameters, matching register-based calling convention

---

### 2a. B Register Parameters (m8 Mode Only)

**In m8 mode** (default, or when A parameter is u8), the B register (high byte of accumulator) can be used for parameters:

**Syntax**: `param @ B: Type`

**Mechanism**: Caller places value in B register (via XBA instruction)

```rust
// m8 mode (default) - B is available
fn pack_word(low @ A: u8, high @ B: u8) -> u16 {
    // Both A and B are available as parameters
    return A as u16 | ((B as u16) << 8);
}

fn process_high_byte(value @ B: u8) -> u8 {
    B = B & 0xF0;
    return B;
}
```

**Generated Assembly**:

**Caller**:
```asm
LDA #$34       ; Load low byte into A
LDX #$12       ; Temporarily store high byte in X
XBA            ; Exchange A and B (A now in B)
TXA            ; Move X to A (high byte)
XBA            ; Exchange again (high byte now in B)
LDA #$34       ; Restore low byte to A
JSR pack_word  ; Call with A=low, B=high
```

**Callee**:
```asm
pack_word:
    ; A contains low byte
    ; B contains high byte (accessible via XBA)
    ; ... function body ...
    RTS
```

**Key Points**:
- **Mode restriction**: B register only available in m8 mode
  - Compiler error if B used with `@ A: u16` (m16 mode)
  - B is meaningless when accumulator is 16-bit
- **Caller responsibility**: Caller must set up B register before call
  - Typically via XBA instruction to exchange A and B
- **Mixed parameters**: B can be combined with A, X, Y
  ```rust
  fn example(a @ A: u8, b @ B: u8, x @ X: u8)  // All valid
  ```
- **Not preserved**: B cannot appear in `#[preserves(...)]` attribute
  - B is part of the same hardware register as A
  - Preserving B without A is meaningless

**Performance**:
- B access via XBA: 3 cycles
- Efficient for byte manipulation patterns
- Common in hand-written 65816 assembly for 16-bit operations

**Use when**: Byte packing/unpacking, 16-bit decomposition, matching hand-written assembly patterns

---

### 3. Variable-Bound Parameters (Direct Page)

**Syntax**: `param @ VARIABLE: Type`

**Mechanism**: Caller writes to specific memory location (typically zero-page)

```rust
#[zeropage(0x10)]
static mut INPUT_A: u8;

#[zeropage(0x11)]
static mut INPUT_B: u8;

fn add(a @ INPUT_A: u8, b @ INPUT_B: u8) -> u8 {
    let sum @ A = a + b;
    return sum;
}

// Caller:
INPUT_A = 10;
INPUT_B = 20;
let result = add(INPUT_A, INPUT_B);
```

**Generated Assembly**:

**Caller**:
```asm
LDA #10
STA $10        ; INPUT_A
LDA #20
STA $11        ; INPUT_B
JSR add
```

**Callee**:
```asm
add:
    LDA $10        ; Load from INPUT_A
    CLC
    ADC $11        ; Add INPUT_B
    RTS
```

**Characteristics**:
- Fast (zero-page is 3-4 cycles)
- Zero overhead when caller has values in those locations
- **Very common in hand-written SNES assembly**
- Global shared communication area

**Use when**: Matching specific memory-based calling convention, shared communication area

---

## Parameter Ordering Rules

### Stack Parameters Must Come First

```rust
// ERROR: Stack parameter after aliased parameter
fn bad(reg @ A: u8, stack: u8) { }

// OK: Stack parameters first
fn good(stack: u8, reg @ A: u8) { }

// OK: All same type
fn all_stack(a: u8, b: u8, c: u8) { }
fn all_register(a @ A: u8, b @ X: u8) { }
```

**Reason**: Stack layout must be determined before register operations

**Compiler error** if stack parameters appear after aliased parameters

---

### Optimization: Zero-Cost Calls

When arguments already match parameter aliases:

```rust
fn process(value @ A: u8, index @ X: u8) { }

fn caller() {
    let v @ A = compute();
    let i @ X = 0;
    process(v, i);  // Zero overhead! Values already in A and X
}
```

**Generated**:
```asm
JSR compute     ; Result in A
LDX #0
JSR process     ; A and X already set!
```

**No setup code** - direct call

---

## Return Value Conventions

### Return Mechanisms

Functions can return values via:
1. **Registers** (A, X, Y)
2. **Zero-page variables** (direct page)
3. **Mixed** (registers + zero-page)

**Critical Rule**: All return paths must have **identical return signatures**

---

### Implicit A Return

Functions without explicit return statement return A:

```rust
fn get_status() -> u8 {
    A = HWREG;
    // Implicitly returns A
}
```

**Generated**:
```asm
get_status:
    LDA HWREG
    RTS            ; A contains return value
```

---

### Explicit Register Return

```rust
fn get_value() -> u8 {
    X = 100;
    return X;
}
```

**Generated**:
```asm
get_value:
    LDX #100
    TXA            ; Transfer X to A (return register)
    RTS
```

**Convention**: Single return value goes in A (transfer if needed)

---

### Multiple Return Values (Registers)

```rust
fn divide(dividend @ A: u8, divisor @ X: u8) -> (u8, u8) {
    // quotient in A, remainder in X
    return (A, X);
}
```

**Convention**:
- First return value: A
- Second return value: X
- Third return value: Y

**Generated**:
```asm
divide:
    ; ... division code ...
    ; Quotient in A, remainder in X
    RTS
```

**Caller**:
```rust
let (q @ A, r @ X) = divide(dividend, divisor);
```

---

### B Register Return Values (m8 Mode Only)

**In `#[mode(m8)]` mode**, B register can be returned alone or with other registers:

```rust
#[mode(m8, x8)]
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // Return B only
}

#[mode(m8, x8)]
fn unpack_word(value: u16) -> (u8, u8) {
    A = value as u8;
    B = (value >> 8) as u8;
    return (A, B);  // Return both A and B
}

#[mode(m8, x8)]
fn swap_bytes(low @ A: u8, high @ B: u8) -> (u8, u8) {
    return (B, A);  // Return B first, A second
}
```

**Return Conventions with B**:

| Return Statement | Meaning | A Status | B Status |
|-----------------|---------|----------|----------|
| `return B;` | Return B only | **Caller must preserve A** | Returns in B |
| `return (A, B);` | Return A first, B second | Returns in A | Returns in B |
| `return (B, A);` | Return B first, A second | Returns in A (but as 2nd value) | Returns in B (as 1st value) |
| `return (B, X);` | Return B and X | **Caller must preserve A** | Returns in B |

**Critical Rule: Caller Responsibility for A**

When a function returns only B (or B with non-A registers), the callee does **not** restore A:

```rust
#[mode(m8, x8)]
fn get_high_byte(value: u16) -> u8 {
    B = (value >> 8) as u8;
    return B;  // A is NOT restored
}

// Caller must preserve A if needed:
#[mode(m8, x8)]
fn caller() {
    let saved_a = A;           // Save A before call
    let high = get_high_byte(0x1234);
    A = saved_a;               // Restore A after call
    // 'high' is in B register
}
```

**Generated Assembly**:

**Return B only**:
```asm
get_high_byte:
    ; ... code that sets B ...
    ; A is NOT saved/restored
    RTS            ; Return with B containing value, A clobbered
```

**Return A and B**:
```asm
unpack_word:
    ; ... code that sets both A and B ...
    RTS            ; Return with both A and B containing values
```

**Caller reading B return**:
```asm
    JSR get_high_byte  ; Call function
    XBA                ; Exchange to get B value into A
    STA result         ; Store the returned value
```

**Key Points**:
- **Caller responsibility**: When B is returned, caller must handle:
  - Preserving A if needed before call
  - Restoring A if needed after call
  - Reading B value (via XBA or direct B access)
- **No implicit A preservation**: Unlike typical calling conventions, returning B doesn't preserve A
- **Explicit contract**: Function signature makes it clear B is being returned
- **Mixed returns allowed**: `return (B, X);` or `return (B, A);` are both valid

**Use when**: Returning high byte, byte unpacking, matching hand-written assembly patterns

---

### Zero-Page Return (Very Common in Hand-Written Assembly)

```rust
#[zeropage(0x10)]
static mut RESULT: u16;

#[zeropage(0x00)]
static mut PARAM: u16;

fn calculate(p @ PARAM: u16) {
    RESULT = p + 100;
    // Returns via RESULT (no explicit return statement)
}

// Caller:
PARAM = 1000;
calculate();
let output = RESULT;  // Read result from zero-page
```

**Generated**:
```asm
calculate:
    LDA $00        ; Load PARAM
    CLC
    ADC #100
    STA $10        ; Store to RESULT
    LDA $01
    ADC #0
    STA $11
    RTS
```

**Common pattern**: Hand-written SNES code heavily uses zero-page for both parameters and returns

---

### Mixed Return (Register + Zero-Page)

```rust
#[zeropage(0x10)]
static mut RESULT: u16;

fn mixed_return() {
    RESULT = 1000;
    A = 42;
    return (X, RESULT);  // Returns X register and RESULT variable
}

// Caller:
mixed_return();
let reg_value @ X = X;     // X contains returned value
let zp_value = RESULT;     // RESULT contains returned value
```

**Flexibility**: Can mix registers and zero-page variables in return

---

## Return Signature Consistency Rule

### Critical Rule: All Return Paths Must Match

**All return statements in a function must return the same registers/variables in the same order.**

### Valid Examples

```rust
// Good: All paths return (X, RESULT)
#[zeropage(0x10)]
static mut RESULT: u16;

fn good(flag: bool) {
    if flag {
        return (X, RESULT);    // Signature: (X, RESULT)
    } else {
        return (X, RESULT);    // Same signature ✓
    }
}

// Good: All paths return A
fn good2(val: u16) -> u8 {
    if val > 100 {
        return A;   // Signature: (A)
    } else {
        return A;   // Same signature ✓
    }
}

// Good: Mixed return, but consistent
fn good3(val: u16) {
    RESULT = 1;
    A = 2;
    if val == 0 {
        return (X, RESULT);
    } else {
        return (X, RESULT);  // Same signature ✓
    }
}
```

---

### Invalid Examples

```rust
#[zeropage(0x10)]
static mut RESULT: u16;

#[zeropage(0x12)]
static mut RESULT2: u16;

// BAD: Different variables returned
fn bad(val: u16) {
    if val == 1 {
        return RESULT;     // Signature: (RESULT)
    } else {
        return RESULT2;    // Signature: (RESULT2) ✗ MISMATCH!
    }
}

// BAD: Different registers returned
fn bad2(val: u16) {
    if val == 1 {
        return A;          // Signature: (A)
    } else {
        return X;          // Signature: (X) ✗ MISMATCH!
    }
}

// BAD: Different number of return values
fn bad3(val: u16) {
    if val == 1 {
        return (A, X);       // Signature: (A, X)
    } else {
        return A;          // Signature: (A) ✗ MISMATCH!
    }
}

// BAD: Different order
fn bad4(val: u16) {
    if val == 1 {
        return (A, X);       // Signature: (A, X)
    } else {
        return (X, A);       // Signature: (X, A) ✗ MISMATCH!
    }
}
```

---

### Compiler Enforcement

The compiler validates return signature consistency:

**Error Message Example**:
```rust
fn inconsistent(val: u16) {
    if val == 1 {
        return RESULT;
    } else {
        return RESULT2;
    }
}
```

**Compiler Error**:
```
error: inconsistent return signature
  --> file.r65:5:16
   |
3  |         return RESULT;
   |                ------ first return has signature: (RESULT)
4  |     } else {
5  |         return RESULT2;
   |                ^^^^^^^ returns (RESULT2), expected (RESULT)
   |
help: all return paths must return the same registers/variables in the same order
```

**Validation Algorithm**:
1. Extract return signature from first return statement
2. Validate all subsequent return statements match exactly
3. Error if any mismatch in registers, variables, or order

---

## Structs and Arrays (Pass by Reference Only)

**Compile Error**: Structs and arrays cannot be passed to functions, returned by value, or directly assigned. This is a deliberate restriction - copying large data structures is expensive on 6502/65816 and the cost should be explicit.

```rust
// ERROR: Cannot pass struct by value
fn bad_param(player: Player) { }

// ERROR: Cannot return struct by value
fn bad_return() -> Player { }

// ERROR: Cannot pass array by value
fn bad_array(data: [u8; 256]) { }

// ERROR: Cannot assign struct by value
PLAYER1 = PLAYER2;

// ERROR: Cannot assign array by value
BUFFER1 = BUFFER2;
```

**Correct patterns** - use pointers or pre-allocated memory:

```rust
// Pass struct by pointer
fn process_player(*player: Player) {
    (*player).health = (*player).health - 1;
}

// Write result to pre-allocated memory
fn init_player(*dest: Player) {
    (*dest).x = 0;
    (*dest).y = 0;
    (*dest).health = 100;
}

// Return pointer to static/global data
fn get_player() -> *Player {
    return &PLAYER;
}

// Copy struct fields individually
PLAYER1.x = PLAYER2.x;
PLAYER1.y = PLAYER2.y;
PLAYER1.health = PLAYER2.health;

// Copy array elements individually (or use mvn/mvp for bulk copy)
BUFFER1[0] = BUFFER2[0];
```

**Rationale**: Explicit pointer usage makes memory operations visible. The programmer can choose between zero-page pointers (fast) or RAM pointers (more available), and the cost of indirection is clear in the code.

---

## Stack Frame Layout

### Stack Organization

```
                  (high address)
    +---------------------------+
    | Parameter N               |  <- SP + (N*param_size) + return_size
    +---------------------------+
    | ...                       |
    +---------------------------+
    | Parameter 2               |  <- SP + (2*param_size) + return_size
    +---------------------------+
    | Parameter 1               |  <- SP + param_size + return_size
    +---------------------------+
    | Return Address (high)     |  <- SP + 2
    +---------------------------+
    | Return Address (low)      |  <- SP + 1
    +---------------------------+
    | Local 1                   |  <- SP
    +---------------------------+
    | Local 2                   |
    +---------------------------+
                  (low address, growing down)
```

**Stack grows downward** (toward lower addresses)

---

### Stack Frame Example

```rust
fn example(a: u8, b: u8) -> u8 {
    let local: u8 = a + b;
    return local;
}
```

**Stack at function entry**:
```
SP+4: b (parameter)
SP+3: a (parameter)
SP+2: return address high
SP+1: return address low
SP:   (empty, will be local)
```

**Function prologue**:
```asm
example:
    TSC
    SEC
    SBC #1         ; Allocate 1 byte for local
    TCS
```

**Function epilogue**:
```asm
    TSC
    CLC
    ADC #3         ; Deallocate local (1) + parameters (2)
    TCS
    RTS
```

---

## Register Preservation

### Caller-Save (Default)

All registers are **caller-save** by default:

```rust
fn caller() {
    let value @ A = 10;

    // Must save A if needed after call
    #[zeropage(0x20)]
    static mut SAVED: u8;

    SAVED = value;
    callee();
    let value @ A = SAVED;

    use_value(value);
}
```

**No automatic preservation** - programmer must save explicitly

---

### Preserves Attribute

Functions declare what they preserve:

```rust
#[preserves(X, Y)]
fn careful(input @ A: u8) -> u8 {
    // Can modify A freely
    // Must preserve X and Y
    return A;
}
```

**Caller advantage**:
```rust
fn caller() {
    let index @ X = 10;
    let result = careful(5);  // X still valid afterward!
    use_index(index);
}
```

**No save/restore needed** for X

---

### Manual Preservation

Programmer must manually save/restore:

```rust
#[preserves(X, Y)]
fn manual_preserve(input @ A: u8) -> u8 {
    let saved_x = X;  // Save X to local variable or memory

    X = 20;           // Temporarily modify X
    // ... use X ...

    X = saved_x;      // Restore X before return
    return A;
}
```

**Compiler validates** but doesn't generate save/restore

---

## Near vs Far Calls

### Near Call (JSR/RTS)

**Syntax**: `fn name() { }`

**Mechanism**: 16-bit address, same bank

```rust
fn local_function() {
    // ...
}

fn caller() {
    local_function();
}
```

**Generated**:
```asm
caller:
    JSR local_function
    ; ...

local_function:
    ; ...
    RTS
```

**Characteristics**:
- 6 cycles (JSR) + 6 cycles (RTS) = 12 cycles overhead
- Same bank only
- Return address: 2 bytes on stack

---

### Far Call (JSL/RTL)

**Syntax**: `far fn name() { }`

**Mechanism**: 24-bit address, cross-bank

**Bank Placement**: `#[bank(n)]` is a global directive that sets the bank context for all following functions and immutable statics (ROM). Use `#[bank(auto)]` for automatic placement (requires `far fn` and `far static` for ROM statics):

```rust
#[bank(1)]
far fn remote_function() {
    // In bank 1
}

far fn another_remote() {
    // Also in bank 1 (inherits from directive)
}

fn caller() {
    remote_function();
}
```

**Generated**:
```asm
caller:
    JSL remote_function
    ; ...

.bank 1
remote_function:
    ; ...
    RTL
```

**Characteristics**:
- 8 cycles (JSL) + 6 cycles (RTL) = 14 cycles overhead
- Cross-bank capable
- Return address: 3 bytes on stack

**Auto-Bank Mode**: Use `#[bank(auto)]` for automatic bank placement:
```rust
#[bank(auto)]
far fn auto_placed() { }      // Automatically placed in available bank space

far static DATA: [u8; 256] = [0; 256];  // Must use 'far static' in auto mode
```

In auto-bank mode:
- Functions must be declared as `far fn` (compile error otherwise)
- ROM statics (immutable) must use `far static` syntax
- RAM statics (`#[ram]`, `#[zeropage]`, `#[lowram]`) are unaffected

---

### Data Bank Register (DBR) Management

Far functions can specify DBR handling via `#[mode(databank=...)]`:

**Option 1: databank=none (default)**
```rust
#[bank(1)]  // Sets bank context for following declarations
far fn no_dbr_change() {
    // Programmer handles DBR manually
}
```

**Option 2: databank=inline**
```rust
#[bank(1)]
#[mode(databank=inline)]
far fn auto_dbr() {
    // Compiler generates DBR save/restore
}
```

**Generated**:
```asm
auto_dbr:
    PHB            ; Save DBR
    LDA #$01
    PHA
    PLB            ; Set DBR = 1
    ; ... function body ...
    PLB            ; Restore DBR
    RTL
```

**Option 3: databank=caller**
```rust
#[bank(1)]
#[mode(databank=caller)]
far fn caller_dbr() {
    // Caller handles DBR
}
```

**Caller generates**:
```asm
caller:
    PHB
    LDA #$01
    PHA
    PLB
    JSL caller_dbr
    PLB
```

---

### Cross-Bank Call Validation

The compiler enforces that near functions can only call near functions in the **same bank**:

**Rule**: JSR uses a 16-bit address and cannot cross bank boundaries. To call a function in a different bank, the callee must be declared as `far fn`.

```rust
#[bank(0)]
fn bank0_caller() {
    same_bank_helper();  // OK: same bank
    far_other_bank();    // OK: far function
    // near_other_bank(); // ERROR: near function in different bank
}

fn same_bank_helper() { }  // Bank 0 (inherits)

#[bank(1)]
fn near_other_bank() { }   // Bank 1 - CANNOT be called from bank 0
far fn far_other_bank() { } // Bank 1 - CAN be called from anywhere
```

**Compile Error**:
```
cannot call near function 'near_other_bank' from bank 0: 'near_other_bank' is in bank 1
hint: near functions use JSR which cannot cross bank boundaries; declare 'near_other_bank' as 'far fn' to allow cross-bank calls
```

**Summary**:
| Caller Bank | Callee Bank | Callee Type | Allowed? |
|-------------|-------------|-------------|----------|
| 0 | 0 | `fn` | Yes |
| 0 | 1 | `fn` | **No** (compile error) |
| 0 | 1 | `far fn` | Yes |
| 1 | 0 | `fn` | **No** (compile error) |
| Any | Any | `far fn` | Yes |

---

## Function Pointers

### Function Pointer Types

```rust
type NearFunc = fn(u8) -> u8;        // JSR/RTS
type FarFunc = far fn(u8) -> u8;     // JSL/RTL
```

**Type system** enforces near vs far

---

### Indirect Call Trampoline

Calling through function pointer requires trampoline:

```rust
type Callback = fn(input @ A: u8) -> u8;

#[ram]
static mut HANDLER: Callback;

fn caller() {
    let result @ A = HANDLER(10);
}
```

**Generated**:
```asm
caller:
    LDA #10        ; Setup parameter

    ; Trampoline for indirect JSR
    LDA HANDLER+1  ; High byte of address
    PHA
    LDA HANDLER    ; Low byte
    PHA
    RTS            ; Jump to handler (uses return address)

    ; Handler returns here
```

**Alternative (using jump vector)**:
```asm
caller:
    LDA #10
    JSR call_trampoline

call_trampoline:
    JMP (HANDLER)  ; Indirect jump
```

---

## Calling Convention Examples

### Example 1: Pure Register Parameters

**Hand-written Assembly**:
```asm
multiply:
    ; A = multiplicand
    ; X = multiplier
    ; Returns A = result
    STX temp
    ; ... multiply A by temp ...
    RTS
```

**R65**:
```rust
fn multiply(a @ A: u8, b @ X: u8) -> u8 {
    let result @ A = mul(a, b);
    return result;
}
```

**Match**: Exact parameter and return convention

---

### Example 2: Zero-Page Parameters and Return

**Hand-written Assembly**:
```asm
; Input: $10 = x, $11 = y
; Output: $20 = result
calculate:
    LDA $10
    CLC
    ADC $11
    STA $20
    RTS
```

**R65**:
```rust
#[zeropage(0x10)]
static mut PARAM_X: u8;

#[zeropage(0x11)]
static mut PARAM_Y: u8;

#[zeropage(0x20)]
static mut RESULT: u8;

fn calculate(x @ PARAM_X: u8, y @ PARAM_Y: u8) {
    RESULT = x + y;
}
```

---

### Example 3: Mixed Parameters

**Hand-written Assembly**:
```asm
; Stack: count
; A = base_value
process:
    TSC
    TAX
    LDA $03,X      ; Load count from stack
    ; ... process with A (base_value) and stack param ...

    ; Cleanup stack
    TSC
    CLC
    ADC #1
    TCS
    RTS
```

**R65**:
```rust
fn process(count: u8, base @ A: u8) -> u8 {
    // count on stack, base in A
    let result @ A = compute(count, base);
    return result;
}
```

---

## Mode Transitions in Calls

### Automatic Mode System

Mode transitions are now handled automatically by the compiler:

- **Default mode**: m8 (8-bit A), x16 (16-bit X/Y)
- **m16 mode**: Inferred when function has `@ A: u16` parameter
- **X/Y always u16**: No x8 mode in R65

```rust
// m8 mode (default)
fn process_byte(value @ A: u8) -> u8 {
    return value + 1;
}

// m16 mode (inferred from u16 @ A)
fn process_word(value @ A: u16) -> u16 {
    return value + 1;
}

fn caller() {
    let byte = process_byte(10);    // Caller in m8, callee in m8
    let word = process_word(1000);  // Callee switches to m16, returns to m8
}
```

**Generated (m16 callee)**:
```asm
process_word:
    REP #$20       ; Switch to m16
    ; ... function body ...
    SEP #$20       ; Restore m8
    RTS
```

---

### Cross-Mode Calls

When caller and callee have different modes, the callee handles the transition:

```rust
fn caller() {
    // caller is m8 (default)
    let result = needs_16bit(0x1234);  // callee handles REP/SEP
}

fn needs_16bit(value @ A: u16) -> u16 {
    // callee is m16 (inferred)
    return value + 1;
}
```

---

## ABI Compatibility

### Matching Existing Code

R65 can match any hand-written calling convention:

**Example: SNES SDK Convention**
```asm
; Convention: A = parameter, X preserved
sdk_function:
    ; X must be preserved
    ; A = input
    ; Returns A
```

**R65**:
```rust
#[preserves(X)]
fn sdk_function(input @ A: u8) -> u8 {
    // Matches SDK convention
    return compute(input);
}
```

---

### Custom Calling Conventions

Create project-specific conventions:

```rust
// Project convention: use zero-page $80-$8F for parameters/returns
#[zeropage(0x80)]
static mut ARG0: u8;

#[zeropage(0x81)]
static mut ARG1: u8;

#[zeropage(0x82)]
static mut RESULT: u16;

// All project functions follow this
fn project_style(a @ ARG0: u8, b @ ARG1: u8) {
    RESULT = compute(a, b);
}
```

---

## Performance Characteristics

### Call Overhead Comparison

```
Register parameters:      0-3 cycles (setup)
Variable-bound (zp):      3-6 cycles (memory writes)
Stack parameters:         5-10 cycles (stack push/pop)

Near call (JSR/RTS):      12 cycles
Far call (JSL/RTL):       14 cycles
Indirect call:            18-24 cycles (trampoline)

Mode transition (auto):   +12 cycles (PHP/PLP)
```

**Fastest**: Register parameters + near call + no mode transition

---

## Summary

### Calling Convention Decision Tree

**1. Choose parameter passing**:
- Few parameters, performance critical → Register aliases
- Shared communication area (hand-written style) → Zero-page variables
- Many parameters, need reentrancy → Stack

**2. Choose return mechanism**:
- Register return → Explicit `return A` or `return (A, X)`
- Zero-page return → Write to zero-page variable
- Mixed → Combine both

**3. Enforce consistency**:
- All return paths must have identical signatures

**4. Choose call type**:
- Same bank → near `fn()`
- Cross-bank → far `far fn()`

**5. Choose mode transition**:
- Manual control → `transition=none`
- Safe/flexible → `transition=inline`
- Performance (batching) → `transition=caller`

**6. Declare preservation**:
- `#[preserves(X, Y)]` for callee-save registers

### Key Principles

1. **Explicit everything** - programmer specifies all conventions
2. **Match assembly patterns** - replicate hand-written code
3. **Zero-cost abstraction** - no hidden overhead
4. **Type safety** - compiler validates conventions
5. **Flexible** - supports any calling pattern
6. **Return consistency** - all paths return same signature

---

**STATUS**: Design Complete
**Last Updated**: 2025-12-31
**Next Steps**: Implement calling convention code generation in backend
