# Register Allocation Strategy Design

## Overview

R65's register allocation is designed for **reverse engineering and replicating hand-written assembly**, not automatic optimization. The strategy prioritizes **explicit programmer control** and **predictable code generation** over clever compiler heuristics.

**Design Principles**:
- Explicit register control - programmer specifies hardware register usage via aliases
- Predictable output - same source always generates same assembly
- Match assembly patterns - replicate hand-written code idioms
- Automatic allocation of temporaries - compiler manages virtual registers
- Trust the programmer - minimal compiler interference

**Philosophy**: The compiler is a **translator**, not an optimizer. It follows programmer directives to generate assembly that matches the mental model.

### Implementation Overview

The actual register allocation pipeline is more sophisticated than a simple scratch pool:

1. **MIR Generation**: All values are assigned **virtual registers** (VReg) by `VirtualRegisterAllocator`
2. **Slot Allocation**: `slot_allocator.py` determines physical locations for each VReg:
   - **HW-coalesceable VRegs**: Values that can stay in their hardware register (A, X, Y, B) without spilling — detected by analyzing whether the HW register is clobbered between definition and last use
   - **Stack slots**: Non-coalesceable values are allocated to stack positions
   - **Scratch registers**: `#[zeropage(addr, register)]` locations are available as call-graph-aware scratch space
3. **Register Allocation**: `register_alloc.py` maps VRegs to physical locations (HW registers or stack offsets)
4. **Code Generation**: Instruction selectors emit loads/stores based on physical locations

**HW Coalescence** uses a two-pass approach:
- **Pass 1**: Find VRegs where the HW register is unclobbered between def and last use
- **Pass 2**: Re-check remaining candidates treating Pass 1 coalesceable Moves as no-ops (enables cascading coalescence, e.g., A can coalesce when its only "clobber" was a coalesceable B parameter save)

---

## Core Strategy: Explicit Everything

### Default: Variables Live in Memory

Variables live in **explicit memory locations** by default:

```rust
#[zeropage(0x10)]
static mut TEMP: u8;

#[zeropage(0x12)]
static mut COUNTER: u16;

fn example() {
    TEMP = 42;         // Goes to $10 (memory)
    COUNTER = 1000;    // Goes to $12-$13 (memory)
}
```

**Generated Assembly**:
```asm
LDA #42
STA $10        ; Direct to memory

LDA #<1000
STA $12
LDA #>1000
STA $13
```

**No automatic register allocation** - values go straight to their memory locations.

---

### Explicit Register Aliases

To use a hardware register, **explicitly alias** it:

```rust
#[zeropage(0x10)]
static mut VALUE: u8;

fn example() {
    let temp @ A = VALUE;  // Load VALUE into A
    temp = temp + 1;       // Operate on A
    VALUE = temp;          // Store A back to VALUE
}
```

**Generated Assembly**:
```asm
LDA $10        ; Load into A
CLC
ADC #1         ; Operate on A
STA $10        ; Store back
```

**Matches hand-written pattern** exactly.

---

## Zero-Page Scratch Registers

### The `register` Attribute

Mark zero-page locations as **compiler-managed scratch space**:

```rust
// Scratch space: compiler can use these for temporaries
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

#[zeropage(0x17, register)]
static mut SCRATCH1: u8;

#[zeropage(0x18, register)]
static mut SCRATCH2: u16;  // Takes 0x18-0x19
```

**Semantics**:
- Marked as `register` - available for compiler temporary allocation
- Not preserved across function calls (caller-save)
- Can be reused by different functions
- Acts like "virtual registers" in zero-page memory

### Compiler Use of Scratch Registers

The compiler can allocate temporaries to scratch space:

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    // Temporary without explicit location
    let temp = compute();  // Compiler allocates to SCRATCH0
    process(temp);
}
```

**Generated Assembly**:
```asm
JSR compute
STA $16        ; Allocated to SCRATCH0
LDA $16
JSR process
```

**Predictable**: Uses designated scratch space, not arbitrary memory

---

### Scratch Register Allocation Rules

**Rule 1**: Scratch registers are function-local

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn func1() {
    let temp = 10;  // Uses SCRATCH0
}

fn func2() {
    let temp = 20;  // Also uses SCRATCH0 (no conflict!)
}
```

**No conflict** - different functions can reuse the same scratch space

**Rule 2**: Not preserved across calls

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    let before = 10;   // Allocated to SCRATCH0
    callee();          // May clobber SCRATCH0
    let after = before; // ERROR: 'before' may be invalid
}
```

**Compiler error**: Value in scratch register doesn't survive function calls

**Rule 3**: Explicit access forbidden

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    SCRATCH0 = 10;  // ERROR: cannot directly access scratch register
}
```

**Only compiler** can allocate to scratch registers, not programmer

---

### Scratch Register Pool

Declare a pool of scratch registers for the compiler:

```rust
// Common pattern: reserve $10-$1F for scratch space
#[zeropage(0x10, register)]
static mut SCRATCH0: u8;

#[zeropage(0x11, register)]
static mut SCRATCH1: u8;

#[zeropage(0x12, register)]
static mut SCRATCH2: u16;

#[zeropage(0x14, register)]
static mut SCRATCH3: u16;

#[zeropage(0x16, register)]
static mut SCRATCH_PTR: *u8;

// Compiler has pool: $10, $11, $12-$13, $14-$15, $16-$17
```

**Allocation**: Compiler assigns temporaries from this pool

---

## Register Allocation Rules

### Rule 1: Hardware Registers Need Explicit Aliases

Operations on A, X, Y require explicit aliases:

```rust
// ERROR: Which register?
let result = a + b;

// OK: Explicit A register
let result @ A = a + b;

// OK: Allocated to scratch register
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

let result = a + b;  // Compiler uses SCRATCH0
```

### Rule 2: Scratch Registers for Unnamed Temporaries

Compiler allocates unnamed temporaries to scratch space:

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    // Anonymous temporary - uses SCRATCH0
    process(a + b);
}
```

**Generated**:
```asm
LDA a
CLC
ADC b
STA $16        ; Store to SCRATCH0
LDA $16
JSR process
```

### Rule 3: Named Temporaries Without Location → Scratch

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    let temp = compute();  // Uses SCRATCH0
    process(temp);
}
```

### Rule 4: Explicit Location → Fixed Memory

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

fn example() {
    TEMP = compute();  // Fixed at $20, not scratch
    process(TEMP);
}
```

---

## Allocation Strategy

### Allocation Priority

1. **Explicit hardware register aliases** (`let x @ A`)
   - Highest priority
   - Mandatory allocation

2. **Explicit memory locations** (`#[zeropage(0x20)] static mut VAR`)
   - Fixed location
   - No allocation needed

3. **Temporaries → Scratch registers** (`let temp = value`)
   - Allocated from scratch pool
   - Function-local only

4. **Stack** (fallback)
   - If no scratch registers available
   - Slowest option

### Scratch Register Allocation Algorithm

**Simple first-fit algorithm**:

```python
class ScratchAllocator:
    def __init__(self, scratch_pool):
        self.pool = scratch_pool  # List of available scratch registers
        self.allocated = {}       # Map: variable -> scratch register

    def allocate(self, variable, size):
        # Find first available scratch register of appropriate size
        for scratch in self.pool:
            if scratch.size >= size and scratch not in self.allocated.values():
                self.allocated[variable] = scratch
                return scratch

        # No scratch available - fallback to stack
        return None

    def release(self, variable):
        # Release at end of variable's lifetime
        if variable in self.allocated:
            del self.allocated[variable]
```

**Deterministic**: Always allocates in same order for predictable output

---

## Common Assembly Patterns

### Pattern 1: Load-Modify-Store

**Assembly**:
```asm
LDA $10
CLC
ADC #1
STA $10
```

**R65**:
```rust
#[zeropage(0x10)]
static mut VAR: u8;

let temp @ A = VAR;
temp = temp + 1;
VAR = temp;
```

---

### Pattern 2: Temporary in Scratch Space

**Assembly**:
```asm
JSR compute
STA $16        ; Store in scratch
; ... other operations ...
LDA $16        ; Reload
JSR process
```

**R65**:
```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

let temp = compute();  // Allocated to SCRATCH0
// ... other operations ...
process(temp);
```

---

### Pattern 3: Register Transfer

**Assembly**:
```asm
LDA $10
TAX
```

**R65**:
```rust
#[zeropage(0x10)]
static mut VALUE: u8;

let temp @ A = VALUE;
let index @ X = temp;
```

---

### Pattern 4: Loop Counter in X

**Assembly**:
```asm
LDX #0
loop:
    ; ... body ...
    INX
    CPX #10
    BCC loop
```

**R65**:
```rust
let i @ X = 0;
loop {
    // ... body ...
    i = i + 1;
    if i >= 10 { break; }
}
```

---

### Pattern 5: Accumulate in A with Scratch

**Assembly**:
```asm
LDA #0
STA $16        ; Save accumulator in scratch
LDX #0
loop:
    LDA $16
    CLC
    ADC array,X
    STA $16
    INX
    CPX #10
    BCC loop
LDA $16
```

**R65**:
```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

let sum = 0;      // Allocated to SCRATCH0
let i @ X = 0;
loop {
    sum = sum + array[i];
    i = i + 1;
    if i >= 10 { break; }
}
let result @ A = sum;
```

---

## Lifetime and Scope

### Scratch Register Lifetime

Scratch registers are **function-scoped**:

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    let temp1 = compute1();  // Uses SCRATCH0
    process(temp1);          // temp1 dies

    let temp2 = compute2();  // Reuses SCRATCH0
    process(temp2);
}
```

**Reuse**: SCRATCH0 reused for temp2 after temp1 dies

### Explicit Scope Control

Use blocks to control lifetime:

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    {
        let temp1 = compute1();  // Uses SCRATCH0
        process(temp1);
    }  // temp1 dies, SCRATCH0 released

    {
        let temp2 = compute2();  // Uses SCRATCH0
        process(temp2);
    }  // temp2 dies
}
```

---

## Function Calls and Scratch Preservation

### Scratch Registers Are Caller-Save

Scratch registers are **not preserved** across function calls:

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    let before = compute1();  // Allocated to SCRATCH0
    callee();                 // May clobber SCRATCH0
    let after = before;       // ERROR: 'before' invalid after call
}
```

**Compiler error**: Variable in scratch register used after function call

### Workaround: Save to Explicit Location

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

#[zeropage(0x20)]
static mut SAVED: u8;

fn example() {
    let before = compute1();  // Allocated to SCRATCH0
    SAVED = before;           // Save to non-scratch memory
    callee();
    let after = SAVED;        // OK: SAVED is preserved
}
```

---

## Scratch Register Sizing

### Scratch Register Types

Scratch registers are declared with explicit types:

```rust
#[zeropage(0x16, register)]
static mut SCRATCH_U8: u8;   // 8-bit scratch

#[zeropage(0x18, register)]
static mut SCRATCH_U16: u16;  // 16-bit scratch (takes 0x18-0x19)
```

The compiler automatically selects appropriately-sized scratch registers based on the operation being performed.

### Mixed Scratch Pools

```rust
// 8-bit scratch space
#[zeropage(0x16, register)]
static mut SCRATCH8_0: u8;

#[zeropage(0x17, register)]
static mut SCRATCH8_1: u8;

// 16-bit scratch space
#[zeropage(0x18, register)]
static mut SCRATCH16_0: u16;

#[zeropage(0x1A, register)]
static mut SCRATCH16_1: u16;
```

**Size-appropriate allocation**: Compiler picks u8 scratch for 8-bit operations, u16 scratch for 16-bit operations.

---

## Examples: Reverse Engineering Workflow

### Example 1: Function with Scratch

**Original Assembly**:
```asm
calculate:
    JSR get_value
    STA $16        ; Temporary in zero-page
    LDA $16
    CLC
    ADC #10
    RTS
```

**Reverse Engineered R65**:
```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn calculate() -> u8 {
    let temp = get_value();  // Allocated to SCRATCH0
    let result @ A = temp + 10;
    return result;
}
```

**Recompiled**: Exact match

---

### Example 2: Multiple Temporaries

**Original Assembly**:
```asm
complex:
    JSR compute_a
    STA $16        ; First temporary

    JSR compute_b
    STA $17        ; Second temporary

    LDA $16
    CLC
    ADC $17
    RTS
```

**Reverse Engineered R65**:
```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

#[zeropage(0x17, register)]
static mut SCRATCH1: u8;

fn complex() -> u8 {
    let a = compute_a();  // SCRATCH0
    let b = compute_b();  // SCRATCH1
    let result @ A = a + b;
    return result;
}
```

**Recompiled**: Exact match

---

### Example 3: Explicit vs Scratch

**Original Assembly**:
```asm
process:
    LDA $10        ; Explicit variable
    STA $16        ; Scratch temporary

    LDA $16
    CLC
    ADC #1
    STA $10

    RTS
```

**Reverse Engineered R65**:
```rust
#[zeropage(0x10)]
static mut VALUE: u8;

#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn process() {
    let temp = VALUE;  // Allocated to SCRATCH0
    VALUE = temp + 1;
}
```

---

## Compiler Validation

### Scratch Register Conflicts

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    SCRATCH0 = 10;  // ERROR: direct access to scratch register
}

// Error message:
// error: cannot directly access scratch register
//  --> file.r65:5:5
//   |
// 5 |     SCRATCH0 = 10;
//   |     ^^^^^^^^ scratch register (marked with `register` attribute)
//   |
// note: scratch registers are compiler-managed
//  --> file.r65:1:1
//   |
// 1 | #[zeropage(0x16, register)]
//   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^ declared as scratch here
//   |
// help: use a regular zero-page variable instead:
//       #[zeropage(0x16)]
//       static mut VAR: u8;
```

### Live Across Call Error

```rust
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

fn example() {
    let temp = compute();
    callee();
    process(temp);
}

// Error message:
// error: variable in scratch register used after function call
//  --> file.r65:6:13
//   |
// 4 |     let temp = compute();
//   |         ---- variable allocated to scratch register $16
// 5 |     callee();
//   |     -------- scratch may be clobbered here
// 6 |     process(temp);
//   |             ^^^^ use of potentially invalid value
//   |
// help: save to non-scratch memory before call:
//       #[zeropage(0x20)]
//       static mut SAVED: u8;
//
//       SAVED = temp;
//       callee();
//       process(SAVED);
```

---

## Performance Considerations

### Zero-Page Scratch is Fast

Scratch registers in zero-page are nearly as fast as hardware registers:

```
Hardware registers (A, X, Y): 2 cycles
Zero-page scratch:            3-4 cycles
RAM:                          4-5 cycles
Stack:                        5-8 cycles
```

**Sweet spot**: Zero-page scratch balances speed and availability

### Minimize Stack Usage

Prefer scratch registers over stack:

```rust
// Good:
#[zeropage(0x16, register)]
static mut SCRATCH0: u8;

let temp = compute();  // SCRATCH0 (3-4 cycles)

// Bad:
let temp = compute();  // Stack (5-8 cycles, no scratch available)
```

### Provide Adequate Scratch Space

Typical recommendation: **8-16 bytes** of scratch space

```rust
// Good scratch pool
#[zeropage(0x10, register)]
static mut SCRATCH0: u8;

#[zeropage(0x11, register)]
static mut SCRATCH1: u8;

#[zeropage(0x12, register)]
static mut SCRATCH2: u16;

#[zeropage(0x14, register)]
static mut SCRATCH3: u16;

#[zeropage(0x16, register)]
static mut SCRATCH_PTR: *u8;
// Total: 8 bytes
```

---

## Summary

### Register Allocation Strategy

1. **Hardware registers** (A, X, Y): Explicit aliases only
2. **Scratch registers**: Zero-page locations marked with `register` attribute
3. **Fixed memory**: Explicit `#[zeropage]` / `#[ram]` locations
4. **Stack**: Fallback for overflow

### Allocation Decision Tree

```
Variable needs allocation
    |
    ├─ Explicit alias (let x @ A)?
    │   └─> Allocate to hardware register
    |
    ├─ Explicit location (#[zeropage(0x20)])?
    │   └─> Use fixed memory location
    |
    ├─ Scratch register available?
    │   └─> Allocate to scratch register
    |
    └─ Otherwise
        └─> Allocate to stack
```

### Key Benefits

- **Explicit control**: Programmer specifies scratch space
- **Predictable**: Deterministic allocation from scratch pool
- **Reverse engineering friendly**: Matches hand-written assembly patterns
- **Fast**: Zero-page scratch is nearly as fast as hardware registers
- **Safe**: Compiler enforces caller-save semantics

---

**STATUS**: Implemented (doc describes design philosophy; actual implementation uses virtual registers, HW coalescence, and slot allocation — see `slot_allocator.py` and `register_alloc.py`)
**Last Updated**: 2026-02-10
