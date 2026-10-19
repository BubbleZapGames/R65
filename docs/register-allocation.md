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
   - **Scratch registers**: `#[zeropage(addr, register)]` locations hold values whose live range contains no call (scratch is caller-save)
3. **Register Allocation**: `register_alloc.py` maps VRegs to physical locations (HW registers or stack offsets)
4. **Code Generation**: Instruction selectors emit loads/stores based on physical locations

**HW Coalescence** uses a two-pass approach:
- **Pass 1**: Find VRegs where the HW register is unclobbered between def and last use
- **Pass 2**: Re-check remaining candidates treating Pass 1 coalesceable Moves as no-ops (enables cascading coalescence, e.g., A can coalesce when its only "clobber" was a coalesceable B parameter save)

A value computed in A coalesces only when each use reads it from A in place (`_use_reads_a_in_place`). Examples are a store, a compare, a BinaryOp's left operand, and the right operand of `+ & | ^` when the left isn't in A or B. Instruction selection swaps the operands of a commutative op whose right operand is in A, so `s + (BUF[i] as u16)` adds the widened byte straight from A to `s` in memory and doesn't spill it.

MIR is not SSA, so a local that is assigned again keeps its vreg, and every per-vreg decision would see all of its values at once. Before allocation, `WebSplitter` (`optimize/web_split.py`) gives each def-use web its own vreg. A web is the set of defs that the same uses read, joined at merges and back edges. When `sieve` reuses `i` for a second loop, the first loop's `i` is a separate vreg. Loop promotion can then put it in X, even though the prime loop reads its own `i` as the right operand of `j + i`. Parameters and vregs with a register hint are not split. A def nothing reads stays with the original vreg. Promotion's check for "another value is moved into X" (`_x_loaded_from_other`) counts only writes made while the counter is still live, so a later loop's index in X doesn't block an earlier loop's counter.

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
   - Not preserved: the alias *is* the register, so code that overwrites
     it before the alias's last use - an expression evaluated through A,
     16-bit work in A (which writes B, its high byte; 8-bit work leaves
     it), an array index loaded into X, a pointer access indexing with Y,
     a call - is a compile error at the overwriting line
     (`mir/register_alias_check.py`), not a spill. Loop promotion keeps
     its counters off a register an `@ X`/`@ Y` alias names.

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

**First-fit, one function at a time** (`RegisterAllocator._try_scratch`,
`codegen/register_alloc.py`): a vreg that is not live across any call takes
the first free scratch register at least its size. A scratch, once taken,
stays taken for the rest of the function — there is no release at the end
of a live range — so a function can hold at most as many scratch values as
the pool has registers; the rest go to the stack. Functions whose prologue
sets D=S (the far-pointer D=S strategy) get no scratch registers, since DP
addressing would then land in the stack frame.

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

Scratch registers are **not preserved** across function calls, so the
allocator never places a value that is live across a call in one: it gets a
stack slot instead.

```rust
fn example() {
    let before = compute1();  // Live across the call: stack slot
    callee();                 // May clobber any scratch register
    let after = before;       // Reads the stack slot
}
```

Only values whose live range contains no call are candidates for scratch.

### Index Registers Live Across a Call

A value the allocator keeps in X or Y - a loop counter promoted there, a
`self` pointer in Y, a register parameter - or a direct `X`/`Y` the program
names, survives a call that may write the register by being pushed before it
and pulled after. The push is placed per *clobber region*
(`ClobberRegionAnalyzer` in `mir/liveness.py`): a run of calls in one block
with no use of the register between them is bracketed once, so a loop body
that makes two calls saves its counter once per pass, not twice. A register
counts as live after a call when a named use of it is, or when any vreg
allocation placed in it is.

The two registers' regions are found separately and need not nest. Regions
starting at the same call push the later-ending one first; when one must be
pulled from under the other, the one above is pulled too and pushed again at
its region's next call. A value in A is saved around each call it is live
across, though the allocator rarely leaves one there: it does not keep a value
in A across a call that may write it.

### Call Effects

What a call does to A, B, X and Y is computed in one place,
`call_effects` (`r65/compiler/mir/call_effects.py`), as a `CallEffects`
record per call:

| Field | Meaning |
|---|---|
| `results` | registers the callee returns in, in order (A, then B or X, then Y) |
| `arg_regs` | registers arguments are passed in (Y for a `self` pointer) |
| `has_stack_args` | whether any argument is pushed |
| `declared` | the callee's `#[preserves(...)]` |
| `inferred` | index registers the callee's code, and its callees', never writes (`codegen/clobber_summary.py`) |

`inferred` is empty for a call with stack arguments (parking A across their
setup and pulling them off afterwards both use X or Y), for a register an
argument goes to, and for anything but a direct call to a generated
function: a trait dispatch, a function pointer, a builtin or an `extern fn`.

Each consumer reads the view it can afford to trust:

- **Call selection** (`codegen/call_select.py`, `codegen/abi_model.py`)
  uses `preserved = declared | inferred` to decide which live registers to
  push around a call and where A may be parked during argument setup. Clobber
  regions (`ClobberRegionAnalyzer`) read the same record.
- **Slot allocation's coalescing** (`codegen/slot_allocator.py`) uses
  `declared` only.
- **The MIR passes** (register alias check, index streams, loop register
  promotion, X reuse in the builder) treat every call as clobbering
  everything (`CLOBBERS_ALL` in `mir/nodes.py`), even a declared
  `#[preserves]`.

The register a result is *collected* into is not part of the record. A return
value the allocator placed in X arrives there by `TAX` after the call, which
depends on slot allocation rather than on the call, so call selection adds
those registers itself (`_call_result_registers`). The region analysis cannot
see them, which is why call selection closes a region whose register turns
out to receive a result.

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
