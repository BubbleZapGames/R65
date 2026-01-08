# Pointer Operations and Memory Model Design

## Overview

R65 provides explicit pointer types that directly reflect the 65816's segmented memory architecture and addressing modes.

**Design Principles**:
- Hardware transparency: Expose 16-bit and 24-bit address spaces
- Zero-cost abstractions: Pointers map directly to addressing modes
- Explicit bank management: Programmer controls DBR
- No safety guarantees: All pointer operations are unchecked

---

## 65816 Memory Architecture

### Address Space

24-bit address space organized into 256 banks of 64KB each (16MB total):

```
$00:0000 - $FF:FFFF  (16MB addressable)
```

### Key Registers

- **PBR** (Program Bank Register): Current execution bank (read-only)
- **DBR** (Data Bank Register): Default bank for data access
- **D** (Direct Page): Base address for zero-page (relocatable)

### Typical SNES Memory Map

```
$00:0000 - $00:1FFF  Low RAM (8KB)
$00:2000 - $00:7FFF  Hardware registers
$00:8000 - $3F:FFFF  ROM (lower)
$7E:0000 - $7F:FFFF  Work RAM (128KB)
$80:0000 - $FF:FFFF  ROM (upper/mirrored)
```

---

## Pointer Types

### Near Pointer: `near<T>`

**Size**: 2 bytes (16-bit)
**Range**: 64KB within current DBR
**Speed**: Fast (5-6 cycles for indirect)

```rust
let ptr: near<u8> = 0x2000;
let value = *ptr;  // Uses DBR for bank
```

**Assembly**: `LDA ($nn)` - uses DBR implicitly

**Use for**: Data within current bank, zero-page pointers, fast indirect addressing

---

### Far Pointer: `far<T>`

**Size**: 3 bytes (24-bit)
**Range**: Full 16MB
**Speed**: Slow (requires DBR manipulation)

```rust
let ptr: far<u8> = 0x01_2000;  // Bank 1, offset 0x2000
let value = *ptr;  // Must manage DBR
```

**Assembly**: Requires PHB/PLB to change DBR

**Use for**: Cross-bank access, ROM in different banks

---

### Null Pointers

**Value**: `0x0000` (near) or `0x00_0000` (far)
**Semantics**: No automatic null checks - dereferencing is **undefined behavior**

```rust
let ptr: near<u8> = 0x0000;
if ptr as u16 != 0 {  // Manual check required
    let value = *ptr;
}
```

---

## Pointer Operations

### Address-Of: `&`

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

#[ram]
static mut BUFFER: [u8; 256];

let zp_ptr = &TEMP;      // near<u8> - zeropage is bank 0
let ram_ptr = &BUFFER;   // far<[u8; 256]> - ram is bank $7E
```

**Automatic type inference**: The compiler infers `near<T>` or `far<T>` based on storage:
- `#[zeropage]`, `#[lowram]`, `#[hw]` → `near<T>` (16-bit, bank 0)
- `#[ram]`, `#[rom]` → `far<T>` (24-bit, includes bank)

**Restrictions**:
- Cannot take address of register aliases (`&A` is error)
- Only works on lvalues (variables, array elements, fields)

---

### Dereference: `*`

```rust
let ptr: near<u8> = 0x2000;
let value = *ptr;        // Read
*ptr = 42;               // Write
```

**Zero-page optimization**:
```rust
#[zeropage(0x42)]
static mut PTR: near<u8>;

*PTR = 5;  // LDA #$05, STA ($42) - very fast!
```

---

### Indexing: `ptr[index]`

Equivalent to `*(ptr + index)`

```rust
let ptr: near<u8> = 0x2000;
let value = ptr[10];      // Constant offset
let value = ptr[index];   // Variable offset
let value @ A = ptr[Y];   // Register Y indexing
```

**Best performance with zero-page + Y**:
```rust
#[zeropage(0x42)]
static mut PTR: near<u8>;

PTR[Y] = value;  // STA ($42),Y - indirect indexed
```

---

### Pointer Arithmetic

```rust
let ptr: near<u8> = 0x2000;
let ptr2 = ptr + 10;      // Add offset
let ptr3 = ptr - 5;       // Subtract offset
ptr += 100;               // Compound assignment

let diff: u16 = ptr2 - ptr;  // Pointer difference
```

**Type scaling**: Automatically scales by `sizeof(T)`
```rust
let ptr: near<u16> = 0x2000;
let ptr2 = ptr + 1;  // Advances by 2 bytes (sizeof(u16))
```

**Wrapping**:
- Near pointers wrap at 64KB (same bank)
- Far pointers wrap at 16MB

---

### Pointer Casting

```rust
// Between near and far
let near_ptr: near<u8> = 0x2000;
let far_ptr: far<u8> = near_ptr as far<u8>;  // Adds DBR

// To/from integers
let addr: u16 = 0x2000;
let ptr: near<u8> = addr as near<u8>;

// Between pointer types
let u8_ptr: near<u8> = 0x2000;
let u16_ptr: near<u16> = u8_ptr as near<u16>;  // Reinterpret
```

---

## Addressing Modes

### Direct Page (Zero-Page)

**Speed**: 3-4 cycles (fastest)

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

TEMP = 42;  // STA $20
```

**With D register**: Access is at `D + offset`

---

### Absolute

**Speed**: 4-5 cycles

```rust
#[ram(0x7E2000)]
static mut BUFFER: u8;

BUFFER = 42;  // STA $2000 (uses DBR)
```

---

### Indexed (X, Y)

**Speed**: 4-5 cycles

```rust
let index @ X = 10;
let value = ARRAY[index];  // LDA ARRAY,X
```

---

### Indirect

**Speed**: 5-6 cycles

```rust
#[zeropage(0x42)]
static mut PTR: near<u8>;

let value = *PTR;  // LDA ($42)
```

---

### Indirect Indexed

**Speed**: 5-6 cycles

```rust
#[zeropage(0x42)]
static mut PTR: near<u8>;

let value = PTR[Y];  // LDA ($42),Y
```

---

## Memory Storage Classes

### `#[zeropage]` - Direct Page

**Size**: 256 bytes
**Speed**: Fastest (3-4 cycles)
**Best for**: Frequently accessed variables, pointers, counters

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

#[zeropage]  // Compiler allocates
static mut FLAGS: u8;
```

---

### `#[ram]` - General RAM

**Size**: Large (up to 128KB on SNES)
**Speed**: Moderate (4-5 cycles)
**Best for**: Arrays, buffers, game state

```rust
#[ram]
static mut BUFFER: [u8; 4096];

#[ram(0x7E2000)]  // Explicit address
static mut DATA: u16;
```

---

### `#[rom]` - Read-Only Memory

**Mutability**: Read-only (writes are compile error)
**Best for**: Graphics, levels, constants, sound data

```rust
#[rom(0x8000)]
static GRAPHICS: [u8; 4096] = include_bytes!("gfx.bin");

GRAPHICS[0] = 1;  // ERROR: cannot write to ROM
```

---

### `#[hw]` - Hardware Registers

**Volatility**: Automatically volatile - every access hits hardware
**Best for**: PPU, APU, DMA, controllers

```rust
#[hw(0x2100)]
static mut INIDISP: u8;

#[hw(0x4212)]
static mut HVBJOY: u8;

// Every read is a hardware access
loop {
    let status = HVBJOY;
    if (status & 0x80) != 0 { break; }
}

// Both writes execute (not optimized away)
INIDISP = 0x0F;
INIDISP = 0x80;
```

---

## Stack

**Location**: Controlled by S register (typically `$0100-$01FF`)
**Usage**: Stack parameters, local variables, function calls

```rust
fn process(a: u8, b: u8) {  // Stack parameters
    let local: u8 = a + b;  // Stack-allocated
}
```

---

## Data Bank Register (DBR)

### Setting DBR

```rust
DBR = 0x7E;  // LDA #$7E, PHA, PLB
```

### DBR in Function Calls

```rust
#[bank(1, data_bank=inline)]
far fn graphics_routine() {
    // Compiler auto-generates DBR save/restore
}
```

---

## Memory Safety (Lack Thereof)

### No Bounds Checking

```rust
let buffer: [u8; 256];
let index: u16 = 300;
let value = buffer[index];  // UB: no check!
```

### Null Dereference

```rust
let ptr: near<u8> = 0x0000;
let value = *ptr;  // UB: no check!
```

### Wild Pointers

```rust
let ptr: near<u8>;  // Uninitialized
let value = *ptr;   // UB: garbage address
```

### Type Punning

```rust
let u8_ptr: near<u8> = 0x2000;
let u16_ptr: near<u16> = u8_ptr as near<u16>;
*u16_ptr = 0x1234;  // Allowed but potentially UB
```

---

## Alignment

**No alignment requirements** - 65816 allows unaligned access with no penalty:

```rust
struct Player {
    x: u8,       // Offset 0
    y: u8,       // Offset 1
    health: u16  // Offset 2 (unaligned - OK!)
}
```

---

## Examples

### Memory Copy

```rust
fn memcpy(dst: near<u8>, src: near<u8>, count @ X: u8) {
    if count == 0 { return; }
    let index @ Y = 0;
    loop {
        dst[index] = src[index];
        index += 1;
        count -= 1;
        if count == 0 { break; }
    }
}
```

### Memory Fill

```rust
fn memset(dst: near<u8>, value @ A: u8, count @ X: u8) {
    if count == 0 { return; }
    let index @ Y = 0;
    loop {
        dst[index] = value;
        index += 1;
        count -= 1;
        if count == 0 { break; }
    }
}
```

### Linked List

```rust
struct Node {
    data: u8,
    next: near<Node>,
}

fn traverse(head: near<Node>) {
    let mut current = head;
    loop {
        if current as u16 == 0 { break; }
        process(current.data);
        current = current.next;
    }
}
```

### ROM Table Lookup

```rust
#[rom]
static SIN_TABLE: [u8; 256] = [ /* ... */ ];

fn get_sin(angle @ A: u8) -> u8 {
    return (&SIN_TABLE)[angle];
}
```

---

## Pointer Comparison

```rust
let ptr1: near<u8> = 0x2000;
let ptr2: near<u8> = 0x2100;

if ptr1 < ptr2 { }        // Address comparison
if ptr1 == ptr2 { }       // Equality
if ptr as u16 != 0 { }    // Null check
```

---

## Type Sizes

```
near<T>:  2 bytes
far<T>:   3 bytes
```

Pointers are just addresses - no metadata, no bounds, no ownership.

---

## Future Enhancements

- **Const pointers**: `near<const u8>` vs `near<mut u8>`
- **Slice types**: Fat pointers with length `{ ptr: near<T>, len: u16 }`
- **Smart pointers**: RAII wrappers for hardware resources

---

**STATUS**: Design Complete
**Last Updated**: 2025-12-31
**Next Steps**: Implement pointer types in type system, codegen for addressing modes
