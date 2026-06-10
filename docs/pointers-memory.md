# Pointer Operations and Memory Model

## Overview

R65 provides explicit pointer types that directly reflect the 65816's segmented memory architecture. Pointers map to hardware addressing modes with no abstraction cost. There are no safety guarantees — no bounds checking, no null checks, no lifetime tracking.

## 65816 Memory Architecture

24-bit address space: 256 banks of 64KB each (16MB total).

**Key Registers**: PBR (execution bank, read-only), DBR (default data bank), D (direct page base).

**SNES Memory Map**:
```
$00:0000 - $00:1FFF  Low RAM (8KB)
$00:2000 - $00:7FFF  Hardware registers
$00:8000 - $3F:FFFF  ROM (lower)
$7E:0000 - $7F:FFFF  Work RAM (128KB)
$80:0000 - $FF:FFFF  ROM (upper/mirrored)
```

## Pointer Types

### Near Pointer: `name: *T`

**Size**: 2 bytes (16-bit) — **Range**: 64KB within current DBR

```rust
let ptr: *u8 = 0x2000;
let value = *ptr;  // Uses DBR for bank
```

**Assembly**: `LDA ($zp)` (indirect), `LDA ($zp),Y` (indirect indexed)

### Far Pointer: `name: far *T`

**Size**: 3 bytes (24-bit) — **Range**: Full 16MB

```rust
let ptr: far *u8 = 0x01_2000;  // Bank 1, offset 0x2000
let value = *ptr;
```

**Assembly**: `LDA [$zp]` (indirect long), `LDA [$zp],Y` (indirect long indexed)

### Pointer to Array: `name: *[T; N]`

Pointer to a fixed-size array — enables compile-time bounds checking on constant indices. A `*[T; N]` implicitly coerces to `*T`, so arrays can be passed to functions taking element pointers.

```rust
fn write_message(msg: *u8) {
    X = 0;
    loop {
        A = msg[X];
        if A == 0 { break; }
        X++;
    }
}
```

### Function Pointers

```rust
fn(u8) -> u8          // Near function pointer (2 bytes, JSR/RTS)
far fn(u8) -> u8      // Far function pointer (3 bytes, JSL/RTL)
```

### Null Pointers

**Value**: `0x0000` (near) or `0x00_0000` (far). No automatic null checks — dereferencing null is UB.

```rust
let ptr: *u8 = 0x0000;
if ptr as u16 != 0 {  // Manual check required
    let value = *ptr;
}
```

## Declaring Pointers

R65 has two syntax forms for pointer declarations. Both parse and both are used in practice:

| Form | Syntax | Star position |
|------|--------|---------------|
| Type-side (canonical) | `name: *T` | In the type |
| Pattern-side | `*name: T` | Before the name |

The type-side form (`name: *T`, `name: far *T`) is the canonical style. The pattern-side form (`*name: T`) is also valid and commonly used for pointer output parameters.

### Local Variables

```rust
let ptr: *u8 = 0x2000;                 // Near pointer
let ptr: far *u8 = 0x01_2000;          // Far pointer
let mut ptr: *u8;                       // Mutable, uninitialized
let *ptr: u8 = 0x210D as *u8;          // Pattern-side form (also valid)
```

### Static Variables

```rust
#[zeropage(0x42)]
static mut PTR: *u8;                    // Near pointer in zero-page (fastest)

#[zeropage]
static mut FAR_PTR: *u16;              // Pattern-side form also valid: *FAR_PTR: u16

#[ram]
static mut BUFFER_PTR: far *u8;        // Far pointer in RAM
```

### Function Parameters

```rust
fn read(src: *u8) { }                  // Near pointer param
fn copy(dst: *u8, src: far *u8) { }    // Mixed near/far
fn print(msg: *u8) { }                // Pointer to array data
fn mul(a: u16, *result: u16) { }       // Pointer output param (pattern-side)
```

### Struct Fields

```rust
struct Node {
    data: u8,
    next: *Node,       // Near pointer to same type
}
```

## Pointer Operations

### Address-Of: `&`

```rust
#[zeropage(0x20)]
static mut TEMP: u8;

#[ram]
static mut BUFFER: [u8; 256];

let zp_ptr: *u8 = &TEMP;              // Near — zeropage is bank 0
let ram_ptr: far *u8 = &BUFFER;       // Far — RAM is bank $7E

static ROM_TABLE: [u8; 16] = [0; 16];
let rom_ptr: *u8 = &ROM_TABLE;        // Near — ROM in bank 0
```

The compiler infers near or far based on storage class:
- `#[zeropage]`, `#[lowram]`, `#[hw]` → near pointer (16-bit, bank 0)
- Immutable statics (ROM) in bank 0 → near pointer (16-bit)
- `#[ram]`, and `static mut` with **no** storage attribute (defaults to `#[ram]`) → far pointer (24-bit, bank $7E)

Cannot take address of register aliases (`&A` is an error). Only works on lvalues.

### Dereference: `*`

```rust
let ptr: *u8 = 0x2000;
let value = *ptr;        // Read
*ptr = 42;               // Write
```

**Zero-page optimization**:
```rust
#[zeropage(0x42)]
static mut PTR: *u8;

*PTR = 5;  // LDA #$05, STA ($42) — very fast!
```

### Auto-Dereference for Field Access

Pointer-to-struct supports direct field access with `.`, like C's `->` operator:

```rust
struct Player { x: u8, y: u8, health: u16 }

#[zeropage]
static mut PLAYER_PTR: *Player;

PLAYER_PTR.x = 10;      // Auto-deref: equivalent to (*PLAYER_PTR).x = 10
let hp = PLAYER_PTR.health;
```

### Indexing: `ptr[index]`

Equivalent to `*(ptr + index)`. Best performance with zero-page pointer + Y register:

```rust
#[zeropage(0x42)]
static mut PTR: *u8;

let value = PTR[Y];   // LDA ($42),Y — indirect indexed
PTR[Y] = value;       // STA ($42),Y
```

### Pointer Arithmetic

```rust
let ptr: *u8 = 0x2000;
let ptr2: *u8 = ptr + 10;      // Add offset
let ptr3: *u8 = ptr - 5;       // Subtract offset
ptr += 100;                    // Compound assignment
let diff: u16 = ptr2 - ptr;    // Pointer difference
```

**Type scaling**: Automatically scales by `sizeof(T)`:
```rust
let ptr: *u16 = 0x2000;
let ptr2: *u16 = ptr + 1;  // Advances by 2 bytes (sizeof(u16))
```

Near pointers wrap at 64KB; far pointers wrap at 16MB.

### Pointer Casting

```rust
// Near to far
let near_ptr: *u8 = 0x2000;
let far_ptr = near_ptr as far *u8;  // Extends with DBR

// Integer to pointer
let addr: u16 = 0x2000;
let ptr = addr as *u8;

// Between pointer types
let u8_ptr: *u8 = 0x2000;
let u16_ptr = u8_ptr as *u16;  // Reinterpret
```

### Pointer Comparison

```rust
let ptr1: *u8 = 0x2000;
let ptr2: *u8 = 0x2100;

if ptr1 < ptr2 { }        // Address comparison
if ptr1 == ptr2 { }       // Equality
if ptr1 as u16 != 0 { }   // Null check
```

### Array Pointer Coercion

A `*[T; N]` (pointer to fixed-size array) can be assigned to `*T` (element pointer):

```rust
static TABLE: [u8; 256] = [0; 256];

fn process(data: *u8) { }

process(&TABLE);  // *[u8; 256] coerces to *u8
```

## Addressing Modes

How pointers map to 65816 instructions:

| Mode | R65 Syntax | Assembly | Cycles | Use |
|------|-----------|----------|--------|-----|
| DP Indirect | `*PTR` | `LDA ($zp)` | 5-6 | Near zeropage pointer deref |
| DP Indirect Indexed | `PTR[Y]` | `LDA ($zp),Y` | 5-6 | Near zeropage pointer + Y |
| DP Indirect Long | `*FAR_PTR` | `LDA [$zp]` | 6-7 | Far zeropage pointer deref |
| DP Indirect Long Indexed | `FAR_PTR[Y]` | `LDA [$zp],Y` | 6-7 | Far zeropage pointer + Y |
| Stack Relative Indirect | *(internal)* | `LDA (d,S),Y` | 7-8 | Stack pointer parameter |

All indirect addressing modes require the pointer to be in zero-page or on the stack. Pointers in RAM cannot be used for indirect addressing without first loading into zero-page.

## Memory Storage Classes

Storage class is determined by mutability and attributes:

| Storage | Attribute | Range | Speed | Best For |
|---------|-----------|-------|-------|----------|
| Direct Page | `#[zeropage]` | `$0000-$00FF` | 3-4 cycles | Pointers, counters, temps |
| Low RAM | `#[lowram]` | `$0000-$1FFF` | 4-5 cycles | Frequently accessed data |
| Main RAM | `#[ram]` | `$7E2000-$7FFFFF` | 4-5 cycles | Arrays, buffers, game state |
| Hardware | `#[hw(addr)]` | I/O addresses | 4-6 cycles | Hardware registers |
| ROM | *(immutable static)* | Bank-dependent | 4-5 cycles | Graphics, tables, constants |
| Stack | *(automatic)* | `S` register area | 5-10 cycles | Locals, parameters |

```rust
#[zeropage(0x20)]
static mut TEMP: u8;                    // Explicit zeropage address

#[zeropage]
static mut FLAGS: u8;                   // Auto-allocated zeropage

#[ram]
static mut BUFFER: [u8; 4096];         // Main RAM, auto-allocated

static SINE_TABLE: [u8; 256] = [0; 256];  // Immutable = ROM (no attribute)
```

## Safety

R65 provides **no memory safety guarantees**. The programmer is responsible for:

- **Bounds checking**: Array and pointer indexing is unchecked. Out-of-bounds is UB.
- **Null safety**: Null dereference is UB. Check manually with `ptr as u16 != 0`.
- **Initialization**: Uninitialized pointers contain garbage. SNES RAM is unpredictable at power-on.
- **Type safety**: Pointer casts (`as *T`) reinterpret memory with no validation.
- **Lifetime**: No tracking of pointer validity. Dangling pointers are the programmer's problem.

## Examples

### Memory Copy

```rust
fn memcpy(dst: *u8, src: *u8, count @ X: u8) {
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
fn memset(dst: *u8, value @ A: u8, count @ X: u8) {
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

### Linked List Traversal

```rust
struct Node {
    data: u8,
    next: *Node,
}

fn traverse(head: *Node) {
    let mut current: *Node = head;
    loop {
        if current as u16 == 0 { break; }
        process(current.data);       // Auto-deref field access
        current = current.next;
    }
}
```

### ROM Table Lookup

```rust
static SIN_TABLE: [u8; 256] = [ /* ... */ ];  // Immutable = ROM

fn get_sin(angle @ A: u8) -> u8 {
    return (&SIN_TABLE)[angle];
}
```

### DMA Transfer (Far Pointers)

```rust
far fn dma_copy_vram(channel: u16, src: far *u8, vram_offset: u16, size: u16) {
    // Set up DMA registers and transfer from ROM/RAM to VRAM
}
```

## Unsupported

- **Reference pointers**: `&name: T` is a compile error (no borrow checker)
- **Sized pointer types**: `*[u8:30]` is not allowed
- **Const pointers**: No `const *T` vs `mut *T` distinction
- **Smart pointers**: No RAII wrappers

## Type Sizes

```
Near pointer   (*T):       2 bytes (16-bit address)
Far pointer    (far *T):   3 bytes (24-bit address)
Near fn ptr    (fn()):     2 bytes
Far fn ptr     (far fn()): 3 bytes
```

Pointers are just addresses — no metadata, no bounds, no ownership.

---

**STATUS**: Implemented
**Last Updated**: 2026-02-05
