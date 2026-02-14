# {{PROJECT_NAME}} - AI Assistant Guide

SNES game project using the R65 compiler. For full R65 language reference, see the main CLAUDE.md in the R65 repository.

## Build Commands

```bash
make          # Build ROM
make run      # Build and launch in emulator
make clean    # Clean build artifacts
make debug    # Build with verbose output
make syntax   # Check syntax only (no ROM generation)
```

## Project Structure

```
{{PROJECT_NAME}}/
├── src/
│   ├── main.r65              # Main entry point (#[entry] fn main)
│   └── lib/                  # Standard libraries
│       ├── 65816.r65         # CPU register definitions
│       ├── sneslib.r65       # SNES hardware registers and helpers
│       └── math.r65          # Math functions (mul8, div8, etc.)
├── build/                    # Compiled output (.asm, .smc, .dbg)
└── Makefile                  # Build system for WLA-DX
```

## Key R65 Patterns for SNES

### Register Binding

```rust
let value @ A = 42;           // Bind variable to A register
fn process(x @ A: u8);        // Parameter in A register
fn indexed(idx @ X: u16);     // X/Y parameters must be u16
```

### Hardware Register Access

```rust
INIDISP = 0x0F;               // Direct write to PPU register
let status = HVBJOY;          // Read from hardware
```

### VBlank Synchronization

```rust
loop {
    wait_nmi();               // Wait for VBlank interrupt
    // Update game state here
}
```

### DMA Transfers

```rust
dma_vram(0, &DATA, dest_addr, size);  // Transfer to VRAM
```

### Memory Storage Classes

```rust
static ROM_DATA: [u8; 16] = [0; 16];  // Immutable = ROM (no attribute)

#[zeropage]
static mut FAST_VAR: u8;              // Zero page (fastest access)

#[lowram]
static mut LOW_VAR: u16;              // Low RAM ($0000-$1FFF)

#[ram]
static mut BUFFER: [u8; 256];         // Main RAM ($7E2000+)

#[hw(0x2100)]
static mut INIDISP: u8;               // Hardware register (volatile)
```

## Common Pitfalls

- Always call `snes_init()` before accessing PPU registers
- Screen must be off (`INIDISP = 0x80`) during VRAM writes outside VBlank
- X/Y registers are always 16-bit (u16), never use u8 for X/Y parameters
- Use `#[ram]` for mutable statics, `#[zeropage]` for fast access variables
- No implicit type conversions - use explicit `as` casts
- Multiply/divide operators only work with constants 1, 2, 4, 8 - use `mul8()`, `div8()` for variables
- Shift operators require constant amounts - use `shl()`, `shr()` for variable shifts

## Debugging

- Build with `--dbg` flag (default in Makefile) for Mesen source debugging
- Load `.dbg` file alongside ROM in Mesen for source-level breakpoints
- Use `asm!("BRK")` to trigger a breakpoint in emulators

## R65 vs Rust Quick Reference

| Rust | R65 | Notes |
|------|-----|-------|
| `fn foo(x: u8)` | `fn foo(x @ A: u8)` | Register binding for parameters |
| `let x = 5;` | `let x @ A = 5;` | Optional register binding |
| `x * y` | `mul8(x, y)` | Variable multiplication |
| `x << n` | `shl(x, n)` | Variable shift amount |
| `unsafe { }` | *(not needed)* | All code has hardware access |
| `Option<T>` | *(not available)* | Use sentinel values |
| `Result<T, E>` | *(not available)* | Use return codes |

## Hardware References

- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)
- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
