# proj2

A new SNES project created with R65.

## Project Structure

```
proj2/
├── src/
│   └── main.r65       # Main program entry point
├── Makefile           # Build system for WLA-DX
└── build/             # Generated assembly and ROM files
```

## Getting Started

### Build the Project

```bash
# Build ROM
make

# Build with verbose output
make debug

# Clean build artifacts
make clean

# Check syntax only (no ROM generation)
make syntax
```

### Manual Compilation

```bash
# Compile R65 to assembly
r65c src/main.r65 -o build/main.asm

# Assemble with WLA-DX
wla-65816 -o build/main.o build/main.asm

# Link to ROM
wlalink linkfile.txt build/game.smc
```

## R65 Language Features

- **Hardware-transparent**: Direct access to SNES registers (A, X, Y, etc.)
- **Type safety**: Catch bank overflow and mode errors at compile time
- **Zero-cost abstractions**: High-level code compiles to efficient assembly
- **Rust-inspired syntax**: Modern, clean language design

### Quick Syntax Examples

```rust
// Function with mode annotation
#[mode(m8, x16)]
fn set_background_color(color @ A: u8) {
    CGDATA = color;
}

// Structs and arrays
struct Player {
    x: u8,
    y: u8,
}

#[ram]
static mut PLAYERS: [Player; 4];

// Loop with break
fn wait_for_vblank() {
    loop {
        let status = HVBJOY;
        if status & 0x80 != 0 {
            break;
        }
    }
}
```

## SNES Specifics

This project is configured for SNES LoROM layout:
- **CPU**: 65816 (16-bit extension of 6502)
- **Memory Map**: LoROM banking ($8000-$FFFF in banks $00-$7F)
- **Assembly Output**: WLA-DX compatible assembly

## Hardware References

- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)
- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
