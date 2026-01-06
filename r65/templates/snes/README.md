# {{PROJECT_NAME}}

A new SNES project created with R65.

## Project Structure

```
{{PROJECT_NAME}}/
├── src/
│   ├── main.r65       # Main program entry point (includes hardware registers)
│   ├── hardware.r65   # Complete SNES hardware register definitions
│   └── data.r65       # Graphics and palette data
├── Makefile           # Build system for WLA-DX
├── build/             # Generated assembly files
├── dist/              # Final ROM files
└── r65x.json          # Project configuration
```

## Getting Started

### Build the Project

```bash
# Using the Makefile (recommended)
make                    # Build ROM
make debug             # Build with verbose output
make clean             # Clean build artifacts
make install-deps      # Install WLA-DX dependencies

# Using r65x tool
r65x build            # Build using r65x (uses Makefile when available)
r65x clean            # Clean build artifacts

# Manual compilation
r65c src/main.r65 -o build/main.asm
```

### Development Workflow

1. Edit `.r65` source files in the `src/` directory
2. Run `r65x build` to compile to assembly
3. Use WLA-DX to assemble to ROM:
   ```bash
   wla-65816 -o main.o build/main.asm
   wlalink linkfile.txt game.smc
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
    // Set background color via hardware register
    BG1HOFS = color;
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
- **Graphics**: Mode 0 (4 background layers, 8 colors per palette)
- **Assembly Output**: WLA-DX compatible assembly

## Hardware References

- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)
- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)

## Useful Commands

```bash
# Initialize new project
r65x init --platform=snes my_game

# Build with verbose output
r65x build --verbose

# Clean build artifacts
r65x clean

# Compile with debugging info
r65c src/main.r65 -o build/main.asm --dump-ast
```