# {{PROJECT_NAME}}

A new SNES project created with R65. Start editing `src/main.r65`: it already
boots the console, handles VBlank (NMI) and runs an empty game loop.

## Project Structure

```
{{PROJECT_NAME}}/
├── src/
│   ├── main.r65       # Reset and NMI handlers, game loop
│   └── lib/           # Standard library (65816, sneslib, math)
├── build/             # Generated assembly, debug symbols and ROM
├── tests/             # mesen-gdb test scripts (*.script)
├── Makefile           # Build system for WLA-DX
└── r65-lint.toml      # Lint rules (r65c --lint)
```

## Getting Started

You need the R65 compiler (`r65c`) and WLA-DX (`wla-65816`, `wlalink`) on your
PATH. `make install-deps` installs WLA-DX via Homebrew.

### Build the Project

```bash
make            # Build build/{{PROJECT_NAME_UPPER}}.smc
make run        # Build and open in Mesen, higan or bsnes
make tests      # Run tests/*.script with mesen-gdb, if installed
make debug      # Clean build with verbose output
make syntax     # Compile to assembly only (no ROM)
make clean      # Remove build artifacts
```

### Manual Compilation

```bash
# Compile R65 to assembly (--dbg also writes symbols for Mesen)
r65c src/main.r65 -o build/main.asm --cfg snes --dbg -I src

# Assemble with WLA-DX
wla-65816 -I src -o build/main.o build/main.asm

# Link to ROM
printf '[objects]\nbuild/main.o\n' > linkfile.txt
wlalink -S linkfile.txt build/{{PROJECT_NAME_UPPER}}.smc
```

## R65 Language Features

- **Hardware-transparent**: CPU registers (A, X, Y, ...), banks and hardware
  registers are part of the language
- **Type safety**: Mode, size and bank mistakes are compile errors
- **Visible cost**: Every construct compiles to the instructions you would
  write by hand; expensive operations are explicit library calls
- **Rust-inspired syntax**: Structs, enums, `match`, `impl` blocks and macros

### Quick Syntax Examples

```rust
// Function with a register parameter
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

The full language reference is at [r65.dev](https://r65.dev/docs).

## Writing R65 with an AI Agent

Point your coding agent at the [R65 agent primer](https://github.com/BubbleZapGames/R65/blob/master/docs/AGENTS_R65.md)
before it writes code, or link it from this project's `CLAUDE.md` / `AGENTS.md`.
It covers the language model, a complete program, the Rust habits that fail,
the standard library's API, and how to build and test a ROM.

## SNES Specifics

This project is configured for SNES LoROM layout:
- **CPU**: 65816 (16-bit extension of 6502)
- **Memory Map**: LoROM banking ($8000-$FFFF in banks $00-$7F)
- **Assembly Output**: WLA-DX compatible assembly

## Hardware References

- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)
- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
