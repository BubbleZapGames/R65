# R65 Compiler

A Rust-inspired compiler for 6502/65816 processors targeting WLA-DX assembly syntax, designed specifically for SNES ROM development and reverse engineering.

## Overview

R65 brings modern type safety and clean syntax to retro hardware programming while maintaining the explicit control and zero-cost abstractions that 6502/65816 development requires. The compiler produces readable, hand-written-quality assembly that matches patterns found in professional SNES games.

**Design Philosophy:**
- **Hardware-First**: CPU registers, bank boundaries, and processor modes are first-class language concepts
- **Type Safety**: Catch bank overflow, mode mismatches, and size errors at compile time
- **Explicit Control**: No hidden costs, all hardware access is visible and direct
- **Zero Abstraction Cost**: High-level constructs compile to efficient assembly
- **Simplicity First**: Omit complex features that don't map well to hardware

## Current Status

**Implementation Progress:**

- ✅ **Lexer & Parser**: Complete with full R65 grammar support (Lark-based)
- ✅ **AST**: Complete abstract syntax tree representation
- ✅ **MIR**: Mid-level IR with CFG construction and virtual registers
- ✅ **Type System**: Mode tracking and type checking implementation
- ✅ **Code Generation**: Complete WLA-DX assembly generation pipeline
  - Memory allocation (zero-page, RAM, hardware registers)
  - Register allocation (hardware, scratch pool, stack)
  - Instruction selection (MIR → 65816 assembly)
  - Addressing mode optimization
  - Function generation with metadata
- 🚧 **Integration**: Connecting all compiler passes into working compiler
- ⏳ **Standard Library**: Not yet started

## Quick Example

```rust
// Hardware registers
#[hw(0x2100)] static mut INIDISP: u8;
#[hw(0x4200)] static mut JOYPAD1: u8;

// Zero-page variables
#[zeropage(0x20)] static mut FRAME_COUNT: u16 = 0;

// Interrupt handler with automatic register preservation
#[interrupt(nmi)]
fn vblank_handler() {
    FRAME_COUNT = FRAME_COUNT + 1;
}

// Function with register preservation guarantee
#[mode(m8, x8)]
#[preserves(X, Y)]
fn wait_vblank() {
    loop {
        let flag @ A = VBLANK_FLAG;
        if flag != 0 { break; }
    }
}

// Entry point
#[entry]
#[mode(m8, x8)]
fn main() -> ! {
    INIDISP = 0x0F;
    loop {
        wait_vblank();
        update_game();
    }
}
```

## Installation

### Requirements
- Python 3.8 or higher
- WLA-DX assembler (for assembling generated code)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/R65.git
cd R65

# Install in development mode
pip install -e .

# Or install with dev dependencies (includes pytest)
pip install -e ".[dev]"
```

### Using Requirements Files

```bash
# Core dependencies only
pip install -r requirements.txt

# Development dependencies (includes testing)
pip install -r requirements-dev.txt
```

## Usage

### Compile R65 Source

```bash
# Basic compilation
r65c examples/simple.r65 -o output.asm

# Compile to stdout
r65c examples/simple.r65

# Compile with verbose output
r65c examples/simple.r65 -o output.asm -v

# Compile from stdin
cat source.r65 | r65c - -o output.asm

# Quiet mode (only errors)
r65c examples/simple.r65 -o output.asm -q
```

### Development/Debugging

```bash
# Dump AST (parser output)
r65c examples/simple.r65 --dump-ast

# Dump HIR (high-level IR)
r65c examples/simple.r65 --dump-hir

# Dump MIR (mid-level IR)
r65c examples/simple.r65 --dump-mir

# Stop after specific phase
r65c examples/simple.r65 --stop-after typecheck

# Dump tokens (lexer output)
r65c examples/simple.r65 --dump-tokens
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=r65
```

## Language Features

R65 includes:

- **Hardware Transparency**: Direct access to A, X, Y, STATUS, D, DBR, S, PBR registers
- **Type System**: `u8`, `i8`, `u16`, `i16`, `bool` with mode-aware register types
- **Storage Classes**: `#[zeropage]`, `#[ram]`, `#[rom]`, `#[hw]` for explicit memory placement
- **Mode Tracking**: Static verification of 8-bit vs 16-bit processor modes
- **Register Aliasing**: `let name @ A = expr` for zero-cost register references
- **Function Parameters**: Three styles - register aliases, variable-bound, or stack
- **Register Preservation**: `#[preserves(...)]` to declare calling conventions
- **Interrupt Handlers**: `#[interrupt(nmi/irq/brk/cop/abort)]` with automatic preservation
- **Cross-Bank Calls**: `far fn()` for JSL/RTL with DBR management
- **Hardware-Aware Operators**: Syntax reflects performance cost (operators = fast, functions = slow)
- **Control Flow**: `if/else`, `loop`, `while`, `break`, `continue`, `return`
- **Const Evaluation**: Compile-time constant expressions
- **Inline Assembly**: Direct `asm!()` for raw 65816 instructions

## Project Structure

```
R65/
├── r65/                    # Main compiler package
│   └── compiler/
│       ├── frontend/       # Lexer, parser, AST
│       ├── hir/           # High-level IR (planned)
│       ├── typeck/        # Type checking, mode checking
│       ├── mir/           # Mid-level IR with CFG
│       ├── codegen/       # Code generation to WLA-DX
│       └── main.py        # CLI entry point
├── docs/                   # Design documentation
│   ├── operators.md       # Operator semantics
│   ├── control-flow.md    # Control flow structures
│   ├── type-system.md     # Type system and mode tracking
│   ├── calling-convention.md  # ABI and calling conventions
│   ├── code-generation.md # Complete codegen reference
│   └── ...
├── examples/              # Example R65 programs
├── CLAUDE.md             # Complete language specification
├── README.md             # This file
├── setup.py              # Python package configuration
├── requirements.txt      # Core dependencies
└── requirements-dev.txt  # Development dependencies
```

## Documentation

### Language Specification
- **[CLAUDE.md](CLAUDE.md)** - Complete language reference and design document

### Detailed Design Docs
- **[Operators](docs/operators.md)** - Hardware-aware operator design with cost model
- **[Control Flow](docs/control-flow.md)** - If/else, loops, break, continue, return semantics
- **[Type System](docs/type-system.md)** - Type checking and mode tracking
- **[B Register](docs/b-register.md)** - Hidden accumulator high byte (m8 mode only)
- **[Calling Convention](docs/calling-convention.md)** - ABI, parameters, register preservation
- **[Pointers & Memory](docs/pointers-memory.md)** - Near/far pointers, addressing modes, memory layout
- **[Code Generation](docs/code-generation.md)** - Complete codegen pipeline reference
- **[Mode Transitions](docs/mode-transition-analysis.md)** - Mode transition strategies
- **[Register Allocation](docs/register-allocation.md)** - Register allocation strategy

## Use Cases

1. **SNES Game Development**: Write new games with modern syntax and type safety
2. **ROM Reverse Engineering**: Disassemble ROMs into readable R65 source
3. **ROM Hacking**: Modify existing games with better tooling than raw assembly
4. **Education**: Learn 6502/65816 architecture with safer, clearer code

## Contributing

This project is in active development. The language design is documented in [CLAUDE.md](CLAUDE.md).

## License

[To be determined]

## References

- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
- [Super Famicom Development Wiki](https://wiki.superfamicom.org/)
- [Rust Language Reference](https://doc.rust-lang.org/reference/) (inspiration)

---

**R65** - Modern language design for retro hardware
