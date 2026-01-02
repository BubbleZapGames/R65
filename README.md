# R65 Compiler

A Rust-inspired compiler for 6502/65816 processors targeting WLA-DX assembly syntax.

## Current Status

**Phase 1 (In Progress)**: Basic compiler infrastructure

- ✅ **Lexer**: Complete and tested (using Lark parser toolkit)
  - Tokenizes R65 source code
  - Recognizes keywords, operators, literals, identifiers
  - Handles hardware register names (A, X, Y, STATUS, D, DBR, PBR, S)
  - Supports comments (line and block)
  - Tracks line/column positions for error reporting
  - Grammar-based approach for easy maintenance
- ⏳ **Parser**: Ready to implement (Lark grammar already defined)
- ⏳ **Code Generation**: Not yet started

## Project Structure

```
r65/
├── compiler/
│   ├── frontend/
│   │   ├── lexer.py       # Lexical analyzer
│   │   ├── tokens.py      # Token definitions
│   │   └── __init__.py
│   ├── main.py            # CLI entry point
│   └── __init__.py
├── tests/
│   └── test_lexer.py      # Lexer test suite
├── examples/
│   └── simple.r65         # Example program
├── docs/
│   ├── operators.md       # Operator design
│   ├── control-flow.md    # Control flow design
│   └── pointers-memory.md # Memory model design
├── CLAUDE.md              # Complete language specification
└── README.md
```

## Installation

```bash
# Requires Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Lexer

Tokenize a source file:
```bash
python -m compiler.main lex examples/simple.r65
```

Tokenize from stdin:
```bash
echo "let x: u8 = 42;" | python -m compiler.main lex -
```

### Running Tests

```bash
python tests/test_lexer.py
```

## Example Program

```rust
// Simple R65 example

#[hw(0x2100)]
static mut INIDISP: u8;

#[zeropage(0x20)]
static mut COUNTER: u16 = 0;

#[mode(m8, x8)]
#[preserves(X, Y)]
fn increment_counter() -> u16 {
    let value @ A = COUNTER;
    value = value + 1;
    COUNTER = value;
    return value;
}

#[entry]
#[mode(m8, x8)]
fn main() -> ! {
    INIDISP = 0x0F;
    loop {
        increment_counter();
    }
}
```

## Language Features (Designed)

See [CLAUDE.md](CLAUDE.md) for the complete language specification, including:

- Hardware-first design with direct register access
- Type safety with mode checking (8-bit vs 16-bit)
- Zero-cost abstractions
- Register aliasing and preservation attributes
- Cross-bank function calls
- Interrupt handlers
- Hardware-aware operator design
- No runtime overhead

## Development Roadmap

- [x] **Phase 1a**: Lexer implementation
- [ ] **Phase 1b**: Parser and AST
- [ ] **Phase 1c**: Basic code generation
- [ ] **Phase 2**: Type system with mode checking
- [ ] **Phase 3**: MIR and optimization
- [ ] **Phase 4**: Full hardware features
- [ ] **Phase 5**: Standard library

## Contributing

This project is currently in early development. The language design is documented in CLAUDE.md.

## License

[To be determined]

## Documentation

- [CLAUDE.md](CLAUDE.md) - Complete language specification
- [Reserved Keywords](docs/reserved-keywords.md) - All 62 reserved keywords (Rust-compatible)
- [Operators and Cost Model](docs/operators.md) - Integer operators with hardware-aware design
- [Control Flow Structures](docs/control-flow.md) - If/else, loops, break, continue, return
- [Pointers and Memory Model](docs/pointers-memory.md) - Near/far pointers, addressing modes, memory layout
- [Implementation Log](docs/implementation-log.md) - Development progress and decisions

## References

- [WLA-DX Documentation](https://wla-dx.readthedocs.io/)
- [65816 Programming Manual](http://archive.6502.org/datasheets/wdc_65816_programming_manual.pdf)
