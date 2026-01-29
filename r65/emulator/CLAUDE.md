# 65816 CPU Emulator

This module provides a cycle-accurate 65816 CPU emulator for SNES development and testing.

## Module Structure

```
r65/emulator/
├── cpu.py          # CPU65816 class - main emulator
├── memory.py       # Memory mapping (LoROM/HiROM, WRAM, vectors)
├── addressing.py   # Addressing mode calculations
├── operations.py   # ALU operations (load, store, arithmetic, logic)
├── disasm.py       # Disassembler
├── trace.py        # Execution trace logging
├── cli.py          # Command-line interface (r65-emu)
├── compare.py      # ROM execution comparison
├── compare_cli.py  # Comparison CLI (python -m r65.emulator.compare_cli)
└── tests/          # Test suite
```

## Key Classes

### CPU65816 (`cpu.py`)

Main emulator class with:
- All 65816 registers (A, X, Y, SP, PC, PBR, DBR, D, P)
- Emulation and native mode support
- Full instruction set (256 opcodes)
- Interrupt handling (NMI, IRQ, BRK, COP)
- Vblank NMI timing support

### Memory (`memory.py`)

SNES memory mapping:
- LoROM and HiROM mapping modes
- 128KB WRAM ($7E0000-$7FFFFF)
- ROM access with bank mapping
- Vector table access (reset, NMI, IRQ)

## Testing

Tests are in `r65/emulator/tests/`. Run with:

```bash
python -m pytest r65/emulator/tests/ -v
```

### Test Organization

| File | Purpose |
|------|---------|
| `test_cpu_state.py` | CPU init, flags, registers, NMI timing |
| `test_addressing.py` | Addressing mode calculations |
| `test_operations.py` | ALU operations |
| `test_instructions.py` | Full instruction execution |
| `test_memory.py` | Memory mapping |
| `test_integration.py` | End-to-end with compiled R65 |

### Test Fixtures

From `conftest.py`:
- `memory` - Fresh Memory with 32KB ROM
- `cpu` - CPU65816 in native mode (16-bit)
- `cpu_emulation` - CPU65816 in emulation mode (8-bit)

### Writing Tests

```python
def test_lda_immediate(self, cpu):
    """LDA immediate loads value into A."""
    cpu.P |= 0x20  # M=1 (8-bit mode)
    cpu.memory.rom[0] = 0xA9  # LDA #imm
    cpu.memory.rom[1] = 0x42
    cpu.PC = 0x8000

    cpu.step()

    assert (cpu.A & 0xFF) == 0x42
```

Key patterns:
- Set PC to 0x8000 (maps to ROM offset 0 in LoROM)
- Use `cpu.step()` for single-instruction tests
- Check both result values and flags
- Test 8-bit and 16-bit modes separately

## CLI Usage

```bash
# Run a ROM
r65-emu game.sfc --trace

# With max cycles
r65-emu game.sfc --max-cycles 1000000

# Disassemble
r65-emu game.sfc --disasm 0x8000 --disasm-count 50

# Set breakpoint
r65-emu game.sfc -b 0x8100 --trace
```

## NMI Timing

Enable automatic vblank NMI:

```python
cpu = CPU65816(memory)
cpu.reset()
cpu.enable_auto_nmi(True)   # Enable scanline tracking
cpu.set_nmi_enabled(True)   # Enable NMI at vblank (NMITIMEN $4200)

# NMI triggers at scanline 225 (~48,762 cycles per frame)
```

## ROM Comparison Tool

Compare execution between two ROMs instruction-by-instruction:

```bash
# Basic comparison
python -m r65.emulator.compare_cli original.smc port.sfc

# With more instructions
python -m r65.emulator.compare_cli original.smc port.sfc --max-instructions 10000

# Verbose parallel trace
python -m r65.emulator.compare_cli original.smc port.sfc --verbose

# Enable vblank NMI timing
python -m r65.emulator.compare_cli original.smc port.sfc --enable-nmi

# Find multiple divergences
python -m r65.emulator.compare_cli original.smc port.sfc --continue-on-diverge
```

### Programmatic Usage

```python
from r65.emulator import RomComparator

comparator = RomComparator(rom1_data, rom2_data, "Original", "Port")
comparator.reset()
comparator.enable_nmi(True)

divergence = comparator.run(max_instructions=10000)
if divergence:
    comparator.format_divergence(divergence)
```

### What Gets Compared

- Opcode (same instruction executed)
- Registers A, X, Y after execution
- Status flags P
- Branch decisions (taken/not taken)

### What Gets Ignored

- Absolute addresses (different code locations expected)
- Exact cycle counts

## Architecture Notes

- Instruction dispatch uses lambda-based table for performance
- Generic handlers reduce code duplication (_load, _store, _alu, _rmw)
- Operations are pure functions in `operations.py`
- Addressing modes return (bank, address, extra_cycles) tuples
