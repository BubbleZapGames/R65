"""
Pytest fixtures for emulator tests.
"""

import pytest
from r65.emulator.cpu import CPU65816
from r65.emulator.memory import Memory, SNESMemory


@pytest.fixture
def empty_rom() -> bytes:
    """32KB empty ROM."""
    return bytes(32768)


@pytest.fixture
def memory(empty_rom) -> SNESMemory:
    """Memory instance with empty ROM."""
    return SNESMemory(empty_rom)


@pytest.fixture
def cpu(memory) -> CPU65816:
    """CPU instance in native mode (16-bit registers)."""
    cpu = CPU65816(memory)
    cpu.emulation_mode = False
    cpu.flag_m = False  # 16-bit accumulator
    cpu.flag_x = False  # 16-bit index
    cpu.PC = 0x8000
    cpu.PBR = 0
    cpu.DBR = 0
    cpu.D = 0
    cpu.SP = 0x1FFF
    return cpu


@pytest.fixture
def cpu_8bit(memory) -> CPU65816:
    """CPU instance in 8-bit mode."""
    cpu = CPU65816(memory)
    cpu.emulation_mode = False
    cpu.flag_m = True   # 8-bit accumulator
    cpu.flag_x = True   # 8-bit index
    cpu.PC = 0x8000
    cpu.PBR = 0
    cpu.DBR = 0
    cpu.D = 0
    cpu.SP = 0x1FFF
    return cpu


@pytest.fixture
def cpu_emulation(memory) -> CPU65816:
    """CPU instance in emulation mode (6502 compatible)."""
    cpu = CPU65816(memory)
    cpu.emulation_mode = True
    cpu.PC = 0x8000
    cpu.PBR = 0
    cpu.DBR = 0
    cpu.D = 0
    cpu.SP = 0x01FF
    return cpu


def load_program(memory: SNESMemory, program: bytes, start: int = 0x8000) -> None:
    """Load a program into ROM at the specified address."""
    # Calculate offset into ROM based on LoROM mapping
    # In LoROM, $8000-$FFFF maps to ROM
    rom_offset = start - 0x8000
    for i, byte in enumerate(program):
        if rom_offset + i < len(memory.rom):
            memory.rom[rom_offset + i] = byte
