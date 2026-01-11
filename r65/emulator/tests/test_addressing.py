"""
Tests for addressing mode handlers.
"""

import pytest
from r65.emulator import addressing as addr
from r65.emulator.cpu import CPU65816
from r65.emulator.memory import Memory


class TestImmediateAddressing:
    """Test immediate addressing modes."""

    def test_immediate_8(self, cpu_8bit):
        """Immediate 8-bit should fetch one byte."""
        cpu_8bit.memory.rom[0] = 0x42
        cpu_8bit.PC = 0x8000

        value, extra = addr.immediate_8(cpu_8bit)

        assert value == 0x42
        assert extra == 0
        assert cpu_8bit.PC == 0x8001

    def test_immediate_16(self, cpu):
        """Immediate 16-bit should fetch two bytes."""
        cpu.memory.rom[0] = 0x34  # Low byte
        cpu.memory.rom[1] = 0x12  # High byte
        cpu.PC = 0x8000

        value, extra = addr.immediate_16(cpu)

        assert value == 0x1234
        assert extra == 0
        assert cpu.PC == 0x8002

    def test_immediate_acc_8bit(self, cpu_8bit):
        """Immediate accumulator should be 8-bit when M=1."""
        cpu_8bit.memory.rom[0] = 0x42
        cpu_8bit.PC = 0x8000

        value, extra = addr.immediate_acc(cpu_8bit)

        assert value == 0x42
        assert cpu_8bit.PC == 0x8001

    def test_immediate_acc_16bit(self, cpu):
        """Immediate accumulator should be 16-bit when M=0."""
        cpu.memory.rom[0] = 0x34
        cpu.memory.rom[1] = 0x12
        cpu.PC = 0x8000

        value, extra = addr.immediate_acc(cpu)

        assert value == 0x1234
        assert cpu.PC == 0x8002

    def test_immediate_idx_8bit(self, cpu_8bit):
        """Immediate index should be 8-bit when X=1."""
        cpu_8bit.memory.rom[0] = 0x42
        cpu_8bit.PC = 0x8000

        value, extra = addr.immediate_idx(cpu_8bit)

        assert value == 0x42

    def test_immediate_idx_16bit(self, cpu):
        """Immediate index should be 16-bit when X=0."""
        cpu.memory.rom[0] = 0x34
        cpu.memory.rom[1] = 0x12
        cpu.PC = 0x8000

        value, extra = addr.immediate_idx(cpu)

        assert value == 0x1234


class TestAbsoluteAddressing:
    """Test absolute addressing modes."""

    def test_absolute(self, cpu):
        """Absolute should use DBR as bank."""
        cpu.memory.rom[0] = 0x00  # Low byte
        cpu.memory.rom[1] = 0x20  # High byte -> $2000
        cpu.DBR = 0x7E
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute(cpu)

        assert bank == 0x7E
        assert address == 0x2000
        assert extra == 0
        assert cpu.PC == 0x8002

    def test_absolute_long(self, cpu):
        """Absolute long should include bank byte."""
        cpu.memory.rom[0] = 0x00  # Low byte
        cpu.memory.rom[1] = 0x20  # High byte
        cpu.memory.rom[2] = 0x7E  # Bank
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_long(cpu)

        assert bank == 0x7E
        assert address == 0x2000
        assert cpu.PC == 0x8003

    def test_absolute_x(self, cpu):
        """Absolute indexed X should add X to address."""
        cpu.memory.rom[0] = 0x00
        cpu.memory.rom[1] = 0x20  # $2000
        cpu.X = 0x0010
        cpu.DBR = 0x00
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_x(cpu)

        assert bank == 0x00
        assert address == 0x2010
        assert cpu.PC == 0x8002

    def test_absolute_x_page_cross(self, cpu):
        """Absolute indexed X should add extra cycle on page cross."""
        cpu.memory.rom[0] = 0xF0
        cpu.memory.rom[1] = 0x20  # $20F0
        cpu.X = 0x0020  # Crosses to $2110
        cpu.DBR = 0x00
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_x(cpu)

        assert address == 0x2110
        assert extra == 1

    def test_absolute_y(self, cpu):
        """Absolute indexed Y should add Y to address."""
        cpu.memory.rom[0] = 0x00
        cpu.memory.rom[1] = 0x20
        cpu.Y = 0x0010
        cpu.DBR = 0x00
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_y(cpu)

        assert address == 0x2010

    def test_absolute_long_x(self, cpu):
        """Absolute long indexed X should handle 24-bit address."""
        cpu.memory.rom[0] = 0xF0
        cpu.memory.rom[1] = 0xFF  # $00FFF0
        cpu.memory.rom[2] = 0x00
        cpu.X = 0x0020
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_long_x(cpu)

        # $00FFF0 + $20 = $010010
        assert bank == 0x01
        assert address == 0x0010


class TestDirectPageAddressing:
    """Test direct page addressing modes."""

    def test_direct(self, cpu):
        """Direct page should add offset to D register."""
        cpu.memory.rom[0] = 0x10  # Offset
        cpu.D = 0x0200
        cpu.PC = 0x8000

        bank, address, extra = addr.direct(cpu)

        assert bank == 0
        assert address == 0x0210
        assert extra == 0  # D is page-aligned at $0200

    def test_direct_d_not_aligned(self, cpu):
        """Direct page should add extra cycle if D not page-aligned."""
        cpu.memory.rom[0] = 0x10
        cpu.D = 0x0201  # Not page-aligned
        cpu.PC = 0x8000

        bank, address, extra = addr.direct(cpu)

        assert address == 0x0211
        assert extra == 1

    def test_direct_x(self, cpu):
        """Direct page indexed X should add X to address."""
        cpu.memory.rom[0] = 0x10
        cpu.D = 0x0200
        cpu.X = 0x0005
        cpu.PC = 0x8000

        bank, address, extra = addr.direct_x(cpu)

        assert address == 0x0215

    def test_direct_y(self, cpu):
        """Direct page indexed Y should add Y to address."""
        cpu.memory.rom[0] = 0x10
        cpu.D = 0x0200
        cpu.Y = 0x0005
        cpu.PC = 0x8000

        bank, address, extra = addr.direct_y(cpu)

        assert address == 0x0215


class TestIndirectAddressing:
    """Test indirect addressing modes."""

    def test_direct_indirect(self, cpu):
        """Direct page indirect should read pointer from DP."""
        cpu.memory.rom[0] = 0x10  # DP offset
        cpu.D = 0x0000
        cpu.DBR = 0x7E
        # Set up pointer at $0010
        cpu.memory.write(0, 0x0010, 0x00)  # Low byte
        cpu.memory.write(0, 0x0011, 0x30)  # High byte -> $3000
        cpu.PC = 0x8000

        bank, address, extra = addr.direct_indirect(cpu)

        assert bank == 0x7E  # Uses DBR
        assert address == 0x3000

    def test_direct_indirect_long(self, cpu):
        """Direct page indirect long should read 24-bit pointer."""
        cpu.memory.rom[0] = 0x10
        cpu.D = 0x0000
        # Set up 24-bit pointer at $0010
        cpu.memory.write(0, 0x0010, 0x00)  # Low
        cpu.memory.write(0, 0x0011, 0x30)  # High
        cpu.memory.write(0, 0x0012, 0x7E)  # Bank
        cpu.PC = 0x8000

        bank, address, extra = addr.direct_indirect_long(cpu)

        assert bank == 0x7E
        assert address == 0x3000

    def test_direct_x_indirect(self, cpu):
        """Direct indexed indirect X should index before indirection."""
        cpu.memory.rom[0] = 0x10
        cpu.D = 0x0000
        cpu.X = 0x0004
        cpu.DBR = 0x00
        # Pointer at $0014
        cpu.memory.write(0, 0x0014, 0x00)
        cpu.memory.write(0, 0x0015, 0x30)
        cpu.PC = 0x8000

        bank, address, extra = addr.direct_x_indirect(cpu)

        assert address == 0x3000

    def test_direct_indirect_y(self, cpu):
        """Direct indirect indexed Y should index after indirection."""
        cpu.memory.rom[0] = 0x10
        cpu.D = 0x0000
        cpu.Y = 0x0004
        cpu.DBR = 0x00
        # Pointer at $0010 -> $3000
        cpu.memory.write(0, 0x0010, 0x00)
        cpu.memory.write(0, 0x0011, 0x30)
        cpu.PC = 0x8000

        bank, address, extra = addr.direct_indirect_y(cpu)

        assert address == 0x3004  # $3000 + Y


class TestStackRelativeAddressing:
    """Test stack relative addressing modes."""

    def test_stack_relative(self, cpu):
        """Stack relative should add offset to SP."""
        cpu.memory.rom[0] = 0x05  # Offset
        cpu.SP = 0x1FF0
        cpu.PC = 0x8000

        bank, address, extra = addr.stack_relative(cpu)

        assert bank == 0
        assert address == 0x1FF5

    def test_stack_relative_indirect_y(self, cpu):
        """Stack relative indirect indexed Y should use pointer from stack."""
        cpu.memory.rom[0] = 0x05  # Offset
        cpu.SP = 0x1FF0
        cpu.Y = 0x0002
        cpu.DBR = 0x7E
        # Pointer at SP+5 = $1FF5
        cpu.memory.write(0, 0x1FF5, 0x00)
        cpu.memory.write(0, 0x1FF6, 0x30)  # -> $3000
        cpu.PC = 0x8000

        bank, address, extra = addr.stack_relative_indirect_y(cpu)

        assert bank == 0x7E
        assert address == 0x3002  # $3000 + Y


class TestRelativeAddressing:
    """Test relative (branch) addressing modes."""

    def test_relative_8_forward(self, cpu):
        """Relative 8-bit forward branch."""
        cpu.memory.rom[0] = 0x10  # +16
        cpu.PC = 0x8000

        target, extra = addr.relative_8(cpu)

        assert target == 0x8011  # PC after fetch + 16

    def test_relative_8_backward(self, cpu):
        """Relative 8-bit backward branch."""
        cpu.memory.rom[0] = 0xF0  # -16 (signed)
        cpu.PC = 0x8000

        target, extra = addr.relative_8(cpu)

        assert target == 0x7FF1  # PC after fetch - 16

    def test_relative_8_page_cross(self, cpu):
        """Relative branch should add cycle on page cross."""
        # PC=0x8080 maps to ROM offset 0x80 in LoROM
        cpu.memory.rom[0x80] = 0x7F  # +127
        cpu.PC = 0x8080

        target, extra = addr.relative_8(cpu)

        # $8081 + 127 = $8100 (page cross)
        assert extra == 1

    def test_relative_16(self, cpu):
        """Relative 16-bit branch (BRL)."""
        cpu.memory.rom[0] = 0x00
        cpu.memory.rom[1] = 0x10  # +$1000
        cpu.PC = 0x8000

        target, extra = addr.relative_16(cpu)

        assert target == 0x9002  # PC after fetch + $1000


class TestJumpAddressing:
    """Test jump/call addressing modes."""

    def test_absolute_indirect(self, cpu):
        """Absolute indirect for JMP ($nnnn)."""
        cpu.memory.rom[0] = 0x00
        cpu.memory.rom[1] = 0x10  # Pointer at $1000
        # Pointer in bank 0
        cpu.memory.write(0, 0x1000, 0x00)
        cpu.memory.write(0, 0x1001, 0x90)  # -> $9000
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_indirect(cpu)

        assert bank == cpu.PBR
        assert address == 0x9000

    def test_absolute_indirect_long(self, cpu):
        """Absolute indirect long for JMP [$nnnn]."""
        cpu.memory.rom[0] = 0x00
        cpu.memory.rom[1] = 0x10  # Pointer at $1000
        cpu.memory.write(0, 0x1000, 0x00)
        cpu.memory.write(0, 0x1001, 0x90)
        cpu.memory.write(0, 0x1002, 0x7E)  # Bank
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_indirect_long(cpu)

        assert bank == 0x7E
        assert address == 0x9000

    def test_absolute_indexed_indirect(self, cpu):
        """Absolute indexed indirect for JMP ($nnnn,X)."""
        cpu.memory.rom[0] = 0x00
        cpu.memory.rom[1] = 0x10  # Base $1000
        cpu.X = 0x0004
        cpu.PBR = 0x00
        # Pointer at $1004
        cpu.memory.write(0, 0x1004, 0x00)
        cpu.memory.write(0, 0x1005, 0x90)
        cpu.PC = 0x8000

        bank, address, extra = addr.absolute_indexed_indirect(cpu)

        assert address == 0x9000
