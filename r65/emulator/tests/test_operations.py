"""
Tests for ALU operations.
"""

import pytest
from r65.emulator import operations as ops
from r65.emulator.cpu import CPU65816
from r65.emulator.memory import Memory


class TestLoadOperations:
    """Test load register operations."""

    def test_lda8(self, cpu_8bit):
        """LDA 8-bit should preserve high byte of A."""
        cpu_8bit.A = 0xFF00
        ops.lda8(cpu_8bit, 0x42)

        assert cpu_8bit.A == 0xFF42
        assert cpu_8bit.flag_z is False
        assert cpu_8bit.flag_n is False

    def test_lda8_zero(self, cpu_8bit):
        """LDA 8-bit zero should set Z flag."""
        ops.lda8(cpu_8bit, 0x00)

        assert cpu_8bit.flag_z is True

    def test_lda8_negative(self, cpu_8bit):
        """LDA 8-bit negative should set N flag."""
        ops.lda8(cpu_8bit, 0x80)

        assert cpu_8bit.flag_n is True

    def test_lda16(self, cpu):
        """LDA 16-bit should load full word."""
        ops.lda16(cpu, 0x1234)

        assert cpu.A == 0x1234

    def test_ldx8(self, cpu_8bit):
        """LDX 8-bit should load and mask."""
        ops.ldx8(cpu_8bit, 0x42)

        assert cpu_8bit.X == 0x42

    def test_ldx16(self, cpu):
        """LDX 16-bit should load full word."""
        ops.ldx16(cpu, 0x1234)

        assert cpu.X == 0x1234

    def test_ldy8(self, cpu_8bit):
        """LDY 8-bit should load and mask."""
        ops.ldy8(cpu_8bit, 0x42)

        assert cpu_8bit.Y == 0x42

    def test_ldy16(self, cpu):
        """LDY 16-bit should load full word."""
        ops.ldy16(cpu, 0x1234)

        assert cpu.Y == 0x1234


class TestAddOperations:
    """Test ADC operations."""

    def test_adc8_simple(self, cpu_8bit):
        """ADC 8-bit simple addition."""
        cpu_8bit.A = 0x10
        cpu_8bit.flag_c = False

        ops.adc8(cpu_8bit, 0x20)

        assert (cpu_8bit.A & 0xFF) == 0x30
        assert cpu_8bit.flag_c is False
        assert cpu_8bit.flag_v is False

    def test_adc8_with_carry_in(self, cpu_8bit):
        """ADC 8-bit with carry in."""
        cpu_8bit.A = 0x10
        cpu_8bit.flag_c = True

        ops.adc8(cpu_8bit, 0x20)

        assert (cpu_8bit.A & 0xFF) == 0x31

    def test_adc8_carry_out(self, cpu_8bit):
        """ADC 8-bit should set carry on overflow."""
        cpu_8bit.A = 0xFF
        cpu_8bit.flag_c = False

        ops.adc8(cpu_8bit, 0x01)

        assert (cpu_8bit.A & 0xFF) == 0x00
        assert cpu_8bit.flag_c is True
        assert cpu_8bit.flag_z is True

    def test_adc8_overflow(self, cpu_8bit):
        """ADC 8-bit should set overflow on signed overflow."""
        cpu_8bit.A = 0x7F  # +127
        cpu_8bit.flag_c = False

        ops.adc8(cpu_8bit, 0x01)  # +1

        assert (cpu_8bit.A & 0xFF) == 0x80  # -128 (overflow)
        assert cpu_8bit.flag_v is True

    def test_adc16_simple(self, cpu):
        """ADC 16-bit simple addition."""
        cpu.A = 0x1000
        cpu.flag_c = False

        ops.adc16(cpu, 0x0234)

        assert cpu.A == 0x1234
        assert cpu.flag_c is False

    def test_adc16_carry_out(self, cpu):
        """ADC 16-bit should set carry on overflow."""
        cpu.A = 0xFFFF
        cpu.flag_c = False

        ops.adc16(cpu, 0x0001)

        assert cpu.A == 0x0000
        assert cpu.flag_c is True


class TestSubtractOperations:
    """Test SBC operations."""

    def test_sbc8_simple(self, cpu_8bit):
        """SBC 8-bit simple subtraction."""
        cpu_8bit.A = 0x30
        cpu_8bit.flag_c = True  # No borrow

        ops.sbc8(cpu_8bit, 0x10)

        assert (cpu_8bit.A & 0xFF) == 0x20
        assert cpu_8bit.flag_c is True  # No borrow needed

    def test_sbc8_with_borrow(self, cpu_8bit):
        """SBC 8-bit with borrow."""
        cpu_8bit.A = 0x30
        cpu_8bit.flag_c = False  # Borrow

        ops.sbc8(cpu_8bit, 0x10)

        assert (cpu_8bit.A & 0xFF) == 0x1F

    def test_sbc8_borrow_out(self, cpu_8bit):
        """SBC 8-bit should clear carry on underflow."""
        cpu_8bit.A = 0x00
        cpu_8bit.flag_c = True

        ops.sbc8(cpu_8bit, 0x01)

        assert (cpu_8bit.A & 0xFF) == 0xFF
        assert cpu_8bit.flag_c is False  # Borrow occurred

    def test_sbc16_simple(self, cpu):
        """SBC 16-bit simple subtraction."""
        cpu.A = 0x1234
        cpu.flag_c = True

        ops.sbc16(cpu, 0x0234)

        assert cpu.A == 0x1000


class TestCompareOperations:
    """Test compare operations."""

    def test_cmp8_equal(self, cpu_8bit):
        """CMP 8-bit equal values should set Z and C."""
        ops.cmp8(cpu_8bit, 0x42, 0x42)

        assert cpu_8bit.flag_z is True
        assert cpu_8bit.flag_c is True
        assert cpu_8bit.flag_n is False

    def test_cmp8_greater(self, cpu_8bit):
        """CMP 8-bit A > value should set C."""
        ops.cmp8(cpu_8bit, 0x50, 0x40)

        assert cpu_8bit.flag_z is False
        assert cpu_8bit.flag_c is True
        assert cpu_8bit.flag_n is False

    def test_cmp8_less(self, cpu_8bit):
        """CMP 8-bit A < value should clear C."""
        ops.cmp8(cpu_8bit, 0x40, 0x50)

        assert cpu_8bit.flag_z is False
        assert cpu_8bit.flag_c is False
        assert cpu_8bit.flag_n is True  # Result is negative

    def test_cmp16_equal(self, cpu):
        """CMP 16-bit equal values."""
        ops.cmp16(cpu, 0x1234, 0x1234)

        assert cpu.flag_z is True
        assert cpu.flag_c is True


class TestLogicOperations:
    """Test logic operations."""

    def test_and8(self, cpu_8bit):
        """AND 8-bit."""
        cpu_8bit.A = 0xFF
        ops.and8(cpu_8bit, 0x0F)

        assert (cpu_8bit.A & 0xFF) == 0x0F

    def test_and16(self, cpu):
        """AND 16-bit."""
        cpu.A = 0xFFFF
        ops.and16(cpu, 0x00FF)

        assert cpu.A == 0x00FF

    def test_ora8(self, cpu_8bit):
        """ORA 8-bit."""
        cpu_8bit.A = 0xF0
        ops.ora8(cpu_8bit, 0x0F)

        assert (cpu_8bit.A & 0xFF) == 0xFF

    def test_ora16(self, cpu):
        """ORA 16-bit."""
        cpu.A = 0xFF00
        ops.ora16(cpu, 0x00FF)

        assert cpu.A == 0xFFFF

    def test_eor8(self, cpu_8bit):
        """EOR 8-bit."""
        cpu_8bit.A = 0xFF
        ops.eor8(cpu_8bit, 0x0F)

        assert (cpu_8bit.A & 0xFF) == 0xF0

    def test_eor16(self, cpu):
        """EOR 16-bit."""
        cpu.A = 0xFFFF
        ops.eor16(cpu, 0x00FF)

        assert cpu.A == 0xFF00

    def test_bit8(self, cpu_8bit):
        """BIT 8-bit should test bits and set N/V from value."""
        cpu_8bit.A = 0x0F
        ops.bit8(cpu_8bit, 0xC0)  # Bits 7 and 6 set

        assert cpu_8bit.flag_z is True  # A & value = 0
        assert cpu_8bit.flag_n is True  # Bit 7 of value
        assert cpu_8bit.flag_v is True  # Bit 6 of value

    def test_bit8_no_nv(self, cpu_8bit):
        """BIT immediate should not set N/V."""
        cpu_8bit.A = 0x0F
        ops.bit8(cpu_8bit, 0xC0, set_nv=False)

        assert cpu_8bit.flag_z is True
        assert cpu_8bit.flag_n is False
        assert cpu_8bit.flag_v is False


class TestShiftOperations:
    """Test shift and rotate operations."""

    def test_asl8(self, cpu_8bit):
        """ASL 8-bit should shift left."""
        result = ops.asl8(cpu_8bit, 0x40)

        assert result == 0x80
        assert cpu_8bit.flag_c is False
        assert cpu_8bit.flag_n is True

    def test_asl8_carry(self, cpu_8bit):
        """ASL 8-bit should set carry from bit 7."""
        result = ops.asl8(cpu_8bit, 0x80)

        assert result == 0x00
        assert cpu_8bit.flag_c is True
        assert cpu_8bit.flag_z is True

    def test_lsr8(self, cpu_8bit):
        """LSR 8-bit should shift right."""
        result = ops.lsr8(cpu_8bit, 0x02)

        assert result == 0x01
        assert cpu_8bit.flag_c is False

    def test_lsr8_carry(self, cpu_8bit):
        """LSR 8-bit should set carry from bit 0."""
        result = ops.lsr8(cpu_8bit, 0x01)

        assert result == 0x00
        assert cpu_8bit.flag_c is True

    def test_rol8(self, cpu_8bit):
        """ROL 8-bit should rotate left through carry."""
        cpu_8bit.flag_c = True
        result = ops.rol8(cpu_8bit, 0x00)

        assert result == 0x01  # Carry rotated in
        assert cpu_8bit.flag_c is False

    def test_rol8_out(self, cpu_8bit):
        """ROL 8-bit should shift bit 7 into carry."""
        cpu_8bit.flag_c = False
        result = ops.rol8(cpu_8bit, 0x80)

        assert result == 0x00
        assert cpu_8bit.flag_c is True

    def test_ror8(self, cpu_8bit):
        """ROR 8-bit should rotate right through carry."""
        cpu_8bit.flag_c = True
        result = ops.ror8(cpu_8bit, 0x00)

        assert result == 0x80  # Carry rotated in
        assert cpu_8bit.flag_c is False

    def test_ror8_out(self, cpu_8bit):
        """ROR 8-bit should shift bit 0 into carry."""
        cpu_8bit.flag_c = False
        result = ops.ror8(cpu_8bit, 0x01)

        assert result == 0x00
        assert cpu_8bit.flag_c is True

    def test_asl16(self, cpu):
        """ASL 16-bit."""
        result = ops.asl16(cpu, 0x8000)

        assert result == 0x0000
        assert cpu.flag_c is True

    def test_lsr16(self, cpu):
        """LSR 16-bit."""
        result = ops.lsr16(cpu, 0x0001)

        assert result == 0x0000
        assert cpu.flag_c is True


class TestIncDecOperations:
    """Test increment and decrement operations."""

    def test_inc8(self, cpu_8bit):
        """INC 8-bit."""
        result = ops.inc8(cpu_8bit, 0x42)

        assert result == 0x43

    def test_inc8_wrap(self, cpu_8bit):
        """INC 8-bit should wrap at 255."""
        result = ops.inc8(cpu_8bit, 0xFF)

        assert result == 0x00
        assert cpu_8bit.flag_z is True

    def test_dec8(self, cpu_8bit):
        """DEC 8-bit."""
        result = ops.dec8(cpu_8bit, 0x42)

        assert result == 0x41

    def test_dec8_wrap(self, cpu_8bit):
        """DEC 8-bit should wrap at 0."""
        result = ops.dec8(cpu_8bit, 0x00)

        assert result == 0xFF
        assert cpu_8bit.flag_n is True

    def test_inc16(self, cpu):
        """INC 16-bit."""
        result = ops.inc16(cpu, 0x1234)

        assert result == 0x1235

    def test_dec16(self, cpu):
        """DEC 16-bit."""
        result = ops.dec16(cpu, 0x1234)

        assert result == 0x1233
