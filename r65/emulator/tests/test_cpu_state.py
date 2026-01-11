"""
Tests for CPU initialization and state management.
"""

import pytest
from r65.emulator.cpu import CPU65816, StopExecution, WaitForInterrupt
from r65.emulator.memory import Memory


class TestCPUInitialization:
    """Test CPU initialization and reset."""

    def test_initial_state(self, memory):
        """CPU should initialize with correct default values."""
        cpu = CPU65816(memory)

        assert cpu.A == 0
        assert cpu.X == 0
        assert cpu.Y == 0
        assert cpu.SP == 0x01FF
        assert cpu.D == 0
        assert cpu.DBR == 0
        assert cpu.PBR == 0
        assert cpu.emulation_mode is True
        assert cpu.cycles == 0
        assert cpu.stopped is False
        assert cpu.waiting is False

    def test_reset(self, cpu):
        """Reset should restore initial state."""
        # Modify state
        cpu.A = 0x1234
        cpu.X = 0x5678
        cpu.Y = 0x9ABC
        cpu.cycles = 1000
        cpu.stopped = True

        cpu.reset()

        assert cpu.A == 0
        assert cpu.X == 0
        assert cpu.Y == 0
        assert cpu.SP == 0x01FF
        assert cpu.cycles == 0
        assert cpu.stopped is False
        assert cpu.emulation_mode is True

    def test_instruction_table_size(self, cpu):
        """Instruction table should have 256 entries."""
        assert len(cpu._instructions) == 256

    def test_all_opcodes_mapped(self, cpu):
        """All opcodes should have handler functions."""
        for opcode in range(256):
            handler = cpu._instructions[opcode]
            assert callable(handler), f"Opcode 0x{opcode:02X} not callable"


class TestStatusFlags:
    """Test processor status flag operations."""

    def test_carry_flag(self, cpu):
        """Test carry flag get/set."""
        cpu.flag_c = True
        assert cpu.flag_c is True
        assert cpu.P & 0x01 == 0x01

        cpu.flag_c = False
        assert cpu.flag_c is False
        assert cpu.P & 0x01 == 0x00

    def test_zero_flag(self, cpu):
        """Test zero flag get/set."""
        cpu.flag_z = True
        assert cpu.flag_z is True
        assert cpu.P & 0x02 == 0x02

        cpu.flag_z = False
        assert cpu.flag_z is False

    def test_interrupt_flag(self, cpu):
        """Test interrupt disable flag get/set."""
        cpu.flag_i = True
        assert cpu.flag_i is True
        assert cpu.P & 0x04 == 0x04

        cpu.flag_i = False
        assert cpu.flag_i is False

    def test_decimal_flag(self, cpu):
        """Test decimal mode flag get/set."""
        cpu.flag_d = True
        assert cpu.flag_d is True
        assert cpu.P & 0x08 == 0x08

        cpu.flag_d = False
        assert cpu.flag_d is False

    def test_index_size_flag(self, cpu):
        """Test index register size flag."""
        cpu.X = 0x1234
        cpu.Y = 0x5678

        # Switch to 8-bit index mode
        cpu.flag_x = True
        assert cpu.flag_x is True
        assert cpu.P & 0x10 == 0x10
        # X and Y should be truncated
        assert cpu.X == 0x34
        assert cpu.Y == 0x78

        cpu.flag_x = False
        assert cpu.flag_x is False

    def test_memory_size_flag(self, cpu):
        """Test accumulator size flag."""
        cpu.flag_m = True
        assert cpu.flag_m is True
        assert cpu.P & 0x20 == 0x20

        cpu.flag_m = False
        assert cpu.flag_m is False

    def test_overflow_flag(self, cpu):
        """Test overflow flag get/set."""
        cpu.flag_v = True
        assert cpu.flag_v is True
        assert cpu.P & 0x40 == 0x40

        cpu.flag_v = False
        assert cpu.flag_v is False

    def test_negative_flag(self, cpu):
        """Test negative flag get/set."""
        cpu.flag_n = True
        assert cpu.flag_n is True
        assert cpu.P & 0x80 == 0x80

        cpu.flag_n = False
        assert cpu.flag_n is False


class TestRegisterSizes:
    """Test register size properties."""

    def test_acc_size_8bit(self, cpu_8bit):
        """Accumulator size should be 1 in 8-bit mode."""
        assert cpu_8bit.acc_size == 1
        assert cpu_8bit.acc_mask == 0xFF

    def test_acc_size_16bit(self, cpu):
        """Accumulator size should be 2 in 16-bit mode."""
        assert cpu.acc_size == 2
        assert cpu.acc_mask == 0xFFFF

    def test_idx_size_8bit(self, cpu_8bit):
        """Index size should be 1 in 8-bit mode."""
        assert cpu_8bit.idx_size == 1
        assert cpu_8bit.idx_mask == 0xFF

    def test_idx_size_16bit(self, cpu):
        """Index size should be 2 in 16-bit mode."""
        assert cpu.idx_size == 2
        assert cpu.idx_mask == 0xFFFF


class TestStackOperations:
    """Test stack push/pull operations."""

    def test_push_byte(self, cpu):
        """Push byte should decrement SP and write to stack."""
        cpu.SP = 0x1FFF
        cpu.push_byte(0x42)

        assert cpu.SP == 0x1FFE
        assert cpu.memory.read(0, 0x1FFF) == 0x42

    def test_pull_byte(self, cpu):
        """Pull byte should read from stack and increment SP."""
        cpu.SP = 0x1FFE
        cpu.memory.write(0, 0x1FFF, 0x42)

        value = cpu.pull_byte()

        assert value == 0x42
        assert cpu.SP == 0x1FFF

    def test_push_word(self, cpu):
        """Push word should push high byte first, then low byte."""
        cpu.SP = 0x1FFF
        cpu.push_word(0x1234)

        assert cpu.SP == 0x1FFD
        # High byte at higher address
        assert cpu.memory.read(0, 0x1FFF) == 0x12
        # Low byte at lower address
        assert cpu.memory.read(0, 0x1FFE) == 0x34

    def test_pull_word(self, cpu):
        """Pull word should pull low byte first, then high byte."""
        cpu.SP = 0x1FFD
        cpu.memory.write(0, 0x1FFE, 0x34)  # Low byte
        cpu.memory.write(0, 0x1FFF, 0x12)  # High byte

        value = cpu.pull_word()

        assert value == 0x1234
        assert cpu.SP == 0x1FFF

    def test_emulation_mode_stack_wrap(self, cpu_emulation):
        """Stack should wrap within page 1 in emulation mode."""
        cpu_emulation.SP = 0x0100
        cpu_emulation.push_byte(0x42)

        # Should wrap to $01FF
        assert cpu_emulation.SP == 0x01FF


class TestNZFlags:
    """Test N and Z flag setting."""

    def test_set_nz_zero_8bit(self, cpu):
        """Zero value should set Z flag in 8-bit mode."""
        cpu.set_nz_flags(0x00, False)
        assert cpu.flag_z is True
        assert cpu.flag_n is False

    def test_set_nz_negative_8bit(self, cpu):
        """Negative value should set N flag in 8-bit mode."""
        cpu.set_nz_flags(0x80, False)
        assert cpu.flag_z is False
        assert cpu.flag_n is True

    def test_set_nz_positive_8bit(self, cpu):
        """Positive non-zero value should clear both flags in 8-bit mode."""
        cpu.set_nz_flags(0x42, False)
        assert cpu.flag_z is False
        assert cpu.flag_n is False

    def test_set_nz_zero_16bit(self, cpu):
        """Zero value should set Z flag in 16-bit mode."""
        cpu.set_nz_flags(0x0000, True)
        assert cpu.flag_z is True
        assert cpu.flag_n is False

    def test_set_nz_negative_16bit(self, cpu):
        """Negative value should set N flag in 16-bit mode."""
        cpu.set_nz_flags(0x8000, True)
        assert cpu.flag_z is False
        assert cpu.flag_n is True


class TestStepExceptions:
    """Test step method exception handling."""

    def test_step_when_stopped(self, cpu):
        """Step should raise StopExecution when CPU is stopped."""
        cpu.stopped = True

        with pytest.raises(StopExecution):
            cpu.step()

    def test_step_when_waiting(self, cpu):
        """Step should raise WaitForInterrupt when CPU is waiting."""
        cpu.waiting = True

        with pytest.raises(WaitForInterrupt):
            cpu.step()
