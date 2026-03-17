# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
        assert cpu.memory.read(0x1FFF) == 0x42

    def test_pull_byte(self, cpu):
        """Pull byte should read from stack and increment SP."""
        cpu.SP = 0x1FFE
        cpu.memory.write(0x1FFF, 0x42)

        value = cpu.pull_byte()

        assert value == 0x42
        assert cpu.SP == 0x1FFF

    def test_push_word(self, cpu):
        """Push word should push high byte first, then low byte."""
        cpu.SP = 0x1FFF
        cpu.push_word(0x1234)

        assert cpu.SP == 0x1FFD
        # High byte at higher address
        assert cpu.memory.read(0x1FFF) == 0x12
        # Low byte at lower address
        assert cpu.memory.read(0x1FFE) == 0x34

    def test_pull_word(self, cpu):
        """Pull word should pull low byte first, then high byte."""
        cpu.SP = 0x1FFD
        cpu.memory.write(0x1FFE, 0x34)  # Low byte
        cpu.memory.write(0x1FFF, 0x12)  # High byte

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


class TestNMITiming:
    """Test vblank NMI timing functionality."""

    def test_auto_nmi_disabled_by_default(self, cpu):
        """Auto NMI should be disabled by default."""
        assert cpu.auto_nmi is False
        assert cpu.nmi_enabled is False

    def test_enable_auto_nmi(self, cpu):
        """Should be able to enable auto NMI timing."""
        cpu.enable_auto_nmi(True)
        assert cpu.auto_nmi is True

        cpu.enable_auto_nmi(False)
        assert cpu.auto_nmi is False

    def test_set_nmi_enabled(self, cpu):
        """Should be able to enable/disable NMI generation."""
        cpu.set_nmi_enabled(True)
        assert cpu.nmi_enabled is True

        cpu.set_nmi_enabled(False)
        assert cpu.nmi_enabled is False

    def test_set_region_ntsc(self, cpu):
        """NTSC region should have 262 scanlines."""
        cpu.set_region(pal=False)
        assert cpu.scanlines_per_frame == 262

    def test_set_region_pal(self, cpu):
        """PAL region should have 312 scanlines."""
        cpu.set_region(pal=True)
        assert cpu.scanlines_per_frame == 312

    def test_scanline_advances_with_cycles(self, cpu):
        """Scanlines should advance as cycles accumulate."""
        cpu.enable_auto_nmi(True)
        cpu.scanline = 0
        cpu.scanline_cycles = 0

        # Execute enough NOPs to advance one scanline (~186 cycles)
        # NOP = 2 cycles, so 93 NOPs = 186 cycles
        cpu.memory.rom[0] = 0xEA  # NOP
        for _ in range(93):
            cpu.PC = 0x8000
            cpu.step()

        assert cpu.scanline == 1

    def test_vblank_flag_set_at_scanline_225(self, cpu):
        """Vblank flag should be set when reaching scanline 225."""
        cpu.enable_auto_nmi(True)
        cpu.scanline = 224
        cpu.scanline_cycles = 185  # Almost at next scanline

        # Execute one NOP to cross into scanline 225
        cpu.memory.rom[0] = 0xEA  # NOP
        cpu.PC = 0x8000
        cpu.step()

        assert cpu.scanline == 225
        assert cpu.vblank_flag is True

    def test_nmi_triggers_at_vblank_when_enabled(self, cpu):
        """NMI should trigger at vblank when nmi_enabled is True."""
        cpu.enable_auto_nmi(True)
        cpu.set_nmi_enabled(True)
        cpu.scanline = 224
        cpu.scanline_cycles = 185
        cpu.emulation_mode = False  # Native mode for cleaner test
        cpu.SP = 0x1FFF

        # Set up NMI vector
        cpu.memory.rom[0x7FEA] = 0x00  # NMI vector low
        cpu.memory.rom[0x7FEB] = 0x90  # NMI vector high ($9000)

        original_pc = cpu.PC = 0x8000
        cpu.memory.rom[0] = 0xEA  # NOP
        cpu.step()

        # PC should now be at NMI handler
        assert cpu.PC == 0x9000
        # Original PC should be on stack
        assert cpu.SP < 0x1FFF  # Stack was used

    def test_nmi_does_not_trigger_when_disabled(self, cpu):
        """NMI should not trigger at vblank when nmi_enabled is False."""
        cpu.enable_auto_nmi(True)
        cpu.set_nmi_enabled(False)  # NMI disabled
        cpu.scanline = 224
        cpu.scanline_cycles = 185
        cpu.SP = 0x1FFF

        cpu.memory.rom[0] = 0xEA  # NOP
        cpu.PC = 0x8000
        cpu.step()

        # Vblank flag should still be set
        assert cpu.vblank_flag is True
        # But NMI should not have triggered - PC continues normally
        assert cpu.PC == 0x8001
        assert cpu.SP == 0x1FFF  # Stack unchanged

    def test_read_nmi_flag_clears_flag(self, cpu):
        """Reading NMI flag should return and clear it."""
        cpu.vblank_flag = True

        result = cpu.read_nmi_flag()

        assert result is True
        assert cpu.vblank_flag is False

        # Second read should return False
        result = cpu.read_nmi_flag()
        assert result is False

    def test_in_vblank_property(self, cpu):
        """in_vblank property should reflect scanline position."""
        cpu.scanline = 100
        assert cpu.in_vblank is False

        cpu.scanline = 224
        assert cpu.in_vblank is False

        cpu.scanline = 225
        assert cpu.in_vblank is True

        cpu.scanline = 261
        assert cpu.in_vblank is True

    def test_frame_counter_increments(self, cpu):
        """Frame counter should increment when scanline wraps."""
        cpu.enable_auto_nmi(True)
        cpu.scanline = 261
        cpu.scanline_cycles = 185
        cpu.frame = 0

        cpu.memory.rom[0] = 0xEA  # NOP
        cpu.PC = 0x8000
        cpu.step()

        assert cpu.scanline == 0
        assert cpu.frame == 1
        assert cpu.vblank_flag is False  # Cleared at frame start

    def test_reset_clears_timing_state(self, cpu):
        """Reset should clear all timing state."""
        cpu.scanline = 100
        cpu.scanline_cycles = 50
        cpu.frame = 10
        cpu.vblank_flag = True
        cpu.nmi_pending = True

        cpu.reset()

        assert cpu.scanline == 0
        assert cpu.scanline_cycles == 0
        assert cpu.frame == 0
        assert cpu.vblank_flag is False
        assert cpu.nmi_pending is False
