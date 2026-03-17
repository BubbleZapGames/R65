# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for CPU instruction execution.
"""

import pytest
from r65.emulator.cpu import CPU65816, StopExecution, WaitForInterrupt
from r65.emulator.memory import Memory
from .conftest import load_program


class TestLoadInstructions:
    """Test load instructions."""

    def test_lda_immediate_8bit(self, cpu_8bit, memory):
        """LDA #$nn in 8-bit mode."""
        load_program(memory, bytes([0xA9, 0x42]))  # LDA #$42
        cpu_8bit.PC = 0x8000

        cycles = cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x42
        assert cycles == 2
        assert cpu_8bit.PC == 0x8002

    def test_lda_immediate_16bit(self, cpu, memory):
        """LDA #$nnnn in 16-bit mode."""
        load_program(memory, bytes([0xA9, 0x34, 0x12]))  # LDA #$1234
        cpu.PC = 0x8000

        cycles = cpu.step()

        assert cpu.A == 0x1234
        assert cycles == 3
        assert cpu.PC == 0x8003

    def test_lda_absolute(self, cpu_8bit, memory):
        """LDA $nnnn."""
        load_program(memory, bytes([0xAD, 0x00, 0x10]))  # LDA $1000
        memory.write(0x1000, 0x42)
        cpu_8bit.PC = 0x8000
        cpu_8bit.DBR = 0

        cycles = cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x42
        assert cycles == 4

    def test_lda_dp(self, cpu_8bit, memory):
        """LDA $nn (direct page)."""
        load_program(memory, bytes([0xA5, 0x10]))  # LDA $10
        memory.write(0x0010, 0x42)
        cpu_8bit.PC = 0x8000
        cpu_8bit.D = 0

        cycles = cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x42
        assert cycles == 3

    def test_ldx_immediate(self, cpu_8bit, memory):
        """LDX #$nn."""
        load_program(memory, bytes([0xA2, 0x42]))  # LDX #$42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.X == 0x42

    def test_ldy_immediate(self, cpu_8bit, memory):
        """LDY #$nn."""
        load_program(memory, bytes([0xA0, 0x42]))  # LDY #$42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.Y == 0x42


class TestStoreInstructions:
    """Test store instructions."""

    def test_sta_absolute(self, cpu_8bit, memory):
        """STA $nnnn."""
        load_program(memory, bytes([0x8D, 0x00, 0x10]))  # STA $1000
        cpu_8bit.A = 0x42
        cpu_8bit.PC = 0x8000
        cpu_8bit.DBR = 0

        cycles = cpu_8bit.step()

        assert memory.read(0x1000) == 0x42
        assert cycles == 4

    def test_sta_dp(self, cpu_8bit, memory):
        """STA $nn."""
        load_program(memory, bytes([0x85, 0x10]))  # STA $10
        cpu_8bit.A = 0x42
        cpu_8bit.PC = 0x8000
        cpu_8bit.D = 0

        cpu_8bit.step()

        assert memory.read(0x0010) == 0x42

    def test_stx_absolute(self, cpu_8bit, memory):
        """STX $nnnn."""
        load_program(memory, bytes([0x8E, 0x00, 0x10]))  # STX $1000
        cpu_8bit.X = 0x42
        cpu_8bit.PC = 0x8000
        cpu_8bit.DBR = 0

        cpu_8bit.step()

        assert memory.read(0x1000) == 0x42

    def test_sty_absolute(self, cpu_8bit, memory):
        """STY $nnnn."""
        load_program(memory, bytes([0x8C, 0x00, 0x10]))  # STY $1000
        cpu_8bit.Y = 0x42
        cpu_8bit.PC = 0x8000
        cpu_8bit.DBR = 0

        cpu_8bit.step()

        assert memory.read(0x1000) == 0x42

    def test_stz_absolute(self, cpu_8bit, memory):
        """STZ $nnnn."""
        load_program(memory, bytes([0x9C, 0x00, 0x10]))  # STZ $1000
        memory.write(0x1000, 0xFF)  # Pre-fill
        cpu_8bit.PC = 0x8000
        cpu_8bit.DBR = 0

        cpu_8bit.step()

        assert memory.read(0x1000) == 0x00


class TestArithmeticInstructions:
    """Test arithmetic instructions."""

    def test_adc_immediate(self, cpu_8bit, memory):
        """ADC #$nn."""
        load_program(memory, bytes([0x69, 0x10]))  # ADC #$10
        cpu_8bit.A = 0x20
        cpu_8bit.flag_c = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x30

    def test_adc_with_carry(self, cpu_8bit, memory):
        """ADC with carry set."""
        load_program(memory, bytes([0x69, 0x10]))
        cpu_8bit.A = 0x20
        cpu_8bit.flag_c = True
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x31

    def test_sbc_immediate(self, cpu_8bit, memory):
        """SBC #$nn."""
        load_program(memory, bytes([0xE9, 0x10]))  # SBC #$10
        cpu_8bit.A = 0x30
        cpu_8bit.flag_c = True  # No borrow
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x20


class TestCompareInstructions:
    """Test compare instructions."""

    def test_cmp_immediate_equal(self, cpu_8bit, memory):
        """CMP #$nn with equal values."""
        load_program(memory, bytes([0xC9, 0x42]))  # CMP #$42
        cpu_8bit.A = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_z is True
        assert cpu_8bit.flag_c is True

    def test_cmp_immediate_greater(self, cpu_8bit, memory):
        """CMP #$nn with A > value."""
        load_program(memory, bytes([0xC9, 0x40]))  # CMP #$40
        cpu_8bit.A = 0x50
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_z is False
        assert cpu_8bit.flag_c is True

    def test_cmp_immediate_less(self, cpu_8bit, memory):
        """CMP #$nn with A < value."""
        load_program(memory, bytes([0xC9, 0x50]))  # CMP #$50
        cpu_8bit.A = 0x40
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_z is False
        assert cpu_8bit.flag_c is False

    def test_cpx_immediate(self, cpu_8bit, memory):
        """CPX #$nn."""
        load_program(memory, bytes([0xE0, 0x42]))  # CPX #$42
        cpu_8bit.X = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_z is True

    def test_cpy_immediate(self, cpu_8bit, memory):
        """CPY #$nn."""
        load_program(memory, bytes([0xC0, 0x42]))  # CPY #$42
        cpu_8bit.Y = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_z is True


class TestLogicInstructions:
    """Test logic instructions."""

    def test_and_immediate(self, cpu_8bit, memory):
        """AND #$nn."""
        load_program(memory, bytes([0x29, 0x0F]))  # AND #$0F
        cpu_8bit.A = 0xFF
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x0F

    def test_ora_immediate(self, cpu_8bit, memory):
        """ORA #$nn."""
        load_program(memory, bytes([0x09, 0x0F]))  # ORA #$0F
        cpu_8bit.A = 0xF0
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0xFF

    def test_eor_immediate(self, cpu_8bit, memory):
        """EOR #$nn."""
        load_program(memory, bytes([0x49, 0x0F]))  # EOR #$0F
        cpu_8bit.A = 0xFF
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0xF0

    def test_bit_immediate(self, cpu_8bit, memory):
        """BIT #$nn - immediate only sets Z."""
        load_program(memory, bytes([0x89, 0xC0]))  # BIT #$C0
        cpu_8bit.A = 0x0F
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_z is True
        # Immediate BIT doesn't affect N/V
        assert cpu_8bit.flag_n is False
        assert cpu_8bit.flag_v is False


class TestShiftRotateInstructions:
    """Test shift and rotate instructions."""

    def test_asl_acc(self, cpu_8bit, memory):
        """ASL A."""
        load_program(memory, bytes([0x0A]))  # ASL A
        cpu_8bit.A = 0x40
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x80

    def test_lsr_acc(self, cpu_8bit, memory):
        """LSR A."""
        load_program(memory, bytes([0x4A]))  # LSR A
        cpu_8bit.A = 0x02
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x01

    def test_rol_acc(self, cpu_8bit, memory):
        """ROL A."""
        load_program(memory, bytes([0x2A]))  # ROL A
        cpu_8bit.A = 0x80
        cpu_8bit.flag_c = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x00
        assert cpu_8bit.flag_c is True

    def test_ror_acc(self, cpu_8bit, memory):
        """ROR A."""
        load_program(memory, bytes([0x6A]))  # ROR A
        cpu_8bit.A = 0x01
        cpu_8bit.flag_c = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x00
        assert cpu_8bit.flag_c is True


class TestIncDecInstructions:
    """Test increment and decrement instructions."""

    def test_inc_acc(self, cpu_8bit, memory):
        """INC A."""
        load_program(memory, bytes([0x1A]))  # INC A
        cpu_8bit.A = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x43

    def test_dec_acc(self, cpu_8bit, memory):
        """DEC A."""
        load_program(memory, bytes([0x3A]))  # DEC A
        cpu_8bit.A = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x41

    def test_inx(self, cpu_8bit, memory):
        """INX."""
        load_program(memory, bytes([0xE8]))  # INX
        cpu_8bit.X = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.X == 0x43

    def test_dex(self, cpu_8bit, memory):
        """DEX."""
        load_program(memory, bytes([0xCA]))  # DEX
        cpu_8bit.X = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.X == 0x41

    def test_iny(self, cpu_8bit, memory):
        """INY."""
        load_program(memory, bytes([0xC8]))  # INY
        cpu_8bit.Y = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.Y == 0x43

    def test_dey(self, cpu_8bit, memory):
        """DEY."""
        load_program(memory, bytes([0x88]))  # DEY
        cpu_8bit.Y = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.Y == 0x41


class TestTransferInstructions:
    """Test transfer instructions."""

    def test_tax(self, cpu_8bit, memory):
        """TAX."""
        load_program(memory, bytes([0xAA]))  # TAX
        cpu_8bit.A = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.X == 0x42

    def test_tay(self, cpu_8bit, memory):
        """TAY."""
        load_program(memory, bytes([0xA8]))  # TAY
        cpu_8bit.A = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.Y == 0x42

    def test_txa(self, cpu_8bit, memory):
        """TXA."""
        load_program(memory, bytes([0x8A]))  # TXA
        cpu_8bit.X = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x42

    def test_tya(self, cpu_8bit, memory):
        """TYA."""
        load_program(memory, bytes([0x98]))  # TYA
        cpu_8bit.Y = 0x42
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x42

    def test_tsx(self, cpu_8bit, memory):
        """TSX."""
        load_program(memory, bytes([0xBA]))  # TSX
        cpu_8bit.SP = 0x01FF
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.X == 0xFF


class TestBranchInstructions:
    """Test branch instructions."""

    def test_bra(self, cpu_8bit, memory):
        """BRA always branches."""
        load_program(memory, bytes([0x80, 0x10]))  # BRA +16
        cpu_8bit.PC = 0x8000

        cycles = cpu_8bit.step()

        assert cpu_8bit.PC == 0x8012  # 0x8002 + 16
        assert cycles >= 3

    def test_beq_taken(self, cpu_8bit, memory):
        """BEQ branches when Z=1."""
        load_program(memory, bytes([0xF0, 0x10]))  # BEQ +16
        cpu_8bit.flag_z = True
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.PC == 0x8012

    def test_beq_not_taken(self, cpu_8bit, memory):
        """BEQ doesn't branch when Z=0."""
        load_program(memory, bytes([0xF0, 0x10]))  # BEQ +16
        cpu_8bit.flag_z = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.PC == 0x8002

    def test_bne_taken(self, cpu_8bit, memory):
        """BNE branches when Z=0."""
        load_program(memory, bytes([0xD0, 0x10]))  # BNE +16
        cpu_8bit.flag_z = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.PC == 0x8012

    def test_bcs_taken(self, cpu_8bit, memory):
        """BCS branches when C=1."""
        load_program(memory, bytes([0xB0, 0x10]))  # BCS +16
        cpu_8bit.flag_c = True
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.PC == 0x8012

    def test_bcc_taken(self, cpu_8bit, memory):
        """BCC branches when C=0."""
        load_program(memory, bytes([0x90, 0x10]))  # BCC +16
        cpu_8bit.flag_c = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.PC == 0x8012


class TestJumpInstructions:
    """Test jump and call instructions."""

    def test_jmp_absolute(self, cpu_8bit, memory):
        """JMP $nnnn."""
        load_program(memory, bytes([0x4C, 0x00, 0x90]))  # JMP $9000
        cpu_8bit.PC = 0x8000

        cycles = cpu_8bit.step()

        assert cpu_8bit.PC == 0x9000
        assert cycles == 3

    def test_jsr_absolute(self, cpu_8bit, memory):
        """JSR $nnnn."""
        load_program(memory, bytes([0x20, 0x00, 0x90]))  # JSR $9000
        cpu_8bit.PC = 0x8000
        cpu_8bit.SP = 0x1FFF

        cycles = cpu_8bit.step()

        assert cpu_8bit.PC == 0x9000
        assert cycles == 6
        # Return address - 1 on stack
        assert memory.read(0x1FFF) == 0x80  # High byte
        assert memory.read(0x1FFE) == 0x02  # Low byte ($8003 - 1)

    def test_rts(self, cpu_8bit, memory):
        """RTS."""
        load_program(memory, bytes([0x60]))  # RTS
        # Push return address - 1
        cpu_8bit.SP = 0x1FFD
        memory.write(0x1FFE, 0xFF)  # Low byte ($8FFF)
        memory.write(0x1FFF, 0x8F)  # High byte
        cpu_8bit.PC = 0x8000

        cycles = cpu_8bit.step()

        assert cpu_8bit.PC == 0x9000  # $8FFF + 1
        assert cycles == 6


class TestStackInstructions:
    """Test stack instructions."""

    def test_pha(self, cpu_8bit, memory):
        """PHA."""
        load_program(memory, bytes([0x48]))  # PHA
        cpu_8bit.A = 0x42
        cpu_8bit.SP = 0x1FFF
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert memory.read(0x1FFF) == 0x42
        assert cpu_8bit.SP == 0x1FFE

    def test_pla(self, cpu_8bit, memory):
        """PLA."""
        load_program(memory, bytes([0x68]))  # PLA
        cpu_8bit.SP = 0x1FFE
        memory.write(0x1FFF, 0x42)
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert (cpu_8bit.A & 0xFF) == 0x42
        assert cpu_8bit.SP == 0x1FFF

    def test_php(self, cpu_8bit, memory):
        """PHP."""
        load_program(memory, bytes([0x08]))  # PHP
        cpu_8bit.P = 0x42
        cpu_8bit.SP = 0x1FFF
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert memory.read(0x1FFF) == 0x42

    def test_plp(self, cpu_8bit, memory):
        """PLP."""
        load_program(memory, bytes([0x28]))  # PLP
        cpu_8bit.SP = 0x1FFE
        memory.write(0x1FFF, 0x42)
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        # In native mode, P is loaded directly
        assert (cpu_8bit.P & 0x42) == 0x42


class TestFlagInstructions:
    """Test flag manipulation instructions."""

    def test_sec(self, cpu_8bit, memory):
        """SEC."""
        load_program(memory, bytes([0x38]))  # SEC
        cpu_8bit.flag_c = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_c is True

    def test_clc(self, cpu_8bit, memory):
        """CLC."""
        load_program(memory, bytes([0x18]))  # CLC
        cpu_8bit.flag_c = True
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_c is False

    def test_sei(self, cpu_8bit, memory):
        """SEI."""
        load_program(memory, bytes([0x78]))  # SEI
        cpu_8bit.flag_i = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_i is True

    def test_cli(self, cpu_8bit, memory):
        """CLI."""
        load_program(memory, bytes([0x58]))  # CLI
        cpu_8bit.flag_i = True
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_i is False

    def test_sed(self, cpu_8bit, memory):
        """SED."""
        load_program(memory, bytes([0xF8]))  # SED
        cpu_8bit.flag_d = False
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_d is True

    def test_cld(self, cpu_8bit, memory):
        """CLD."""
        load_program(memory, bytes([0xD8]))  # CLD
        cpu_8bit.flag_d = True
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_d is False

    def test_clv(self, cpu_8bit, memory):
        """CLV."""
        load_program(memory, bytes([0xB8]))  # CLV
        cpu_8bit.flag_v = True
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_v is False


class TestModeInstructions:
    """Test processor mode instructions."""

    def test_sep(self, cpu, memory):
        """SEP #$nn."""
        load_program(memory, bytes([0xE2, 0x30]))  # SEP #$30 (set M and X)
        cpu.PC = 0x8000
        cpu.X = 0x1234
        cpu.Y = 0x5678

        cpu.step()

        assert cpu.flag_m is True
        assert cpu.flag_x is True
        # X and Y should be truncated
        assert cpu.X == 0x34
        assert cpu.Y == 0x78

    def test_rep(self, cpu_8bit, memory):
        """REP #$nn."""
        load_program(memory, bytes([0xC2, 0x30]))  # REP #$30 (clear M and X)
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.flag_m is False
        assert cpu_8bit.flag_x is False

    def test_xce(self, cpu, memory):
        """XCE - exchange carry and emulation."""
        load_program(memory, bytes([0xFB]))  # XCE
        cpu.flag_c = True
        cpu.emulation_mode = False
        cpu.PC = 0x8000

        cpu.step()

        assert cpu.emulation_mode is True
        assert cpu.flag_c is False


class TestMiscInstructions:
    """Test miscellaneous instructions."""

    def test_nop(self, cpu_8bit, memory):
        """NOP."""
        load_program(memory, bytes([0xEA]))  # NOP
        cpu_8bit.PC = 0x8000

        cycles = cpu_8bit.step()

        assert cpu_8bit.PC == 0x8001
        assert cycles == 2

    def test_xba(self, cpu_8bit, memory):
        """XBA - exchange B and A."""
        load_program(memory, bytes([0xEB]))  # XBA
        cpu_8bit.A = 0x1234
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.A == 0x3412

    def test_stp(self, cpu_8bit, memory):
        """STP - stop processor."""
        load_program(memory, bytes([0xDB]))  # STP
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.stopped is True

    def test_wai(self, cpu_8bit, memory):
        """WAI - wait for interrupt."""
        load_program(memory, bytes([0xCB]))  # WAI
        cpu_8bit.PC = 0x8000

        cpu_8bit.step()

        assert cpu_8bit.waiting is True


class TestIntegration:
    """Integration tests with multi-instruction programs."""

    def test_simple_loop(self, cpu_8bit, memory):
        """Test a simple counting loop."""
        # LDX #$05; loop: DEX; BNE loop; STP
        program = bytes([
            0xA2, 0x05,  # LDX #$05
            0xCA,        # DEX
            0xD0, 0xFD,  # BNE -3 (back to DEX)
            0xDB,        # STP
        ])
        load_program(memory, program)
        cpu_8bit.PC = 0x8000

        # Run until stopped
        try:
            for _ in range(100):
                cpu_8bit.step()
        except StopExecution:
            pass

        assert cpu_8bit.X == 0x00
        assert cpu_8bit.stopped is True

    def test_memory_copy(self, cpu_8bit, memory):
        """Test loading and storing."""
        # LDA #$42; STA $10; LDA #$00; LDA $10; STP
        program = bytes([
            0xA9, 0x42,  # LDA #$42
            0x85, 0x10,  # STA $10
            0xA9, 0x00,  # LDA #$00
            0xA5, 0x10,  # LDA $10
            0xDB,        # STP
        ])
        load_program(memory, program)
        cpu_8bit.PC = 0x8000
        cpu_8bit.D = 0

        try:
            for _ in range(10):
                cpu_8bit.step()
        except StopExecution:
            pass

        assert (cpu_8bit.A & 0xFF) == 0x42
        assert memory.read(0x10) == 0x42

    def test_subroutine_call(self, cpu_8bit, memory):
        """Test JSR/RTS."""
        # JSR $8010; STP; (at $8010) LDA #$42; RTS
        program = bytes([
            0x20, 0x10, 0x80,  # JSR $8010
            0xDB,              # STP
        ])
        subroutine = bytes([
            0xA9, 0x42,  # LDA #$42
            0x60,        # RTS
        ])
        load_program(memory, program)
        load_program(memory, subroutine, start=0x8010)
        cpu_8bit.PC = 0x8000
        cpu_8bit.SP = 0x1FFF

        try:
            for _ in range(10):
                cpu_8bit.step()
        except StopExecution:
            pass

        assert (cpu_8bit.A & 0xFF) == 0x42
        assert cpu_8bit.stopped is True
