"""Tests for bank size validation."""

import pytest
from r65.compiler.codegen.bank_size import (
    calculate_bank_sizes,
    validate_bank_sizes,
    BankSizeError,
    LOROM_BANK_SIZE,
    HIROM_BANK_SIZE,
    SNES_HEADER_SIZE,
)
from r65.compiler.codegen.asm_nodes import (
    Instruction, Directive, Label, Comment, BlankLine,
)
from r65.compiler.codegen.opcodes import Opcode


class TestCalculateBankSizes:
    """Tests for bank size calculation."""

    def test_empty_program(self):
        """Empty program should have zero size for bank 0."""
        nodes = []
        result = calculate_bank_sizes(nodes)

        assert 0 in result
        assert result[0].size == 0

    def test_single_instruction(self):
        """Single instruction should contribute its size."""
        nodes = [
            Instruction(Opcode.NOP),  # 1 byte
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].size == 1

    def test_multiple_instructions(self):
        """Multiple instructions should sum their sizes."""
        nodes = [
            Instruction(Opcode.NOP),  # 1 byte
            Instruction(Opcode.NOP),  # 1 byte
            Instruction(Opcode.RTS),  # 1 byte
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].size == 3

    def test_bank_switch(self):
        """Bank directive should switch to new bank."""
        nodes = [
            Instruction(Opcode.NOP),  # Bank 0: 1 byte
            Directive(".BANK", ["1", "SLOT", "0"]),
            Instruction(Opcode.NOP),  # Bank 1: 1 byte
            Instruction(Opcode.NOP),  # Bank 1: 1 byte
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].size == 1
        assert result[1].size == 2

    def test_data_directive_db(self):
        """.db directive should count bytes."""
        nodes = [
            Directive(".db", ["$01", "$02", "$03"]),  # 3 bytes
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].size == 3

    def test_data_directive_dw(self):
        """.dw directive should count words (2 bytes each)."""
        nodes = [
            Directive(".dw", ["$1234", "$5678"]),  # 4 bytes
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].size == 4

    def test_labels_dont_count(self):
        """Labels should not contribute to size."""
        nodes = [
            Label("test_label"),
            Instruction(Opcode.NOP),  # 1 byte
            Label("another_label"),
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].size == 1

    def test_comments_dont_count(self):
        """Comments should not contribute to size."""
        nodes = [
            Comment("This is a comment"),
            Instruction(Opcode.NOP),  # 1 byte
            Comment("Another comment", section_header=True),
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].size == 1

    def test_lorom_limit(self):
        """LoROM bank limit should be 32KB."""
        nodes = []
        result = calculate_bank_sizes(nodes, is_hirom=False, has_header=False)

        assert result[0].limit == LOROM_BANK_SIZE

    def test_hirom_limit(self):
        """HiROM bank limit should be 64KB."""
        nodes = []
        result = calculate_bank_sizes(nodes, is_hirom=True, has_header=False)

        assert result[0].limit == HIROM_BANK_SIZE

    def test_header_bank_reduced_capacity(self):
        """Header bank should have reduced capacity."""
        nodes = []
        result = calculate_bank_sizes(nodes, has_header=True)

        assert result[0].is_header_bank
        assert result[0].limit == LOROM_BANK_SIZE - SNES_HEADER_SIZE

    def test_non_header_bank_full_capacity(self):
        """Non-header banks should have full capacity."""
        nodes = [
            Directive(".BANK", ["1", "SLOT", "0"]),
            Instruction(Opcode.NOP),
        ]
        result = calculate_bank_sizes(nodes, has_header=True)

        assert not result[1].is_header_bank
        assert result[1].limit == LOROM_BANK_SIZE

    def test_no_header_full_capacity(self):
        """Bank 0 without header should have full capacity."""
        nodes = []
        result = calculate_bank_sizes(nodes, has_header=False)

        assert not result[0].is_header_bank
        assert result[0].limit == LOROM_BANK_SIZE


class TestValidateBankSizes:
    """Tests for bank size validation."""

    def test_valid_small_program(self):
        """Small program should pass validation."""
        nodes = [
            Instruction(Opcode.NOP),
            Instruction(Opcode.RTS),
        ]
        # Should not raise
        validate_bank_sizes(nodes)

    def test_overflow_raises_error(self):
        """Bank overflow should raise BankSizeError."""
        # Create enough instructions to overflow bank 0
        # LoROM header bank limit is 32KB - 64 bytes = 32704 bytes
        # NOP is 1 byte, so 33000 NOPs will overflow
        nodes = [Instruction(Opcode.NOP) for _ in range(33000)]

        with pytest.raises(BankSizeError) as exc_info:
            validate_bank_sizes(nodes, has_header=True)

        assert "bank 0 exceeds" in str(exc_info.value)

    def test_error_message_includes_hint(self):
        """Error message should include helpful hint."""
        # 33000 NOPs overflow the header bank
        nodes = [Instruction(Opcode.NOP) for _ in range(33000)]

        with pytest.raises(BankSizeError) as exc_info:
            validate_bank_sizes(nodes, has_header=True)

        assert exc_info.value.hint is not None
        assert "bank" in exc_info.value.hint.lower()

    def test_multiple_banks_validated(self):
        """All banks should be validated."""
        # Bank 0 - small, Bank 1 - overflow
        nodes = [
            Instruction(Opcode.NOP),  # Bank 0 - 1 byte
            Directive(".BANK", ["1", "SLOT", "0"]),
        ] + [Instruction(Opcode.NOP) for _ in range(33000)]  # Bank 1 - overflow

        with pytest.raises(BankSizeError) as exc_info:
            validate_bank_sizes(nodes, has_header=True)

        assert "bank 1 exceeds" in str(exc_info.value)

    def test_hirom_larger_limit(self):
        """HiROM should allow larger banks."""
        # 40000 NOPs (40KB) - too big for LoROM (32KB) but OK for HiROM (64KB)
        nodes = [Instruction(Opcode.NOP) for _ in range(40000)]

        # Should fail for LoROM
        with pytest.raises(BankSizeError):
            validate_bank_sizes(nodes, is_hirom=False, has_header=False)

        # Should pass for HiROM (64KB limit)
        validate_bank_sizes(nodes, is_hirom=True, has_header=False)


class TestBankInfoProperties:
    """Tests for BankInfo computed properties."""

    def test_overflow_when_over_limit(self):
        """Overflow should report bytes over limit."""
        # 33024 bytes = 32KB + 256 bytes over
        nodes = [Instruction(Opcode.NOP) for _ in range(33024)]
        result = calculate_bank_sizes(nodes, has_header=False)

        # Should be 256 bytes over (33024 - 32768 = 256)
        assert result[0].overflow == 256

    def test_no_overflow_when_under_limit(self):
        """Overflow should be 0 when under limit."""
        nodes = [
            Instruction(Opcode.NOP),
        ]
        result = calculate_bank_sizes(nodes)

        assert result[0].overflow == 0

    def test_usage_percent(self):
        """Usage percent should calculate correctly."""
        # 16384 NOPs = 16KB = 50% of 32KB
        nodes = [Instruction(Opcode.NOP) for _ in range(16384)]
        result = calculate_bank_sizes(nodes, has_header=False)

        assert 49.9 < result[0].usage_percent < 50.1
