#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for peephole optimization passes."""

import pytest

from r65.compiler.optimize.peephole import PeepholeOptimizer
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Instruction, Address, StackOffset


class TestIdentityCopyElimination:
    """Test _eliminate_identity_copies pass."""

    def test_dp_identity_copy_eliminated(self):
        """LDA $10; STA $10 should be eliminated."""
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x10)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 0
        assert opt.stats.identity_copies_eliminated == 1

    def test_stack_identity_copy_eliminated(self):
        """LDA $03,S; STA $03,S should be eliminated."""
        nodes = [
            Instruction(opcode=Opcode.LDA_STACK, operand=StackOffset(0x03)),
            Instruction(opcode=Opcode.STA_STACK, operand=StackOffset(0x03)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 0
        assert opt.stats.identity_copies_eliminated == 1

    def test_different_addresses_not_eliminated(self):
        """LDA $10; STA $12 should NOT be eliminated."""
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x12)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 2
        assert opt.stats.identity_copies_eliminated == 0

    def test_different_addressing_modes_not_eliminated(self):
        """LDA $10 (DP); STA $10,S (stack) should NOT be eliminated."""
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STA_STACK, operand=StackOffset(0x10)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 2
        assert opt.stats.identity_copies_eliminated == 0

    def test_non_adjacent_not_eliminated(self):
        """LDA $10; TAX; STA $10 should NOT be eliminated (not adjacent)."""
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.TAX),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x10)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 3
        assert opt.stats.identity_copies_eliminated == 0

    def test_multiple_identity_copies(self):
        """Multiple identity copies should all be eliminated."""
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x12)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x12)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 0
        assert opt.stats.identity_copies_eliminated == 2

    def test_absolute_identity_copy_eliminated(self):
        """LDA $1234; STA $1234 should be eliminated."""
        nodes = [
            Instruction(opcode=Opcode.LDA_ABSOLUTE, operand=Address(0x1234)),
            Instruction(opcode=Opcode.STA_ABSOLUTE, operand=Address(0x1234)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 0
        assert opt.stats.identity_copies_eliminated == 1

    def test_surrounding_instructions_preserved(self):
        """Instructions around an identity copy should be preserved."""
        nodes = [
            Instruction(opcode=Opcode.TAX),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.TAY),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert len(result) == 2
        assert result[0].opcode == Opcode.TAX
        assert result[1].opcode == Opcode.TAY
