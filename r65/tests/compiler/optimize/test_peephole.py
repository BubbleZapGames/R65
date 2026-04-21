#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for peephole optimization passes."""

import pytest

from r65.compiler.optimize.peephole import PeepholeOptimizer
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import (
    Instruction, Address, StackOffset, Label, RawAsm,
)


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


class TestBranchThreadingRawAsm:
    """Test that _thread_branches treats RawAsm as executable code.

    Bug: _thread_branches skipped RawAsm nodes when scanning for the first
    instruction after a label. If a block had inline asm followed by BRA,
    the pass saw the BRA as the "first instruction" and incorrectly threaded
    branches targeting that label through to the BRA's destination.
    This made the else clause in long if-else-if chains unreachable when
    branches used inline asm (asm! macro).
    """

    def test_rawasm_blocks_branch_threading(self):
        """BNE to label with RawAsm before BRA should NOT be threaded."""
        nodes = [
            # BNE targets else_label
            Instruction(opcode=Opcode.BNE, operand=Address("else_label")),
            # ... then block ...
            Label("then_label"),
            Instruction(opcode=Opcode.NOP),
            Instruction(opcode=Opcode.BRA, operand=Address("merge_label")),
            # merge block (just a BRA to epilogue)
            Label("merge_label"),
            Instruction(opcode=Opcode.BRA, operand=Address("epilogue")),
            # else block: RawAsm (inline asm) THEN BRA
            Label("else_label"),
            RawAsm("LDA #$42"),
            RawAsm("STA $4302"),
            Instruction(opcode=Opcode.BRA, operand=Address("merge_label")),
            # epilogue
            Label("epilogue"),
            Instruction(opcode=Opcode.RTS),
        ]
        opt = PeepholeOptimizer()
        result = opt._thread_branches(nodes)
        # BNE should still target else_label, NOT epilogue
        assert result[0].opcode == Opcode.BNE
        assert result[0].operand.value == "else_label"

    def test_empty_label_before_bra_is_threaded(self):
        """BNE to label with only BRA (no RawAsm) should be threaded."""
        nodes = [
            Instruction(opcode=Opcode.BNE, operand=Address("merge_label")),
            Label("merge_label"),
            Instruction(opcode=Opcode.BRA, operand=Address("epilogue")),
            Label("epilogue"),
            Instruction(opcode=Opcode.RTS),
        ]
        opt = PeepholeOptimizer()
        result = opt._thread_branches(nodes)
        # BNE should be threaded to epilogue
        assert result[0].opcode == Opcode.BNE
        assert result[0].operand.value == "epilogue"


class TestBranchOverBranchEliminationSelfLoopGuard:
    """Regression: `_eliminate_branch_over_branch` must not drop a BRA that
    self-loops to a label in its `between` region. That pattern is the body
    of a `-> !` / `#[entry]` halt loop — removing the BRA lets control fall
    through past the label into whatever follows.

    Bug reproducer: `loop { if X == 5 { break; } X++; }` in an entry fn
    compiled to `BNE body / halt: BRA halt / body: INX / BRA loop`. The pass
    rewrote BNE → BEQ halt and dropped `BRA halt`, letting X run unbounded.
    """

    def test_self_looping_halt_bra_preserved(self):
        """BNE skipping a `halt: BRA halt` block must not eliminate the BRA."""
        nodes = [
            Instruction(opcode=Opcode.BNE, operand=Address("body")),
            Label("halt_label"),
            Instruction(opcode=Opcode.BRA, operand=Address("halt_label")),
            Label("body"),
            Instruction(opcode=Opcode.INX),
        ]
        opt = PeepholeOptimizer()
        result = opt._eliminate_branch_over_branch(nodes)
        # The halt self-loop must survive intact.
        bra_nodes = [n for n in result
                     if isinstance(n, Instruction) and n.opcode == Opcode.BRA]
        assert len(bra_nodes) == 1
        assert bra_nodes[0].operand.value == "halt_label"
        # And the original BNE is untouched (no inversion-retarget happened).
        bne_nodes = [n for n in result
                     if isinstance(n, Instruction) and n.opcode == Opcode.BNE]
        assert len(bne_nodes) == 1
        assert bne_nodes[0].operand.value == "body"

    def test_ordinary_branch_over_branch_still_eliminated(self):
        """Sanity: pattern with a non-self-loop BRA is still optimized."""
        nodes = [
            Instruction(opcode=Opcode.BNE, operand=Address("skip")),
            Instruction(opcode=Opcode.BRA, operand=Address("elsewhere")),
            Label("skip"),
            Instruction(opcode=Opcode.INX),
            Label("elsewhere"),
            Instruction(opcode=Opcode.RTS),
        ]
        opt = PeepholeOptimizer()
        result = opt._eliminate_branch_over_branch(nodes)
        # BNE → BEQ elsewhere (inverted + retargeted), BRA dropped.
        first = [n for n in result if isinstance(n, Instruction)][0]
        assert first.opcode == Opcode.BEQ
        assert first.operand.value == "elsewhere"
