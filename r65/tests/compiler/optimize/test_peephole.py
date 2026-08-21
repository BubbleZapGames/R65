#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for peephole optimization passes."""

import pytest

from r65.compiler.optimize.peephole import PeepholeOptimizer
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import (
    Instruction, Address, StackOffset, Label, RawAsm, Immediate,
    ModeChange, Comment,
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


class TestDeadModeChangeElimination:
    """Test _eliminate_dead_mode_changes pass."""

    def test_sep_killed_by_following_rep(self):
        """SEP #$20; REP #$20 — the SEP is overwritten before anything reads P."""
        nodes = [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert [n.opcode for n in result if isinstance(n, Instruction)] == [
            Opcode.REP_IMMEDIATE, Opcode.LDA_DP,
        ]
        assert opt.stats.dead_mode_changes_eliminated == 1

    def test_partial_mask_not_eliminated(self):
        """SEP #$30; REP #$20 leaves the x flag set — the SEP still matters."""
        nodes = [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x30)),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert opt.stats.dead_mode_changes_eliminated == 0

    def test_label_between_blocks_elimination(self):
        """A label between the two switches means another path can observe the SEP."""
        nodes = [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Label(name="target"),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
        ]
        opt = PeepholeOptimizer()
        opt.optimize(nodes)
        assert opt.stats.dead_mode_changes_eliminated == 0

    def test_instruction_between_blocks_elimination(self):
        """An instruction between the switches runs in the mode the SEP set."""
        nodes = [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
        ]
        opt = PeepholeOptimizer()
        opt.optimize(nodes)
        assert opt.stats.dead_mode_changes_eliminated == 0


class TestCarrySetupFoldedIntoRep:
    """Test _fold_carry_setup_into_rep pass."""

    def test_clc_folded_into_rep(self):
        """REP #$20; TSC; CLC; ADC — REP #$21 clears carry for free."""
        nodes = [
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.TSC),
            Instruction(opcode=Opcode.CLC),
            Instruction(opcode=Opcode.ADC_IMMEDIATE, operand=Immediate(0x0A)),
            Instruction(opcode=Opcode.TCS),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        instrs = [n for n in result if isinstance(n, Instruction)]
        assert [n.opcode for n in instrs] == [
            Opcode.REP_IMMEDIATE, Opcode.TSC, Opcode.ADC_IMMEDIATE, Opcode.TCS,
        ]
        assert instrs[0].operand.value == 0x21
        assert instrs[2].operand.value == 0x0A
        assert opt.stats.carry_ops_folded_into_rep == 1

    def test_sec_folded_into_rep_with_borrow(self):
        """REP #$20; TSC; SEC; SBC #$0A — carry-clear SBC needs one less."""
        nodes = [
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.TSC),
            Instruction(opcode=Opcode.SEC),
            Instruction(opcode=Opcode.SBC_IMMEDIATE, operand=Immediate(0x0A)),
            Instruction(opcode=Opcode.TCS),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        instrs = [n for n in result if isinstance(n, Instruction)]
        assert [n.opcode for n in instrs] == [
            Opcode.REP_IMMEDIATE, Opcode.TSC, Opcode.SBC_IMMEDIATE, Opcode.TCS,
        ]
        assert instrs[0].operand.value == 0x21
        assert instrs[2].operand.value == 0x09
        assert opt.stats.carry_ops_folded_into_rep == 1

    def test_sec_not_folded_when_operand_is_zero(self):
        """SBC #$00 has no borrow-adjusted form — leave the SEC alone."""
        nodes = [
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.TSC),
            Instruction(opcode=Opcode.SEC),
            Instruction(opcode=Opcode.SBC_IMMEDIATE, operand=Immediate(0x00)),
        ]
        opt = PeepholeOptimizer()
        opt.optimize(nodes)
        assert opt.stats.carry_ops_folded_into_rep == 0

    def test_intervening_instruction_blocks_fold(self):
        """Only a TSC may sit between the REP and the carry op."""
        nodes = [
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.CLC),
            Instruction(opcode=Opcode.ADC_IMMEDIATE, operand=Immediate(0x0A)),
        ]
        opt = PeepholeOptimizer()
        opt.optimize(nodes)
        assert opt.stats.carry_ops_folded_into_rep == 0


class TestLoopModeSwitchHoistRetagsAnchor:
    """Test _hoist_loop_mode_switches keeps the label's `.ACCU` anchor honest."""

    @staticmethod
    def _loop_nodes(*, before_anchor=()):
        """A bottom-tested loop whose header carries a REP and an anchor."""
        return [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Label(name="loop"),
            *before_anchor,
            ModeChange(flag="ACCU", bits=8),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.INC),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.BNE, operand=Address("loop")),
        ]

    def _hoist(self, nodes):
        opt = PeepholeOptimizer()
        return opt._hoist_loop_mode_switches(nodes)

    def test_anchor_follows_the_hoisted_switch(self):
        """REP moves above the label, so the label is now entered in m16."""
        result = self._hoist(self._loop_nodes())
        label_at = next(i for i, n in enumerate(result)
                        if isinstance(n, Label) and n.name == "loop")
        # The hoisted REP now precedes the label...
        assert result[label_at - 1].opcode == Opcode.REP_IMMEDIATE
        # ...and the anchor says m16, not the stale m8.
        anchor = next(n for n in result[label_at:]
                      if isinstance(n, ModeChange) and n.flag == "ACCU")
        assert anchor.bits == 16

    def test_anchor_found_past_a_comment(self):
        """A comment between label and anchor must not hide it from the retag."""
        result = self._hoist(
            self._loop_nodes(before_anchor=(Comment(text="block entry"),))
        )
        anchor = next(n for n in result if isinstance(n, ModeChange)
                      and n.flag == "ACCU")
        assert anchor.bits == 16
