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
            Label('fn'), ModeChange('ACCU', 8),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x12)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x12)),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes, entry_modes={'fn': 8}, call_exit_modes={'fn': 8})
        assert not [n for n in result if isinstance(n, Instruction)]
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

    def test_load_kept_when_a_is_read_after(self):
        """TAY reads the loaded value: only the identity STA goes."""
        nodes = [
            Instruction(opcode=Opcode.TAX),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.TAY),
        ]
        opt = PeepholeOptimizer()
        result = opt.optimize(nodes)
        assert [n.opcode for n in result] == [Opcode.TAX, Opcode.LDA_DP, Opcode.TAY]


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
        """A label something branches to, between the two switches: the
        branch's path can observe the SEP."""
        nodes = [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Label(name="target"),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            # Two references, so the loop-switch hoist (single-reference
            # labels) leaves the REP where it is.
            Instruction(opcode=Opcode.BEQ, operand=Address("target")),
            Instruction(opcode=Opcode.BRA, operand=Address("target")),
        ]
        opt = PeepholeOptimizer()
        opt.optimize(nodes)
        assert opt.stats.dead_mode_changes_eliminated == 0

    def test_unreferenced_label_between_switches(self):
        """A label nothing names is only fallen into, so the SEP before it
        is overwritten unseen."""
        nodes = [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Label(name="fallen_into"),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
        ]
        opt = PeepholeOptimizer()
        opt.optimize(nodes)
        assert opt.stats.dead_mode_changes_eliminated == 1

    def test_redundant_switch_keeps_colocated_branch_target(self):
        """Dropping a SEP every predecessor already arrives in keeps all the
        labels before it, not only the first: the second is branched to."""
        nodes = [
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.BEQ, operand=Address("target")),
            Instruction(opcode=Opcode.BRA, operand=Address("first")),
            Label(name="first"),
            Label(name="target"),
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x12)),
            Instruction(opcode=Opcode.RTS),
        ]
        opt = PeepholeOptimizer()
        opt.optimize([])                  # sets up the dataflow seeds
        out = opt._eliminate_cross_block_mode_changes(nodes)
        assert opt.stats.redundant_mode_changes_eliminated == 1
        assert "target" in {n.name for n in out if isinstance(n, Label)}

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

    def test_folded_carry_survives_redundant_mode_switch(self):
        """Already in m16, `REP #$21` is redundant as a mode switch but its
        carry clear is what `SBC #(n-1)` relies on — an interrupt prologue
        (REP #$30 ... SEP #$20; frame setup) lost it and allocated its
        frame one byte short whenever the interrupted code left carry set."""
        nodes = [
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x30)),
            Instruction(opcode=Opcode.PHA),
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.TSC),
            Instruction(opcode=Opcode.SEC),
            Instruction(opcode=Opcode.SBC_IMMEDIATE, operand=Immediate(0x02)),
            Instruction(opcode=Opcode.TCS),
        ]
        result = PeepholeOptimizer().optimize(nodes)
        # Follow carry to the SBC: it must take exactly 2 off S.
        carry = None
        for n in result:
            if not isinstance(n, Instruction):
                continue
            if n.opcode == Opcode.SBC_IMMEDIATE:
                assert carry is not None, result
                assert n.operand.value + (1 - carry) == 2, result
                break
            if n.opcode in (Opcode.CLC, Opcode.SEC):
                carry = int(n.opcode == Opcode.SEC)
            elif (n.opcode in (Opcode.REP_IMMEDIATE, Opcode.SEP_IMMEDIATE)
                    and n.operand.value & 0x01):
                carry = int(n.opcode == Opcode.SEP_IMMEDIATE)


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


class TestLoopInvariantLoadHoistGuards:
    """_hoist_loop_invariant_loads moves an `LDA #imm` above a loop when
    nothing else in the loop writes A. A call does - its return value comes
    back in A - and a loop that switches the accumulator's width cannot
    have its immediate moved to where the width differs. Both held in
    `while step(0xFFFF) == 0 {}`: the 16-bit constant was hoisted above the
    REP into 8-bit mode (WLA-DX: "Couldn't parse LDA"), and every pass but
    the first passed the call's last return value instead of $FFFF."""

    SRC = '''
        #[zeropage(0x16, register)]
        static mut SCRATCH0: u16;
        #[zeropage(0x18, register)]
        static mut SCRATCH1: u16;
        #[lowram] static mut left: u16;
        far fn step(budget: u16) -> u8 {
            if left > budget { left = left - budget; return 0; }
            left = 0;
            return 1;
        }
        far fn finish() { while step(0xFFFF) == 0 {} }
        #[entry] fn main() -> ! { left = 5; finish(); loop {} }
    '''

    def test_constant_argument_stays_in_a_loop_with_a_call(self):
        from r65.compiler.main import compile_string
        asm = compile_string(self.SRC, filename='<test>')
        body = asm[asm.index('finish:'):]
        body = body[:body.index(' step')]          # the call: JSR or JSL
        label = body.index('finish__L1:')
        load = body.index('LDA #$FFFF')
        assert load > label, "the call's constant argument was hoisted out of the loop:\n" + body

    def test_call_in_the_body_blocks_the_hoist(self):
        loop = [
            Label('L'),
            Instruction(opcode=Opcode.LDA_IMMEDIATE, operand=Immediate(5)),
            Instruction(opcode=Opcode.STA_DP, operand=Address(0x16)),
            Instruction(opcode=Opcode.JSL, operand=Address('step')),
            Instruction(opcode=Opcode.CMP_IMMEDIATE, operand=Immediate(0)),
            Instruction(opcode=Opcode.BEQ, operand=Address('L')),
        ]
        opt = PeepholeOptimizer()
        opt._hoist_loop_invariant_loads(loop)
        assert opt.stats.loop_invariant_loads_hoisted == 0


class TestRedundantCmpZeroNeedsAFlags:
    """`CMP #0` tests A, so it is redundant only after an instruction whose
    N/Z come from A at A's width."""

    @staticmethod
    def _run(prev):
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            prev,
            Instruction(opcode=Opcode.CMP_IMMEDIATE, operand=Immediate(0)),
            Instruction(opcode=Opcode.BEQ, operand=Address(0x20)),
        ]
        return PeepholeOptimizer()._eliminate_redundant_cmp_zero(nodes)

    @pytest.mark.parametrize("prev", [
        Instruction(opcode=Opcode.LDX_DP, operand=Address(0x12)),
        Instruction(opcode=Opcode.INY),
        Instruction(opcode=Opcode.TAX),
        Instruction(opcode=Opcode.XBA),
        Instruction(opcode=Opcode.BIT_DP, operand=Address(0x12)),
    ], ids=lambda i: i.opcode.name)
    def test_kept_after_flags_from_elsewhere(self, prev):
        assert any(n.opcode == Opcode.CMP_IMMEDIATE for n in self._run(prev))

    def test_removed_after_an_a_result(self):
        prev = Instruction(opcode=Opcode.AND_ABSOLUTE_X, operand=Address(0x1234))
        assert not any(n.opcode == Opcode.CMP_IMMEDIATE for n in self._run(prev))


class TestTrackedLoadForgetsWrittenMemory:
    """A remembered `LDA addr` is stale once anything else writes `addr`."""

    @pytest.mark.parametrize("write", [
        Instruction(opcode=Opcode.INC_DP, operand=Address(0x10)),
        Instruction(opcode=Opcode.STZ_DP, operand=Address(0x10)),
        Instruction(opcode=Opcode.TSB_DP, operand=Address(0x10)),
        Instruction(opcode=Opcode.STZ_DP_X, operand=Address(0x0F)),
    ], ids=lambda i: i.opcode.name)
    def test_reload_kept(self, write):
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            write,
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
        ]
        result = PeepholeOptimizer()._eliminate_redundant_loads_tracked(nodes)
        assert len(result) == 3

    def test_write_elsewhere_keeps_tracking(self):
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.STZ_DP, operand=Address(0x12)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
        ]
        result = PeepholeOptimizer()._eliminate_redundant_loads_tracked(nodes)
        assert len(result) == 2


class TestStzKeepsZeroThatIsRead:
    """`LDA #0; STA x` becomes `STZ x`, and the LDA may go only if A is
    replaced before it is read. Most writers of A read it first."""

    @pytest.mark.parametrize("reader", [
        Instruction(opcode=Opcode.XBA),
        Instruction(opcode=Opcode.AND_IMMEDIATE, operand=Immediate(0xFF)),
        Instruction(opcode=Opcode.ASL),
        Instruction(opcode=Opcode.ADC_IMMEDIATE, operand=Immediate(1)),
    ], ids=lambda i: i.opcode.name)
    def test_lda_kept(self, reader):
        nodes = [
            Instruction(opcode=Opcode.LDA_IMMEDIATE, operand=Immediate(0)),
            Instruction(opcode=Opcode.STA_ABSOLUTE, operand=Address(0x0300)),
            reader,
        ]
        result = PeepholeOptimizer()._convert_zero_stores_to_stz(nodes)
        assert [n.opcode for n in result][:2] == [Opcode.LDA_IMMEDIATE, Opcode.STZ_ABSOLUTE]

    def test_lda_dropped_before_reload(self):
        nodes = [
            Instruction(opcode=Opcode.LDA_IMMEDIATE, operand=Immediate(0)),
            Instruction(opcode=Opcode.STA_ABSOLUTE, operand=Address(0x0300)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
        ]
        result = PeepholeOptimizer()._convert_zero_stores_to_stz(nodes)
        assert [n.opcode for n in result] == [Opcode.STZ_ABSOLUTE, Opcode.LDA_DP]


class TestIncFoldKeepsCarryForIndexedRotate:
    """`CLC; ADC #1` folds to `INC A` only when nothing reads its carry-out.
    The carry readers were listed by opcode, and `ROL dp,X` was missing."""

    @pytest.mark.parametrize("reader", [
        Instruction(opcode=Opcode.ROL_DP_X, operand=Address(0x10)),
        Instruction(opcode=Opcode.ROR_ABSOLUTE_X, operand=Address(0x1234)),
    ], ids=lambda i: i.opcode.name)
    def test_not_folded(self, reader):
        nodes = [
            Instruction(opcode=Opcode.CLC),
            Instruction(opcode=Opcode.ADC_IMMEDIATE, operand=Immediate(1)),
            reader,
            Instruction(opcode=Opcode.CLC),
        ]
        result = PeepholeOptimizer()._fold_inc_dec_accumulator(nodes)
        assert result[0].opcode == Opcode.CLC


class TestPerFunctionRegions:
    """optimize() iterates each region separately but must decide as a
    whole-stream run would."""

    def test_stack_store_before_bra_sees_other_regions(self):
        """A push in another function still keeps the stack store: the
        BRA rule asks whether *any* instruction may read the slot."""
        second = Comment("second function")
        nodes = [
            Label("first"),
            Instruction(opcode=Opcode.STA_STACK, operand=StackOffset(0x03)),
            Instruction(opcode=Opcode.BRA, operand=Address("first_end")),
            Label("first_end"),
            Instruction(opcode=Opcode.RTS),
            second,
            Label("other"),
            Instruction(opcode=Opcode.PHA),
            Instruction(opcode=Opcode.PLA),
            Instruction(opcode=Opcode.RTS),
        ]
        result = PeepholeOptimizer().optimize(nodes, region_starts=[second])
        assert any(isinstance(n, Instruction) and n.opcode == Opcode.STA_STACK
                   for n in result)


class TestDroppedWritesMustBeDead:
    """A pass may drop an instruction only if nothing reads what it wrote:
    an LDA's N/Z, or the zeroed B of `AND #$00FF`."""

    def test_reload_kept_for_a_branch(self):
        """After a call N/Z are the callee's; the reload sets them for BNE."""
        nodes = [
            Instruction(opcode=Opcode.JSR, operand=Address("f")),
            Instruction(opcode=Opcode.STA_STACK, operand=StackOffset(0x01)),
            Instruction(opcode=Opcode.LDA_STACK, operand=StackOffset(0x01)),
            Instruction(opcode=Opcode.BNE, operand=Address("f")),
        ]
        result = PeepholeOptimizer()._eliminate_redundant_load_after_store(nodes)
        assert len(result) == 4

    def test_tracked_reload_kept_for_a_branch(self):
        """A holds the value again, but the CMP between rewrote N/Z."""
        nodes = [
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.CMP_IMMEDIATE, operand=Immediate(5)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.BEQ, operand=Address(0x20)),
        ]
        result = PeepholeOptimizer()._eliminate_redundant_loads_tracked(nodes)
        assert len(result) == 4

    def test_mask_kept_before_tax(self):
        """In x16, TAX copies B into X's high byte, so the m16 mask stays."""
        nodes = [
            Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.LDA_DP, operand=Address(0x10)),
            Instruction(opcode=Opcode.AND_IMMEDIATE, operand=Immediate(0xFF)),
            Instruction(opcode=Opcode.SEP_IMMEDIATE, operand=Immediate(0x20)),
            Instruction(opcode=Opcode.TAX),
            Instruction(opcode=Opcode.LDA_ABSOLUTE_X, operand=Address(0x1000)),
        ]
        result = PeepholeOptimizer()._eliminate_redundant_and_before_sep(nodes)
        assert any(n.opcode == Opcode.AND_IMMEDIATE for n in result)


def test_x_only_rep_does_not_declare_the_accumulator_mode():
    """`REP #$10` widens X/Y only; the `.ACCU 8` after it is the signal."""
    from r65.compiler.optimize.asm_mode_dataflow import (
        compute_modes, first_constraining_mode_after,
    )
    nodes = [
        Instruction(opcode=Opcode.REP_IMMEDIATE, operand=Immediate(0x10)),
        ModeChange('ACCU', 8),
        Instruction(opcode=Opcode.RTS),
    ]
    assert first_constraining_mode_after(nodes, compute_modes(nodes), 0) == 8
