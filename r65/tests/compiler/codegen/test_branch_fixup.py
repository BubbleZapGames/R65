"""Tests for branch fixup pass using typed Instruction nodes."""

import pytest
from r65.compiler.codegen.branch_fixup import (
    fixup_nodes,
    BranchFixup,
    CONDITIONAL_BRANCH_OPCODES,
    MAX_BRANCH_DISTANCE,
)
from r65.compiler.codegen.asm_nodes import (
    Instruction, Label, Comment, Directive, BlankLine,
    Address, invert_branch, BRANCH_INVERSIONS,
)
from r65.compiler.codegen.opcodes import Opcode


class TestBranchFixup:
    """Tests for branch fixup logic using Instruction nodes."""

    def test_short_branch_unchanged(self):
        """Branches within range should not be modified."""
        nodes = [
            Label("func"),
            Instruction(Opcode.LDA_IMMEDIATE, Address(0x00)),
            Instruction(Opcode.BEQ, Address("short_target")),
            Instruction(Opcode.NOP),
            Label("short_target"),
            Instruction(Opcode.RTS),
        ]

        fixed, num_fixups = fixup_nodes(nodes)

        assert num_fixups == 0
        # Original BEQ should still be there
        beq_instrs = [n for n in fixed if isinstance(n, Instruction) and n.opcode == Opcode.BEQ]
        assert len(beq_instrs) == 1

    def test_long_branch_fixed(self):
        """Branches exceeding 127 bytes should be fixed."""
        nodes = [
            Label("func"),
            Instruction(Opcode.BEQ, Address("far_target")),
        ]
        # Add 150 NOPs (150 bytes) to exceed branch range
        for _ in range(150):
            nodes.append(Instruction(Opcode.NOP))
        nodes.append(Label("far_target"))
        nodes.append(Instruction(Opcode.RTS))

        fixed, num_fixups = fixup_nodes(nodes)

        assert num_fixups == 1
        # Check for inversion pattern: BNE (inverted) + JMP (to original target)
        bne_instrs = [n for n in fixed if isinstance(n, Instruction) and n.opcode == Opcode.BNE]
        jmp_instrs = [n for n in fixed if isinstance(n, Instruction) and n.opcode == Opcode.JMP_ABSOLUTE]
        assert len(bne_instrs) == 1
        assert len(jmp_instrs) == 1
        # JMP should target far_target
        assert jmp_instrs[0].operand.value == "far_target"

    def test_branch_inversion_table(self):
        """Verify all conditional branches have inversions."""
        for opcode in CONDITIONAL_BRANCH_OPCODES:
            inverse = invert_branch(opcode)
            assert inverse is not None, f"{opcode} has no inverse"
            # Double inversion should return original
            assert invert_branch(inverse) == opcode

    def test_branch_inversions_dict(self):
        """Verify BRANCH_INVERSIONS dict is complete and symmetric."""
        for opcode in CONDITIONAL_BRANCH_OPCODES:
            assert opcode in BRANCH_INVERSIONS
            inverse = BRANCH_INVERSIONS[opcode]
            assert BRANCH_INVERSIONS[inverse] == opcode

    def test_multiple_long_branches(self):
        """Multiple long branches in same function."""
        nodes = [
            Label("func"),
            Instruction(Opcode.BEQ, Address("target1")),
            Instruction(Opcode.BNE, Address("target2")),
        ]
        # Add padding
        for _ in range(150):
            nodes.append(Instruction(Opcode.NOP))
        nodes.extend([
            Label("target1"),
            Instruction(Opcode.NOP),
            Label("target2"),
            Instruction(Opcode.RTS),
        ])

        fixed, num_fixups = fixup_nodes(nodes)

        # Both branches should be fixed
        assert num_fixups == 2

    def test_backward_branch(self):
        """Backward branches can also exceed range."""
        nodes = [
            Label("func"),
            Label("loop_start"),
        ]
        # Add 150 NOPs
        for _ in range(150):
            nodes.append(Instruction(Opcode.NOP))
        nodes.extend([
            Instruction(Opcode.BNE, Address("loop_start")),  # Backward branch
            Instruction(Opcode.RTS),
        ])

        fixed, num_fixups = fixup_nodes(nodes)

        # Backward branch should be fixed
        assert num_fixups == 1


class TestIntegration:
    """Integration tests with realistic assembly patterns."""

    def test_real_world_pattern(self):
        """Test pattern similar to actual generated code."""
        nodes = [
            Comment("Function with while loop"),
            Label("test_while"),
            Directive(".ACCU", ["8"]),
            Directive(".INDEX", ["8"]),
            Instruction(Opcode.LDA_IMMEDIATE, Address(0x0A)),
            Instruction(Opcode.STA_DP, Address(0x10)),
            Label("test_while__L1"),
            Instruction(Opcode.LDA_DP, Address(0x10)),
            Instruction(Opcode.BEQ, Address("test_while__L3"), "Exit if zero"),
            Instruction(Opcode.JMP_ABSOLUTE, Address("test_while__L2")),
            Label("test_while__L2"),
        ]
        # Large loop body - JSR is 3 bytes each
        for i in range(50):
            nodes.append(Instruction(Opcode.JSR, Address(f"nop{i % 4}")))
        nodes.extend([
            Instruction(Opcode.JMP_ABSOLUTE, Address("test_while__L1")),
            Label("test_while__L3"),
            Instruction(Opcode.RTS),
        ])

        fixed, num_fixups = fixup_nodes(nodes)

        # The BEQ test_while__L3 should be fixed (50 JSRs = 150 bytes)
        assert num_fixups == 1

        # Verify output still has proper structure
        labels = [n.name for n in fixed if isinstance(n, Label)]
        assert "test_while" in labels
        assert "test_while__L3" in labels


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        fixed, num_fixups = fixup_nodes([])
        assert fixed == []
        assert num_fixups == 0

    def test_no_branches(self):
        """Code without branches should be unchanged."""
        nodes = [
            Label("func"),
            Instruction(Opcode.LDA_IMMEDIATE, Address(0x42)),
            Instruction(Opcode.STA_DP, Address(0x20)),
            Instruction(Opcode.RTS),
        ]

        fixed, num_fixups = fixup_nodes(nodes)
        assert num_fixups == 0
        assert len(fixed) == len(nodes)

    def test_jmp_not_affected(self):
        """JMP instructions should never be 'fixed'."""
        nodes = [
            Label("func"),
            Instruction(Opcode.JMP_ABSOLUTE, Address("far_target")),
        ]
        for _ in range(150):
            nodes.append(Instruction(Opcode.NOP))
        nodes.extend([
            Label("far_target"),
            Instruction(Opcode.RTS),
        ])

        fixed, num_fixups = fixup_nodes(nodes)

        # JMP is not a conditional branch, should not be fixed
        assert num_fixups == 0

    def test_bra_not_conditional(self):
        """BRA (unconditional relative) is not in the fixup set."""
        nodes = [
            Label("func"),
            Instruction(Opcode.BRA, Address("target")),
            Label("target"),
            Instruction(Opcode.RTS),
        ]

        fixed, num_fixups = fixup_nodes(nodes)

        # BRA is not in CONDITIONAL_BRANCH_OPCODES
        assert num_fixups == 0

    def test_comments_preserved(self):
        """Comments should be preserved through fixup."""
        nodes = [
            Comment("Header comment"),
            Label("func"),
            Instruction(Opcode.BEQ, Address("far_target"), "Branch comment"),
        ]
        for _ in range(150):
            nodes.append(Instruction(Opcode.NOP))
        nodes.extend([
            Label("far_target"),
            Instruction(Opcode.RTS),
        ])

        fixed, num_fixups = fixup_nodes(nodes)

        assert num_fixups == 1
        comments = [n for n in fixed if isinstance(n, Comment)]
        assert len(comments) >= 1

    def test_directives_preserved(self):
        """Directives should be preserved through fixup."""
        nodes = [
            Label("func"),
            Directive(".ACCU", ["16"]),
            Instruction(Opcode.BEQ, Address("far_target")),
        ]
        for _ in range(150):
            nodes.append(Instruction(Opcode.NOP))
        nodes.extend([
            Label("far_target"),
            Instruction(Opcode.RTS),
        ])

        fixed, num_fixups = fixup_nodes(nodes)

        assert num_fixups == 1
        directives = [n for n in fixed if isinstance(n, Directive)]
        assert len(directives) == 1
        assert directives[0].name == ".ACCU"


class TestBranchInversion:
    """Test branch inversion functionality."""

    def test_all_conditional_branches_invertible(self):
        """All conditional branches should be invertible."""
        conditional = [
            Opcode.BCC, Opcode.BCS,
            Opcode.BEQ, Opcode.BNE,
            Opcode.BMI, Opcode.BPL,
            Opcode.BVC, Opcode.BVS,
        ]
        for opcode in conditional:
            inverse = invert_branch(opcode)
            assert inverse is not None
            assert inverse != opcode

    def test_unconditional_branches_not_invertible(self):
        """Unconditional branches (BRA, BRL) should not be invertible."""
        assert invert_branch(Opcode.BRA) is None
        assert invert_branch(Opcode.BRL) is None

    def test_non_branches_not_invertible(self):
        """Non-branch instructions should not be invertible."""
        assert invert_branch(Opcode.LDA_IMMEDIATE) is None
        assert invert_branch(Opcode.JMP_ABSOLUTE) is None
        assert invert_branch(Opcode.NOP) is None

    def test_specific_inversions(self):
        """Test specific branch inversions."""
        assert invert_branch(Opcode.BEQ) == Opcode.BNE
        assert invert_branch(Opcode.BNE) == Opcode.BEQ
        assert invert_branch(Opcode.BCS) == Opcode.BCC
        assert invert_branch(Opcode.BCC) == Opcode.BCS
        assert invert_branch(Opcode.BMI) == Opcode.BPL
        assert invert_branch(Opcode.BPL) == Opcode.BMI
        assert invert_branch(Opcode.BVS) == Opcode.BVC
        assert invert_branch(Opcode.BVC) == Opcode.BVS
