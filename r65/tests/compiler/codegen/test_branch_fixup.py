"""Tests for branch fixup pass."""

import pytest
from r65.compiler.codegen.branch_fixup import (
    fixup_long_branches,
    AssemblyParser,
    BranchFixup,
    AsmInstruction,
    AsmLabel,
    CONDITIONAL_BRANCHES,
    BRANCH_INVERSION,
    MAX_BRANCH_DISTANCE,
)


class TestAssemblyParser:
    """Tests for assembly line parsing."""

    def test_parse_instruction(self):
        """Parse a simple instruction."""
        parser = AssemblyParser()
        elements = parser.parse_lines(["    LDA #$42"])

        assert len(elements) == 1
        assert isinstance(elements[0], AsmInstruction)
        assert elements[0].mnemonic == "LDA"
        assert elements[0].operand == "#$42"

    def test_parse_instruction_with_comment(self):
        """Parse instruction with inline comment."""
        parser = AssemblyParser()
        elements = parser.parse_lines(["    LDA #$42  ; Load value"])

        assert len(elements) == 1
        instr = elements[0]
        assert instr.mnemonic == "LDA"
        assert instr.operand == "#$42"
        assert instr.comment == "Load value"

    def test_parse_label(self):
        """Parse a label."""
        parser = AssemblyParser()
        elements = parser.parse_lines(["my_label:"])

        assert len(elements) == 1
        assert isinstance(elements[0], AsmLabel)
        assert elements[0].name == "my_label"

    def test_parse_branch_instruction(self):
        """Parse branch instruction."""
        parser = AssemblyParser()
        elements = parser.parse_lines(["    BEQ target_label"])

        assert len(elements) == 1
        instr = elements[0]
        assert instr.mnemonic == "BEQ"
        assert instr.operand == "target_label"
        assert instr.size == 2  # All branches are 2 bytes

    def test_instruction_sizes(self):
        """Test instruction size calculation."""
        parser = AssemblyParser()

        test_cases = [
            ("    NOP", 1),           # Implied
            ("    RTS", 1),           # Return
            ("    BEQ label", 2),     # Branch
            ("    LDA #$42", 2),      # Immediate (8-bit mode)
            ("    JMP label", 3),     # Absolute jump
            ("    JSR func", 3),      # Subroutine call
            ("    JSL far_func", 4),  # Long subroutine call
        ]

        for line, expected_size in test_cases:
            elements = parser.parse_lines([line])
            assert elements[0].size == expected_size, f"Failed for: {line}"


class TestBranchFixup:
    """Tests for branch fixup logic."""

    def test_short_branch_unchanged(self):
        """Branches within range should not be modified."""
        lines = [
            "func:",
            "    LDA #$00",
            "    BEQ short_target",
            "    NOP",
            "short_target:",
            "    RTS",
        ]

        fixed, num_fixups = fixup_long_branches(lines)

        assert num_fixups == 0
        # Check BEQ is still there unchanged
        assert any("BEQ short_target" in line for line in fixed)

    def test_long_branch_fixed(self):
        """Branches exceeding 127 bytes should be fixed."""
        # Create a long branch scenario
        lines = ["func:"]
        lines.append("    BEQ far_target")  # This branch needs to skip many NOPs

        # Add 150 NOPs (150 bytes) to exceed branch range
        for _ in range(150):
            lines.append("    NOP")

        lines.append("far_target:")
        lines.append("    RTS")

        fixed, num_fixups = fixup_long_branches(lines)

        assert num_fixups == 1
        # Check the inversion pattern
        assert any("BNE __branch_skip_" in line for line in fixed)
        assert any("JMP far_target" in line for line in fixed)

    def test_branch_inversion_table(self):
        """Verify all conditional branches have inversions."""
        for branch in CONDITIONAL_BRANCHES:
            assert branch in BRANCH_INVERSION
            inverse = BRANCH_INVERSION[branch]
            # Double inversion should return original
            assert BRANCH_INVERSION[inverse] == branch

    def test_multiple_long_branches(self):
        """Multiple long branches in same function."""
        lines = ["func:"]
        lines.append("    BEQ target1")
        lines.append("    BNE target2")

        # Add padding
        for _ in range(150):
            lines.append("    NOP")

        lines.append("target1:")
        lines.append("    NOP")
        lines.append("target2:")
        lines.append("    RTS")

        fixed, num_fixups = fixup_long_branches(lines)

        # Both branches should be fixed
        assert num_fixups == 2

    def test_backward_branch(self):
        """Backward branches can also exceed range."""
        lines = ["func:"]
        lines.append("loop_start:")

        # Add 150 NOPs
        for _ in range(150):
            lines.append("    NOP")

        lines.append("    BNE loop_start")  # Backward branch
        lines.append("    RTS")

        fixed, num_fixups = fixup_long_branches(lines)

        # Backward branch should be fixed
        assert num_fixups == 1


class TestIntegration:
    """Integration tests with full assembly."""

    def test_real_world_pattern(self):
        """Test pattern similar to actual generated code."""
        lines = [
            "; Function with while loop",
            "test_while:",
            ".ACCU 8",
            ".INDEX 8",
            "    LDA #$0A",
            "    STA $10",
            "test_while__L1:",
            "    LDA $10",
            "    BEQ test_while__L3  ; Exit if zero",
            "    JMP test_while__L2",
            "test_while__L2:",
        ]

        # Large loop body
        for i in range(50):
            lines.append(f"    JSR nop{i % 4}")

        lines.extend([
            "    JMP test_while__L1",
            "test_while__L3:",
            "    RTS",
        ])

        fixed, num_fixups = fixup_long_branches(lines)

        # The BEQ test_while__L3 should be fixed
        assert num_fixups == 1

        # Verify output still has proper structure
        assert any("test_while:" in line for line in fixed)
        assert any("test_while__L3:" in line for line in fixed)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        fixed, num_fixups = fixup_long_branches([])
        assert fixed == []
        assert num_fixups == 0

    def test_no_branches(self):
        """Code without branches should be unchanged."""
        lines = [
            "func:",
            "    LDA #$42",
            "    STA $20",
            "    RTS",
        ]

        fixed, num_fixups = fixup_long_branches(lines)
        assert num_fixups == 0
        assert len(fixed) == len(lines)

    def test_jmp_not_affected(self):
        """JMP instructions should never be 'fixed'."""
        lines = ["func:"]
        lines.append("    JMP far_target")

        for _ in range(150):
            lines.append("    NOP")

        lines.append("far_target:")
        lines.append("    RTS")

        fixed, num_fixups = fixup_long_branches(lines)

        # JMP is not a conditional branch, should not be fixed
        assert num_fixups == 0

    def test_bra_not_conditional(self):
        """BRA (unconditional relative) is not in the fixup set."""
        # BRA has 8-bit range but is handled differently
        # For now we don't fix BRA - it would need BRL replacement
        lines = ["func:"]
        lines.append("    BRA target")
        lines.append("target:")
        lines.append("    RTS")

        fixed, num_fixups = fixup_long_branches(lines)

        # BRA is not in CONDITIONAL_BRANCHES
        assert num_fixups == 0
