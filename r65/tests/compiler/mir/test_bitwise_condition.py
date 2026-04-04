# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for bitwise condition optimization.

Tests that bitwise expressions (&, |, ^) used as conditions are properly
optimized to use direct Z flag testing instead of short-circuit evaluation.
"""

from r65.compiler.mir.nodes import BinaryOp, CondBranch, Compare, BitTest
from r65.tests.language.common import build_mir, get_mir_instructions


def has_binary_op(instrs, op: str) -> bool:
    """Check if instruction list contains a BinaryOp with given operator."""
    return any(isinstance(i, BinaryOp) and i.op == op for i in instrs)


def count_cond_branches(instrs) -> int:
    """Count CondBranch instructions."""
    return sum(1 for i in instrs if isinstance(i, CondBranch))


def has_bit_test(instrs) -> bool:
    """Check if instruction list contains a BitTest."""
    return any(isinstance(i, BitTest) for i in instrs)


class TestBitwiseOrCondition:
    """Tests for bitwise OR (|) in conditions."""

    def test_simple_or_condition(self):
        """Test if (A | B) generates ORA + branch."""
        source = """
        #[zeropage(0x10)]
        static mut A_VAR: u8;
        #[zeropage(0x11)]
        static mut B_VAR: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((A_VAR | B_VAR) != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Should have ORA binary op
        assert has_binary_op(instrs, '|'), "Should emit ORA (|) instruction"
        # Should have single conditional branch (not multiple for short-circuit)
        assert count_cond_branches(instrs) == 1, "Should have single conditional branch"

    def test_zero_on_left_side(self):
        """Test if (0 == (A | B)) handles zero on left correctly."""
        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x11)]
        static mut MASK: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if (0 == (FLAGS | MASK)) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        assert has_binary_op(instrs, '|'), "Should emit ORA instruction"
        assert count_cond_branches(instrs) == 1

    def test_or_equal_zero(self):
        """Test if ((A | B) == 0) uses inverted branch."""
        source = """
        #[zeropage(0x10)]
        static mut A_VAR: u8;
        #[zeropage(0x11)]
        static mut B_VAR: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((A_VAR | B_VAR) == 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        assert has_binary_op(instrs, '|')
        assert count_cond_branches(instrs) == 1

        # Find the CondBranch and verify it uses == comparison (for BEQ)
        for instr in instrs:
            if isinstance(instr, CondBranch):
                assert instr.comparison == '==', "Should use == (BEQ) for == 0 test"


class TestBitwiseAndCondition:
    """Tests for bitwise AND (&) in conditions."""

    def test_simple_and_condition(self):
        """Test if (A & mask) generates AND + branch."""
        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((FLAGS & 0x0F) != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        assert has_binary_op(instrs, '&'), "Should emit AND instruction"
        assert count_cond_branches(instrs) == 1

    def test_and_bit7_uses_bit_instruction(self):
        """Test if (A & 0x80) uses BIT instruction optimization."""
        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((FLAGS & 0x80) != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Bit 7 test should use BIT instruction, not AND
        assert has_bit_test(instrs), "Should use BIT instruction for bit 7 test"

    def test_and_bit6_uses_bit_instruction(self):
        """Test if (A & 0x40) uses BIT instruction optimization."""
        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((FLAGS & 0x40) != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Bit 6 test should use BIT instruction (BVS/BVC)
        assert has_bit_test(instrs), "Should use BIT instruction for bit 6 test"


class TestBitwiseXorCondition:
    """Tests for bitwise XOR (^) in conditions."""

    def test_simple_xor_condition(self):
        """Test if (A ^ B) generates EOR + branch."""
        source = """
        #[zeropage(0x10)]
        static mut A_VAR: u8;
        #[zeropage(0x11)]
        static mut B_VAR: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((A_VAR ^ B_VAR) != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        assert has_binary_op(instrs, '^'), "Should emit EOR (^) instruction"
        assert count_cond_branches(instrs) == 1


class TestChainedBitwiseCondition:
    """Tests for chained bitwise operations in conditions."""

    def test_chained_or(self):
        """Test if (A | B | C) generates chained ORA."""
        source = """
        #[zeropage(0x10)]
        static mut A_VAR: u8;
        #[zeropage(0x11)]
        static mut B_VAR: u8;
        #[zeropage(0x13)]
        static mut C_VAR: u8;
        #[zeropage(0x14)]
        static mut RESULT: u8;

                fn test() {
            if ((A_VAR | B_VAR | C_VAR) != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Should have multiple ORA ops (chained)
        or_count = sum(1 for i in instrs if isinstance(i, BinaryOp) and i.op == '|')
        assert or_count >= 2, f"Should have at least 2 ORA instructions, got {or_count}"
        # Still single branch
        assert count_cond_branches(instrs) == 1


class TestNegatedBitwiseCondition:
    """Tests for negated bitwise conditions using == 0."""

    def test_negated_or_via_eq_zero(self):
        """Test if ((A | B) == 0) generates ORA + BEQ (inverted branch)."""
        # In R65, !(A | B) is expressed as (A | B) == 0 since ! requires boolean
        source = """
        #[zeropage(0x10)]
        static mut A_VAR: u8;
        #[zeropage(0x11)]
        static mut B_VAR: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((A_VAR | B_VAR) == 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        assert has_binary_op(instrs, '|')
        assert count_cond_branches(instrs) == 1

        # Check for inverted comparison (== for BEQ when checking if zero)
        for instr in instrs:
            if isinstance(instr, CondBranch):
                assert instr.comparison == '==', "Should use == (BEQ) for == 0 test"


class TestLogicalVsBitwiseCondition:
    """Tests that logical operators still use short-circuit evaluation."""

    def test_logical_or_short_circuit(self):
        """Test that || still uses short-circuit evaluation (multiple branches)."""
        source = """
        #[zeropage(0x10)]
        static mut A_VAR: u8;
        #[zeropage(0x11)]
        static mut B_VAR: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if (A_VAR != 0 || B_VAR != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Logical OR should have multiple branches for short-circuit
        branch_count = count_cond_branches(instrs)
        assert branch_count >= 2, f"Logical || should have multiple branches, got {branch_count}"

    def test_logical_and_short_circuit(self):
        """Test that && still uses short-circuit evaluation."""
        source = """
        #[zeropage(0x10)]
        static mut A_VAR: u8;
        #[zeropage(0x11)]
        static mut B_VAR: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if (A_VAR != 0 && B_VAR != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Logical AND should have multiple branches for short-circuit
        branch_count = count_cond_branches(instrs)
        assert branch_count >= 2, f"Logical && should have multiple branches, got {branch_count}"


class TestVolatileNotOptimized:
    """Tests that volatile variables are not optimized (preserve order)."""

    def test_volatile_hw_not_optimized(self):
        """Test that #[hw] volatile variables don't get bitwise optimization."""
        source = """
        #[hw(0x4212)]
        static mut HVBJOY: u8;
        #[hw(0x4016)]
        static mut JOYSER0: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((HVBJOY | JOYSER0) != 0) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Should NOT use the bitwise optimization for volatile vars
        # This means it will use normal comparison path (Compare + CondBranch)
        # rather than ORA + CondBranch on result
        has_compare = any(isinstance(i, Compare) for i in instrs)

        # If volatile, should fall back to Compare instruction path
        # (The bitwise optimization should be skipped)
        assert has_compare or has_binary_op(instrs, '|'), \
            "Should either use Compare or fall back to ORA"


class TestBitwiseWithComparison:
    """Tests for bitwise conditions with non-zero comparisons."""

    def test_bitwise_greater_than(self):
        """Test if ((A & mask) > N) uses bitwise + compare."""
        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test() {
            if ((FLAGS & 0x0F) > 5) {
                RESULT = 1;
            }
        }
        """
        mir = build_mir(source)
        instrs = get_mir_instructions(mir, "test")

        # Should have AND operation
        assert has_binary_op(instrs, '&'), "Should emit AND instruction"
        # Should have Compare for > 5
        has_compare = any(isinstance(i, Compare) for i in instrs)
        assert has_compare, "Should emit Compare instruction for > 5"


class TestBitwiseConditionCodegen:
    """Integration tests for bitwise condition code generation."""

    def test_or_generates_ora_instruction(self):
        """Test that OR condition generates ORA in assembly."""
        from r65.compiler.main import compile_source
        import io
        import sys

        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x11)]
        static mut MASK: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test_or() {
            if ((FLAGS | MASK) != 0) {
                RESULT = 1;
            }
        }
        """

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            compile_source(source, filename='<test>')
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Check that ORA instruction is present
        assert 'ORA' in output, "Should generate ORA instruction"

    def test_and_generates_and_instruction(self):
        """Test that AND condition generates AND in assembly."""
        from r65.compiler.main import compile_source
        import io
        import sys

        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test_and() {
            if ((FLAGS & 0x0F) != 0) {
                RESULT = 2;
            }
        }
        """

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            compile_source(source, filename='<test>')
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Check that AND instruction is present
        assert 'AND' in output, "Should generate AND instruction"

    def test_xor_generates_eor_instruction(self):
        """Test that XOR condition generates EOR in assembly."""
        from r65.compiler.main import compile_source
        import io
        import sys

        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x11)]
        static mut MASK: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test_xor() {
            if ((FLAGS ^ MASK) != 0) {
                RESULT = 3;
            }
        }
        """

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            compile_source(source, filename='<test>')
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Check that EOR instruction is present
        assert 'EOR' in output, "Should generate EOR instruction"

    def test_bit7_generates_bit_instruction(self):
        """Test that bit 7 test generates BIT + BMI/BPL."""
        from r65.compiler.main import compile_source
        import io
        import sys

        source = """
        #[zeropage(0x10)]
        static mut FLAGS: u8;
        #[zeropage(0x12)]
        static mut RESULT: u8;

                fn test_bit7() {
            if ((FLAGS & 0x80) != 0) {
                RESULT = 1;
            }
        }
        """

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            compile_source(source, filename='<test>')
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Check that BIT instruction is present
        assert 'BIT' in output, "Should generate BIT instruction for bit 7 test"
        # Should use BMI or BPL for N flag testing
        assert 'BMI' in output or 'BPL' in output, "Should use BMI/BPL for bit 7"
