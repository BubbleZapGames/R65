"""
Tests for register-specific operator restrictions.

Based on 65816 hardware capabilities:
- A: Full ALU support (+, -, &, |, ^, <<, >>)
- X/Y: Only increment (++), decrement (--), comparison, load, transfer
- B: No binary operations (accessed via XBA swap)
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.typeck.errors import TypeCheckError


def compile_and_type_check(source: str):
    """Helper to compile and type check source code."""
    program = parse(source)
    program = expand_macros(program)
    builder = HIRBuilder()
    hir = builder.build_program(program)
    type_checker = TypeChecker(hir)
    type_checker.check()
    return hir


class TestXRegisterRestrictions:
    """Tests for X register operation restrictions."""

    def test_x_increment_allowed(self):
        """X++ should be allowed (uses INX instruction)."""
        source = """
        fn test() {
            X++;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_decrement_allowed(self):
        """X-- should be allowed (uses DEX instruction)."""
        source = """
        fn test() {
            X--;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_load_immediate_allowed(self):
        """X = constant should be allowed (uses LDX instruction)."""
        source = """
        fn test() {
            X = 10;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_comparison_allowed(self):
        """X < constant comparison should be allowed (uses CPX instruction)."""
        source = """
        fn test() {
            if X < 100 {
                A = 1;
            }
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_transfer_from_a_allowed(self):
        """X = A should be allowed (uses TAX instruction)."""
        source = """
        fn test() {
            X = A as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_add_rejected(self):
        """X = X + 5 should be rejected (X cannot add)."""
        source = """
        fn test() {
            X = X + 5;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '+' not allowed on register X" in str(exc_info.value)

    def test_x_subtract_rejected(self):
        """X = X - 5 should be rejected (X cannot subtract)."""
        source = """
        fn test() {
            X = X - 5;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '-' not allowed on register X" in str(exc_info.value)

    def test_x_bitwise_and_rejected(self):
        """X = X & 0xFF should be rejected (X cannot do bitwise ops)."""
        source = """
        fn test() {
            X = X & 0xFF;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '&' not allowed on register X" in str(exc_info.value)

    def test_x_bitwise_or_rejected(self):
        """X = X | 0x80 should be rejected."""
        source = """
        fn test() {
            X = X | 0x80;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '|' not allowed on register X" in str(exc_info.value)

    def test_x_bitwise_xor_rejected(self):
        """X = X ^ 0xFF should be rejected."""
        source = """
        fn test() {
            X = X ^ 0xFF;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '^' not allowed on register X" in str(exc_info.value)

    def test_x_shift_left_rejected(self):
        """X = X << 1 should be rejected."""
        source = """
        fn test() {
            X = X << 1;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '<<' not allowed on register X" in str(exc_info.value)

    def test_x_shift_right_rejected(self):
        """X = X >> 1 should be rejected."""
        source = """
        fn test() {
            X = X >> 1;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '>>' not allowed on register X" in str(exc_info.value)

    def test_x_compound_add_rejected(self):
        """X += 5 should be rejected (desugars to X = X + 5)."""
        source = """
        fn test() {
            X += 5;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '+' not allowed on register X" in str(exc_info.value)

    def test_x_multiply_rejected(self):
        """X = X * 2 should be rejected."""
        source = """
        fn test() {
            X = X * 2;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '*' not allowed on register X" in str(exc_info.value)


class TestYRegisterRestrictions:
    """Tests for Y register operation restrictions (same as X)."""

    def test_y_increment_allowed(self):
        """Y++ should be allowed (uses INY instruction)."""
        source = """
        fn test() {
            Y++;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_y_decrement_allowed(self):
        """Y-- should be allowed (uses DEY instruction)."""
        source = """
        fn test() {
            Y--;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_y_add_rejected(self):
        """Y = Y + 5 should be rejected."""
        source = """
        fn test() {
            Y = Y + 5;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '+' not allowed on register Y" in str(exc_info.value)

    def test_y_compound_subtract_rejected(self):
        """Y -= 3 should be rejected (desugars to Y = Y - 3)."""
        source = """
        fn test() {
            Y -= 3;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '-' not allowed on register Y" in str(exc_info.value)


class TestARegisterOperations:
    """Tests that A register allows all operations."""

    def test_a_add_allowed(self):
        """A = A + 5 should be allowed (uses ADC instruction)."""
        source = """
        fn test() {
            A = A + 5;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_subtract_allowed(self):
        """A = A - 3 should be allowed (uses SBC instruction)."""
        source = """
        fn test() {
            A = A - 3;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_bitwise_and_allowed(self):
        """A = A & 0x0F should be allowed (uses AND instruction)."""
        source = """
        fn test() {
            A = A & 0x0F;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_bitwise_or_allowed(self):
        """A = A | 0x80 should be allowed (uses ORA instruction)."""
        source = """
        fn test() {
            A = A | 0x80;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_bitwise_xor_allowed(self):
        """A = A ^ 0xFF should be allowed (uses EOR instruction)."""
        source = """
        fn test() {
            A = A ^ 0xFF;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_shift_left_allowed(self):
        """A = A << 1 should be allowed (uses ASL instruction)."""
        source = """
        fn test() {
            A = A << 1;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_shift_right_allowed(self):
        """A = A >> 1 should be allowed (uses LSR instruction)."""
        source = """
        fn test() {
            A = A >> 1;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_increment_allowed(self):
        """A++ should be allowed (uses INC A instruction)."""
        source = """
        fn test() {
            A++;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_decrement_allowed(self):
        """A-- should be allowed (uses DEC A instruction)."""
        source = """
        fn test() {
            A--;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_compound_add_allowed(self):
        """A += 5 should be allowed."""
        source = """
        fn test() {
            A += 5;
        }
        """
        compile_and_type_check(source)  # Should not raise


class TestRegisterAliasTracking:
    """Tests that register aliases (let x @ X = ...) are properly tracked."""

    def test_aliased_x_increment_allowed(self):
        """let x @ X = ...; x++ should be allowed."""
        source = """
        fn test() {
            let x @ X = 10;
            x++;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_aliased_x_add_rejected(self):
        """let x @ X = ...; x = x + 5 should be rejected."""
        source = """
        fn test() {
            let x @ X = 10;
            x = x + 5;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '+' not allowed on register X" in str(exc_info.value)

    def test_aliased_y_subtract_rejected(self):
        """let idx @ Y = ...; idx = idx - 3 should be rejected."""
        source = """
        fn test() {
            let idx @ Y = 0;
            idx = idx - 3;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '-' not allowed on register Y" in str(exc_info.value)

    def test_aliased_a_operations_allowed(self):
        """let acc @ A = ...; acc = acc + 5 should be allowed."""
        source = """
        fn test() {
            let acc @ A = 10;
            acc = acc + 5;
            acc = acc & 0x0F;
            acc = acc << 2;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_parameter_alias_x_rejected(self):
        """Parameter bound to X should have restrictions."""
        source = """
        fn test(idx @ X: u16) {
            idx = idx + 5;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '+' not allowed on register X" in str(exc_info.value)

    def test_parameter_alias_a_allowed(self):
        """Parameter bound to A should allow all operations."""
        source = """
        fn test(value @ A: u8) {
            value = value + 5;
            value = value & 0x0F;
        }
        """
        compile_and_type_check(source)  # Should not raise


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_x_load_from_memory_allowed(self):
        """X = memory_value should be allowed (just a load)."""
        source = """
        #[zeropage]
        static mut TEMP: u16;
        fn test() {
            X = TEMP;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_non_self_operation_allowed(self):
        """X = A + 5 should be allowed (A is doing the op, not X)."""
        source = """
        fn test() {
            X = (A + 5) as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_nested_expression_check(self):
        """Complex nested expression should still check."""
        source = """
        fn test() {
            X = X + (5 + 3);
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "operator '+' not allowed on register X" in str(exc_info.value)

    def test_error_message_includes_hint(self):
        """Error message should include helpful hint."""
        source = """
        fn test() {
            X = X + 5;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        error = exc_info.value
        # Check the error message
        assert "operator '+' not allowed on register X" in error.message
        # Check the hint mentions increment/decrement and transfer
        assert error.hint is not None
        hint_lower = error.hint.lower()
        assert "increment" in hint_lower or "decrement" in hint_lower
        assert "move" in hint_lower or "transfer" in hint_lower or "a" in hint_lower


class TestIndexRegisterComparison:
    """Tests for index register comparison restrictions.

    X and Y cannot be compared directly because there's no CPX Y or CPY X
    instruction - it would require using an intermediate register.
    """

    def test_x_vs_y_rejected(self):
        """X == Y should be rejected (no CPX Y instruction)."""
        source = """
        fn test() {
            if X == Y {
                A = 1;
            }
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot compare X with Y directly" in str(exc_info.value)

    def test_x_less_than_y_rejected(self):
        """X < Y should be rejected."""
        source = """
        fn test() {
            if X < Y {
                A = 1;
            }
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot compare X with Y directly" in str(exc_info.value)

    def test_y_vs_x_rejected(self):
        """Y > X should be rejected."""
        source = """
        fn test() {
            if Y > X {
                A = 1;
            }
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot compare Y with X directly" in str(exc_info.value)

    def test_aliased_x_vs_y_rejected(self):
        """Aliased X vs Y should be rejected."""
        source = """
        fn test() {
            let x @ X = 100;
            let y @ Y = 200;
            if x == y {
                A = 1;
            }
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot compare X with Y directly" in str(exc_info.value)

    def test_x_vs_immediate_allowed(self):
        """X == constant should be allowed (uses CPX instruction)."""
        source = """
        fn test() {
            if X == 100 {
                A = 1;
            }
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_y_vs_immediate_allowed(self):
        """Y < constant should be allowed (uses CPY instruction)."""
        source = """
        fn test() {
            if Y < 200 {
                A = 1;
            }
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_vs_a_allowed(self):
        """X == A should be allowed (can transfer A to temp and CPX)."""
        source = """
        fn test() {
            if X == (A as u16) {
                A = 1;
            }
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_vs_memory_allowed(self):
        """X == memory should be allowed (uses CPX addr)."""
        source = """
        #[zeropage]
        static mut TEMP: u16;
        fn test() {
            if X == TEMP {
                A = 1;
            }
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_error_has_helpful_hint(self):
        """Error message should suggest storing to variable first."""
        source = """
        fn test() {
            if X == Y {
                A = 1;
            }
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        error = exc_info.value
        assert error.hint is not None
        assert "store" in error.hint.lower() or "variable" in error.hint.lower()


class TestRegisterTransferRestrictions:
    """Tests for register-to-register transfer restrictions.

    Some register pairs don't have direct transfer instructions and would
    require an intermediate register, which we reject.
    """

    def test_d_to_x_rejected(self):
        """D = X should be rejected (no TDX instruction)."""
        source = """
        fn test() {
            X = D;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot transfer D to X directly" in str(exc_info.value)

    def test_d_to_y_rejected(self):
        """Y = D should be rejected (no TDY instruction)."""
        source = """
        fn test() {
            Y = D;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot transfer D to Y directly" in str(exc_info.value)

    def test_x_to_d_rejected(self):
        """D = X should be rejected (no TXD instruction)."""
        source = """
        fn test() {
            D = X;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot transfer X to D directly" in str(exc_info.value)

    def test_y_to_d_rejected(self):
        """D = Y should be rejected (no TYD instruction)."""
        source = """
        fn test() {
            D = Y;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        assert "cannot transfer Y to D directly" in str(exc_info.value)

    def test_x_to_y_allowed(self):
        """Y = X should be allowed (TXY instruction exists)."""
        source = """
        fn test() {
            Y = X;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_y_to_x_allowed(self):
        """X = Y should be allowed (TYX instruction exists)."""
        source = """
        fn test() {
            X = Y;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_to_x_allowed(self):
        """X = A should be allowed (TAX instruction exists)."""
        source = """
        fn test() {
            X = A as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_to_y_allowed(self):
        """Y = A should be allowed (TAY instruction exists)."""
        source = """
        fn test() {
            Y = A as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_to_a_allowed(self):
        """A = X should be allowed (TXA instruction exists)."""
        source = """
        fn test() {
            A = X as u8;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_y_to_a_allowed(self):
        """A = Y should be allowed (TYA instruction exists)."""
        source = """
        fn test() {
            A = Y as u8;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_error_suggests_intermediate(self):
        """Error should suggest using A as intermediate."""
        source = """
        fn test() {
            X = D;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            compile_and_type_check(source)
        error = exc_info.value
        assert error.hint is not None
        assert "A" in error.hint or "through" in error.hint.lower()

    def test_status_to_a_allowed(self):
        """A = STATUS should be allowed (PHP + PLA)."""
        source = """
        fn test() {
            A = STATUS;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_to_status_allowed(self):
        """STATUS = A should be allowed (PHA + PLP)."""
        source = """
        fn test() {
            STATUS = A;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_status_to_x_allowed(self):
        """X = STATUS should be allowed (goes through A)."""
        source = """
        fn test() {
            X = STATUS as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_status_to_y_allowed(self):
        """Y = STATUS should be allowed (goes through A)."""
        source = """
        fn test() {
            Y = STATUS as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise


class TestDBRTransfers:
    """Tests for DBR register transfers via stack operations."""

    def test_dbr_to_a_allowed(self):
        """A = DBR should be allowed (PHB + PLA)."""
        source = """
        fn test() {
            A = DBR;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_a_to_dbr_allowed(self):
        """DBR = A should be allowed (PHA + PLB)."""
        source = """
        fn test() {
            DBR = A;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_dbr_to_x_allowed(self):
        """X = DBR should be allowed (through A)."""
        source = """
        fn test() {
            X = DBR as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_dbr_to_y_allowed(self):
        """Y = DBR should be allowed (through A)."""
        source = """
        fn test() {
            Y = DBR as u16;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_x_to_dbr_allowed(self):
        """DBR = X should be allowed (through A)."""
        source = """
        fn test() {
            DBR = X as u8;
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_y_to_dbr_allowed(self):
        """DBR = Y should be allowed (through A)."""
        source = """
        fn test() {
            DBR = Y as u8;
        }
        """
        compile_and_type_check(source)  # Should not raise
