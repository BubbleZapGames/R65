"""Tests for cross-bank function call validation."""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.typeck.errors import TypeCheckError


def typecheck(source: str):
    """Parse, build HIR, and type check source code."""
    program = parse(source)
    program = expand_macros(program)
    builder = HIRBuilder()
    hir_program = builder.build_program(program)
    checker = TypeChecker(hir_program)
    checker.check()
    return hir_program


class TestCrossBankCalls:
    """Tests for cross-bank function call validation."""

    def test_same_bank_near_call_allowed(self):
        """Near functions in the same bank can call each other."""
        # Both functions default to bank 0
        source = """
        #[mode(m8, x8)]
        fn helper() {
            A = 1;
        }

        #[mode(m8, x8)]
        fn main() {
            helper();
        }
        """
        # Should not raise
        typecheck(source)

    def test_same_bank_explicit_near_call_allowed(self):
        """Near functions in explicitly same bank can call each other."""
        source = """
        #[bank(1)]
        #[mode(m8, x8)]
        fn helper() {
            A = 1;
        }

        #[mode(m8, x8)]
        fn caller() {
            helper();
        }
        """
        # Should not raise - both in bank 1 due to directive
        typecheck(source)

    def test_cross_bank_near_call_error(self):
        """Near function in different bank cannot be called by near function."""
        source = """
        #[bank(0)]
        #[mode(m8, x8)]
        fn bank0_func() {
            bank1_func();
        }

        #[bank(1)]
        #[mode(m8, x8)]
        fn bank1_func() {
            A = 1;
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            typecheck(source)

        error = exc_info.value
        assert "cannot call near function 'bank1_func'" in str(error)
        assert "bank 0" in str(error)
        assert "bank 1" in str(error)
        # The hint suggests using far fn
        assert error.hint is not None
        assert "far fn" in error.hint

    def test_cross_bank_far_call_allowed(self):
        """Far functions can be called from any bank."""
        source = """
        #[bank(0)]
        #[mode(m8, x8)]
        fn bank0_func() {
            bank1_func();
        }

        #[bank(1)]
        #[mode(m8, x8)]
        far fn bank1_func() {
            A = 1;
        }
        """
        # Should not raise - bank1_func is far so JSL is used
        typecheck(source)

    def test_far_caller_can_call_near_in_same_bank(self):
        """Far function can call near function in same bank."""
        source = """
        #[bank(1)]
        #[mode(m8, x8)]
        fn helper() {
            A = 1;
        }

        #[mode(m8, x8)]
        far fn main() {
            helper();
        }
        """
        # Both in bank 1, should not raise
        typecheck(source)

    def test_far_caller_cannot_call_near_in_different_bank(self):
        """Far function cannot call near function in different bank."""
        source = """
        #[bank(0)]
        #[mode(m8, x8)]
        fn bank0_helper() {
            A = 1;
        }

        #[bank(1)]
        #[mode(m8, x8)]
        far fn bank1_main() {
            bank0_helper();
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            typecheck(source)

        error = str(exc_info.value)
        assert "cannot call near function 'bank0_helper'" in error

    def test_default_bank_zero(self):
        """Functions without explicit bank default to bank 0."""
        source = """
        #[mode(m8, x8)]
        fn helper() {
            A = 1;
        }

        #[bank(1)]
        #[mode(m8, x8)]
        fn bank1_caller() {
            helper();
        }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            typecheck(source)

        error = str(exc_info.value)
        assert "cannot call near function 'helper'" in error
        assert "bank 1" in error
        assert "bank 0" in error

    def test_multiple_bank_switches(self):
        """Test multiple bank directive switches."""
        source = """
        #[bank(0)]
        #[mode(m8, x8)]
        fn bank0_a() { A = 1; }
        #[mode(m8, x8)]
        fn bank0_b() { bank0_a(); }  // OK - same bank

        #[bank(1)]
        #[mode(m8, x8)]
        fn bank1_a() { A = 2; }
        #[mode(m8, x8)]
        fn bank1_b() { bank1_a(); }  // OK - same bank

        #[bank(0)]
        #[mode(m8, x8)]
        fn bank0_c() { bank0_b(); }  // OK - both in bank 0
        """
        # Should not raise
        typecheck(source)

    def test_cross_bank_with_mode_attributes(self):
        """Bank check works together with mode attributes."""
        source = """
        #[bank(0)]
        #[mode(m8)]
        fn bank0_func() {
            bank1_func();
        }

        #[bank(1)]
        #[mode(m8, transition=inline)]
        far fn bank1_func() {
            A = 1;
        }
        """
        # Should not raise - bank1_func is far
        typecheck(source)
