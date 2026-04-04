# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for overflow-aware literal promotion.

When unsuffixed integer literals produce a compile-time result that overflows
the inferred type, the compiler should promote to u16/i16 automatically.
"""

from r65.tests.e2e import ExpectedState


class TestLiteralPromotion:
    """Test that unsuffixed literals are promoted when compile-time result overflows."""

    def test_shift_overflow_promotes(self, e2e):
        """8 << 5 = 256, overflows u8 -> promote to u16."""
        result = e2e.run('''
            #[zeropage]
            static mut VAL: u16;

            #[entry]
            fn main() {
                VAL = 8 << 5;
                X = VAL;
            }
        ''', ExpectedState(X=256))
        assert result.success, f"Failures: {result.failures}"

    def test_shift_no_overflow_stays_u8(self, e2e):
        """1 << 0 = 1, fits u8 -> stays u8."""
        result = e2e.run('''
            #[entry]
            fn main() {
                A = 1 << 0;
            }
        ''', ExpectedState(A=1))
        assert result.success, f"Failures: {result.failures}"

    def test_shift_max_u8_stays(self, e2e):
        """1 << 7 = 128, fits u8 -> stays u8."""
        result = e2e.run('''
            #[entry]
            fn main() {
                A = 1 << 7;
            }
        ''', ExpectedState(A=128))
        assert result.success, f"Failures: {result.failures}"

    def test_shift_just_overflows(self, e2e):
        """1 << 8 = 256, overflows u8 -> promote to u16."""
        result = e2e.run('''
            #[zeropage]
            static mut VAL: u16;

            #[entry]
            fn main() {
                VAL = 1 << 8;
                X = VAL;
            }
        ''', ExpectedState(X=256))
        assert result.success, f"Failures: {result.failures}"

    def test_multiply_overflow_promotes(self, e2e):
        """32 * 32 = 1024, overflows u8 -> promote to u16."""
        result = e2e.run('''
            #[zeropage]
            static mut VAL: u16;

            #[entry]
            fn main() {
                VAL = 32 * 32;
                X = VAL;
            }
        ''', ExpectedState(X=1024))
        assert result.success, f"Failures: {result.failures}"

    def test_add_overflow_promotes(self, e2e):
        """200 + 100 = 300, overflows u8 -> promote to u16."""
        result = e2e.run('''
            #[zeropage]
            static mut VAL: u16;

            #[entry]
            fn main() {
                VAL = 200 + 100;
                X = VAL;
            }
        ''', ExpectedState(X=300))
        assert result.success, f"Failures: {result.failures}"

    def test_nested_no_promotion(self, e2e):
        """(8 << 4) + 1 = 129, fits u8 at each step."""
        result = e2e.run('''
            #[entry]
            fn main() {
                A = (8 << 4) + 1;
            }
        ''', ExpectedState(A=129))
        assert result.success, f"Failures: {result.failures}"

    def test_macro_shift_promotes(self, e2e):
        """Macro arguments should also benefit from promotion."""
        result = e2e.run('''
            #[zeropage]
            static mut VAL: u16;

            macro_rules! make_val($v:expr, $s:expr) {
                VAL = $v << $s;
            }

            #[entry]
            fn main() {
                make_val!(8, 5);
                X = VAL;
            }
        ''', ExpectedState(X=256))
        assert result.success, f"Failures: {result.failures}"
