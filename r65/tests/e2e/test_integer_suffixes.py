"""
End-to-end tests for integer literal type suffixes (u8, u16, i8, i16).
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestIntegerSuffixes:
    """Test integer literal suffix support."""

    @pytest.fixture
    def e2e(self):
        """Create E2ETest instance."""
        return E2ETest()

    def test_u8_suffix(self, e2e):
        """Test u8 suffix on integer literal."""
        result = e2e.run('''
            #[entry]
            fn main() {
                A = 255u8;
            }
        ''', ExpectedState(A=0xFF))

        assert result.success, f"Failures: {result.failures}"

    def test_u16_suffix_forces_wide(self, e2e):
        """Test u16 suffix forces 16-bit type even for small values."""
        result = e2e.run('''
            #[zeropage]
            static mut VAL: u16;

            #[entry]
            fn main() {
                VAL = 0u16;
                X = VAL;
            }
        ''', ExpectedState(X=0))

        assert result.success, f"Failures: {result.failures}"

    def test_hex_with_suffix(self, e2e):
        """Test hex literal with u8 suffix."""
        result = e2e.run('''
            #[entry]
            fn main() {
                A = 0xFFu8;
            }
        ''', ExpectedState(A=0xFF))

        assert result.success, f"Failures: {result.failures}"

    def test_binary_with_suffix(self, e2e):
        """Test binary literal with u16 suffix."""
        result = e2e.run('''
            #[zeropage]
            static mut VAL: u16;

            #[entry]
            fn main() {
                VAL = 0b0001_0000u16;
                X = VAL;
            }
        ''', ExpectedState(X=0x10))

        assert result.success, f"Failures: {result.failures}"

    def test_suffix_in_arithmetic(self, e2e):
        """Test suffixed literals in arithmetic expressions."""
        result = e2e.run('''
            #[entry]
            fn main() {
                A = 10u8 + 20u8;
            }
        ''', ExpectedState(A=30))

        assert result.success, f"Failures: {result.failures}"
