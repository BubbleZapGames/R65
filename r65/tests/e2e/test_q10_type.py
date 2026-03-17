# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for the q10 type alias and Q10.6 fixed-point operations.

Verifies that the `type q10 = i16;` alias and q10_neg, q10_abs, q10_mul
work through the full pipeline.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
Q10_PATH = STDLIB_DIR / "q10_type.r65"


class TestQ10TypeAlias:
    """Test the q10 type alias from q10_type.r65."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_q10_static_variable(self, e2e):
        """Declare a static variable with the q10 type alias."""
        result = e2e.run('''
            type q10 = i16;

            #[zeropage(0x10)]
            static mut VELOCITY: q10;

            #[entry]
            fn main() {
                VELOCITY = 64;
            }
        ''', ExpectedState(memory={
            0x7E0010: 64,
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_q10_function_param_and_return(self, e2e):
        """Use q10 type alias in function parameter (stack) and return type."""
        result = e2e.run('''
            type q10 = i16;

            #[zeropage(0x10)]
            static mut RESULT: q10;

            fn add_one(val: q10) -> q10 {
                return val + 1;
            }

            #[entry]
            fn main() {
                RESULT = add_one(63);
            }
        ''', ExpectedState(memory={
            0x7E0010: 64,
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_q10_let_binding(self, e2e):
        """Use q10 type alias in let binding type annotation."""
        result = e2e.run('''
            type q10 = i16;

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {
                let val: q10 = 128;
                OUT = val;
            }
        ''', ExpectedState(memory={
            0x7E0010: 128,
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_q10_with_macros(self, e2e):
        """Use q10 type alias together with Q10 macros."""
        result = e2e.run('''
            type q10 = i16;

            const Q10_FRAC_MASK: q10 = 0x3F;

            macro_rules! q10_from_int($n:expr) { $n << 6 }
            macro_rules! q10_to_int($q:expr) { $q >> 6 }

            #[zeropage(0x10)]
            static mut POS: q10;
            #[zeropage(0x12)]
            static mut INT_PART: i16;

            #[entry]
            fn main() {
                POS = q10_from_int!(3);
                INT_PART = q10_to_int!(POS);
            }
        ''', ExpectedState(memory={
            0x7E0010: 192,  # 3 << 6 = 192
            0x7E0011: 0,
            0x7E0012: 3,    # 192 >> 6 = 3
            0x7E0013: 0,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestQ10Neg:
    """Test q10_neg macro."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_neg_positive(self, e2e):
        """Negate a positive Q10.6 value: neg(3.0) = -3.0."""
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_neg!(192);
            }}
        ''', ExpectedState(memory={
            # -192 as u16 = 0xFF40
            0x7E0010: 0x40,
            0x7E0011: 0xFF,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_neg_negative(self, e2e):
        """Negate a negative Q10.6 value: neg(-2.0) = 2.0."""
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                let val: q10 = q10_neg!(128);
                OUT = q10_neg!(val);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 128,  # 2.0 in Q10.6
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestQ10Abs:
    """Test q10_abs function."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_abs_positive(self, e2e):
        """Absolute value of positive: abs(5.0) = 5.0."""
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_abs(320);
            }}
        ''', ExpectedState(memory={
            # 320 = 5.0 in Q10.6 = 0x0140
            0x7E0010: 0x40,
            0x7E0011: 0x01,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_abs_negative(self, e2e):
        """Absolute value of negative: abs(-3.0) = 3.0."""
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                let val: q10 = q10_neg!(192);
                OUT = q10_abs(val);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 192,  # 3.0 in Q10.6
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_abs_zero(self, e2e):
        """Absolute value of zero: abs(0) = 0."""
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_abs(0);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 0,
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestQ10Mul:
    """Test q10_mul function (SNES hardware multiply)."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_mul_integers(self, e2e):
        """3.0 * 4.0 = 12.0."""
        # 3.0 = 192, 4.0 = 256, 12.0 = 768
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_mul(192, 256);
            }}
        ''', ExpectedState(memory={
            # 768 = 0x0300
            0x7E0010: 0x00,
            0x7E0011: 0x03,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_fractional(self, e2e):
        """1.5 * 2.0 = 3.0."""
        # 1.5 = 1*64+32 = 96, 2.0 = 128, 3.0 = 192
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_mul(96, 128);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 192,
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_larger_values(self, e2e):
        """100.0 * 5.0 = 500.0."""
        # 100.0 = 6400, 5.0 = 320, 500.0 = 32000
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_mul(6400, 320);
            }}
        ''', ExpectedState(memory={
            # 32000 = 0x7D00
            0x7E0010: 0x00,
            0x7E0011: 0x7D,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_negative(self, e2e):
        """(-2.0) * 3.5 = -7.0."""
        # -2.0 = -128 (0xFF80), 3.5 = 224, -7.0 = -448 (0xFE40)
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                let a: q10 = q10_neg!(128);
                OUT = q10_mul(a, 224);
            }}
        ''', ExpectedState(memory={
            # -448 = 0xFE40
            0x7E0010: 0x40,
            0x7E0011: 0xFE,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_both_negative(self, e2e):
        """(-3.0) * (-4.0) = 12.0."""
        # -3.0 = -192 (0xFF40), -4.0 = -256 (0xFF00), 12.0 = 768
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                let a: q10 = q10_neg!(192);
                let b: q10 = q10_neg!(256);
                OUT = q10_mul(a, b);
            }}
        ''', ExpectedState(memory={
            # 768 = 0x0300
            0x7E0010: 0x00,
            0x7E0011: 0x03,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_by_zero(self, e2e):
        """5.0 * 0 = 0."""
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_mul(320, 0);
            }}
        ''', ExpectedState(memory={
            0x7E0010: 0,
            0x7E0011: 0,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_carry_path(self, e2e):
        """5.75 * 5.75 = 33.0625 — exercises carry in p0 + mid<<8."""
        # 5.75 in Q10.6 = 5*64+48 = 368 = 0x0170
        # 33.0625 in Q10.6 = 33*64+4 = 2116 = 0x0844
        result = e2e.run(f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")

            #[zeropage(0x10)]
            static mut OUT: q10;

            #[entry]
            fn main() {{
                OUT = q10_mul(368, 368);
            }}
        ''', ExpectedState(memory={
            # 2116 = 0x0844
            0x7E0010: 0x44,
            0x7E0011: 0x08,
        }))
        assert result.success, f"Failures: {result.failures}"
