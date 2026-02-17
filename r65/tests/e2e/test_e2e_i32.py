"""
End-to-end tests for I32 (32-bit signed integer) stdlib.

Compiles R65 source through the full pipeline, executes on the emulator,
and validates mathematical correctness of all I32 operations.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
I32_PATH = STDLIB_DIR / "I32.r65"


def i32_bytes(value):
    """Convert a 32-bit signed value to 4-byte little-endian list (two's complement)."""
    if value < 0:
        value = value + 0x100000000
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


RESULT_ADDR = 0x10
RESULT_SNES = 0x7E0000 + RESULT_ADDR

HEADER = f'''
    include!("{SNESLIB_PATH}")
    include!("{I32_PATH}")

    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;

    #[zeropage({RESULT_ADDR})]
    static mut V: I32;

    #[ram]
    static mut W: I32;
'''

HEADER2 = f'''
    include!("{SNESLIB_PATH}")
    include!("{I32_PATH}")

    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;

    #[zeropage({RESULT_ADDR})]
    static mut V: I32;

    #[zeropage(0x14)]
    static mut R: u16;

    #[zeropage(0x16)]
    static mut CMP_RESULT: u8;

    #[zeropage(0x17)]
    static mut BOOL_RESULT: u8;

    #[ram]
    static mut W: I32;
'''


class TestI32FromI16:
    """Test I32::from_i16 sign extension."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_from_i16_positive(self, e2e):
        """Positive i16: 1000 -> lo=1000, hi=0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_i16(1000 as i16); }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(1000)}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_i16_negative(self, e2e):
        """Negative i16: -100 -> sign extended to 0xFFFFFF9C."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_i16(-100 as i16); }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-100)}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_i16_zero(self, e2e):
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_i16(0 as i16); }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(0)}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_i16_minus_one(self, e2e):
        """-1 -> 0xFFFFFFFF."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_i16(-1 as i16); }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-1)}))
        assert result.success, f"Failures: {result.failures}"


class TestI32FromU16:
    """Test I32::from_u16 zero extension."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_from_u16_basic(self, e2e):
        """1000 zero-extends to lo=1000, hi=0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_u16(1000); }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(1000)}))
        assert result.success, f"Failures: {result.failures}"


class TestI32Neg:
    """Test I32::neg (two's complement negation)."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_neg_positive(self, e2e):
        """neg(100) = -100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                V.neg();
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-100)}))
        assert result.success, f"Failures: {result.failures}"

    def test_neg_negative(self, e2e):
        """neg(-100) = 100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-100 as i16);
                V.neg();
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(100)}))
        assert result.success, f"Failures: {result.failures}"

    def test_neg_zero(self, e2e):
        """neg(0) = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(0 as i16);
                V.neg();
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(0)}))
        assert result.success, f"Failures: {result.failures}"


class TestI32Abs:
    """Test I32::abs."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_abs_positive(self, e2e):
        """abs(100) = 100 (unchanged)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                V.abs();
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(100)}))
        assert result.success, f"Failures: {result.failures}"

    def test_abs_negative(self, e2e):
        """abs(-100) = 100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-100 as i16);
                V.abs();
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(100)}))
        assert result.success, f"Failures: {result.failures}"

    def test_abs_zero(self, e2e):
        """abs(0) = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(0 as i16);
                V.abs();
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(0)}))
        assert result.success, f"Failures: {result.failures}"


class TestI32IsNegative:
    """Test I32::is_negative."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_positive_is_not_negative(self, e2e):
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                if V.is_negative() {
                    BOOL_RESULT = 1;
                } else {
                    BOOL_RESULT = 0;
                }
            }
        ''', ExpectedState(memory={0x7E0017: 0}))
        assert result.success, f"Failures: {result.failures}"

    def test_negative_is_negative(self, e2e):
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_i16(-100 as i16);
                if V.is_negative() {
                    BOOL_RESULT = 1;
                } else {
                    BOOL_RESULT = 0;
                }
            }
        ''', ExpectedState(memory={0x7E0017: 1}))
        assert result.success, f"Failures: {result.failures}"


class TestI32Add:
    """Test I32::add arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_add_positive(self, e2e):
        """100 + 200 = 300."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(200 as i16);
                V.add(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(300)}))
        assert result.success, f"Failures: {result.failures}"

    def test_add_negative(self, e2e):
        """100 + (-50) = 50."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(-50 as i16);
                V.add(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(50)}))
        assert result.success, f"Failures: {result.failures}"

    def test_add_both_negative(self, e2e):
        """-100 + (-200) = -300."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-100 as i16);
                W.from_i16(-200 as i16);
                V.add(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-300)}))
        assert result.success, f"Failures: {result.failures}"


class TestI32Sub:
    """Test I32::sub arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_sub_basic(self, e2e):
        """300 - 100 = 200."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(300 as i16);
                W.from_i16(100 as i16);
                V.sub(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(200)}))
        assert result.success, f"Failures: {result.failures}"

    def test_sub_to_negative(self, e2e):
        """100 - 200 = -100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(200 as i16);
                V.sub(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-100)}))
        assert result.success, f"Failures: {result.failures}"


class TestI32Mul:
    """Test I32::mul arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_mul_positive(self, e2e):
        """100 * 200 = 20000."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(200 as i16);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(20000)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_mul_negative(self, e2e):
        """100 * (-5) = -500."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(-5 as i16);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-500)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_mul_both_negative(self, e2e):
        """(-10) * (-20) = 200."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-10 as i16);
                W.from_i16(-20 as i16);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(200)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_mul_by_zero(self, e2e):
        """100 * 0 = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(0 as i16);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(0)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"


class TestI32Div:
    """Test I32::div arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_div_exact(self, e2e):
        """1000 / 10 = 100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(1000 as i16);
                W.from_i16(10 as i16);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(100)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_negative_dividend(self, e2e):
        """-1000 / 10 = -100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-1000 as i16);
                W.from_i16(10 as i16);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-100)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_negative_divisor(self, e2e):
        """1000 / (-10) = -100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(1000 as i16);
                W.from_i16(-10 as i16);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-100)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_both_negative(self, e2e):
        """-1000 / (-10) = 100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-1000 as i16);
                W.from_i16(-10 as i16);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(100)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_truncation_toward_zero(self, e2e):
        """7 / 2 = 3 (not 3.5, truncates toward zero)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(7 as i16);
                W.from_i16(2 as i16);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(3)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_by_zero(self, e2e):
        """1000 / 0 = MIN_I32 (0x80000000) sentinel."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(1000 as i16);
                W.from_i16(0 as i16);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-2147483648)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"


class TestI32Mod:
    """Test I32::mod arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_mod_basic(self, e2e):
        """1000 % 7 = 6."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(1000 as i16);
                W.from_i16(7 as i16);
                V.mod(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(6)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"

    def test_mod_negative_dividend(self, e2e):
        """-1000 % 7 = -6 (remainder has sign of dividend)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-1000 as i16);
                W.from_i16(7 as i16);
                V.mod(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-6)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"

    def test_mod_no_remainder(self, e2e):
        """1000 % 10 = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(1000 as i16);
                W.from_i16(10 as i16);
                V.mod(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(0)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"


class TestI32Cmp:
    """Test I32::cmp signed comparison."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_cmp_equal(self, e2e):
        """100 == 100 returns 0."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(100 as i16);
                A = V.cmp(&W);
                CMP_RESULT = A;
            }
        ''', ExpectedState(memory={0x7E0016: 0}))
        assert result.success, f"Failures: {result.failures}"

    def test_cmp_greater(self, e2e):
        """100 > -100 returns 1."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_i16(100 as i16);
                W.from_i16(-100 as i16);
                A = V.cmp(&W);
                CMP_RESULT = A;
            }
        ''', ExpectedState(memory={0x7E0016: 1}))
        assert result.success, f"Failures: {result.failures}"

    def test_cmp_less(self, e2e):
        """-100 < 100 returns 0xFF."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_i16(-100 as i16);
                W.from_i16(100 as i16);
                A = V.cmp(&W);
                CMP_RESULT = A;
            }
        ''', ExpectedState(memory={0x7E0016: 0xFF}))
        assert result.success, f"Failures: {result.failures}"

    def test_cmp_both_negative(self, e2e):
        """-10 > -100 returns 1."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_i16(-10 as i16);
                W.from_i16(-100 as i16);
                A = V.cmp(&W);
                CMP_RESULT = A;
            }
        ''', ExpectedState(memory={0x7E0016: 1}))
        assert result.success, f"Failures: {result.failures}"


class TestI32Shl:
    """Test I32::shl (shift left)."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_shl_basic(self, e2e):
        """1 << 4 = 16."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(1 as i16);
                V.shl(4);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(16)}))
        assert result.success, f"Failures: {result.failures}"


class TestI32Sar:
    """Test I32::sar (arithmetic shift right, preserves sign)."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_sar_positive(self, e2e):
        """16 >> 2 = 4."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(16 as i16);
                V.sar(2);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(4)}))
        assert result.success, f"Failures: {result.failures}"

    def test_sar_negative_preserves_sign(self, e2e):
        """-16 >> 2 = -4 (sign bit preserved)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-16 as i16);
                V.sar(2);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-4)}))
        assert result.success, f"Failures: {result.failures}"

    def test_sar_minus_one(self, e2e):
        """-1 >> 1 = -1 (all bits set, sign preserved)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_i16(-1 as i16);
                V.sar(1);
            }
        ''', ExpectedState(memory={RESULT_SNES: i32_bytes(-1)}))
        assert result.success, f"Failures: {result.failures}"
