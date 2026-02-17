"""
End-to-end tests for U32 (32-bit unsigned integer) stdlib.

Compiles R65 source through the full pipeline, executes on the emulator,
and validates mathematical correctness of all U32 operations.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
U32_PATH = STDLIB_DIR / "U32.r65"


def u32_bytes(value):
    """Convert a 32-bit unsigned value to 4-byte little-endian list."""
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


# Zeropage address for the primary result variable
RESULT_ADDR = 0x10
RESULT_SNES = 0x7E0000 + RESULT_ADDR

# Common header: includes sneslib + U32, declares result at known address,
# secondary operand in #[ram] (far pointer via &), and scratch registers.
HEADER = f'''
    include!("{SNESLIB_PATH}")
    include!("{U32_PATH}")

    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;

    #[zeropage({RESULT_ADDR})]
    static mut V: U32;

    #[ram]
    static mut W: U32;
'''

# Header with a second zeropage result for tests needing two checks
HEADER2 = f'''
    include!("{SNESLIB_PATH}")
    include!("{U32_PATH}")

    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;

    #[zeropage({RESULT_ADDR})]
    static mut V: U32;

    #[zeropage(0x14)]
    static mut R: u16;

    #[zeropage(0x16)]
    static mut CMP_RESULT: u8;

    #[ram]
    static mut W: U32;
'''


class TestU32FromU16:
    """Test U32::from_u16 initialization."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_from_u16_zero(self, e2e):
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_u16(0); }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(0)}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_u16_small(self, e2e):
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_u16(1000); }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(1000)}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_u16_max(self, e2e):
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() { V.from_u16(65535); }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(65535)}))
        assert result.success, f"Failures: {result.failures}"


class TestU32ToU16:
    """Test U32::to_u16 truncation."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_to_u16_basic(self, e2e):
        """to_u16 returns low 16 bits."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_u16(1234);
                R = V.to_u16();
            }
        ''', ExpectedState(memory={
            0x7E0014: [0xD2, 0x04],  # 1234 = 0x04D2 LE
        }))
        assert result.success, f"Failures: {result.failures}"


class TestU32Copy:
    """Test U32::copy."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_copy_value(self, e2e):
        """Copy transfers both lo and hi words."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                W.from_u16(42);
                V.copy(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(42)}))
        assert result.success, f"Failures: {result.failures}"


class TestU32Add:
    """Test U32::add arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_add_basic(self, e2e):
        """100 + 200 = 300."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(100);
                W.from_u16(200);
                V.add(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(300)}))
        assert result.success, f"Failures: {result.failures}"

    def test_add_carry(self, e2e):
        """65535 + 1 = 65536 (carry from lo to hi)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(65535);
                W.from_u16(1);
                V.add(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(65536)}))
        assert result.success, f"Failures: {result.failures}"

    def test_add_large(self, e2e):
        """40000 + 30000 = 70000 (crosses u16 boundary)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(40000);
                W.from_u16(30000);
                V.add(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(70000)}))
        assert result.success, f"Failures: {result.failures}"

    def test_add_zero(self, e2e):
        """1000 + 0 = 1000 (identity)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1000);
                W.from_u16(0);
                V.add(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(1000)}))
        assert result.success, f"Failures: {result.failures}"


class TestU32Sub:
    """Test U32::sub arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_sub_basic(self, e2e):
        """300 - 100 = 200."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(300);
                W.from_u16(100);
                V.sub(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(200)}))
        assert result.success, f"Failures: {result.failures}"

    def test_sub_to_zero(self, e2e):
        """100 - 100 = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(100);
                W.from_u16(100);
                V.sub(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(0)}))
        assert result.success, f"Failures: {result.failures}"

    def test_sub_borrow(self, e2e):
        """Build 65536 via add, then sub 1 = 65535 (borrow from hi to lo)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(65535);
                W.from_u16(1);
                V.add(&W);
                // V = 65536
                W.from_u16(1);
                V.sub(&W);
                // V = 65535
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(65535)}))
        assert result.success, f"Failures: {result.failures}"


class TestU32Mul:
    """Test U32::mul arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_mul_basic(self, e2e):
        """100 * 200 = 20000."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(100);
                W.from_u16(200);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(20000)}),
                          max_instructions=50000)
        assert result.success, f"Failures: {result.failures}"

    def test_mul_by_zero(self, e2e):
        """12345 * 0 = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(12345);
                W.from_u16(0);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(0)}),
                          max_instructions=50000)
        assert result.success, f"Failures: {result.failures}"

    def test_mul_by_one(self, e2e):
        """12345 * 1 = 12345."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(12345);
                W.from_u16(1);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(12345)}),
                          max_instructions=50000)
        assert result.success, f"Failures: {result.failures}"

    def test_mul_large(self, e2e):
        """1000 * 100 = 100000 (exceeds u16)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1000);
                W.from_u16(100);
                V.mul(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(100000)}),
                          max_instructions=50000)
        assert result.success, f"Failures: {result.failures}"


class TestU32Div:
    """Test U32::div arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_div_exact(self, e2e):
        """1000 / 10 = 100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1000);
                W.from_u16(10);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(100)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_truncation(self, e2e):
        """1000 / 7 = 142 (truncates toward zero)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1000);
                W.from_u16(7);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(142)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_by_one(self, e2e):
        """42 / 1 = 42."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(42);
                W.from_u16(1);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(42)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_zero_numerator(self, e2e):
        """0 / 5 = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(0);
                W.from_u16(5);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(0)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_div_by_zero(self, e2e):
        """1000 / 0 = 0xFFFFFFFF (sentinel)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1000);
                W.from_u16(0);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(0xFFFFFFFF)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"


class TestU32Mod:
    """Test U32::mod arithmetic."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_mod_basic(self, e2e):
        """1000 % 7 = 6."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1000);
                W.from_u16(7);
                V.mod(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(6)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_mod_no_remainder(self, e2e):
        """1000 % 10 = 0."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1000);
                W.from_u16(10);
                V.mod(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(0)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_mod_smaller_than_divisor(self, e2e):
        """3 % 5 = 3."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(3);
                W.from_u16(5);
                V.mod(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(3)}),
                          max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"


class TestU32Cmp:
    """Test U32::cmp comparison."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_cmp_equal(self, e2e):
        """100 == 100 returns 0."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_u16(100);
                W.from_u16(100);
                A = V.cmp(&W);
                CMP_RESULT = A;
            }
        ''', ExpectedState(memory={0x7E0016: 0}))
        assert result.success, f"Failures: {result.failures}"

    def test_cmp_greater(self, e2e):
        """200 > 100 returns 1."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_u16(200);
                W.from_u16(100);
                A = V.cmp(&W);
                CMP_RESULT = A;
            }
        ''', ExpectedState(memory={0x7E0016: 1}))
        assert result.success, f"Failures: {result.failures}"

    def test_cmp_less(self, e2e):
        """100 < 200 returns 0xFF."""
        result = e2e.run(HEADER2 + '''
            #[entry]
            fn main() {
                V.from_u16(100);
                W.from_u16(200);
                A = V.cmp(&W);
                CMP_RESULT = A;
            }
        ''', ExpectedState(memory={0x7E0016: 0xFF}))
        assert result.success, f"Failures: {result.failures}"


class TestU32Shl:
    """Test U32::shl (shift left)."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_shl_basic(self, e2e):
        """1 << 4 = 16."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1);
                V.shl(4);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(16)}))
        assert result.success, f"Failures: {result.failures}"

    def test_shl_cross_word(self, e2e):
        """1 << 16 = 65536 (bit moves from lo to hi word)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1);
                V.shl(16);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(65536)}),
                          max_instructions=20000)
        assert result.success, f"Failures: {result.failures}"

    def test_shl_zero(self, e2e):
        """42 << 0 = 42 (no shift)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(42);
                V.shl(0);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(42)}))
        assert result.success, f"Failures: {result.failures}"


class TestU32Shr:
    """Test U32::shr (shift right)."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_shr_basic(self, e2e):
        """1024 >> 2 = 256."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1024);
                V.shr(2);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(256)}))
        assert result.success, f"Failures: {result.failures}"

    def test_shr_to_zero(self, e2e):
        """1 >> 1 = 0 (bit shifted out)."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1);
                V.shr(1);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(0)}))
        assert result.success, f"Failures: {result.failures}"

    def test_shr_cross_word(self, e2e):
        """Build 65536 (0x10000) then >> 16 = 1."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(1);
                V.shl(16);
                // V = 65536
                V.shr(16);
                // V = 1
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(1)}),
                          max_instructions=50000)
        assert result.success, f"Failures: {result.failures}"


class TestU32Combined:
    """Test combined U32 operations for consistency."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_add_then_sub(self, e2e):
        """(100 + 200) - 200 = 100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(100);
                W.from_u16(200);
                V.add(&W);
                V.sub(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(100)}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_then_div(self, e2e):
        """(100 * 10) / 10 = 100."""
        result = e2e.run(HEADER + '''
            #[entry]
            fn main() {
                V.from_u16(100);
                W.from_u16(10);
                V.mul(&W);
                V.div(&W);
            }
        ''', ExpectedState(memory={RESULT_SNES: u32_bytes(100)}),
                          max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"
