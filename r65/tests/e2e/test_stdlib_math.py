"""
End-to-end tests for stdlib math functions.

Tests div8, mod8, shl8, shr8, and type casts.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
MATH_PATH = STDLIB_DIR / "math.r65"


class TestDiv8:
    """Test 8-bit division."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_div8_basic(self, e2e):
        """Test basic division: 100/10=10, 255/5=51, 7/2=3."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;

            #[entry]
            fn main() {{
                R1 = div8(100, 10);
                R2 = div8(255, 5);
                R3 = div8(7, 2);
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 10,
            0x7E0011: 51,
            0x7E0012: 3,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_div8_edge_cases(self, e2e):
        """Test edge cases: 42/1=42, 128/128=1, 0/5=0."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;

            #[entry]
            fn main() {{
                R1 = div8(42, 1);
                R2 = div8(128, 128);
                R3 = div8(0, 5);
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 42,
            0x7E0011: 1,
            0x7E0012: 0,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMod8:
    """Test 8-bit modulo."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_mod8_basic(self, e2e):
        """Test basic modulo: 10%3=1, 100%7=2, 15%5=0."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;

            #[entry]
            fn main() {{
                R1 = mod8(10, 3);
                R2 = mod8(100, 7);
                R3 = mod8(15, 5);
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 1,
            0x7E0011: 2,
            0x7E0012: 0,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestShift:
    """Test variable-amount shift operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_shl8_variable(self, e2e):
        """Test variable left shift: shl8(1,0)=1, shl8(1,3)=8, shl8(3,4)=48, shl8(1,7)=128."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;
            #[zeropage(0x13)]
            static mut R4: u8;

            #[entry]
            fn main() {{
                R1 = shl8(1, 0);
                R2 = shl8(1, 3);
                R3 = shl8(3, 4);
                R4 = shl8(1, 7);
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 1,
            0x7E0011: 8,
            0x7E0012: 48,
            0x7E0013: 128,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_shr8_variable(self, e2e):
        """Test variable right shift: shr8(128,1)=64, shr8(255,4)=15, shr8(1,0)=1."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut R1: u8;
            #[zeropage(0x11)]
            static mut R2: u8;
            #[zeropage(0x12)]
            static mut R3: u8;

            #[entry]
            fn main() {{
                R1 = shr8(128, 1);
                R2 = shr8(255, 4);
                R3 = shr8(1, 0);
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 64,
            0x7E0011: 15,
            0x7E0012: 1,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestTypeCasts:
    """Test type conversion operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_u8_to_u16_cast(self, e2e):
        """Test u8 -> u16 zero extension: 0xAB -> [0xAB, 0x00]."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut WIDE: u16;

            #[entry]
            fn main() {
                A = 0xAB;
                WIDE = A as u16;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0xAB, 0x00],  # LE: low byte first
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_to_u8_truncation(self, e2e):
        """Test u16 -> u8 truncation: 0x1234 -> 0x34."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut WIDE: u16 = 0x1234;
            #[zeropage(0x12)]
            static mut NARROW: u8;

            #[entry]
            fn main() {
                A = WIDE as u8;
                NARROW = A;
            }
        ''', ExpectedState(A=0x34, memory={
            0x7E0012: 0x34,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_cast_then_add(self, e2e):
        """Test u8 -> u16 cast followed by u16 addition to avoid overflow.
        200 + 100 = 300 (overflow in u8, correct in u16).
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            fn add_wide(a @ A: u8, b: u8) -> u16 {
                let wide @ A : u16 = a as u16;
                A = A + (b as u16);
                return A;
            }

            #[entry]
            fn main() {
                RESULT = add_wide(200, 100);
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x2C, 0x01],  # 300 = 0x012C LE
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_cast_then_multiply(self, e2e):
        """Test u8 -> u16 cast followed by u16 constant multiply.
        200 * 2 = 400 (overflow in u8, correct in u16).
        Regression: ASL was emitted in m8 mode, truncating values > 127.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            fn double_wide(val: u8) -> u16 {
                let wide: u16 = val as u16 * 2;
                return wide;
            }

            #[entry]
            fn main() {
                RESULT = double_wide(200);
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x90, 0x01],  # 400 = 0x0190 LE
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_cast_then_divide(self, e2e):
        """Test u16 constant divide uses 16-bit LSR.
        0x0180 / 2 = 0x00C0 (192). In m8 mode LSR would only shift low byte.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            #[entry]
            fn main() {
                let val: u16 = 0x0180;
                RESULT = val / 2;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0xC0, 0x00],  # 0x00C0 LE
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_sign_extension_negative(self, e2e):
        """Test i8 -> i16 sign extension of negative value: -5 (0xFB) -> 0xFFFB."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: i16;

            #[entry]
            fn main() {
                A = 0xFB;
                let signed: i8 = A as i8;
                let wide @ A : i16 = signed as i16;
                RESULT = A;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0xFB, 0xFF],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_sign_extension_positive(self, e2e):
        """Test i8 -> i16 sign extension of positive value: +5 -> 0x0005."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: i16;

            #[entry]
            fn main() {
                A = 5;
                let signed: i8 = A as i8;
                let wide @ A : i16 = signed as i16;
                RESULT = A;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x05, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMathPipeline:
    """Test combining multiple math operations."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_mul_then_div(self, e2e):
        """Test div8(mul8(15,8), 4) = 30."""
        source = f'''
            include!("{SNESLIB_PATH}")
            include!("{MATH_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                // mul8(15, 8) = 120 (lo=120, hi=0)
                let lo: u8 = mul8(15, 8);
                // div8(120, 4) = 30
                RESULT = div8(lo, 4);
            }}
        '''
        result = e2e.run(source, ExpectedState(memory={
            0x7E0010: 30,
        }))
        assert result.success, f"Failures: {result.failures}"
