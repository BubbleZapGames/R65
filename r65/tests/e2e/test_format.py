"""
End-to-end tests for the format! built-in macro.

Tests that format! generates correct output in memory by compiling R65 source,
running on the emulator, and validating buffer contents.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest, ExpectedState

# Path to stdlib
STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
STRING_PATH = STDLIB_DIR / "string.r65"


def buf_addr(offset=0):
    """SNES address for BUF (auto-allocated RAM at 0x7E2000)."""
    return 0x7E2000 + offset


def result_addr(offset=0):
    """SNES address for RESULT zeropage var at 0x10."""
    return 0x7E0010 + offset


def ascii_bytes_null(s):
    """Convert a Python string to list of ASCII byte values with null terminator."""
    return [ord(c) for c in s] + [0]


# ============================================================================
# Basic Format Tests
# ============================================================================

class TestFormatLiteral:
    """Test format! with literal-only format strings."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_literal_only(self, e2e):
        """format! with only literal text produces correct output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "Hello");
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
            }}
        ''', ExpectedState(memory={
            # "Hello\0"
            result_addr(): ascii_bytes_null("Hello")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_empty_string(self, e2e):
        """format! with empty format string produces just null."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "");
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x00, 0xFF]
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Numeric Format Tests
# ============================================================================

class TestFormatU8:
    """Test format! with {u8} specifier."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_u8_single(self, e2e):
        """format! with single {u8} value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "N:{{u8}}", 42);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
            }}
        ''', ExpectedState(memory={
            # "N:42\0"
            result_addr(): ascii_bytes_null("N:42") + [0xFF]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestFormatU16:
    """Test format! with {u16} specifier."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_u16_single(self, e2e):
        """format! with single {u16} value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "{{u16}}", 65535);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
            }}
        ''', ExpectedState(memory={
            # "65535\0"
            result_addr(): ascii_bytes_null("65535")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u16_zero(self, e2e):
        """format! with {u16} value 0."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "{{u16}}", 0);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
            }}
        ''', ExpectedState(memory={
            # "0\0"
            result_addr(): ascii_bytes_null("0")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Hex Format Tests
# ============================================================================

class TestFormatHex:
    """Test format! with hex specifiers."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_u8_hex(self, e2e):
        """format! with {u8:x} produces 2-char hex."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "${{u8:x}}", 0xAB);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
            }}
        ''', ExpectedState(memory={
            # "$AB\0"
            result_addr(): ascii_bytes_null("$AB")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u16_hex(self, e2e):
        """format! with {u16:x} produces 4-char hex."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 7];

            #[entry]
            fn main() {{
                format!(BUF, "0x{{u16:x}}", 0xDEAD);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
                RESULT[6] = BUF[6];
            }}
        ''', ExpectedState(memory={
            # "0xDEAD\0"
            result_addr(): ascii_bytes_null("0xDEAD")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Mixed Format Tests
# ============================================================================

class TestFormatMixed:
    """Test format! with multiple specifiers."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_mixed_u8_u16hex(self, e2e):
        """format! with {u8} and {u16:x} together."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 12];

            #[entry]
            fn main() {{
                format!(BUF, "HP:{{u8}} ${{u16:x}}", 99, 0x00AB);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
                RESULT[6] = BUF[6];
                RESULT[7] = BUF[7];
                RESULT[8] = BUF[8];
                RESULT[9] = BUF[9];
                RESULT[10] = BUF[10];
                RESULT[11] = BUF[11];
            }}
        ''', ExpectedState(memory={
            # "HP:99 $00AB\0"
            result_addr(): ascii_bytes_null("HP:99 $00AB")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Char Format Tests
# ============================================================================

class TestFormatChar:
    """Test format! with {c} specifier."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_char(self, e2e):
        """format! with {c} writes single byte."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "A{{c}}B", 0x58);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
            }}
        ''', ExpectedState(memory={
            # "AXB\0"
            result_addr(): ascii_bytes_null("AXB")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# String Format Tests
# ============================================================================

class TestFormatString:
    """Test format! with {s} specifier."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_string(self, e2e):
        """format! with {s} copies string."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];
            #[ram]
            static mut NAME: [u8; 8] = "World\\0";

            #[zeropage(0x10)]
            static mut RESULT: [u8; 10];

            #[entry]
            fn main() {{
                format!(BUF, "Hi {{s}}!", &NAME as far *u8);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
                RESULT[6] = BUF[6];
                RESULT[7] = BUF[7];
                RESULT[8] = BUF[8];
                RESULT[9] = BUF[9];
            }}
        ''', ExpectedState(memory={
            # "Hi World!\0"
            result_addr(): ascii_bytes_null("Hi World!")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Escaped Brace Tests
# ============================================================================

class TestFormatEscapedBraces:
    """Test format! with {{ and }} for literal braces."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_escaped_braces(self, e2e):
        """format! with {{braces}} produces {braces}."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 9];

            #[entry]
            fn main() {{
                format!(BUF, "{{{{braces}}}}");
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
                RESULT[6] = BUF[6];
                RESULT[7] = BUF[7];
                RESULT[8] = BUF[8];
            }}
        ''', ExpectedState(memory={
            # "{braces}\0"
            result_addr(): ascii_bytes_null("{braces}")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Padded Format Tests
# ============================================================================

class TestFormatPadded:
    """Test format! with {u16:Nd} padded specifier."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_padded(self, e2e):
        """format! with {u16:5d} produces space-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                format!(BUF, "[{{u16:5d}}]", 42);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
                RESULT[6] = BUF[6];
                RESULT[7] = BUF[7];
            }}
        ''', ExpectedState(memory={
            # "[   42]\0"
            result_addr(): ascii_bytes_null("[   42]")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u16_zero_padded(self, e2e):
        """format! with {u16:05d} produces zero-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "{{u16:05d}}", 42);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
            }}
        ''', ExpectedState(memory={
            # "00042\0"
            result_addr(): ascii_bytes_null("00042")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u8_padded(self, e2e):
        """format! with {u8:3d} produces space-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "{{u8:3d}}", 7);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
            }}
        ''', ExpectedState(memory={
            # "  7\0"
            result_addr(): ascii_bytes_null("  7")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u8_zero_padded(self, e2e):
        """format! with {u8:03d} produces zero-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "{{u8:03d}}", 7);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
            }}
        ''', ExpectedState(memory={
            # "007\0"
            result_addr(): ascii_bytes_null("007")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Bool Format Tests
# ============================================================================

class TestFormatBool:
    """Test format! with {bool} specifier."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_bool_true(self, e2e):
        """format! with {bool} and true produces "1"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "{{bool}}", true);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes_null("1")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_bool_false(self, e2e):
        """format! with {bool} and false produces "0"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "{{bool}}", false);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes_null("0")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Signed Integer Format Tests
# ============================================================================

class TestFormatSigned:
    """Test format! with signed integer specifiers."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_i8_positive(self, e2e):
        """format! with {i8} and positive value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            #[entry]
            fn main() {{
                let v: i8 = 42;
                format!(BUF, "{{i8}}", v);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
            }}
        ''', ExpectedState(memory={
            # "42\0"
            result_addr(): ascii_bytes_null("42")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_i8_negative(self, e2e):
        """format! with {i8} and negative value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            #[entry]
            fn main() {{
                let v: i8 = 0 - 1;
                format!(BUF, "{{i8}}", v);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
            }}
        ''', ExpectedState(memory={
            # "-1\0"
            result_addr(): ascii_bytes_null("-1")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_i16_positive(self, e2e):
        """format! with {i16} and positive value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                let v: i16 = 1000;
                format!(BUF, "{{i16}}", v);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
            }}
        ''', ExpectedState(memory={
            # "1000\0"
            result_addr(): ascii_bytes_null("1000")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_i16_negative(self, e2e):
        """format! with {i16} and negative value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                let v: i16 = 0 - 42;
                format!(BUF, "{{i16}}", v);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
            }}
        ''', ExpectedState(memory={
            # "-42\0"
            result_addr(): ascii_bytes_null("-42")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Small Literal Inlining Tests
# ============================================================================

class TestFormatLiteralInlining:
    """Test that small literals are inlined (1-3 bytes) and large use memcpy."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_format_small_literal(self, e2e):
        """format! with 2-byte literal produces correct output via inlining."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "X={{u8}}", 42);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = BUF[5];
            }}
        ''', ExpectedState(memory={
            # "X=42\0"
            result_addr(): ascii_bytes_null("X=42") + [0xFF]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_single_byte_literal(self, e2e):
        """format! with 1-byte literal produces correct output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "Z");
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes_null("Z")
        }))
        assert result.success, f"Failures: {result.failures}"
