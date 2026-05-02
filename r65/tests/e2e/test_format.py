# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for the format! built-in macro.

Tests that format! generates correct output in memory by compiling R65 source,
running on the emulator, and validating buffer contents.
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

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

    def test_format_literal_only(self, e2e):
        """format! with only literal text produces correct output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "Hello");
            }}
        ''', ExpectedState(memory={
            # "Hello\0"
            0x7E2000: ascii_bytes_null("Hello")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_empty_string(self, e2e):
        """format! with empty format string produces just null."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "");
            }}
        ''', ExpectedState(memory={
            0x7E2000: [0x00, 0xFF]
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Numeric Format Tests
# ============================================================================

class TestFormatU8:
    """Test format! with {u8} specifier."""

    def test_format_u8_single(self, e2e):
        """format! with single {u8} value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "N:{{u8}}", 42);
            }}
        ''', ExpectedState(memory={
            # "N:42\0"
            0x7E2000: ascii_bytes_null("N:42") + [0xFF]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestFormatU16:
    """Test format! with {u16} specifier."""

    def test_format_u16_single(self, e2e):
        """format! with single {u16} value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "{{u16}}", 65535);
            }}
        ''', ExpectedState(memory={
            # "65535\0"
            0x7E2000: ascii_bytes_null("65535")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u16_zero(self, e2e):
        """format! with {u16} value 0."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "{{u16}}", 0);
            }}
        ''', ExpectedState(memory={
            # "0\0"
            0x7E2000: ascii_bytes_null("0")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Hex Format Tests
# ============================================================================

class TestFormatHex:
    """Test format! with hex specifiers."""

    def test_format_u8_hex(self, e2e):
        """format! with {u8:x} produces 2-char hex."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "${{u8:x}}", 0xAB);
            }}
        ''', ExpectedState(memory={
            # "$AB\0"
            0x7E2000: ascii_bytes_null("$AB")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u16_hex(self, e2e):
        """format! with {u16:x} produces 4-char hex."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 7];

            #[entry]
            fn main() {{
                format!(BUF, "0x{{u16:x}}", 0xDEAD);
            }}
        ''', ExpectedState(memory={
            # "0xDEAD\0"
            0x7E2000: ascii_bytes_null("0xDEAD")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Mixed Format Tests
# ============================================================================

class TestFormatMixed:
    """Test format! with multiple specifiers."""

    def test_format_mixed_u8_u16hex(self, e2e):
        """format! with {u8} and {u16:x} together."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 12];

            #[entry]
            fn main() {{
                format!(BUF, "HP:{{u8}} ${{u16:x}}", 99, 0x00AB);
            }}
        ''', ExpectedState(memory={
            # "HP:99 $00AB\0"
            0x7E2000: ascii_bytes_null("HP:99 $00AB")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Char Format Tests
# ============================================================================

class TestFormatChar:
    """Test format! with {c} specifier."""

    def test_format_char(self, e2e):
        """format! with {c} writes single byte."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "A{{c}}B", 0x58);
            }}
        ''', ExpectedState(memory={
            # "AXB\0"
            0x7E2000: ascii_bytes_null("AXB")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# String Format Tests
# ============================================================================

class TestFormatString:
    """Test format! with {s} specifier."""

    def test_format_string(self, e2e):
        """format! with {s} copies string."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];
            #[ram(0x7E2020)]
            static mut NAME: [u8; 8] = "World\\0";

            #[zeropage(0x10)]
            static mut RESULT: [u8; 10];

            #[entry]
            fn main() {{
                format!(BUF, "Hi {{s}}!", &NAME as far *u8);
            }}
        ''', ExpectedState(memory={
            # "Hi World!\0"
            0x7E2000: ascii_bytes_null("Hi World!")
        }))
        assert result.success, f"Failures: {result.failures}"


class TestFormatToString:
    """Test format! {s} dispatch to ToString impls."""

    def test_format_tostring_u32(self, e2e):
        """format! {s} on a U32 dispatches to its ToString impl."""
        u32_path = STDLIB_DIR / "U32.r65"
        result = e2e.run(f'''
            include!("{STRING_PATH}")
            include!("{u32_path}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];
            #[ram(0x7E2020)]
            static mut N: U32;

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                N.lo = 12345;
                N.hi = 0;
                format!(BUF, "n={{s}}!", N);
            }}
        ''', ExpectedState(memory={
            # "n=12345!\0"
            0x7E2000: [ord('n'), ord('='), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('!')]
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Escaped Brace Tests
# ============================================================================

class TestFormatEscapedBraces:
    """Test format! with {{ and }} for literal braces."""

    def test_format_escaped_braces(self, e2e):
        """format! with {{braces}} produces {braces}."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 9];

            #[entry]
            fn main() {{
                format!(BUF, "{{{{braces}}}}");
            }}
        ''', ExpectedState(memory={
            # "{braces}\0"
            0x7E2000: ascii_bytes_null("{braces}")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Padded Format Tests
# ============================================================================

class TestFormatPadded:
    """Test format! with {u16:Nd} padded specifier."""

    def test_format_padded(self, e2e):
        """format! with {u16:5d} produces space-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                format!(BUF, "[{{u16:5d}}]", 42);
            }}
        ''', ExpectedState(memory={
            # "[   42]\0"
            0x7E2000: ascii_bytes_null("[   42]")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u16_zero_padded(self, e2e):
        """format! with {u16:05d} produces zero-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "{{u16:05d}}", 42);
            }}
        ''', ExpectedState(memory={
            # "00042\0"
            0x7E2000: ascii_bytes_null("00042")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u8_padded(self, e2e):
        """format! with {u8:3d} produces space-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "{{u8:3d}}", 7);
            }}
        ''', ExpectedState(memory={
            # "  7\0"
            0x7E2000: ascii_bytes_null("  7")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_u8_zero_padded(self, e2e):
        """format! with {u8:03d} produces zero-padded output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                format!(BUF, "{{u8:03d}}", 7);
            }}
        ''', ExpectedState(memory={
            # "007\0"
            0x7E2000: ascii_bytes_null("007")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Bool Format Tests
# ============================================================================

class TestFormatBool:
    """Test format! with {bool} specifier."""

    def test_format_bool_true(self, e2e):
        """format! with {bool} and true produces "1"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "{{bool}}", true);
            }}
        ''', ExpectedState(memory={
            0x7E2000: ascii_bytes_null("1")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_bool_false(self, e2e):
        """format! with {bool} and false produces "0"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "{{bool}}", false);
            }}
        ''', ExpectedState(memory={
            0x7E2000: ascii_bytes_null("0")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Signed Integer Format Tests
# ============================================================================

class TestFormatSigned:
    """Test format! with signed integer specifiers."""

    def test_format_i8_positive(self, e2e):
        """format! with {i8} and positive value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            #[entry]
            fn main() {{
                let v: i8 = 42;
                format!(BUF, "{{i8}}", v);
            }}
        ''', ExpectedState(memory={
            # "42\0"
            0x7E2000: ascii_bytes_null("42")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_i8_negative(self, e2e):
        """format! with {i8} and negative value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            #[entry]
            fn main() {{
                let v: i8 = 0 - 1;
                format!(BUF, "{{i8}}", v);
            }}
        ''', ExpectedState(memory={
            # "-1\0"
            0x7E2000: ascii_bytes_null("-1")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_i16_positive(self, e2e):
        """format! with {i16} and positive value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                let v: i16 = 1000;
                format!(BUF, "{{i16}}", v);
            }}
        ''', ExpectedState(memory={
            # "1000\0"
            0x7E2000: ascii_bytes_null("1000")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_i16_negative(self, e2e):
        """format! with {i16} and negative value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                let v: i16 = 0 - 42;
                format!(BUF, "{{i16}}", v);
            }}
        ''', ExpectedState(memory={
            # "-42\0"
            0x7E2000: ascii_bytes_null("-42")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Small Literal Inlining Tests
# ============================================================================

class TestFormatLiteralInlining:
    """Test that small literals are inlined (1-3 bytes) and large use memcpy."""

    def test_format_small_literal(self, e2e):
        """format! with 2-byte literal produces correct output via inlining."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "X={{u8}}", 42);
            }}
        ''', ExpectedState(memory={
            # "X=42\0"
            0x7E2000: ascii_bytes_null("X=42") + [0xFF]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_single_byte_literal(self, e2e):
        """format! with 1-byte literal produces correct output."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                format!(BUF, "Z");
            }}
        ''', ExpectedState(memory={
            0x7E2000: ascii_bytes_null("Z")
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Multiple format! Calls in Same Scope
# ============================================================================

class TestFormatMultipleCalls:
    """Test that multiple format! calls in the same scope compile and run correctly."""

    def test_format_two_calls_same_buffer(self, e2e):
        """Two format! calls to the same buffer; second overwrites first."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                format!(BUF, "FIRST");
                format!(BUF, "ABCDE");
            }}
        ''', ExpectedState(memory={
            0x7E2000: ascii_bytes_null("ABCDE")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_two_calls_different_buffers(self, e2e):
        """Two format! calls to different buffers in same scope."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF_A: [u8; 32] = [0xFF; 32];
            #[ram(0x7E2020)]
            static mut BUF_B: [u8; 32] = [0xFF; 32];

            #[entry]
            fn main() {{
                format!(BUF_A, "X:{{u16:04d}}", 123);
                format!(BUF_B, "Y:{{u16:04d}}", 456);
            }}
        ''', ExpectedState(memory={
            0x7E2000: list(b"X:0123"),
            0x7E2020: list(b"Y:0456"),
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_format_three_calls_with_specifiers(self, e2e):
        """Three format! calls with different specifiers in same scope."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram(0x7E2000)]
            static mut BUF: [u8; 32] = [0xFF; 32];

            #[zeropage(0x10)]
            static mut R1: [u8; 4];
            #[zeropage(0x14)]
            static mut R2: [u8; 3];
            #[zeropage(0x17)]
            static mut R3: [u8; 3];

            #[entry]
            fn main() {{
                format!(BUF, "{{u8:x}}", 255);
                R1[0] = BUF[0];
                R1[1] = BUF[1];
                R1[2] = BUF[2];

                format!(BUF, "{{u8}}", 42);
                R2[0] = BUF[0];
                R2[1] = BUF[1];
                R2[2] = BUF[2];

                format!(BUF, "{{bool}}", true);
                R3[0] = BUF[0];
                R3[1] = BUF[1];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes_null("FF") + [0x00] + ascii_bytes_null("42") + ascii_bytes_null("1"),
        }))
        assert result.success, f"Failures: {result.failures}"
