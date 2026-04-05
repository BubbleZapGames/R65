# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for stdlib/string.r65.

Tests memory primitives, string operations, character classification,
and number formatting functions.
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

# Path to stdlib
STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
STRING_PATH = STDLIB_DIR / "string.r65"


def buf_addr(offset=0):
    """SNES address for BUF (auto-allocated at 0x7E2000)."""
    return 0x7E2000 + offset


def result_addr(offset=0):
    """SNES address for RESULT zeropage var at 0x10."""
    return 0x7E0010 + offset


def ascii_bytes(s):
    """Convert a Python string to list of ASCII byte values."""
    return [ord(c) for c in s]


def ascii_bytes_null(s):
    """Convert a Python string to list of ASCII byte values with null terminator."""
    return [ord(c) for c in s] + [0]


# ============================================================================
# Memory Primitives
# ============================================================================

class TestStrlen:
    """Test strlen function."""

    def test_strlen_empty(self, e2e):
        """strlen of empty string returns 0."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut STR: [u8; 4] = "\\0";

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                let mut len: u16 = strlen(&STR as far *u8);
                RESULT[0] = len as u8;
                RESULT[1] = (len >> 8) as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x00, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strlen_short(self, e2e):
        """strlen of "Hello" returns 5."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut STR: [u8; 8] = "Hello\\0";

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                let mut len: u16 = strlen(&STR as far *u8);
                RESULT[0] = len as u8;
                RESULT[1] = (len >> 8) as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x05, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strlen_single_char(self, e2e):
        """strlen of single character string returns 1."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut STR: [u8; 4] = "X\\0";

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                let mut len: u16 = strlen(&STR as far *u8);
                RESULT[0] = len as u8;
                RESULT[1] = (len >> 8) as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x01, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMemcpy:
    """Test memcpy function."""

    def test_memcpy_basic(self, e2e):
        """memcpy copies bytes from src to dst."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut SRC: [u8; 8] = "ABCD\\0";
            #[ram]
            static mut DST: [u8; 8] = [0; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                memcpy(&DST as far *u8, &SRC as far *u8, 4);
                RESULT[0] = DST[0];
                RESULT[1] = DST[1];
                RESULT[2] = DST[2];
                RESULT[3] = DST[3];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("ABCD")
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_memcpy_zero_length(self, e2e):
        """memcpy with n=0 does nothing."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut SRC: [u8; 4] = "AB\\0";
            #[ram]
            static mut DST: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                memcpy(&DST as far *u8, &SRC as far *u8, 0);
                RESULT = DST[0];
            }}
        ''', ExpectedState(memory={
            result_addr(): 0xFF
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMemset:
    """Test memset function."""

    def test_memset_fill(self, e2e):
        """memset fills buffer with value."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                memset(&BUF as far *u8, 0xAA, 4);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
            }}
        ''', ExpectedState(memory={
            result_addr(): [0xAA, 0xAA, 0xAA, 0xAA]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_memset_zero_length(self, e2e):
        """memset with n=0 does nothing."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                memset(&BUF as far *u8, 0x00, 0);
                RESULT = BUF[0];
            }}
        ''', ExpectedState(memory={
            result_addr(): 0xFF
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMemcmp:
    """Test memcmp function."""

    def test_memcmp_equal(self, e2e):
        """memcmp returns 0 for equal regions."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_BUF: [u8; 4] = "ABC\\0";
            #[ram]
            static mut B_BUF: [u8; 4] = "ABC\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = memcmp(&A_BUF as far *u8, &B_BUF as far *u8, 3);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x00
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_memcmp_less(self, e2e):
        """memcmp returns 0xFF when a < b."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_BUF: [u8; 4] = "ABC\\0";
            #[ram]
            static mut B_BUF: [u8; 4] = "ABD\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = memcmp(&A_BUF as far *u8, &B_BUF as far *u8, 3);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0xFF
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_memcmp_greater(self, e2e):
        """memcmp returns 1 when a > b."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_BUF: [u8; 4] = "ABD\\0";
            #[ram]
            static mut B_BUF: [u8; 4] = "ABC\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = memcmp(&A_BUF as far *u8, &B_BUF as far *u8, 3);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x01
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_memcmp_zero_length(self, e2e):
        """memcmp with n=0 returns 0."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_BUF: [u8; 4] = "ZZZ\\0";
            #[ram]
            static mut B_BUF: [u8; 4] = "AAA\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = memcmp(&A_BUF as far *u8, &B_BUF as far *u8, 0);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x00
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# String Operations
# ============================================================================

class TestStrcpy:
    """Test strcpy function."""

    def test_strcpy_basic(self, e2e):
        """strcpy copies string including null, returns length."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut SRC: [u8; 8] = "Hello\\0";
            #[ram]
            static mut DST: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                let mut len: u16 = strcpy(&DST as far *u8, &SRC as far *u8);
                RESULT[0] = DST[0];
                RESULT[1] = DST[1];
                RESULT[2] = DST[2];
                RESULT[3] = DST[3];
                RESULT[4] = DST[4];
                RESULT[5] = DST[5];
                RESULT[6] = len as u8;
                RESULT[7] = (len >> 8) as u8;
            }}
        ''', ExpectedState(memory={
            # "Hello\0" copied, len=5
            result_addr(): ascii_bytes("Hello") + [0x00, 0x05, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strcpy_empty(self, e2e):
        """strcpy of empty string copies just null, returns 0."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut SRC: [u8; 4] = "\\0";
            #[ram]
            static mut DST: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            #[entry]
            fn main() {{
                let mut len: u16 = strcpy(&DST as far *u8, &SRC as far *u8);
                RESULT[0] = DST[0];
                RESULT[1] = DST[1];
                RESULT[2] = len as u8;
            }}
        ''', ExpectedState(memory={
            # null copied, rest untouched, len=0
            result_addr(): [0x00, 0xFF, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestStrncpy:
    """Test strncpy function."""

    def test_strncpy_truncates(self, e2e):
        """strncpy truncates when src is longer than max-1."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut SRC: [u8; 8] = "Hello\\0";
            #[ram]
            static mut DST: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                let mut len: u16 = strncpy(&DST as far *u8, &SRC as far *u8, 4);
                RESULT[0] = DST[0];
                RESULT[1] = DST[1];
                RESULT[2] = DST[2];
                RESULT[3] = DST[3];
                RESULT[4] = len as u8;
            }}
        ''', ExpectedState(memory={
            # "Hel\0" (max=4 → copies 3 chars + null), len=3
            result_addr(): [0x48, 0x65, 0x6C, 0x00, 0x03]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strncpy_short_src(self, e2e):
        """strncpy copies full src when shorter than max."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut SRC: [u8; 4] = "Hi\\0";
            #[ram]
            static mut DST: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                let mut len: u16 = strncpy(&DST as far *u8, &SRC as far *u8, 8);
                RESULT[0] = DST[0];
                RESULT[1] = DST[1];
                RESULT[2] = DST[2];
                RESULT[3] = len as u8;
            }}
        ''', ExpectedState(memory={
            # "Hi\0", len=2
            result_addr(): [0x48, 0x69, 0x00, 0x02]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strncpy_zero_max(self, e2e):
        """strncpy with max=0 returns 0 and writes nothing."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut SRC: [u8; 4] = "Hi\\0";
            #[ram]
            static mut DST: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            #[entry]
            fn main() {{
                let mut len: u16 = strncpy(&DST as far *u8, &SRC as far *u8, 0);
                RESULT[0] = DST[0];
                RESULT[1] = len as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): [0xFF, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestStrcat:
    """Test strcat function."""

    def test_strcat_basic(self, e2e):
        """strcat appends src after dst's null."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut DST: [u8; 16] = "Hi\\0";
            #[ram]
            static mut SRC: [u8; 8] = "Lo\\0";

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                let mut len: u16 = strcat(&DST as far *u8, &SRC as far *u8);
                RESULT[0] = DST[0];
                RESULT[1] = DST[1];
                RESULT[2] = DST[2];
                RESULT[3] = DST[3];
                RESULT[4] = DST[4];
                RESULT[5] = len as u8;
            }}
        ''', ExpectedState(memory={
            # "HiLo\0", len=4
            result_addr(): ascii_bytes("HiLo") + [0x00, 0x04]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strcat_to_empty(self, e2e):
        """strcat onto empty dst copies src."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut DST: [u8; 16] = "\\0";
            #[ram]
            static mut SRC: [u8; 8] = "AB\\0";

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                let mut len: u16 = strcat(&DST as far *u8, &SRC as far *u8);
                RESULT[0] = DST[0];
                RESULT[1] = DST[1];
                RESULT[2] = DST[2];
                RESULT[3] = len as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x41, 0x42, 0x00, 0x02]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestStrcmp:
    """Test strcmp function."""

    def test_strcmp_equal(self, e2e):
        """strcmp returns 0 for equal strings."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_STR: [u8; 4] = "AB\\0";
            #[ram]
            static mut B_STR: [u8; 4] = "AB\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = strcmp(&A_STR as far *u8, &B_STR as far *u8);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x00
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strcmp_less(self, e2e):
        """strcmp returns 0xFF when a < b."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_STR: [u8; 4] = "AA\\0";
            #[ram]
            static mut B_STR: [u8; 4] = "AB\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = strcmp(&A_STR as far *u8, &B_STR as far *u8);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0xFF
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strcmp_greater(self, e2e):
        """strcmp returns 1 when a > b."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_STR: [u8; 4] = "B\\0";
            #[ram]
            static mut B_STR: [u8; 4] = "A\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = strcmp(&A_STR as far *u8, &B_STR as far *u8);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x01
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strcmp_prefix(self, e2e):
        """strcmp: shorter string is less than longer with same prefix."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_STR: [u8; 4] = "A\\0";
            #[ram]
            static mut B_STR: [u8; 4] = "AB\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = strcmp(&A_STR as far *u8, &B_STR as far *u8);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            # "A\0" vs "AB\0": at index 1, 0x00 < 0x42 → a < b → 0xFF
            result_addr(): 0xFF
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strcmp_both_empty(self, e2e):
        """strcmp of two empty strings returns 0."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut A_STR: [u8; 4] = "\\0";
            #[ram]
            static mut B_STR: [u8; 4] = "\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                let r: i8 = strcmp(&A_STR as far *u8, &B_STR as far *u8);
                RESULT = r as u8;
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x00
        }))
        assert result.success, f"Failures: {result.failures}"


class TestStrchr:
    """Test strchr function."""

    def test_strchr_found(self, e2e):
        """strchr returns index of first match."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut STR: [u8; 8] = "Hello\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                // 'l' = 0x6C, first at index 2
                RESULT = strchr(&STR as far *u8, 0x6C);
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x02
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strchr_not_found(self, e2e):
        """strchr returns 0xFF when char not in string."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut STR: [u8; 8] = "Hello\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                // 'Z' = 0x5A, not in "Hello"
                RESULT = strchr(&STR as far *u8, 0x5A);
            }}
        ''', ExpectedState(memory={
            result_addr(): 0xFF
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_strchr_first_char(self, e2e):
        """strchr finds character at index 0."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut STR: [u8; 8] = "Hello\\0";

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {{
                // 'H' = 0x48, at index 0
                RESULT = strchr(&STR as far *u8, 0x48);
            }}
        ''', ExpectedState(memory={
            result_addr(): 0x00
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Character Classification and Conversion
# ============================================================================

class TestCharClassification:
    """Test is_digit, is_upper, is_lower, is_alpha."""

    def test_char_classification_batch(self, e2e):
        """Batch test: digit, upper, lower, alpha classification."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                // is_digit
                RESULT[0] = is_digit(0x35) as u8;   // '5' → true (1)
                RESULT[1] = is_digit(0x41) as u8;   // 'A' → false (0)

                // is_upper
                RESULT[2] = is_upper(0x41) as u8;   // 'A' → true
                RESULT[3] = is_upper(0x61) as u8;   // 'a' → false

                // is_lower
                RESULT[4] = is_lower(0x61) as u8;   // 'a' → true
                RESULT[5] = is_lower(0x41) as u8;   // 'A' → false

                // is_alpha
                RESULT[6] = is_alpha(0x5A) as u8;   // 'Z' → true
                RESULT[7] = is_alpha(0x30) as u8;   // '0' → false
            }}
        ''', ExpectedState(memory={
            result_addr(): [1, 0, 1, 0, 1, 0, 1, 0]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_char_boundaries(self, e2e):
        """Test classification at ASCII boundary values."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                // is_digit boundaries
                RESULT[0] = is_digit(0x30) as u8;   // '0' → true
                RESULT[1] = is_digit(0x39) as u8;   // '9' → true
                RESULT[2] = is_digit(0x2F) as u8;   // '/' → false (below '0')
                RESULT[3] = is_digit(0x3A) as u8;   // ':' → false (above '9')

                // is_upper boundaries
                RESULT[4] = is_upper(0x41) as u8;   // 'A' → true
                RESULT[5] = is_upper(0x5A) as u8;   // 'Z' → true
                RESULT[6] = is_upper(0x40) as u8;   // '@' → false
                RESULT[7] = is_upper(0x5B) as u8;   // '[' → false
            }}
        ''', ExpectedState(memory={
            result_addr(): [1, 1, 0, 0, 1, 1, 0, 0]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestCharConversion:
    """Test to_upper and to_lower."""

    def test_to_upper(self, e2e):
        """to_upper converts lowercase, leaves others unchanged."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                RESULT[0] = to_upper(0x61);   // 'a' → 'A' (0x41)
                RESULT[1] = to_upper(0x7A);   // 'z' → 'Z' (0x5A)
                RESULT[2] = to_upper(0x41);   // 'A' → 'A' (unchanged)
                RESULT[3] = to_upper(0x35);   // '5' → '5' (unchanged)
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x41, 0x5A, 0x41, 0x35]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_to_lower(self, e2e):
        """to_lower converts uppercase, leaves others unchanged."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                RESULT[0] = to_lower(0x41);   // 'A' → 'a' (0x61)
                RESULT[1] = to_lower(0x5A);   // 'Z' → 'z' (0x7A)
                RESULT[2] = to_lower(0x61);   // 'a' → 'a' (unchanged)
                RESULT[3] = to_lower(0x35);   // '5' → '5' (unchanged)
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x61, 0x7A, 0x61, 0x35]
        }))
        assert result.success, f"Failures: {result.failures}"


# ============================================================================
# Number Formatting
# ============================================================================

class TestU8ToDec:
    """Test u8_to_dec function."""

    def test_u8_to_dec_batch(self, e2e):
        """Batch: u8_to_dec for 0, single digit, two digit, three digit."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF0: [u8; 4] = [0xFF; 4];
            #[ram]
            static mut BUF1: [u8; 4] = [0xFF; 4];
            #[ram]
            static mut BUF2: [u8; 4] = [0xFF; 4];
            #[ram]
            static mut BUF3: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 8];

            #[entry]
            fn main() {{
                // 0 → "0"
                let l0: u8 = u8_to_dec(&BUF0 as far *u8, 0);
                // 7 → "7"
                let l1: u8 = u8_to_dec(&BUF1 as far *u8, 7);
                // 42 → "42"
                let l2: u8 = u8_to_dec(&BUF2 as far *u8, 42);
                // 255 → "255"
                let l3: u8 = u8_to_dec(&BUF3 as far *u8, 255);

                RESULT[0] = BUF0[0];  // '0'
                RESULT[1] = l0;       // 1
                RESULT[2] = BUF1[0];  // '7'
                RESULT[3] = l1;       // 1
                RESULT[4] = BUF2[0];  // '4'
                RESULT[5] = BUF2[1];  // '2'
                RESULT[6] = l2;       // 2
                RESULT[7] = l3;       // 3
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x30, 1, 0x37, 1, 0x34, 0x32, 2, 3]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_to_dec_255(self, e2e):
        """u8_to_dec(255) produces "255\\0"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                u8_to_dec(&BUF as far *u8, 255);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
            }}
        ''', ExpectedState(memory={
            # "255\0"
            result_addr(): [0x32, 0x35, 0x35, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_to_dec_100(self, e2e):
        """u8_to_dec(100) produces "100\\0"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                u8_to_dec(&BUF as far *u8, 100);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
            }}
        ''', ExpectedState(memory={
            # "100\0"
            result_addr(): [0x31, 0x30, 0x30, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestU16ToDec:
    """Test u16_to_dec function."""

    def test_u16_to_dec_zero(self, e2e):
        """u16_to_dec(0) produces "0"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            #[entry]
            fn main() {{
                let mut len: u8 = u16_to_dec(&BUF as far *u8, 0);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = len;
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x30, 0x00, 1]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_to_dec_12345(self, e2e):
        """u16_to_dec(12345) produces "12345"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                let mut len: u8 = u16_to_dec(&BUF as far *u8, 12345);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = len;
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("12345") + [5]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_to_dec_65535(self, e2e):
        """u16_to_dec(65535) produces "65535"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                let mut len: u8 = u16_to_dec(&BUF as far *u8, 65535);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = len;
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("65535") + [5]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_to_dec_1000(self, e2e):
        """u16_to_dec(1000) produces "1000"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                let mut len: u8 = u16_to_dec(&BUF as far *u8, 1000);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = len;
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("1000") + [4]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestU16ToDecPad:
    """Test u16_to_dec_pad function."""

    def test_pad_short_number(self, e2e):
        """u16_to_dec_pad(42, 5) produces "   42"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 6];

            #[entry]
            fn main() {{
                let mut len: u8 = u16_to_dec_pad(&BUF as far *u8, 42, 5, 0x20);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
                RESULT[5] = len;
            }}
        ''', ExpectedState(memory={
            # "   42" (3 spaces + "42"), len=5
            result_addr(): [0x20, 0x20, 0x20, 0x34, 0x32, 5]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_pad_exact_width(self, e2e):
        """u16_to_dec_pad where number fills width exactly."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 4];

            #[entry]
            fn main() {{
                let mut len: u8 = u16_to_dec_pad(&BUF as far *u8, 123, 3, 0x20);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = len;
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("123") + [3]
        }))
        assert result.success, f"Failures: {result.failures}"


class TestU8ToHex:
    """Test u8_to_hex function."""

    def test_u8_to_hex_batch(self, e2e):
        """Batch: u8_to_hex for 0x00, 0xAB, 0xFF."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF0: [u8; 4] = [0xFF; 4];
            #[ram]
            static mut BUF1: [u8; 4] = [0xFF; 4];
            #[ram]
            static mut BUF2: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 9];

            #[entry]
            fn main() {{
                u8_to_hex(&BUF0 as far *u8, 0x00);
                u8_to_hex(&BUF1 as far *u8, 0xAB);
                u8_to_hex(&BUF2 as far *u8, 0xFF);

                RESULT[0] = BUF0[0];  // '0'
                RESULT[1] = BUF0[1];  // '0'
                RESULT[2] = BUF0[2];  // null
                RESULT[3] = BUF1[0];  // 'A'
                RESULT[4] = BUF1[1];  // 'B'
                RESULT[5] = BUF1[2];  // null
                RESULT[6] = BUF2[0];  // 'F'
                RESULT[7] = BUF2[1];  // 'F'
                RESULT[8] = BUF2[2];  // null
            }}
        ''', ExpectedState(memory={
            result_addr(): [
                0x30, 0x30, 0x00,  # "00\0"
                0x41, 0x42, 0x00,  # "AB\0"
                0x46, 0x46, 0x00,  # "FF\0"
            ]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_to_hex_mixed_nibbles(self, e2e):
        """u8_to_hex with mixed high/low nibbles (0x3C → "3C")."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 4] = [0xFF; 4];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            #[entry]
            fn main() {{
                u8_to_hex(&BUF as far *u8, 0x3C);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x33, 0x43, 0x00]  # "3C\0"
        }))
        assert result.success, f"Failures: {result.failures}"


class TestU16ToHex:
    """Test u16_to_hex function."""

    def test_u16_to_hex_1234(self, e2e):
        """u16_to_hex(0x1234) produces "1234"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                u16_to_hex(&BUF as far *u8, 0x1234);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("1234") + [0x00]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_to_hex_0000(self, e2e):
        """u16_to_hex(0x0000) produces "0000"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                u16_to_hex(&BUF as far *u8, 0x0000);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
            }}
        ''', ExpectedState(memory={
            result_addr(): [0x30, 0x30, 0x30, 0x30, 0x00]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_to_hex_FFFF(self, e2e):
        """u16_to_hex(0xFFFF) produces "FFFF"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                u16_to_hex(&BUF as far *u8, 0xFFFF);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("FFFF") + [0x00]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_u16_to_hex_DEAD(self, e2e):
        """u16_to_hex(0xDEAD) produces "DEAD"."""
        result = e2e.run(f'''
            include!("{STRING_PATH}")

            #[ram]
            static mut BUF: [u8; 8] = [0xFF; 8];

            #[zeropage(0x10)]
            static mut RESULT: [u8; 5];

            #[entry]
            fn main() {{
                u16_to_hex(&BUF as far *u8, 0xDEAD);
                RESULT[0] = BUF[0];
                RESULT[1] = BUF[1];
                RESULT[2] = BUF[2];
                RESULT[3] = BUF[3];
                RESULT[4] = BUF[4];
            }}
        ''', ExpectedState(memory={
            result_addr(): ascii_bytes("DEAD") + [0x00]
        }))
        assert result.success, f"Failures: {result.failures}"
