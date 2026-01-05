"""
Tests for string literal initialization of byte arrays.
"""
import pytest

from r65.compiler.frontend import Parser
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.mir import MIRBuilder


def compile_to_mir(source: str):
    """Helper to compile source through type checking to MIR."""
    parser = Parser()
    ast = parser.parse(source)
    hir_builder = HIRBuilder()
    hir = hir_builder.build_program(ast)
    type_checker = TypeChecker(hir)
    type_checker.check()
    mir_builder = MIRBuilder()
    mir = mir_builder.build_program(hir)
    return hir, mir


def type_check(source: str):
    """Helper to compile source through type checking."""
    parser = Parser()
    ast = parser.parse(source)
    hir_builder = HIRBuilder()
    hir = hir_builder.build_program(ast)
    type_checker = TypeChecker(hir)
    type_checker.check()
    return hir


class TestStringLiteralBasic:
    """Basic string literal functionality."""

    def test_simple_string_literal(self):
        """Test basic string literal initialization."""
        source = '''
#[ram]
static mut MSG: [u8; 16] = "Hello";
'''
        hir, mir = compile_to_mir(source)

        init = hir.declarations[0].initializer
        assert init.processed_bytes == [72, 101, 108, 108, 111]  # "Hello"

        # Check ROM data is zero-padded to 16 bytes
        assert len(mir.rom_data_sections) == 1
        assert len(mir.rom_data_sections[0].data) == 16
        assert mir.rom_data_sections[0].data[:5] == [72, 101, 108, 108, 111]
        assert mir.rom_data_sections[0].data[5:] == [0] * 11

    def test_immutable_string_literal(self):
        """Test string literal works with non-mutable static."""
        source = '''
#[rom]
static MSG: [u8; 8] = "Test";
'''
        hir, mir = compile_to_mir(source)

        assert hir.declarations[0].is_mutable == False
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [84, 101, 115, 116]  # "Test"

    def test_exact_size_string(self):
        """Test string that exactly matches array size."""
        source = '''
#[ram]
static mut MSG: [u8; 5] = "Hello";
'''
        hir, mir = compile_to_mir(source)

        init = hir.declarations[0].initializer
        assert init.processed_bytes == [72, 101, 108, 108, 111]
        assert len(mir.rom_data_sections[0].data) == 5


class TestEscapeSequences:
    """Escape sequence handling."""

    def test_newline_escape(self):
        """Test \\n escape sequence."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "A\nB";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x41, 0x0A, 0x42]  # A, newline, B

    def test_tab_escape(self):
        """Test \\t escape sequence."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "A\tB";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x41, 0x09, 0x42]  # A, tab, B

    def test_carriage_return_escape(self):
        """Test \\r escape sequence."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "A\rB";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x41, 0x0D, 0x42]  # A, CR, B

    def test_null_escape(self):
        """Test \\0 escape sequence."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "A\0B";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x41, 0x00, 0x42]  # A, null, B

    def test_backslash_escape(self):
        """Test \\\\ escape sequence."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "A\\B";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x41, 0x5C, 0x42]  # A, backslash, B

    def test_hex_escape(self):
        """Test \\x## escape sequence."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "\x41\x42";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x41, 0x42]  # A, B

    def test_hex_escape_high_bytes(self):
        """Test \\x## with high byte values (Extended ASCII)."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "\xC0\xFF";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0xC0, 0xFF]

    def test_multiple_escapes(self):
        """Test multiple escape sequences in one string."""
        source = r'''
#[ram]
static mut MSG: [u8; 10] = "A\n\t\0\x42";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x41, 0x0A, 0x09, 0x00, 0x42]


class TestErrorCases:
    """Error detection tests."""

    def test_utf8_multibyte_rejected(self):
        """Test that UTF-8 multi-byte characters are rejected."""
        source = '''
#[ram]
static mut MSG: [u8; 8] = "Hello€";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "not valid Extended ASCII" in str(excinfo.value)
        assert "U+20AC" in str(excinfo.value)  # Euro sign code point

    def test_emoji_rejected(self):
        """Test that emoji (high Unicode) are rejected."""
        source = '''
#[ram]
static mut MSG: [u8; 8] = "Hi😀";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "not valid Extended ASCII" in str(excinfo.value)

    def test_chinese_character_rejected(self):
        """Test that Chinese characters are rejected."""
        source = '''
#[ram]
static mut MSG: [u8; 8] = "Hello中";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "not valid Extended ASCII" in str(excinfo.value)

    def test_string_too_long(self):
        """Test error when string exceeds array size."""
        source = '''
#[ram]
static mut MSG: [u8; 4] = "Hello, World!";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "larger than array size" in str(excinfo.value)

    def test_invalid_escape_sequence(self):
        """Test error on invalid escape sequence."""
        source = r'''
#[ram]
static mut MSG: [u8; 8] = "Hello\q";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "Unknown escape sequence" in str(excinfo.value)

    def test_string_in_non_array_type(self):
        """Test error when string assigned to non-array type."""
        source = '''
#[ram]
static mut VAL: u8 = "x";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "non-array type" in str(excinfo.value)

    def test_string_in_non_u8_array(self):
        """Test error when string assigned to non-u8 array."""
        source = '''
#[ram]
static mut MSG: [u16; 8] = "Hello";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "only initialize [u8; N]" in str(excinfo.value)

    def test_incomplete_hex_escape(self):
        """Test error on incomplete hex escape at end of string."""
        source = r'''
#[ram]
static mut MSG: [u8; 4] = "A\x4";
'''
        with pytest.raises(TypeCheckError) as excinfo:
            type_check(source)
        assert "Invalid hex escape" in str(excinfo.value)


class TestExtendedASCII:
    """Extended ASCII (0x80-0xFF) support."""

    def test_extended_ascii_via_hex_escape(self):
        """Test Extended ASCII bytes via hex escape."""
        source = r'''
#[ram]
static mut MSG: [u8; 8] = "\x80\x81\xFE\xFF";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == [0x80, 0x81, 0xFE, 0xFF]

    def test_full_byte_range(self):
        """Test full byte range 0x00-0xFF is supported."""
        # Build a string with all possible byte values via hex escapes
        hex_chars = ''.join(f'\\x{i:02X}' for i in range(256))
        source = f'''
#[ram]
static mut ALL_BYTES: [u8; 256] = "{hex_chars}";
'''
        hir = type_check(source)
        init = hir.declarations[0].initializer
        assert init.processed_bytes == list(range(256))
