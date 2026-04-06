# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for character literal parsing.

Character literals ('a', 'A', '\\n', '\\x7F', b'a') produce u8 integer literals.
Only 7-bit ASCII is accepted as raw characters; use \\xNN escapes for bytes 0x80-0xFF.
Unicode/UTF-8 characters are rejected.
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend import ast
from r65.compiler.errors import ParseError


def parse_char_value(source_expr: str) -> int:
    """Parse `let c: u8 = <source_expr>;` and return the resulting integer value."""
    source = f'fn test() {{ let c: u8 = {source_expr}; }}'
    program = parse(source, '<test>')
    func = program.items[0]
    let_stmt = func.body.statements[0]
    init = let_stmt.initializer
    assert isinstance(init, ast.IntegerLiteral)
    assert init.suffix == 'u8'
    return init.value


class TestBasicChars:
    """Basic ASCII character literals."""

    def test_lowercase_letter(self):
        assert parse_char_value("'a'") == 97

    def test_uppercase_letter(self):
        assert parse_char_value("'Z'") == 90

    def test_digit(self):
        assert parse_char_value("'0'") == 48

    def test_space(self):
        assert parse_char_value("' '") == 32

    def test_punctuation(self):
        assert parse_char_value("'!'") == 33

    def test_del_0x7F(self):
        assert parse_char_value("'\\x7F'") == 127


class TestEscapeSequences:
    """Escape sequences in character literals."""

    def test_newline(self):
        assert parse_char_value("'\\n'") == 10

    def test_tab(self):
        assert parse_char_value("'\\t'") == 9

    def test_carriage_return(self):
        assert parse_char_value("'\\r'") == 13

    def test_null(self):
        assert parse_char_value("'\\0'") == 0

    def test_backslash(self):
        assert parse_char_value("'\\\\'") == 92

    def test_single_quote(self):
        assert parse_char_value("'\\''") == 39

    def test_double_quote(self):
        assert parse_char_value("'\\\"'") == 34


class TestHexEscapes:
    """\\xNN hex escape sequences."""

    def test_hex_uppercase_a(self):
        assert parse_char_value("'\\x41'") == 0x41

    def test_hex_null(self):
        assert parse_char_value("'\\x00'") == 0x00

    def test_hex_high_bit(self):
        """0x80-0xFF range via hex escape."""
        assert parse_char_value("'\\x80'") == 0x80

    def test_hex_max(self):
        assert parse_char_value("'\\xFF'") == 0xFF

    def test_hex_lowercase(self):
        assert parse_char_value("'\\xab'") == 0xAB


class TestByteLiteralPrefix:
    """b'...' byte literal prefix is accepted as an alias."""

    def test_byte_literal_basic(self):
        assert parse_char_value("b'a'") == 97

    def test_byte_literal_escape(self):
        assert parse_char_value("b'\\n'") == 10

    def test_byte_literal_hex(self):
        assert parse_char_value("b'\\xFF'") == 0xFF


class TestUnicodeRejected:
    """Non-ASCII characters are rejected."""

    def test_accented_char_rejected(self):
        with pytest.raises(ParseError, match="not 7-bit ASCII"):
            parse_char_value("'é'")

    def test_emoji_rejected(self):
        with pytest.raises(ParseError, match="not 7-bit ASCII"):
            parse_char_value("'€'")

    def test_cjk_rejected(self):
        with pytest.raises(ParseError, match="not 7-bit ASCII"):
            parse_char_value("'日'")


class TestInvalidSyntax:
    """Invalid character literal forms."""

    def test_empty_char_rejected(self):
        with pytest.raises(ParseError):
            parse_char_value("''")

    def test_multi_char_rejected(self):
        with pytest.raises(ParseError):
            parse_char_value("'ab'")

    def test_unknown_escape_rejected(self):
        with pytest.raises(ParseError):
            parse_char_value("'\\q'")


class TestTypeCompatibility:
    """Character literals integrate with the type system as u8."""

    def test_assign_to_u8(self):
        """let c: u8 = 'a' works."""
        from r65.compiler.main import compile_string
        asm = compile_string('''
            #[zeropage(0x10)]
            static mut R: u8;
            #[entry]
            fn main() {
                let c: u8 = 'A';
                R = c;
            }
        ''')
        # Verify 'A' compiled to 0x41
        assert 'LDA #$41' in asm

    def test_char_in_comparison(self):
        """Character literal usable in comparison."""
        from r65.compiler.main import compile_string
        compile_string('''
            fn test(ch @ A: u8) -> bool {
                if ch == 'a' { return true; }
                return false;
            }
        ''')
