"""Tests for parse errors: missing tokens, braces, semicolons."""

import pytest
from r65.compiler.frontend.parser import parse, ParseError


class TestMissingSemicolons:
    """Tests for missing semicolon errors."""

    def test_missing_semicolons(self):
        """Test statements without semicolons fail.

        Note: 'fn test() { A = 10 }' is now valid — it's a trailing
        expression (Rust-style implicit return). Only true statements
        (let, return, break, continue) still require semicolons.
        """
        cases = [
            "fn test() { let x = 10 }",
            "fn test() { return A }",
            "fn test() { loop { break } }",
            "fn test() { loop { continue } }",
            "#[ram] static mut X: u8",
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestMissingBraces:
    """Tests for missing brace errors."""

    def test_missing_braces(self):
        """Test blocks without braces fail."""
        cases = [
            "fn test() let x = 10; }",  # Missing open brace
            "fn test() { let x = 10;",   # Missing close brace
            "fn test() { if A == 0 A = 1; } }",  # If without braces
            "fn test() { loop A++; }",  # Loop without braces
            "fn test() { while A != 0 A--; }",  # While without braces
            "struct Point x: u8, y: u8",  # Struct without braces
            "enum Dir North, South",  # Enum without braces
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestMissingParentheses:
    """Tests for missing parenthesis errors."""

    def test_missing_parens(self):
        """Test missing parentheses fail."""
        cases = [
            "fn test) { }",  # Missing open paren
            "fn test( { }",  # Missing close paren
            "#[mode m8] fn test() { }",  # Attribute without parens
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestMissingBrackets:
    """Tests for missing bracket errors."""

    def test_missing_brackets(self):
        """Test missing brackets fail."""
        cases = [
            "#[ram] static mut X: [u8; 10;",  # Missing close bracket
            # NOTE: "#[ram] static mut X: [u8];" is now valid as a slice type
            "fn test() { let x: u8 = ARR[0; }",  # Index without close bracket
            "#[ram] static mut X: [u8; 3] = [1, 2, 3;",  # Literal without close
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestInvalidLiterals:
    """Tests for invalid literal errors."""

    def test_invalid_literals(self):
        """Test invalid literals fail."""
        cases = [
            "fn test() { let x: u8 = 0xGG; }",  # Invalid hex
            "fn test() { let x: u8 = 0b123; }",  # Invalid binary
            "fn test() { let x: u8 = 0x; }",  # Hex no digits
            "fn test() { let x: u8 = 0b; }",  # Binary no digits
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)
