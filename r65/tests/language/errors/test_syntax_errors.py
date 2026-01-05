"""Tests for syntax errors: invalid declarations, expressions, operators."""

import pytest
from r65.compiler.frontend.parser import parse, ParseError
from r65.tests.language.common import parse_succeeds


class TestInvalidDeclarations:
    """Tests for invalid declaration syntax."""

    def test_invalid_declarations(self):
        """Test various invalid declarations fail."""
        cases = [
            "fn () { }",  # Function no name
            "fn test();",  # Function no body
            "#[ram] static mut X;",  # Static no type
            "const X: u8;",  # Const no value
            "const X = 10;",  # Const no type
            "fn test(x) { }",  # Parameter no type
            "struct Point { x, y }",  # Field no type
            "enum Empty { }",  # Empty enum
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestInvalidOperators:
    """Tests for invalid operator usage."""

    def test_invalid_operators(self):
        """Test invalid operator usage fails."""
        cases = [
            "fn test() { let x: u8 = 1 ++ 2; }",  # Double plus
            "fn test() { let x: u8 = 1 -- 2; }",  # Double minus
            "fn test() { let x: u8 = + 1; }",  # Missing left operand
            "fn test() { let x: u8 = 1 +; }",  # Missing right operand
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestInvalidStatements:
    """Tests for invalid statement syntax."""

    def test_invalid_statements(self):
        """Test invalid statements fail."""
        cases = [
            "fn test() { let x; }",  # Let no init or type
            "fn test() { if { A = 1; } }",  # If no condition
            "fn test() { while { A--; } }",  # While no condition
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestInvalidExpressions:
    """Tests for invalid expression syntax."""

    def test_invalid_expressions(self):
        """Test invalid expressions fail."""
        cases = [
            "fn test() { let x: u8 = (); }",  # Empty parens
            "fn test() { let x: u8 = (1 + 2; }",  # Unmatched open
            "fn test() { let x: u8 = 1 + 2); }",  # Unmatched close
            "fn test() { let x: u8 = A == 0 ? 1 : 0; }",  # Ternary not supported
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestUnsupportedSyntax:
    """Tests for unsupported Rust-like syntax."""

    def test_unsupported_features(self):
        """Test unsupported Rust features fail."""
        cases = [
            "fn test() { for i in 0..10 { } }",  # For loops
            "fn test() { let f = |x| x + 1; }",  # Closures
            "trait Foo { fn bar(); }",  # Traits
            "impl Foo { fn bar() { } }",  # Impl blocks
            "fn test<T>(x: T) { }",  # Generics
            "fn test<'a>(x: &'a u8) { }",  # Lifetimes
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestValidEdgeCases:
    """Tests for valid edge cases that should NOT error."""

    def test_valid_edge_cases(self):
        """Test valid edge cases parse successfully."""
        cases = [
            "fn empty() { }",
            "struct Empty { }",
            "fn test() { { { { A = 1; } } } }",  # Nested blocks
            "fn test() { return A, X, Y; }",  # Multiple returns
        ]
        for source in cases:
            prog = parse_succeeds(source)
            assert len(prog.items) >= 1
