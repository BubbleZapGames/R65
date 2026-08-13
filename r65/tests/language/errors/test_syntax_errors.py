# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
            # For loops are now supported
            "fn test() { let f = |x| x + 1; }",  # Closures
            "trait Foo { fn bar(); }",  # Traits
            # impl blocks are now supported
            "fn test<T>(x: T) { }",  # Generics
            "fn test<'a>(x: &'a u8) { }",  # Lifetimes
        ]
        for source in cases:
            with pytest.raises(ParseError):
                parse(source)


class TestTupleStructs:
    """Tuple structs (newtypes) are unsupported and must say so by name."""

    def test_newtype_declaration_names_the_feature(self):
        with pytest.raises(ParseError, match="tuple structs \\(newtypes\\) are not supported"):
            parse("struct TileId(u8);")

    def test_hint_suggests_a_named_field_of_the_same_type(self):
        with pytest.raises(ParseError) as exc:
            parse("struct TileId(u8);")
        assert "struct TileId { value: u8 }" in exc.value.hint

    def test_multi_field_tuple_struct(self):
        with pytest.raises(ParseError) as exc:
            parse("struct Point(u8, u16);")
        assert "field0: u8, field1: u16" in exc.value.hint

    def test_hint_renders_compound_types(self):
        """The suggested replacement must itself be valid R65."""
        for source, expected in [
            ("struct Buf([u8; 4]);", "value: [u8; 4]"),
            ("struct Ref(far *u8);", "value: far *u8"),
            ("struct Wrap(Other);", "value: Other"),
        ]:
            with pytest.raises(ParseError) as exc:
                parse(source)
            assert expected in exc.value.hint
        # Each suggestion parses as written
        for decl in ["struct Buf { value: [u8; 4] }",
                     "struct Ref { value: far *u8 }",
                     "struct Wrap { value: Other }"]:
            parse_succeeds(decl)

    def test_empty_tuple_struct(self):
        with pytest.raises(ParseError, match="tuple structs"):
            parse("struct Empty();")

    def test_doc_commented_tuple_struct(self):
        with pytest.raises(ParseError, match="tuple structs"):
            parse("/// A tile index\nstruct TileId(u8);")

    def test_tuple_field_access(self):
        with pytest.raises(ParseError, match="tuple field access '.0' is not supported"):
            parse("fn f() { let a: u8 = x.0; }")

    def test_named_field_access_still_works(self):
        parse_succeeds("fn f() { let a: u8 = x.value; }")


class TestValidEdgeCases:
    """Tests for valid edge cases that should NOT error."""

    def test_valid_edge_cases(self):
        """Test valid edge cases parse successfully."""
        cases = [
            "fn empty() { }",
            "struct Empty { }",
            "fn test() { { { { A = 1; } } } }",  # Nested blocks
            "fn test() { return A, X, Y; }",  # Multiple returns (tuple syntax)
        ]
        for source in cases:
            prog = parse_succeeds(source)
            assert len(prog.items) >= 1
