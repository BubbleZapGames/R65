"""
Tests for offset_of() built-in function.

Tests compile-time evaluation of struct field byte offsets.
"""
import pytest

from r65.compiler.frontend import Parser
from r65.compiler.hir import (
    HIRBuilder, HIRError, HIRProgram,
    HIRIntegerLiteral, HIRConstDecl,
)


def build_hir(source: str) -> HIRProgram:
    """Helper to parse and build HIR from source."""
    parser = Parser()
    ast = parser.parse(source)
    builder = HIRBuilder()
    return builder.build_program(ast)


class TestOffsetOf:
    """Test offset_of() built-in function."""

    def test_first_field_offset_is_zero(self):
        """First field should have offset 0."""
        source = """
        struct Player { x: u8, y: u8, health: u16 }
        fn foo() {
            A = offset_of(Player, x);
        }
        """
        hir = build_hir(source)
        func = hir.declarations[1]  # skip struct decl
        stmt = func.body.statements[0]
        # HIRExprStmt -> .expr (HIRAssignment) -> .value
        value = stmt.expr.value
        assert isinstance(value, HIRIntegerLiteral)
        assert value.value == 0

    def test_second_field_offset(self):
        """Second field offset after u8 should be 1."""
        source = """
        struct Player { x: u8, y: u8, health: u16 }
        fn foo() {
            A = offset_of(Player, y);
        }
        """
        hir = build_hir(source)
        func = hir.declarations[1]
        stmt = func.body.statements[0]
        value = stmt.expr.value
        assert isinstance(value, HIRIntegerLiteral)
        assert value.value == 1

    def test_third_field_offset_after_two_u8s(self):
        """Third field offset after two u8s should be 2."""
        source = """
        struct Player { x: u8, y: u8, health: u16 }
        fn foo() {
            A = offset_of(Player, health);
        }
        """
        hir = build_hir(source)
        func = hir.declarations[1]
        stmt = func.body.statements[0]
        value = stmt.expr.value
        assert isinstance(value, HIRIntegerLiteral)
        assert value.value == 2

    def test_offset_with_u16_field(self):
        """Field after u16 should account for 2-byte size."""
        source = """
        struct Data { a: u16, b: u8, c: u16 }
        fn foo() {
            A = offset_of(Data, b);
        }
        """
        hir = build_hir(source)
        func = hir.declarations[1]
        stmt = func.body.statements[0]
        value = stmt.expr.value
        assert isinstance(value, HIRIntegerLiteral)
        assert value.value == 2  # after u16

    def test_offset_after_u16_then_u8(self):
        """Field after u16 + u8 should be at offset 3."""
        source = """
        struct Data { a: u16, b: u8, c: u16 }
        fn foo() {
            A = offset_of(Data, c);
        }
        """
        hir = build_hir(source)
        func = hir.declarations[1]
        stmt = func.body.statements[0]
        value = stmt.expr.value
        assert isinstance(value, HIRIntegerLiteral)
        assert value.value == 3  # u16(2) + u8(1)

    def test_const_evaluation(self):
        """offset_of should work in const declarations."""
        source = """
        struct Player { x: u8, y: u8, health: u16 }
        const OFF: u8 = offset_of(Player, health);
        """
        hir = build_hir(source)
        const_decl = hir.declarations[1]
        assert isinstance(const_decl, HIRConstDecl)
        assert const_decl.name == "OFF"
        assert isinstance(const_decl.value, HIRIntegerLiteral)
        assert const_decl.value.value == 2

    def test_error_wrong_arg_count(self):
        """offset_of with wrong number of args should error."""
        source = """
        struct Player { x: u8 }
        fn foo() {
            A = offset_of(Player);
        }
        """
        with pytest.raises(HIRError, match="offset_of.*2 argument"):
            build_hir(source)

    def test_error_nonexistent_struct(self):
        """offset_of with unknown struct should error."""
        source = """
        fn foo() {
            A = offset_of(NoSuchStruct, x);
        }
        """
        with pytest.raises(HIRError, match="Undefined"):
            build_hir(source)

    def test_error_nonexistent_field(self):
        """offset_of with unknown field should error."""
        source = """
        struct Player { x: u8, y: u8 }
        fn foo() {
            A = offset_of(Player, z);
        }
        """
        with pytest.raises(HIRError, match="has no field 'z'"):
            build_hir(source)
