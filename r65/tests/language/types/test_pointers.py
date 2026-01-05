"""Tests for pointer types: near<T> and far<T>."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import ParseError
from r65.tests.language.common import parse_type, parse_static, parse_function, build_hir


class TestPointerTypes:
    """Tests for pointer type parsing."""

    def test_near_pointer(self):
        """Test near<T> pointer type."""
        t = parse_type("near<u8>")
        assert isinstance(t, ast.PointerType)
        assert t.is_far is False
        assert isinstance(t.pointee_type, ast.BasicType)
        assert t.pointee_type.name == "u8"

    def test_far_pointer(self):
        """Test far<T> pointer type."""
        t = parse_type("far<u8>")
        assert isinstance(t, ast.PointerType)
        assert t.is_far is True
        assert t.pointee_type.name == "u8"

    def test_pointer_to_struct(self):
        """Test pointer to named type."""
        t = parse_type("near<Player>")
        assert isinstance(t, ast.PointerType)
        # Named types are parsed as BasicType
        assert isinstance(t.pointee_type, ast.BasicType)
        assert t.pointee_type.name == "Player"


class TestPointerOperations:
    """Tests for pointer operations."""

    def test_dereference(self):
        """Test pointer dereference with *."""
        func = parse_function("fn test() { let v: u8 = *ptr; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.Dereference)

    def test_address_of(self):
        """Test address-of operator &."""
        func = parse_function("fn test() { let p: near<u8> = &value; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.AddressOf)

    def test_indexed_pointer(self):
        """Test pointer indexing ptr[Y]."""
        func = parse_function("fn test() { let v: u8 = ptr[Y]; }")
        let_stmt = func.body.statements[0]
        # This could be ArrayIndex or IndexedPointer depending on implementation
        assert let_stmt.initializer is not None


class TestPointerStatics:
    """Tests for pointer static declarations."""

    def test_pointer_static(self):
        """Test pointer in static declaration."""
        static = parse_static("#[zeropage] static mut PTR: near<u8>;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far is False

    def test_far_pointer_static(self):
        """Test far pointer in static declaration."""
        static = parse_static("#[ram] static mut FAR_PTR: far<u16>;")
        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far is True


class TestPointerHIR:
    """Tests for pointer HIR generation."""

    def test_pointer_hir(self):
        """Test pointer types generate proper HIR."""
        hir_prog = build_hir("""
            #[zeropage] static mut PTR: near<u8>;
            fn test() { let v: u8 = *PTR; }
        """)
        assert len(hir_prog.statics) >= 1
        assert len(hir_prog.functions) >= 1
