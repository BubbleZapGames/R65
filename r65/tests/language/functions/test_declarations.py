# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for function declarations."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import ParseError
from r65.tests.language.common import parse_function, parse_program, build_hir


class TestBasicFunctions:
    """Tests for basic function declaration."""

    def test_empty_function(self):
        """Test empty function body."""
        func = parse_function("fn empty() { }")
        assert func.name == "empty"
        assert len(func.params) == 0
        assert len(func.body.statements) == 0

    def test_function_with_body(self):
        """Test function with statements."""
        func = parse_function("fn work() { A = 1; X = 2; }")
        assert len(func.body.statements) == 2

    def test_function_return_type(self):
        """Test function with return type."""
        func = parse_function("fn get_value() -> u8 { return A; }")
        assert isinstance(func.return_type, ast.BasicType)
        assert func.return_type.name == "u8"

    def test_function_no_return_type(self):
        """Test function without explicit return type."""
        func = parse_function("fn no_return() { A = 1; }")
        # No return type specified - may be None or implicit
        assert func.name == "no_return"


class TestFarFunctions:
    """Tests for far function declarations."""

    def test_far_function(self):
        """Test far function declaration."""
        func = parse_function("far fn cross_bank() { }")
        assert func.is_far is True

    def test_near_function_default(self):
        """Test normal function is not far."""
        func = parse_function("fn local() { }")
        assert func.is_far is False


class TestFunctionTypes:
    """Tests for function type declarations."""

    def test_function_type_static(self):
        """Test function type in static declaration."""
        from r65.tests.language.common import parse_static
        static = parse_static("#[ram] static mut HANDLER: fn() -> u8;")
        assert isinstance(static.var_type, ast.FunctionType)

    def test_far_function_type(self):
        """Test far function type."""
        from r65.tests.language.common import parse_static
        static = parse_static("#[ram] static mut FAR_HANDLER: far fn();")
        assert isinstance(static.var_type, ast.FunctionType)
        assert static.var_type.is_far is True


class TestMultiReturnFunctions:
    """Tests for functions returning multiple values via a return type list (e.g. `-> u8, u16`)."""

    def test_parse_multi_return_type(self):
        """Test parsing function with multi-return type list."""
        func = parse_function("fn get_pair() -> u8, u16 { return A, X; }")
        assert isinstance(func.return_type, ast.MultiReturnType)
        assert [t.name for t in func.return_type.element_types] == ['u8', 'u16']

    def test_parse_triple_return_type(self):
        """Test parsing function with three return values."""
        func = parse_function("fn get_triple() -> u8, u16, u16 { return A, X, Y; }")
        assert isinstance(func.return_type, ast.MultiReturnType)
        assert [t.name for t in func.return_type.element_types] == ['u8', 'u16', 'u16']

    def test_parse_multi_return_statement(self):
        """Test parsing return statement with multiple values."""
        func = parse_function("fn get_pair() -> u8, u16 { return A, X; }")
        ret_stmt = func.body.statements[0]
        assert isinstance(ret_stmt, ast.ReturnStmt)
        assert len(ret_stmt.values) == 2

    def test_parse_multi_assignment(self):
        """Test parsing multi-assignment from function call."""
        func = parse_function("""
            fn test() {
                let mut a: u8;
                let mut b: u8;
                a, b = get_pair();
            }
        """)
        stmts = func.body.statements
        assert len(stmts) == 3

    def test_parse_let_multi_stmt(self):
        """Test parsing let a, b = func() multi-let syntax."""
        func = parse_function("""
            fn test() {
                let a, b = get_pair();
            }
        """)
        stmts = func.body.statements
        assert len(stmts) == 1
        assert isinstance(stmts[0], ast.MultiLetStmt)
        assert stmts[0].names == ['a', 'b']

    def test_hir_multi_return_type(self):
        """Test HIR building for multi-return type: TupleTypeInfo is produced internally."""
        hir_prog = build_hir("""
            fn get_pair() -> u8, u16 { return A, X; }
        """)
        func = hir_prog.functions[0]
        from r65.compiler.hir.types import TupleTypeInfo
        assert isinstance(func.return_type, TupleTypeInfo)
        assert len(func.return_type.element_types) == 2

    def test_hir_multi_return_m16(self):
        """A function in m16 mode returns u16 in A; the type list reflects that."""
        hir_prog = build_hir("""
            fn split(value @ A: u16) -> u16, u16 { return A, X; }
        """)
        from r65.compiler.hir.types import TupleTypeInfo
        ret = hir_prog.functions[0].return_type
        assert isinstance(ret, TupleTypeInfo)
        assert [str(t) for t in ret.element_types] == ['u16', 'u16']

    def test_hir_multi_return_four_values(self):
        """Four values are allowed when the second is 8-bit in m8 mode (B is free)."""
        hir_prog = build_hir("""
            fn quad(a @ A: u8, b @ B: u8) -> u8, u8, u16, u16 { return A, B, X, Y; }
        """)
        from r65.compiler.hir.types import TupleTypeInfo
        ret = hir_prog.functions[0].return_type
        assert isinstance(ret, TupleTypeInfo)
        assert len(ret.element_types) == 4

    def test_legacy_register_syntax_rejected(self):
        """The old `-> rA, rB` register spelling is no longer valid syntax."""
        from r65.compiler.errors import HIRError
        with pytest.raises(HIRError, match="Undefined type: rA"):
            build_hir("fn pair() -> rA, rB { return A, B; }")

    def test_too_many_return_values_rejected(self):
        """A third value forced into X must be 16-bit; three u8s cannot be returned."""
        from r65.compiler.errors import HIRError
        with pytest.raises(HIRError, match="register X"):
            build_hir("fn three(a @ A: u8, b @ B: u8) -> u8, u8, u8 { return A, B, X; }")

    def test_u16_in_a_under_m8_rejected(self):
        """In m8 mode register A holds one byte; a leading u16 return is rejected."""
        from r65.compiler.errors import HIRError
        with pytest.raises(HIRError, match="register A"):
            build_hir("fn bad() -> u16, u8 { return X, A; }")

