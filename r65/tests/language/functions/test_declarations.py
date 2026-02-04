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
    """Tests for functions returning multiple values."""

    def test_parse_tuple_return_type(self):
        """Test parsing function with tuple return type."""
        func = parse_function("fn get_pair() -> (u8, u8) { return A, X; }")
        assert isinstance(func.return_type, ast.TupleType)
        assert len(func.return_type.element_types) == 2
        assert func.return_type.element_types[0].name == "u8"
        assert func.return_type.element_types[1].name == "u8"

    def test_parse_triple_return_type(self):
        """Test parsing function with three return values."""
        func = parse_function("fn get_triple() -> (u8, u8, u8) { return A, X, Y; }")
        assert isinstance(func.return_type, ast.TupleType)
        assert len(func.return_type.element_types) == 3

    def test_parse_multi_return_statement(self):
        """Test parsing return statement with multiple values."""
        func = parse_function("fn get_pair() -> (u8, u8) { return A, X; }")
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
        # The multi-assignment should be in the body
        stmts = func.body.statements
        assert len(stmts) == 3

    def test_hir_tuple_return_type(self):
        """Test HIR building for tuple return type."""
        hir_prog = build_hir("""
            fn get_pair() -> (u8, u8) { return A, X; }
        """)
        func = hir_prog.functions[0]
        from r65.compiler.hir.types import TupleTypeInfo
        assert isinstance(func.return_type, TupleTypeInfo)
        assert len(func.return_type.element_types) == 2

