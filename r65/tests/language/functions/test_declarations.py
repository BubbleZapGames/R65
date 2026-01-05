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


class TestFunctionHIR:
    """Tests for function HIR generation."""

    def test_function_hir(self):
        """Test functions generate proper HIR."""
        hir_prog = build_hir("""
            fn helper() -> u8 { return A; }
            fn main() { let x: u8 = helper(); }
        """)
        assert len(hir_prog.functions) == 2
