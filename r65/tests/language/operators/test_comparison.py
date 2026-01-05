"""Tests for comparison operators."""

import pytest
from r65.compiler.frontend import ast
from r65.tests.language.common import parse_expr, parse_function, build_hir


class TestComparisonOperators:
    """Tests for comparison operator parsing."""

    def test_all_comparison_operators(self):
        """Test ==, !=, <, <=, >, >= operators."""
        ops = [
            ("==", "A == 0"),
            ("!=", "A != 0"),
            ("<", "A < 10"),
            ("<=", "A <= 10"),
            (">", "A > 10"),
            (">=", "A >= 10"),
        ]
        for op, source in ops:
            expr = parse_expr(source)
            assert isinstance(expr, ast.BinaryOp)
            assert expr.op == op

    def test_comparison_with_expressions(self):
        """Test comparisons with complex expressions."""
        expr = parse_expr("A + 1 == X - 1")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == "=="
        assert isinstance(expr.left, ast.BinaryOp)
        assert isinstance(expr.right, ast.BinaryOp)

    def test_comparison_in_conditionals(self):
        """Test comparisons in if conditions."""
        func = parse_function("fn test() { if A == 0 { X = 1; } }")
        if_stmt = func.body.statements[0]
        assert isinstance(if_stmt.condition, ast.BinaryOp)
        assert if_stmt.condition.op == "=="


class TestComparisonHIR:
    """Tests for comparison HIR generation."""

    def test_comparison_hir(self):
        """Test comparison expressions generate proper HIR."""
        hir_prog = build_hir("fn test() { if A == 0 && X != 0 { Y = 1; } }")
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 1
