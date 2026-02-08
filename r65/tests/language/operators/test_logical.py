"""Tests for logical operators."""

import pytest
from r65.compiler.frontend import ast
from r65.tests.language.common import parse_expr, parse_function


class TestLogicalOperators:
    """Tests for logical operator parsing."""

    def test_binary_logical_operators(self):
        """Test && and || operators."""
        expr = parse_expr("A == 0 && X == 0")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == "&&"

        expr = parse_expr("A == 0 || X == 0")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == "||"

    def test_logical_not(self):
        """Test ! unary operator."""
        expr = parse_expr("!flag")
        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == "!"

    def test_logical_precedence(self):
        """Test && has higher precedence than ||."""
        expr = parse_expr("A == 0 || X == 0 && Y == 0")
        # Should parse as: A == 0 || (X == 0 && Y == 0)
        assert expr.op == "||"
        assert isinstance(expr.right, ast.BinaryOp)
        assert expr.right.op == "&&"

    def test_chained_logical(self):
        """Test chained logical operators."""
        expr = parse_expr("A == 0 && X == 0 && Y == 0")
        assert isinstance(expr, ast.BinaryOp)
        # Chained && - structure depends on associativity
        assert expr.op == "&&"


class TestLogicalInConditions:
    """Tests for logical operators in control flow."""

    def test_logical_in_if(self):
        """Test logical operators in if conditions."""
        func = parse_function("""
            fn test() {
                if A == 0 && X == 0 { Y = 1; }
                if A == 0 || X == 0 { Y = 2; }
                if !done { Y = 3; }
            }
        """)
        assert len(func.body.statements) == 3
        for stmt in func.body.statements:
            assert isinstance(stmt, ast.IfStmt)

    def test_logical_in_while(self):
        """Test logical operators in while conditions."""
        func = parse_function("fn test() { while A != 0 && X != 0 { A--; } }")
        while_stmt = func.body.statements[0]
        assert isinstance(while_stmt.condition, ast.BinaryOp)
        assert while_stmt.condition.op == "&&"


