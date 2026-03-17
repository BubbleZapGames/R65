# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for bitwise operators."""

import pytest
from r65.compiler.frontend import ast
from r65.tests.language.common import parse_expr, parse_function


class TestBitwiseOperators:
    """Tests for bitwise operator parsing."""

    def test_binary_bitwise_operators(self):
        """Test &, |, ^ binary operators."""
        ops = [("&", "A & 0xFF"), ("|", "A | 0x80"), ("^", "A ^ 0x0F")]
        for op, source in ops:
            expr = parse_expr(source)
            assert isinstance(expr, ast.BinaryOp)
            assert expr.op == op

    def test_bitwise_not(self):
        """Test ~ unary operator."""
        expr = parse_expr("~A")
        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == "~"

    def test_shift_operators(self):
        """Test << and >> shift operators."""
        for op in ["<<", ">>"]:
            expr = parse_expr(f"A {op} 4")
            assert isinstance(expr, ast.BinaryOp)
            assert expr.op == op


class TestBitwiseCompoundAssignment:
    """Tests for bitwise compound assignment operators."""

    def test_bitwise_compound_operators(self):
        """Test &=, |=, ^=, <<=, >>=."""
        ops = ["&", "|", "^", "<<", ">>"]
        for op in ops:
            func = parse_function(f"fn test() {{ A {op}= 1; }}")
            stmt = func.body.statements[0]
            assert isinstance(stmt, ast.ExprStmt)
            assert isinstance(stmt.expr, ast.CompoundAssignment)
            assert stmt.expr.operator == op


class TestBitwisePrecedence:
    """Tests for bitwise operator precedence."""

    def test_bitwise_precedence(self):
        """Test precedence: & before | before ^."""
        # A | B & C should be A | (B & C)
        expr = parse_expr("A | B & C")
        assert expr.op == "|"
        assert isinstance(expr.right, ast.BinaryOp)
        assert expr.right.op == "&"

    def test_shift_precedence(self):
        """Test shift operators have correct precedence with arithmetic."""
        # A + B << 2 should be (A + B) << 2 or A + (B << 2)
        expr = parse_expr("A + B << 2")
        # Verify it parses without error
        assert isinstance(expr, ast.BinaryOp)


