"""Tests for arithmetic operators."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import ParseError
from r65.tests.language.common import parse_expr, parse_function, build_hir


class TestArithmeticOperators:
    """Tests for arithmetic operator parsing."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operators: +, -, *, /."""
        ops = [("+", "1 + 2"), ("-", "1 - 2"), ("*", "1 * 2"), ("/", "1 / 2")]
        for op, source in ops:
            expr = parse_expr(source)
            assert isinstance(expr, ast.BinaryOp)
            assert expr.op == op

    def test_unary_minus(self):
        """Test unary minus operator."""
        expr = parse_expr("-5")
        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == "-"

    def test_compound_arithmetic(self):
        """Test compound arithmetic expressions."""
        expr = parse_expr("1 + 2 * 3")
        assert isinstance(expr, ast.BinaryOp)
        # Check precedence: 1 + (2 * 3)
        assert expr.op == "+"
        assert isinstance(expr.right, ast.BinaryOp)
        assert expr.right.op == "*"

    def test_associativity(self):
        """Test left-to-right associativity of arithmetic."""
        expr = parse_expr("A - B - C")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == "-"
        # Left associative: (A - B) - C
        assert isinstance(expr.left, ast.BinaryOp)


class TestCompoundAssignment:
    """Tests for compound assignment operators."""

    def test_compound_assignment_operators(self):
        """Test +=, -=, *=, /=, %=."""
        ops = ["+", "-", "*", "/", "%"]
        for op in ops:
            func = parse_function(f"fn test() {{ A {op}= 1; }}")
            stmt = func.body.statements[0]
            assert isinstance(stmt, ast.ExprStmt)
            assert isinstance(stmt.expr, ast.CompoundAssignment)
            assert stmt.expr.operator == op


class TestIncrementDecrement:
    """Tests for increment/decrement operators."""

    def test_postfix_operators(self):
        """Test A++ and A-- postfix operators."""
        for op in ["++", "--"]:
            func = parse_function(f"fn test() {{ A{op}; }}")
            stmt = func.body.statements[0]
            assert isinstance(stmt, ast.ExprStmt)
            # Postfix operators are desugared to compound assignment
            assert isinstance(stmt.expr, ast.CompoundAssignment)


class TestArithmeticHIR:
    """Tests for arithmetic HIR generation."""

    def test_arithmetic_hir(self):
        """Test arithmetic expressions generate proper HIR."""
        hir_prog = build_hir("fn test() { let x: u8 = 1 + 2 * 3; }")
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 1
