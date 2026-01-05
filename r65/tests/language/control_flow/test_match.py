"""Tests for match expressions."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function, build_hir


class TestMatchExpression:
    """Tests for match expression parsing."""

    def test_basic_match(self):
        """Test basic match expression."""
        func = parse_function("""
            fn test() {
                let result: u8 = match A {
                    0 => 1,
                    1 => 2,
                    _ => 0
                };
            }
        """)
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.MatchExpression)
        assert len(let_stmt.initializer.arms) == 3

    def test_match_with_expressions(self):
        """Test match arms with expressions."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    0 => A + 1,
                    _ => A - 1
                };
            }
        """)
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.MatchExpression)

    def test_match_scrutinee(self):
        """Test match with different scrutinee expressions."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A + 1 {
                    0 => 1,
                    _ => 0
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        assert isinstance(match_expr.scrutinee, ast.BinaryOp)


class TestMatchPatterns:
    """Tests for match patterns."""

    def test_literal_pattern(self):
        """Test literal patterns in match arms."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    0 => 1,
                    1 => 2,
                    2 => 3,
                    _ => 0
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        arm = match_expr.arms[0]
        assert isinstance(arm, ast.MatchArm)

    def test_wildcard_pattern(self):
        """Test wildcard pattern _ in match."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    _ => 42
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        assert len(match_expr.arms) == 1


class TestMatchHIR:
    """Tests for match HIR generation."""

    def test_match_hir(self):
        """Test match expressions generate proper HIR."""
        hir_prog = build_hir("""
            fn test() {
                let x: u8 = match A {
                    0 => 1,
                    _ => 0
                };
            }
        """)
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 1
