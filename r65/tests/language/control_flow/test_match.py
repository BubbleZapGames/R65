# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for match expressions."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function


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

    def test_range_pattern_exclusive(self):
        """Test exclusive range pattern 0..5."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    0..5 => 1,
                    _ => 0
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        pat = match_expr.arms[0].pattern
        assert isinstance(pat, ast.RangePattern)
        assert pat.start == 0
        assert pat.end == 5
        assert pat.inclusive is False

    def test_range_pattern_inclusive(self):
        """Test inclusive range pattern 0..=5."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    0..=5 => 1,
                    _ => 0
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        pat = match_expr.arms[0].pattern
        assert isinstance(pat, ast.RangePattern)
        assert pat.start == 0
        assert pat.end == 5
        assert pat.inclusive is True

    def test_range_pattern_in_or(self):
        """Test range pattern inside OR pattern: 0..=3 | 10..=13."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    0..=3 | 10..=13 => 1,
                    _ => 0
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        pat = match_expr.arms[0].pattern
        assert isinstance(pat, ast.OrPattern)
        assert len(pat.patterns) == 2
        assert isinstance(pat.patterns[0], ast.RangePattern)
        assert pat.patterns[0].start == 0
        assert pat.patterns[0].end == 3
        assert pat.patterns[0].inclusive is True
        assert isinstance(pat.patterns[1], ast.RangePattern)
        assert pat.patterns[1].start == 10
        assert pat.patterns[1].end == 13
        assert pat.patterns[1].inclusive is True

    def test_range_pattern_hex(self):
        """Test range pattern with hex literals."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    0x00..=0x0F => 1,
                    _ => 0
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        pat = match_expr.arms[0].pattern
        assert isinstance(pat, ast.RangePattern)
        assert pat.start == 0
        assert pat.end == 15
        assert pat.inclusive is True

    def test_constant_in_match_parses_as_identifier(self):
        """Constants in match arms parse as IdentifierPattern at AST level."""
        func = parse_function("""
            fn test() {
                let x: u8 = match A {
                    MY_CONST => 1,
                    _ => 0
                };
            }
        """)
        match_expr = func.body.statements[0].initializer
        pat = match_expr.arms[0].pattern
        assert isinstance(pat, ast.IdentifierPattern)
        assert pat.name == "MY_CONST"


