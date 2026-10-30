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

    def test_empty_arm_block(self):
        """An empty `{}` arm body is a valid no-op (statement-form match)."""
        func = parse_function("""
            fn test(x @ A: u8) {
                match x {
                    0 => { A = 1; },
                    _ => {},
                };
            }
        """)
        match_expr = func.body.statements[0].expr
        assert isinstance(match_expr, ast.MatchExpression)
        assert len(match_expr.arms) == 2
        wildcard_body = match_expr.arms[1].body
        assert isinstance(wildcard_body, ast.Block)
        assert wildcard_body.statements == []


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




class TestMatchStatement:
    """A match in statement position ends its statement; no `;` needed."""

    ARMS = "{ 0 => { G = 1; } _ => { G = 2; } }"

    def body(self, source):
        return parse_function(f"fn test(t: u8) {{ {source} }}").body.statements

    def test_statement_follows(self):
        stmts = self.body(f"match t {self.ARMS} G = 3;")
        assert isinstance(stmts[0].expr, ast.MatchExpression)
        assert isinstance(stmts[1].expr, ast.Assignment)

    def test_deref_starts_next_statement(self):
        """`*P = 3` after a match is a new statement, not a multiply."""
        stmts = self.body(f"match t {self.ARMS} *P = 3;")
        assert isinstance(stmts[0].expr, ast.MatchExpression)
        assert isinstance(stmts[1].expr.target, ast.Dereference)

    def test_explicit_semicolon_still_parses(self):
        stmts = self.body(f"match t {self.ARMS}; G = 3;")
        assert isinstance(stmts[0].expr, ast.MatchExpression)
        assert len(stmts) == 2

    def test_loop_bodies_end_in_match(self):
        for head in ("loop", "while t != 0", "for i in 0..4"):
            loop = self.body(f"{head} {{ match t {self.ARMS} }}")[0]
            assert isinstance(loop.body.statements[-1].expr, ast.MatchExpression), head

    def test_if_body_ends_in_match(self):
        if_stmt = self.body(f"if t == 0 {{ match t {self.ARMS} }} G = 3;")[0]
        assert isinstance(if_stmt, ast.IfStmt)
        assert isinstance(if_stmt.then_block.statements[-1].expr, ast.MatchExpression)

    def test_if_with_valued_else_stays_an_expression(self):
        """`if c { match .. } else { 3 }` is still an if expression, so it can
        be a function's value."""
        stmts = self.body("if t == 0 { match t { 0 => 1, _ => 2 } } else { 3 }")
        if_expr = stmts[-1].expr
        assert isinstance(if_expr, ast.IfExpression)
        assert isinstance(if_expr.then_block.final_expr, ast.MatchExpression)

    def test_else_if_let_is_kept(self):
        """HIR builds an else only from a Block or IfStmt; `else if let` is an
        ExprStmt, so it is wrapped rather than dropped."""
        if_stmt = self.body("if t == 5 { G = 1; } else if let 0 = t { G = 2; }")[0]
        assert isinstance(if_stmt.else_block, ast.Block)
        assert isinstance(if_stmt.else_block.statements[0].expr, ast.MatchExpression)
