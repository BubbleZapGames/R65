"""Tests for jump statements: break, continue, return."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import ParseError
from r65.tests.language.common import parse_function, build_hir


class TestBreakContinue:
    """Tests for break and continue statements."""

    def test_break_continue_in_loops(self):
        """Test break and continue in various loop contexts."""
        # Break in loop
        func = parse_function("fn test() { loop { break; } }")
        loop = func.body.statements[0]
        stmt = loop.body.statements[0]
        assert isinstance(stmt, ast.BreakStmt)

        # Continue in loop
        func = parse_function("fn test() { loop { continue; } }")
        loop = func.body.statements[0]
        stmt = loop.body.statements[0]
        assert isinstance(stmt, ast.ContinueStmt)

        # Break in while
        func = parse_function("fn test() { while A != 0 { break; } }")
        while_stmt = func.body.statements[0]
        stmt = while_stmt.body.statements[0]
        assert isinstance(stmt, ast.BreakStmt)

        # Continue in while
        func = parse_function("fn test() { while A != 0 { continue; } }")
        while_stmt = func.body.statements[0]
        stmt = while_stmt.body.statements[0]
        assert isinstance(stmt, ast.ContinueStmt)

    def test_conditional_break_continue(self):
        """Test break and continue inside conditionals."""
        func = parse_function("""
            fn test() {
                loop {
                    if A == 0 { break; }
                    if X == 0 { continue; }
                    A--;
                }
            }
        """)
        loop = func.body.statements[0]
        if1 = loop.body.statements[0]
        if2 = loop.body.statements[1]
        assert isinstance(if1.then_block.statements[0], ast.BreakStmt)
        assert isinstance(if2.then_block.statements[0], ast.ContinueStmt)


class TestReturnStatements:
    """Tests for return statements."""

    def test_return_variants(self):
        """Test various return statement forms."""
        # Empty return (implicit A)
        func = parse_function("fn test() { return; }")
        ret = func.body.statements[0]
        assert isinstance(ret, ast.ReturnStmt)
        assert ret.values == []

        # Return single value
        func = parse_function("fn test() { return A; }")
        ret = func.body.statements[0]
        assert len(ret.values) == 1
        assert isinstance(ret.values[0], ast.Register)

        # Return expression
        func = parse_function("fn test() { return A + 1; }")
        ret = func.body.statements[0]
        assert len(ret.values) == 1
        assert isinstance(ret.values[0], ast.BinaryOp)

        # Return multiple values (parenthesized tuple syntax)
        func = parse_function("fn test() { return (A, X); }")
        ret = func.body.statements[0]
        assert len(ret.values) == 2

        # Return three values (parenthesized tuple syntax)
        func = parse_function("fn test() { return (A, X, Y); }")
        ret = func.body.statements[0]
        assert len(ret.values) == 3

    def test_return_register_combinations(self):
        """Test returning different register combinations."""
        # Single values don't need parentheses
        single = ["A", "X", "Y"]
        for reg in single:
            func = parse_function(f"fn test() {{ return {reg}; }}")
            ret = func.body.statements[0]
            assert isinstance(ret, ast.ReturnStmt)
            assert len(ret.values) == 1

        # Multiple values require parenthesized tuple syntax
        multi = ["(A, X)", "(A, Y)", "(X, Y)", "(A, X, Y)"]
        for combo in multi:
            func = parse_function(f"fn test() {{ return {combo}; }}")
            ret = func.body.statements[0]
            assert isinstance(ret, ast.ReturnStmt)
            # Count values by counting commas + 1
            expected_count = combo.count(",") + 1
            assert len(ret.values) == expected_count

    def test_return_in_conditionals(self):
        """Test return statements in conditional branches."""
        func = parse_function("""
            fn test() {
                if A == 0 { return 0; }
                return 1;
            }
        """)
        if_stmt = func.body.statements[0]
        ret1 = if_stmt.then_block.statements[0]
        ret2 = func.body.statements[1]
        assert isinstance(ret1, ast.ReturnStmt)
        assert isinstance(ret2, ast.ReturnStmt)


class TestJumpHIR:
    """Tests for jump statement HIR generation."""

    def test_break_continue_hir(self):
        """Test break/continue HIR generation."""
        hir_prog = build_hir("""
            fn test() {
                loop {
                    if A == 0 { break; }
                    continue;
                }
            }
        """)
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 1

    def test_return_hir(self):
        """Test return HIR generation."""
        hir_prog = build_hir("fn test() { return A; }")
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 1


class TestJumpErrors:
    """Tests for jump statement errors."""

    def test_break_missing_semicolon(self):
        """Test break without semicolon fails."""
        with pytest.raises(ParseError):
            parse_function("fn test() { loop { break } }")

    def test_continue_missing_semicolon(self):
        """Test continue without semicolon fails."""
        with pytest.raises(ParseError):
            parse_function("fn test() { loop { continue } }")

    def test_return_missing_semicolon(self):
        """Test return without semicolon fails."""
        with pytest.raises(ParseError):
            parse_function("fn test() { return A }")
