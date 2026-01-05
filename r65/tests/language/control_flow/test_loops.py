"""Tests for loop control flow statements."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import ParseError
from r65.tests.language.common import parse_function, build_hir


class TestLoopStatements:
    """Tests for infinite loop parsing."""

    def test_basic_loop_structures(self):
        """Test basic loop and while structures."""
        # Infinite loop
        func = parse_function("fn test() { loop { A++; } }")
        loop = func.body.statements[0]
        assert isinstance(loop, ast.LoopStmt)
        assert len(loop.body.statements) == 1

        # While loop
        func = parse_function("fn test() { while A != 0 { A--; } }")
        while_stmt = func.body.statements[0]
        assert isinstance(while_stmt, ast.WhileStmt)
        assert while_stmt.condition is not None

    def test_loop_with_control_flow(self):
        """Test loops with break and continue."""
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
        assert isinstance(loop, ast.LoopStmt)
        assert len(loop.body.statements) == 3

    def test_nested_loops(self):
        """Test nested loop structures."""
        func = parse_function("""
            fn test() {
                loop {
                    while X != 0 {
                        loop {
                            Y--;
                            if Y == 0 { break; }
                        }
                        X--;
                    }
                    if A == 0 { break; }
                }
            }
        """)
        outer = func.body.statements[0]
        assert isinstance(outer, ast.LoopStmt)
        inner_while = outer.body.statements[0]
        assert isinstance(inner_while, ast.WhileStmt)
        innermost = inner_while.body.statements[0]
        assert isinstance(innermost, ast.LoopStmt)


class TestWhileConditions:
    """Tests for while loop conditions."""

    def test_while_condition_expressions(self):
        """Test various while conditions."""
        conditions = [
            ("A != 0", ast.BinaryOp),
            ("X > 0", ast.BinaryOp),
            ("flag", ast.Identifier),
            ("A & 0x80", ast.BinaryOp),
            ("A != 0 && X != 0", ast.BinaryOp),
        ]
        for cond, expected_type in conditions:
            func = parse_function(f"fn test() {{ while {cond} {{ A--; }} }}")
            while_stmt = func.body.statements[0]
            assert isinstance(while_stmt.condition, expected_type)


class TestLoopHIR:
    """Tests for loop HIR generation."""

    def test_loop_hir_generation(self):
        """Test loop HIR generation."""
        hir_prog = build_hir("fn test() { loop { A++; if A == 10 { break; } } }")
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 1

    def test_while_hir_generation(self):
        """Test while loop HIR generation."""
        hir_prog = build_hir("fn test() { while A != 0 { A--; } }")
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 1


class TestLoopErrors:
    """Tests for loop parse errors."""

    def test_loop_missing_braces(self):
        """Test loop without braces fails."""
        with pytest.raises(ParseError):
            parse_function("fn test() { loop A++; }")

    def test_while_missing_condition(self):
        """Test while without condition fails."""
        with pytest.raises(ParseError):
            parse_function("fn test() { while { A--; } }")

    def test_while_missing_braces(self):
        """Test while without braces fails."""
        with pytest.raises(ParseError):
            parse_function("fn test() { while A != 0 A--; }")
