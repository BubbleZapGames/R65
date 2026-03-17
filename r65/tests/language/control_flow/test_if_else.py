# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for if/else control flow statements."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import ParseError
from r65.tests.language.common import parse_function, parse_fails


class TestIfStatements:
    """Tests for if statement parsing and structure."""

    def test_basic_if_structures(self):
        """Test basic if, if-else, and else-if chains."""
        # Simple if
        func = parse_function("fn test() { if A == 0 { X = 1; } }")
        if_stmt = func.body.statements[0]
        assert isinstance(if_stmt, ast.IfStmt)
        assert if_stmt.else_block is None

        # If-else
        func = parse_function("fn test() { if A == 0 { X = 1; } else { X = 2; } }")
        if_stmt = func.body.statements[0]
        assert isinstance(if_stmt.else_block, ast.Block)

        # Else-if chain
        func = parse_function("""
            fn test() {
                if A == 0 { X = 1; }
                else if A == 1 { X = 2; }
                else { X = 3; }
            }
        """)
        if_stmt = func.body.statements[0]
        assert isinstance(if_stmt.else_block, ast.IfStmt)
        nested_if = if_stmt.else_block
        assert isinstance(nested_if.else_block, ast.Block)

    def test_if_conditions(self):
        """Test various condition expressions in if statements."""
        conditions = [
            "A == 0",      # Equality
            "A != 0",      # Inequality
            "A < 10",      # Less than
            "A > 10",      # Greater than
            "A <= 10",     # Less or equal
            "A >= 10",     # Greater or equal
            "A & 0x80",    # Bitwise (truthy)
            "A",           # Simple register
            "flag",        # Identifier
            "A == 0 && X == 0",  # Logical AND
            "A == 0 || X == 0",  # Logical OR
            "!flag",       # Logical NOT
        ]
        for cond in conditions:
            func = parse_function(f"fn test() {{ if {cond} {{ X = 1; }} }}")
            if_stmt = func.body.statements[0]
            assert isinstance(if_stmt, ast.IfStmt)
            assert if_stmt.condition is not None

    def test_nested_if_statements(self):
        """Test deeply nested if statements."""
        func = parse_function("""
            fn test() {
                if A == 0 {
                    if X == 0 {
                        if Y == 0 {
                            STATUS = 0;
                        }
                    }
                }
            }
        """)
        if_stmt = func.body.statements[0]
        assert isinstance(if_stmt, ast.IfStmt)
        nested1 = if_stmt.then_block.statements[0]
        assert isinstance(nested1, ast.IfStmt)
        nested2 = nested1.then_block.statements[0]
        assert isinstance(nested2, ast.IfStmt)

    def test_if_with_multiple_statements(self):
        """Test if blocks with multiple statements."""
        func = parse_function("""
            fn test() {
                if A == 0 {
                    X = 1;
                    Y = 2;
                    A = 3;
                }
            }
        """)
        if_stmt = func.body.statements[0]
        assert len(if_stmt.then_block.statements) == 3



class TestIfErrors:
    """Tests for if statement parse errors."""

    def test_else_without_if(self):
        """Test else without if fails."""
        with pytest.raises(ParseError):
            parse_function("fn test() { else { X = 1; } }")
