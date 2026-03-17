# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for register aliasing with @ syntax."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function, build_hir


class TestRegisterAliasing:
    """Tests for let name @ register syntax."""

    def test_basic_alias(self):
        """Test basic register aliasing."""
        func = parse_function("fn test() { let counter @ A = 10; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt, ast.LetStmt)
        assert let_stmt.name == "counter"
        assert let_stmt.binding is not None
        assert let_stmt.binding.name == "A"

    def test_alias_all_registers(self):
        """Test aliasing with different registers."""
        registers = ["A", "X", "Y"]
        for reg in registers:
            func = parse_function(f"fn test() {{ let val @ {reg} = 0; }}")
            let_stmt = func.body.statements[0]
            assert let_stmt.binding.name == reg

    def test_alias_usage(self):
        """Test using aliased register name."""
        func = parse_function("""
            fn test() {
                let count @ A = 10;
                count = count + 1;
            }
        """)
        assert len(func.body.statements) == 2


class TestAliasHIR:
    """Tests for alias HIR generation."""

    def test_alias_hir(self):
        """Test aliases generate proper HIR."""
        hir_prog = build_hir("""
            fn test() {
                let x @ A = 10;
                let y @ X = 20;
            }
        """)
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 2
