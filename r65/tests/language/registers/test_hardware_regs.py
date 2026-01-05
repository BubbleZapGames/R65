"""Tests for hardware registers: A, X, Y, STATUS, D, DBR, PBR, S, B."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function, build_hir


class TestRegisterAccess:
    """Tests for reading and writing hardware registers."""

    def test_all_registers_read(self):
        """Test reading all hardware registers."""
        registers = ["A", "X", "Y", "STATUS", "D", "DBR", "PBR", "S"]
        for reg in registers:
            func = parse_function(f"fn test() {{ let v: u8 = {reg}; }}")
            let_stmt = func.body.statements[0]
            assert isinstance(let_stmt.initializer, ast.Register)
            assert let_stmt.initializer.name == reg

    def test_register_assignment(self):
        """Test writing to registers."""
        func = parse_function("""
            fn test() {
                A = 10;
                X = 20;
                Y = 30;
                STATUS = 0;
            }
        """)
        assert len(func.body.statements) == 4
        for stmt in func.body.statements:
            assert isinstance(stmt, ast.ExprStmt)
            assert isinstance(stmt.expr, ast.Assignment)

    def test_register_in_expression(self):
        """Test registers in expressions."""
        func = parse_function("fn test() { let v: u8 = A + X; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.BinaryOp)
        assert isinstance(let_stmt.initializer.left, ast.Register)
        assert isinstance(let_stmt.initializer.right, ast.Register)


class TestRegisterOperations:
    """Tests for operations on registers."""

    def test_register_increment(self):
        """Test register increment."""
        func = parse_function("fn test() { A++; X++; Y++; }")
        assert len(func.body.statements) == 3
        for stmt in func.body.statements:
            assert isinstance(stmt.expr, ast.CompoundAssignment)

    def test_register_compound_assignment(self):
        """Test compound assignment on registers."""
        func = parse_function("fn test() { A += 10; X &= 0xFF; }")
        assert len(func.body.statements) == 2


class TestRegisterHIR:
    """Tests for register HIR generation."""

    def test_register_hir(self):
        """Test registers generate proper HIR."""
        hir_prog = build_hir("""
            fn test() {
                A = 10;
                X = A + 1;
                Y = X;
            }
        """)
        func = hir_prog.functions[0]
        assert len(func.body.statements) >= 3
