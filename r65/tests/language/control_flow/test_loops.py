# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for loop control flow statements."""

from r65.compiler.frontend import ast
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



class TestForLoops:
    """Tests for for loop statements."""

    def test_basic_for_loop(self):
        """Test basic for loop parsing."""
        func = parse_function("fn test() { for i in 0..10 { A++; } }")
        for_stmt = func.body.statements[0]
        assert isinstance(for_stmt, ast.ForStmt)
        assert for_stmt.variable == "i"

    def test_for_loop_with_array_len(self):
        """Test for loop using array.len() as range end."""
        from r65.compiler.hir import nodes as hir
        hir_prog = build_hir("""
            #[ram] static mut BUFFER: [u8; 64] = [0; 64];
            fn test() {
                for index in 0..BUFFER.len() {
                    BUFFER[index] = 0;
                }
            }
        """)
        func = hir_prog.functions[0]
        # For loop desugars to block with let + while
        block = func.body.statements[0]
        assert isinstance(block, hir.HIRBlock)
        let_stmt = block.statements[0]
        assert isinstance(let_stmt, hir.HIRLetStmt)
        assert let_stmt.name == "index"
        while_stmt = block.statements[1]
        assert isinstance(while_stmt, hir.HIRWhileStmt)
        # Condition is: index < BUFFER.len() where len() is const-evaluated to 64
        assert isinstance(while_stmt.condition, hir.HIRBinaryOp)
        assert while_stmt.condition.op == '<'
        # Right side should be const-evaluated to 64
        assert isinstance(while_stmt.condition.right, hir.HIRIntegerLiteral)
        assert while_stmt.condition.right.value == 64

    def test_for_loop_with_array_len_expression(self):
        """Test for loop using array.len() in arithmetic expression."""
        from r65.compiler.hir import nodes as hir
        hir_prog = build_hir("""
            #[ram] static mut DATA: [u8; 100] = [0; 100];
            fn test() {
                for i in 0..DATA.len() - 1 {
                    A = i as u8;
                }
            }
        """)
        func = hir_prog.functions[0]
        # For loop desugars to block with let + while
        block = func.body.statements[0]
        assert isinstance(block, hir.HIRBlock)
        while_stmt = block.statements[1]
        assert isinstance(while_stmt, hir.HIRWhileStmt)
        # Right side is: DATA.len() - 1 where len() is const-evaluated to 100
        # The subtraction remains as HIRBinaryOp at HIR stage
        end_expr = while_stmt.condition.right
        assert isinstance(end_expr, hir.HIRBinaryOp)
        assert end_expr.op == '-'
        # Left side should be const-evaluated len() = 100
        assert isinstance(end_expr.left, hir.HIRIntegerLiteral)
        assert end_expr.left.value == 100
        # Right side is literal 1
        assert isinstance(end_expr.right, hir.HIRIntegerLiteral)
        assert end_expr.right.value == 1
