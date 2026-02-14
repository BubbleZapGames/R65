"""Tests for block expressions, if expressions, and trailing return expressions."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function


class TestBlockExpression:
    """Tests for block expression parsing."""

    def test_simple_block_expr(self):
        """Test block expression with just a final expression."""
        func = parse_function("""
            fn test() {
                let x: u8 = { 42 };
            }
        """)
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.BlockExpression)
        assert isinstance(let_stmt.initializer.final_expr, ast.IntegerLiteral)
        assert let_stmt.initializer.final_expr.value == 42
        assert len(let_stmt.initializer.statements) == 0

    def test_block_expr_with_statements(self):
        """Test block expression with statements and a final expression."""
        func = parse_function("""
            fn test() {
                let x: u8 = { let temp: u8 = 5; temp + 1 };
            }
        """)
        let_stmt = func.body.statements[0]
        block_expr = let_stmt.initializer
        assert isinstance(block_expr, ast.BlockExpression)
        assert len(block_expr.statements) == 1
        assert isinstance(block_expr.statements[0], ast.LetStmt)
        assert isinstance(block_expr.final_expr, ast.BinaryOp)

    def test_block_expr_with_multiple_statements(self):
        """Test block expression with multiple statements."""
        func = parse_function("""
            fn test() {
                let result: u8 = {
                    let a: u8 = 1;
                    let b: u8 = 2;
                    a + b
                };
            }
        """)
        let_stmt = func.body.statements[0]
        block_expr = let_stmt.initializer
        assert isinstance(block_expr, ast.BlockExpression)
        assert len(block_expr.statements) == 2
        assert isinstance(block_expr.final_expr, ast.BinaryOp)

    def test_nested_block_expr(self):
        """Test nested block expressions."""
        func = parse_function("""
            fn test() {
                let x: u8 = { { 42 } };
            }
        """)
        let_stmt = func.body.statements[0]
        outer = let_stmt.initializer
        assert isinstance(outer, ast.BlockExpression)
        assert isinstance(outer.final_expr, ast.BlockExpression)
        assert isinstance(outer.final_expr.final_expr, ast.IntegerLiteral)


class TestIfExpression:
    """Tests for if expression parsing."""

    def test_basic_if_expr(self):
        """Test basic if expression."""
        func = parse_function("""
            fn test(val @ A: u8) {
                let x: u8 = if val > 0 { 1 } else { 0 };
            }
        """)
        let_stmt = func.body.statements[0]
        if_expr = let_stmt.initializer
        assert isinstance(if_expr, ast.IfExpression)
        assert isinstance(if_expr.condition, ast.BinaryOp)
        assert isinstance(if_expr.then_block, ast.BlockExpression)
        assert isinstance(if_expr.else_block, ast.BlockExpression)

    def test_if_expr_with_block_bodies(self):
        """Test if expression with multi-statement block bodies."""
        func = parse_function("""
            fn test(val @ A: u8) {
                let x: u8 = if val > 10 {
                    let temp: u8 = val - 10;
                    temp
                } else {
                    val
                };
            }
        """)
        let_stmt = func.body.statements[0]
        if_expr = let_stmt.initializer
        assert isinstance(if_expr, ast.IfExpression)
        assert len(if_expr.then_block.statements) == 1
        assert len(if_expr.else_block.statements) == 0

    def test_else_if_chain(self):
        """Test else-if chain in if expression."""
        func = parse_function("""
            fn test(val @ A: u8) {
                let x: u8 = if val > 10 { 2 } else if val > 5 { 1 } else { 0 };
            }
        """)
        let_stmt = func.body.statements[0]
        if_expr = let_stmt.initializer
        assert isinstance(if_expr, ast.IfExpression)
        # else branch is another IfExpression (else if)
        assert isinstance(if_expr.else_block, ast.IfExpression)
        inner_if = if_expr.else_block
        assert isinstance(inner_if.then_block, ast.BlockExpression)
        assert isinstance(inner_if.else_block, ast.BlockExpression)

    def test_if_expr_in_binary_op(self):
        """Test if expression used in a larger expression."""
        func = parse_function("""
            fn test(val @ A: u8) {
                let x: u8 = (if val > 0 { 1 } else { 0 }) + 5;
            }
        """)
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.BinaryOp)
        assert isinstance(let_stmt.initializer.left, ast.IfExpression)


class TestTrailingReturn:
    """Tests for trailing return expression parsing.

    Trailing returns are parsed as normal expression statements, then
    converted to return statements during HIR building.
    """

    def test_trailing_return_literal(self):
        """Test trailing return with a literal (no semicolon)."""
        func = parse_function("""
            fn test() -> u8 {
                42
            }
        """)
        # Parser wraps trailing expr in ExprStmt for HIR builder
        last_stmt = func.body.statements[-1]
        assert isinstance(last_stmt, ast.ExprStmt)
        assert isinstance(last_stmt.expr, ast.IntegerLiteral)
        assert last_stmt.expr.value == 42

    def test_trailing_return_expression(self):
        """Test trailing return with a complex expression (no semicolon)."""
        func = parse_function("""
            fn test(val @ A: u8) -> u8 {
                val + 1
            }
        """)
        last_stmt = func.body.statements[-1]
        assert isinstance(last_stmt, ast.ExprStmt)
        assert isinstance(last_stmt.expr, ast.BinaryOp)

    def test_trailing_return_if_expr(self):
        """Test trailing return with an if expression (no semicolon)."""
        func = parse_function("""
            fn test(val @ A: u8) -> u8 {
                if val > 0 { val } else { 0 }
            }
        """)
        last_stmt = func.body.statements[-1]
        assert isinstance(last_stmt, ast.ExprStmt)
        assert isinstance(last_stmt.expr, ast.IfExpression)

    def test_trailing_return_with_preceding_stmts(self):
        """Test trailing return with statements before it."""
        func = parse_function("""
            fn test(val @ A: u8) -> u8 {
                let x: u8 = val + 1;
                x
            }
        """)
        assert len(func.body.statements) == 2
        assert isinstance(func.body.statements[0], ast.LetStmt)
        last_stmt = func.body.statements[-1]
        assert isinstance(last_stmt, ast.ExprStmt)
        assert isinstance(last_stmt.expr, ast.Identifier)
