# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for block expressions, if expressions, and trailing return expressions.

Validates the HIR shape produced by the parser/HIR builder. Runtime arithmetic
("does 5 + 1 == 6") is exercised by every other test in the suite — what we
care about here is that the desugaring produces the right tree.
"""
from r65.compiler.frontend import Parser
from r65.compiler.hir import (
    HIRBuilder,
    HIRBlockExpression,
    HIRIfExpression,
    HIRIntegerLiteral,
    HIRBinaryOp,
    HIRReturnStmt,
    HIRLetStmt,
)
from r65.compiler.typeck import TypeChecker


def build(source: str):
    parser = Parser()
    ast = parser.parse(source)
    hir = HIRBuilder().build_program(ast)
    TypeChecker(hir).check()
    return hir


def first_let_init(hir, fn_name: str = "main"):
    """Return the initializer expression of the first `let` in fn_name."""
    for decl in hir.declarations:
        if getattr(decl, 'name', None) == fn_name:
            for stmt in decl.body.statements:
                if isinstance(stmt, HIRLetStmt) and stmt.initializer is not None:
                    return stmt.initializer
    raise AssertionError(f"no let-initializer in {fn_name}")


class TestBlockExpressions:
    """Block expressions desugar to HIRBlockExpression with a final_expr."""

    def test_simple_block_expr(self):
        """{ 42 } → HIRBlockExpression, final_expr is the literal."""
        hir = build("fn main() { let x: u8 = { 42 }; }")
        init = first_let_init(hir)
        assert isinstance(init, HIRBlockExpression)
        assert init.statements == []
        assert isinstance(init.final_expr, HIRIntegerLiteral)
        assert init.final_expr.value == 42
        assert init.expr_type.name == 'u8'

    def test_block_expr_with_let(self):
        """{ let temp = 5; temp + 1 } — statements + trailing expr."""
        hir = build('''
            fn main() {
                let x: u8 = {
                    let temp: u8 = 5;
                    temp + 1
                };
            }
        ''')
        init = first_let_init(hir)
        assert isinstance(init, HIRBlockExpression)
        assert len(init.statements) == 1
        assert isinstance(init.statements[0], HIRLetStmt)
        assert isinstance(init.final_expr, HIRBinaryOp)
        assert init.final_expr.op == '+'
        assert init.expr_type.name == 'u8'

    def test_block_expr_multiple_stmts(self):
        """{ let a; let b; a + b } — multiple lets before trailing expr."""
        hir = build('''
            fn main() {
                let r: u8 = {
                    let a: u8 = 10;
                    let b: u8 = 20;
                    a + b
                };
            }
        ''')
        init = first_let_init(hir)
        assert isinstance(init, HIRBlockExpression)
        assert len(init.statements) == 2
        assert isinstance(init.final_expr, HIRBinaryOp)

    def test_nested_block_expr(self):
        """Nested block exprs nest as HIRBlockExpression in HIRBlockExpression."""
        hir = build('''
            fn main() {
                let x: u8 = {
                    let inner: u8 = { 10 };
                    inner + 5
                };
            }
        ''')
        init = first_let_init(hir)
        assert isinstance(init, HIRBlockExpression)
        inner_let = init.statements[0]
        assert isinstance(inner_let, HIRLetStmt)
        assert isinstance(inner_let.initializer, HIRBlockExpression)
        assert isinstance(inner_let.initializer.final_expr, HIRIntegerLiteral)


class TestIfExpressions:
    """If expressions desugar to HIRIfExpression with then/else blocks."""

    def test_if_expr_basic(self):
        """if cond { 1 } else { 0 } — condition, then_block, else_block all set."""
        hir = build('''
            fn choose(val @ A: u8) -> u8 {
                let x: u8 = if val > 0 { 1 } else { 0 };
                return x;
            }
        ''')
        init = first_let_init(hir, "choose")
        assert isinstance(init, HIRIfExpression)
        assert isinstance(init.condition, HIRBinaryOp)
        assert init.condition.op == '>'
        assert isinstance(init.then_block, HIRBlockExpression)
        assert isinstance(init.else_block, HIRBlockExpression)
        assert init.expr_type.name == 'u8'
        # Both branches must be u8 for unified type
        assert init.then_block.expr_type.name == 'u8'
        assert init.else_block.expr_type.name == 'u8'

    def test_if_expr_else_if_chain_nests(self):
        """else if … chains nest: else_block is itself a HIRIfExpression."""
        hir = build('''
            fn classify(val @ A: u8) -> u8 {
                let x: u8 = if val > 10 { 2 } else if val > 5 { 1 } else { 0 };
                return x;
            }
        ''')
        init = first_let_init(hir, "classify")
        assert isinstance(init, HIRIfExpression)
        assert isinstance(init.then_block, HIRBlockExpression)
        # Outer else is itself an if-expression
        assert isinstance(init.else_block, HIRIfExpression)
        # Innermost else is a plain block
        assert isinstance(init.else_block.else_block, HIRBlockExpression)

    def test_if_expr_with_block_bodies(self):
        """Multi-statement if-expr branches are HIRBlockExpression with statements."""
        hir = build('''
            fn compute(val @ A: u8) -> u8 {
                let r: u8 = if val > 10 {
                    let excess: u8 = val - 10;
                    excess
                } else {
                    val
                };
                return r;
            }
        ''')
        init = first_let_init(hir, "compute")
        assert isinstance(init, HIRIfExpression)
        # Then branch has a statement before final_expr
        assert len(init.then_block.statements) == 1
        assert isinstance(init.then_block.statements[0], HIRLetStmt)
        assert init.then_block.final_expr is not None
        # Else branch is a bare value
        assert init.else_block.statements == []
        assert init.else_block.final_expr is not None


class TestTrailingReturn:
    """Trailing expression in fn body becomes HIRReturnStmt."""

    def test_trailing_literal(self):
        """fn get() -> u8 { 42 } — body ends with a return of the literal."""
        hir = build("fn get() -> u8 { 42 }")
        fn = hir.declarations[0]
        last = fn.body.statements[-1]
        assert isinstance(last, HIRReturnStmt)
        assert len(last.values) == 1
        assert isinstance(last.values[0], HIRIntegerLiteral)
        assert last.values[0].value == 42

    def test_trailing_binary_op(self):
        """fn add_one(v) -> u8 { v + 1 } — trailing binop becomes return."""
        hir = build('''
            fn add_one(val @ A: u8) -> u8 {
                val + 1
            }
        ''')
        fn = hir.declarations[0]
        last = fn.body.statements[-1]
        assert isinstance(last, HIRReturnStmt)
        assert isinstance(last.values[0], HIRBinaryOp)
        assert last.values[0].op == '+'

    def test_trailing_if_expr(self):
        """Trailing if-expression: return wraps a HIRIfExpression."""
        hir = build('''
            fn abs_diff(a: u8, b: u8) -> u8 {
                if a > b { a - b } else { b - a }
            }
        ''')
        fn = hir.declarations[0]
        last = fn.body.statements[-1]
        assert isinstance(last, HIRReturnStmt)
        assert isinstance(last.values[0], HIRIfExpression)

    def test_trailing_after_statements(self):
        """Statements before the trailing expr stay as statements; expr returns."""
        hir = build('''
            fn compute(val @ A: u8) -> u8 {
                A = val + 10;
                A + 1
            }
        ''')
        fn = hir.declarations[0]
        # All but the last are non-return statements
        for s in fn.body.statements[:-1]:
            assert not isinstance(s, HIRReturnStmt)
        last = fn.body.statements[-1]
        assert isinstance(last, HIRReturnStmt)
        assert isinstance(last.values[0], HIRBinaryOp)
