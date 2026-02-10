"""
End-to-end tests for block expressions, if expressions, and trailing return expressions.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestBlockExpressions:
    """Test block expressions compile and execute correctly."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_simple_block_expr(self, e2e):
        """Test block expression with just a value."""
        result = e2e.run('''
            #[entry]
            fn main() {
                let x: u8 = { 42 };
                A = x;
            }
        ''', ExpectedState(A=42))
        assert result.success, f"Failures: {result.failures}"

    def test_block_expr_with_let(self, e2e):
        """Test block expression with local variable."""
        result = e2e.run('''
            #[entry]
            fn main() {
                let x: u8 = {
                    let temp: u8 = 5;
                    temp + 1
                };
                A = x;
            }
        ''', ExpectedState(A=6))
        assert result.success, f"Failures: {result.failures}"

    def test_block_expr_multiple_stmts(self, e2e):
        """Test block expression with multiple statements."""
        result = e2e.run('''
            #[entry]
            fn main() {
                let result: u8 = {
                    let a: u8 = 10;
                    let b: u8 = 20;
                    a + b
                };
                A = result;
            }
        ''', ExpectedState(A=30))
        assert result.success, f"Failures: {result.failures}"

    def test_nested_block_expr(self, e2e):
        """Test nested block expressions."""
        result = e2e.run('''
            #[entry]
            fn main() {
                let x: u8 = {
                    let inner: u8 = { 10 };
                    inner + 5
                };
                A = x;
            }
        ''', ExpectedState(A=15))
        assert result.success, f"Failures: {result.failures}"


class TestIfExpressions:
    """Test if expressions compile and execute correctly."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_if_expr_true_branch(self, e2e):
        """Test if expression takes true branch."""
        result = e2e.run('''
            fn choose(val @ A: u8) -> u8 {
                let x: u8 = if val > 0 { 1 } else { 0 };
                return x;
            }

            #[entry]
            fn main() {
                A = choose(5);
            }
        ''', ExpectedState(A=1))
        assert result.success, f"Failures: {result.failures}"

    def test_if_expr_false_branch(self, e2e):
        """Test if expression takes false branch."""
        result = e2e.run('''
            fn choose(val @ A: u8) -> u8 {
                let x: u8 = if val > 10 { 1 } else { 0 };
                return x;
            }

            #[entry]
            fn main() {
                A = choose(5);
            }
        ''', ExpectedState(A=0))
        assert result.success, f"Failures: {result.failures}"

    def test_if_expr_with_register_values(self, e2e):
        """Test if expression using register values in branches."""
        result = e2e.run('''
            fn max(a: u8, b: u8) -> u8 {
                let result: u8 = if a > b { a } else { b };
                return result;
            }

            #[entry]
            fn main() {
                A = max(10, 20);
            }
        ''', ExpectedState(A=20))
        assert result.success, f"Failures: {result.failures}"

    def test_if_expr_else_if_chain(self, e2e):
        """Test if-else if-else chain expression."""
        result = e2e.run('''
            fn classify(val @ A: u8) -> u8 {
                let x: u8 = if val > 10 { 2 } else if val > 5 { 1 } else { 0 };
                return x;
            }

            #[entry]
            fn main() {
                A = classify(3);
            }
        ''', ExpectedState(A=0))
        assert result.success, f"Failures: {result.failures}"

    def test_if_expr_else_if_middle(self, e2e):
        """Test if-else if-else chain takes middle branch."""
        result = e2e.run('''
            fn classify(val @ A: u8) -> u8 {
                let x: u8 = if val > 10 { 2 } else if val > 5 { 1 } else { 0 };
                return x;
            }

            #[entry]
            fn main() {
                A = classify(8);
            }
        ''', ExpectedState(A=1))
        assert result.success, f"Failures: {result.failures}"

    def test_if_expr_with_block_bodies(self, e2e):
        """Test if expression with multi-statement blocks."""
        result = e2e.run('''
            fn compute(val @ A: u8) -> u8 {
                let result: u8 = if val > 10 {
                    let excess: u8 = val - 10;
                    excess
                } else {
                    val
                };
                return result;
            }

            #[entry]
            fn main() {
                A = compute(15);
            }
        ''', ExpectedState(A=5))
        assert result.success, f"Failures: {result.failures}"


class TestTrailingReturn:
    """Test trailing return expressions compile and execute correctly."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_trailing_return_literal(self, e2e):
        """Test function with trailing literal return (no semicolon)."""
        result = e2e.run('''
            fn get_value() -> u8 {
                42
            }

            #[entry]
            fn main() {
                A = get_value();
            }
        ''', ExpectedState(A=42))
        assert result.success, f"Failures: {result.failures}"

    def test_trailing_return_expression(self, e2e):
        """Test function with trailing expression return (no semicolon)."""
        result = e2e.run('''
            fn add_one(val @ A: u8) -> u8 {
                val + 1
            }

            #[entry]
            fn main() {
                A = add_one(9);
            }
        ''', ExpectedState(A=10))
        assert result.success, f"Failures: {result.failures}"

    def test_trailing_return_if_expr(self, e2e):
        """Test function with trailing if expression return (no semicolon)."""
        result = e2e.run('''
            fn abs_diff(a: u8, b: u8) -> u8 {
                if a > b { a - b } else { b - a }
            }

            #[entry]
            fn main() {
                A = abs_diff(3, 10);
            }
        ''', ExpectedState(A=7))
        assert result.success, f"Failures: {result.failures}"

    def test_trailing_return_with_statements(self, e2e):
        """Test function with statements before trailing return (no semicolon)."""
        result = e2e.run('''
            fn compute(val @ A: u8) -> u8 {
                A = val + 10;
                A + 1
            }

            #[entry]
            fn main() {
                A = compute(20);
            }
        ''', ExpectedState(A=31))
        assert result.success, f"Failures: {result.failures}"
