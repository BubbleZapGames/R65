# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end smoke tests for block expressions, if expressions, and trailing
return expressions. HIR shape and desugaring are covered by
compiler/hir/test_block_if_expressions.py; this file covers the codegen path.
"""

from r65.tests.e2e import ExpectedState


class TestBlockIfExprE2E:
    def test_block_expr_runtime(self, e2e):
        """Block expr with statements compiles and executes."""
        result = e2e.run('''
            #[entry]
            fn main() {
                let x: u8 = {
                    let a: u8 = 10;
                    let b: u8 = 20;
                    a + b
                };
                A = x;
            }
        ''', ExpectedState(A=30))
        assert result.success, f"Failures: {result.failures}"

    def test_if_expr_else_if_runtime(self, e2e):
        """if-else if-else chain selects the right branch at runtime."""
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

    def test_trailing_if_return_runtime(self, e2e):
        """Trailing if-expression return evaluates and returns correctly."""
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
