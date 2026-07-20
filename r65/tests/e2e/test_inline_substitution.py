# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end behavioral tests for inline parameter substitution (copy-propagation
at inline time). A wrong substitution corrupts the computed value silently, so
these compile at -O2 (which enables implicit inlining) and assert the RESULT is
correct — the only guard that catches a bad splice.
"""

from r65.tests.e2e import ExpectedState


class TestInlineSubstitutionBehavior:
    def test_register_param_const_arg(self, e2e):
        """f(v @ A){ return v + 3 } called with a constant must still yield 10."""
        result = e2e.run('''
            #[inline]
            fn add3(v @ A: u8) -> u8 { return v + 3; }
            #[entry]
            fn main() { A = add3(7); }
        ''', ExpectedState(A=10), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_stack_param_const_arg(self, e2e):
        """Stack param, constant arg — the bridge store is elided; value stands."""
        result = e2e.run('''
            #[inline]
            fn add3s(v: u8) -> u8 { return v + 3; }
            #[entry]
            fn main() { A = add3s(7); }
        ''', ExpectedState(A=10), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_register_param_variable_arg(self, e2e):
        """Variable (vreg) arg substituted into the body."""
        result = e2e.run('''
            #[inline]
            fn dbl(v @ A: u8) -> u8 { return v + v; }
            #[entry]
            fn main() { let x: u8 = 21; A = dbl(x); }
        ''', ExpectedState(A=42), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_param_used_multiple_times(self, e2e):
        """A param used more than once: substituting the operand at every use
        must not re-evaluate anything (it is already a single value)."""
        result = e2e.run('''
            #[inline]
            fn poly(v @ A: u8) -> u8 { return v + v + v; }
            #[entry]
            fn main() { A = poly(4); }
        ''', ExpectedState(A=12), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_two_params_const(self, e2e):
        """Two params, both constants — ordering/substitution must be correct."""
        result = e2e.run('''
            #[inline]
            fn sub(a @ A: u8, b @ X: u16) -> u8 { return a - (b as u8); }
            #[entry]
            fn main() { A = sub(50, 8); }
        ''', ExpectedState(A=42), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_multiple_inlined_calls_share_result(self, e2e):
        """Several inlined calls in one function — no cross-contamination."""
        result = e2e.run('''
            #[inline]
            fn add1(v @ A: u8) -> u8 { return v + 1; }
            #[entry]
            fn main() {
                let a: u8 = add1(10);
                let b: u8 = add1(20);
                A = a + b;
            }
        ''', ExpectedState(A=32), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_pointer_param_immediate_addr(self, e2e):
        """Constant-address pointer arg: the Immediate must NOT be spliced into
        the pointer slot (it stays a register/DP value), so the store lands."""
        result = e2e.run('''
            #[zeropage(0x40)]
            static mut OUT: u8;
            #[inline]
            fn store5(p: *u8) { *p = 5; }
            #[entry]
            fn main() { store5(&OUT); A = OUT; }
        ''', ExpectedState(A=5, memory={0x40: 5}), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"
