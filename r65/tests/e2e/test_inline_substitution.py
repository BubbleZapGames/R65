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

    def test_method_variable_bound_param(self, e2e):
        """`t @ TEMP` on a method: inlining drops the Call, so the argument
        must already be stored to TEMP (both bytes of a u16)."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut TEMP: u16;
            #[zeropage(0x24)]
            static mut OUT: u16;
            struct Thing { v: u8 }
            #[lowram]
            static mut THING: Thing;
            impl Thing {
                fn put(*self, t @ TEMP: u16) { OUT = t; }
            }
            #[entry]
            fn main() { THING.put(0x1234); }
        ''', ExpectedState(memory={0x7E0024: [0x34, 0x12]}), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"


class TestInlineMultiReturnBehavior:
    """A multi-return Call names no result vregs: the caller captures each
    value from its ABI register after the call. Inlining must deliver every
    value to its capture, not leave the registers holding stale bytes."""

    def test_let_capture_lands_each_value(self, e2e):
        """`let v, j = h(..)` inlined: v (A) and j (X) both reach ARR[j]."""
        result = e2e.run('''
            #[lowram(0x0300)] static mut ARR: [u8; 16];
            fn h(a: u8, b: u16) -> u8, u16 { return a + 1, b + 2; }
            #[entry]
            fn main() { let v, j = h(3, 4); ARR[j] = v; }
        ''', ExpectedState(memory={0x0306: 4}), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_static_targets_read_return_registers(self, e2e):
        """`P, Q = h(..)` stores straight from the return registers, so the
        inlined values must be placed in A and X."""
        result = e2e.run('''
            #[zeropage(0x40)] static mut P: u8;
            #[zeropage(0x42)] static mut Q: u16;
            #[inline(always)]
            fn h(a: u8, b: u16) -> u8, u16 { return a + 1, b + 2; }
            #[entry]
            fn main() { P, Q = h(3, 0x1230); }
        ''', ExpectedState(memory={0x40: 4, 0x42: [0x32, 0x12]}),
            extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"

    def test_u8_pair_via_b(self, e2e):
        """`-> u8, u8` returns in A and B; both captures must see their value."""
        result = e2e.run('''
            #[zeropage(0x40)] static mut P: u8;
            #[zeropage(0x41)] static mut Q: u8;
            #[inline(always)]
            fn h(a: u8, b: u8) -> u8, u8 { return a + 1, b + 2; }
            #[entry]
            fn main() { let v, w = h(3, 7); P = v; Q = w; }
        ''', ExpectedState(memory={0x40: 4, 0x41: 9}), extra_args=["-O2"])
        assert result.success, f"Failures: {result.failures}"
