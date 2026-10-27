# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A bare `fn` is far: callable from any bank, and a bare `fn()` pointer holds it.

`far_to_near` still demotes a function whose direct callers share its bank,
so `twice` below is a JSL from bank 0 while `bump`, called only from bank 1,
comes out near.
"""

from r65.tests.e2e import ExpectedState


def test_bare_fn_in_another_bank(e2e):
    result = e2e.run('''
        #[zeropage(0x10)] static mut O: [u8; 3];
        #[zeropage(0x20)] static mut FP: fn();

        #[bank(1)]

        #[inline(never)]
        fn bump() { O[0] = O[0] + 1; }

        #[inline(never)]
        fn twice() { bump(); bump(); }

        #[inline(never)]
        fn mark() { O[2] = 7; }

        #[bank(0)]

        #[entry]
        fn main() {
            O[0] = 0;
            O[1] = 0;
            twice();
            FP = mark;
            FP();
            O[1] = 5;
        }
    ''', ExpectedState(memory={0x7E0010: [2, 5, 7]}))
    assert result.success, f"Failures: {result.failures}"
