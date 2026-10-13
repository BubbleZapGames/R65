# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Both arms of an `if` expression read a static from memory.

The `if` statement drops cached static loads between its arms; the `if`
expression did not. A load in the `then` arm cached `V` in a vreg, and the
`else` arm - a sibling, not a successor - read that vreg, which held nothing
on its path: `pick()` with C == 0 returned 2 instead of 12.
"""

from r65.tests.e2e import ExpectedState


def test_each_arm_loads_the_static_itself(e2e):
    result = e2e.run('''
        #[zeropage(0x10)] static mut C: u8;
        #[zeropage(0x11)] static mut V: u8;
        #[zeropage(0x12)] static mut O: [u8; 2];

        #[inline(never)]
        fn pick() -> u8 { return if C != 0 { V + 1 } else { V + 2 }; }

        #[entry]
        fn main() { V = 10; C = 1; O[0] = pick(); C = 0; O[1] = pick(); }
    ''', ExpectedState(memory={0x7E0012: [11, 12]}))
    assert result.success, f"Failures: {result.failures}"
