# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
An array index in X is reloaded after a struct copy.

MIR lowering keeps `ARR[i]`'s index in X for the next access to `ARR[i]`,
and drops it after anything that writes X. Its list missed AggregateCopy,
which copies more than four bytes with MVN and leaves X past the source:
the second read used X = $0306 and read the wrong byte.
"""

from r65.tests.e2e import ExpectedState


def test_index_reloaded_after_clone_from(e2e):
    result = e2e.run('''
        struct Big { a: u8, b: u8, c: u8, d: u8, e: u8, f: u8 }
        impl Clone for Big {}

        #[lowram(0x0300)] static mut SRC: Big;
        #[lowram(0x0310)] static mut DST: Big;
        #[lowram(0x0320)] static mut ARR: [u8; 8];
        #[zeropage(0x10)] static mut O: [u8; 2];

        #[inline(never)]
        fn f(i: u16) {
            O[0] = ARR[i];
            DST.clone_from(&SRC);
            O[1] = ARR[i];
        }

        #[entry]
        fn main() { ARR[2] = 5; f(2); }
    ''', ExpectedState(memory={0x7E0010: [5, 5]}))
    assert result.success, f"Failures: {result.failures}"
