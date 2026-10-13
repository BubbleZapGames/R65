# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A pointer held in X or Y converts with both of its bytes.

The conversion copied it with `TXA` in m8, which moves only X's low byte, and
then took the high byte from B with `XBA` - whatever B held.
"""

from r65.tests.e2e import ExpectedState


def test_a_near_pointer_in_x_widens_to_far(e2e):
    result = e2e.run('''
        #[lowram(0x0300)] static mut BUF: [u8; 600];
        #[zeropage(0x10)] static mut FP: far *u8;
        #[zeropage(0x14)] static mut O: u8;

        #[inline(never)]
        fn widen(p @ X: *u8) { B = 0x77; FP = p as far *u8; }

        #[entry]
        fn main() {
            BUF[0x120] = 0x5A;
            widen(&BUF[0x120]);
            O = *FP;
        }
    ''', ExpectedState(memory={0x7E0010: [0x20, 0x04, 0x00], 0x7E0014: 0x5A}))
    assert result.success, f"Failures: {result.failures}"
