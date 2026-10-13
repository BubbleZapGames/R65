# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A fill or block move restores 8-bit A, and codegen's mode tracking knows it.

These emitted `REP #$20 ... SEP #$20` directly. Entered with A already
16-bit, the `SEP` left it 8-bit while the tracker still said 16, so the next
16-bit operation got no `REP`: `W + 0x0101` came out as `ADC #$0101` in m8,
which does not even assemble.
"""

from r65.tests.e2e import ExpectedState


def test_u16_arithmetic_after_a_u16_fill(e2e):
    result = e2e.run('''
        #[zeropage(0x10)] static mut W: u16;
        #[zeropage(0x12)] static mut O: [u16; 2];

        #[inline(never)]
        fn f() {
            W = W + 0x0101;
            let a: [u16; 4] = [0x0302; 4];
            O[0] = W + 0x0101;
            O[1] = a[3];
        }

        #[entry]
        fn main() { W = 0x1000; f(); }
    ''', ExpectedState(memory={0x7E0012: [0x02, 0x12, 0x02, 0x03]}))
    assert result.success, f"Failures: {result.failures} {result.error}"
