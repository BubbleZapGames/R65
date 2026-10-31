# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`*` and `/` take a named const or constant expression as their power-of-2
operand, on either side of `*`.

The validator accepted only a literal, so `y * SCREEN_WIDTH` was rejected
though the const was 32; and codegen multiplies by its right operand, so a
constant on the left (`32 * y`) failed there.
"""

from r65.tests.e2e import ExpectedState


def test_named_constant_multiply_and_divide(e2e):
    result = e2e.run('''
        const W: u16 = 32;
        const DV: u16 = 4;

        #[zeropage(0x10)]
        static mut N: u16;
        #[zeropage(0x20)]
        static mut OUT: [u16; 5];

        #[entry]
        fn main() {
            N = 3;
            OUT[0] = N * W;
            OUT[1] = W * N;
            OUT[2] = 32 * N;
            OUT[3] = N * (W / 2);
            OUT[4] = N / DV;
        }
    ''', ExpectedState(memory={
        0x20: [96, 0, 96, 0, 96, 0, 48, 0, 0, 0],
    }))
    assert result.success, f"Failures: {result.failures}"
