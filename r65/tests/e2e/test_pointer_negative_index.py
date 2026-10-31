# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A negative pointer index reaches the element below the pointer.

`-1: i8` reaches MIR lowering masked to 255, and a byte pointer's index went
into Y at its own 1-byte width, so `c[-1]` loaded `LDY #$FF` and read 255
bytes above c. Y now always gets the u16 byte offset: a negative constant
wraps to $FFxx and a run-time i8 index is sign-extended.
"""

from r65.tests.e2e import ExpectedState


def test_negative_byte_pointer_indices(e2e):
    result = e2e.run('''
        #[lowram(0x0300)]
        static mut GRID: [u8; 64];
        #[zeropage(0x10)]
        static mut I: i8;
        #[zeropage(0x20)]
        static mut OUT: [u8; 3];

        #[entry]
        fn main() {
            GRID[31] = 0x11;
            GRID[62] = 0x22;
            GRID[61] = 0x33;
            I = -2;
            let c: *u8 = &GRID[63] as *u8;
            OUT[0] = c[-1];         // GRID[62]
            OUT[1] = c[-32];        // GRID[31]
            OUT[2] = c[I];          // GRID[61], run-time i8
            c[-3] = 0x44;           // GRID[60]
        }
    ''', ExpectedState(memory={
        0x20: [0x22, 0x11, 0x33],
        0x033C: 0x44,
    }))
    assert result.success, f"Failures: {result.failures}"
