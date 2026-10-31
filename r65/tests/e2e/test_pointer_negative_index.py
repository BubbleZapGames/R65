# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A negative pointer index reaches the element below the pointer, in its bank.

`-1: i8` reaches MIR lowering masked to 255, and a byte pointer's index went
into Y at its own 1-byte width, so `c[-1]` loaded `LDY #$FF` and read 255
bytes above c. Nor can Y hold $FFFF: `(dp),Y` adds Y to the 24-bit address,
reaching the next bank, which low RAM mirrors but ROM does not. A negative
constant or a run-time signed index is added to the pointer in 16 bits.
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


def test_negative_index_stays_in_the_pointers_bank(e2e):
    """With DBR at $7E, Y = $FFFF would reach bank $7F, which is not a mirror."""
    result = e2e.run('''
        #[ram]
        static mut BUF: [u8; 4];
        #[zeropage(0x10)]
        static mut I: i8;
        #[zeropage(0x20)]
        static mut OUT: [u8; 2];

        #[entry]
        fn main() {
            BUF[0] = 0x51;
            BUF[2] = 0x53;
            I = -3;
            DBR = 0x7E;
            let c: *u8 = &BUF[3] as *u8;
            OUT[0] = c[-1];         // BUF[2]
            OUT[1] = c[I];          // BUF[0]
        }
    ''', ExpectedState(memory={0x20: [0x53, 0x51]}))
    assert result.success, f"Failures: {result.failures}"
