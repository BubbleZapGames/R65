# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A block move keeps DBR.

MVN sets DBR to its destination bank. A zero fill over 32 bytes and an array
literal both copy with MVN, and neither restored DBR: after `__init_start`
filled a `#[ram]` array, DBR was $7E, so every absolute write that followed -
`NMITIMEN`, `INIDISP` in snes_init - landed in WRAM instead of the hardware
register, and `wait_nmi`'s read of $4210 spun forever on a black screen. Both
now save DBR around the move (PHB/PLB), as the clone copy already did.
"""

from r65.tests.e2e import ExpectedState


def test_a_large_zero_fill_keeps_dbr(e2e):
    result = e2e.run('''
        #[ram]
        static mut BUF: [u8; 64] = [0; 64];
        #[zeropage(0x10)]
        static mut LAST: u8;

        #[entry]
        fn main() {
            LAST = BUF[63];
        }
    ''', ExpectedState(memory={0x7E0010: 0}))
    assert result.success, f"Failures: {result.failures}"
    assert result.cpu.DBR == 0


def test_an_array_literal_initialiser_keeps_dbr(e2e):
    result = e2e.run('''
        #[ram]
        static mut TBL: [u8; 4] = [1, 2, 3, 4];
        #[zeropage(0x10)]
        static mut LAST: u8;

        #[entry]
        fn main() {
            LAST = TBL[3];
        }
    ''', ExpectedState(memory={0x7E0010: 4}))
    assert result.success, f"Failures: {result.failures}"
    assert result.cpu.DBR == 0


def test_a_local_array_literal_keeps_dbr(e2e):
    """A local array literal is the same ROM-to-RAM copy, in an ordinary
    function. Its destination here is low RAM, so it used to reset DBR to $00
    - whatever bank the caller had set (a far function's own, say)."""
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut LAST: u8;

        fn last() -> u8 {
            let a: [u8; 4] = [5, 6, 7, 8];
            return a[3];
        }

        #[entry]
        fn main() {
            asm!("LDA #$7E", "PHA", "PLB");
            LAST = last();
        }
    ''', ExpectedState(memory={0x7E0010: 8}))
    assert result.success, f"Failures: {result.failures}"
    assert result.cpu.DBR == 0x7E
