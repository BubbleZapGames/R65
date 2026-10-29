# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for `#[clobbers(...)]`.

The `asm!` callees leave the listed registers changed: X written, DBR set
to $7F, D moved to $0100. Each caller must find its own Y and X, and store
through the D and DBR it had, after the call.
"""

from r65.tests.e2e import ExpectedState


def test_clobbered_registers_are_restored_by_the_caller(e2e):
    result = e2e.run('''
        #[zeropage(0x10)] static mut H: u8;
        #[lowram(0x0300)] static mut OUT: [u16; 4];

        #[clobbers(X)]
        fn writes_x() { asm!("LDX #$9999"); }
        #[clobbers(DBR)]
        fn leaves_dbr() { asm!("LDA #$7F", "PHA", "PLB"); }
        #[clobbers(D)]
        fn leaves_d() { asm!("PEA $0100", "PLD"); }

        #[entry]
        fn main() {
            Y = 0x1234; X = 0x4321; writes_x();
            OUT[0] = Y;                                 // $1234, kept unsaved
            OUT[1] = X;                                 // $4321, saved
            leaves_dbr(); OUT[2] = 0x5555;              // $00:0302, not $7F:0302
            leaves_d(); H = 0x66;                       // $0010, not $0110
        }
    ''', ExpectedState(memory={
        0x0300: [0x34, 0x12, 0x21, 0x43, 0x55, 0x55],
        0x0010: 0x66,
    }))
    assert result.success, f"Failures: {result.failures}"
    assert result.cpu.DBR == 0
