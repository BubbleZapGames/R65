# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A `databank=inline` function whose callers all share its bank.

`far_to_near` demotes such a function to JSR/RTS, and the DBR prologue
(PHB / set DBR to the function's bank / PLB) is only emitted for a far
function, so the demoted one runs with its caller's DBR. The compiler's own
static and `match`-table reads go long once the prologue is gone, but a
near pointer is DBR-relative by definition: `p[i]` below reads bank 0 at
TABLE's offset instead of bank 2.

The hardware write keeps the prologue from being skipped as unneeded, so
the same function left far (called from bank 0 too) reads bank 2.
"""

from r65.tests.e2e import ExpectedState


def test_demoted_databank_inline_function_keeps_its_dbr(e2e):
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut OUT: [u8; 2];
        #[hw(0x4202)]
        static mut WRMPYA: u8;

        #[bank(2)]

        static TABLE: [u8; 2] = [0xA5, 0x5A];

        #[inline(never)]
        #[mode(databank=inline)]
        fn read_near(i: u16) -> u8 {
            WRMPYA = 0;
            let p: *u8 = &TABLE as *u8;
            return p[i];
        }

        // The only caller, in read_near's bank: far_to_near demotes it.
        #[inline(never)]
        fn collect() {
            OUT[0] = read_near(0);
            OUT[1] = read_near(1);
        }

        #[bank(0)]

        #[entry]
        fn main() { collect(); }
    ''', ExpectedState(memory={0x7E0010: [0xA5, 0x5A]}))
    assert result.success, f"Failures: {result.failures}"

