# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A static read in a loop is read again on every pass.

The MIR builder keeps a static's last load in a vreg and reuses it until a
store, a call or a merge (`symbol_to_vreg`). A loop header is a merge with
its back edge: `G` loaded before the loop was reused inside it, so
`G = G + 2` stored the first value + 2 each pass. A load made in the body
is not on the path that skips it either.
"""

from r65.tests.e2e import ExpectedState


class TestLoopStaticReload:

    def test_static_reloaded_across_the_back_edge_and_after_the_loop(self, e2e):
        result = e2e.run('''
            #[zeropage(0x40)] static mut G: u8;
            #[zeropage(0x41)] static mut H: u8;
            #[zeropage(0x42)] static mut OUT: u16;
            #[zeropage(0x44)] static mut SEEN: u8;
            #[zeropage(0x45)] static mut N: u8;

            #[inline(never)]
            fn bump() -> u16 {
                let mut i: u16 = G as u16;
                while i < 8 {
                    G = G + 2;
                    i = i + 1;
                }
                return i;
            }

            #[inline(never)]
            fn skipped() {
                let mut k: u8 = 0;
                while k < N {
                    SEEN = H;
                    k = k + 1;
                }
                SEEN = SEEN + H;
            }

            #[entry]
            fn main() {
                G = 5;
                OUT = bump();     // three passes: G = 5 + 6
                H = 9;
                N = 0;
                SEEN = 1;
                skipped();        // no pass: SEEN = 1 + 9
            }
        ''', ExpectedState(memory={
            0x7E0040: [11],
            0x7E0042: [8, 0],
            0x7E0044: [10],
        }))
        assert result.success, f"Failures: {result.failures}"
