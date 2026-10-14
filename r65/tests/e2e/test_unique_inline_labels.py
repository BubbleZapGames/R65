# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for inline branch labels that are not unique.

Bug: an `i8 as i16` widen (and `!x`) branched to WLA-DX anonymous labels,
`+` and `++`. The assembler resolves each `+` to the next `+:`, but the
compiler's passes key labels by name: the mode dataflow took the first `+:`
in the program for every one of them. The real `+:` of a later widen then
had no predecessor, so it and everything after it in the function looked
unreachable, and the mode-directive rewrite dropped the `.ACCU 16` after
the next REP #$20 and kept stale `.ACCU 8` anchors. An immediate assembled
for 8-bit A then ran in 16-bit A and the CPU fell into the next
instruction's bytes.
"""

from r65.tests.e2e import ExpectedState


class TestUniqueInlineLabels:

    def test_second_widen_keeps_16bit_immediates(self, e2e):
        """Two widens in the program, then 16-bit adds behind branches."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[lowram] static mut offs: [i8; 2];
            #[lowram] static mut base: i16;
            #[lowram] static mut first: i16;
            #[lowram] static mut result: i16;

            #[inline(never)]
            fn widen_first() {
                first = base + (offs[0] as i16);
            }

            #[inline(never)]
            fn widen_then_add(k: u8) {
                let mut v: i16 = base + (offs[1] as i16);
                if k != 0 {
                    if v < 128 {
                        v = v + 6;
                    }
                }
                if v >= 0x2D {
                    v = v + 3;
                }
                result = v;
            }

            #[entry]
            fn main() {
                offs[0] = -8;
                offs[1] = -8;
                base = 100;
                widen_first();
                widen_then_add(1);
            }
        ''', ExpectedState(memory={
            0x7E0204: [92, 0],     # first  = 100 - 8
            0x7E0206: [101, 0],    # result = 100 - 8 + 6 + 3
        }))
        assert result.success, f"Failures: {result.failures}"
