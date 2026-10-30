# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for folding `SEC; SBC #1` into `DEC A` ahead of a V read.

Bug: the peephole folded the subtract of a signed `x <= 1` into `DEC A`
after checking only that the carry was dead. DEC sets N and Z but not V,
so the compare's `BVS` read whatever an earlier ADC left there, and after
an overflowing add `5 <= 1` came out true. classickong's enemy_process
had two such compares (`dx <= 1`, `dx > 1`).
"""

from r65.tests.e2e import ExpectedState


class TestDecFoldOverflow:
    def test_signed_compare_after_overflowing_add(self, e2e):
        result = e2e.run('''
            #[zeropage(0xB0)] static mut A8: i8;
            #[zeropage(0xB1)] static mut B8: i8;
            #[zeropage(0xB2)] static mut S8: i8;
            #[zeropage(0xB3)] static mut DX: i8;
            #[zeropage(0xB4)] static mut OUT: u8;

            fn check() {
                S8 = A8 + B8;       // 0x7F + 1 overflows: V = 1
                if DX <= 1 { OUT = 1; } else { OUT = 2; }
            }

            #[entry]
            fn main() {
                A8 = 0x7F;
                B8 = 1;
                DX = 5;
                check();
            }
        ''', ExpectedState(memory={
            0x7E00B4: [2],      # 5 <= 1 is false
        }))
        assert result.success, f"Failures: {result.failures}"
