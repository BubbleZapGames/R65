# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for `!` and bare-bool conditions (mir/lowerers/condition.py).

`!e` lowers by swapping the branch targets and a bare scalar as `e != 0`;
each case records which way a branch went for both truth values.
"""

from r65.tests.e2e import ExpectedState


class TestNegatedCondition:
    def test_values(self, e2e):
        result = e2e.run('''
            #[zeropage(0x10)] static mut T: bool;
            #[zeropage(0x11)] static mut F: bool;
            #[zeropage(0x12)] static mut N: u8;
            #[zeropage(0x13)] static mut M: u8;
            #[zeropage(0x14)] static mut W: u16;
            #[lowram(0x0300)] static mut OUT: [u8; 12];

            fn is_three() -> bool { return N == 3; }

            #[entry]
            fn main() {
                T = true;
                F = false;
                N = 3;
                M = 5;
                W = 0x0100;
                OUT[0] = 0; OUT[1] = 0; OUT[2] = 0; OUT[3] = 0;
                OUT[4] = 0; OUT[5] = 0; OUT[6] = 0; OUT[7] = 0;
                OUT[8] = 0; OUT[9] = 0;
                if !T { OUT[0] = 1; }                   // 0
                if !F { OUT[1] = 1; }                   // 1
                if !F && !F { OUT[2] = 1; }             // 1
                if !F && !T { OUT[3] = 1; }             // 0
                if !T || F { OUT[4] = 1; }              // 0
                if !(N < M) { OUT[5] = 1; }             // 0
                if !(M < N) { OUT[6] = 1; }             // 1
                if !is_three() { OUT[7] = 1; }          // 0
                if W != 0 && T { OUT[8] = 1; }          // 1
                let mut done = false;
                let mut passes: u8 = 0;
                while !done {
                    passes++;
                    if passes == 4 { done = true; }
                }
                OUT[9] = passes;                        // 4
                OUT[10] = 0;
                if !!T { OUT[10] = 1; }                 // 1
                OUT[11] = 0;
                if T { OUT[11] = 1; }                   // 1
            }
        ''', ExpectedState(memory={
            0x0300: [0, 1, 1, 0, 0, 0, 1, 0, 1, 4, 1, 1],
        }))
        assert result.success, f"Failures: {result.failures}"
