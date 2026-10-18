# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for skipping X/Y saves around calls (codegen/clobber_summary.py).

Each case holds a value in an index register across a call. A callee that
never writes the register keeps it unsaved; one that does, an argument
bound to the register, and an `@ A` argument parked across the setup of
the others must still find it intact afterwards.
"""

from r65.tests.e2e import ExpectedState


class TestClobberSummary:
    def test_values(self, e2e):
        result = e2e.run('''
            #[zeropage(0x10)] static mut H: u8;
            #[zeropage(0x11)] static mut T: u8;
            #[lowram(0x0300)] static mut OUT: [u16; 6];

            #[inline(never)] fn clean() { H = H + 1; }
            #[inline(never)] fn dirty() { Y = 7; X = 9; H = 1; }
            #[inline(never)] fn takes_x(v @ X: u16) { H = v as u8; }
            #[inline(never)] fn vb(a @ A: u8, t @ T: u8) { H = a + t; }
            #[inline(never)] fn three() -> u8 { return 3; }

            #[entry]
            fn main() {
                H = 0;
                Y = 0x1234; clean(); OUT[0] = Y;            // $1234
                Y = 0x2345; dirty(); OUT[1] = Y;            // $2345
                X = 0x3456; takes_x(5); OUT[2] = X;         // $3456
                X = 0x4567; clean(); OUT[3] = X;            // $4567
                Y = 0x5678; vb(three(), H); OUT[4] = Y;     // $5678
                OUT[5] = H as u16;                          // 3 + 6
            }
        ''', ExpectedState(memory={
            0x0300: [0x34, 0x12, 0x45, 0x23, 0x56, 0x34, 0x67, 0x45, 0x78, 0x56, 9, 0],
        }))
        assert result.success, f"Failures: {result.failures}"
