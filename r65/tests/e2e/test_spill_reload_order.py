# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for X and Y saved across calls in regions that do not nest.

Each register's clobber region is found on its own, so one may open inside
another and outlive it. The reload must still take each register's own bytes
off the stack. Compile-level shape: tests/compiler/codegen/test_hw_spilling.py.
"""

from r65.tests.e2e import ExpectedState


class TestSpillReloadOrder:
    def test_regions_that_do_not_nest(self, e2e):
        result = e2e.run('''
            #[lowram(0x0300)] static mut OUT: [u16; 6];
            #[inline(never)] fn f() { X = 7; Y = 7; }
            #[inline(never)] fn y_only() { Y = 5; }
            #[inline(never)] fn g() { X = 8; Y = 8; }
            #[inline(never)] fn h() { X = 9; Y = 9; }

            #[inline(never)] fn x_then_y() {
                X = 0x1111;
                f();
                Y = 0x2222;
                g();
                OUT[0] = X;                 // $1111
                h();
                OUT[1] = Y;                 // $2222
            }
            #[inline(never)] fn y_then_x() {
                Y = 0x3333;
                f();
                X = 0x4444;
                g();
                OUT[2] = Y;                 // $3333
                h();
                OUT[3] = X;                 // $4444
            }

            // Both regions open at f(); X's ends there, Y's at y_only().
            #[inline(never)] fn same_start() {
                X = 0x5555;
                Y = 0x6666;
                f();
                y_only();
                OUT[4] = X;                 // $5555
                OUT[5] = Y;                 // $6666
            }

            #[entry]
            fn main() { x_then_y(); y_then_x(); same_start(); }
        ''', ExpectedState(memory={
            0x0300: [0x11, 0x11, 0x22, 0x22, 0x33, 0x33, 0x44, 0x44,
                     0x55, 0x55, 0x66, 0x66],
        }))
        assert result.success, f"Failures: {result.failures}"
