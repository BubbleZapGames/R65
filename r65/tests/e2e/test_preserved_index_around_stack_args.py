# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for a call with stack arguments to a callee that preserves
an index register.

The caller does not spill a register the callee preserves, so whatever the
call's own setup and cleanup use - a register to park the `@ A` argument in
while the stack arguments are pushed, a register to pull them back off into -
must not be that one. Compile-level shapes: tests/compiler/codegen/
test_a_arg_preservation.py.
"""

from r65.tests.e2e import ExpectedState


class TestPreservedIndexAroundStackArgs:
    def test_values(self, e2e):
        result = e2e.run('''
            #[zeropage(0x10)] static mut H: u8;
            #[lowram(0x0300)] static mut OUT: [u16; 7];

            #[preserves(Y)]
            #[inline(never)] fn py(s: u8, a @ A: u8) -> u8 { Y = 2; return a + s; }
            #[preserves(X, Y)]
            #[inline(never)] fn pxy(s: u8, a @ A: u8) -> u8 { X = 1; Y = 2; return a + s; }
            #[inline(never)]
            fn xy(s: u8, a @ A: u8, p @ X: u16, q @ Y: u16) -> u16 {
                return p + q + (s as u16) + (a as u16);
            }
            #[preserves(X)]
            #[inline(never)] fn px(s: u8, t: u8) -> u8 { X = 1; return s + t; }

            #[inline(never)] fn c1(v @ A: u8, w: u8) -> u16 {
                Y = 0x1234; H = py(w, v); return Y;
            }
            #[inline(never)] fn c2(v @ A: u8, w: u8) -> u16 {
                X = 0x3456; Y = 0x2345; H = pxy(w, v); OUT[2] = X; return Y;
            }
            #[inline(never)] fn c3(v @ A: u8, w: u8, p: u16, q: u16) -> u16 {
                return xy(w, v, p, q);
            }
            #[inline(never)] fn c4(w: u8) -> u16 {
                X = 0x4567; H = px(w, 3); return X;
            }

            #[entry]
            fn main() {
                OUT[0] = c1(7, 9);          // $1234
                OUT[1] = H as u16;          // 16
                OUT[3] = c2(7, 9);          // $2345, OUT[2] = $3456
                OUT[4] = c3(7, 9, 1, 2);    // 19
                OUT[5] = c4(9);             // $4567
                OUT[6] = H as u16;          // 12
            }
        ''', ExpectedState(memory={
            0x0300: [0x34, 0x12, 16, 0, 0x56, 0x34, 0x45, 0x23, 19, 0, 0x67, 0x45, 12, 0],
        }))
        assert result.success, f"Failures: {result.failures}"
