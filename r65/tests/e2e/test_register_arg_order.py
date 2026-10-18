# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for register arguments whose sources are one another's
targets (ABIModel._order_index_reads / _read_a_source_first).

Register arguments load in rank order - B, X and Y, then A - which did not see
that one argument's source may be another's target: `f(Y, X)` into
`a @ X, b @ Y` passed Y twice. Each case below passes registers that the
same call overwrites.
"""

from r65.tests.e2e import ExpectedState


class TestRegisterArgumentOrder:
    def test_values(self, e2e):
        result = e2e.run('''
            #[lowram(0x0300)] static mut OUT: [u16; 20];
            #[lowram(0x0340)] static mut M: u16;
            #[zeropage(0x60, register)] static mut S0: u16;

            #[inline(never)] fn xy(a @ X: u16, b @ Y: u16, k @ A: u8) {
                OUT[k] = a; OUT[k + 1] = b;
            }
            #[inline(never)] fn ax(a @ A: u16, b @ X: u16) { OUT[10] = a; OUT[11] = b; }
            #[inline(never)] fn axy(a @ A: u16, b @ X: u16, c @ Y: u16) {
                OUT[12] = a; OUT[13] = b; OUT[14] = c;
            }
            #[inline(never)] fn a8x(a @ A: u8, b @ X: u16) { OUT[15] = a as u16; OUT[16] = b; }
            #[inline(never)] fn sy(s: u16, y @ Y: u16) { OUT[17] = s; OUT[18] = y; }
            #[inline(never)] fn abx(a @ A: u8, b @ B: u8, c @ X: u16) {
                OUT[19] = ((a as u16) << 8) | (b as u16); OUT[9] = c;
            }

            #[inline(never)] fn swap()     { X = 0x1111; Y = 0x2222; xy(Y, X, 0); }
            #[inline(never)] fn y_then_m() { Y = 0x2222; xy(Y, M, 2); }
            #[inline(never)] fn m_then_x() { X = 0x1111; xy(M, X, 4); }
            #[inline(never)] fn a_from_x() { X = 0x1111; ax(X, M); }
            #[inline(never)] fn three()    { X = 0x1111; Y = 0x2222; axy(Y, Y, X); }
            #[inline(never)] fn narrow_a() { X = 0x1144; a8x(X as u8, M + 1); }
            #[inline(never)] fn scratch()  { Y = 0x2222; sy(Y, M + 1); }
            #[inline(never)] fn with_b()   { X = 0x0011; abx(5, X as u8, M + 1); }

            #[entry]
            fn main() {
                M = 0x3333;
                swap(); y_then_m(); m_then_x(); a_from_x(); three();
                narrow_a(); scratch(); with_b();
            }
        ''', ExpectedState(memory={
            0x0300: [0x22, 0x22, 0x11, 0x11,      # OUT[0..1]   swap
                     0x22, 0x22, 0x33, 0x33,      # OUT[2..3]   X <- Y, Y <- M
                     0x33, 0x33, 0x11, 0x11,      # OUT[4..5]   X <- M, Y <- X
                     0, 0, 0, 0, 0, 0,            # OUT[6..8]   unused
                     0x34, 0x33,                  # OUT[9]      with_b: X <- M + 1
                     0x11, 0x11, 0x33, 0x33,      # OUT[10..11] A <- X, X <- M
                     0x22, 0x22, 0x22, 0x22, 0x11, 0x11,   # OUT[12..14] A <- Y, X <- Y, Y <- X
                     0x44, 0x00, 0x34, 0x33,      # OUT[15..16] u8 A <- X, X <- M + 1
                     0x22, 0x22, 0x34, 0x33,      # OUT[17..18] scratch <- Y, Y <- M + 1
                     0x11, 0x05],                 # OUT[19]     with_b: B <- X, A = 5
        }))
        assert result.success, f"Failures: {result.failures}"
