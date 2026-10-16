# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for the precedence of a macro's `expr` arguments.

Bug: an `$e:expr` argument was pasted into the body token for token, so
`DBL!(FY + 1)` with body `$e * 2` computed `FY + 2`, and classickong's
`POS!(x, barrel_fire_y >> 3)` with body `($y as u16) << 5` became
`y >> (3 as u16)`, shifted left in 8 bits, and lost the row's high bits.
An argument of more than one token is now substituted in parentheses.
"""

from r65.tests.e2e import ExpectedState


class TestMacroExprPrecedence:
    def test_binary_argument_keeps_its_grouping(self, e2e):
        result = e2e.run('''
            macro_rules! DBL($e:expr) { $e * 2 }

            #[zeropage(0x30)] static mut FY: u8;
            #[zeropage(0x31)] static mut OUT: u8;

            #[entry]
            fn main() {
                FY = 5;
                OUT = DBL!(FY + 1);
            }
        ''', ExpectedState(memory={
            0x7E0031: [12],     # (5 + 1) * 2, not 5 + 1 * 2
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_pos_row_from_shifted_y(self, e2e):
        """classickong's POS! with a shifted row: y = 200 is row 25, offset 800."""
        result = e2e.run('''
            macro_rules! POS($x:expr, $y:expr) { ((($y as u16) << 5) + ($x as u16)) }

            #[zeropage(0x30)] static mut FX: u8;
            #[zeropage(0x31)] static mut FY: u8;
            #[zeropage(0x32)] static mut OUT: u16;

            #[entry]
            fn main() {
                FX = 24;
                FY = 200;
                OUT = POS!(FX >> 3, FY >> 3);
            }
        ''', ExpectedState(memory={
            0x7E0032: [0x23, 0x03],     # 25 * 32 + 3 = 803
        }))
        assert result.success, f"Failures: {result.failures}"
