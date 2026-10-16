# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for a shift packed into a wider value.

Bug: a shift takes its destination's width (`let w: u16 = n << 2` widens n),
but not when the shift sits under `|`, `&`, `+`, ...: sneslib's
`Color::rgb!(r, g, b)` builds `((b & 0x1F) << 10) | ((g & 0x1F) << 5) | r`,
so with u8 components the blue shift ran in 8 bits and folded to 0, and
green lost its top bits — classickong's palette fade lost all its blue.
"""

from r65.tests.e2e import ExpectedState


class TestShiftPackingWidth:
    def test_rgb_from_u8_components(self, e2e):
        result = e2e.run('''
            struct Color(u16);
            macro_rules! rgb($r:expr, $g:expr, $b:expr) {
                { Color(((($b) & 0x1F) << 10) | ((($g) & 0x1F) << 5) | (($r) & 0x1F)) }
            }

            #[zeropage(0x30)] static mut R: u8;
            #[zeropage(0x31)] static mut G: u8;
            #[zeropage(0x32)] static mut BL: u8;
            #[zeropage(0x34)] static mut OUT: Color;

            #[entry]
            fn main() {
                R = 3; G = 10; BL = 20;
                OUT = rgb!(R, G, BL);
            }
        ''', ExpectedState(memory={
            0x7E0034: [0x43, 0x51],     # (20 << 10) | (10 << 5) | 3 = 0x5143
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_high_byte_plus_low_byte(self, e2e):
        result = e2e.run('''
            #[zeropage(0x30)] static mut HI: u8;
            #[zeropage(0x31)] static mut LO: u8;
            #[zeropage(0x32)] static mut OUT: u16;

            #[entry]
            fn main() {
                HI = 0x12; LO = 0x34;
                OUT = (HI << 8) + LO;
            }
        ''', ExpectedState(memory={
            0x7E0032: [0x34, 0x12],
        }))
        assert result.success, f"Failures: {result.failures}"
