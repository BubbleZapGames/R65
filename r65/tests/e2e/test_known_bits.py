# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for known-bits simplification (optimize/known_bits.py).

The pass drops a mask when the bits it clears are known zero or no reader
looks at them. Each case leaves junk where a dropped mask would have
cleaned it - a stale high byte from an earlier 16-bit value - so a mask
dropped wrongly shows in the result.
"""

from r65.tests.e2e import ExpectedState


class TestKnownBits:
    def test_values(self, e2e):
        result = e2e.run('''
            #[zeropage(0x10)] static mut BB: u8;
            #[zeropage(0x12)] static mut W: u16;
            #[zeropage(0x14)] static mut JUNK: u16;
            #[lowram(0x0300)] static mut OUT: [u16; 8];
            #[lowram(0x0320)] static mut OUT8: [u8; 4];
            #[lowram(0x0400)] static mut TBL: [u8; 256];

            #[entry]
            fn main() {
                TBL[200] = 0x5A;
                TBL[0x34] = 0x77;
                BB = 0x9C;
                JUNK = 0xABCD;
                W = JUNK;                                   // B now holds $AB
                OUT[0] = (BB as u16) << 8;                  // $9C00
                W = 0x0180;
                OUT[1] = ((W & 0x1FF) >> 8) & 1;            // 1
                W = 0x0080;
                OUT[2] = ((W & 0x1FF) >> 8) & 1;            // 0
                W = 0x7FFF;
                OUT[3] = (W & 0x7C00) >> 10;                // 31
                W = 0xFF7C;
                OUT[4] = ((W >> 2) & 31) << 8;              // $1F00
                OUT[5] = ((BB as u16) >> 8) & 1;            // 0
                BB = 200;
                let x: u16 = BB as u16;
                OUT8[0] = TBL[(x as u8) as u16];            // TBL[200]
                W = 0x1234;
                let s: u16 = W;
                OUT8[1] = TBL[(s as u8) as u16];            // TBL[$34], not $1234
                OUT8[2] = (s & 0xFF) as u8;                 // $34
                OUT8[3] = ((W + 0x0100) >> 8) as u8;        // $13
            }
        ''', ExpectedState(memory={
            0x0300: [0x00, 0x9C, 1, 0, 0, 0, 31, 0, 0x00, 0x1F, 0, 0],
            0x0320: [0x5A, 0x77, 0x34, 0x13],
        }))
        assert result.success, f"Failures: {result.failures}"
