# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`(x & 0x80) != 0` tests bit 7 of x's low byte, whatever x's width.

It lowers to BIT (or AND #$80) and a BMI/BPL. In m16 those set N from bit
15, so a u16 value tested while A was 16-bit always read as clear: F32's
multiply never negated a negative mantissa.
"""

from r65.tests.e2e import ExpectedState


def test_bit7_of_a_u16(e2e):
    result = e2e.run('''
        #[lowram(0x0300)] static mut WORD: u16;
        #[lowram(0x0310)] static mut LOCAL: u8;
        #[lowram(0x0311)] static mut STATIC: u8;

        #[inline(never)]
        fn probe(p: far *u16) {
            let v: u16 = *p & 0xFF;
            if (v & 0x80) != 0 { LOCAL = 1; } else { LOCAL = 2; }
            if (WORD & 0x80) != 0 { STATIC = 1; } else { STATIC = 2; }
        }

        #[entry]
        fn main() { WORD = 0x00C0; probe(&WORD); }
    ''', ExpectedState(memory={0x7E0310: [1, 1]}))
    assert result.success, f"Failures: {result.failures}"
