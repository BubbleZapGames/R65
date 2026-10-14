# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`(x & mask) != 0` with a mask of the top two bits of a width becomes BIT and
a branch on N or V.

0x80 / 0x40 test bits 7 / 6 of x's low byte, whatever x's width: in m16 BIT
sets N from bit 15, so a u16 value tested while A was 16-bit always read as
clear and F32's multiply never negated a negative mantissa. On a u16,
0x8000 / 0x4000 test bits 15 / 14 the same way, in m16.
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


def test_top_bits_of_a_u16(e2e):
    """0x8000 and 0x4000 on a u16 test bits 15 and 14 (BIT in m16: N and V)."""
    result = e2e.run('''
        #[lowram(0x0300)] static mut WORD: u16;
        #[lowram(0x0310)] static mut OUT: [u8; 5];

        #[inline(never)]
        fn probe(k @ A: u16) {
            let v: u16 = k + WORD;
            if (WORD & 0x8000) != 0 { OUT[0] = 1; } else { OUT[0] = 2; }
            if (WORD & 0x4000) != 0 { OUT[1] = 1; } else { OUT[1] = 2; }
            if (WORD & 0x4000) == 0 { OUT[2] = 1; } else { OUT[2] = 2; }
            if (v & 0x8000) != 0 { OUT[3] = 1; } else { OUT[3] = 2; }
            if (v & 0x4000) != 0 { OUT[4] = 1; } else { OUT[4] = 2; }
        }

        #[entry]
        fn main() { WORD = 0x8080; probe(0x4000); }
    ''', ExpectedState(memory={0x7E0310: [1, 2, 1, 1, 1]}))
    assert result.success, f"Failures: {result.failures}"


def test_bit6_of_a_computed_byte(e2e):
    """Only BIT on memory sets V; a computed value's bit 6 is tested by AND."""
    result = e2e.run('''
        #[lowram(0x0300)] static mut BYTE: u8;
        #[lowram(0x0310)] static mut OUT: u8;

        #[inline(never)]
        fn probe(p: far *u8) {
            let v: u8 = *p | 0x01;
            if (v & 0x40) != 0 { OUT = 1; } else { OUT = 2; }
        }

        #[entry]
        fn main() { BYTE = 0x40; probe(&BYTE); }
    ''', ExpectedState(memory={0x7E0310: [1]}))
    assert result.success, f"Failures: {result.failures}"


def test_inlined_constant_is_not_bit_tested_as_an_immediate(e2e):
    """BIT #imm sets only Z, so the inliner must not put a constant argument
    into a BitTest: `BIT #$90` would leave N from whatever ran before."""
    result = e2e.run('''
        #[lowram(0x0310)] static mut OUT: [u8; 2];

        #[inline(always)]
        fn sign8(v: u8) -> u8 { if (v & 0x80) != 0 { return 1; } return 2; }

        #[inline(always)]
        fn sign16(v: u16) -> u8 { if (v & 0x8000) != 0 { return 1; } return 2; }

        #[entry]
        fn main() { A = 0; OUT[0] = sign8(0x90); A = 0; OUT[1] = sign16(0x9000); }
    ''', ExpectedState(memory={0x7E0310: [1, 1]}))
    assert result.success, f"Failures: {result.failures}"


def test_top_bits_of_an_index_register(e2e):
    """A value in X or Y tests bit 15 by CPX/CPY #0 and bit 7 through TXA/TYA."""
    result = e2e.run('''
        #[lowram(0x0310)] static mut OUT: [u8; 4];

        #[inline(never)]
        fn probe(x @ X: u16, y @ Y: u16) {
            if (x & 0x8000) != 0 { OUT[0] = 1; } else { OUT[0] = 2; }
            if (x & 0x80) != 0 { OUT[1] = 1; } else { OUT[1] = 2; }
            if (y & 0x8000) != 0 { OUT[2] = 1; } else { OUT[2] = 2; }
            if (y & 0x80) != 0 { OUT[3] = 1; } else { OUT[3] = 2; }
        }

        #[entry]
        fn main() { probe(0x8001, 0x0080); }
    ''', ExpectedState(memory={0x7E0310: [1, 2, 2, 1]}))
    assert result.success, f"Failures: {result.failures}"
