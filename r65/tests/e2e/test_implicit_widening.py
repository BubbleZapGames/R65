# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A one-byte value stored into a two-byte place is extended, as `as` does.

The type checker lets a `u8` flow into a `u16`, but only an operator's operands
were converted. Everywhere else the u8 was loaded in m8 and stored after
`REP #$20`, so whatever B held became the high byte. B is set to 0xFF before
each case so a stale high byte shows.
"""

from r65.tests.e2e import ExpectedState


def test_every_assignment_shaped_site_extends(e2e):
    result = e2e.run('''
        #[zeropage(0x20)] static mut I: u8;
        #[zeropage(0x21)] static mut N: i8;
        #[zeropage(0x22)] static mut P: i8;
        #[zeropage(0x60)] static mut O: [u16; 12];
        #[zeropage(0x80)] static mut SO: [i16; 2];

        struct Pair { w: u16 }
        struct Wide(u16);

        #[inline(never)]
        fn ret() -> u16 { B = 0xFF; return I; }
        #[inline(never)]
        fn via_a(v @ A: u16) -> u16 { return v; }
        #[inline(never)]
        fn via_x(v @ X: u16) -> u16 { return X; }

        #[entry]
        fn main() {
            I = 0x05; N = -3; P = 5;
            B = 0xFF; O[0] = I;
            B = 0xFF; let l: u16 = I; O[1] = l;
            O[2] = ret();
            B = 0xFF; O[3] = via_a(I);
            B = 0xFF; O[4] = via_x(I);
            B = 0xFF; X = I; O[5] = X;
            B = 0xFF; let p = Pair { w: I }; O[6] = p.w;
            B = 0xFF; let w: Wide = I; O[7] = w.0;
            B = 0xFF; let v = Wide(I); O[8] = v.0;
            B = 0xFF; SO[0] = N;
            B = 0xFF; SO[1] = P;
            B = 0xFF; O[9] = via_x(I + 1);
            B = 0xFF; O[10] = via_a(I + 1);
            B = 0xFF; O[11] = (I + 1) as u16;
        }
    ''', ExpectedState(memory={
        0x7E0060: [0x05, 0x00] * 9 + [0x06, 0x00] * 3,
        0x7E0080: [0xFD, 0xFF, 0x05, 0x00],
    }))
    assert result.success, f"Failures: {result.failures}"
