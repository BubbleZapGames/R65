# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A fieldless enum with a variant above 255 is two bytes, end to end.

`size_bytes` already said 2, but codegen sized loads, stores and compares by
type *name* (`u16`/`i16`), so a `Wide` was moved in m8: `W = Wide::C` stored
`$2C`, and a match on it could not tell `B` (256) from `A` (0).
"""

from r65.tests.e2e import ExpectedState


def test_a_two_byte_enum_stores_matches_and_converts(e2e):
    result = e2e.run('''
        enum Wide { A = 0, B = 256, C = 300 }

        impl Wide {
            #[inline(never)]
            fn code(self) -> u8 {
                match self { Wide::A => 1, Wide::B => 2, Wide::C => 3 }
            }
        }

        #[zeropage(0x10)] static mut W: Wide;
        #[zeropage(0x12)] static mut O: [u8; 6];
        #[zeropage(0x18)] static mut W16: u16;
        #[zeropage(0x1A)] static mut B8: u8;

        #[entry]
        fn main() {
            W = Wide::B;
            match W { Wide::A => O[0] = 1, Wide::B => O[0] = 2, Wide::C => O[0] = 3 };
            O[1] = W.code();
            O[2] = Wide::C.code();
            W16 = Wide::C as u16;
            O[3] = W as u8;
            B8 = 0;
            W = B8 as Wide;
            O[4] = W.code();
            let w: Wide = Wide::C;
            O[5] = w.code();
        }
    ''', ExpectedState(memory={0x7E0012: [2, 2, 3, 0x00, 1, 3],
                               0x7E0018: [0x2C, 0x01]}))
    assert result.success, f"Failures: {result.failures}"
