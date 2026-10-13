# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
An immutable scalar static is ROM data, read in place.

Arrays, strings and struct literals in ROM got a data label, but a scalar
got nothing: no bytes were emitted, the symbol had no `rom_label`, and a read
fell through to the RAM allocator, which answered address 0. So
`static W: u16 = 0x1234; O = W;` loaded `$00`, and a `fn()` static jumped
wherever `$00` pointed.
"""

from r65.tests.e2e import ExpectedState


def test_scalar_rom_statics_read_their_values(e2e):
    result = e2e.run('''
        enum Dir { North, East, South }
        const BASE: u16 = 0x100;

        static W: u16 = 0x1234;
        static BT: u8 = 0x56;
        static SG: i16 = -2;
        static FLAG: bool = true;
        static HEADING: Dir = Dir::South;
        static K: u16 = BASE * 2 + 3;
        static HP: fn() = hit;

        #[zeropage(0x10)] static mut O: [u8; 12];
        #[zeropage(0x20)] static mut P: *u16;

        fn hit() { O[11] = 0x77; }

        #[entry]
        fn main() {
            O[0] = W as u8;
            O[1] = (W >> 8) as u8;
            O[2] = BT;
            let s: i16 = SG;
            O[3] = s as u8;
            O[4] = (s >> 8) as u8;
            if FLAG { O[5] = 1; }
            O[6] = HEADING as u8;
            O[7] = K as u8;
            O[8] = (K >> 8) as u8;
            P = &W;
            let v: u16 = *P;
            O[9] = v as u8;
            O[10] = (v >> 8) as u8;
            HP();
        }
    ''', ExpectedState(memory={0x7E0010: [0x34, 0x12, 0x56, 0xFE, 0xFF, 1,
                                          2, 0x03, 0x02, 0x34, 0x12, 0x77]}))
    assert result.success, f"Failures: {result.failures}"
