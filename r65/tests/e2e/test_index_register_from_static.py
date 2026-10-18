# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for `X = G` / `Y = G` straight from a static.

The direct load into an index register was typed by the register (u16), and
codegen then judged the source's width by its location, which says one byte
for any static: a u16 lost its high byte (G = $1234 gave $0034). The load
now has the static's width - a u16 read whole, a u8 zero-extended.
"""

from r65.tests.e2e import ExpectedState


class TestIndexRegisterFromStatic:
    def test_values(self, e2e):
        result = e2e.run('''
            #[lowram(0x0300)] static mut OUT: [u16; 6];
            #[zeropage(0x10)] static mut GZ: u16;
            #[lowram(0x0320)] static mut GL: u16;
            #[ram] static mut GR: u16;
            #[zeropage(0x12)] static mut HZ: u8;
            #[zeropage(0x13)] static mut JUNK: u8;
            #[ram] static mut HR: u8;

            #[inline(never)] fn go() {
                Y = GZ; OUT[0] = Y;            // $1234
                X = GL; OUT[1] = X;            // $5678
                Y = GR; OUT[2] = Y;            // $9ABC
                X = HZ; OUT[3] = X;            // $0012, not JUNK's byte
                Y = HR; OUT[4] = Y;            // $0034
                X = GZ; OUT[5] = X;            // $1234
            }

            #[entry]
            fn main() {
                GZ = 0x1234; GL = 0x5678; GR = 0x9ABC;
                HZ = 0x12; JUNK = 0xEE; HR = 0x34;
                go();
            }
        ''', ExpectedState(memory={
            0x0300: [0x34, 0x12, 0x78, 0x56, 0xBC, 0x9A,
                     0x12, 0x00, 0x34, 0x00, 0x34, 0x12],
        }))
        assert result.success, f"Failures: {result.failures}"
