# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A commutative op keeps its right operand in A, checked on the hardware.

`select_binary_op` swaps `+ & | ^` when the right operand is in A, so the
slot allocator lets that operand stay there (`_use_reads_a_in_place`).
Each shape here has the left operand somewhere else: a spilled widened
byte, a u8 local on the stack, X, and a second load.
"""

from r65.tests.e2e import ExpectedState


class TestRightOperandInA:

    def test_commutative_ops_read_the_right_operand_from_a(self, e2e):
        result = e2e.run('''
            #[lowram(0x0300)] static mut BUF: [u8; 4];
            #[lowram(0x0304)] static mut AFTER: u8;
            #[zeropage(0xC0)] static mut O1: u16;
            #[zeropage(0xC2)] static mut O2: u16;
            #[zeropage(0xC4)] static mut O3: u8;
            #[zeropage(0xC6)] static mut O4: u16;

            #[inline(never)]
            fn pair(i @ X: u16, j @ Y: u16) -> u16 {
                return (BUF[i] as u16) + (BUF[j] as u16);
            }

            #[inline(never)]
            fn mix() -> u8 {
                let mut s: u8 = 0x5A;
                let mut i: u16 = 0;
                while i < 4 { s = s ^ (BUF[i] & 0x0F); i = i + 1; }
                return s;
            }

            #[inline(never)]
            fn with_x(k @ X: u16) -> u16 {
                let v: u16 = BUF[1] as u16;
                return k + v;
            }

            #[entry]
            fn main() {
                BUF[0] = 0xFF; BUF[1] = 0x81; BUF[2] = 0x07; BUF[3] = 0xFE;
                AFTER = 0xEE;
                O1 = pair(0, 3);
                O3 = mix();
                O2 = with_x(0x1234);
                O4 = (BUF[2] as u16) | (BUF[1] as u16);
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0xFD, 0x01],   # 0xFF + 0xFE
            0x7E00C2: [0xB5, 0x12],   # 0x1234 + 0x81
            0x7E00C4: [0x5D],         # 0x5A ^ 0x0F ^ 0x01 ^ 0x07 ^ 0x0E
            0x7E00C6: [0x87, 0x00],   # 0x07 | 0x81
        }))
        assert result.success, f"Failures: {result.failures}"
