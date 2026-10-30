# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Bytes widened in m16, checked on the hardware.

`ByteWidener` (optimize/byte_widen.py) loads a widened byte with one m16
read and masks it, so the byte after it is read too and must not leak into
the result: each program puts a non-zero byte there. The emitted shape is
pinned in tests/compiler/codegen/test_byte_widen.py.
"""

from r65.tests.e2e import ExpectedState


class TestByteWiden:

    def test_widened_bytes_keep_their_high_byte_clear(self, e2e):
        result = e2e.run('''
            #[lowram(0x0300)] static mut BUF: [u8; 4];
            #[lowram(0x0304)] static mut AFTER: u8;
            #[zeropage(0xC0)] static mut OUT_STATIC: u16;
            #[zeropage(0xC2)] static mut OUT_PTR: u16;
            #[zeropage(0xC4)] static mut OUT_MASK: u16;

            #[inline(always)]
            fn sum_bytes(p: *u8, n: u16) -> u16 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < n {
                    s = s + (p[i] as u16);
                    i = i + 1;
                }
                return s;
            }

            #[inline(never)]
            fn sum_static() -> u16 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < 4 {
                    s = s + (BUF[i] as u16);
                    i = i + 1;
                }
                return s;
            }

            #[inline(never)]
            fn low_bits() -> u16 {
                let mut c: u16 = 0;
                let mut i: u16 = 0;
                while i < 4 {
                    let v: u8 = BUF[i];
                    c = c + ((v & 3) as u16);
                    i = i + 1;
                }
                return c;
            }

            #[entry]
            fn main() {
                BUF[0] = 0xFF;
                BUF[1] = 0x81;
                BUF[2] = 0x07;
                BUF[3] = 0xFE;
                AFTER = 0xEE;
                OUT_STATIC = sum_static();               // 0x0285
                OUT_PTR = sum_bytes(&BUF as *u8, 4);     // 0x0285
                OUT_MASK = low_bits();                   // 3 + 1 + 3 + 2
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0x85, 0x02],
            0x7E00C2: [0x85, 0x02],
            0x7E00C4: [0x09, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"
