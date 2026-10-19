# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Loads and stores through a pointer known to hold one address.

A pointer whose one def is `&STATIC` or a number is read and written with
absolute addressing (`LDA BUF,Y`, memory_select._const_pointer), a u16 as one
m16 access that leaves Y - here the loop counter - alone. A pointer written
again (`p = p + 1`) is not constant: it used to be taken for its first value,
so every pass read BUF[0].
"""

from r65.tests.e2e import ExpectedState


class TestConstPointer:

    def test_constant_and_walking_pointers(self, e2e):
        result = e2e.run('''
            #[lowram(0x0300)] static mut BUF: [u8; 4];
            #[lowram(0x0310)] static mut WORDS: [u16; 4];
            #[lowram(0x0318)] static mut AFTER: u16;
            #[zeropage(0x40)] static mut WALKED: u16;
            #[zeropage(0x42)] static mut SUMMED: u16;
            #[zeropage(0x44)] static mut WORDSUM: u16;

            #[inline(never)]
            fn walk() -> u16 {
                let mut p: *u8 = 0x0300 as *u8;
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < 4 {
                    s = s + (*p as u16);
                    p = p + 1;
                    i = i + 1;
                }
                return s;
            }

            #[inline(always)]
            fn sum_bytes(p: *u8, n: u16) -> u16 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < n { s = s + (p[i] as u16); i = i + 1; }
                return s;
            }

            #[inline(always)]
            fn fill_words(p: *u16, n: u16) {
                let mut i: u16 = 0;
                while i < n { p[i] = 0x1100 + i; i = i + 1; }
            }

            #[inline(always)]
            fn sum_words(p: *u16, n: u16) -> u16 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < n { s = s + p[i]; i = i + 1; }
                return s;
            }

            #[inline(never)]
            fn run() {
                SUMMED = sum_bytes(&BUF as *u8, 4);
                fill_words(&WORDS as *u16, 4);
                WORDSUM = sum_words(&WORDS as *u16, 4);
            }

            #[entry]
            fn main() {
                BUF[0] = 1; BUF[1] = 2; BUF[2] = 3; BUF[3] = 0xF4;
                AFTER = 0xBEEF;
                WALKED = walk();     // 1 + 2 + 3 + 0xF4
                run();
            }
        ''', ExpectedState(memory={
            0x7E0040: [0xFA, 0x00],
            0x7E0042: [0xFA, 0x00],
            0x7E0044: [0x06, 0x44],                 # 4 * 0x1100 + 0 + 1 + 2 + 3
            0x7E0310: [0x00, 0x11, 0x01, 0x11, 0x02, 0x11, 0x03, 0x11],
            0x7E0318: [0xEF, 0xBE],                 # past the end: untouched
        }), max_instructions=50000)
        assert result.success, f"Failures: {result.failures}"
