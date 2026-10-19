# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Counters carried doubled, checked on the hardware.

`IndexStrideReducer` (optimize/index_stride.py) starts a counter read only
as a doubled u16 index at twice its start, steps it by 2 and tests it
against twice the bound. These pin a non-zero start, a start already past
the bound (no pass), `<=`, and three arrays written through one index. The
shapes are pinned in tests/compiler/optimize/test_index_stride.py.
"""

from r65.tests.e2e import ExpectedState


def test_doubled_counters(e2e):
    result = e2e.run('''
        #[lowram(0x0300)] static mut W: [u16; 16];
        #[lowram(0x0320)] static mut A1: [u16; 4];
        #[lowram(0x0328)] static mut A2: [u16; 4];
        #[lowram(0x0330)] static mut A3: [u16; 4];
        #[lowram(0x0338)] static mut AFTER: u16;
        #[zeropage(0x40)] static mut FROM3: u16;
        #[zeropage(0x42)] static mut PAST: u16;
        #[zeropage(0x44)] static mut THROUGH: u16;

        #[inline(never)]
        fn sum_from(start: u16) -> u16 {
            let mut s: u16 = 0;
            let mut i: u16 = 3;
            while i < 10 { s = s + W[i]; i = i + 1; }
            return s + start;
        }

        #[inline(never)]
        fn none() -> u16 {
            let mut s: u16 = 0x55;
            let mut i: u16 = 12;
            while i < 10 { s = s + W[i]; i = i + 1; }
            return s;
        }

        #[inline(never)]
        fn through() -> u16 {
            let mut s: u16 = 0;
            let mut i: u16 = 0;
            while i <= 15 { s = s + W[i]; i = i + 1; }
            return s;
        }

        #[inline(never)]
        fn fill() {
            let mut i: u16 = 0;
            while i < 4 {
                A1[i] = 0x1111;
                A2[i] = 0x2222;
                A3[i] = 0;
                i = i + 1;
            }
        }

        #[entry]
        fn main() {
            let mut k: u16 = 0;
            while k < 16 { W[k] = 0x0100 + k; k = k + 1; }
            AFTER = 0xBEEF;
            FROM3 = sum_from(0);   // 7 * 0x100 + (3 + ... + 9)
            PAST = none();         // no pass
            THROUGH = through();   // 16 * 0x100 + (0 + ... + 15)
            fill();
        }
    ''', ExpectedState(memory={
        0x7E0040: [0x2A, 0x07],
        0x7E0042: [0x55, 0x00],
        0x7E0044: [0x78, 0x10],
        0x7E0320: [0x11, 0x11] * 4 + [0x22, 0x22] * 4 + [0x00, 0x00] * 4,
        0x7E0338: [0xEF, 0xBE],
    }), max_instructions=50000)
    assert result.success, f"Failures: {result.failures}"
