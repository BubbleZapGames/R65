# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Counters carried scaled, checked on the hardware.

`IndexStrideReducer` (optimize/index_stride.py) starts a counter read only
as a scaled index (u16 arrays, 3- and 4-byte elements) at its scaled
start, steps it by the element size and tests it against the scaled bound.
These pin a non-zero start, a start already past the bound (no pass), `<=`,
three arrays written through one index, a 4-byte struct array from element
1, and a far-pointer table and 3-byte struct array (stride 3). The shapes are pinned in tests/compiler/optimize/test_index_stride.py.
"""

from r65.tests.e2e import ExpectedState


def test_scaled_counters(e2e):
    result = e2e.run('''
        struct E { pos: u16, hp: u8, kind: u8 }
        #[lowram(0x0340)] static mut POOL: [E; 4];
        #[lowram(0x0350)] static mut AFTER2: u16;
        struct T { tag: u8, val: u16 }
        #[lowram(0x0360)] static mut TRIP: [T; 5];
        #[lowram(0x036F)] static mut AFTER3: u16;
        #[ram] static mut CELLS: [u8; 6];
        #[ram] static mut PTRS: [far *u8; 6];
        #[zeropage(0xC6)] static mut VIA: u16;
        #[lowram(0x0300)] static mut W: [u16; 16];
        #[lowram(0x0320)] static mut A1: [u16; 4];
        #[lowram(0x0328)] static mut A2: [u16; 4];
        #[lowram(0x0330)] static mut A3: [u16; 4];
        #[lowram(0x0338)] static mut AFTER: u16;
        #[zeropage(0xC0)] static mut FROM3: u16;
        #[zeropage(0xC2)] static mut PAST: u16;
        #[zeropage(0xC4)] static mut THROUGH: u16;

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

        #[inline(never)]
        fn tag() {
            let mut i: u16 = 1;
            while i < 4 { POOL[i].pos = 0x1234; POOL[i].kind = 9; i = i + 1; }
        }

        #[inline(never)]
        fn via_ptrs() -> u16 {
            let mut s: u16 = 0;
            let mut i: u16 = 1;
            while i <= 4 { let p: far *u8 = PTRS[i]; s = s + (*p as u16); i = i + 1; }
            return s;
        }

        #[inline(never)]
        fn trip() {
            let mut i: u16 = 1;
            while i < 5 { TRIP[i].tag = 0x77; TRIP[i].val = 0xA1B2; i = i + 1; }
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
            let mut e: u16 = 0;
            while e < 4 { POOL[e].hp = 0x5A; e = e + 1; }
            POOL[0].pos = 0xAAAA;
            POOL[0].kind = 0x66;
            AFTER2 = 0xCAFE;
            tag();                 // elements 1..3; element 0 untouched
            let mut c: u16 = 0;
            while c < 6 { CELLS[c] = (c as u8) * 16; PTRS[c] = &CELLS[c]; c = c + 1; }
            VIA = via_ptrs();      // 0x10 + 0x20 + 0x30 + 0x40
            TRIP[0].tag = 0x11;
            TRIP[0].val = 0x2233;
            AFTER3 = 0xD00D;
            trip();                // elements 1..4; element 0 untouched
        }
    ''', ExpectedState(memory={
        0x7E00C0: [0x2A, 0x07],
        0x7E00C2: [0x55, 0x00],
        0x7E00C4: [0x78, 0x10],
        0x7E0320: [0x11, 0x11] * 4 + [0x22, 0x22] * 4 + [0x00, 0x00] * 4,
        0x7E0338: [0xEF, 0xBE],
        0x7E0340: [0xAA, 0xAA, 0x5A, 0x66] + [0x34, 0x12, 0x5A, 0x09] * 3,
        0x7E0350: [0xFE, 0xCA],
        0x7E00C6: [0xA0, 0x00],
        0x7E0360: [0x11, 0x33, 0x22] + [0x77, 0xB2, 0xA1] * 4 + [0x0D, 0xD0],
    }), max_instructions=50000)
    assert result.success, f"Failures: {result.failures}"
