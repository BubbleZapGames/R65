# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Values with several defs kept in A, checked on the hardware.

The slot allocator keeps such a value in A when nothing it is live across
touches A (`_find_a_multi_def_vregs`). These pin a u16 sum, a u8 sum that
wraps, the value stored every pass, a loop that runs no pass, a sum over a
nested loop (the outer counter in Y, the inner in X), and a byte clamp and
a word clamp into a table index, each on both sides of its limit. The
shapes are pinned in tests/compiler/codegen/test_loop_carried_a.py.
"""

from r65.tests.e2e import ExpectedState


def test_loop_carried_values_in_a(e2e):
    result = e2e.run('''
        #[lowram(0x0300)] static mut W: [u16; 16];
        #[lowram(0x0320)] static mut V: [u8; 16];
        #[zeropage(0x40)] static mut SUM: u16;
        #[zeropage(0x42)] static mut SUM8: u8;
        #[zeropage(0x44)] static mut LAST: u16;
        #[zeropage(0x46)] static mut PAST: u16;
        #[zeropage(0x48)] static mut NESTED: u16;
        #[lowram(0x0330)] static mut T: [u8; 8];
        #[zeropage(0x4A)] static mut CLAMPED: [u8; 4];

        #[inline(never)]
        fn sum() -> u16 {
            let mut s: u16 = 0x1000;
            let mut i: u16 = 0;
            while i < 16 { s = s + W[i]; i = i + 1; }
            return s;
        }

        #[inline(never)]
        fn sum8() -> u8 {
            let mut s: u8 = 0;
            let mut i: u16 = 0;
            while i < 16 { s = s + V[i]; i = i + 1; }
            return s;
        }

        #[inline(never)]
        fn running() {
            let mut s: u16 = 0;
            let mut i: u16 = 0;
            while i < 16 { s = s + W[i]; LAST = s; i = i + 1; }
        }

        #[inline(never)]
        fn none() -> u16 {
            let mut s: u16 = 0x55;
            let mut i: u16 = 20;
            while i < 16 { s = s + W[i]; i = i + 1; }
            return s;
        }

        #[inline(never)]
        fn nested() -> u16 {
            let mut s: u16 = 0;
            let mut i: u16 = 0;
            while i < 3 {
                let mut j: u16 = 0;
                while j < 16 { s = s + W[j]; j = j + 1; }
                i = i + 1;
            }
            return s;
        }

        #[inline(never)]
        fn clamp(x: u8) -> u8 {
            let mut n: u8 = x;
            if n > 9 { n = 9; }
            return n;
        }

        #[inline(never)]
        fn level_row(level: u8) -> u8 {
            let mut l: u16 = (level as u16) - 1;
            if l > 5 { l = 5; }
            return T[l];
        }

        #[entry]
        fn main() {
            let mut k: u16 = 0;
            while k < 16 { let v: u16 = k + 1; W[k] = (v << 8) + v; V[k] = 0x20; k = k + 1; }
            SUM = sum();         // 0x1000 + 0x0101 * (1 + ... + 16)
            SUM8 = sum8();       // 16 * 0x20 wraps to 0
            running();           // the last running total
            PAST = none();       // no pass
            NESTED = nested();   // 3 * 0x0101 * 136
            k = 0;
            while k < 8 { T[k] = 0x60 + (k as u8); k = k + 1; }
            CLAMPED[0] = clamp(7);         // under the limit: kept
            CLAMPED[1] = clamp(200);       // over: 9
            CLAMPED[2] = level_row(3);     // T[2]
            CLAMPED[3] = level_row(40);    // T[5]
        }
    ''', ExpectedState(memory={
        0x7E0040: [0x88, 0x98],
        0x7E0042: [0x00],
        0x7E0044: [0x88, 0x88],
        0x7E0046: [0x55, 0x00],
        0x7E0048: [0x98, 0x99],
        0x7E004A: [0x07, 0x09, 0x62, 0x65],
    }), max_instructions=50000)
    assert result.success, f"Failures: {result.failures}"
