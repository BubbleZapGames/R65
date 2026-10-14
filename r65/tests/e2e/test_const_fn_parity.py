# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A const fn gives the same result folded at compile time as it does run on the
65816: values wrap at their type's width in both. Each result is stored twice,
once from constant arguments (folded) and once from arguments read out of RAM
(a real call).
"""

from r65.tests.e2e import ExpectedState


class TestConstFnParity:
    def test_folded_matches_runtime(self, e2e):
        result = e2e.run('''
            const fn add(a: u8, b: u8) -> u8 { return a + b; }
            const fn under(a: u16, b: u16) -> u8 {
                let d: u16 = a - b;
                if d > 100 { return 1; }
                return 0;
            }
            const fn xs(s: u16) -> u16 {
                let mut x: u16 = s;
                x = x ^ (x << 7); x = x ^ (x >> 9); x = x ^ (x << 8);
                return x;
            }
            const fn sum_to(n: u8) -> u8 { let mut s: u8 = 0; for i in 0..=n { s = s + i; } s }

            #[lowram(0x0300)]
            static mut FOLDED: [u16; 4];
            #[lowram(0x0340)]
            static mut RAN: [u16; 4];
            #[lowram(0x0380)]
            static mut IN: [u16; 2];

            #[entry]
            fn main() {
                FOLDED[0] = add(200, 100) as u16;
                FOLDED[1] = under(1, 2) as u16;
                FOLDED[2] = xs(0xFFFF);
                FOLDED[3] = sum_to(30) as u16;
                IN[0] = 200; IN[1] = 100;
                RAN[0] = add(IN[0] as u8, IN[1] as u8) as u16;
                IN[0] = 1; IN[1] = 2;
                RAN[1] = under(IN[0], IN[1]) as u16;
                IN[0] = 0xFFFF;
                RAN[2] = xs(IN[0]);
                IN[0] = 30;
                RAN[3] = sum_to(IN[0] as u8) as u16;
            }
        ''', ExpectedState(memory={
            # 44, 1, 0x7F7F, 465 & 0xFF = 0xD1 — little-endian words
            0x0300: [0x2C, 0, 0x01, 0, 0x7F, 0x7F, 0xD1, 0],
            0x0340: [0x2C, 0, 0x01, 0, 0x7F, 0x7F, 0xD1, 0],
        }), max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"
