# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A parameter FixedStack promotes to A survives the entry moves before its own.

`a` and `c` arrive in X and Y, `e` in A, and all three are copied to the
frame at entry, `a` first. Homing X goes through A (there is no STX d,S): A
read as free there, since only explicit `@ A` bindings were tracked, so
`e` was stored as `a`'s low byte. Tracked, A is saved around the transfer.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def test_a_promoted_param_survives_homing_x():
    result = E2ETest(abi="FixedStack").run('''
        #[lowram(0x0300)] static mut OUT: [u16; 2];

        #[inline(never)]
        fn h(a: u16, b: u16) -> u16 { return a + b + 1; }

        #[inline(never)]
        fn many(a: u16, c: u16, e: u8) -> u16 {
            let x1: u16 = h(a, 1);
            let x2: u16 = h(c, 2);
            let mut acc: u16 = 0;
            for i in 0..3 {
                acc = h(acc, i as u16);
            }
            return x1 + x2 + acc + e as u16;
        }

        #[entry]
        fn main() {
            OUT[0] = many(1, 3, 5);
            OUT[1] = many(0x100, 0x300, 0x50);
        }
    ''', ExpectedState(memory={0x7E0300: [20, 0, 91, 4]}))
    assert result.success, f"{result.error} {result.failures}"
