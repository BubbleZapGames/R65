# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A `@ B` argument is loaded after the X/Y arguments, whose stack-slot loads
write B.

A call result bound for B was moved there first (it sits in A, which the
other loads clobber), and the X argument's `LDA d,S; TAX` then ran in m16:
the callee got the high byte of X's value as its B argument.
"""

from r65.tests.e2e import E2ETest, ExpectedState

SOURCE = '''
#[lowram(0x0320)] static mut W: [u16; 2];
#[lowram(0x0340)] static mut G16: u16;

#[inline(never)]
fn get8(v: u8) -> u8 { return v + 1; }
'''


def test_b_argument_from_a_call_survives_an_x_stack_load():
    result = E2ETest().run(SOURCE + '''
        #[inline(never)]
        fn mix(a @ A: u8, c @ B: u8, b @ X: u16) -> u16 {
            return (a as u16) + b + (c as u16);
        }
        #[inline(never)]
        fn deep(n: u8, acc: u16) -> u16 {
            return mix(n, get8(n), acc);
        }

        #[entry]
        fn main() {
            G16 = 0x100;
            W[0] = deep(10, 0x100);
            W[1] = mix(10, get8(10), G16);
        }
    ''', ExpectedState(memory={0x7E0320: [0x15, 0x01, 0x15, 0x01]}))
    assert result.success, f"{result.error} {result.failures}"


def test_promoted_b_argument_survives_an_x_stack_load():
    """FixedStack promotes `c` to B and `b` to X on its own."""
    result = E2ETest(abi="FixedStack").run(SOURCE + '''
        #[inline(never)]
        fn mix(a: u8, b: u16, c: u8, d: u16) -> u16 {
            return (a as u16) + b + (c as u16) + d;
        }
        #[inline(never)]
        fn sub3(a: u8, b: u8, c: u8) -> u8 { return a - b - c; }
        #[inline(never)]
        fn deep(n: u8, acc: u16) -> u16 {
            let t: u16 = mix(n, acc, sub3(n, 1, 1), 0x100);
            return t + mix(1, 2, 3, 4);
        }

        #[entry]
        fn main() {
            W[0] = deep(10, 0x100);
        }
    ''', ExpectedState(memory={0x7E0320: [0x1C, 0x02]}))
    assert result.success, f"{result.error} {result.failures}"
