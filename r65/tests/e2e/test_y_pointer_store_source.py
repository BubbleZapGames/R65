# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A word stored through a pointer in Y may come from X or Y.

The Y-pointer store (`STA off,Y`) loaded its u16 source as an immediate, from
A or from memory, so a value in an index register - including the pointer
itself, `p.me = p` - was a compile error.
"""

from r65.tests.e2e import E2ETest, ExpectedState

STATICS = '''
struct Pair { lo: u16, hi: u16 }
struct N { a: u8, me: *N }
#[lowram(0x0300)] static mut P: [Pair; 2];
#[lowram(0x0310)] static mut NN: [N; 2];
'''


def test_word_from_x_and_the_pointer_itself():
    result = E2ETest().run(STATICS + '''
        #[inline(never)]
        fn put(p @ Y: *Pair, v @ X: u16) { p.hi = v; }
        #[inline(never)]
        fn me(p @ Y: *N) { p.me = p; }

        #[entry]
        fn main() {
            put(&P[0] as *Pair, 0x1234);
            put(&P[1] as *Pair, 0x5678);
            me(&NN[0] as *N);
            me(&NN[1] as *N);
        }
    ''', ExpectedState(memory={
        0x7E0302: [0x34, 0x12],
        0x7E0306: [0x78, 0x56],
        0x7E0311: [0x10, 0x03, 0x00, 0x13, 0x03],
    }))
    assert result.success, f"{result.error} {result.failures}"


def test_fixedstack_pointer_in_y_word_in_x():
    """FixedStack puts `p` in Y beside the bound `v @ X`."""
    result = E2ETest(abi="FixedStack").run(STATICS + '''
        #[inline(never)]
        fn put(p: *Pair, v @ X: u16) { p.hi = v; }

        #[entry]
        fn main() {
            put(&P[0] as *Pair, 0x1234);
            put(&P[1] as *Pair, 0x5678);
        }
    ''', ExpectedState(memory={0x7E0302: [0x34, 0x12], 0x7E0306: [0x78, 0x56]}))
    assert result.success, f"{result.error} {result.failures}"
