# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A value in Y stored through a pointer is read before Y is reloaded.

The field offset (`LDY #offset`), or the `LDY #0` a word or stack store
emits, replaced the value before the `TYA` that moved it to A for the store:
`p.hi = v` with `v` in Y wrote the offset.
"""

from r65.tests.e2e import E2ETest, ExpectedState

PAIRS = '''
struct Pair { lo: u16, hi: u16 }
#[lowram(0x0300)] static mut P: [Pair; 2];
'''


def run(source, expected, abi=None):
    result = E2ETest(abi=abi).run(PAIRS + source, ExpectedState(memory={0x7E0300: expected}))
    assert result.success, f"{result.error} {result.failures}"


def test_field_store_of_a_promoted_y_param():
    """FixedStack promotes `v` to Y."""
    run('''
        #[inline(never)]
        fn put(p: *Pair, v: u16) { p.hi = v; }

        #[entry]
        fn main() {
            for i in 0..2 { P[i].lo = 0; P[i].hi = 0; }
            put(&P[0] as *Pair, 0x1234);
            put(&P[1] as *Pair, 0x5678);
        }
    ''', [0, 0, 0x34, 0x12, 0, 0, 0x78, 0x56], abi="FixedStack")


def test_field_and_deref_stores_of_a_y_param():
    run('''
        #[inline(never)]
        fn put(p: *Pair, v @ Y: u16) { p.hi = v; }
        #[inline(never)]
        fn set(p: *u16, v @ Y: u16) { *p = v; }

        #[entry]
        fn main() {
            for i in 0..2 { P[i].lo = 0; P[i].hi = 0; }
            put(&P[0] as *Pair, 0x1234);
            put(&P[1] as *Pair, 0x5678);
            set(&P[0].lo as *u16, 0x1111);
            set(&P[1].lo as *u16, 0x2222);
        }
    ''', [0x11, 0x11, 0x34, 0x12, 0x22, 0x22, 0x78, 0x56])


def test_far_pointer_field_store_of_a_y_param():
    run('''
        #[ram] static mut FP: [Pair; 2];

        #[inline(never)]
        fn put(p: far *Pair, v @ Y: u16) { p.hi = v; }

        #[entry]
        fn main() {
            for i in 0..2 { P[i].lo = 0; }
            put(&FP[0] as far *Pair, 0x1234);
            put(&FP[1] as far *Pair, 0x5678);
            P[0].hi = FP[0].hi;
            P[1].hi = FP[1].hi;
        }
    ''', [0, 0, 0x34, 0x12, 0, 0, 0x78, 0x56])
