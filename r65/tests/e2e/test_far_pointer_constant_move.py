# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A constant far pointer moved into a local or parameter gets all 3 bytes.

The immediate move split on 8 vs 16 bits only, so a `far *T` constant took
the 8-bit path: the low byte was stored and the high and bank bytes were
left as they were, and writes through the pointer landed in bank 0. At -O1
the constant-argument pass makes a constant far argument the same move in
the callee.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def run(source, expected, abi=None, opt="-O1"):
    result = E2ETest(abi=abi).run(source, ExpectedState(memory=expected),
                                  extra_args=[opt])
    assert result.success, f"{result.error} {result.failures}"


LOCAL = '''
#[inline(never)]
fn put(v: u8) {
    let q: far *u8 = 0x7F1040 as far *u8;
    q[0] = v;
    q[1] = v + 1;
}
#[entry]
fn main() { put(5); }
'''

FOLDED = '''
#[inline(never)]
fn fill(p: far *u8, r: u8) {
    for i in 0..4 { p[i] = p[i] + r; }
}
#[entry]
fn main() {
    for r in 1..4 { fill(0x7F1000 as far *u8, r as u8); }
}
'''


def test_local():
    for opt in ("-O0", "-O1"):
        run(LOCAL, {0x7F1040: [5, 6], 0x7E0040: [0]}, opt=opt)


def test_folded_argument():
    run(FOLDED, {0x7F1000: [6, 6, 6, 6], 0x7E0000: [0, 0, 0, 0]})


def test_folded_argument_fixedstack():
    run(FOLDED, {0x7F1000: [6, 6, 6, 6], 0x7E0000: [0, 0, 0, 0]}, abi="FixedStack")
