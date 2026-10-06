# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Signed relational compares must not read a stale overflow flag.

The signed branches test N XOR V. They were fed by CMP, which sets N, Z and C
but never V - so the branch read whatever V the last ADC/SBC left behind. Any
arithmetic that overflowed before an unrelated signed compare inverted it.
Rotor Raiders' enemy airframe hit this: `approach(cur, target, step)` returned
its target outright whenever the caller's `aim - y` had overflowed i16, so the
vertical speed snapped between rest and full and the airframe crashed on its
own deck.

The compare is now a subtraction (SEC/SBC, A restored under PHP/PLP). These
cases set V with an overflowing add immediately before the compare, cover
compares whose own subtraction overflows, and check that A still holds the
left operand afterwards, which `approach`-shaped code relies on.

Every input arrives through a `static`, so nothing folds at compile time.
"""

import pytest
from r65.tests.e2e import ExpectedState

OUT = 0x7E0010


def program(ty: str, a: str, b: str, op: str, poison: bool = True) -> str:
    # POISON + POISON overflows i16 (30000 + 30000), leaving V set just
    # before the compare under test.
    poison_stmt = "SINK = POISON + POISON;" if poison else ""
    return f'''
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x12)] static mut A_IN: {ty};
#[zeropage(0x14)] static mut B_IN: {ty};
#[zeropage(0x16)] static mut POISON: i16;
#[zeropage(0x18)] static mut SINK: i16;
#[entry]
fn main() {{
    A_IN = {a};
    B_IN = {b};
    POISON = 30000;
    let a: {ty} = A_IN;
    let b: {ty} = B_IN;
    {poison_stmt}
    if a {op} b {{ OUT = 1; }} else {{ OUT = 2; }}
}}
'''


def truth(a: int, b: int, op: str) -> int:
    return 1 if {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op] else 2


def lit(v: int) -> str:
    return str(v) if v >= 0 else f"0 - {-v}"


class TestStaleOverflow:
    """V set by an earlier add must not flip an unrelated compare."""

    @pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
    @pytest.mark.parametrize("a,b", [(5, 3), (3, 5), (4, 4), (-7, 2), (2, -7)])
    def test_i16(self, e2e, op, a, b):
        r = e2e.run(program("i16", lit(a), lit(b), op),
                    ExpectedState(memory={OUT: truth(a, b, op)}))
        assert r.success, f"{a} {op} {b}: {r.error} {r.failures}"

    @pytest.mark.parametrize("op", ["<", ">="])
    @pytest.mark.parametrize("a,b", [(5, 3), (3, 5), (-7, 2)])
    def test_i8(self, e2e, op, a, b):
        r = e2e.run(program("i8", lit(a), lit(b), op),
                    ExpectedState(memory={OUT: truth(a, b, op)}))
        assert r.success, f"{a} {op} {b}: {r.error} {r.failures}"

    @pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
    @pytest.mark.parametrize("a", [-3, 0, 3])
    def test_against_zero(self, e2e, op, a):
        r = e2e.run(program("i16", lit(a), "0", op),
                    ExpectedState(memory={OUT: truth(a, 0, op)}))
        assert r.success, f"{a} {op} 0: {r.error} {r.failures}"


class TestCompareOverflow:
    """The compare's own subtraction overflows: N alone would be wrong."""

    @pytest.mark.parametrize("op", ["<", ">"])
    @pytest.mark.parametrize("a,b", [(30000, -30000), (-30000, 30000)])
    def test_i16(self, e2e, op, a, b):
        r = e2e.run(program("i16", lit(a), lit(b), op, poison=False),
                    ExpectedState(memory={OUT: truth(a, b, op)}))
        assert r.success, f"{a} {op} {b}: {r.error} {r.failures}"

    @pytest.mark.parametrize("a,b", [(100, -100), (-100, 100)])
    def test_i8(self, e2e, a, b):
        r = e2e.run(program("i8", lit(a), lit(b), "<", poison=False),
                    ExpectedState(memory={OUT: truth(a, b, "<")}))
        assert r.success, f"{a} < {b}: {r.error} {r.failures}"


class TestApproach:
    """The shape that found it: the compared value is the one returned."""

    SRC = '''
#[zeropage(0x10)] static mut OUT: i16;
#[zeropage(0x12)] static mut CUR: i16;
#[zeropage(0x14)] static mut TGT: i16;
#[zeropage(0x16)] static mut STEP: i16;
#[zeropage(0x18)] static mut SINK: i16;
fn approach(cur: i16, target: i16, step: i16) -> i16 {{
    if cur < target {{
        let next: i16 = cur + step;
        if next > target {{ return target; }}
        return next;
    }}
    if cur > target {{
        let next: i16 = cur - step;
        if next < target {{ return target; }}
        return next;
    }}
    return cur;
}}
#[entry]
fn main() {{
    CUR = {cur}; TGT = {tgt}; STEP = {step};
    let big: i16 = 0 - 23552;
    SINK = big - 24576;
    OUT = approach(CUR, TGT, STEP);
}}
'''

    @pytest.mark.parametrize("cur,tgt,step,expected", [
        (0, 256, 6, 6),         # the airframe's climb: one step, not the target
        (256, 0, 8, 248),       # ... and its drag
        (250, 256, 6, 256),     # last step lands on the target
        (0, 0 - 256, 6, -6),
    ])
    def test_steps(self, e2e, cur, tgt, step, expected):
        src = self.SRC.format(cur=lit(cur), tgt=lit(tgt), step=lit(step))
        r = e2e.run(src, ExpectedState(memory={OUT: [expected & 0xFF, (expected >> 8) & 0xFF]}))
        assert r.success, f"approach({cur}, {tgt}, {step}): {r.error} {r.failures}"
