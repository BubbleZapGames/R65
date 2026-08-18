# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Match dispatch on signed scrutinees, including negative literal patterns.

Nothing covered this. It is worth pinning because signed comparison on the
65816 is not a plain `CMP` — a correct signed `<` has to account for the
overflow flag, and match lowering has to pick that form rather than the
unsigned one. An unsigned compare would still dispatch *some* arms correctly,
so the failure would be partial rather than obvious.

Every scrutinee arrives through a `static`, so it is a runtime value rather
than a constant the folder could resolve at compile time.
"""

import pytest
from r65.tests.e2e import ExpectedState

OUT = 0x7E0010


def program(ty: str, arms: str, value: str) -> str:
    return f'''
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x12)] static mut IN: {ty};
#[entry]
fn main() {{
    IN = {value};
    let v: {ty} = IN;
    match v {{ {arms} }};
}}
'''


class TestNegativeLiteralPatterns:
    """`match v {{ -1 => ... }}` — a negative literal is a pattern, where
    `0 - 1` would be an expression and is correctly rejected."""

    ARMS = ("-1 => { OUT = 11; }, -5 => { OUT = 55; }, "
            "3 => { OUT = 33; }, _ => { OUT = 99; }")

    @pytest.mark.parametrize("value,expected", [
        ("0 - 1", 11),      # matches the first negative arm
        ("0 - 5", 55),      # matches a later negative arm
        ("3", 33),          # a positive arm alongside negatives
        ("7", 99),          # positive, no arm
        ("0 - 9", 99),      # negative, no arm -- must reach the wildcard
    ])
    def test_i8(self, e2e, value, expected):
        r = e2e.run(program("i8", self.ARMS, value),
                    ExpectedState(memory={OUT: expected}))
        assert r.success, f"IN={value}: {r.error} {r.failures}"

    @pytest.mark.parametrize("value,expected", [
        ("0 - 300", 3),
        ("300", 4),         # the same magnitude, opposite sign
        ("5", 9),
    ])
    def test_i16(self, e2e, value, expected):
        """A 16-bit scrutinee, and a +/- pair of the same magnitude so a sign
        error cannot pass by matching the wrong arm."""
        arms = "-300 => { OUT = 3; }, 300 => { OUT = 4; }, _ => { OUT = 9; }"
        r = e2e.run(program("i16", arms, value),
                    ExpectedState(memory={OUT: expected}))
        assert r.success, f"IN={value}: {r.error} {r.failures}"


class TestNegativeRangePatterns:
    """A range spanning negatives, where an unsigned compare would invert the
    ordering entirely."""

    ARMS = "-5..0 => { OUT = 1; }, _ => { OUT = 0; }"

    @pytest.mark.parametrize("value,expected", [
        ("0 - 3", 1),       # inside
        ("0 - 5", 1),       # the inclusive low bound
        ("0", 0),           # the exclusive high bound
        ("2", 0),           # above
        ("0 - 9", 0),       # below
    ])
    def test_range(self, e2e, value, expected):
        r = e2e.run(program("i8", self.ARMS, value),
                    ExpectedState(memory={OUT: expected}))
        assert r.success, f"IN={value}: {r.error} {r.failures}"


class TestNewtypeOverSignedPayload:
    """The same, through a newtype — the case that prompted the check."""

    def test_negative_literal_through_a_wrapper(self, e2e):
        src = '''
struct Rot(i16);
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x12)] static mut IN: Rot;
#[entry]
fn main() {
    IN = 0 - 90;
    let r: Rot = IN;
    match r { -90 => { OUT = 1; }, 90 => { OUT = 2; }, _ => { OUT = 0; } };
}
'''
        r = e2e.run(src, ExpectedState(memory={OUT: 1}))
        assert r.success, f"{r.error} {r.failures}"
