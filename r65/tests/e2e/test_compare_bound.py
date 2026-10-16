# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`x > K` and `x <= K` against a constant compile as `x >= K+1` and
`x < K+1` (optimize/compare_bound.py): one branch instead of two. These
check the unsigned edges - K next to the type's maximum, where K+1 must
still fit, and K at it, where the rewrite must not happen. The signed
edges are in test_signed_compare.py's TestConstantRhs.
"""

import pytest
from r65.tests.e2e import ExpectedState

OUT = 0x7E0010


def program(ty: str, a: int, k: int, op: str) -> str:
    return f'''
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x12)] static mut A_IN: {ty};
#[entry]
fn main() {{
    A_IN = {a};
    let a: {ty} = A_IN;
    if a {op} {k} {{ OUT = 1; }} else {{ OUT = 2; }}
}}
'''


def truth(a: int, k: int, op: str) -> int:
    return 1 if {">": a > k, "<=": a <= k}[op] else 2


class TestUnsignedBound:
    @pytest.mark.parametrize("op", [">", "<="])
    @pytest.mark.parametrize("k", [1, 254, 255])
    @pytest.mark.parametrize("a", [0, 1, 2, 254, 255])
    def test_u8(self, e2e, op, k, a):
        r = e2e.run(program("u8", a, k, op), ExpectedState(memory={OUT: truth(a, k, op)}))
        assert r.success, f"{a} {op} {k}: {r.error} {r.failures}"

    @pytest.mark.parametrize("op", [">", "<="])
    @pytest.mark.parametrize("k", [1, 300, 65534, 65535])
    @pytest.mark.parametrize("a", [0, 1, 300, 301, 65534, 65535])
    def test_u16(self, e2e, op, k, a):
        r = e2e.run(program("u16", a, k, op), ExpectedState(memory={OUT: truth(a, k, op)}))
        assert r.success, f"{a} {op} {k}: {r.error} {r.failures}"
