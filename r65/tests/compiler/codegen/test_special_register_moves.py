# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Moves between D/S and A/X: the 65816's own transfers, no recursion.

The transfer table had no TDC/TCD/TSC/TCS/TSX/TXS, and its fallback (route
through A) asked for the same missing pair again: `A = D;`, `X = S;`,
`D = W;` and `S = W;` all died of "maximum recursion depth exceeded".
"""

import pytest

from r65.compiler.main import compile_string


def _fn(body: str) -> list:
    asm = compile_string(f"""
#[lowram] static mut W: u16;
#[inline(never)]
fn f() {{ {body} }}
#[entry]
fn main() {{ f(); }}
""", "t.r65")
    text = asm.split("\nf:", 1)[1].split("\n\n", 1)[0]
    return [l.split(';')[0].strip() for l in text.splitlines()
            if l.split(';')[0].strip() and not l.strip().startswith('.')]


@pytest.mark.parametrize("body, op", [
    ("A = D; W = A;", "TDC"),
    ("W = S;", "TSC"),
    ("X = S; W = X;", "TSX"),
    ("D = W;", "TCD"),
    ("S = W;", "TCS"),
])
def test_transfer(body, op):
    assert op in _fn(body)
