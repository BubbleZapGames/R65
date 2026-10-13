# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Moves between special registers: D/S and A/X, and X/Y into B.

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


def test_index_to_b_in_m16_saves_and_restores_a_at_one_width():
    # X -> B saves A around the XBA; push and pull must be the same width,
    # or the return address is off by a byte.
    asm = compile_string("""
#[zeropage] static mut R: u16;
#[inline(never)]
fn g(v @ A: u16) -> u16 { X = 0x1234; B = X; return A; }
#[entry]
fn main() -> ! { R = g(0x5678); loop {} }
""", "t.r65")
    text = asm.split("\ng:", 1)[1].split("\n\n", 1)[0]
    lines = [l.split(';')[0].strip() for l in text.splitlines()]
    pha, pla = lines.index("PHA"), lines.index("PLA")
    width_at = lambda i: [l for l in lines[:i] if l.startswith(".ACCU")][-1]
    assert width_at(pha) == width_at(pla) == ".ACCU 8", text
