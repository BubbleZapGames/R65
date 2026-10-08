# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for `if let`, `while let` and `let ... else`.

The parser desugars all three to a two-arm `match`, so each is pinned by
compiling to the same assembly as the `match` written by hand. A payload
binding in one is the same view of the pool a hand-written arm gets.

See docs/control-flow.md.
"""

import pytest

from r65.compiler.errors import HIRError
from r65.compiler.main import compile_string


SHAPES = """
struct Circle { r: u8 }
struct Rect { w: u8, h: u8 }
enum Shape { Empty, C(Circle), R(Rect) }
enum Dir { North, East, South, West }
#[lowram]
static mut POOL: [Shape; 8];
#[lowram]
static mut OUT: u8;
fn halt() -> ! { loop {} }
"""

MAIN = """
fn main() -> ! {
    POOL[0] = Shape::C(Circle { r: 5 });
    OUT = f(0);
    loop {}
}
"""


def asm(fn_source: str) -> str:
    """The program's assembly, without comment lines (they carry line numbers)."""
    out = compile_string(SHAPES + fn_source + MAIN, "test.r65")
    return "\n".join(l for l in out.splitlines() if not l.startswith(";"))


EQUIVALENT = {
    "if_let_payload_view": (
        "fn f(i: u16) -> u8 { if let Shape::C(c) = POOL[i] { c.r = c.r + 1; } return OUT; }",
        "fn f(i: u16) -> u8 { match POOL[i] { Shape::C(c) => { c.r = c.r + 1; } _ => {} }; return OUT; }",
    ),
    "else_if_let": (
        "fn f(i: u16) -> u8 { if let Shape::C(c) = POOL[i] { OUT = c.r; }"
        " else if let Shape::R(r) = POOL[i] { OUT = r.h; } else { OUT = 9; } return OUT; }",
        "fn f(i: u16) -> u8 { match POOL[i] { Shape::C(c) => { OUT = c.r; } _ => {"
        " match POOL[i] { Shape::R(r) => { OUT = r.h; } _ => { OUT = 9; } }; } }; return OUT; }",
    ),
    "if_let_expression": (
        "fn f(i: u16) -> u8 { let d: Dir = OUT as Dir;"
        " let v: u8 = if let Dir::North | Dir::South = d { 1 } else { 2 }; return v; }",
        "fn f(i: u16) -> u8 { let d: Dir = OUT as Dir;"
        " let v: u8 = match d { Dir::North | Dir::South => { 1 } _ => { 2 } }; return v; }",
    ),
    "labeled_while_let": (
        "fn f(i: u16) -> u8 { let mut j: u16 = i; 'scan: while let Shape::C(c) = POOL[j] {"
        " c.r = 0; j = j + 1; if j == 8 { break 'scan; } } return OUT; }",
        "fn f(i: u16) -> u8 { let mut j: u16 = i; 'scan: loop { match POOL[j] { Shape::C(c) => {"
        " c.r = 0; j = j + 1; if j == 8 { break 'scan; } } _ => { break 'scan; } }; } return OUT; }",
    ),
    "let_else_return": (
        "fn f(i: u16) -> u8 { let Shape::R(r) = POOL[i] else { return 0; }; r.w }",
        "fn f(i: u16) -> u8 { match POOL[i] { Shape::R(r) => { return r.w; } _ => { return 0; } }; return 0; }",
    ),
}


@pytest.mark.parametrize("sugar,match_form", EQUIVALENT.values(), ids=EQUIVALENT.keys())
def test_compiles_to_the_hand_written_match(sugar, match_form):
    assert asm(sugar) == asm(match_form)


class TestLetElseDiverges:
    """The else block runs when the pattern fails, so it must leave the block."""

    @pytest.mark.parametrize("else_block", [
        "{ return 0; }",
        "{ halt(); }",
        "{ loop {} }",
        "{ if OUT == 0 { return 1; } else { return 2; } }",
    ])
    def test_accepted(self, else_block):
        asm(f"fn f(i: u16) -> u8 {{ let Shape::C(c) = POOL[i] else {else_block}; c.r }}")

    @pytest.mark.parametrize("else_block", [
        "{ OUT = 1; }",
        "{ loop { break; } }",
        "{ if OUT == 0 { return 1; } }",
    ])
    def test_fall_through_rejected(self, else_block):
        with pytest.raises(HIRError, match="must not fall through"):
            asm(f"fn f(i: u16) -> u8 {{ let Shape::C(c) = POOL[i] else {else_block}; c.r }}")


def test_binding_used_after_rewrite_rejected():
    with pytest.raises(HIRError, match="used after its element was rewritten"):
        asm("fn f(i: u16) -> u8 { if let Shape::C(c) = POOL[i] {"
            " POOL[i] = Shape::Empty; OUT = c.r; } return OUT; }")
