# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""`a op b` on two statics should read `b` in the ALU instruction itself.

The compiler used to load both operands into temporaries and operate on
the temporaries — eleven instructions and a stack frame for `G_X = G_X +
G_VX`, against the four a person would write:

    LDA G_X / CLC / ADC G_VX / STA G_X

`MemoryOperandFolder` deletes the second load and makes its location the
operation's right operand. Removing the load is also what lets the *first*
operand stay in A: with the second `LDA` gone, nothing clobbers A between
the first load and its use.

These tests pin the emitted shape, and the cases that must not fold.
"""

import re

from r65.compiler.main import compile_string


def _fn_asm(source: str, name: str = "main") -> str:
    asm = compile_string(source, cfg_options=["snes"])
    body = asm.split(f"\n{name}:", 1)[1]
    return body.split("\nRTS", 1)[0].split("__SCMP", 1)[0]


TWO_GLOBALS = """
#[zeropage(0x20)] static mut G_X: u8;
#[zeropage(0x21)] static mut G_VX: u8;
#[zeropage(0x22)] static mut R: u8;
#[entry] fn main() {{ {body} }}
"""


def test_add_reads_the_second_operand_directly():
    asm = _fn_asm(TWO_GLOBALS.format(body="G_X = G_X + G_VX;"))
    assert re.search(r"LDA \$20\s*\n\s*CLC\s*\n\s*ADC \$21\s*\n\s*STA \$20", asm)
    assert "Allocate frame" not in asm


def test_every_foldable_op():
    for op, mnem in (("-", "SBC"), ("&", "AND"), ("|", "ORA"), ("^", "EOR")):
        asm = _fn_asm(TWO_GLOBALS.format(body=f"R = G_X {op} G_VX;"))
        assert re.search(rf"^\s*{mnem} \$21\b", asm, re.M), (op, asm)
        assert "Allocate frame" not in asm


def test_compare_reads_the_second_operand_directly():
    asm = _fn_asm(TWO_GLOBALS.format(body="if G_X < G_VX { R = 1; }"))
    assert re.search(r"^\s*CMP \$21\b", asm, re.M)
    assert "Allocate frame" not in asm


def test_indexed_planes_keep_their_index():
    """`P[i] + Q[i]` reads both through the same X."""
    asm = _fn_asm("""
        #[lowram] static mut P: [u8; 16];
        #[lowram] static mut Q: [u8; 16];
        fn f(i @ X: u16) { P[i] = P[i] + Q[i]; }
        #[entry] fn main() { f(3); }
    """, "f")
    assert re.search(r"LDA \$0200,X\s*\n\s*CLC\s*\n\s*ADC \$0210,X\s*\n\s*STA \$0200,X", asm)
    assert "Allocate frame" not in asm


def test_reinterpreting_cast_is_followed():
    """`x + (y as u8)` on an `i8` is a width-preserving copy in MIR; the
    fold looks through it."""
    asm = _fn_asm("""
        #[zeropage(0x20)] static mut G_X: u8;
        #[zeropage(0x21)] static mut G_VX: i8;
        #[entry] fn main() { G_X = G_X + (G_VX as u8); }
    """)
    assert re.search(r"^\s*ADC \$21\b", asm, re.M)
    assert "Allocate frame" not in asm


def test_sixteen_bit_operands():
    asm = _fn_asm("""
        #[zeropage(0x20)] static mut A16: u16;
        #[zeropage(0x22)] static mut B16: u16;
        #[entry] fn main() { A16 = A16 + B16; }
    """)
    assert re.search(r"REP #\$20.*\n(.*\n)*?\s*ADC \$22\b", asm)


def test_volatile_operand_is_left_alone():
    """A hardware read is not moved past anything; the load stays."""
    asm = _fn_asm("""
        #[hw(0x4218)] static mut JOY: u8;
        #[zeropage(0x20)] static mut G_X: u8;
        #[entry] fn main() { G_X = G_X + JOY; }
    """)
    assert re.search(r"^\s*LDA \$4218\b", asm, re.M)
    assert not re.search(r"^\s*ADC \$4218\b", asm, re.M)


def test_shared_load_is_left_alone():
    """A value read twice keeps its load; the fold needs a single reader."""
    asm = _fn_asm(TWO_GLOBALS.format(body="R = G_X + G_VX; G_X = G_VX;"))
    assert re.search(r"^\s*LDA \$21\b", asm, re.M)


def test_store_between_load_and_use_is_left_alone():
    """The load's value must be the one read *before* the store."""
    asm = _fn_asm("""
        #[zeropage(0x20)] static mut G_X: u8;
        #[zeropage(0x21)] static mut G_VX: u8;
        #[entry] fn main() {
            let v: u8 = G_VX;
            G_VX = 0;
            G_X = G_X + v;
        }
    """)
    assert not re.search(r"^\s*ADC \$21\b", asm, re.M)


def test_two_loads_fold_the_right_one():
    """With both operands loads, `a - b` keeps `LDA a` and reads b directly."""
    asm = _fn_asm(TWO_GLOBALS.format(body="R = G_X - G_VX;"))
    assert re.search(r"LDA \$20\s*\n\s*SEC\s*\n\s*SBC \$21", asm)


def test_left_load_moves_past_the_computed_right_operand():
    """`a & ~b` and `a - (b + c)`: the load of a waits for the right side,
    then is read by the AND / reverse-subtract ADC itself."""
    asm = _fn_asm(TWO_GLOBALS.format(body="R = G_X & ~G_VX;"))
    assert re.search(r"LDA \$21\s*\n\s*EOR #\$FF.*\n\s*AND \$20\s*\n\s*STA \$22", asm)
    asm = _fn_asm(TWO_GLOBALS.format(body="R = R - (G_X + G_VX);"))
    assert re.search(r"EOR #\$FF.*\n\s*SEC\s*\n\s*ADC \$22\s*\n\s*STA \$22", asm)
    assert "Allocate frame" not in asm


def test_indexed_left_load_keeps_its_index():
    """`EXT[i] & ~m`: the read moves past the complement, still through X."""
    asm = _fn_asm("""
        #[zeropage(0x20)] static mut M: u8;
        #[lowram(0x0300)] static mut EXT: [u8; 32];
        #[used] fn clear(i @ X: u16) { EXT[i] = EXT[i] & ~M; }
        #[entry] fn main() { clear(3); }
    """, "clear")
    assert re.search(r"LDA \$20\s*\n\s*EOR #\$FF.*\n\s*AND \$0300,X\s*\n\s*STA \$0300,X", asm)
