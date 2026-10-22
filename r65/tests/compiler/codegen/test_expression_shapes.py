# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Instruction shapes of small expressions: `!` on a bool, a subtract whose
right operand is computed, constant chains, an inlined bool result tested by
`if`. Values: tests/e2e/test_expression_shapes.py.
"""

import re

from r65.compiler.main import compile_string

GLOBALS = """
#[zeropage(0x10)] static mut F: bool;
#[zeropage(0x11)] static mut G: u8;
#[zeropage(0x12)] static mut H: u8;
#[zeropage(0x14)] static mut W: u16;
#[zeropage(0x16)] static mut V: u16;
#[used] fn probe() {{ {body} }}
#[entry] fn main() -> ! {{ probe(); loop {{}} }}
"""


def _probe(body: str):
    asm = compile_string(GLOBALS.format(body=body), cfg_options=["snes"])
    fn = re.split(r"\n\s+RT[SL]\b", asm.split("\nprobe:", 1)[1], 1)[0]
    return [line.split(";")[0].strip() for line in fn.splitlines()
            if line.strip() and not line.strip().startswith((".", ";"))]


def test_logical_not_of_bool_flips_bit_zero():
    assert _probe("F = !F;") == ["LDA $10", "EOR #$01", "STA $10"]


def test_constant_minus_computed_complements_and_adds():
    """`k - v` with v just computed in A: EOR / SEC / ADC #k, no spill."""
    assert _probe("G = 5 - (H + G);") == [
        "LDA $12", "CLC", "ADC $11", "EOR #$FF", "SEC", "ADC #$05", "STA $11"]


def test_constant_chain_is_one_operation():
    """MIR lowering (_lower_constant_chain): one constant per `+`/`-` or
    bitwise chain."""
    assert _probe("G = G + 8 - 2;") == ["LDA $11", "CLC", "ADC #$06", "STA $11"]
    assert _probe("G = 10 - H - 3;") == ["LDA #$07", "SEC", "SBC $12", "STA $11"]
    assert _probe("G = H & 0xF0 & 0x3F;") == ["LDA $12", "AND #$30", "STA $11"]


INLINED_BOOL = """
#[zeropage(0x11)] static mut G: u8;
#[zeropage(0x12)] static mut H: u8;
#[inline] fn is3() -> bool {{ return G == 3; }}
#[used] fn probe() {{ {body} }}
#[entry] fn main() -> ! {{ probe(); loop {{}} }}
"""


def _probe_src(src: str):
    asm = compile_string(src, cfg_options=["snes"])
    fn = re.split(r"\n\s+RT[SL]\b", asm.split("\nprobe:", 1)[1], 1)[0]
    return [line.split(";")[0].strip() for line in fn.splitlines()
            if line.strip() and not line.strip().startswith((".", ";"))
            and not line.strip().endswith(":")]


def test_inlined_bool_result_branches_on_the_compare():
    """optimize/jump_thread.py: no 0/1 value is built for `if !is3()`."""
    assert _probe_src(INLINED_BOOL.format(body="if !is3() { H = 1; }"))[:3] == [
        "LDA $11", "EOR #$03", "BNE probe__L1"]


def test_inlined_bool_read_twice_is_kept():
    code = _probe_src(INLINED_BOOL.format(body="let b = is3(); if b { H = 1; } if b { G = 0; }"))
    assert "LDA #$01" in code
