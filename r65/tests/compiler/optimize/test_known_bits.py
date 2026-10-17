# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Masks that known-zero or unread bits make moot (optimize/known_bits.py).

Codegen ANDs wherever a value changes width - a zero-extend, the XBA of a
shift by 8 - without knowing the high byte is 0 already or that nothing
reads it. These pin the AND count of each shape the MIR pass sees through,
and the one it must not touch. Values: tests/e2e/test_known_bits.py.
"""

import re

from r65.compiler.main import compile_string

GLOBALS = """
#[zeropage(0x10)] static mut BB: u8;
#[zeropage(0x12)] static mut W: u16;
#[zeropage(0x14)] static mut O: u16;
#[zeropage(0x16)] static mut OB: u8;
#[lowram] static mut TBL: [u8; 256];
fn wide() -> u16 {{ return W + 1; }}
#[used] fn probe() {{ {body} }}
#[entry] fn main() -> ! {{ probe(); loop {{}} }}
"""


def _ands(body: str):
    asm = compile_string(GLOBALS.format(body=body), cfg_options=["snes"])
    fn = re.split(r"\n\s+RT[SL]\b", asm.split("\nprobe:", 1)[1], 1)[0]
    return re.findall(r"AND #\$([0-9A-F]+)", fn)


def test_value_below_256_narrowed_and_widened_masks_once():
    assert len(_ands("let x: u16 = BB as u16; OB = TBL[(x as u8) as u16];")) == 1


def test_shifted_index_needs_no_second_mask():
    assert len(_ands("OB = TBL[((W + 1) >> 8) as u8];")) == 1


def test_zero_extend_then_shift_left_by_eight():
    assert len(_ands("O = (BB as u16) << 8;")) == 1


def test_masked_field_then_shift():
    assert _ands("O = (W & 0x7C00) >> 10;") == ["7C00"]
    assert _ands("O = ((W >> 2) & 31) << 8;") == ["1F"]


def test_ninth_bit_test_reads_one_bit():
    assert _ands("O = ((W & 0x1FF) >> 8) & 1;") == ["01"]


def test_widened_u8_shifted_out_is_zero():
    assert _ands("O = ((BB as u16) >> 8) & 1;") == []


def test_unknown_u16_keeps_its_byte_mask():
    """A call's u16 is not known below 256: `as u8` still truncates."""
    assert "FF" in _ands("let s: u16 = wide(); OB = TBL[(s as u8) as u16];")
