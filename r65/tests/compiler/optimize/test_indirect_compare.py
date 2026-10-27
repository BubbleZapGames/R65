# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A byte read through a pointer parameter only to be compared is read by
the CMP itself (`IndirectCompareFolder`), so the other operand stays in A:
strcmp's loop is `LDA [a],Y / CMP [b],Y` with no stack frame, where it used
to spill both bytes and reload them for each compare.
"""

import re

from r65.compiler.main import compile_string


def _fn_asm(source: str, name: str) -> str:
    asm = compile_string(source, cfg_options=["snes"])
    body = asm.split(f"\n{name}:", 1)[1]
    return body.split("\n    RTS", 1)[0]


CMP_LOOP = """
#[ram] static mut A1: [u8; 8];
#[ram] static mut B1: [u8; 8];
#[ram] static mut R: u8;
#[used]
far fn cmp(a: far *u8, b: far *u8) -> {ret} {{
    let mut i: u16 = 0;
    loop {{
        let ca: {ty} = a[i];
        let cb: {ty} = b[i];
        {between}
        if ca < cb {{ return 1; }}
        if ca > cb {{ return 2; }}
        if ca == 0 {{ return 0; }}
        i++;
    }}
}}
#[entry] fn main() -> ! {{ R = cmp(&A1, &B1) as u8; loop {{}} }}
"""


def test_both_compares_read_through_the_pointer():
    asm = _fn_asm(CMP_LOOP.format(ret="u8", ty="u8", between=""), "cmp")
    assert re.search(r"LDA \[\$[0-9A-F]+\],Y.*\n\s*CMP \[\$[0-9A-F]+\],Y", asm)
    assert len(re.findall(r"CMP \[\$[0-9A-F]+\],Y", asm)) == 2
    assert "Allocate frame" not in asm


def test_store_between_load_and_compare_is_left_alone():
    asm = _fn_asm(CMP_LOOP.format(ret="u8", ty="u8", between="b[i] = 0;"), "cmp")
    assert not re.search(r"CMP \[\$[0-9A-F]+\],Y", asm)


def test_vreg_defined_before_the_compare_is_left_alone():
    """Loop promotion may later pin `j` to Y, over the index the CMP needs."""
    asm = _fn_asm(CMP_LOOP.format(ret="u8", ty="u8", between="let mut j: u16 = 3; R = j as u8;"),
                  "cmp")
    assert not re.search(r"CMP \[\$[0-9A-F]+\],Y", asm)


def test_signed_ordered_compare_is_left_alone():
    """A signed `<` is an SBC, not a CMP."""
    asm = _fn_asm(CMP_LOOP.format(ret="u8", ty="i8", between=""), "cmp")
    assert not re.search(r"CMP \[\$[0-9A-F]+\],Y", asm)
