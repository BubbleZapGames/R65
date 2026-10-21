# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A constant far address passed on the stack (`FarConstantArgFolder`) is
pushed as `LDA #bank / PHA / PEA addr`, not built in a frame slot and pushed
from it byte by byte. A pointer the function changes is not a constant.
"""

from r65.compiler.main import compile_string

SOURCE = """
#[ram(0x7E2000)] static mut BUF: [u8; 8];
#[zeropage(0x10)] static mut R: u8;
#[inline(never)]
fn rd(q: far *u8) -> u8 {{ return *q; }}
#[entry] fn main() {{ {body} }}
"""


def _main_asm(body: str) -> str:
    asm = compile_string(SOURCE.format(body=body), cfg_options=["snes"])
    return asm.split("\nmain:", 1)[1].split("\nrd:", 1)[0]


def test_static_address_is_pushed_with_pea():
    asm = _main_asm("R = rd(&BUF[3] as far *u8);")
    assert "LDA #$7E" in asm and "PEA $2003" in asm
    assert "Allocate frame" not in asm


def test_reassigned_pointer_is_pushed_from_its_slot():
    asm = _main_asm("let mut p: far *u8 = &BUF[3] as far *u8; if R == 0 { p = &BUF[4] as far *u8; } R = rd(p);")
    assert "PEA" not in asm
