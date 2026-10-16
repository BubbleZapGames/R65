# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Two index streams alternating in X: one takes Y (optimize/index_streams.py).

`DST[j].a = SRC[i].a; DST[j].b = SRC[i].b` used to reload X for every
access. The stream whose accesses abs,Y can make moves to Y; a long (#[ram])
stream cannot, since there is no long,Y. Runtime behaviour is pinned by
tests/e2e/test_index_streams.py; these pin the shapes.
"""

import re

from r65.compiler.main import compile_string


def _fn_asm(source: str, name: str = "main") -> str:
    asm = compile_string(source, cfg_options=["snes"])
    body = asm.split(f"\n{name}:", 1)[1]
    return body.split("\nRTS", 1)[0].split("__SCMP", 1)[0]


COPY = """
struct Spr {{ x: u8, y: u8, tile: u8 }}
{src_attr} static mut PX: [u8; 8];
{src_attr} static mut PY: [u8; 8];
#[lowram] static mut OAM: [Spr; 8];
#[zeropage(0x30)] static mut I: u16;
#[zeropage(0x32)] static mut J: u16;
#[entry] fn main() {{
    let i = I;
    let j = J;
    OAM[j].x = PX[i];
    OAM[j].y = PY[i];
    OAM[j].tile = {tile};
}}
"""


def test_destination_stream_takes_y():
    asm = _fn_asm(COPY.format(src_attr="#[ram]", tile=7))
    assert len(re.findall(r"STA \$[0-9A-F]{4},Y", asm)) == 3, asm
    # X is set for the source once; the old per-field reload is gone.
    assert len(re.findall(r"^\s*(TAX|LDX)\b", asm, re.M)) == 1, asm


def test_two_long_streams_stay_on_x():
    """Both sides #[ram]: no long,Y, so nothing moves."""
    asm = _fn_asm(COPY.format(src_attr="#[ram]", tile=7).replace(
        "#[lowram] static mut OAM", "#[ram] static mut OAM"))
    assert ",Y" not in asm


def test_zero_store_through_y_is_not_stz():
    """There is no STZ abs,Y: a zero goes through A."""
    asm = _fn_asm(COPY.format(src_attr="#[ram]", tile=0))
    assert not re.search(r"STZ \$[0-9A-F]{4},Y", asm)
    assert re.search(r"LDA #\$00\s*\n\s*STA \$[0-9A-F]{4},Y", asm), asm
