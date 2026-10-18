# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""An index register live across a call is saved only if the callee can
change it (codegen/clobber_summary.py).

Functions are generated callees first and each one's emitted code says
which of X and Y it writes; a call whose callee - and everything it calls -
leaves the register alone skips the PHY/PLY. Values:
tests/e2e/test_clobber_summary.py.
"""

import re

from r65.compiler.main import compile_string

GLOBALS = """
#[zeropage(0x10)] static mut H: u8;
#[zeropage(0x12)] static mut O: u16;
#[inline(never)] fn clean() {{ H = H + 1; }}
#[inline(never)] fn dirty() {{ Y = 7; H = 1; }}
#[inline(never)] fn mid() {{ clean(); dirty(); }}
#[inline(never)] fn rec() {{ if H != 0 {{ H = H - 1; rec(); }} }}
#[inline(never)] fn raw() {{ asm!("NOP"); }}
#[inline(never)] fn xw() {{ X = 3; H = 4; }}
#[used] fn probe() {{ {body} }}
#[entry] fn main() -> ! {{ probe(); loop {{}} }}
"""


def _probe(body: str) -> str:
    asm = compile_string(GLOBALS.format(body=body), cfg_options=["snes"])
    return re.split(r"\n\s+RT[SL]\b", asm.split("\nprobe:", 1)[1], 1)[0]


def _saves(body: str, reg: str) -> int:
    return len(re.findall(rf"\bPH{reg}\b.*region start", _probe(body)))


def test_callee_that_leaves_y_needs_no_save():
    assert _saves("Y = 0x1234; clean(); O = Y;", "Y") == 0


def test_callee_that_writes_y_keeps_the_save():
    assert _saves("Y = 0x1234; dirty(); O = Y;", "Y") == 1


def test_write_two_calls_down_counts():
    assert _saves("Y = 0x1234; mid(); O = Y;", "Y") == 1


def test_recursion_and_inline_asm_are_unknown():
    assert _saves("Y = 0x1234; rec(); O = Y;", "Y") == 1
    assert _saves("Y = 0x1234; raw(); O = Y;", "Y") == 1


def test_x_follows_the_same_rule():
    assert _saves("X = 0x1234; clean(); O = X;", "X") == 0
    assert _saves("X = 0x1234; xw(); O = X;", "X") == 1
