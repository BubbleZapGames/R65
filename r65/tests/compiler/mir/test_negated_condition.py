# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A `!` or bare-bool condition branches on the value where it sits.

`!e` swaps the branch targets and a bare scalar is `e != 0`, so neither
builds a 0/1 value in a stack slot first. Values:
tests/e2e/test_negated_condition.py.
"""

import re

from r65.compiler.main import compile_string

GLOBALS = """
#[zeropage(0x10)] static mut FLAG: bool;
#[zeropage(0x11)] static mut G: u8;
#[used] fn probe() {{ {body} }}
#[entry] fn main() -> ! {{ probe(); loop {{}} }}
"""


def _probe(body: str) -> list:
    asm = compile_string(GLOBALS.format(body=body), cfg_options=["snes"])
    fn = re.split(r"\n\s+RT[SL]\b", asm.split("\nprobe:", 1)[1], 1)[0]
    return [line.split(";")[0].strip() for line in fn.splitlines()
            if line.strip() and not line.strip().startswith((";", "."))
            and not line.rstrip().endswith(":")]


def test_negated_flag_branches_off_its_load():
    assert _probe("if !FLAG { G = 1; }")[:2] == ["LDA $10", "BNE probe__L2"]


def test_bare_flag_takes_no_slot():
    code = _probe("if FLAG { G = 2; }")
    assert code[:2] == ["LDA $10", "BEQ probe__L2"]
    assert not any(",S" in op for op in code)
