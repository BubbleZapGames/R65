# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""An interrupt handler sets its own register widths and saves what it writes.

An interrupt keeps the M and X flags of the code it interrupted, so the
prologue has to put the body in m8/x16 itself — even a body of one
`INC counter`, which is a 16-bit increment when the NMI lands in m16 code.
And the registers saved are checked against the emitted body: a constant
store (`V = 2`) loads A with no MIR register def, and once went out unsaved.
"""

import re

from r65.compiler.main import compile_string


def _handler(body: str) -> list:
    src = f"""
#[lowram]
static mut V: u8;

#[interrupt(nmi)]
fn nmi() {{ {body} }}

#[entry]
fn main() {{ }}
"""
    asm = compile_string(src)
    text = asm.split("\nnmi:", 1)[1].split("\n\n", 1)[0]
    return [l.split(';')[0].strip() for l in text.splitlines()
            if l.split(';')[0].strip() and not l.strip().startswith('.')]


def test_memory_only_body_still_sets_widths():
    """`V = V + 1` folds to INC on memory; the switches must precede it."""
    code = _handler("V = V + 1;")
    inc = code.index("INC $0200")
    assert "SEP #$20" in code[:inc], code
    assert any(re.fullmatch(r"REP #\$[13]0", c) for c in code[:inc]), code


def test_constant_store_saves_a_on_every_return():
    """Two returns, each storing a constant through A: both restore it."""
    code = _handler("if X == 0 { V = 2; return; } V = 3;")
    assert code[1] == "PHA", code
    for i, c in enumerate(code):
        if c == "RTI":
            assert code[i - 1] == "PLA", code
    assert code.count("RTI") == 2, code
