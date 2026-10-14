# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Inline branches use unique labels, never assembler anonymous labels.

The asm passes (branch fixup, mode dataflow, peephole) key labels by name.
An anonymous `+`/`-` label is defined many times and means "the nearest
one", so a pass that looks it up by name finds the wrong definition: the
mode dataflow then lost track of the accumulator width (see
tests/e2e/test_unique_inline_labels.py).
"""

import re

import pytest

from r65.compiler.codegen.asm_nodes import Address, Instruction, Label
from r65.compiler.codegen.branch_fixup import fixup_nodes
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.errors import InternalCompilerError
from r65.compiler.main import compile_string

ANONYMOUS_LABEL = re.compile(r'^\s*[+-]+:|\b(B[A-Z]{2}|JMP(\.w)?)\s+[+-]+\s*(;|$)', re.M)


def _compile(body: str, decls: str = "") -> str:
    return compile_string(f"""
#[zeropage] static mut R8: u8;
#[zeropage] static mut S8: i8;
#[lowram] static mut R16: i16;
#[lowram] static mut FLAG: bool;
{decls}
#[entry]
fn main() -> ! {{
    {body}
    loop {{}}
}}
""")


@pytest.mark.parametrize("body", [
    "FLAG = !FLAG;",                 # logical not as a value
    "R16 = S8 as i16;",              # sign extension
    "R16 = R16 >> 9;",               # arithmetic shift right through XBA
    "R16 = R16 / 4;",                # signed divide by a power of two
])
def test_no_anonymous_labels(body):
    asm = _compile(body)
    assert not ANONYMOUS_LABEL.search(asm), asm


def test_two_sign_extensions_far_apart_need_no_long_branch():
    # Each sign extension's branches reach their own labels a few bytes
    # away, however much code separates the two.
    padding = "\n    ".join("R8 = R8 + 1;" for _ in range(80))
    asm = _compile(f"R16 = S8 as i16;\n    {padding}\n    R16 = R16 + (S8 as i16);")
    assert "long branch" not in asm, asm


def test_fixup_rejects_a_label_defined_twice():
    nodes = [
        Instruction(Opcode.BEQ, Address("x")),
        Label("x"),
        Instruction(Opcode.NOP),
        Label("x"),
        Instruction(Opcode.RTS),
    ]
    with pytest.raises(InternalCompilerError, match="defined more than once"):
        fixup_nodes(nodes)
