# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A block move clobbers A, X and Y — the peephole has to know that.

`MVN`/`MVP` run the accumulator down to $FFFF and walk X and Y across
the copied region. They were missing from `MODIFIES_A_OPCODES`, so
`_eliminate_redundant_loads_tracked` believed A still held the byte
count after a copy and deleted the *second* count load in a two-copy
`__init_start`:

    REP #$30 / LDA #$03 / LDX .. / LDY .. / MVN    <- A comes out $FFFF
    LDA #$03 / LDX .. / LDY .. / MVN               <- this LDA vanished

The second copy then ran with a length of $FFFF, spraying ROM across
WRAM. The bug sat latent for a long time because codegen used to emit a
`SEP #$20`/`REP #$30` pair between the two copies, and a mode change
resets the tracker; it only surfaced once that dead pair was optimized
away.
"""

import re

from r65.compiler.codegen.asm_nodes import Address, BlockMove, Immediate, Instruction
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.main import compile_string
from r65.compiler.optimize.peephole import PeepholeOptimizer


SOURCE = """
#[ram]
static mut A_STR: [u8; 4] = "AB\\0";
#[ram]
static mut B_STR: [u8; 4] = "CD\\0";

#[entry]
fn main() {
    A_STR[0] = B_STR[0];
}
"""


def test_each_block_copy_keeps_its_own_length():
    """Two ROM-to-RAM copies, two length loads."""
    asm = compile_string(SOURCE, cfg_options=["snes"])
    init = asm.split("__init_start:", 1)[1].split("RTS", 1)[0]
    moves = len(re.findall(r"^\s*MVN\b", init, re.M))
    lengths = len(re.findall(r"^\s*LDA #\$03\b", init, re.M))
    assert moves == 2
    assert lengths == moves


def test_load_after_mvn_not_treated_as_redundant():
    """The tracker must drop what it knew about A across a block move."""
    nodes = [
        Instruction(Opcode.LDA_IMMEDIATE, Immediate(0x03)),
        Instruction(Opcode.MVN, BlockMove(0x00, 0x7E)),
        Instruction(Opcode.LDA_IMMEDIATE, Immediate(0x03)),
        Instruction(Opcode.MVN, BlockMove(0x00, 0x7E)),
    ]
    opt = PeepholeOptimizer()
    result = opt.optimize(nodes)
    assert [n.opcode for n in result if isinstance(n, Instruction)] == [
        Opcode.LDA_IMMEDIATE, Opcode.MVN, Opcode.LDA_IMMEDIATE, Opcode.MVN,
    ]
