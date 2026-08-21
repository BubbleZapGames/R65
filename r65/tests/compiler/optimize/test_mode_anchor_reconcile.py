# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A label's `.ACCU` anchor must not outlive the SEP/REP that made it true.

`compute_modes` *freezes* a label it has an entry-mode seed for: the
anchored `.ACCU` wins and real predecessor edges are ignored. That is
right for a function entry, which the asm dataflow can only reach
through a JSR/JSL it does not model, and wrong for an ordinary block
label — peephole may have deleted the block-entry SEP/REP that made the
anchor true, and a stale anchor makes WLA-DX *encode* the block's
immediates at the wrong width. It doesn't misbehave at runtime; it
assembles to different bytes.

The bug that motivated this: `_eliminate_dead_mode_changes` dropped a
`SEP #$20` whose only effect was to be overwritten by the next
instruction's `REP #$20`. The block was then entered in m16, but its
anchor still claimed m8, and `CMP #$FFFF` inside it was sized as an
8-bit immediate — which WLA-DX rejects outright ("INPUT_NUMBER: Out of
8-bit range").
"""

from r65.compiler.codegen.asm_nodes import (
    Address, Immediate, Instruction, Label, ModeChange, RawAsm,
)
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.optimize.mode_directive_rewrite import normalize_mode_directives


def _anchor_for(nodes, label_name):
    """The bits of the `.ACCU` directly after `label_name`, or None."""
    for i, node in enumerate(nodes):
        if isinstance(node, Label) and node.name == label_name:
            for nj in nodes[i + 1:]:
                if isinstance(nj, ModeChange) and nj.flag == 'ACCU':
                    return nj.bits
                if isinstance(nj, Instruction):
                    return None
    return None


def _reached_only_by_branch(tail):
    """A function whose second block is entered in m16 from one branch."""
    return [
        Label('fn'), ModeChange('ACCU', 8),
        Instruction(Opcode.REP_IMMEDIATE, Immediate(0x20)),
        Instruction(Opcode.LDA_ABSOLUTE, Address(0x1234)),
        Instruction(Opcode.CMP_IMMEDIATE, Immediate(0x0400)),
        Instruction(Opcode.BCC, Address('blk')),
        Instruction(Opcode.RTS),
    ] + tail + [
        # Stale: the SEP that made `.ACCU 8` true here is gone.
        Label('blk'), ModeChange('ACCU', 8),
        Instruction(Opcode.CMP_IMMEDIATE, Immediate(0xFFFF)),
        Instruction(Opcode.RTS),
    ]


def test_stale_block_anchor_corrected_from_cfg():
    """One m16 branch in, no other edge — the CFG overrules the anchor."""
    result = normalize_mode_directives(_reached_only_by_branch([]))
    assert _anchor_for(result, 'blk') == 16


def test_function_entry_anchor_survives_a_call():
    """A JSL target's arriving mode isn't constrained by branches."""
    called = _reached_only_by_branch(
        [Instruction(Opcode.JSL, Address('blk'))]
    )
    result = normalize_mode_directives(called)
    assert _anchor_for(result, 'blk') == 8


def test_address_taken_label_anchor_survives():
    """A label loaded as a pointer can be entered by an edge we can't see."""
    escaped = _reached_only_by_branch(
        [Instruction(Opcode.LDA_IMMEDIATE, Immediate('<blk'))]
    )
    result = normalize_mode_directives(escaped)
    assert _anchor_for(result, 'blk') == 8


def test_jump_table_entry_anchor_survives():
    """`.DW blk` in a dispatch table is an entry the dataflow can't model."""
    tabled = _reached_only_by_branch([RawAsm(".DW blk")])
    result = normalize_mode_directives(tabled)
    assert _anchor_for(result, 'blk') == 8


def test_unreachable_label_anchor_survives():
    """A vector-table handler has no in-file edges at all — trust codegen."""
    nodes = [
        Label('fn'), ModeChange('ACCU', 8),
        Instruction(Opcode.RTS),
        Label('nmi_handler'), ModeChange('ACCU', 8),
        Instruction(Opcode.RTI),
    ]
    result = normalize_mode_directives(nodes)
    assert _anchor_for(result, 'nmi_handler') == 8
