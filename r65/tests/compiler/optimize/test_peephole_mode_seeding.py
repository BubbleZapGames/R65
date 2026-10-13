# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""The peephole's mode dataflow starts each function at its entry mode.

Function entries are reached by JSR/JSL, which the dataflow does not
follow, so unseeded it saw every function body as unreachable and the
mode-driven passes never fired. Codegen now hands the optimizer each
function's entry and exit mode. Seeding makes calls and inline asm matter:
a JSR leaves the callee's exit mode (unknown for an unknown callee), and a
SEP/REP inside `asm!` text takes effect.
"""

from r65.compiler.codegen.asm_nodes import (
    Address, Immediate, Instruction, Label, ModeChange, RawAsm,
)
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.optimize.peephole import PeepholeOptimizer


def _rep():
    return Instruction(Opcode.REP_IMMEDIATE, Immediate(0x20))


def _sep():
    return Instruction(Opcode.SEP_IMMEDIATE, Immediate(0x20))


def _optimize(body, entry=8, exits=None):
    """Optimize function `f` (entered in `entry`) after an unrelated one,
    anchored the way codegen anchors a function (m8 where unknown)."""
    nodes = [Label('g'), Instruction(Opcode.RTS),
             Label('f'), ModeChange('ACCU', entry or 8)] + body
    return PeepholeOptimizer().optimize(
        nodes, entry_modes={'f': entry}, call_exit_modes=exits or {})


def _ops(nodes):
    return [n.opcode for n in nodes if isinstance(n, Instruction)]


def test_entry_mode_makes_a_switch_redundant():
    """`f` is entered in m16, so its REP does nothing."""
    result = _optimize([_rep(), Instruction(Opcode.LDA_ABSOLUTE, Address(0x1234)),
                        Instruction(Opcode.RTS)], entry=16)
    assert Opcode.REP_IMMEDIATE not in _ops(result)


def test_without_entry_modes_the_body_is_left_alone():
    nodes = [Label('g'), Instruction(Opcode.RTS), Label('f'), _rep(),
             Instruction(Opcode.LDA_ABSOLUTE, Address(0x1234)), Instruction(Opcode.RTS)]
    assert Opcode.REP_IMMEDIATE in _ops(PeepholeOptimizer().optimize(nodes))


def test_call_returns_in_the_callees_exit_mode():
    """After `JSR wide` (exits m16) the REP is redundant, and the directives
    follow the call: the m16 immediate is sized 16-bit."""
    result = _optimize([
        Instruction(Opcode.JSR, Address('wide')), _rep(),
        Instruction(Opcode.LDA_IMMEDIATE, Immediate(0x0100)),
        Instruction(Opcode.RTS),
    ], exits={'wide': 16})
    assert Opcode.REP_IMMEDIATE not in _ops(result)
    lda = next(i for i, n in enumerate(result)
               if isinstance(n, Instruction) and n.opcode == Opcode.LDA_IMMEDIATE)
    accu = [n.bits for n in result[:lda] if isinstance(n, ModeChange) and n.flag == 'ACCU']
    assert accu[-1] == 16


def test_restore_after_a_wide_return_is_kept():
    """`f` called `wide` in m8, but it returns in m16: the SEP is needed."""
    result = _optimize([Instruction(Opcode.JSR, Address('wide')), _sep(),
                        Instruction(Opcode.RTS)], exits={'wide': 16})
    assert Opcode.SEP_IMMEDIATE in _ops(result)


def test_unknown_callee_returns_in_an_unknown_mode():
    result = _optimize([Instruction(Opcode.JSR, Address('extern_fn')), _sep(),
                        Instruction(Opcode.RTS)])
    assert Opcode.SEP_IMMEDIATE in _ops(result)


def test_inline_asm_switch_is_seen():
    """`f` is entered in m16 but its asm drops to m8, so the REP after it
    is needed."""
    result = _optimize([RawAsm('SEP #$20'), _rep(),
                        Instruction(Opcode.LDA_ABSOLUTE, Address(0x1234)),
                        Instruction(Opcode.RTS)], entry=16)
    assert Opcode.REP_IMMEDIATE in _ops(result)


def test_interrupt_entry_is_unknown():
    """A handler interrupts code in either mode: its SEP is never redundant."""
    result = _optimize([_sep(), Instruction(Opcode.LDA_ABSOLUTE, Address(0x1234)),
                        Instruction(Opcode.RTI)], entry=None)
    assert Opcode.SEP_IMMEDIATE in _ops(result)


def test_loop_switch_hoist_does_not_follow_text_past_a_return():
    """`L4` looks like a loop header - a later BRA targets it - but that
    BRA comes from `L5`, entered in m16. Hoisting the SEP above `L4` would
    run `LDA #0` in m16 on that path. (It also looped forever: the
    cross-block pass re-inserted the SEP after the label each round.)"""
    result = _optimize([
        _rep(),
        Instruction(Opcode.LDA_ABSOLUTE, Address(0x1234)),
        Instruction(Opcode.BEQ, Address('L5')),
        _sep(),
        Label('L4'), _sep(),
        Instruction(Opcode.LDA_IMMEDIATE, Immediate(0)),
        Instruction(Opcode.RTS),
        Label('L5'),
        Instruction(Opcode.LDA_ABSOLUTE, Address(0x1236)),
        Instruction(Opcode.BRA, Address('L4')),
    ])
    l4 = next(i for i, n in enumerate(result) if isinstance(n, Label) and n.name == 'L4')
    after = next(n for n in result[l4 + 1:] if isinstance(n, Instruction))
    assert after.opcode == Opcode.SEP_IMMEDIATE
