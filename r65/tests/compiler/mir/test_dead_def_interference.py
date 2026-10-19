# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A dead Move still writes its slot (LivenessAnalyzer.interferes).

`let mut digit: u8 = 0; ... digit = 0;` left `digit = 0` read by nothing
once web splitting gave it a vreg of its own; slot allocation put it in the
slot of `v`, live across it, and its STA overwrote `v` (u16_to_dec printed
the wrong digits). The copy a Move makes from its source is the one value
it may share a slot with.
"""

from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.liveness import LivenessAnalyzer
from r65.compiler.mir.nodes import (
    BasicBlock, Immediate, MIRFunction, Move, Return, VirtualRegister,
)

U8 = BasicTypeInfo('u8')


def _analyzer(instrs):
    func = MIRFunction(name='t', blocks={0: BasicBlock(block_id=0, instructions=instrs)})
    analyzer = LivenessAnalyzer(func)
    analyzer.analyze()
    return analyzer


def test_dead_move_interferes_with_a_value_live_across_it():
    v, dead = VirtualRegister(0, U8), VirtualRegister(1, U8)
    a = _analyzer([Move(dest=v, source=Immediate(5), type_info=U8),
                   Move(dest=dead, source=Immediate(0), type_info=U8),
                   Return(values=[v])])
    assert a.interferes(v, dead) and a.interferes(dead, v)


def test_dead_copy_of_a_live_value_may_share_its_slot():
    v, copy = VirtualRegister(0, U8), VirtualRegister(1, U8)
    a = _analyzer([Move(dest=v, source=Immediate(5), type_info=U8),
                   Move(dest=copy, source=v, type_info=U8),
                   Return(values=[v])])
    assert not a.interferes(v, copy)
