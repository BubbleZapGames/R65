# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""The register effects of a call (mir/call_effects.py)."""

from types import SimpleNamespace

from r65.compiler.hir.types import BasicTypeInfo, MultiReturnTypeInfo
from r65.compiler.mir.call_effects import call_effects
from r65.compiler.mir.nodes import (
    Argument, ArgumentMechanism, Call, HardwareRegister, Immediate, TraitDispatch,
)
from r65.compiler.typeck.processor_mode import ModeState

U8, U16 = BasicTypeInfo('u8'), BasicTypeInfo('u16')


def _arg(mechanism, reg=None):
    return Argument(value=Immediate(0), mechanism=mechanism,
                    location=HardwareRegister(reg) if reg else None)


def test_results():
    assert call_effects(Call(function='f')).results == ()
    assert call_effects(Call(function='f', callee_return_type=U8)).results == ('A',)
    pair = MultiReturnTypeInfo(element_types=[U8, U8])
    assert call_effects(Call(function='f', callee_return_type=pair)).results == ('A', 'B')
    assert call_effects(Call(function='f', callee_return_type=pair,
                             callee_entry_m_mode=ModeState.M16)).results == ('A', 'X')
    wide = MultiReturnTypeInfo(element_types=[U8, U16, U16])
    assert call_effects(Call(function='f', callee_return_type=wide)).results == ('A', 'X', 'Y')
    void = TraitDispatch(callee_return_type=BasicTypeInfo('void'))
    assert call_effects(void).results == ()


def test_inferred_preserves():
    summaries = {'f': frozenset({'X'})}
    assert call_effects(Call(function='f'), summaries).inferred == {'Y'}
    assert call_effects(Call(function='f')).inferred == frozenset()
    assert call_effects(Call(function='g'), summaries).inferred == frozenset()
    # Not a register an argument goes to, self included.
    assert call_effects(Call(function='f', args=[_arg(ArgumentMechanism.REGISTER, 'Y')]),
                        summaries).inferred == frozenset()
    assert call_effects(Call(function='f', args=[_arg(ArgumentMechanism.SELF_Y)]),
                        summaries).inferred == frozenset()
    # Nothing with stack arguments, for a builtin, or through a dispatch.
    assert call_effects(Call(function='f', args=[_arg(ArgumentMechanism.STACK)]),
                        summaries).inferred == frozenset()
    assert call_effects(Call(function='f', builtin_name='mul8'), summaries).inferred == frozenset()
    assert call_effects(TraitDispatch(), summaries).preserved == frozenset()


def test_may_write():
    keeps_x = Call(function='f', preserves_attr=SimpleNamespace(registers=['A', 'X']))
    effects = call_effects(keeps_x, {'f': frozenset({'X'})})
    assert effects.preserved == {'A', 'X', 'Y'}
    assert not effects.may_write('Y')
    assert effects.may_write('Y', declared_only=True)
    assert not effects.may_write('A')
    assert effects.may_write('B')     # A's high byte: never preserved
