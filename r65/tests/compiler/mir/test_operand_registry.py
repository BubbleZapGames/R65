# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for the MIR operand registry (mir/nodes.py).

The registry (OPERAND_SPECS + iter_operands/map_operands) is the single source
of truth for which fields of each MIR node are register operands. These tests
guard the "footgun": a new MIRInstruction subclass that isn't added to the
registry fails loudly here instead of silently dropping out of liveness /
loop-promotion / inlining.
"""
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.nodes import (
    MIRInstruction, OPERAND_SPECS, OperandRole,
    iter_operands, map_operands,
    VirtualRegister, HardwareRegister, Immediate, Argument, ArgumentMechanism,
    Move, BinaryOp, Call, TraitDispatch, Return,
)


def vr(i):
    return VirtualRegister(id=i, type_info=BasicTypeInfo('u16'))


def hr(name):
    return HardwareRegister(name=name)


def test_registry_covers_every_instruction_subclass():
    """Every MIRInstruction subclass must be in OPERAND_SPECS (no silent drift)."""
    subclasses = set(MIRInstruction.__subclasses__())
    keys = set(OPERAND_SPECS)
    missing = {c.__name__ for c in subclasses - keys}
    extra = {c.__name__ for c in keys - subclasses}
    assert not missing, f"MIR nodes missing from OPERAND_SPECS: {missing}"
    assert not extra, f"OPERAND_SPECS has non-node keys: {extra}"


def test_iter_operands_reads_and_writes():
    """BinaryOp yields its dest (write) and left/right (read) operands."""
    instr = BinaryOp(dest=vr(1), left=vr(2), right=hr('X'), op='+',
                     type_info=BasicTypeInfo('u16'))
    all_ops = [v for _, v in iter_operands(instr)]
    assert all_ops == [vr(1), vr(2), hr('X')]
    reads = [v for _, v in iter_operands(instr, role=OperandRole.READ)]
    assert reads == [vr(2), hr('X')]
    writes = [v for _, v in iter_operands(instr, role=OperandRole.WRITE)]
    assert writes == [vr(1)]


def test_iter_operands_accepts_hr_filter_and_skips_nonreg():
    """Immediate/str/None slots are skipped; accepts_hr filters by declared type."""
    # Move.source can be Immediate -> skipped entirely.
    instr = Move(dest=vr(1), source=Immediate(5), type_info=BasicTypeInfo('u8'))
    assert [v for _, v in iter_operands(instr)] == [vr(1)]

    # Call.function (str) is never a register and self_ptr=None is skipped.
    call = Call(function='foo', args=[Argument(value=hr('A'),
                mechanism=ArgumentMechanism.REGISTER)], returns=[vr(3)])
    hr_reads = [v for _, v in iter_operands(call, role=OperandRole.READ,
                                            accepts_hr=True)]
    assert hr_reads == [hr('A')]  # the arg value; function (str) excluded


def test_map_operands_replaces_only_matching_vregs():
    """map_operands rewrites vreg slots via fn and leaves non-vregs untouched."""
    instr = BinaryOp(dest=vr(1), left=vr(2), right=Immediate(7), op='+',
                     type_info=BasicTypeInfo('u16'))
    repl = {1: vr(10), 2: vr(20)}
    map_operands(instr, lambda v: repl.get(v.id, v) if isinstance(v, VirtualRegister) else v)
    assert instr.dest == vr(10)
    assert instr.left == vr(20)
    assert instr.right == Immediate(7)  # untouched


def test_map_operands_handles_lists_and_args_and_none():
    """Return.values list, Call args/returns lists, and None self_ptr are safe."""
    ret = Return(values=[vr(1), Immediate(0)])
    map_operands(ret, lambda v: vr(99) if isinstance(v, VirtualRegister) else v)
    assert ret.values == [vr(99), Immediate(0)]

    td = TraitDispatch(self_ptr=None, args=[Argument(value=vr(2),
                       mechanism=ArgumentMechanism.STACK)], returns=[vr(3)],
                       trait_name='T', method_name='m', method_index=0)
    map_operands(td, lambda v: vr(99) if isinstance(v, VirtualRegister) else v)
    assert td.self_ptr is None
    assert td.args[0].value == vr(99)
    assert td.returns == [vr(99)]


def test_nodes_without_operands_yield_nothing():
    """A node mapped to () (e.g. Push) yields no operands and maps cleanly."""
    from r65.compiler.mir.nodes import Push
    push = Push(register=hr('A'))
    assert list(iter_operands(push)) == []
    map_operands(push, lambda v: vr(0))  # no-op, must not raise
    assert push.register == hr('A')
