# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Unit tests for AssignmentLowerer._emit_moves_with_cycle_handling.

Tuple-return destructuring produces a set of *parallel* register copies
(dest <- source) that must be serialized without clobbering a value another
copy still needs. The previous implementation only detected 2-cycles, so a
3-register rotation (A->X->Y->A) and even a non-cyclic chain (X<-A then
Y<-X) were emitted in an order that corrupted values.

These tests run the emitted Move/Push/Pull sequence against a simulated
register file and assert each destination ends up holding the *original*
value of its source (parallel-copy semantics).
"""

import pytest

from r65.compiler.mir.nodes import HardwareRegister, Move, Push, Pull
from r65.compiler.mir.lowerers.assignment import AssignmentLowerer
from r65.compiler.errors import MIRLoweringError


class _StubBuilder:
    def __init__(self):
        self.emitted = []
        self._current_source_loc = None

    def emit(self, instr):
        self.emitted.append(instr)


def _run(copies):
    """copies: list of (dest_name, src_name). Returns the final reg file."""
    builder = _StubBuilder()
    lowerer = AssignmentLowerer(builder)
    assignments = [
        (HardwareRegister(d), HardwareRegister(s), None) for d, s in copies
    ]
    lowerer._emit_moves_with_cycle_handling(assignments)

    # Simulate. Every register involved starts holding a unique sentinel
    # equal to its own original value.
    names = {d for d, _ in copies} | {s for _, s in copies}
    regs = {n: f"orig_{n}" for n in names}
    stack = []

    for instr in builder.emitted:
        if isinstance(instr, Move):
            regs[instr.dest.name] = regs[instr.source.name]
        elif isinstance(instr, Push):
            stack.append(regs[instr.register.name])
        elif isinstance(instr, Pull):
            regs[instr.register.name] = stack.pop()
        else:
            raise AssertionError(f"unexpected instr {instr!r}")

    assert not stack, "hardware stack left unbalanced"
    return regs, builder.emitted


def _assert_parallel_copy(copies):
    regs, emitted = _run(copies)
    for dest, src in copies:
        assert regs[dest] == f"orig_{src}", (
            f"{dest} should hold original {src}; copies={copies} "
            f"emitted={[type(i).__name__ for i in emitted]}"
        )


class TestParallelCopyMoves:
    def test_empty(self):
        builder = _StubBuilder()
        lowerer = AssignmentLowerer(builder)
        lowerer._emit_moves_with_cycle_handling([])
        assert builder.emitted == []

    def test_independent_moves(self):
        # No shared registers — order irrelevant, must just be correct.
        _assert_parallel_copy([("X", "A"), ("Y", "B")])

    def test_chain_hazard_no_cycle(self):
        # Y<-X then X<-A: emitting in list order would clobber X before
        # Y reads it. Not a cycle, but still order-sensitive.
        _assert_parallel_copy([("Y", "X"), ("X", "A")])

    def test_two_cycle_with_free_temp(self):
        # A<->X swap; Y is free and used as the temporary.
        regs, _ = _run([("X", "A"), ("A", "X")])
        assert regs["X"] == "orig_A"
        assert regs["A"] == "orig_X"

    def test_three_cycle_rotation_no_free_temp(self):
        # A->X->Y->A. All of A/X/Y occupied: broken via the stack on the
        # X<->Y edge so push/pull widths match.
        _assert_parallel_copy([("X", "A"), ("Y", "X"), ("A", "Y")])

    def test_three_cycle_other_direction(self):
        # A->Y->X->A.
        _assert_parallel_copy([("Y", "A"), ("X", "Y"), ("A", "X")])

    def test_path_component(self):
        # Pure path B<-A<-X<-Y (distinct sources). Order-sensitive: must
        # emit B<-A before A<-X before X<-Y.
        _assert_parallel_copy([("B", "A"), ("A", "X"), ("X", "Y")])

    def test_two_cycle_plus_independent_chain(self):
        # 2-cycle (A X) plus an independent chain B<-Y. Y is only a source,
        # so after the chain drains it must be reusable as the swap temp
        # even though it was "involved" up front.
        _assert_parallel_copy([("A", "X"), ("X", "A"), ("B", "Y")])

    def test_two_disjoint_crossing_cycles_unsupported(self):
        # (A X) and (B Y) crossing 2-cycles, no free temp and no
        # width-safe X/Y-only edge: must raise rather than miscompile.
        with pytest.raises(MIRLoweringError):
            _run([("X", "A"), ("A", "X"), ("Y", "B"), ("B", "Y")])
