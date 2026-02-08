"""
Tests for stack depth overflow analysis.
"""

import pytest
from r65.compiler.mir.nodes import (
    MIRFunction, MIRProgram, BasicBlock, Call, Return, VirtualRegister
)
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.attributes import InterruptAttribute, InterruptVector
from r65.compiler.analysis.stack_depth import StackDepthAnalyzer, MAX_SPILL_BYTES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_func(name, *, is_entry=False, is_far=False,
               calls=None, interrupt_attr=None,
               frame_size=0, prologue_bytes=0):
    """
    Build a minimal MIRFunction with optional Call instructions.

    *calls* is a list of callee names (strings).
    """
    instructions = []
    if calls:
        for callee in calls:
            instructions.append(
                Call(function=callee, is_far=is_far)
            )
    instructions.append(Return())

    block = BasicBlock(
        block_id=0,
        instructions=instructions,
        predecessors=[],
        successors=[],
    )

    func = MIRFunction(
        name=name,
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: block},
        entry_block_id=0,
        exit_block_ids=[0],
        is_entry=is_entry,
        is_far=is_far,
        interrupt_attr=interrupt_attr,
        vreg_allocator=VirtualRegisterAllocator(),
    )
    func.codegen_frame_size = frame_size
    func.codegen_prologue_bytes = prologue_bytes
    return func


def _make_program(*funcs):
    return MIRProgram(functions=list(funcs))


def _run(funcs, stack_lower=0x0100, stack_upper=0x01FF):
    """Build program, run analyzer, return warnings list."""
    prog = _make_program(*funcs)
    analyzer = StackDepthAnalyzer(prog, stack_lower, stack_upper)
    return analyzer.analyze()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoOverflow:
    """Cases where the stack fits — expect no W003/W005 warnings."""

    def test_single_entry_no_calls(self):
        """Entry function with small frame fits in 256-byte stack."""
        main = _make_func("main", is_entry=True, frame_size=4, prologue_bytes=2)
        warnings = _run([main])
        assert not any("W003" in w for w in warnings)
        assert not any("W005" in w for w in warnings)

    def test_shallow_chain(self):
        """main -> a -> b, all small — should fit."""
        b = _make_func("b", frame_size=2, prologue_bytes=0)
        a = _make_func("a", calls=["b"], frame_size=2, prologue_bytes=0)
        main = _make_func("main", is_entry=True, calls=["a"],
                          frame_size=2, prologue_bytes=0)
        warnings = _run([main, a, b])
        assert not any("W003" in w for w in warnings)

    def test_custom_stack_fits(self):
        """Large custom stack accommodates deep chain."""
        c = _make_func("c", frame_size=10, prologue_bytes=0)
        b = _make_func("b", calls=["c"], frame_size=10, prologue_bytes=0)
        main = _make_func("main", is_entry=True, calls=["b"],
                          frame_size=10, prologue_bytes=0)
        # root (non-leaf): 10+0+6 = 16
        # call b (non-leaf): 2 + 10+0+6 = 18
        # call c (leaf): 2 + 10+0 = 12
        # total = 16 + 18 + 12 = 46
        warnings = _run([main, b, c], stack_lower=0x0100, stack_upper=0x01FF)
        assert not any("W003" in w for w in warnings)


class TestOverflow:
    """W003: worst-case depth exceeds stack size."""

    def test_deep_chain_overflows(self):
        """A chain of calls that exceeds the 256-byte default stack."""
        # Build a chain: main -> f1 -> f2 -> ... -> f30
        # Each function has frame_size=8, prologue_bytes=0
        # Non-leaf own_cost = 8 + 6 = 14, leaf own_cost = 8
        # near call edge cost = 2
        # total = 14 + 29*(2+14) + (2+8) = 14 + 464 + 10 = 488 > 256
        funcs = []
        prev_name = None
        for i in range(30, 0, -1):
            name = f"f{i}"
            calls = [prev_name] if prev_name else None
            funcs.append(_make_func(name, frame_size=8, prologue_bytes=0,
                                    calls=calls))
            prev_name = name

        main = _make_func("main", is_entry=True, calls=[prev_name],
                          frame_size=8, prologue_bytes=0)
        funcs.append(main)

        warnings = _run(funcs)
        w003 = [w for w in warnings if "W003" in w]
        assert len(w003) == 1
        assert "exceeds stack size" in w003[0]

    def test_tiny_stack_overflows(self):
        """Even two functions overflow a tiny (16-byte) stack."""
        b = _make_func("b", frame_size=4, prologue_bytes=0)
        main = _make_func("main", is_entry=True, calls=["b"],
                          frame_size=4, prologue_bytes=0)
        # root (non-leaf): 4+0+6 = 10, call b (leaf): 2+4+0 = 6, total = 16
        # stack_size = 16, total = 16 — exactly fits
        # Shrink stack by 1 to trigger overflow
        warnings = _run([main, b], stack_lower=0x0100, stack_upper=0x010E)
        # stack_size = 15, total = 16 > 15
        assert any("W003" in w for w in warnings)


class TestRecursion:
    """W004: recursive call chains."""

    def test_direct_recursion(self):
        """Function calls itself."""
        f = _make_func("f", calls=["f"])
        main = _make_func("main", is_entry=True, calls=["f"])
        warnings = _run([main, f])
        w004 = [w for w in warnings if "W004" in w]
        assert len(w004) == 1
        assert "recursive" in w004[0].lower()

    def test_mutual_recursion(self):
        """a -> b -> a cycle."""
        a = _make_func("a", calls=["b"])
        b = _make_func("b", calls=["a"])
        main = _make_func("main", is_entry=True, calls=["a"])
        warnings = _run([main, a, b])
        w004 = [w for w in warnings if "W004" in w]
        assert len(w004) >= 1


class TestInterrupts:
    """W005: combined normal + interrupt depth exceeds stack."""

    def test_interrupt_adds_to_depth(self):
        """Normal chain + NMI handler together overflow."""
        # NMI handler (leaf): 4 (hw push) + 10 (auto-save) + own_cost
        nmi = _make_func("nmi_handler",
                         interrupt_attr=InterruptAttribute(
                             name="interrupt",
                             vector=InterruptVector.NMI, preserve=True),
                         frame_size=20, prologue_bytes=0)
        # nmi own_cost (leaf) = 20 + 0 = 20; total interrupt = 4 + 10 + 20 = 34

        # main chain uses most of the stack
        main = _make_func("main", is_entry=True, frame_size=200,
                          prologue_bytes=10)
        # main own_cost (leaf) = 200 + 10 = 210; normal = 210
        # combined = 210 + 34 = 244  — fits in 256-byte stack

        # Use a stack where 244 overflows
        warnings = _run([main, nmi], stack_lower=0x0100, stack_upper=0x01F2)
        # stack_size = 243, combined = 244 > 243
        assert any("W005" in w for w in warnings)

    def test_interrupt_preserve_false(self):
        """preserve=false skips auto-save bytes."""
        nmi = _make_func("nmi_handler",
                         interrupt_attr=InterruptAttribute(
                             name="interrupt",
                             vector=InterruptVector.NMI, preserve=False),
                         frame_size=0, prologue_bytes=0)
        # interrupt cost = 4 (hw) + 0 (no auto-save) + 0 (leaf) = 4
        main = _make_func("main", is_entry=True, frame_size=0,
                          prologue_bytes=0)
        # normal = 0 (leaf);  combined = 0 + 4 = 4
        warnings = _run([main, nmi], stack_lower=0x0100, stack_upper=0x0103)
        # stack_size = 4, combined = 4 — exactly fits, no warning
        assert not any("W005" in w for w in warnings)

    def test_no_w005_without_both_roots(self):
        """W005 only fires if there are BOTH entry and interrupt roots."""
        nmi = _make_func("nmi_handler",
                         interrupt_attr=InterruptAttribute(
                             name="interrupt",
                             vector=InterruptVector.NMI, preserve=True),
                         frame_size=200, prologue_bytes=0)
        # No entry function — W005 should not fire
        warnings = _run([nmi])
        assert not any("W005" in w for w in warnings)


class TestDiamondCallGraph:
    """Diamond: main -> a -> c, main -> b -> c. Worst path wins."""

    def test_diamond_takes_worst_path(self):
        """The deeper branch (through a) determines the depth, not summed."""
        c = _make_func("c", frame_size=4, prologue_bytes=0)
        a = _make_func("a", calls=["c"], frame_size=50, prologue_bytes=0)
        b = _make_func("b", calls=["c"], frame_size=2, prologue_bytes=0)
        main = _make_func("main", is_entry=True, calls=["a", "b"],
                          frame_size=2, prologue_bytes=0)
        # Worst path: main -> a -> c
        # main (non-leaf): 2+0+6 = 8
        # call a (non-leaf): 2 + 50+0+6 = 58
        # call c (leaf): 2 + 4+0 = 6
        # total = 8 + 58 + 6 = 72  (fits in 256)
        warnings = _run([main, a, b, c])
        assert not any("W003" in w for w in warnings)


class TestEntryOnly:
    """Entry function has no return address overhead from a caller."""

    def test_entry_no_return_addr(self):
        """Leaf entry function cost is just frame + prologue, no spill."""
        main = _make_func("main", is_entry=True, frame_size=0,
                          prologue_bytes=0)
        # own_cost (leaf) = 0 + 0 = 0
        # A 1-byte stack should suffice
        warnings = _run([main], stack_lower=0x0100, stack_upper=0x0100)
        assert not any("W003" in w for w in warnings)

    def test_entry_barely_overflows(self):
        """Entry with frame_size that barely exceeds a tiny stack."""
        main = _make_func("main", is_entry=True, frame_size=5,
                          prologue_bytes=0)
        # own_cost (leaf) = 5 + 0 = 5
        warnings = _run([main], stack_lower=0x0100, stack_upper=0x0103)
        # stack_size = 4, own_cost = 5 > 4
        assert any("W003" in w for w in warnings)


class TestFarCalls:
    """Far calls push 3-byte return address instead of 2."""

    def test_far_call_costs_three(self):
        """Far callee adds 3 bytes for return address instead of 2."""
        far_func = _make_func("far_func", is_far=True, frame_size=0,
                              prologue_bytes=0)
        main = _make_func("main", is_entry=True, calls=["far_func"],
                          frame_size=0, prologue_bytes=0)
        # main own (non-leaf): 0+0+6 = 6
        # call far_func (leaf): 3 + 0+0 = 3
        # total = 9
        warnings = _run([main, far_func], stack_lower=0x0100, stack_upper=0x0108)
        # stack = 9, total = 9 — fits
        assert not any("W003" in w for w in warnings)

        # Shrink by 1
        warnings = _run([main, far_func], stack_lower=0x0100, stack_upper=0x0107)
        # stack = 8, total = 9 — overflows
        assert any("W003" in w for w in warnings)


class TestLeafVsNonLeaf:
    """Leaf functions don't get spill byte allowance."""

    def test_leaf_no_spill_bytes(self):
        """A leaf function's own cost is just frame + prologue."""
        leaf = _make_func("leaf", frame_size=10, prologue_bytes=2)
        main = _make_func("main", is_entry=True, calls=["leaf"],
                          frame_size=0, prologue_bytes=0)
        # main (non-leaf): 0+0+6 = 6
        # call leaf: 2 + 10+2 = 14  (no spill bytes for leaf)
        # total = 20
        warnings = _run([main, leaf], stack_lower=0x0100, stack_upper=0x0113)
        # stack = 20, total = 20 — fits
        assert not any("W003" in w for w in warnings)

        # With spill bytes it would be 26 — verify it doesn't overflow at 20
        warnings = _run([main, leaf], stack_lower=0x0100, stack_upper=0x0112)
        # stack = 19, total = 20 — overflows
        assert any("W003" in w for w in warnings)

    def test_non_leaf_gets_spill_bytes(self):
        """A non-leaf function gets MAX_SPILL_BYTES added."""
        inner = _make_func("inner", frame_size=0, prologue_bytes=0)
        mid = _make_func("mid", calls=["inner"], frame_size=0, prologue_bytes=0)
        main = _make_func("main", is_entry=True, calls=["mid"],
                          frame_size=0, prologue_bytes=0)
        # main (non-leaf): 0+0+6 = 6
        # call mid (non-leaf): 2 + 0+0+6 = 8
        # call inner (leaf): 2 + 0+0 = 2
        # total = 16
        expected = MAX_SPILL_BYTES + (2 + MAX_SPILL_BYTES) + 2
        warnings = _run([main, mid, inner],
                        stack_lower=0x0100,
                        stack_upper=0x0100 + expected - 1)
        # Exactly fits
        assert not any("W003" in w for w in warnings)

        # One byte less — overflows
        warnings = _run([main, mid, inner],
                        stack_lower=0x0100,
                        stack_upper=0x0100 + expected - 2)
        assert any("W003" in w for w in warnings)
