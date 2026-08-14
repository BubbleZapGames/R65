# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for a binary operation whose operands are the same value.

`x + x` has no "ADC A" addressing mode to reach for, so when no scratch register
is free the operand is pushed and the operation runs against the stack slot. The
pushed byte then has to come off the stack — and a bare `PLA` lands in A, which
by then holds the *result*, not the operand. That silently made `x + x` evaluate
to `x` and `x - x` evaluate to `x` rather than 0.

`&` and `|` masked the bug: for those the result equals the restored operand, so
the wrong value happened to be the right one. Any test using only those would
have passed throughout.
"""

import pytest
from r65.tests.e2e import ExpectedState

# The stdlib scratch bank rather than a hand-picked address: the FixedStack ABI
# reserves zero-page ranges of its own, and a collision is a compile error.
from pathlib import Path as _Path
SCRATCH = f'include!("{_Path(__file__).parent.parent.parent.parent / "stdlib" / "scratch_regs.r65"}")\n' 


def program(body: str, prelude: str = "") -> str:
    return (prelude
            + "#[zeropage(0x10)]\nstatic mut V: u8;\n"
            + "#[zeropage(0x11)]\nstatic mut OUT: u8;\n"
            + "#[entry]\nfn main() { V = 5; " + body + " }")


class TestRepeatedOperand:
    """Both operands the same, with no scratch register available."""

    @pytest.mark.parametrize("op,expected", [
        ("+", 10), ("-", 0), ("^", 0), ("&", 5), ("|", 5),
    ])
    def test_local_against_itself(self, e2e, op, expected):
        result = e2e.run(program(f"let t: u8 = V; OUT = t {op} t;"),
                         ExpectedState(memory={0x7E0011: expected}))
        assert result.success, f"t {op} t: {result.failures}"

    @pytest.mark.parametrize("op,expected", [
        ("+", 10), ("-", 0), ("^", 0), ("&", 5), ("|", 5),
    ])
    def test_static_against_itself(self, e2e, op, expected):
        result = e2e.run(program(f"OUT = V {op} V;"),
                         ExpectedState(memory={0x7E0011: expected}))
        assert result.success, f"V {op} V: {result.failures}"

    def test_still_correct_with_a_scratch_register(self, e2e):
        """The scratch path was always correct; it must stay that way."""
        result = e2e.run(program("OUT = V - V;", prelude=SCRATCH),
                         ExpectedState(memory={0x7E0011: 0}))
        assert result.success, f"Failures: {result.failures}"

    def test_distinct_operands_unaffected(self, e2e):
        result = e2e.run(program("let t: u8 = V; let u: u8 = V; OUT = t + u;"),
                         ExpectedState(memory={0x7E0011: 10}))
        assert result.success, f"Failures: {result.failures}"

    def test_literal_operand_unaffected(self, e2e):
        result = e2e.run(program("let t: u8 = V; OUT = t + 3;"),
                         ExpectedState(memory={0x7E0011: 8}))
        assert result.success, f"Failures: {result.failures}"

    def test_chained_self_operations(self, e2e):
        """Two in a row, so a stack left unbalanced by the fix would show."""
        result = e2e.run(program("let t: u8 = V; let d: u8 = t + t; OUT = d + d;"),
                         ExpectedState(memory={0x7E0011: 20}))
        assert result.success, f"Failures: {result.failures}"

    def test_stack_stays_balanced(self, e2e):
        """The push/pop pair must net to zero — a leak would corrupt the frame
        and take the return address with it."""
        result = e2e.run('''
            #[zeropage(0x10)] static mut V: u8;
            #[zeropage(0x11)] static mut OUT: u8;
            fn doubled(n: u8) -> u8 { return n + n; }
            #[entry]
            fn main() { V = 5; OUT = doubled(V) + doubled(V); }
        ''', ExpectedState(memory={0x7E0011: 20}))
        assert result.success, f"Failures: {result.failures}"

    def test_sixteen_bit(self, e2e):
        result = e2e.run('''
            #[zeropage(0x10)] static mut V: u16;
            #[zeropage(0x12)] static mut OUT: u16;
            #[entry]
            fn main() { V = 300; let t: u16 = V; OUT = t + t; }
        ''', ExpectedState(memory={0x7E0012: [0x58, 0x02]}))
        assert result.success, f"Failures: {result.failures}"
