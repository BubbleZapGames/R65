# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""On-hardware checks that an A-resident `@ A` argument survives argument setup.

`r65/tests/compiler/codegen/test_a_arg_preservation.py` asserts on emitted
instructions and runs in milliseconds; this file runs the same shapes on the
emulator, so the two disagree if the instruction-level model is ever wrong about
what the hardware does.

The motivating case is `mul8`, whose second parameter arrives in B:

    far fn mul8(multA @ A: u8, multB @ B: u8) -> u16
    fn scaled(v @ A: u8, k: u8) -> u16 { return mul8(v, k); }

`v` lives in A, and loading `k` for the `XBA` destroyed it. The value had no
memory home to be reloaded from, which is why the identical call is correct when
`v` happens to be a stack parameter.
"""

from pathlib import Path as _Path

import pytest
from r65.tests.e2e import ExpectedState

_STDLIB = _Path(__file__).parent.parent.parent.parent / "stdlib"
# math.r65 documents sneslib.r65 as a prerequisite -- it provides WRMPYA etc.
MATH = (f'include!("{_STDLIB / "sneslib.r65"}")\n'
        f'include!("{_STDLIB / "math.r65"}")\n')


def program(decls: str, body: str) -> str:
    return (MATH
            + "#[zeropage(0x20)]\nstatic mut OUT: u16;\n"
            + "#[zeropage(0x24)]\nstatic mut OUT2: u16;\n"
            + decls
            + "#[entry]\nfn main() { " + body + " }")


class TestStdlibMultiply:
    """The real shape, against the real stdlib multiply."""

    def test_a_bound_argument(self, e2e):
        result = e2e.run(
            program("fn scaled(v @ A: u8, k: u8) -> u16 { return mul8(v, k); }\n",
                    "OUT = scaled(12, 10);"),
            ExpectedState(memory={0x7E0020: [120, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_stack_bound_argument_still_correct(self, e2e):
        """The control: this shape was always correct and must stay so."""
        result = e2e.run(
            program("fn scaled(v: u8, k: u8) -> u16 { return mul8(v, k); }\n",
                    "OUT = scaled(12, 10);"),
            ExpectedState(memory={0x7E0020: [120, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_immediate_second_argument(self, e2e):
        result = e2e.run(
            program("fn scaled(v @ A: u8) -> u16 { return mul8(v, 10); }\n",
                    "OUT = scaled(12);"),
            ExpectedState(memory={0x7E0020: [120, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_same_value_squared(self, e2e):
        """`XBA` alone gives B the value but leaves A holding the old B."""
        result = e2e.run(
            program("fn sq(v @ A: u8) -> u16 { return mul8(v, v); }\n",
                    "OUT = sq(7);"),
            ExpectedState(memory={0x7E0020: [49, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_back_to_back_calls(self, e2e):
        """A save that leaked a stack byte would corrupt the frame and take the
        return address with it."""
        result = e2e.run(
            program("fn scaled(v @ A: u8, k: u8) -> u16 { return mul8(v, k); }\n",
                    "OUT = scaled(12, 10); OUT2 = scaled(3, 4);"),
            ExpectedState(memory={0x7E0020: [120, 0], 0x7E0024: [12, 0]}))
        assert result.success, f"Failures: {result.failures}"


class TestIndexTargets:
    """X and Y route through A only when the source is memory or stack."""

    XY = ("far fn addxy(a @ A: u8, p @ X: u16, q @ Y: u16) -> u16 "
          "{ return (a as u16) + p + q; }\n")

    def test_x_and_y_both_bound(self, e2e):
        """Neither index register is free, so the value goes on the stack."""
        result = e2e.run(
            program(self.XY + "fn go(v @ A: u8, p: u16, q: u16) -> u16 "
                              "{ return addxy(v, p, q); }\n",
                    "OUT = go(5, 100, 1000);"),
            ExpectedState(memory={0x7E0020: [0x51, 0x04]}))   # 5+100+1000 = 0x0451
        assert result.success, f"Failures: {result.failures}"

    def test_x_from_memory(self, e2e):
        decls = ("far fn addx(a @ A: u8, p @ X: u16) -> u16 "
                 "{ return (a as u16) + p; }\n"
                 "fn go(v @ A: u8, p: u16) -> u16 { return addx(v, p); }\n")
        result = e2e.run(program(decls, "OUT = go(5, 1000);"),
                         ExpectedState(memory={0x7E0020: [0xED, 0x03]}))
        assert result.success, f"Failures: {result.failures}"

    def test_x_from_immediate_unaffected(self, e2e):
        decls = ("far fn addx(a @ A: u8, p @ X: u16) -> u16 "
                 "{ return (a as u16) + p; }\n"
                 "fn go(v @ A: u8) -> u16 { return addx(v, 1000); }\n")
        result = e2e.run(program(decls, "OUT = go(5);"),
                         ExpectedState(memory={0x7E0020: [0xED, 0x03]}))
        assert result.success, f"Failures: {result.failures}"


class TestSixteenBitAccumulator:
    """A u16 `@ A` argument saved and restored under x16."""

    def test_u16_a_with_an_x_argument(self, e2e):
        decls = ("far fn addw(a @ A: u16, p @ X: u16) -> u16 { return a + p; }\n"
                 "fn go(v @ A: u16, p: u16) -> u16 { return addw(v, p); }\n")
        result = e2e.run(program(decls, "OUT = go(300, 1000);"),
                         ExpectedState(memory={0x7E0020: [0x14, 0x05]}))
        assert result.success, f"Failures: {result.failures}"
