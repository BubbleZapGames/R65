# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for the JML [d] far indirect call fast path.

The fast path lowers a far indirect call through a DP-addressable function
pointer to PHK / PEA / JML [d]. Saves ~62 cycles per call vs the generic
RTS/RTL trampoline. These tests verify that the lowered call is actually
correct: control returns to the caller with the original PBR restored, the
callee's effects are visible, and chained calls don't corrupt state.

See r65/compiler/codegen/call_select.py:_emit_dp_indirect_far_call.
"""

import pytest
from r65.tests.e2e import ExpectedState


class TestFarIndirectJmlFastPath:
    """End-to-end correctness for the JML [d] fast path."""

    def test_far_indirect_via_scratch_runtime(self, e2e):
        """A far fn ptr in scratch invokes correctly: the callee runs and
        we return to the caller with PBR restored."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut S0: u8;
            #[zeropage(0x11, register)]
            static mut S1: u8;
            #[zeropage(0x12, register)]
            static mut S2: u8;
            #[zeropage(0x13, register)]
            static mut S3: u8;

            #[zeropage(0x40)]
            static mut RESULT: u8;

            far fn target() {
                RESULT = 0x42;
            }

            fn invoke(handler: far fn()) {
                handler();
            }

            #[entry]
            fn main() {
                invoke(target);
            }
        ''', ExpectedState(memory={
            0x40: 0x42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_chained_indirect_calls_through_same_dp_slot(self, e2e):
        """Two sequential far indirect calls through the same scratch slot
        execute correctly — the JML [d] sequence doesn't leave residual
        state on the stack or in DP."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut S0: u8;
            #[zeropage(0x11, register)]
            static mut S1: u8;
            #[zeropage(0x12, register)]
            static mut S2: u8;
            #[zeropage(0x13, register)]
            static mut S3: u8;

            #[zeropage(0x40)]
            static mut R0: u8;
            #[zeropage(0x41)]
            static mut R1: u8;

            far fn target_one() {
                R0 = 0x11;
            }

            far fn target_two() {
                R1 = 0x22;
            }

            fn invoke(handler: far fn()) {
                handler();
            }

            #[entry]
            fn main() {
                invoke(target_one);
                invoke(target_two);
            }
        ''', ExpectedState(memory={
            0x40: 0x11,
            0x41: 0x22,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_callee_returns_value_via_global(self, e2e):
        """Verify the callee can compute a value and store it via DP, then
        we observe the value after the call returns. The interesting part
        is that PBR is restored correctly — if it weren't, the post-call
        ``A = 0x77`` store would land in the wrong bank."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut S0: u8;
            #[zeropage(0x11, register)]
            static mut S1: u8;
            #[zeropage(0x12, register)]
            static mut S2: u8;
            #[zeropage(0x13, register)]
            static mut S3: u8;

            #[zeropage(0x40)]
            static mut CALLEE_VAL: u8;

            #[zeropage(0x41)]
            static mut POST_CALL: u8;

            far fn target() {
                CALLEE_VAL = 0x55;
            }

            fn invoke(handler: far fn()) {
                handler();
                POST_CALL = 0x77;
            }

            #[entry]
            fn main() {
                invoke(target);
            }
        ''', ExpectedState(memory={
            0x40: 0x55,
            0x41: 0x77,
        }))
        assert result.success, f"Failures: {result.failures}"
