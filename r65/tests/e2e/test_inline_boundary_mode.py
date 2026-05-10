# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end smoke tests for the inliner's MIR-level boundary SetMode.

Two thin runtime checks live here. The MIR-shape correctness of the
boundary `SetMode` (entry, exit, when-skipped, both directions) is
covered by `r65/tests/compiler/optimize/test_inline.py::TestBoundarySetMode`
without needing the assembler or emulator. The codegen's emitter-mode
tracking for `SetMode` is covered by
`r65/tests/compiler/codegen/test_instruction_select.py::test_select_set_mode_*`.

What's still e2e:

  - `test_m16_callee_into_m8_caller_preserves_u16_result` exercises the
    full pipeline: the inliner inserts a SEP at the exit boundary, the
    codegen's `select_set_mode` updates the emitter's tracked mode, and
    the subsequent u16 STA emits the REP it now knows it needs. Catches
    a regression in any layer (inliner, codegen tracker, peephole).
  - `test_chained_m16_inlines_preserve_value` exercises hardware
    semantics — A's high byte must survive across a SEP/REP pair via
    the hidden B register. Only the emulator can verify that; no MIR
    or asm assertion would catch a B-clobber regression.
"""

from r65.tests.e2e import ExpectedState


class TestInlineBoundaryMode:
    """Boundary SetMode insertion at inline call sites."""

    def test_m16_callee_into_m8_caller_preserves_u16_result(self, e2e):
        """Full-pipeline smoke: m16-entry / m16-exit callee inlined into
        an m8 caller, result stored as u16. Verifies inliner + codegen
        tracker + peephole compose correctly.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            #[inline(always)]
            fn make16(x @ A: u16) -> u16 { return x + 1; }

            #[entry]
            fn main() {
                let r @ A: u16 = make16(0x1000);
                RESULT = A;
            }
        ''', ExpectedState(memory={
            0x000010: 0x01,  # RESULT low  (= 0x1001 & 0xFF)
            0x000011: 0x10,  # RESULT high (= 0x1001 >> 8)
        }))
        assert result.success, f"Failures: {result.failures}, error: {result.error}"

    def test_chained_m16_inlines_preserve_value(self, e2e):
        """Two chained m16 inline calls — the first call's u16 result
        must survive the entry-boundary REP of the second call. With a
        properly bracketed boundary SetMode pair, A's high byte stashes
        in B during the exit SEP and restores when the next entry REP
        brings m16 back. Pure hardware semantics; not testable in MIR.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            #[inline(always)]
            fn add_lit(x @ A: u16) -> u16 { return x + 7; }

            #[entry]
            fn main() {
                let r @ A: u16 = add_lit(0x1000);  // 0x1007
                let s @ A: u16 = add_lit(A);       // 0x100E
                RESULT = A;
            }
        ''', ExpectedState(memory={
            0x000010: 0x0E,
            0x000011: 0x10,
        }))
        assert result.success, f"Failures: {result.failures}, error: {result.error}"
