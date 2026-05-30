# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for scratch param argument sourced from a hardware index
register (X or Y).

Bug: `emit_scratch_param_argument` in call_select.py routed the u16-param
path through `_emit_load('LDA', arg_loc)`, which fails when `arg_loc` is a
HARDWARE X or Y register — the location resolver raises "Cannot resolve
hardware register X as memory operand".

Trigger: a `for i in 0..N` loop where `i` is promoted to X by the loop
register-promotion pass, and `i` is then passed as a u16 scratch param to
a callee. Repro mirrors `oam_buffer.set_size(i, 1)` in classickong.r65.
"""

from r65.tests.e2e import ExpectedState


class TestScratchParamFromHwRegister:
    """Loop counter in X/Y as u16 scratch param argument."""

    def test_loop_counter_in_x_as_u16_scratch_param(self, e2e):
        """X-held loop counter passed as u16 scratch param.

        Pre-fix: codegen error "Cannot resolve hardware register X as memory
        operand" when emitting the LDA for the scratch param store.
        Post-fix: REP+TXA copies the 16-bit X into A, then STA dp.
        """
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[lowram]
            static mut sink_lo: u8;
            #[lowram]
            static mut sink_hi: u8;
            #[lowram]
            static mut sink_count: u8;

            fn record(idx: u16, sz: u8) {
                sink_lo = idx as u8;
                sink_hi = (idx >> 8) as u8;
                sink_count = sink_count + sz;
            }

            #[entry]
            fn main() {
                sink_count = 0;
                for i in 0..16 {
                    record(i, 1);
                }
            }
        ''', ExpectedState(memory={
            0x7E0200: 15,    # last idx low byte
            0x7E0201: 0,     # last idx high byte
            0x7E0202: 16,    # called 16 times
        }))
        assert result.success, f"Failures: {result.failures}"
