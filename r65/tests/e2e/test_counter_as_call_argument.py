# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for a loop counter, promoted into X or Y, passed to a call
whose callee clobbers both index registers. Promotion rule:
tests/compiler/codegen/test_loop_promotion.py::TestCounterAsCallArgument.
"""

from r65.tests.e2e import ExpectedState


class TestCounterAsCallArgument:
    def test_values(self, e2e):
        result = e2e.run('''
            #[lowram(0x0300)] static mut LOG: [u16; 16];
            #[lowram(0x0340)] static mut N: u16;
            #[inline(never)] fn note(v: u16) { LOG[N] = v; N = N + 1; X = 0x7777; Y = 0x7777; }

            #[inline(never)] fn on_stack(v: u16) { note(v); }
            #[inline(never)] fn narrow(v: u8) { note(v as u16); }
            #[inline(never)] fn in_a(v @ A: u8) { note(v as u16 | 0x100); }
            #[inline(never)] fn in_y(v @ Y: u16) { note(v | 0x200); }

            #[inline(never)] fn go() {
                let mut i: u16 = 0;
                while i < 2 { on_stack(i); in_y(i); i = i + 1; }
                let mut j: u8 = 0;
                while j < 2 { narrow(j); in_a(j); j = j + 1; }
            }

            #[entry]
            fn main() { N = 0; go(); }
        ''', ExpectedState(memory={
            0x0300: [0x00, 0x00, 0x00, 0x02, 0x01, 0x00, 0x01, 0x02,
                     0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_beside_a_value_in_a(self, e2e):
        """`pair(i + 0x10, i)`: A holds the first argument while the counter is
        pushed, so A may not be parked in the counter's register."""
        result = e2e.run('''
            #[lowram(0x0300)] static mut LOG: [u16; 8];
            #[lowram(0x0340)] static mut N: u16;
            #[inline(never)] fn note(v: u16) { LOG[N] = v; N = N + 1; X = 0x7777; Y = 0x7777; }
            #[inline(never)] fn pair(a: u16, b: u16) { note(a); note(b); }
            #[inline(never)] fn go() {
                let mut i: u16 = 0;
                while i < 2 { pair(i + 0x10, i); i = i + 1; }
            }
            #[entry]
            fn main() { N = 0; go(); }
        ''', ExpectedState(memory={
            0x0300: [0x10, 0x00, 0x00, 0x00, 0x11, 0x00, 0x01, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"
