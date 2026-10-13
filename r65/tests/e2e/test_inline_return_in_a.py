# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
An inlined return reads A only when the producer leaves its result there.

The inliner returned the callee's value as `A` when the last instruction
was a TypeConvert, but a signed widening ends with A holding the extension
byte and writes the result through memory. `run` stored that byte as `O`.
"""

from r65.tests.e2e import ExpectedState


def test_inlined_signed_widening(e2e):
    result = e2e.run('''
        #[zeropage(0x10)] static mut V: i8;
        #[zeropage(0x12)] static mut O: i16;

        #[inline(always)]
        fn widen(k @ A: u16) -> i16 { return V as i16; }

        #[inline(never)]
        fn run(k @ A: u16) { O = widen(k); }

        #[entry]
        fn main() { V = -3; run(1); }
    ''', ExpectedState(memory={0x7E0012: [0xFD, 0xFF]}))
    assert result.success, f"Failures: {result.failures}"
