# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A u16 compare of a loop counter in X/Y against a stack slot.

CPX/CPY have no stack-relative mode, so with no scratch free the compare goes
through A as `TXA / CMP off,S`. That path never set A to 16 bits, so it
compared only the low bytes whenever A was left in m8: counting to 273 stopped
at 17.
"""

from r65.tests.e2e import ExpectedState


def test_a_u16_counter_compares_all_sixteen_bits(e2e):
    result = e2e.run('''
        #[zeropage(0x10)] static mut O: u16;
        #[zeropage(0x12)] static mut N: u16;
        #[zeropage(0x14)] static mut C: u8;

        #[inline(never)]
        fn f(v @ A: u16) -> u8 {
            let mut n: u8 = 0;
            let mut i: u16 = 0;
            while i < v + 1 { i = i + 1; n = n + 1; }
            O = i;
            return n;
        }

        #[entry]
        fn main() { N = 0x110; C = f(N); C = f(N); }
    ''', ExpectedState(memory={0x7E0010: [0x11, 0x01]}), max_instructions=50000)
    assert result.success, f"Failures: {result.failures}"
