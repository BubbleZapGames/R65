# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Near pointers compare all sixteen bits.

The compare emitter sized its operands by integer type name, so a near
pointer took the 8-bit path: `&BUF[0] == &BUF[256]` ($0300 vs $0400) compared
only the low bytes and came out equal.
"""

from r65.tests.e2e import ExpectedState


def test_near_pointers_differing_only_in_the_high_byte(e2e):
    result = e2e.run('''
        #[lowram(0x0300)] static mut BUF: [u8; 600];
        #[zeropage(0x10)] static mut P: *u8;
        #[zeropage(0x12)] static mut Q: *u8;
        #[zeropage(0x14)] static mut O: [u8; 2];

        #[entry]
        fn main() {
            P = &BUF[0];
            Q = &BUF[256];
            if P == Q { O[0] = 1; } else { O[0] = 2; }
            if P < Q { O[1] = 3; } else { O[1] = 4; }
        }
    ''', ExpectedState(memory={0x7E0014: [2, 3]}))
    assert result.success, f"Failures: {result.failures}"
