# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
An inlined m16 function in one arm of an `if` returns the caller to its mode.

The inliner switched back from the callee's exit_m_mode - the ABI exit its
epilogue makes, m8 - but an inlined body has no epilogue: one ending in a call
made in m16 is still in m16. With no else arm the merge saw m16 from one
predecessor and m8 from the other, and mode tracking failed.
"""

from r65.tests.e2e import ExpectedState


def test_inlined_m16_wrapper_in_one_arm(e2e):
    result = e2e.run('''
        #[zeropage(0x20)]
        static mut SND: u16;
        #[zeropage(0x22)]
        static mut F1: u8;
        #[zeropage(0x23)]
        static mut OUT: u8;

        #[inline(never)]
        fn sound_play(word @ A: u16) { SND = word; }

        #[inline(always)]
        fn wrap(packed @ A: u16) { sound_play(packed); }

        #[inline(never)]
        fn f() {
            if F1 != 0 { wrap(0x0300 | (F1 as u16)); }
            OUT = F1 + 1;               // m8 again after the merge
        }

        #[entry]
        fn main() {
            F1 = 7;
            f();
        }
    ''', ExpectedState(memory={0x20: [0x07, 0x03], 0x23: 8}))
    assert result.success, f"Failures: {result.failures}"
