# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Frames shrunk to the locals left on the stack, checked on the hardware.

`RegisterAllocator._compact_frame` drops the frame bytes of locals that went
to scratch. A function keeping one local across a call keeps a frame for it
alone, and one whose locals all fit in scratch has none; both must still
read their values and return the right one (the shape is pinned in
tests/compiler/codegen/test_frame_compaction.py).
"""

from r65.tests.e2e import ExpectedState
from r65.tests.e2e.framework import E2ETest


def test_frames_without_their_scratch_locals():
    result = E2ETest(abi="FixedStack").run('''
        #[lowram] static mut BUF: [u8; 8];
        #[zeropage(0x20)] static mut ALL_SCRATCH: u16;
        #[zeropage(0x22)] static mut ACROSS_CALL: u16;

        #[inline(never)]
        fn bump(v: u16) -> u16 { return v + 5; }

        #[inline(never)]
        fn sums() -> u16 {
            let mut s: u16 = 0;
            let mut t: u16 = 0;
            let mut i: u16 = 0;
            while i < 8 { s = s + (BUF[i] as u16); t = t ^ s; i = i + 1; }
            return s + t;
        }

        #[inline(never)]
        fn keeps(k: u16) -> u16 {
            let keep: u16 = k + 0x100;
            let mut s: u16 = 0;
            let mut i: u16 = 0;
            while i < 8 { s = s + (BUF[i] as u16); i = i + 1; }
            let r: u16 = bump(s);
            return keep + r;
        }

        #[entry]
        fn main() {
            let mut i: u16 = 0;
            while i < 8 { BUF[i] = (i as u8) + 1; i = i + 1; }
            ALL_SCRATCH = sums();     // s = 36; t = 1^3^6^10^15^21^28^36 = 44
            ACROSS_CALL = keeps(7);   // 0x107 + 36 + 5
        }
    ''', ExpectedState(memory={
        0x7E0020: [36 + 44, 0],
        0x7E0022: [0x07 + 36 + 5, 0x01],
    }), max_instructions=20000)
    assert result.success, f"Failures: {result.failures}"
