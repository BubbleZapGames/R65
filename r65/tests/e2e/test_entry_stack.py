# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""`#[stack(lo, hi)]` moves S at the entry point, checked on the hardware.

The entry prologue set S with `REP #$20 / LDA #top / TCS` before its
`CLC / XCE`. At reset the CPU is in emulation mode, where REP cannot clear
M: the LDA took a one-byte immediate, the TCS was never reached, and S
stayed at $01FF. The low-RAM allocator had trusted the declared region and
put a 512-byte array at $0100, so the stack grew down into it. S is now set
after the XCE (function_gen._emit_entry_setup).
"""

from r65.tests.e2e import ExpectedState


def test_declared_stack_is_used_from_reset(e2e):
    result = e2e.run('''
        #[stack(0x1F00, 0x1FFF)]
        #[lowram] static mut BIG: [u8; 512];
        #[zeropage(0x40)] static mut OUT: u16;

        #[inline(never)]
        fn deep(n: u16) -> u16 {
            let mut s: u16 = n;
            let mut i: u16 = 0;
            while i < 4 { s = s + (BIG[0xF0 + i] as u16); i = i + 1; }
            return s;
        }

        #[entry]
        fn main() {
            let mut i: u16 = 0;
            while i < 512 { BIG[i] = 0xA5; i = i + 1; }
            OUT = deep(1);
        }
    ''', ExpectedState(
        SP=0x1FFF,
        memory={
            0x7E0040: [0x95, 0x02],          # 1 + 4 * 0xA5
            0x7E01F0: [0xA5] * 16,           # where S used to be
        }), max_instructions=20000)
    assert result.success, f"Failures: {result.failures}"
