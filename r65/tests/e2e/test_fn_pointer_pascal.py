# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Under Pascal a call through a function pointer pushes the result space.

A Pascal callee stores its result above its parameters, in space the caller
pushed before them. Only direct calls pushed it, so a callee reached through
a pointer stored its result over the caller's return address.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def run(source, expected):
    result = E2ETest(abi="Pascal").run(source, ExpectedState(memory={0x7E0300: expected}))
    assert result.success, f"{result.error} {result.failures}"


def test_u8_result():
    run('''
        #[lowram(0x0300)] static mut OUT: [u8; 3];
        #[lowram(0x0318)] static mut FP: fn(u8) -> u8;
        #[inline(never)] fn inc(v: u8) -> u8 { return v + 1; }
        #[inline(never)] fn go(k: u8) -> u8 { return FP(k) + 10; }

        #[entry]
        fn main() { FP = inc; OUT[0] = go(5); OUT[1] = go(7); OUT[2] = 0x5A; }
    ''', [16, 18, 0x5A])


def test_far_and_near_u16_results():
    run('''
        #[lowram(0x0300)] static mut OUT: [u16; 3];
        #[lowram(0x0318)] static mut FP: fn(u16) -> u16;
        #[lowram(0x031C)] static mut NP: near fn(u16) -> u16;
        #[inline(never)] fn inc16(v: u16) -> u16 { return v + 1; }
        #[inline(never)] near fn dec16(v: u16) -> u16 { return v - 1; }
        #[inline(never)] fn go(w: u16) -> u16 { let r: u16 = FP(w); return r + NP(w); }

        #[entry]
        fn main() {
            FP = inc16;
            NP = dec16;
            OUT[0] = go(0x1000);
            OUT[1] = FP(0x00FF);
            OUT[2] = 0xBEEF;
        }
    ''', [0x00, 0x20, 0x00, 0x01, 0xEF, 0xBE])
