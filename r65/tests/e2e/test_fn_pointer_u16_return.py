# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A call through a function pointer resumes in the callee's exit mode.

Indirect calls carried no modes, so after a u16 result (an m16 exit) the
caller went on assembling for m8: the PLA that pops a one-byte argument
pulled two, and the next RTS returned into garbage. The result was also
collected as one byte, which the wrong mode had hidden.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def run(source, expected, abi="Default"):
    result = E2ETest(abi=abi).run(source, ExpectedState(memory={0x7E0300: expected}))
    assert result.success, f"{result.error} {result.failures}"


def test_u16_result_with_a_u8_stack_argument():
    run('''
        #[lowram(0x0300)] static mut OUT: [u16; 3];
        #[lowram(0x0318)] static mut FP: fn(u8) -> u16;
        #[inline(never)] fn widen(v: u8) -> u16 { return (v as u16) + 0x100; }
        #[inline(never)] fn go(k: u8) -> u16 { return FP(k) + 1; }

        #[entry]
        fn main() {
            FP = widen;
            OUT[0] = go(5);
            OUT[1] = go(7);
            OUT[2] = 0xBEEF;
        }
    ''', [0x06, 0x01, 0x08, 0x01, 0xEF, 0xBE])


def test_far_and_near_u16_results_are_collected_whole():
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


def test_fixedstack_u16_result_then_8_bit_work():
    """FixedStack points only at parameterless functions. 8-bit work right
    after the u16 result needs the SEP that a caller still tracking m8
    leaves out, so its `LDA #imm` would be read a byte too wide."""
    run('''
        #[lowram(0x0300)] static mut OUT: [u16; 2];
        #[lowram(0x0304)] static mut BYTES: [u8; 2];
        #[lowram(0x031C)] static mut WP: fn() -> u16;
        #[inline(never)] fn wide() -> u16 { return 0x1234; }
        #[inline(never)] fn go(k: u8) -> u16 { let x: u16 = WP(); BYTES[0] = 0x42; BYTES[1] = k; return x; }

        #[entry]
        fn main() {
            WP = wide;
            OUT[0] = go(7);
            OUT[1] = 0xBEEF;
        }
    ''', [0x34, 0x12, 0xEF, 0xBE, 0x42, 0x07], abi="FixedStack")
