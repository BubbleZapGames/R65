# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""D, S, DBR and PBR read as values, and D/DBR restore from a local.

None of them has a load or store instruction, so every read goes through A
(TDC, TSC, PHB/PLA, PHK/PLA). Each used to fail: `let d = D;` crashed with a
KeyError (the local was "kept in D"), `W = D;` had no store, a comparison
had no route, `A = D;` and `D = W;` recursed forever, and `D = saved;`
had no load.
"""

import pytest

from r65.tests.e2e import ExpectedState


def test_read_compare_and_restore(e2e):
    result = e2e.run('''
        // Absolute, not zeropage: with D moved, a direct-page store would
        // land at D + offset.
        #[lowram(0x0300)] static mut OUT: [u16; 4];
        #[lowram(0x0308)] static mut BYTES: [u8; 4];

        #[inline(never)]
        fn probe() {
            let saved = D;                      // a local read of D
            D = 0x1234;
            OUT[0] = D;                         // straight to memory
            if D == 0x1234 { BYTES[0] = 1; }    // compared
            D = saved;                          // restored from the local
            OUT[1] = D;

            let bank = DBR;
            DBR = 0x7E;
            BYTES[1] = DBR;
            DBR = bank;
            BYTES[2] = DBR;
            BYTES[3] = PBR;
        }

        #[entry]
        fn main() {
            BYTES[0] = 0;
            probe();
        }
    ''', ExpectedState(memory={
        0x7E0300: [0x34, 0x12, 0x00, 0x00],    # D while set, then restored to 0
        0x7E0308: [1, 0x7E, 0x00, 0x00],       # compare hit, DBR set/restored, PBR
    }))
    assert result.success, f"Failures: {result.failures} {result.error}"


def test_b_survives_a_later_load_into_a(e2e):
    """`B = t` then a load into A, then a read of B.

    The A-to-B transfer's XBA was tracked as a swap, so the read of B saw
    "A already holds B" and skipped its XBA: R got Q's 9, not t's 7.
    """
    result = e2e.run('''
        #[zeropage(0xC0)] static mut P: u8;
        #[zeropage(0xC1)] static mut Q: u8;
        #[zeropage(0xC2)] static mut R: u8;
        #[entry]
        fn main() -> ! { P = 5; Q = 9; let t: u8 = P + 2; B = t; A = Q; R = B; loop {} }
    ''', ExpectedState(memory={0x7E00C2: [7]}))
    assert result.success, f"Failures: {result.failures} {result.error}"


@pytest.mark.parametrize("opt", ["-O0", "-O1", "-O2"])
def test_b_in_m16_keeps_the_live_accumulator_value(e2e, opt):
    """In m16, B is the high byte of A: reaching it with XBA must not cost the
    16-bit value still live in A, which waits on the stack instead."""
    result = e2e.run('''
        #[lowram(0x0300)] static mut HI: u8;
        #[lowram(0x0302)] static mut OUT: [u16; 2];
        #[inline(never)]
        fn read_hi(v @ A: u16) -> u16 { let z: u8 = B; HI = z; return v; }
        #[inline(never)]
        fn write_hi(v @ A: u16) -> u16 { B = 0x56; return v + 1; }
        #[entry]
        fn main() -> ! { OUT[0] = read_hi(0x1234); OUT[1] = write_hi(0x1234); loop {} }
    ''', ExpectedState(memory={0x7E0300: [0x12], 0x7E0302: [0x34, 0x12, 0x35, 0x12]}),
        extra_args=[opt])
    assert result.success, f"Failures: {result.failures} {result.error}"
