# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A register spilled around a call in a far-pointer function nests with the
D or DBR the call saves.

Under D=S the prologue's saved D sits on top of the frame: the PLD before a
call must come before a spill is pushed over it (it popped the spilled loop
counter into D, and the PLY after the call took the caller's D), and nothing
may stay pushed across the PHD after it. Under SET_DBR the PHB before a call
goes on above the spills, so its PLB comes off before they are reloaded.
"""

import subprocess
import sys

from r65.tests.e2e import E2ETest, ExpectedState

# `add1` indexes through a pointer, so it writes Y: the loop counter in Y is
# spilled around every call.
COMMON = '''
#[ram] static mut BUF: [u8; 32];
#[lowram(0x0300)] static mut OUT: [u8; 8];
#[lowram(0x0310)] static mut TAB: [u8; 8];

#[inline(never)]
fn add1(v: u8) -> u8 {
    let p: *u8 = &TAB as *u8;
    return v + p[1];
}
'''


def run(source, expected, opt="-O1"):
    result = E2ETest().run(COMMON + source, ExpectedState(memory={0x7E0300: expected}),
                           max_instructions=100000, extra_args=[opt])
    assert result.success, f"{result.error} {result.failures}"


def test_d_equals_s_spill_goes_above_the_saved_d():
    run('''
        #[inline(never)]
        fn copy_inc(dst: far *u8, src: far *u8, n: u8) -> u8 {
            let mut last: u8 = 0;
            for i in 0..n {
                last = add1(src[i]);
                dst[i] = last;
            }
            return last;
        }

        #[entry]
        fn main() {
            for i in 0..8 { TAB[i] = 1; }
            for i in 0..16 { BUF[i] = i as u8; }
            OUT[0] = copy_inc(&BUF[16] as far *u8, &BUF[0] as far *u8, 8);
            OUT[1] = copy_inc(&BUF[24] as far *u8, &BUF[0] as far *u8, 4);
            for i in 0..6 { OUT[2 + i] = BUF[16 + i]; }
        }
    ''', [8, 4, 1, 2, 3, 4, 5, 6])


def test_d_equals_s_region_spanning_calls_is_pulled_before_each_phd():
    """One Y region covers all three calls; D=S is set up between them."""
    run('''
        #[inline(never)]
        fn thrice(dst: far *u8, src: far *u8, n: u8) -> u8 {
            let mut last: u8 = 0;
            for i in 0..n {
                let a: u8 = add1(src[i]);
                let b: u8 = add1(a);
                let c: u8 = add1(b);
                dst[i] = c + src[i];
                last = c;
            }
            return last;
        }

        #[entry]
        fn main() {
            for i in 0..8 { TAB[i] = 1; }
            for i in 0..16 { BUF[i] = i as u8; }
            OUT[0] = thrice(&BUF[16] as far *u8, &BUF[0] as far *u8, 8);
            OUT[1] = thrice(&BUF[24] as far *u8, &BUF[0] as far *u8, 4);
            for i in 0..6 { OUT[2 + i] = BUF[16 + i]; }
        }
    ''', [10, 6, 3, 5, 7, 9, 11, 13])


SET_DBR = '''
#[zeropage(0x70)] static mut Z: [u8; 8];

#[inline(never)]
fn inc_all(buf: far *u8, n: u8) -> u8 {
    let mut last: u8 = 0;
    for i in 0..n {
        Z[0] = Z[0] + 1; Z[1] = Z[1] + 1; Z[2] = Z[2] + 1; Z[3] = Z[3] + 1;
        Z[4] = Z[4] + 1; Z[5] = Z[5] + 1; Z[6] = Z[6] + 1; Z[7] = Z[7] + 1;
        last = add1(buf[i]);
        buf[i] = last;
    }
    return last;
}

#[entry]
fn main() {
    for i in 0..8 { TAB[i] = 1; Z[i] = 0; }
    for i in 0..16 { BUF[i] = i as u8; }
    OUT[0] = inc_all(&BUF[0] as far *u8, 8);
    OUT[1] = inc_all(&BUF[3] as far *u8, 2);
    for i in 0..6 { OUT[2 + i] = BUF[i]; }
}
'''


def test_set_dbr_plb_comes_off_before_the_spill():
    # The zero-page traffic makes the cost model pick SET_DBR at -O0.
    asm = subprocess.run(
        [sys.executable, "-m", "r65.compiler.main", "-", "--cfg", "snes", "-O0", "-q"],
        input=COMMON + SET_DBR, capture_output=True, text=True, check=True).stdout
    assert 'Save ptr-bank DBR before call' in asm, "inc_all no longer uses SET_DBR"
    run(SET_DBR, [8, 6, 1, 2, 3, 5, 6, 6], opt="-O0")
