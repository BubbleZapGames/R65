# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A `*dyn` call from a far-pointer function hands the method the program's D.

Under D=S the caller's D is its stack frame. A direct call pulls the saved D
before it and pushes it back after; a dispatch did neither, so a method that
touched the direct page read and wrote the caller's stack instead.
"""

from r65.tests.e2e import E2ETest, ExpectedState

TRAIT = '''
#[zeropage(0x70)] static mut ZK: u8;
#[lowram(0x0300)] static mut OUT: [u8; 8];
#[ram] static mut BUF: [u8; 8];
struct Ctr { k: u8 }
trait Step { fn kick(*self, b: u8) -> u8; fn add(*self, b: u8) -> u8; }
impl Step for Ctr {
    fn kick(*self, b: u8) -> u8 { ZK = ZK + b; self.k = self.k + 1; return ZK; }
    fn add(*self, b: u8) -> u8 { self.k = self.k + b; return self.k; }
}
'''


def run(source, expected, opt):
    result = E2ETest().run(TRAIT + source, ExpectedState(memory={0x7E0300: expected}),
                           extra_args=[opt])
    assert result.success, f"{result.error} {result.failures}"


def test_method_reaches_the_direct_page():
    run('''
        #[lowram(0x0310)] static mut CC: Ctr = Ctr { k: 0 };
        #[inline(never)]
        fn far_kick(buf: far *u8, o: *dyn Step, n: u8) -> u8 {
            let mut last: u8 = 0;
            for i in 0..n { last = o.kick(buf[i]); buf[i] = last; }
            return last;
        }

        #[entry]
        fn main() {
            ZK = 0;
            for i in 0..8 { BUF[i] = (i as u8) + 1; }
            OUT[0] = far_kick(&BUF[0] as far *u8, &CC as *dyn Step, 3);
            OUT[1] = far_kick(&BUF[4] as far *u8, &CC as *dyn Step, 2);
            OUT[2] = ZK; OUT[3] = CC.k;
            for i in 0..3 { OUT[4 + i] = BUF[i]; }
        }
    ''', [6, 17, 17, 5, 1, 3, 6], "-O1")


def test_far_self_chain_stands_alone():
    """Two far-self dispatches in a row would hold self's DBR across both;
    here each must also nest its own PLD/PHD."""
    run('''
        #[ram] static mut CC: Ctr = Ctr { k: 0 };
        #[inline(never)]
        fn far_kick(buf: far *u8, o: far *dyn Step, n: u8) -> u8 {
            let mut last: u8 = 0;
            for i in 0..n {
                let a: u8 = o.kick(buf[i]);
                let b: u8 = o.add(1);
                last = a + b;
                buf[i] = last;
            }
            return last;
        }

        #[entry]
        fn main() {
            ZK = 0;
            for i in 0..8 { BUF[i] = (i as u8) + 1; }
            OUT[0] = far_kick(&BUF[0] as far *u8, &CC as far *dyn Step, 3);
            OUT[1] = ZK; OUT[2] = CC.k;
            for i in 0..3 { OUT[3 + i] = BUF[i]; }
        }
    ''', [12, 6, 6, 3, 7, 12], "-O0")
