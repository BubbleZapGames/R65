# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Rotated loops, checked on the hardware.

A rotated loop leaves through the back edge, so its exit is entered in the
width the body ends in rather than the header's; the loop counter must
still hold N afterwards when it is read; and each `continue` tests on its
own (see tests/compiler/codegen/test_loop_rotation.py for the emitted
shape).
"""

from r65.tests.e2e import ExpectedState


class TestLoopRotation:

    def test_exit_width_and_counter_after_the_loop(self, e2e):
        result = e2e.run('''
            #[lowram] static mut DATA: [u16; 16];
            #[zeropage(0xC0)] static mut G: u8;
            #[zeropage(0xC2)] static mut OUT8: u8;
            #[zeropage(0xC4)] static mut OUT16: u16;

            #[inline(never)]
            fn sum_then_m8() -> u8 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < 16 {
                    s = s + DATA[i];
                    i = i + 1;
                }
                G = G + 3;
                return (s as u8) + G;
            }

            #[inline(never)]
            fn count() -> u16 {
                let mut i: u16 = 0;
                while i < 8 {
                    G = G + 2;
                    i = i + 1;
                }
                return i;
            }

            #[entry]
            fn main() {
                let mut i: u16 = 0;
                while i < 16 {
                    DATA[i] = i + 1;
                    i = i + 1;
                }
                G = 5;
                OUT8 = sum_then_m8();   // 136 + (5 + 3) = 144
                OUT16 = count();        // 8, and G = 8 + 16
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0x18],
            0x7E00C2: [0x90],
            0x7E00C4: [0x08, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_loop_blocks_placed_after_the_back_edge(self, e2e):
        # The else arm is laid out after the loop's `BRA header` and jumps
        # back into the loop, so the rotated loop cannot fall into the exit.
        result = e2e.run('''
            #[lowram] static mut DATA: [u16; 8];
            #[zeropage(0xC0)] static mut OUT: u16;

            #[inline(never)]
            fn score() -> u16 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < 8 {
                    let v: u16 = DATA[i];
                    if v & 1 != 0 {
                        s = s + 100;
                    } else {
                        if v >= 4 {
                            s = s + 10;
                        }
                    }
                    i = i + 1;
                }
                return s;
            }

            #[entry]
            fn main() {
                let mut i: u16 = 0;
                while i < 8 {
                    DATA[i] = i;
                    i = i + 1;
                }
                OUT = score();   // odd 1,3,5,7: 400; even >= 4 (4, 6): 20
            }
        ''', ExpectedState(memory={0x7E00C0: [0xA4, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_continue_and_nested_counters(self, e2e):
        result = e2e.run('''
            #[zeropage(0xC0)] static mut OUT: u16;
            #[zeropage(0xC2)] static mut G: u8;
            #[zeropage(0xC3)] static mut H: u8;
            #[zeropage(0xC4)] static mut K: u8;

            #[inline(never)]
            fn evens() -> u16 {
                let mut s: u16 = 0;
                let mut i: u16 = 0;
                while i < 10 {
                    i = i + 1;
                    if i & 1 != 0 {
                        continue;
                    }
                    s = s + i;
                }
                return s;
            }

            #[inline(never)]
            fn halves() {
                let mut i: u16 = 0;     // read only by its test: counts down
                while i < 6 {
                    i = i + 1;
                    G = G + 1;
                    if G & 1 != 0 {
                        continue;
                    }
                    H = H + 1;
                }
            }

            #[inline(never)]
            fn nested() {
                let mut y: u16 = 0;     // stepped in the inner loop: counts up
                while y < 4 {
                    let mut x: u16 = 0;
                    while x < 3 {
                        x = x + 1;
                        y = y + 1;
                        K = K + 1;
                    }
                }
            }

            #[entry]
            fn main() {
                G = 0;
                H = 0;
                K = 0;
                OUT = evens();   // 2 + 4 + 6 + 8 + 10
                halves();        // G = 6, H = 3
                nested();        // two passes of three: K = 6
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0x1E, 0x00],
            0x7E00C2: [0x06],
            0x7E00C3: [0x03],
            0x7E00C4: [0x06],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_guarded_loops_run_zero_and_more_times(self, e2e):
        # The first test may fail: the guard copy must skip the body.
        result = e2e.run('''
            #[lowram] static mut BUF: [u8; 4];
            #[zeropage(0xC0)] static mut BITS: u16;
            #[zeropage(0xC2)] static mut FROM_START: u16;
            #[zeropage(0xC4)] static mut FROM_END: u16;
            #[zeropage(0xC6)] static mut ODD: u16;
            #[zeropage(0xC8)] static mut START: u16;

            #[inline(never)]
            fn popcount() -> u16 {
                let mut c: u16 = 0;
                let mut i: u16 = 0;
                while i < 4 {
                    let mut v: u8 = BUF[i];
                    while v != 0 {
                        c = c + ((v & 1) as u16);
                        v = v >> 1;
                    }
                    i = i + 1;
                }
                return c;
            }

            #[inline(never)]
            fn steps(from: u16) -> u16 {
                let mut i: u16 = from;
                let mut n: u16 = 0;
                while i < 10 {
                    n = n + 1;
                    i = i + 1;
                }
                return n;
            }

            #[inline(never)]
            fn odd_sum(from: u16) -> u16 {
                let mut i: u16 = from;
                let mut s: u16 = 0;
                while i < 10 {
                    i = i + 1;
                    if i & 1 == 0 { continue; }
                    s = s + i;
                }
                return s;
            }

            #[entry]
            fn main() {
                BUF[0] = 0;
                BUF[1] = 0xFF;
                BUF[2] = 0x81;
                BUF[3] = 0;
                BITS = popcount();          // 0 + 8 + 2 + 0
                START = 3;
                FROM_START = steps(START);  // 7
                START = 12;
                FROM_END = steps(START);    // 0: the guard skips the loop
                ODD = odd_sum(4);           // 5 + 7 + 9
            }
        ''', ExpectedState(memory={
            0x7E00C0: [10, 0],
            0x7E00C2: [7, 0],
            0x7E00C4: [0, 0],
            0x7E00C6: [21, 0],
        }))
        assert result.success, f"Failures: {result.failures}"
