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
            #[zeropage(0x40)] static mut G: u8;
            #[zeropage(0x42)] static mut OUT8: u8;
            #[zeropage(0x44)] static mut OUT16: u16;

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
            0x7E0040: [0x18],
            0x7E0042: [0x90],
            0x7E0044: [0x08, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_loop_blocks_placed_after_the_back_edge(self, e2e):
        # The else arm is laid out after the loop's `BRA header` and jumps
        # back into the loop, so the rotated loop cannot fall into the exit.
        result = e2e.run('''
            #[lowram] static mut DATA: [u16; 8];
            #[zeropage(0x40)] static mut OUT: u16;

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
        ''', ExpectedState(memory={0x7E0040: [0xA4, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_continue_and_nested_counters(self, e2e):
        result = e2e.run('''
            #[zeropage(0x40)] static mut OUT: u16;
            #[zeropage(0x42)] static mut G: u8;
            #[zeropage(0x43)] static mut H: u8;
            #[zeropage(0x44)] static mut K: u8;

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
            0x7E0040: [0x1E, 0x00],
            0x7E0042: [0x06],
            0x7E0043: [0x03],
            0x7E0044: [0x06],
        }))
        assert result.success, f"Failures: {result.failures}"
