# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Locals split into one vreg per value, checked on the hardware.

`WebSplitter` (optimize/web_split.py) gives a reused local a vreg per
def-use web, and loop promotion may then put one of them in X while
another value takes X once it is dead. The vreg counts are pinned in
tests/compiler/optimize/test_web_split.py.
"""

from r65.tests.e2e import ExpectedState


class TestWebSplit:

    def test_reused_counter(self, e2e):
        result = e2e.run('''
            #[lowram] static mut FLAGS: [u8; 1024];
            #[lowram] static mut ROW: [u8; 32];
            #[zeropage(0xC0)] static mut PRIMES: u16;
            #[zeropage(0xC2)] static mut MIXED: u16;

            #[inline(never)]
            fn sieve() -> u16 {
                let mut count: u16 = 0;
                let mut i: u16 = 0;
                while i < 1024 { FLAGS[i] = 0; i = i + 1; }
                i = 2;
                while i < 1024 {
                    if FLAGS[i] == 0 {
                        count = count + 1;
                        let mut j: u16 = i + i;
                        while j < 1024 { FLAGS[j] = 1; j = j + i; }
                    }
                    i = i + 1;
                }
                return count;
            }

            // `j` is moved into X while `i` is live, and `c` takes Y: `i`
            // must stay out of X.
            #[inline(never)]
            fn interleaved() -> u16 {
                let mut c: u16 = 0;
                let mut i: u16 = 0;
                let mut j: u16 = 0;
                while i < 8 {
                    ROW[i] = 1;
                    ROW[j + 8] = 2;
                    j = j + 3;
                    i = i + 1;
                    c = c + 1;
                }
                return c;
            }

            #[entry]
            fn main() {
                PRIMES = sieve();
                let c: u16 = interleaved();
                MIXED = c + (ROW[7] as u16) + (ROW[29] as u16);
            }
        ''', ExpectedState(memory={
            0x7E00C0: [172, 0],        # primes below 1024
            0x7E00C2: [8 + 1 + 2, 0],
        }), max_instructions=200000)
        assert result.success, f"Failures: {result.failures}"
