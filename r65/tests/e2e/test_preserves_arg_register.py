# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A callee's `#[preserves(X)]` does not keep the caller's X when an argument
goes in X.

The callee restores X to what it was given - the argument - so a value the
caller held in X before setting the argument up is gone. The caller trusted
the declaration, skipped the spill, and lost its loop bound.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def test_bound_in_x_survives_an_x_argument():
    result = E2ETest().run('''
        #[lowram(0x0300)] static mut OUT: [u8; 8];
        #[inline(never)]
        #[preserves(X)]
        fn put(i @ X: u16, v @ A: u8) { OUT[i] = v; }
        #[inline(never)]
        fn fill(n @ X: u16) {
            let mut i: u16 = 0;
            while i < n { put(i, i as u8 + 1); i = i + 1; }
        }

        #[entry]
        fn main() { for k in 0..8 { OUT[k] = 0; } fill(4); }
    ''', ExpectedState(memory={0x7E0300: [1, 2, 3, 4, 0, 0, 0, 0]}))
    assert result.success, f"{result.error} {result.failures}"


def test_promoted_bound_survives_a_promoted_argument():
    """FixedStack promotes `n` and `keep`'s `w` both to X."""
    result = E2ETest(abi="FixedStack").run('''
        #[lowram(0x0300)] static mut OUT: [u16; 2];
        #[inline(never)]
        #[preserves(X, Y)]
        fn keep(v: u8, w: u16) -> u8 { return v + (w as u8); }
        #[inline(never)]
        fn loopy(n: u16) -> u16 {
            let mut s: u16 = 0;
            let mut i: u16 = 0;
            while i < n { s = s + (keep(i as u8, i) as u16); i = i + 1; }
            return s;
        }

        #[entry]
        fn main() { OUT[0] = loopy(5); OUT[1] = loopy(3); }
    ''', ExpectedState(memory={0x7E0300: [20, 0, 6, 0]}))
    assert result.success, f"{result.error} {result.failures}"
