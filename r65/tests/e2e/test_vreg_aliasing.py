# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end regression tests for variable aliasing bugs.

When `let x = y` where y is a VirtualRegister bound to another symbol,
the builder must create a fresh vreg so modifications to one variable
don't corrupt the other.  These tests verify that:

1. Independent copies stay independent under mutation.
2. Dead-after-copy variables are still coalesced (no perf regression).
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestVregAliasing:
    """Regression tests for the vreg aliasing correctness fix."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_let_copy_independent_of_original(self, e2e):
        """let x = y; y--; x should retain original value."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            fn use_copy(height @ A: u8) {
                let cy: u8 = height;
                cy--;
                // height should still be original, cy should be original - 1
                RESULT[0] = height;
                RESULT[1] = cy;
            }

            #[entry]
            fn main() {
                use_copy(10);
            }
        ''', ExpectedState(memory={
            0x7E0010: [10, 9],  # height=10 unchanged, cy=9
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_let_copy_in_loop_bound(self, e2e):
        """Loop bound from copied variable must not see mutations."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut COUNT: u8;

            fn count_down(n @ A: u8) {
                let bound: u8 = n;
                let mut i: u8 = 0;
                loop {
                    if i >= bound { break; }
                    n--;  // mutate original, should NOT affect bound
                    i++;
                }
                COUNT = i;
            }

            #[entry]
            fn main() {
                count_down(5);
            }
        ''', ExpectedState(memory={
            0x7E0010: 5,  # loop ran exactly 5 times
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_dead_copy_coalesces(self, e2e):
        """let x = y where y is dead after copy should produce correct result."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn compute(val @ A: u8) -> u8 {
                let x: u8 = val;
                // val is never used again — x should work correctly
                return x + 1;
            }

            #[entry]
            fn main() {
                RESULT = compute(41);
            }
        ''', ExpectedState(memory={
            0x7E0010: 42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_multiple_copies_independent(self, e2e):
        """Multiple copies of same variable stay independent."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: [u8; 3];

            fn triple_copy(val @ A: u8) {
                let a: u8 = val;
                let b: u8 = val;
                let c: u8 = val;
                a = a + 1;
                b = b + 2;
                c = c + 3;
                RESULT[0] = a;
                RESULT[1] = b;
                RESULT[2] = c;
            }

            #[entry]
            fn main() {
                triple_copy(10);
            }
        ''', ExpectedState(memory={
            0x7E0010: [11, 12, 13],
        }))
        assert result.success, f"Failures: {result.failures}"
