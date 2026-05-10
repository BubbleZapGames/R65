# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end smoke test for the `let x = y` fresh-vreg fix.

Distinct-vreg shape is asserted by compiler/mir/test_vreg_aliasing.py; this
file confirms the fix holds through codegen with one runtime check.
"""

from r65.tests.e2e import ExpectedState


class TestVregAliasingE2E:
    def test_let_copy_independent_runtime(self, e2e):
        """let mut cy = height; cy--; — height keeps its value at runtime."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: [u8; 2];

            fn use_copy(height @ A: u8) {
                let mut cy: u8 = height;
                cy--;
                RESULT[0] = height;
                RESULT[1] = cy;
            }

            #[entry]
            fn main() {
                use_copy(10);
            }
        ''', ExpectedState(memory={
            0x7E0010: [10, 9],
        }))
        assert result.success, f"Failures: {result.failures}"
