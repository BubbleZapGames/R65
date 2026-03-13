"""
End-to-end tests for array indexing with variables across conditional branches.

Regression tests for a bug where the compiler promotes a variable-based array
index to a scratch register inside a conditional block, but then uses that
scratch register on the path that skipped the conditional — reading stale or
uninitialized scratch data instead of the actual variable.

Example of the bug pattern:
    if condition {
        variable++;          // compiler stores variable to scratch $00 HERE
    }
    result = table[variable]; // compiler reads scratch $00 — wrong on else path!

The fix must ensure the array index is loaded from the authoritative variable
location on every path, not from a scratch register that may be stale.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestArrayIndexConditional:
    """Test array indexing with variables across conditional branches."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_array_index_after_conditional_update(self, e2e):
        """Array index variable modified in if-block, used after if-block.

        The index is incremented inside a conditional. The array lookup
        happens unconditionally after. Both paths (incremented and not)
        must read the correct index value from the variable, not from
        a scratch register that was only written inside the if-block.
        """
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;

            static TABLE: [u8; 4] = [10, 20, 30, 40];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[zeropage(0x11)]
            static mut IDX: u8;

            #[zeropage(0x12)]
            static mut COUNTER: u8;

            fn update_and_lookup() {
                // Counter controls whether IDX is incremented
                COUNTER++;
                if COUNTER >= 2 {
                    COUNTER = 0;
                    IDX++;
                    if IDX >= 4 {
                        IDX = 0;
                    }
                }

                // This lookup must use the ACTUAL value of IDX,
                // not a stale scratch register
                RESULT = TABLE[IDX];
            }

            #[entry]
            fn main() {
                IDX = 0;
                COUNTER = 0;

                // Call 1: COUNTER becomes 1, IDX stays 0 (no increment)
                // TABLE[0] = 10
                update_and_lookup();
                // RESULT should be 10

                // Call 2: COUNTER becomes 2 >= 2, so IDX becomes 1
                // TABLE[1] = 20
                update_and_lookup();
                // RESULT should be 20
            }
        ''', ExpectedState(memory={
            0x7E0010: 20,  # TABLE[1] after second call
            0x7E0011: 1,   # IDX = 1
            0x7E0012: 0,   # COUNTER reset to 0
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_array_index_skip_path_uses_original(self, e2e):
        """When conditional block is NOT entered, array index must use
        the original variable value, not garbage from a scratch register.

        This is the minimal reproduction: a conditional that modifies and
        stores a variable to scratch, followed by an array access that
        the compiler incorrectly reads from that scratch register.
        """
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;

            static TABLE: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[zeropage(0x11)]
            static mut IDX: u8 = 2;

            #[zeropage(0x12)]
            static mut FLAG: u8 = 0;

            #[entry]
            fn main() {
                // FLAG is 0, so this block is skipped entirely.
                // The compiler must NOT rely on scratch $00 for IDX.
                if FLAG != 0 {
                    IDX++;
                    if IDX >= 4 {
                        IDX = 0;
                    }
                }

                // Must use IDX (=2) from memory, not stale scratch
                RESULT = TABLE[IDX];
            }
        ''', ExpectedState(memory={
            0x7E0010: 0xCC,  # TABLE[2]
            0x7E0011: 2,     # IDX unchanged
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_array_index_both_paths_correct(self, e2e):
        """Verify both the taken and not-taken paths produce correct
        array lookups when the index variable is conditionally modified.
        """
        # Test the taken path (FLAG=1, IDX goes from 1 to 2)
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;

            static TABLE: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[zeropage(0x11)]
            static mut IDX: u8 = 1;

            #[zeropage(0x12)]
            static mut FLAG: u8 = 1;

            #[entry]
            fn main() {
                if FLAG != 0 {
                    IDX++;
                    if IDX >= 4 {
                        IDX = 0;
                    }
                }
                RESULT = TABLE[IDX];
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x33,  # TABLE[2] (IDX was 1, incremented to 2)
            0x7E0011: 2,
        }))
        assert result.success, f"Taken path failures: {result.failures}"

    def test_local_var_assigned_in_both_branches(self, e2e):
        """Local variable assigned in both branches of if/else must keep
        a single vreg so the post-merge code reads the correct value
        regardless of which branch was taken.

        Regression: save/restore of symbol_to_vreg was too aggressive,
        causing the else-branch to allocate a second vreg for the same
        local.  The merge-point code then referenced only one vreg,
        reading garbage on the other path.
        """
        # Test the then-path (FLAG=1 → tile = 0xAA)
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[zeropage(0x11)]
            static mut FLAG: u8 = 1;

            #[entry]
            fn main() {
                let tile: u8;
                if FLAG != 0 {
                    tile = 0xAA;
                } else {
                    tile = 0xBB;
                }
                RESULT = tile;
            }
        ''', ExpectedState(memory={0x7E0010: 0xAA}))
        assert result.success, f"Then-path: {result.failures}"

        # Test the else-path (FLAG=0 → tile = 0xBB)
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[zeropage(0x11)]
            static mut FLAG: u8 = 0;

            #[entry]
            fn main() {
                let tile: u8;
                if FLAG != 0 {
                    tile = 0xAA;
                } else {
                    tile = 0xBB;
                }
                RESULT = tile;
            }
        ''', ExpectedState(memory={0x7E0010: 0xBB}))
        assert result.success, f"Else-path: {result.failures}"

    def test_u16_array_index_after_conditional(self, e2e):
        """Same pattern but with u16 array elements."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;

            static TABLE: [u16; 4] = [0x1111, 0x2222, 0x3333, 0x4444];

            #[zeropage(0x10)]
            static mut RESULT_LO: u8;

            #[zeropage(0x11)]
            static mut RESULT_HI: u8;

            #[zeropage(0x12)]
            static mut IDX: u8 = 0;

            #[zeropage(0x13)]
            static mut FLAG: u8 = 0;

            #[entry]
            fn main() {
                if FLAG != 0 {
                    IDX++;
                }

                let val: u16 = TABLE[IDX];
                RESULT_LO = val as u8;
                RESULT_HI = (val >> 8) as u8;
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x11,  # TABLE[0] low byte
            0x7E0011: 0x11,  # TABLE[0] high byte
        }))
        assert result.success, f"Failures: {result.failures}"
