# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for scratch param to local variable assignment.

Bug: `let pos = ptr;` where `ptr` is a scratch-promoted parameter was
incorrectly optimized away by Move coalescing. The coalescer saw that
the dest vreg (pos) and source vreg (ptr) didn't interfere, so it
merged them. But ptr was pre-allocated to a scratch DP address, and
pos needed its own stack slot — the Move that copies the scratch value
to the stack was eliminated, leaving pos uninitialized.
"""

from r65.tests.e2e import ExpectedState


class TestScratchParamLocalAssignment:
    """Test that assigning scratch params to locals preserves the value."""

    def test_let_local_equals_scratch_param(self, e2e):
        """let pos = ptr; must copy scratch param value to local."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[lowram]
            static mut result1: u16;
            #[lowram]
            static mut result2: u16;

            // 4 params forces scratch promotion for off and ptr
            fn process(off: u16, ptr: u16, wdt: u8, dx: u8) {
                let idx: u16 = off << 5;
                let pos: u16 = ptr;

                // Use both locals to prevent dead code elimination
                result1 = idx;
                result2 = pos;
            }

            #[entry]
            fn main() {
                process(3, 827, 27, 1);
            }
        ''', ExpectedState(memory={
            # idx = 3 << 5 = 96 = 0x0060
            0x7E0200: 0x60,
            0x7E0201: 0x00,
            # pos = ptr = 827 = 0x033B
            0x7E0202: 0x3B,
            0x7E0203: 0x03,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_scratch_param_modified_in_loop(self, e2e):
        """Local assigned from scratch param, then modified in a loop."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[lowram]
            static mut result: u16;

            fn accumulate(start: u16, count: u8, step: u8) {
                let pos: u16 = start;
                let i: u8 = 0;
                loop {
                    if i >= count {
                        break;
                    }
                    pos = pos + (step as u16);
                    i++;
                }
                result = pos;
            }

            #[entry]
            fn main() {
                accumulate(100, 5, 10);
            }
        ''', ExpectedState(memory={
            # pos = 100 + 5*10 = 150 = 0x0096
            0x7E0200: 0x96,
            0x7E0201: 0x00,
        }))
        assert result.success, f"Failures: {result.failures}"
