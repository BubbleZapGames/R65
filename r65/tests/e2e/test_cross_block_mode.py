# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for cross-block A mode mismatch via JMP.

Bug: the peephole optimizer's cross-block mode elimination pass only
recorded source modes at branch targets (BCC, BEQ, BRA etc.) but NOT
at JMP targets. When a JMP from a 16-bit A context targeted a label
where other paths arrived in 8-bit, the optimizer only saw the 8-bit
modes and incorrectly eliminated the SEP #$20 at the target. This
caused subsequent instructions to decode with wrong operand sizes.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestCrossBlockMode:
    """Test cross-block A mode tracking with JMP instructions."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_jmp_from_m16_to_m8_label(self, e2e):
        """JMP from 16-bit A context to label expecting 8-bit A.

        Pattern from unpack_level: if/else chain where some branches
        do 16-bit operations then JMP to a shared label that does an
        8-bit CMP. Without proper mode tracking, CMP #$80 in m16
        reads an extra byte and decodes subsequent code as operands.
        """
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
            static mut result_val: u8;
            #[lowram]
            static mut wide_val: u16;

            fn process(n: u8, flag: u8) -> u8 {
                let val: u8 = n;

                if flag == 1 {
                    // This path does 16-bit operations then falls through
                    wide_val = (n as u16) + 0x0100;
                } else if flag == 2 {
                    // Another 16-bit path
                    wide_val = (n as u16) * 2;
                }
                // After the if/else, execution merges here.
                // One path arrives in m16, the other in m8.
                // The peephole must NOT eliminate SEP #$20 here.

                // This comparison needs 8-bit A mode
                if val >= 128 {
                    val = 0;
                }

                return val;
            }

            #[entry]
            fn main() {
                // flag=1 triggers the 16-bit path before the CMP
                result_val = process(200, 1);  // 200 >= 128, should return 0
            }
        ''', ExpectedState(memory={
            0x7E0200: 0,   # result_val = 0 (200 >= 128 -> val = 0)
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_many_branches_force_jmp(self, e2e):
        """Many if/else branches with 16-bit ops force JMP to shared label.

        With enough branches and code in each, the compiler emits JMP
        (not BRA) to reach the shared merge point. This triggers the
        bug where JMP source modes aren't recorded at the target label.
        """
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
            static mut result_a: u8;
            #[lowram]
            static mut result_b: u8;
            #[lowram]
            static mut wide_a: u16;
            #[lowram]
            static mut wide_b: u16;
            #[lowram]
            static mut wide_c: u16;
            #[lowram]
            static mut wide_d: u16;

            fn big_switch(n: u8, code: u8) -> u8 {
                let val: u8 = n;

                if code == 0xFE {
                    wide_a = (n as u16) + 0x100;
                    wide_b = wide_a + 0x200;
                } else if code == 0xF4 {
                    wide_a = (n as u16) + 0x300;
                    wide_c = wide_a + 0x400;
                } else if code == 0xF6 {
                    wide_a = (n as u16) + 0x500;
                    wide_d = wide_a + 0x600;
                } else if code == 0xF7 {
                    wide_b = (n as u16) + 0x700;
                    wide_c = wide_b + 0x800;
                } else if code == 0xF8 {
                    wide_c = (n as u16) + 0x900;
                    wide_d = wide_c + 0xA00;
                } else if code == 0xF9 {
                    wide_d = (n as u16) + 0xB00;
                    wide_a = wide_d + 0xC00;
                } else if code == 0xFA {
                    wide_a = (n as u16) + 0xD00;
                    wide_b = wide_a + 0xE00;
                    wide_c = wide_b + 0xF00;
                } else if code == 0xFB {
                    wide_b = (n as u16) + 0x1000;
                    wide_c = wide_b + 0x1100;
                    wide_d = wide_c + 0x1200;
                } else if code == 0xFC {
                    wide_c = (n as u16) + 0x1300;
                    wide_d = wide_c + 0x1400;
                    wide_a = wide_d + 0x1500;
                }

                // After all branches merge: 8-bit comparison
                // Some branches left A in 16-bit mode (from u16 additions)
                // The SEP #$20 here must NOT be eliminated
                if val >= 128 {
                    val = 0;
                }

                return val;
            }

            #[entry]
            fn main() {
                // code=0xFE triggers first 16-bit branch, val=200 >= 128 -> 0
                result_a = big_switch(200, 0xFE);
                // code=0xFC triggers last 16-bit branch, val=50 < 128 -> 50
                result_b = big_switch(50, 0xFC);
            }
        ''', ExpectedState(memory={
            0x7E0200: 0,   # result_a: 200 >= 128 -> 0
            0x7E0201: 50,  # result_b: 50 < 128 -> 50
        }))
        assert result.success, f"Failures: {result.failures}"
