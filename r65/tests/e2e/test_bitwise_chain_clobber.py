# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Regression test for the bitwise-condition accumulator-chain clobber.

`_emit_bitwise_chain` lowers `if (a | b | c)` to `LDA a; ORA b; ORA c`.
For a *mixed-operator* nested operand like `a | (b & c)` it used to flatten
by `|` into `[a, (b & c)]`, load `a` into A, then evaluate `(b & c)` —
which itself loads `b` into A, destroying the accumulated `a` before the
ORA reads it. The condition then evaluated against garbage.

Chosen so the two behaviours diverge:
    a=0x01, b=0x00, c=0x00
    correct : a | (b & c) = 0x01 | 0x00 = 0x01  -> truthy
    buggy   : A=a; A clobbered by (b&c) eval -> 0x00 -> falsy
"""

from r65.tests.e2e import ExpectedState


class TestBitwiseChainClobber:
    def test_mixed_operator_nested_operand(self, e2e):
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x11)]
            static mut AV: u8;
            #[zeropage(0x12)]
            static mut BV: u8;
            #[zeropage(0x13)]
            static mut CV: u8;

            #[entry]
            fn main() {
                AV = 0x01;
                BV = 0x00;
                CV = 0x00;
                RESULT = 0x55;
                if (AV | (BV & CV)) != 0 {
                    RESULT = 0xAA;   // taken iff condition correct (0x01 != 0)
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 0xAA,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_operator_nested_operand_falsy(self, e2e):
        """Symmetric case: correct result is falsy; a clobber would flip it."""
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u8;
            #[zeropage(0x11)]
            static mut AV: u8;
            #[zeropage(0x12)]
            static mut BV: u8;
            #[zeropage(0x13)]
            static mut CV: u8;

            #[entry]
            fn main() {
                AV = 0x00;
                BV = 0xFF;
                CV = 0x00;
                RESULT = 0x55;
                // correct: 0x00 | (0xFF & 0x00) = 0 -> falsy, stays 0x55
                if (AV | (BV & CV)) != 0 {
                    RESULT = 0xAA;
                }
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x55,
        }))
        assert result.success, f"Failures: {result.failures}"
