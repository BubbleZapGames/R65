"""
End-to-end tests for parameter forwarding optimization.

Tests that forwarded parameters produce correct results when the compiler
eliminates redundant identity copies (scratch param → same scratch param).
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestParamForwarding:
    """Test parameter forwarding produces correct results."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_forward_u8_stack_params(self, e2e):
        """Forward u8 stack params through nested call."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn inner(a: u8, b: u8) -> u8 {
                return a + b;
            }

            fn outer(a: u8, b: u8) -> u8 {
                return inner(a, b);
            }

            #[entry]
            fn main() {
                RESULT = outer(10, 32);
            }
        ''', ExpectedState(memory={0x7E0010: 42}))
        assert result.success, f"Failures: {result.failures}"

    def test_forward_u16_stack_param(self, e2e):
        """Forward u16 stack param through nested call."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            #[zeropage(0x10)]
            static mut RESULT_LO: u8;
            #[zeropage(0x11)]
            static mut RESULT_HI: u8;

            fn inner(val: u16) -> u16 {
                return val;
            }

            fn outer(val: u16) -> u16 {
                return inner(val);
            }

            #[entry]
            fn main() {
                let result: u16 = outer(0x1234);
                RESULT_LO = result as u8;
                RESULT_HI = (result >> 8) as u8;
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x34,  # low byte
            0x7E0011: 0x12,  # high byte
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_forward_with_computation(self, e2e):
        """Forward some params, compute others — both params used."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn inner(a: u8, b: u16) -> u8 {
                return a + b as u8;
            }

            fn outer(a: u8, b: u16) -> u8 {
                let x: u16 = 3 + b;
                return inner(a, x);
            }

            #[entry]
            fn main() {
                RESULT = outer(10, 100);
            }
        ''', ExpectedState(memory={0x7E0010: 113}))  # 10 + (3+100) = 113
        assert result.success, f"Failures: {result.failures}"

    def test_forward_far_pointer(self, e2e):
        """Forward a far pointer through nested call."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            #[ram]
            static mut BUFFER: [u8; 4] = [10, 20, 30, 40];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn read_first(ptr: far *u8) -> u8 {
                return *ptr;
            }

            fn wrapper(ptr: far *u8) -> u8 {
                return read_first(ptr);
            }

            #[entry]
            fn main() {
                RESULT = wrapper(&BUFFER as far *u8);
            }
        ''', ExpectedState(memory={0x7E0010: 10}))
        assert result.success, f"Failures: {result.failures}"
