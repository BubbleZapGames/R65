# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end smoke tests for parameter forwarding.

Identity-copy elimination is unit-tested in
compiler/codegen/test_scratch_forwarding.py. This file keeps a couple of
integration cases through a real call chain to catch issues the unit test
mocks away.
"""

from r65.tests.e2e import ExpectedState


class TestParamForwardingE2E:
    def test_forward_u16_through_chain(self, e2e):
        """Forward u16 stack param through outer→inner — value preserved."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            #[zeropage(0x10)]
            static mut RESULT_LO: u8;
            #[zeropage(0x11)]
            static mut RESULT_HI: u8;

            fn inner(val: u16) -> u16 { return val; }
            fn outer(val: u16) -> u16 { return inner(val); }

            #[entry]
            fn main() {
                let mut result: u16 = outer(0x1234);
                RESULT_LO = result as u8;
                RESULT_HI = (result >> 8) as u8;
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x34,
            0x7E0011: 0x12,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_forward_far_pointer_through_chain(self, e2e):
        """Forward far pointer through outer→inner — dereference works."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            #[ram]
            static mut BUFFER: [u8; 4] = [10, 20, 30, 40];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn read_first(ptr: far *u8) -> u8 { return *ptr; }
            fn wrapper(ptr: far *u8) -> u8 { return read_first(ptr); }

            #[entry]
            fn main() {
                RESULT = wrapper(&BUFFER as far *u8);
            }
        ''', ExpectedState(memory={0x7E0010: 10}))
        assert result.success, f"Failures: {result.failures}"
