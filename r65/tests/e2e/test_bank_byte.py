# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for bank_byte() far pointer method.

Tests compile-time const evaluation and runtime bank byte extraction.
"""

from r65.tests.e2e import ExpectedState


class TestBankByte:
    """Test bank_byte() method on far pointers."""

    def test_const_eval_bank_byte(self, e2e):
        """Const-evaluated bank_byte() should produce correct immediate."""
        result = e2e.run('''
            const BANK: u8 = (0x7E2000 as far *u8).bank_byte();

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                RESULT = BANK;
            }
        ''', ExpectedState(memory={0x7E0010: 0x7E}))
        assert result.success, f"Failures: {result.failures}"

    def test_const_eval_bank_byte_zero(self, e2e):
        """bank_byte() of bank 0 address should be 0x00."""
        result = e2e.run('''
            const BANK: u8 = (0x002000 as far *u8).bank_byte();

            #[zeropage(0x10)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                RESULT = BANK;
            }
        ''', ExpectedState(memory={0x7E0010: 0x00}))
        assert result.success, f"Failures: {result.failures}"

    def test_runtime_bank_byte_from_address_of(self, e2e):
        """bank_byte() on far pointer from &ram_array (bank $7E)."""
        result = e2e.run('''
            #[ram]
            static mut DATA: [u8; 4] = [0, 0, 0, 0];

            #[zeropage(0x10)]
            static mut RESULT: u8;

            fn get_bank(ptr: far *u8) -> u8 {
                return ptr.bank_byte();
            }

            #[entry]
            fn main() {
                let p: far *u8 = &DATA;
                RESULT = get_bank(p);
            }
        ''', ExpectedState(memory={0x7E0010: 0x7E}))
        assert result.success, f"Failures: {result.failures}"
