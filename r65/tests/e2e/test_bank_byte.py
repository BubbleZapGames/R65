# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for bank_byte() far pointer method.

Const-evaluation cases live in compiler/typeck/test_bank_byte.py; this file
covers runtime extraction only.
"""

from r65.tests.e2e import ExpectedState


class TestBankByte:
    """Test bank_byte() method on far pointers."""

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
