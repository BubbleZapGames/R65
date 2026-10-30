# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A bank filled to its last byte passes the compiler's overflow check
(tests/compiler/codegen/test_bank_overflow.py) and WLA-DX alike."""

from r65.tests.e2e import ExpectedState


def test_full_bank_assembles_and_reads_back(e2e):
    result = e2e.run('''
        #[zeropage(0xC0)] static mut R: u8;
        #[entry]
        fn main() { R = TA[32767]; }
        #[bank(1)]
        static TA: [u8; 32768] = [0x5A; 32768];
    ''', ExpectedState(memory={0x7E00C0: [0x5A]}))
    assert result.success, f"Failures: {result.failures}"
