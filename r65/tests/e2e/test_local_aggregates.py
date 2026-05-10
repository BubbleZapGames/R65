# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end smoke tests for local aggregate (struct/array) variables.

The promotion vs decomposition logic is unit-tested in
compiler/mir/test_aggregate_promotion.py. This file just verifies the codegen
path produces correct runtime values for one struct, one array, and one
string-literal-initialized array.
"""

from r65.tests.e2e import ExpectedState


class TestLocalAggregatesE2E:
    def test_local_struct_runtime(self, e2e):
        """Decomposed local struct: field writes and reads round-trip."""
        result = e2e.run('''
            struct Entity { kind: u8, health: u16 }

            #[zeropage(0x20)]
            static mut RKIND: u8;
            #[zeropage(0x22)]
            static mut RHEALTH: u16;

            #[entry]
            fn main() {
                let e: Entity;
                e.kind = 5;
                e.health = 1000;
                RKIND = e.kind;
                RHEALTH = e.health;
            }
        ''', ExpectedState(memory={
            0x7E0020: 5,
            0x7E0022: [0xE8, 0x03],  # 1000 little-endian
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_local_array_runtime(self, e2e):
        """Promoted local array: literal init + indexed reads."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut R0: u8;
            #[zeropage(0x21)]
            static mut R1: u8;
            #[zeropage(0x22)]
            static mut R2: u8;

            #[entry]
            fn main() {
                let data: [u8; 4] = [10, 20, 30, 40];
                R0 = data[0];
                R1 = data[1];
                R2 = data[3];
            }
        ''', ExpectedState(memory={
            0x7E0020: 10,
            0x7E0021: 20,
            0x7E0022: 40,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_local_string_literal_runtime(self, e2e):
        """Local array initialized with a string literal."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut R0: u8;
            #[zeropage(0x21)]
            static mut R1: u8;

            #[entry]
            fn main() {
                let msg: [u8; 8] = "Hi";
                R0 = msg[0];
                R1 = msg[1];
            }
        ''', ExpectedState(memory={
            0x7E0020: 72,   # 'H'
            0x7E0021: 105,  # 'i'
        }))
        assert result.success, f"Failures: {result.failures}"
