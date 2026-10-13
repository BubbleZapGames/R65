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


class TestRuntimeArrayLiteralsE2E:
    """A local literal's run-time elements are stored into their slots."""

    def test_runtime_and_constant_elements(self, e2e):
        """`[x, 1, 2]` (constants as immediates) and a literal whose constant
        part is block-copied from ROM before its run-time elements land."""
        result = e2e.run('''
            #[zeropage(0x10)] static mut IN: u8;
            #[zeropage(0x20)] static mut R: [u8; 6];

            #[entry]
            fn main() {
                IN = 9;
                let x: u8 = IN;
                let a: [u8; 3] = [x, 1, 2];
                let b: [u8; 8] = [1, 2, 3, x, 5, 6, 7, x];
                R[0] = a[0]; R[1] = a[1]; R[2] = a[2];
                R[3] = b[2]; R[4] = b[3]; R[5] = b[7];
            }
        ''', ExpectedState(memory={0x7E0020: [9, 1, 2, 3, 9, 9]}))
        assert result.success, f"Failures: {result.failures}"

    def test_runtime_fill(self, e2e):
        """`[v; N]` stores v N times when short and replicates element 0
        when long, evaluating v once either way."""
        result = e2e.run('''
            #[zeropage(0x10)] static mut COUNT: u8;
            #[zeropage(0x20)] static mut R: [u8; 5];

            fn next() -> u8 { COUNT = COUNT + 1; return COUNT + 6; }

            #[entry]
            fn main() {
                COUNT = 0;
                let a: [u8; 4] = [next(); 4];
                let b: [u8; 20] = [next(); 20];
                R[0] = a[0]; R[1] = a[3]; R[2] = b[0]; R[3] = b[19];
                R[4] = COUNT;
            }
        ''', ExpectedState(memory={0x7E0020: [7, 7, 8, 8, 2]}))
        assert result.success, f"Failures: {result.failures}"

    def test_runtime_u16_elements(self, e2e):
        """u16 run-time elements, in a literal and in a long fill."""
        result = e2e.run('''
            #[zeropage(0x10)] static mut IN: u16;
            #[zeropage(0x20)] static mut R: [u16; 5];

            #[entry]
            fn main() {
                IN = 0xBEEF;
                let w: u16 = IN;
                let a: [u16; 3] = [w, 0x1234, w];
                let b: [u16; 12] = [w; 12];
                R[0] = a[0]; R[1] = a[1]; R[2] = a[2];
                R[3] = b[1]; R[4] = b[11];
            }
        ''', ExpectedState(memory={0x7E0020: [0xEF, 0xBE, 0x34, 0x12, 0xEF, 0xBE,
                                              0xEF, 0xBE, 0xEF, 0xBE]}))
        assert result.success, f"Failures: {result.failures}"

    def test_runtime_struct_literal_fields(self, e2e):
        """A promoted local struct's run-time field, and one inside its
        nested array field."""
        result = e2e.run('''
            struct Rec { a: u8, b: u16, arr: [u8; 3] }

            #[zeropage(0x10)] static mut IN: u8;
            #[zeropage(0x20)] static mut R: [u8; 4];

            #[entry]
            fn main() {
                IN = 5;
                let x: u8 = IN;
                let r = Rec { a: x, b: 300, arr: [4, x, 6] };
                R[0] = r.a; R[1] = r.b as u8; R[2] = r.arr[1]; R[3] = r.arr[2];
            }
        ''', ExpectedState(memory={0x7E0020: [5, 44, 5, 6]}))
        assert result.success, f"Failures: {result.failures}"
