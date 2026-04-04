# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for local aggregate (struct/array) variables.

Local structs and arrays are promoted to static lowram storage.
These tests verify the full compilation and emulation pipeline.
"""

from r65.tests.e2e import ExpectedState


class TestLocalStructE2E:
    """Test local struct variables compile and execute correctly."""

    def test_local_struct_field_write_read(self, e2e):
        """Local struct field writes and reads produce correct results."""
        result = e2e.run('''
            struct Point { x: u8, y: u8 }

            #[zeropage(0x20)]
            static mut RX: u8;
            #[zeropage(0x21)]
            static mut RY: u8;

            #[entry]
            fn main() {
                let p: Point;
                p.x = 42;
                p.y = 99;
                RX = p.x;
                RY = p.y;
            }
        ''', ExpectedState(
            memory={
                0x7E0020: 42,
                0x7E0021: 99,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_local_struct_with_initializer(self, e2e):
        """Local struct with struct literal initializer."""
        result = e2e.run('''
            struct Point { x: u8, y: u8 }

            #[zeropage(0x20)]
            static mut RX: u8;
            #[zeropage(0x21)]
            static mut RY: u8;

            #[entry]
            fn main() {
                let p: Point = Point { x: 10, y: 20 };
                RX = p.x;
                RY = p.y;
            }
        ''', ExpectedState(
            memory={
                0x7E0020: 10,
                0x7E0021: 20,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_local_struct_u16_field(self, e2e):
        """Local struct with u16 field."""
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
        ''', ExpectedState(
            memory={
                0x7E0020: 5,
                0x7E0022: [0xE8, 0x03],  # 1000 = 0x03E8 little-endian
            }
        ))
        assert result.success, f"Failures: {result.failures}"


class TestLocalArrayE2E:
    """Test local array variables compile and execute correctly."""

    def test_local_array_fill_and_access(self, e2e):
        """Local array with fill initializer, then constant index access."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut R0: u8;
            #[zeropage(0x21)]
            static mut R1: u8;

            #[entry]
            fn main() {
                let buf: [u8; 8] = [0; 8];
                buf[0] = 42;
                buf[3] = 99;
                R0 = buf[0];
                R1 = buf[3];
            }
        ''', ExpectedState(
            memory={
                0x7E0020: 42,
                0x7E0021: 99,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_local_array_literal(self, e2e):
        """Local array with literal initializer."""
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
        ''', ExpectedState(
            memory={
                0x7E0020: 10,
                0x7E0021: 20,
                0x7E0022: 40,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_local_array_in_function(self, e2e):
        """Local array inside a non-entry function."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn fill_and_read() {
                let buf: [u8; 4] = [0; 4];
                buf[2] = 77;
                RESULT = buf[2];
            }

            #[entry]
            fn main() {
                fill_and_read();
            }
        ''', ExpectedState(
            memory={
                0x7E0020: 77,
            }
        ))
        assert result.success, f"Failures: {result.failures}"


class TestLocalStringLiteralE2E:
    """Test local array initialized with string literal."""

    def test_local_string_literal(self, e2e):
        """Local array initialized with string literal."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut R0: u8;
            #[zeropage(0x21)]
            static mut R1: u8;

            #[entry]
            fn main() {
                let msg: [u8; 8] = "Hi";
                R0 = msg[0];  // 'H' = 72
                R1 = msg[1];  // 'i' = 105
            }
        ''', ExpectedState(
            memory={
                0x7E0020: 72,   # 'H'
                0x7E0021: 105,  # 'i'
            }
        ))
        assert result.success, f"Failures: {result.failures}"
