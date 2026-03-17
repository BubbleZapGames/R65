# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for &array[index] address-of array element operations.

Tests that taking the address of an array element produces the correct pointer
(base + index * element_size) for both constant and variable indices, near and
far pointers, and different element sizes.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestAddressOfArrayIndex:
    """Test &array[index] produces correct pointer values."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_zp_array_constant_index_u8(self, e2e):
        """&zp_array[2] should produce near pointer to base+2."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0x30)]
            static mut RESULT: u8;

            fn read_via_ptr(ptr: *u8) -> u8 {
                return *ptr;
            }

            #[entry]
            fn main() {
                let p: *u8 = &ARR[2];
                RESULT = read_via_ptr(p);
            }
        ''', ExpectedState(memory={
            0x7E0030: 30,  # ARR[2] == 30
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_zp_array_constant_index_u16_elements(self, e2e):
        """&zp_array_u16[2] should produce near pointer to base+4 (element_size=2)."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u16; 4] = [0x1111, 0x2222, 0x3333, 0x4444];

            #[zeropage(0x30)]
            static mut RESULT: u16;

            #[entry]
            fn main() {
                // &ARR[2] should be address 0x24 (0x20 + 2*2)
                let p: *u16 = &ARR[2];
                RESULT = *p;
            }
        ''', ExpectedState(memory={
            0x7E0030: [0x33, 0x33],  # ARR[2] == 0x3333
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_zp_array_variable_index_u8(self, e2e):
        """&zp_array[x] with variable index should compute base+x."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0x30)]
            static mut RESULT: u8;

            #[zeropage(0x31)]
            static mut IDX: u8 = 3;

            fn read_via_ptr(ptr: *u8) -> u8 {
                return *ptr;
            }

            #[entry]
            fn main() {
                let i: u8 = IDX;
                let p: *u8 = &ARR[i];
                RESULT = read_via_ptr(p);
            }
        ''', ExpectedState(memory={
            0x7E0030: 40,  # ARR[3] == 40
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_zp_array_variable_index_u16_elements(self, e2e):
        """&zp_array_u16[x] with variable index and element_size=2."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u16; 4] = [0x1111, 0x2222, 0x3333, 0x4444];

            #[zeropage(0x30)]
            static mut RESULT: u16;

            #[zeropage(0x32)]
            static mut IDX: u8 = 1;

            #[entry]
            fn main() {
                let i: u8 = IDX;
                let p: *u16 = &ARR[i];
                RESULT = *p;
            }
        ''', ExpectedState(memory={
            0x7E0030: [0x22, 0x22],  # ARR[1] == 0x2222
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_ram_array_constant_index_far_pointer(self, e2e):
        """&ram_array[2] should produce far pointer (bank $7E)."""
        result = e2e.run('''
            #[ram]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0x30)]
            static mut RESULT: u8;

            fn read_via_far_ptr(ptr: far *u8) -> u8 {
                return *ptr;
            }

            #[entry]
            fn main() {
                let p: far *u8 = &ARR[2];
                RESULT = read_via_far_ptr(p);
            }
        ''', ExpectedState(memory={
            0x7E0030: 30,  # ARR[2] == 30
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_ram_array_variable_index_far_pointer(self, e2e):
        """&ram_array[x] with variable index should produce far pointer."""
        result = e2e.run('''
            #[ram]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0x30)]
            static mut RESULT: u8;

            #[zeropage(0x31)]
            static mut IDX: u8 = 5;

            fn read_via_far_ptr(ptr: far *u8) -> u8 {
                return *ptr;
            }

            #[entry]
            fn main() {
                let i: u8 = IDX;
                let p: far *u8 = &ARR[i];
                RESULT = read_via_far_ptr(p);
            }
        ''', ExpectedState(memory={
            0x7E0030: 60,  # ARR[5] == 60
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_address_of_first_element(self, e2e):
        """&array[0] should equal &array (base address)."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];

            #[zeropage(0x30)]
            static mut RESULT: u8;

            fn read_via_ptr(ptr: *u8) -> u8 {
                return *ptr;
            }

            #[entry]
            fn main() {
                let p: *u8 = &ARR[0];
                RESULT = read_via_ptr(p);
            }
        ''', ExpectedState(memory={
            0x7E0030: 0xAA,  # ARR[0] == 0xAA
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_write_via_address_of_element(self, e2e):
        """Write through pointer from &array[index]."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u8; 4] = [0, 0, 0, 0];

            fn write_via_ptr(ptr: *u8, val @ A: u8) {
                *ptr = val;
            }

            #[entry]
            fn main() {
                let p: *u8 = &ARR[2];
                write_via_ptr(p, 0x42);
            }
        ''', ExpectedState(memory={
            0x7E0022: 0x42,  # ARR[2] at address 0x20+2
        }))
        assert result.success, f"Failures: {result.failures}"
