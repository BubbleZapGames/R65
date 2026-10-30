# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for &array[index] address-of array element operations.

Tests that taking the address of an array element produces the correct pointer
(base + index * element_size) for both constant and variable indices, near and
far pointers, and different element sizes.
"""

from r65.tests.e2e import ExpectedState


class TestAddressOfArrayIndex:
    """Test &array[index] produces correct pointer values."""

    def test_zp_array_constant_index_u8(self, e2e):
        """&zp_array[2] should produce near pointer to base+2."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0xB0)]
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
            0x7E00B0: 30,  # ARR[2] == 30
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_zp_array_constant_index_u16_elements(self, e2e):
        """&zp_array_u16[2] should produce near pointer to base+4 (element_size=2)."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u16; 4] = [0x1111, 0x2222, 0x3333, 0x4444];

            #[zeropage(0xB0)]
            static mut RESULT: u16;

            #[entry]
            fn main() {
                // &ARR[2] should be address 0x24 (0x20 + 2*2)
                let p: *u16 = &ARR[2];
                RESULT = *p;
            }
        ''', ExpectedState(memory={
            0x7E00B0: [0x33, 0x33],  # ARR[2] == 0x3333
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_zp_array_variable_index_u8(self, e2e):
        """&zp_array[x] with variable index should compute base+x."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0xB0)]
            static mut RESULT: u8;

            #[zeropage(0xB1)]
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
            0x7E00B0: 40,  # ARR[3] == 40
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_zp_array_variable_index_u16_elements(self, e2e):
        """&zp_array_u16[x] with variable index and element_size=2."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u16; 4] = [0x1111, 0x2222, 0x3333, 0x4444];

            #[zeropage(0xB0)]
            static mut RESULT: u16;

            #[zeropage(0xB2)]
            static mut IDX: u8 = 1;

            #[entry]
            fn main() {
                let i: u8 = IDX;
                let p: *u16 = &ARR[i];
                RESULT = *p;
            }
        ''', ExpectedState(memory={
            0x7E00B0: [0x22, 0x22],  # ARR[1] == 0x2222
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_ram_array_constant_index_far_pointer(self, e2e):
        """&ram_array[2] should produce far pointer (bank $7E)."""
        result = e2e.run('''
            #[ram]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0xB0)]
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
            0x7E00B0: 30,  # ARR[2] == 30
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_ram_array_variable_index_far_pointer(self, e2e):
        """&ram_array[x] with variable index should produce far pointer."""
        result = e2e.run('''
            #[ram]
            static mut ARR: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

            #[zeropage(0xB0)]
            static mut RESULT: u8;

            #[zeropage(0xB1)]
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
            0x7E00B0: 60,  # ARR[5] == 60
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_address_of_first_element(self, e2e):
        """&array[0] should equal &array (base address)."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut ARR: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];

            #[zeropage(0xB0)]
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
            0x7E00B0: 0xAA,  # ARR[0] == 0xAA
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

    def test_pointer_to_array_element_steps_by_the_element(self, e2e):
        """&p[i] on a `*[u8; 16]` is p + i, as p[i] reads it - not p + 16 * i."""
        result = e2e.run('''
            #[lowram(0x0300)] static mut ONE: [u8; 16];
            #[zeropage(0x10)] static mut P: *[u8; 16];
            #[zeropage(0x12)] static mut Q: *u8;
            #[zeropage(0x14)] static mut I: u16;
            #[zeropage(0xB0)] static mut RESULT: u8;
            #[entry]
            fn main() {
                ONE[5] = 0x66;
                P = &ONE; I = 5;
                Q = &P[I];
                RESULT = *Q;
            }
        ''', ExpectedState(memory={0x7E0012: [0x05, 0x03], 0x7E00B0: 0x66}))
        assert result.success, f"Failures: {result.failures}"

    def test_nested_ram_element_is_far(self, e2e):
        """An element below a field or a second index of a #[ram] static is
        in bank $7E, so its address is far; a `*u8` parameter it reaches is
        promoted."""
        result = e2e.run('''
            struct R { z: u8, cells: [u8; 4] }
            #[ram] static mut RT: [[u8; 4]; 4];
            #[ram] static mut RR: [R; 2];
            #[zeropage(0x14)] static mut I: u16;
            #[zeropage(0x1C)] static mut FQ: far *u8;
            #[zeropage(0xB0)] static mut RESULT: [u8; 3];
            fn rd(p: *u8) -> u8 { return *p; }
            #[entry]
            fn main() {
                RT[1][2] = 0x12; RR[1].cells[2] = 0x31;
                I = 1;
                FQ = &RT[I][2];          RESULT[0] = *FQ;
                FQ = &RR[I].cells[2];    RESULT[1] = *FQ;
                RESULT[2] = rd(&RT[I][2]);
            }
        ''', ExpectedState(memory={0x7E00B0: [0x12, 0x31, 0x12]}))
        assert result.success, f"Failures: {result.failures}"

    def test_element_through_a_far_pointer_is_far(self, e2e):
        """An element reached through a far pointer lives where the pointer
        points, so its address keeps the bank."""
        result = e2e.run('''
            struct R { z: u8, cells: [u8; 4] }
            #[ram] static mut RONE: R;
            #[ram] static mut RT: [[u8; 4]; 4];
            #[zeropage(0x10)] static mut FR: far *R;
            #[zeropage(0x13)] static mut FS: far *[u8; 4];
            #[zeropage(0x16)] static mut FP: far *[[u8; 4]; 4];
            #[zeropage(0x1C)] static mut FQ: far *u8;
            #[zeropage(0x20)] static mut I: u16;
            #[zeropage(0xB0)] static mut RESULT: [u8; 4];
            #[entry]
            fn main() {
                RONE.z = 0x41; RONE.cells[3] = 0x42; RT[2][1] = 0x21;
                FR = &RONE; FS = &RONE.cells; FP = &RT;
                I = 2;
                FQ = &FR.z;          RESULT[0] = *FQ;
                FQ = &FR.cells[3];   RESULT[1] = *FQ;
                FQ = &(*FS)[3];      RESULT[2] = *FQ;
                FQ = &FP[I][1];      RESULT[3] = *FQ;
            }
        ''', ExpectedState(memory={0x7E00B0: [0x41, 0x42, 0x42, 0x21]}))
        assert result.success, f"Failures: {result.failures}"
