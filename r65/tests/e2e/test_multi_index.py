# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Indexing an element that is itself reached by an index.

`T[i][j]`, `ROWS[i].name[j]` and `POOL[i].arr[j]` (an `#[soa]` array field,
also `self.arr[j]` in its methods) used to fail in MIR lowering: an array base
had to fold into one MemoryLocation, and the inner index has a runtime part.
X now holds the sum of every variable index scaled by its stride; constant
indices fold into the address.

The indices live in zero page so nothing folds them away.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import MIRLoweringError
from r65.tests.e2e import ExpectedState


INDICES = '''
    #[zeropage(0x20)] static mut I: u16;
    #[zeropage(0x22)] static mut J: u16;
    #[zeropage(0x24)] static mut K: u16;
    #[zeropage(0x26)] static mut I8: u8;
'''


def test_a_two_dimensional_array(e2e):
    result = e2e.run(INDICES + '''
        #[lowram(0x0300)]
        static mut TWO: [[u8; 3]; 2];
        #[zeropage(0x10)] static mut O: [u8; 5];

        #[entry]
        fn main() {
            I = 1; J = 2; I8 = 1;
            let mut n: u8 = 1;
            for i in 0..2 {
                for j in 0..3 {
                    TWO[i][j] = n;
                    n = n + 1;
                }
            }
            O[0] = TWO[I][J];
            O[1] = TWO[1][J];
            O[2] = TWO[I][0];
            O[3] = TWO[0][1];
            O[4] = TWO[I8][J];
        }
    ''', ExpectedState(memory={0x7E0300: [1, 2, 3, 4, 5, 6],
                               0x7E0010: [6, 6, 4, 2, 6]}))
    assert result.success, f"Failures: {result.failures}"


def test_u16_elements_scale_the_inner_index(e2e):
    result = e2e.run(INDICES + '''
        #[lowram(0x0300)]
        static mut W: [[u16; 3]; 2];
        #[zeropage(0x10)] static mut O: u16;

        #[entry]
        fn main() {
            I = 1; J = 2;
            W[I][J] = 0xBEEF;
            W[I][J - 1] = 0x1234;
            O = W[I][J];
        }
    ''', ExpectedState(memory={0x7E0308: [0x34, 0x12, 0xEF, 0xBE],
                               0x7E0010: [0xEF, 0xBE]}))
    assert result.success, f"Failures: {result.failures}"


def test_a_three_dimensional_array(e2e):
    result = e2e.run(INDICES + '''
        #[lowram(0x0300)]
        static mut C: [[[u8; 2]; 2]; 2];
        #[zeropage(0x10)] static mut O: u8;

        #[entry]
        fn main() {
            I = 1; J = 0; K = 1;
            C[I][J][K] = 0x55;
            O = C[I][J][K];
        }
    ''', ExpectedState(memory={0x7E0305: 0x55, 0x7E0010: 0x55}))
    assert result.success, f"Failures: {result.failures}"


def test_an_array_field_of_a_struct_table(e2e):
    result = e2e.run(INDICES + '''
        struct Row { name: [u8; 4], z: u8 }
        #[lowram(0x0300)]
        static mut ROWS: [Row; 2];
        #[zeropage(0x10)] static mut O: u8;

        #[entry]
        fn main() {
            I = 1; J = 2;
            ROWS[I].name[J] = 0x41;
            ROWS[I].z = 0x07;
            O = ROWS[I].name[J];
        }
    ''', ExpectedState(memory={0x7E0307: [0x41, 0x00, 0x07], 0x7E0010: 0x41}))
    assert result.success, f"Failures: {result.failures}"


SOA_POOL = '''
    #[soa]
    struct G { a: u8, arr: [u8; 3] }
    impl G {
        fn get(self, j @ Y: u16) -> u8 { return self.arr[j]; }
    }
    #[lowram(0x0300)]
    static mut POOL: [G; 2];
'''


def test_an_soa_array_field(e2e):
    """`arr` is a 3-byte plane padded to a 4-byte stride after the 2-byte
    `a` plane: `POOL[1].arr[2]` is 0x0302 + 4 + 2."""
    result = e2e.run(INDICES + SOA_POOL + '''
        #[zeropage(0x10)] static mut O: [u8; 2];

        #[entry]
        fn main() {
            I = 1; J = 2;
            POOL[I].arr[J] = 9;
            POOL[I].arr[J] += 1;
            POOL[0].arr[J] = 3;
            O[0] = POOL[I].arr[J];
            O[1] = POOL[I].get(J);
        }
    ''', ExpectedState(memory={0x7E0304: 3, 0x7E0308: 10, 0x7E0010: [10, 10]}))
    assert result.success, f"Failures: {result.failures}"


def test_the_address_of_an_inner_element(e2e):
    result = e2e.run(INDICES + '''
        #[lowram(0x0300)]
        static mut TWO: [[u8; 3]; 2];
        #[zeropage(0x10)] static mut O: u8;
        #[zeropage(0x12)] static mut P: *u8;

        #[entry]
        fn main() {
            I = 1; J = 1;
            TWO[1][1] = 0x66;
            P = &TWO[I][J];
            O = *P;
        }
    ''', ExpectedState(memory={0x7E0010: 0x66, 0x7E0012: [0x04, 0x03]}))
    assert result.success, f"Failures: {result.failures}"


def test_an_element_reached_through_a_pointer_index_is_an_error():
    with pytest.raises(MIRLoweringError, match="reached through a pointer"):
        compile_string('''
            struct Row { name: [u8; 4], z: u8 }
            #[lowram] static mut ROWS: [Row; 2];
            #[zeropage(0x10)] static mut O: u8;
            fn read(p: *[Row; 2], i: u16, j: u16) -> u8 { return p[i].name[j]; }
            #[entry]
            fn main() { O = read(&ROWS, 1, 2); }
        ''', cfg_options=['snes'])
