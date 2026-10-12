# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Static initializers with aggregates inside them.

An initializer element used to have to be an integer or `&STATIC`, so an
array or struct literal *inside* one - an `#[soa]` field `arr: [u8; 3]`, a
2-D array, a struct's array or string field - failed to compile. Fills were
worse, and silent: an immutable `[5; 4]` emitted no data, so a read landed in
zero page, and a mutable fill of an aggregate (`[[7; 2]; 3]`) stored zeros.
And a single `#[soa]` instance would not take its own struct literal.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import HIRError, MIRLoweringError
from r65.tests.e2e import ExpectedState


SOA_POOL = '''
    struct Pos { x: u8, y: u16 }
    #[soa]
    struct G { a: u8, arr: [u8; 3], p: Pos, b: u16 }
    #[{storage}]
    static mut POOL: [G; 2] = [
        G { a: 1, arr: [4, 5, 6], p: Pos { x: 7, y: 0x0809 }, b: 0x1234 },
        G { a: 2, arr: [10, 11, 12], p: Pos { x: 13, y: 0x0E0F }, b: 0x5678 },
    ];
    #[zeropage(0x11)] static mut PY: u16;
    #[zeropage(0x13)] static mut OB: u16;
    #[zeropage(0x15)] static mut OA: u8;

    #[entry]
    fn main() {
        PY = POOL[0].p.y;
        OB = POOL[1].b;
        OA = POOL[1].a;
    }
'''

# The planes, in field order from the pool's address: `a`, then `arr` and
# `p` each padded to a 4-byte stride, then `b`.
SOA_PLANES = [1, 2,
              4, 5, 6, 0, 10, 11, 12, 0,
              7, 0x09, 0x08, 0, 13, 0x0F, 0x0E, 0,
              0x34, 0x12, 0x78, 0x56]


def _soa_expected(base):
    return ExpectedState(memory={
        base: SOA_PLANES,
        0x7E0011: [0x09, 0x08],
        0x7E0013: [0x78, 0x56],
        0x7E0015: 2,
    })


def test_a_lowram_soa_pool_with_array_and_struct_fields(e2e):
    result = e2e.run(SOA_POOL.replace('{storage}', 'lowram(0x0300)'), _soa_expected(0x7E0300))
    assert result.success, f"Failures: {result.failures}"


def test_a_ram_soa_pool_with_array_and_struct_fields(e2e):
    result = e2e.run(SOA_POOL.replace('{storage}', 'ram'), _soa_expected(0x7E2000))
    assert result.success, f"Failures: {result.failures}"
    assert result.cpu.DBR == 0


def test_a_single_soa_instance_takes_its_struct_literal(e2e):
    result = e2e.run('''
        #[soa]
        struct One { a: u8, b: u16 }
        #[lowram]
        static mut P: One = One { a: 0x42, b: 0xBEEF };
        #[zeropage(0x10)] static mut OA: u8;
        #[zeropage(0x11)] static mut OB: u16;

        #[entry]
        fn main() {
            OA = P.a;
            OB = P.b;
        }
    ''', ExpectedState(memory={0x7E0010: 0x42, 0x7E0011: [0xEF, 0xBE]}))
    assert result.success, f"Failures: {result.failures}"


def test_a_two_dimensional_array_initializer(e2e):
    result = e2e.run('''
        #[lowram(0x0300)]
        static mut TWO: [[u8; 3]; 2] = [[1, 2, 3], [4, 5, 6]];

        #[entry]
        fn main() {}
    ''', ExpectedState(memory={0x7E0300: [1, 2, 3, 4, 5, 6]}))
    assert result.success, f"Failures: {result.failures}"


def test_a_struct_table_with_array_and_string_fields(e2e):
    """The bytes are the same table a ROM static would get; here it is
    copied to a fixed address so the test can read it whole."""
    result = e2e.run('''
        struct Row { name: [u8; 4], tiles: [u8; 2], z: u8 }
        #[lowram(0x0300)]
        static mut ROWS: [Row; 2] = [
            Row { name: "AB", tiles: [1, 2], z: 3 },
            Row { name: "WXYZ", tiles: [4, 5], z: 6 },
        ];

        #[entry]
        fn main() {}
    ''', ExpectedState(memory={0x7E0300: [0x41, 0x42, 0, 0, 1, 2, 3,
                                           0x57, 0x58, 0x59, 0x5A, 4, 5, 6]}))
    assert result.success, f"Failures: {result.failures}"


def test_an_immutable_fill_has_its_bytes_in_rom(e2e):
    """Used to emit no table: the read went to zero page."""
    result = e2e.run('''
        static T: [u8; 4] = [5; 4];
        #[zeropage(0x02)] static mut DECOY: u8;
        #[zeropage(0x10)] static mut OUT: u8;

        #[entry]
        fn main() {
            DECOY = 0x99;
            OUT = T[2];
        }
    ''', ExpectedState(memory={0x7E0010: 5}))
    assert result.success, f"Failures: {result.failures}"


def test_a_fill_of_arrays(e2e):
    """Used to fill with 0: the value was no single byte."""
    result = e2e.run('''
        #[lowram(0x0300)]
        static mut GRID: [[u8; 2]; 3] = [[7; 2]; 3];

        #[entry]
        fn main() {}
    ''', ExpectedState(memory={0x7E0300: [7, 7, 7, 7, 7, 7]}))
    assert result.success, f"Failures: {result.failures}"


def test_a_single_soa_instance_rejects_an_array_literal():
    with pytest.raises(HIRError, match="takes a `One"):
        compile_string('''
            #[soa]
            struct One { a: u8, b: u16 }
            #[lowram]
            static mut P: One = [One { a: 1, b: 2 }];
            #[entry]
            fn main() { P.a = 3; }
        ''', cfg_options=['snes'])


def test_a_fill_of_a_runtime_value_is_an_error():
    """Used to be a fill with 0."""
    with pytest.raises(MIRLoweringError, match="compile-time constant"):
        compile_string('''
            #[lowram] static mut V: u8;
            #[lowram] static mut BAD: [u8; 4] = [V; 4];
            #[entry]
            fn main() { V = 1; }
        ''', cfg_options=['snes'])
