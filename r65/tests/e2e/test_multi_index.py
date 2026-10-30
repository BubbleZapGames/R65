# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Indexing an element that is itself reached by an index.

`T[i][j]`, `ROWS[i].name[j]` and `POOL[i].arr[j]` (an `#[soa]` array field,
also `self.arr[j]` in its methods) used to fail in MIR lowering: an array base
had to fold into one MemoryLocation, and the inner index has a runtime part.
X now holds the sum of every variable index scaled by its stride; constant
indices fold into the address. Through a pointer (`p[i][j]`, `self.c[i][j]`)
the sum goes to Y instead, for `(p),Y`.

The indices live in zero page so nothing folds them away.
"""

from r65.compiler.main import compile_string
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


# Through a pointer the same terms sum into Y, for `(p),Y`.

def test_a_two_dimensional_array_through_a_pointer(e2e):
    result = e2e.run(INDICES + '''
        #[lowram(0x0300)]
        static mut TWO: [[u8; 4]; 4];
        #[zeropage(0x10)] static mut O: [u8; 6];
        #[zeropage(0x18)] static mut P: *[[u8; 4]; 4];
        #[zeropage(0x1A)] static mut Q: *u8;

        #[entry]
        fn main() {
            P = &TWO;
            let mut n: u8 = 1;
            for i in 0..4 {
                for j in 0..4 {
                    P[i][j] = n;
                    n = n + 1;
                }
            }
            I = 2; J = 3; I8 = 1;
            O[0] = P[I][J];
            O[1] = P[1][2];
            O[2] = P[I][0];
            O[3] = (*P)[I][J];
            O[4] = P[I8][J];
            Q = &P[I][J];
            O[5] = *Q;
        }
    ''', ExpectedState(memory={0x7E0300: list(range(1, 17)),
                               0x7E0010: [12, 7, 9, 12, 8, 12]}))
    assert result.success, f"Failures: {result.failures}"


def test_u16_elements_through_a_pointer_scale_the_inner_index(e2e):
    result = e2e.run(INDICES + '''
        #[lowram(0x0300)]
        static mut W: [[u16; 3]; 2];
        #[zeropage(0x10)] static mut O: u16;
        #[zeropage(0x18)] static mut P: *[[u16; 3]; 2];

        #[entry]
        fn main() {
            P = &W;
            I = 1; J = 2;
            P[I][J] = 0xBEEF;
            P[I][J - 1] = 0x1234;
            O = P[I][J];
        }
    ''', ExpectedState(memory={0x7E0308: [0x34, 0x12, 0xEF, 0xBE],
                               0x7E0010: [0xEF, 0xBE]}))
    assert result.success, f"Failures: {result.failures}"


def test_a_two_dimensional_field_through_a_pointer(e2e):
    """`cells` sits after `z`, so `P.cells[1][2]` is 0x0300 + 1 + 3 + 2."""
    result = e2e.run(INDICES + '''
        struct Board { z: u8, cells: [[u8; 3]; 2] }
        impl Board {
            near fn get(*self, i: u16, j: u16) -> u8 { return self.cells[i][j]; }
        }
        #[lowram(0x0300)]
        static mut BOARD: Board;
        #[zeropage(0x10)] static mut O: [u8; 2];
        #[zeropage(0x18)] static mut P: *Board;

        #[entry]
        fn main() {
            P = &BOARD;
            I = 1; J = 2;
            P.cells[I][J] = 0x5A;
            O[0] = P.cells[I][J];
            O[1] = BOARD.get(I, J);
        }
    ''', ExpectedState(memory={0x7E0306: 0x5A, 0x7E0010: [0x5A, 0x5A]}))
    assert result.success, f"Failures: {result.failures}"


def test_an_element_reached_through_a_pointer_index(e2e):
    result = e2e.run(INDICES + '''
        struct Row { name: [u8; 4], z: u8 }
        #[lowram(0x0300)] static mut ROWS: [Row; 2];
        #[zeropage(0x10)] static mut O: u8;
        fn read(p: *[Row; 2], i: u16, j: u16) -> u8 { return p[i].name[j]; }
        #[entry]
        fn main() {
            ROWS[1].name[2] = 0x41;
            O = read(&ROWS, 1, 2);
        }
    ''', ExpectedState(memory={0x7E0010: 0x41}))
    assert result.success, f"Failures: {result.failures}"


def test_a_field_of_an_indexed_pointer(e2e):
    """`p[i].b` reads the pointee. It used to index the pointer variable's
    own bytes, and a parameter `p` failed in codegen."""
    result = e2e.run(INDICES + '''
        struct Pair { a: u8, b: u8 }
        #[lowram(0x0300)] static mut PAIRS: [Pair; 3];
        #[zeropage(0x10)] static mut O: [u8; 2];
        #[zeropage(0x18)] static mut P: *Pair;
        #[zeropage(0x1A)] static mut GUARD: [u8; 4] = [0; 4];
        fn second(p: *Pair, i: u16) -> u8 { return p[i].b; }
        #[entry]
        fn main() {
            P = &PAIRS[0];
            I = 2;
            P[I].b = 0x77;
            P[I].a = 0x66;
            O[0] = P[I].b;
            O[1] = second(P, 2);
        }
    ''', ExpectedState(memory={0x7E0304: [0x66, 0x77], 0x7E0010: [0x77, 0x77],
                               0x7E001A: [0, 0, 0, 0]}))
    assert result.success, f"Failures: {result.failures}"


# An inner index the loop never changes is scaled once, before the loop.

ROW_SUM = '''
    #[lowram(0x0300)]
    static mut T: [[u8; 4]; 4];
    #[zeropage(0x10)] static mut O: [u8; 3];

    fn fill() {
        let mut n: u8 = 1;
        for i in 0..4 {
            for j in 0..4 {
                T[i][j] = n;
                n = n + 1;
            }
        }
    }

    #[used]
    fn row_sum(i: u16) -> u8 {
        let mut s: u8 = 0;
        for j in 0..4 { s = s + T[i][j]; }
        return s;
    }

    // `i` changes inside the loop, so it is rescaled every pass.
    #[used]
    fn diagonal_sum(i: u16) -> u8 {
        let mut s: u8 = 0;
        let mut r: u16 = i;
        for j in 0..4 {
            s = s + T[r][j];
            r = r + 1;
        }
        return s;
    }

    // `r` is declared in the loop, so it does not exist before it.
    #[used]
    fn column_sum(j: u16) -> u8 {
        let mut s: u8 = 0;
        for k in 0..4 {
            let r: u16 = 3 - k;
            s = s + T[r][j];
        }
        return s;
    }

    #[entry]
    fn main() {
        fill();
        O[0] = row_sum(2);
        O[1] = diagonal_sum(0);
        O[2] = column_sum(1);
    }
'''


def test_an_invariant_inner_index_is_scaled_before_the_loop():
    asm = compile_string(ROW_SUM, cfg_options=['snes']).splitlines()
    start = asm.index('row_sum:')
    body = asm[start:asm.index('    RTS', start)]
    loop = next(n for n, line in enumerate(body) if line.startswith('row_sum__L'))
    shifts = [n for n, line in enumerate(body) if line.strip() == 'ASL A']
    assert shifts and max(shifts) < loop, "i * 4 is recomputed inside the loop"


def test_hoisted_and_unhoisted_inner_indices(e2e):
    """Row 2 is 9..12; the diagonal 1 + 6 + 11 + 16; column 1, bottom up,
    14 + 10 + 6 + 2."""
    result = e2e.run(ROW_SUM, ExpectedState(memory={0x7E0010: [42, 34, 32]}))
    assert result.success, f"Failures: {result.failures}"


def test_an_invariant_inner_index_through_a_pointer_is_scaled_before_the_loop():
    asm = compile_string('''
        #[lowram(0x0300)] static mut T: [[u8; 4]; 4];
        #[zeropage(0x10)] static mut O: u8;
        #[used]
        fn row_sum(p: *[[u8; 4]; 4], i: u16) -> u8 {
            let mut s: u8 = 0;
            for j in 0..4 { s = s + p[i][j]; }
            return s;
        }
        #[entry]
        fn main() { O = row_sum(&T, 2); }
    ''', cfg_options=['snes']).splitlines()
    start = asm.index('row_sum:')
    body = asm[start:asm.index('    RTS', start)]
    loop = next(n for n, line in enumerate(body) if line.startswith('row_sum__L'))
    shifts = [n for n, line in enumerate(body) if line.strip() == 'ASL A']
    assert shifts and max(shifts) < loop, "i * 4 is recomputed inside the loop"
