# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Behavioural checks for memory-operand folding.

`MemoryOperandFolder` makes the right operand of `a op b` a direct memory
read. These run the folded code and confirm the arithmetic — and that a
write between the read and its use still sees the old value.
"""

from r65.tests.e2e.framework import ExpectedState


TWO = '''
    #[zeropage(0x20)] static mut G_X: u8;
    #[zeropage(0x21)] static mut G_VX: u8;
    #[zeropage(0x22)] static mut R: u8;
'''


def test_add_sub_and_or_xor(e2e):
    result = e2e.run(TWO + '''
        #[zeropage(0x23)] static mut OUT: [u8; 5];
        #[entry] fn main() {
            G_X = 0x5A; G_VX = 0x0F;
            OUT[0] = G_X + G_VX;
            OUT[1] = G_X - G_VX;
            OUT[2] = G_X & G_VX;
            OUT[3] = G_X | G_VX;
            OUT[4] = G_X ^ G_VX;
        }
    ''', ExpectedState(memory={0x7E0023: [0x69, 0x4B, 0x0A, 0x5F, 0x55]}))
    assert result.success, f"Failures: {result.failures}"


def test_compare_both_ways(e2e):
    result = e2e.run(TWO + '''
        #[entry] fn main() {
            G_X = 3; G_VX = 7; R = 0;
            if G_X < G_VX { R = R | 1; }
            if G_VX < G_X { R = R | 2; }
            if G_X == G_VX { R = R | 4; }
            if G_X != G_VX { R = R | 8; }
        }
    ''', ExpectedState(memory={0x7E0022: 0x09}))
    assert result.success, f"Failures: {result.failures}"


def test_indexed_elements(e2e):
    result = e2e.run('''
        #[zeropage(0x20)] static mut P: [u8; 4];
        #[zeropage(0x24)] static mut Q: [u8; 4];
        fn f(i @ X: u16) { P[i] = P[i] + Q[i]; }
        #[entry] fn main() {
            P[2] = 0x10; Q[2] = 0x05;
            P[3] = 0x20; Q[3] = 0x01;
            f(2); f(3);
        }
    ''', ExpectedState(memory={0x7E0022: [0x15, 0x21]}))
    assert result.success, f"Failures: {result.failures}"


def test_sixteen_bit(e2e):
    result = e2e.run('''
        #[zeropage(0x20)] static mut A16: u16;
        #[zeropage(0x22)] static mut B16: u16;
        #[entry] fn main() { A16 = 0x12F0; B16 = 0x0120; A16 = A16 + B16; }
    ''', ExpectedState(memory={0x7E0020: [0x10, 0x14]}))
    assert result.success, f"Failures: {result.failures}"


def test_signed_reinterpret(e2e):
    """`x + (v as u8)` with `v: i8 = -1` subtracts one."""
    result = e2e.run('''
        #[zeropage(0x20)] static mut G_X: u8;
        #[zeropage(0x21)] static mut G_VX: i8;
        #[entry] fn main() { G_X = 10; G_VX = -1; G_X = G_X + (G_VX as u8); }
    ''', ExpectedState(memory={0x7E0020: 9}))
    assert result.success, f"Failures: {result.failures}"


def test_write_between_read_and_use_sees_the_old_value(e2e):
    result = e2e.run(TWO + '''
        #[entry] fn main() {
            G_X = 1; G_VX = 5;
            let v: u8 = G_VX;
            G_VX = 0x40;
            R = G_X + v;
        }
    ''', ExpectedState(memory={0x7E0022: 6}))
    assert result.success, f"Failures: {result.failures}"
