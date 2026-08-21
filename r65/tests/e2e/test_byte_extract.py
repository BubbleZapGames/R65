# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Behavioural checks for the `(mem16 >> 8) as u8` fold.

`ByteExtractOptimizer` rewrites the expression into a load of the word's
high byte. The rewrite is only valid because the 65816 is little-endian
and because the cast keeps just the low byte of the shift result — these
run the folded code and confirm the value that comes out.
"""

from r65.tests.e2e import E2ETest
from r65.tests.e2e.framework import ExpectedState


def test_unsigned_high_byte(e2e):
    """The plain case: high byte of a zero-page word."""
    result = e2e.run('''
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x10)] static mut R: u8;
        #[entry] fn main() { W = 0x1234; R = (W >> 8) as u8; }
    ''', ExpectedState(memory={0x7E0010: 0x12}))
    assert result.success, f"Failures: {result.failures}"


def test_signed_negative_high_byte(e2e):
    """An arithmetic shift fills the top with sign, but the cast drops it.

    -2 is $FFFE; >> 8 is $FFFF signed or $00FF logical, and either way the
    byte that survives the cast is $FF — the original high byte.
    """
    result = e2e.run('''
        #[zeropage(0x20)] static mut W: i16;
        #[zeropage(0x10)] static mut R: u8;
        #[entry] fn main() { W = -2; R = (W >> 8) as u8; }
    ''', ExpectedState(memory={0x7E0010: 0xFF}))
    assert result.success, f"Failures: {result.failures}"


def test_far_ram_word(e2e):
    """A #[ram] static is reached by long addressing; +1 still applies."""
    result = e2e.run('''
        #[ram] static mut W: u16;
        #[zeropage(0x10)] static mut R: u8;
        #[entry] fn main() { W = 0xABCD; R = (W >> 8) as u8; }
    ''', ExpectedState(memory={0x7E0010: 0xAB}))
    assert result.success, f"Failures: {result.failures}"


def test_indexed_array_element(e2e):
    """The +1 must land inside the indexed element, not the array base."""
    result = e2e.run('''
        #[ram] static mut TBL: [u16; 4];
        #[zeropage(0x10)] static mut R: u8;
        #[entry] fn main() {
            TBL[0] = 0x1111; TBL[1] = 0x2233; TBL[2] = 0x4455;
            Y = 2;
            R = (TBL[Y] >> 8) as u8;
        }
    ''', ExpectedState(memory={0x7E0010: 0x44}))
    assert result.success, f"Failures: {result.failures}"


def test_struct_field(e2e):
    """A field's own offset composes with the high-byte offset."""
    result = e2e.run('''
        struct P { a: u16, b: u16 }
        #[ram] static mut PT: P;
        #[zeropage(0x10)] static mut R: u8;
        #[entry] fn main() { PT.a = 0x1111; PT.b = 0x9977; R = (PT.b >> 8) as u8; }
    ''', ExpectedState(memory={0x7E0010: 0x99}))
    assert result.success, f"Failures: {result.failures}"


def test_word_still_readable_after_fold(e2e):
    """Folding one use must not consume the load the other use needs."""
    result = e2e.run('''
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x10)] static mut HI: u8;
        #[zeropage(0x12)] static mut FULL: u16;
        #[entry] fn main() { W = 0x8844; HI = (W >> 8) as u8; FULL = W; }
    ''', ExpectedState(memory={0x7E0010: 0x88, 0x7E0012: 0x44, 0x7E0013: 0x88}))
    assert result.success, f"Failures: {result.failures}"


def test_reload_after_intervening_write(e2e):
    """Two reads of a variable written in between must see different values."""
    result = e2e.run('''
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x10)] static mut R: u8;
        #[entry] fn main() {
            W = 0x1234;
            R = (W >> 8) as u8;
            W = 0x5678;
            R = R + (W >> 8) as u8;
        }
    ''', ExpectedState(memory={0x7E0010: 0x12 + 0x56}))
    assert result.success, f"Failures: {result.failures}"


def test_shift_result_reused_is_not_folded(e2e):
    """A shift whose full result is used elsewhere keeps its real semantics."""
    result = e2e.run('''
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x10)] static mut BYTE: u8;
        #[zeropage(0x12)] static mut WORD: u16;
        #[entry] fn main() {
            W = 0xC0DE;
            let s: u16 = W >> 8;
            BYTE = s as u8;
            WORD = s;
        }
    ''', ExpectedState(memory={0x7E0010: 0xC0, 0x7E0012: 0xC0, 0x7E0013: 0x00}))
    assert result.success, f"Failures: {result.failures}"


def test_cast_to_signed_byte(e2e):
    """`as i8` reinterprets the same byte rather than normalizing it."""
    result = e2e.run('''
        #[zeropage(0x20)] static mut W: u16;
        #[zeropage(0x10)] static mut R: i8;
        #[entry] fn main() { W = 0xFE00; R = (W >> 8) as i8; }
    ''', ExpectedState(memory={0x7E0010: 0xFE}))
    assert result.success, f"Failures: {result.failures}"


def test_rom_array_element(e2e):
    """The high byte of a ROM table element, read through its label."""
    result = e2e.run('''
        static TBL: [u16; 4] = [0x1122, 0x3344, 0x5566, 0x7788];
        #[zeropage(0x10)] static mut HI: u8;
        #[zeropage(0x11)] static mut LO: u8;
        #[entry] fn main() {
            Y = 2;
            HI = (TBL[Y] >> 8) as u8;
            LO = TBL[Y] as u8;
        }
    ''', ExpectedState(memory={0x7E0010: 0x55, 0x7E0011: 0x66}))
    assert result.success, f"Failures: {result.failures}"
