# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Behavioural checks for 16-bit shifts of 8 or more.

`_emit_shift_by` starts such a shift with XBA — the byte swap that *is* a
shift by 8 — masks off the half being discarded, and shifts the remainder.
The substitution is only sound because the mask runs before the residual
shifts; without it XBA's other half walks back into the result. These run
the emitted code and check the value that comes out.

`x * 256` and `x / 256` route through the same path, so they are covered
here rather than with the other arithmetic tests.
"""

from r65.tests.e2e import E2ETest
from r65.tests.e2e.framework import ExpectedState


def test_shift_right_10_masked(e2e):
    """The ClassicKong case: `(target >> 10) & 0x1F`.

    $9C40 >> 10 is $0027; the mask trims bit 5, leaving $0007.
    """
    result = e2e.run('''
        #[ram] static mut TARGET: u16;
        #[zeropage(0x10)] static mut R: u16;
        #[entry] fn main() {
            TARGET = 0x9C40;
            R = (TARGET >> 10) & 0x1F;
        }
    ''', ExpectedState(memory={0x7E0010: [0x07, 0x00]}))
    assert result.success, f"Failures: {result.failures}"


def test_shift_right_residual_counts(e2e):
    """9, 12, and 15 — one past the XBA boundary, mid-range, and the top.

    $ABCD >> 9 is $0055, >> 12 is $000A, >> 15 is $0001.
    """
    result = e2e.run('''
        #[ram] static mut W: u16;
        #[zeropage(0x10)] static mut A9: u16;
        #[zeropage(0x12)] static mut A12: u16;
        #[zeropage(0x14)] static mut A15: u16;
        #[entry] fn main() {
            W = 0xABCD;
            A9 = W >> 9;
            A12 = W >> 12;
            A15 = W >> 15;
        }
    ''', ExpectedState(memory={
        0x7E0010: [0x55, 0x00],
        0x7E0012: [0x0A, 0x00],
        0x7E0014: [0x01, 0x00],
    }))
    assert result.success, f"Failures: {result.failures}"


def test_discarded_half_does_not_leak_back(e2e):
    """The mask *before* the residual shifts is what this pins down.

    Both values have a non-zero half that XBA parks where the residual
    shifts would walk it back into the result:

    - $FF12 >> 12 is $000F. Without the AND the residual LSRs see $12FF
      and yield $012F.
    - $12FF << 12 is $F000. Without the AND the residual ASLs see $FF12
      and yield $F120.
    """
    result = e2e.run('''
        #[ram] static mut W1: u16;
        #[ram] static mut W2: u16;
        #[zeropage(0x10)] static mut R1: u16;
        #[zeropage(0x12)] static mut R2: u16;
        #[entry] fn main() {
            W1 = 0xFF12; R1 = W1 >> 12;
            W2 = 0x12FF; R2 = W2 << 12;
        }
    ''', ExpectedState(memory={
        0x7E0010: [0x0F, 0x00],
        0x7E0012: [0x00, 0xF0],
    }))
    assert result.success, f"Failures: {result.failures}"


def test_shift_left_residual_counts(e2e):
    """$ABCD << 9 is $9A00, << 12 is $D000, << 15 is $8000."""
    result = e2e.run('''
        #[ram] static mut W: u16;
        #[zeropage(0x10)] static mut A9: u16;
        #[zeropage(0x12)] static mut A12: u16;
        #[zeropage(0x14)] static mut A15: u16;
        #[entry] fn main() {
            W = 0xABCD;
            A9 = W << 9;
            A12 = W << 12;
            A15 = W << 15;
        }
    ''', ExpectedState(memory={
        0x7E0010: [0x00, 0x9A],
        0x7E0012: [0x00, 0xD0],
        0x7E0014: [0x00, 0x80],
    }))
    assert result.success, f"Failures: {result.failures}"


def test_divide_and_multiply_by_256(e2e):
    """`/ 256` and `* 256` are shifts of exactly 8, and take the XBA path."""
    result = e2e.run('''
        #[ram] static mut W: u16;
        #[zeropage(0x10)] static mut DIV: u16;
        #[zeropage(0x12)] static mut MUL: u16;
        #[entry] fn main() {
            W = 0xABCD;
            DIV = W / 256;
            MUL = W * 256;
        }
    ''', ExpectedState(memory={
        0x7E0010: [0xAB, 0x00],
        0x7E0012: [0x00, 0xCD],
    }))
    assert result.success, f"Failures: {result.failures}"


def test_shift_right_8_unchanged(e2e):
    """The count == 8 case had the XBA path already; it still works."""
    result = e2e.run('''
        #[ram] static mut W: u16;
        #[zeropage(0x10)] static mut R: u16;
        #[entry] fn main() { W = 0x1234; R = W >> 8; }
    ''', ExpectedState(memory={0x7E0010: [0x12, 0x00]}))
    assert result.success, f"Failures: {result.failures}"
