# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Behavioural checks for narrowing casts whose result stays in A.

`TypeConvert` is now an A-coalescence def type in the slot allocator, so a
`u16 -> u8` cast no longer round-trips through a stack slot. These run the
coalesced code and confirm the byte that comes out, including the paths
where the truncated value has to survive a mode switch or a call.
"""

from r65.tests.e2e import E2ETest
from r65.tests.e2e.framework import ExpectedState


def test_register_parameter_narrowed(e2e):
    """A u16 arriving in A, truncated and returned."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut R: u8;
        far fn low(v @ A: u16) -> u8 { return v as u8; }
        #[entry] fn main() { R = low(0x1234); }
    ''', ExpectedState(memory={0x7E0010: 0x34}))
    assert result.success, f"Failures: {result.failures}"


def test_computed_value_narrowed(e2e):
    """The truncation reads the arithmetic result straight out of A."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut R: u8;
        far fn f(v @ A: u16) -> u8 { return (v + 0xFF) as u8; }
        #[entry] fn main() { R = f(0x1201); }
    ''', ExpectedState(memory={0x7E0010: 0x00}))
    assert result.success, f"Failures: {result.failures}"


def test_shift_then_narrow_in_register(e2e):
    """`(x >> 8) as u8` on a register value — XBA with no mask."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut R: u8;
        far fn hi(v @ A: u16) -> u8 { return ((v + 1) >> 8) as u8; }
        #[entry] fn main() { R = hi(0x7EFF); }
    ''', ExpectedState(memory={0x7E0010: 0x7F}))
    assert result.success, f"Failures: {result.failures}"


def test_signed_word_narrowed(e2e):
    """A negative i16 keeps its low byte, not its sign."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut R: u8;
        far fn low(v @ A: i16) -> u8 { return v as u8; }
        #[entry] fn main() { R = low(-2); }
    ''', ExpectedState(memory={0x7E0010: 0xFE}))
    assert result.success, f"Failures: {result.failures}"


def test_local_word_narrowed(e2e):
    """A local that never leaves A still narrows correctly."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut R: u8;
        #[zeropage(0x12)] static mut W: u16;
        #[entry] fn main() {
            W = 0xBEEF;
            let t: u16 = W;
            R = (t + 1) as u8;
        }
    ''', ExpectedState(memory={0x7E0010: 0xF0}))
    assert result.success, f"Failures: {result.failures}"


def test_narrowed_value_survives_being_stored_twice(e2e):
    """Two consumers of one coalesced result must both see it."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut A1: u8;
        #[zeropage(0x11)] static mut B1: u8;
        #[zeropage(0x12)] static mut W: u16;
        #[entry] fn main() {
            W = 0x4455;
            let b: u8 = W as u8;
            A1 = b;
            B1 = b + 1;
        }
    ''', ExpectedState(memory={0x7E0010: 0x55, 0x7E0011: 0x56}))
    assert result.success, f"Failures: {result.failures}"


def test_narrowed_value_used_as_call_argument(e2e):
    """A coalesced byte passed straight into a call."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut R: u8;
        #[zeropage(0x12)] static mut W: u16;
        far fn twice(v @ A: u8) -> u8 { return v + v; }
        #[entry] fn main() { W = 0x1122; R = twice(W as u8); }
    ''', ExpectedState(memory={0x7E0010: 0x44}))
    assert result.success, f"Failures: {result.failures}"


def test_widening_still_correct(e2e):
    """The direction that was deliberately left alone still works."""
    result = e2e.run('''
        #[zeropage(0x10)] static mut W: u16;
        #[zeropage(0x12)] static mut SW: i16;
        #[entry] fn main() {
            let b: u8 = 0x80;
            let n: i8 = -3;
            W = b as u16;
            SW = n as i16;
        }
    ''', ExpectedState(memory={0x7E0010: 0x80, 0x7E0011: 0x00,
                               0x7E0012: 0xFD, 0x7E0013: 0xFF}))
    assert result.success, f"Failures: {result.failures}"
