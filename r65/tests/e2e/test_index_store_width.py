# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A store from X or Y must write exactly the destination's width, on hardware.

R65 is unconditionally x16, so `STX`/`STY` store two bytes. Nothing checked the
destination width before selecting them, so a 1-byte destination silently took
its neighbour with it. `r65/tests/compiler/codegen/test_store_from_index_register.py`
asserts on the emitted instructions; this file proves the resulting memory.

The guard byte placed immediately after each destination is the whole point of
these tests -- the stored value itself was always correct, so only the neighbour
reveals the bug.
"""

import pytest
from r65.tests.e2e import ExpectedState


class TestOneByteDestination:
    """A u8 counter promoted into Y, then stored."""

    SRC = """
#[zeropage(0x10)]
static mut OUT: u8;
#[zeropage(0x11)]
static mut GUARD: u8;
#[entry]
fn main() {
    GUARD = 0xEE;
    let mut t: u8 = 0;
    while t < 10 { t = t + 1; }
    OUT = t as u8;
}
"""

    def test_neighbour_byte_survives(self, e2e):
        result = self.SRC
        r = e2e.run(result, ExpectedState(memory={0x7E0010: 10, 0x7E0011: 0xEE}))
        assert r.success, f"Failures: {r.failures}"


class TestRamDestination:
    """A #[ram] destination is long-addressed, and `STY` has no long form -- this
    used to be a hard codegen error rather than a wrong store."""

    SRC = """
#[ram]
static mut OUT: u8;
#[ram]
static mut GUARD: u8;
#[entry]
fn main() {
    GUARD = 0xEE;
    let mut t: u8 = 0;
    while t < 10 { t = t + 1; }
    OUT = t as u8;
}
"""

    def test_compiles_and_stores_one_byte(self, e2e):
        r = e2e.run(self.SRC, ExpectedState(memory={0x7E2000: 10, 0x7E2001: 0xEE}))
        assert r.success, f"Failures: {r.failures}"


class TestTwoByteDestinationStillCorrect:
    """The anti-pessimization case must stay correct as well as direct: a real
    16-bit value in an index register still writes both its bytes."""

    SRC = """
#[zeropage(0x10)]
static mut W: u16;
#[zeropage(0x12)]
static mut V: u16;
far fn storex(p @ X: u16) { W = p; }
far fn storey(p @ Y: u16) { V = p; }
#[entry]
fn main() { storex(0x1234); storey(0x5678); }
"""

    def test_full_width_written(self, e2e):
        r = e2e.run(self.SRC, ExpectedState(memory={
            0x7E0010: [0x34, 0x12],
            0x7E0012: [0x78, 0x56],
        }))
        assert r.success, f"Failures: {r.failures}"


class TestAccumulatorNotClobbered:
    """The store routes through A, so a value already live in A has to survive
    it. The helper pushes and pulls around the transfer; if the two ran in
    different accumulator modes the frame would unbalance."""

    SRC = """
#[zeropage(0x10)]
static mut OUT: u8;
#[zeropage(0x11)]
static mut GUARD: u8;
#[zeropage(0x12)]
static mut KEPT: u8;
fn work(v @ A: u8) -> u8 {
    let mut t: u8 = 0;
    while t < 10 { t = t + 1; }
    OUT = t as u8;
    return v;
}
#[entry]
fn main() { GUARD = 0xEE; KEPT = work(0x5A); }
"""

    def test_a_bound_parameter_survives_the_store(self, e2e):
        r = e2e.run(self.SRC, ExpectedState(memory={
            0x7E0010: 10,
            0x7E0011: 0xEE,
            0x7E0012: 0x5A,
        }))
        assert r.success, f"Failures: {r.failures}"
