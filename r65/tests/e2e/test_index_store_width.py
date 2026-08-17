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


class TestPromotedCounterStoredToMemory:
    """A loop counter that is stored to memory is now kept in an index register.

    That promotion and the STX/STY width rule are coupled: before the width fix
    this exact shape would have emitted `STX $10` and zeroed the guard byte, so
    relaxing the promotion guard first would have turned a slow loop into a
    wrong one.
    """

    ARR = "#[ram]\nstatic mut ARR: [u8; 64];\n"

    def test_u8_counter_stores_one_byte(self, e2e):
        src = (self.ARR
               + "#[zeropage(0x10)]\nstatic mut OUT: u8;\n"
               + "#[zeropage(0x11)]\nstatic mut GUARD: u8;\n"
               + "#[entry]\nfn main() {\n"
                 "    GUARD = 0xEE;\n"
                 "    let mut i: u8 = 0;\n"
                 "    while i < 8 { ARR[i] = 1; i = i + 1; }\n"
                 "    OUT = i;\n"
                 "}")
        r = e2e.run(src, ExpectedState(memory={0x7E0010: 8, 0x7E0011: 0xEE}))
        assert r.success, f"Failures: {r.failures}"

    def test_u16_counter_stores_both_bytes(self, e2e):
        src = (self.ARR
               + "#[zeropage(0x10)]\nstatic mut OUT: u16;\n"
               + "#[entry]\nfn main() {\n"
                 "    let mut i: u16 = 0;\n"
                 "    while i < 300 { i = i + 1; }\n"
                 "    OUT = i;\n"
                 "}")
        r = e2e.run(src, ExpectedState(memory={0x7E0010: [0x2C, 0x01]}))
        assert r.success, f"Failures: {r.failures}"

    def test_counter_stored_inside_the_loop(self, e2e):
        """The counter is written every iteration, so the transfer is paid each
        time rather than once. Correctness must hold regardless of whether that
        is a net win."""
        src = (self.ARR
               + "#[zeropage(0x10)]\nstatic mut OUT: u8;\n"
               + "#[zeropage(0x11)]\nstatic mut GUARD: u8;\n"
               + "#[entry]\nfn main() {\n"
                 "    GUARD = 0xEE;\n"
                 "    let mut i: u8 = 0;\n"
                 "    while i < 8 { OUT = i; ARR[i] = 1; i = i + 1; }\n"
                 "}")
        r = e2e.run(src, ExpectedState(memory={0x7E0010: 7, 0x7E0011: 0xEE}))
        assert r.success, f"Failures: {r.failures}"


class TestIndirectStoreFromPromotedCounter:
    """`StoreIndirect` with the counter as source, on hardware.

    `(zp),Y` addressing needs Y for the index, so a counter promoted to Y is in
    contention with it. Correct results here are the point: the unit tests pin
    which register holds what, these pin that the right bytes reach memory.
    """

    DECL = ("#[zeropage(0x40)]\nstatic mut PTR: *u8;\n"
            "#[lowram]\nstatic mut BUF: [u8; 32];\n"
            "#[zeropage(0x50)]\nstatic mut K: u8;\n"
            "#[zeropage(0x51)]\nstatic mut GUARD: u8;\n")

    def run(self, e2e, body: str):
        src = (self.DECL + "#[entry]\nfn main() { GUARD = 0xEE; PTR = &BUF;"
               " K = 0; let mut i: u8 = 0; " + body + " }")
        r = e2e.run(src, ExpectedState(memory={0x7E0051: 0xEE}))
        assert r.success, f"Failures: {r.failures}"
        return r

    def test_index_and_value_are_the_same_counter(self, e2e):
        r = self.run(e2e, "while i < 8 { PTR[i] = i; i = i + 1; }")
        for n in range(8):
            got = r.cpu.memory.read(0x7E0200 + n)
            assert got == n, f"BUF[{n}] = {got}, expected {n}"

    def test_index_and_value_differ(self, e2e):
        """K walks 0,2,4.. while i walks 0..7, so a Y serving both roles would
        write the counter to the wrong slots."""
        r = self.run(e2e, "while i < 8 { PTR[K] = i; i = i + 1; K = K + 2; }")
        for n in range(8):
            got = r.cpu.memory.read(0x7E0200 + n * 2)
            assert got == n, f"BUF[{n * 2}] = {got}, expected {n}"
            gap = r.cpu.memory.read(0x7E0200 + n * 2 + 1)
            assert gap == 0, f"BUF[{n * 2 + 1}] = {gap}, expected untouched"
