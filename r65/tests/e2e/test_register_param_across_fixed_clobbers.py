# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A value kept in A, X or Y survives a node whose fixed expansion overwrites it.

The slot allocator left a register parameter in its register across a clone
of more than AGGREGATE_COPY_UNROLL_MAX bytes (MVN over A, X and Y) and a
lookup-table match (TAX): its clobber rules knew calls and `asm!` but not
`fixed_clobbers`. `OUT = i` stored what the MVN or TAX left behind.
"""

import pytest

from r65.tests.e2e import ExpectedState

PRELUDE = '''
struct Big { a: u16, b: u16, c: u16, d: u16, e: u16 }
impl Clone for Big {}
#[zeropage(0x10, register)] static mut S0: u8;
#[zeropage(0x12, register)] static mut S1: u16;
#[lowram(0x0300)] static mut ARR: [u8; 16];
#[lowram(0x0320)] static mut BIG_A: Big;
#[lowram(0x0340)] static mut BIG_B: Big;
#[lowram(0x0380)] static mut OUT: u16;
#[lowram(0x0382)] static mut K: u8;
'''

CLONE = "BIG_B.clone_from(&BIG_A);"
LOOKUP = "K = match ARR[0] { 0 => 10, 1 => 11, 2 => 12, 3 => 13, _ => 14 };"


@pytest.mark.parametrize("reg, between", [
    ('X', CLONE), ('Y', CLONE), ('A', CLONE), ('X', LOOKUP),
], ids=["X-clone", "Y-clone", "A-clone", "X-lookup"])
def test_register_param_survives(e2e, reg, between):
    result = e2e.run(PRELUDE + f'''
        #[inline(never)]
        fn f(i @ {reg}: u16) {{ {between} OUT = i; }}

        #[entry]
        fn main() {{ ARR[0] = 2; f(0x1234); }}
    ''', ExpectedState(memory={0x7E0380: [0x34, 0x12]}), max_instructions=50000)
    assert result.success, f"Failures: {result.failures}"
