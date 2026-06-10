# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for `Clone` on the I32/U32/F32 stdlib types.

Each type has an empty `impl Clone for T {}` (auto bitwise copy). Verified
through the `==` operator: a clone equals its source, stays independent when the
source is mutated, and `clone_from` overwrites the destination.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"


def read_u8(cpu, zp_addr):
    return cpu.memory.read(0x7E0000 + zp_addr)


def _src(inc, T, ctor, cast, v1, v2):
    return f'''
    include!("{SNESLIB_PATH}")
    include!("{STDLIB_DIR / inc}")

    #[zeropage(0x02, register)] static mut SCRATCH0: u8;
    #[zeropage(0x04, register)] static mut SCRATCH1: u16;

    #[ram] static mut SRC: {T};
    #[ram] static mut DST: {T};

    #[zeropage(0x20)] static mut R_EQ: u8;     // clone == source
    #[zeropage(0x21)] static mut R_INDEP: u8;  // clone tracks mutated source? (want 0)
    #[zeropage(0x22)] static mut R_CF: u8;     // clone_from copied

    #[entry]
    fn main() {{
        SRC.{ctor}({v1} as {cast});
        let c = SRC.clone();                          // sugar: auto bitwise copy
        if c == SRC {{ R_EQ = 1; }} else {{ R_EQ = 0; }}

        SRC.{ctor}({v2} as {cast});                   // mutate the source
        if c == SRC {{ R_INDEP = 1; }} else {{ R_INDEP = 0; }}  // c must be unchanged -> 0

        DST.clone_from(&SRC);                          // in-place copy (SRC now v2)
        if DST == SRC {{ R_CF = 1; }} else {{ R_CF = 0; }}
    }}
'''


class TestNumericClone:
    @pytest.mark.parametrize("inc,T,ctor,cast,v1,v2", [
        ("I32.r65", "I32", "from_i16", "i16", 1234, 42),
        ("U32.r65", "U32", "from_u16", "u16", 50000, 100),
        ("F32.r65", "F32", "from_i16", "i16", 7, 3),
    ])
    def test_clone(self, inc, T, ctor, cast, v1, v2):
        e2e = E2ETest()
        rom = e2e.compile(_src(inc, T, ctor, cast, v1, v2))
        cpu = e2e.execute(rom, max_instructions=2_000_000)
        assert read_u8(cpu, 0x20) == 1, f"{T}: clone == source"
        assert read_u8(cpu, 0x21) == 0, f"{T}: clone independent of mutated source"
        assert read_u8(cpu, 0x22) == 1, f"{T}: clone_from copied"
