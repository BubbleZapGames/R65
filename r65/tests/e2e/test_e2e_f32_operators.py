# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for operator overloading on the F32 stdlib type.

Values are built with `from_i16` and arithmetic is verified *through* the
comparison operators (e.g. `V += W; if V == EXP`), so no float bit-patterns are
hardcoded — small-integer float results (3+4, 3*4, 12/4) are exact.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
F32_PATH = STDLIB_DIR / "F32.r65"


def read_u8(cpu, zp_addr):
    return cpu.memory.read(0x7E0000 + zp_addr)


SOURCE = f'''
    include!("{SNESLIB_PATH}")
    include!("{F32_PATH}")

    #[zeropage(0x02, register)] static mut SCRATCH0: u8;
    #[zeropage(0x04, register)] static mut SCRATCH1: u16;

    #[lowram(0x0300)] static mut V: F32;
    #[lowram(0x0304)] static mut W: F32;
    #[lowram(0x0308)] static mut EXP: F32;

    #[zeropage(0x20)] static mut R_ADD: u8;
    #[zeropage(0x21)] static mut R_SUB: u8;
    #[zeropage(0x22)] static mut R_MUL: u8;
    #[zeropage(0x23)] static mut R_DIV: u8;
    #[zeropage(0x24)] static mut R_LT: u8;
    #[zeropage(0x25)] static mut R_GE: u8;
    #[zeropage(0x26)] static mut R_NE: u8;

    #[entry]
    fn main() {{
        // 3 + 4 == 7
        V.from_i16(3 as i16); W.from_i16(4 as i16); V += W; EXP.from_i16(7 as i16);
        if V == EXP {{ R_ADD = 1; }} else {{ R_ADD = 0; }}

        // 10 - 4 == 6
        V.from_i16(10 as i16); W.from_i16(4 as i16); V -= W; EXP.from_i16(6 as i16);
        if V == EXP {{ R_SUB = 1; }} else {{ R_SUB = 0; }}

        // 3 * 4 == 12
        V.from_i16(3 as i16); W.from_i16(4 as i16); V *= W; EXP.from_i16(12 as i16);
        if V == EXP {{ R_MUL = 1; }} else {{ R_MUL = 0; }}

        // 12 / 4 == 3
        V.from_i16(12 as i16); W.from_i16(4 as i16); V /= W; EXP.from_i16(3 as i16);
        if V == EXP {{ R_DIV = 1; }} else {{ R_DIV = 0; }}

        // ordering: 3 < 4 (true), 3 >= 4 (false), 3 != 4 (true)
        V.from_i16(3 as i16); W.from_i16(4 as i16);
        if V < W  {{ R_LT = 1; }} else {{ R_LT = 0; }}
        if V >= W {{ R_GE = 1; }} else {{ R_GE = 0; }}
        if V != W {{ R_NE = 1; }} else {{ R_NE = 0; }}
    }}
'''


class TestF32Operators:
    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(SOURCE)
        return e2e.execute(rom, max_instructions=5_000_000)

    def test_compound_assign(self, cpu):
        assert read_u8(cpu, 0x20) == 1, "3 + 4 == 7"
        assert read_u8(cpu, 0x21) == 1, "10 - 4 == 6"
        assert read_u8(cpu, 0x22) == 1, "3 * 4 == 12"
        assert read_u8(cpu, 0x23) == 1, "12 / 4 == 3"

    def test_comparisons(self, cpu):
        assert read_u8(cpu, 0x24) == 1, "3 < 4"
        assert read_u8(cpu, 0x25) == 0, "3 >= 4 is false"
        assert read_u8(cpu, 0x26) == 1, "3 != 4"
