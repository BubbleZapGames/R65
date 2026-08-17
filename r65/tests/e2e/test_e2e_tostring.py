# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for ToString trait implementations in U32, I32, and F32.

Each test class compiles one ROM with several to_string calls whose results
land in fixed lowram buffers at $0300+, then reads back and compares bytes.

Buffers start at $0300 to avoid overlap with the stdlib scratch variables
(__u32_tostr_tmp etc.) which are auto-allocated starting at $0200.
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
U32_PATH     = STDLIB_DIR / "U32.r65"
I32_PATH     = STDLIB_DIR / "I32.r65"
F32_PATH     = STDLIB_DIR / "F32.r65"


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_exact(cpu, lowram_addr, n):
    """Read exactly `n` bytes from lowram; return as Python str.

    `to_string` writes no terminator -- the count it returns is what delimits
    the string -- so scanning for a NUL would read whatever follows in the
    buffer. Callers pass the expected length.
    """
    return "".join(chr(cpu.memory.read(0x7E0000 + lowram_addr + i))
                   for i in range(n))


def read_u16(cpu, lowram_addr):
    """Read 16-bit little-endian value from lowram."""
    lo = cpu.memory.read(0x7E0000 + lowram_addr)
    hi = cpu.memory.read(0x7E0000 + lowram_addr + 1)
    return lo | (hi << 8)


SCRATCH = """
    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;
"""


# ── U32 ToString ─────────────────────────────────────────────────────────────

U32_SOURCE = f"""
    include!("{SNESLIB_PATH}")
    include!("{U32_PATH}")
    {SCRATCH}

    #[lowram]
    static mut VAL: U32;

    // Buffers at $0300+ to avoid stdlib auto-allocated scratch at $0200-$0214
    #[lowram(0x0300)] static mut BUF0: [u8; 16];   // U32 0
    #[lowram(0x0310)] static mut BUF1: [u8; 16];   // U32 1
    #[lowram(0x0320)] static mut BUF2: [u8; 16];   // U32 4294967295
    #[lowram(0x0330)] static mut BUF3: [u8; 16];   // U32 100
    #[lowram(0x0340)] static mut BUF4: [u8; 16];   // U32 65536
    #[lowram(0x0350)] static mut LEN0: [u8; 2];    // return value for case 0
    #[lowram(0x0352)] static mut LEN3: [u8; 2];    // return value for case 100

    #[entry]
    fn main() {{
        // Sentinel-fill the buffers: to_string writes no terminator, so the byte
        // at the returned count must still hold 0xFF afterwards.
        let mut f: u16 = 0;
        while f < 16 {{ BUF0[f] = 0xFF; BUF3[f] = 0xFF; f = f + 1; }}

        // 0
        VAL.lo = 0; VAL.hi = 0;
        let r0: u16 = VAL.to_string(&BUF0 as far *u8);
        LEN0[0] = r0 as u8; LEN0[1] = (r0 >> 8) as u8;

        // 1
        VAL.lo = 1; VAL.hi = 0;
        VAL.to_string(&BUF1 as far *u8);

        // 4294967295 (0xFFFF_FFFF)
        VAL.lo = 0xFFFF; VAL.hi = 0xFFFF;
        VAL.to_string(&BUF2 as far *u8);

        // 100
        VAL.lo = 100; VAL.hi = 0;
        let r3: u16 = VAL.to_string(&BUF3 as far *u8);
        LEN3[0] = r3 as u8; LEN3[1] = (r3 >> 8) as u8;

        // 65536
        VAL.lo = 0; VAL.hi = 1;
        VAL.to_string(&BUF4 as far *u8);
    }}
"""


class TestU32ToString:
    """Batched U32 to_string tests."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(U32_SOURCE)
        return e2e.execute(rom, max_instructions=2_000_000)

    def test_zero(self, cpu):
        assert read_exact(cpu, 0x0300, len("0")) == "0", "U32(0)"

    def test_zero_returns_1(self, cpu):
        assert read_u16(cpu, 0x0350) == 1, "to_string(0) returns 1"

    def test_no_terminator_written(self, cpu):
        """No NUL: the returned count delimits the string. The zero case takes an
        early return and the 100 case the main digit loop -- a terminator was
        removed from each, so both are checked."""
        assert cpu.memory.read(0x7E0000 + 0x0300 + 1) == 0xFF, "U32(0) zero path"
        assert cpu.memory.read(0x7E0000 + 0x0330 + 3) == 0xFF, "U32(100) main path"

    def test_one(self, cpu):
        assert read_exact(cpu, 0x0310, len("1")) == "1", "U32(1)"

    def test_max(self, cpu):
        assert read_exact(cpu, 0x0320, len("4294967295")) == "4294967295", "U32(0xFFFFFFFF)"

    def test_hundred(self, cpu):
        assert read_exact(cpu, 0x0330, len("100")) == "100", "U32(100)"

    def test_hundred_returns_3(self, cpu):
        assert read_u16(cpu, 0x0352) == 3, "to_string(100) returns 3"

    def test_65536(self, cpu):
        assert read_exact(cpu, 0x0340, len("65536")) == "65536", "U32(65536)"


# ── I32 ToString ─────────────────────────────────────────────────────────────

I32_SOURCE = f"""
    include!("{SNESLIB_PATH}")
    include!("{I32_PATH}")
    {SCRATCH}

    #[lowram]
    static mut VAL: I32;

    // Buffers at $0300+ to avoid stdlib auto-allocated scratch at $0200-$0214
    #[lowram(0x0300)] static mut BUF0: [u8; 16];   // I32  0
    #[lowram(0x0310)] static mut BUF1: [u8; 16];   // I32 -1
    #[lowram(0x0320)] static mut BUF2: [u8; 16];   // I32  2147483647
    #[lowram(0x0330)] static mut BUF3: [u8; 16];   // I32 -2147483648  (INT_MIN)
    #[lowram(0x0340)] static mut BUF4: [u8; 16];   // I32  100
    #[lowram(0x0350)] static mut BUF5: [u8; 16];   // I32 -100
    #[lowram(0x0360)] static mut LEN3: [u8; 2];    // return value for INT_MIN

    #[entry]
    fn main() {{
        let mut f: u16 = 0;
        while f < 16 {{ BUF0[f] = 0xFF; BUF4[f] = 0xFF; f = f + 1; }}

        // 0
        VAL.lo = 0; VAL.hi = 0;
        VAL.to_string(&BUF0 as far *u8);

        // -1
        VAL.lo = 0xFFFF; VAL.hi = 0xFFFF;
        VAL.to_string(&BUF1 as far *u8);

        // INT_MAX = 2147483647 (0x7FFF_FFFF)
        VAL.lo = 0xFFFF; VAL.hi = 0x7FFF;
        VAL.to_string(&BUF2 as far *u8);

        // INT_MIN = -2147483648
        VAL.lo = 0; VAL.hi = 0x8000;
        let r3: u16 = VAL.to_string(&BUF3 as far *u8);
        LEN3[0] = r3 as u8; LEN3[1] = (r3 >> 8) as u8;

        // 100
        VAL.lo = 100; VAL.hi = 0;
        VAL.to_string(&BUF4 as far *u8);

        // -100
        VAL.lo = 0xFF9C; VAL.hi = 0xFFFF;
        VAL.to_string(&BUF5 as far *u8);
    }}
"""


class TestI32ToString:
    """Batched I32 to_string tests."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(I32_SOURCE)
        return e2e.execute(rom, max_instructions=5_000_000)

    def test_zero(self, cpu):
        assert read_exact(cpu, 0x0300, len("0")) == "0", "I32(0)"

    def test_no_terminator_written(self, cpu):
        assert cpu.memory.read(0x7E0000 + 0x0300 + 1) == 0xFF, "I32(0) zero path"
        assert cpu.memory.read(0x7E0000 + 0x0340 + 3) == 0xFF, "I32(100) main path"

    def test_neg_one(self, cpu):
        assert read_exact(cpu, 0x0310, len("-1")) == "-1", "I32(-1)"

    def test_int_max(self, cpu):
        assert read_exact(cpu, 0x0320, len("2147483647")) == "2147483647", "I32(INT_MAX)"

    def test_int_min(self, cpu):
        assert read_exact(cpu, 0x0330, len("-2147483648")) == "-2147483648", "I32(INT_MIN)"

    def test_int_min_returns_11(self, cpu):
        assert read_u16(cpu, 0x0360) == 11, "to_string(INT_MIN) returns 11"

    def test_hundred(self, cpu):
        assert read_exact(cpu, 0x0340, len("100")) == "100", "I32(100)"

    def test_neg_hundred(self, cpu):
        assert read_exact(cpu, 0x0350, len("-100")) == "-100", "I32(-100)"


# ── F32 ToString ─────────────────────────────────────────────────────────────
# F32 constants (from F32.r65 pre-computed values):
#   F32_ZERO:    exp_hi=0x0000, mant_lo=0x0000  → 0.0
#   F32_ONE:     exp_hi=0x8140, mant_lo=0x0000  → 1.0
#   F32_TEN:     exp_hi=0x8450, mant_lo=0x0000  → 10.0
#   F32_HUNDRED: exp_hi=0x8764, mant_lo=0x0000  → 100.0
#   F32_PI:      exp_hi=0x8264, mant_lo=0x87ED  → ~3.14159
#   F32(-42.0):  exp_hi=0x86AC, mant_lo=0x0000  → -42.0

F32_SOURCE = f"""
    include!("{SNESLIB_PATH}")
    include!("{F32_PATH}")
    {SCRATCH}

    #[lowram]
    static mut VAL: F32;

    // Buffers at $0300+ to avoid stdlib auto-allocated scratch at $0200-$0219
    #[lowram(0x0300)] static mut BUF0: [u8; 16];   // 0.0
    #[lowram(0x0310)] static mut BUF1: [u8; 16];   // 1.0
    #[lowram(0x0320)] static mut BUF2: [u8; 16];   // 10.0
    #[lowram(0x0330)] static mut BUF3: [u8; 16];   // 100.0
    #[lowram(0x0340)] static mut BUF4: [u8; 16];   // ~3.14159 (pi)
    #[lowram(0x0350)] static mut BUF5: [u8; 16];   // -42.0
    #[lowram(0x0360)] static mut LEN0: [u8; 2];    // return value for 0.0

    #[entry]
    fn main() {{
        let mut f: u16 = 0;
        while f < 16 {{ BUF0[f] = 0xFF; BUF1[f] = 0xFF; f = f + 1; }}

        // 0.0
        VAL.mant_lo = 0; VAL.exp_hi = 0;
        let r0: u16 = VAL.to_string(&BUF0 as far *u8);
        LEN0[0] = r0 as u8; LEN0[1] = (r0 >> 8) as u8;

        // 1.0 (F32_ONE)
        VAL.mant_lo = 0; VAL.exp_hi = 0x8140;
        VAL.to_string(&BUF1 as far *u8);

        // 10.0 (F32_TEN)
        VAL.mant_lo = 0; VAL.exp_hi = 0x8450;
        VAL.to_string(&BUF2 as far *u8);

        // 100.0 (F32_HUNDRED)
        VAL.mant_lo = 0; VAL.exp_hi = 0x8764;
        VAL.to_string(&BUF3 as far *u8);

        // ~3.14159 (F32_PI)
        VAL.mant_lo = 0x87ED; VAL.exp_hi = 0x8264;
        VAL.to_string(&BUF4 as far *u8);

        // -42.0
        VAL.mant_lo = 0; VAL.exp_hi = 0x86AC;
        VAL.to_string(&BUF5 as far *u8);
    }}
"""


class TestF32ToString:
    """Batched F32 to_string tests."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(F32_SOURCE)
        return e2e.execute(rom, max_instructions=5_000_000)

    def test_zero(self, cpu):
        assert read_exact(cpu, 0x0300, len("0.0")) == "0.0", "F32(0.0)"

    def test_zero_returns_3(self, cpu):
        assert read_u16(cpu, 0x0360) == 3, "to_string(0.0) returns 3"

    def test_no_terminator_written(self, cpu):
        assert cpu.memory.read(0x7E0000 + 0x0300 + 3) == 0xFF, "F32(0.0) zero path"
        assert cpu.memory.read(0x7E0000 + 0x0310 + 3) == 0xFF, "F32(1.0) main path"

    def test_one(self, cpu):
        assert read_exact(cpu, 0x0310, len("1.0")) == "1.0", "F32(1.0)"

    def test_ten(self, cpu):
        assert read_exact(cpu, 0x0320, len("10.0")) == "10.0", "F32(10.0)"

    def test_hundred(self, cpu):
        assert read_exact(cpu, 0x0330, len("100.0")) == "100.0", "F32(100.0)"

    def test_pi(self, cpu):
        assert read_exact(cpu, 0x0340, len("3.14159")) == "3.14159", "F32(pi ~3.14159)"

    def test_neg_42(self, cpu):
        assert read_exact(cpu, 0x0350, len("-42.0")) == "-42.0", "F32(-42.0)"
