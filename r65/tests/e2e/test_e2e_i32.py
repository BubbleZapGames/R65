# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for I32 (32-bit signed integer) stdlib.

Batched: multiple operations compiled into a single ROM per test class
to reduce compile+assemble+link overhead from ~50 compilations to 4.

Result slots use lowram addresses at $0200+ to avoid stack overlap
(stack starts at $01FF and grows downward).
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
I32_PATH = STDLIB_DIR / "I32.r65"


def i32_bytes(value):
    """Convert a 32-bit signed value to 4-byte little-endian list (two's complement)."""
    if value < 0:
        value = value + 0x100000000
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def read_i32(cpu, lowram_addr):
    """Read 4 bytes (LE) from lowram via SNES address."""
    snes = 0x7E0000 + lowram_addr
    return [cpu.memory.read(snes + i) for i in range(4)]


def read_u8(cpu, zp_addr):
    """Read 1 byte from zeropage via SNES address."""
    return cpu.memory.read(0x7E0000 + zp_addr)


# ── Common header ────────────────────────────────────────────────────────────

COMMON_HEADER = f'''
    include!("{SNESLIB_PATH}")
    include!("{I32_PATH}")

    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;

    #[zeropage(0x10)]
    static mut V: I32;
    #[ram]
    static mut W: I32;
'''

# ── Fast operations batch 1: literals, from_i16, from_u16, neg, abs, is_neg ─

FAST_SOURCE_1 = COMMON_HEADER + f'''
    // ROM-initialized sources for literal tests
    #[ram] static mut SRC_0: I32 = I32!(0);
    #[ram] static mut SRC_1000: I32 = I32!(1000);
    #[ram] static mut SRC_100000: I32 = I32!(100000);
    #[ram] static mut SRC_NEG1: I32 = I32!(-1);
    #[ram] static mut SRC_NEG42: I32 = I32!(-42);
    #[ram] static mut SRC_NEG100000: I32 = I32!(-100000);

    // Result slots: literals
    #[lowram(0x0200)] static mut LIT0: I32;
    #[lowram(0x0204)] static mut LIT1: I32;
    #[lowram(0x0208)] static mut LIT2: I32;
    #[lowram(0x020C)] static mut LIT3: I32;
    #[lowram(0x0210)] static mut LIT4: I32;
    #[lowram(0x0214)] static mut LIT5: I32;

    // Result slots: from_i16
    #[lowram(0x0218)] static mut FI0: I32;
    #[lowram(0x021C)] static mut FI1: I32;
    #[lowram(0x0220)] static mut FI2: I32;
    #[lowram(0x0224)] static mut FI3: I32;

    // Result slots: from_u16
    #[lowram(0x0228)] static mut FU0: I32;

    // Result slots: neg
    #[lowram(0x022C)] static mut NG0: I32;
    #[lowram(0x0230)] static mut NG1: I32;
    #[lowram(0x0234)] static mut NG2: I32;

    // Result slots: abs
    #[lowram(0x0238)] static mut AB0: I32;
    #[lowram(0x023C)] static mut AB1: I32;
    #[lowram(0x0240)] static mut AB2: I32;

    // Scalar: is_negative
    #[zeropage(0x20)] static mut IS_NEG0: u8;
    #[zeropage(0x21)] static mut IS_NEG1: u8;

    #[entry]
    fn main() {{
        // === Literals ===
        V.lo = SRC_0.lo; V.hi = SRC_0.hi;
        LIT0.lo = V.lo; LIT0.hi = V.hi;

        V.lo = SRC_1000.lo; V.hi = SRC_1000.hi;
        LIT1.lo = V.lo; LIT1.hi = V.hi;

        V.lo = SRC_100000.lo; V.hi = SRC_100000.hi;
        LIT2.lo = V.lo; LIT2.hi = V.hi;

        V.lo = SRC_NEG1.lo; V.hi = SRC_NEG1.hi;
        LIT3.lo = V.lo; LIT3.hi = V.hi;

        V.lo = SRC_NEG42.lo; V.hi = SRC_NEG42.hi;
        LIT4.lo = V.lo; LIT4.hi = V.hi;

        V.lo = SRC_NEG100000.lo; V.hi = SRC_NEG100000.hi;
        LIT5.lo = V.lo; LIT5.hi = V.hi;

        // === from_i16 ===
        V.from_i16(1000 as i16);
        FI0.lo = V.lo; FI0.hi = V.hi;

        V.from_i16(-100 as i16);
        FI1.lo = V.lo; FI1.hi = V.hi;

        V.from_i16(0 as i16);
        FI2.lo = V.lo; FI2.hi = V.hi;

        V.from_i16(-1 as i16);
        FI3.lo = V.lo; FI3.hi = V.hi;

        // === from_u16 ===
        V.from_u16(1000);
        FU0.lo = V.lo; FU0.hi = V.hi;

        // === neg ===
        V.from_i16(100 as i16); V.neg();
        NG0.lo = V.lo; NG0.hi = V.hi;

        V.from_i16(-100 as i16); V.neg();
        NG1.lo = V.lo; NG1.hi = V.hi;

        V.from_i16(0 as i16); V.neg();
        NG2.lo = V.lo; NG2.hi = V.hi;

        // === abs ===
        V.from_i16(100 as i16); V.abs();
        AB0.lo = V.lo; AB0.hi = V.hi;

        V.from_i16(-100 as i16); V.abs();
        AB1.lo = V.lo; AB1.hi = V.hi;

        V.from_i16(0 as i16); V.abs();
        AB2.lo = V.lo; AB2.hi = V.hi;

        // === is_negative ===
        V.from_i16(100 as i16);
        if V.is_negative() {{
            IS_NEG0 = 1;
        }} else {{
            IS_NEG0 = 0;
        }}

        V.from_i16(-100 as i16);
        if V.is_negative() {{
            IS_NEG1 = 1;
        }} else {{
            IS_NEG1 = 0;
        }}
    }}
'''

# ── Fast operations batch 2: mul, add, sub, shl, sar, cmp ───────────────────

FAST_SOURCE_2 = COMMON_HEADER + f'''
    // Result slots: mul (first to avoid state interaction)
    #[lowram(0x0200)] static mut MU0: I32;
    #[lowram(0x0204)] static mut MU1: I32;
    #[lowram(0x0208)] static mut MU2: I32;
    #[lowram(0x020C)] static mut MU3: I32;

    // Result slots: add
    #[lowram(0x0210)] static mut AD0: I32;
    #[lowram(0x0214)] static mut AD1: I32;
    #[lowram(0x0218)] static mut AD2: I32;

    // Result slots: sub
    #[lowram(0x021C)] static mut SU0: I32;
    #[lowram(0x0220)] static mut SU1: I32;

    // Result slots: shl
    #[lowram(0x0224)] static mut SL0: I32;

    // Result slots: sar
    #[lowram(0x0228)] static mut SA0: I32;
    #[lowram(0x022C)] static mut SA1: I32;
    #[lowram(0x0230)] static mut SA2: I32;

    // Scalar: comparison operators
    #[zeropage(0x20)] static mut CMP_EQ: u8;
    #[zeropage(0x21)] static mut CMP_NE: u8;
    #[zeropage(0x22)] static mut CMP_GT: u8;
    #[zeropage(0x23)] static mut CMP_GE: u8;
    #[zeropage(0x24)] static mut CMP_LT: u8;
    #[zeropage(0x25)] static mut CMP_LE: u8;

    #[entry]
    fn main() {{
        // === mul (first to avoid state interaction) ===
        V.from_i16(100 as i16); W.from_i16(200 as i16); V *= W;
        MU0.lo = V.lo; MU0.hi = V.hi;

        V.from_i16(100 as i16); W.from_i16(-5 as i16); V *= W;
        MU1.lo = V.lo; MU1.hi = V.hi;

        V.from_i16(-10 as i16); W.from_i16(-20 as i16); V *= W;
        MU2.lo = V.lo; MU2.hi = V.hi;

        V.from_i16(100 as i16); W.from_i16(0 as i16); V *= W;
        MU3.lo = V.lo; MU3.hi = V.hi;

        // === add ===
        V.from_i16(100 as i16); W.from_i16(200 as i16); V += W;
        AD0.lo = V.lo; AD0.hi = V.hi;

        V.from_i16(100 as i16); W.from_i16(-50 as i16); V += W;
        AD1.lo = V.lo; AD1.hi = V.hi;

        V.from_i16(-100 as i16); W.from_i16(-200 as i16); V += W;
        AD2.lo = V.lo; AD2.hi = V.hi;

        // === sub ===
        V.from_i16(300 as i16); W.from_i16(100 as i16); V -= W;
        SU0.lo = V.lo; SU0.hi = V.hi;

        V.from_i16(100 as i16); W.from_i16(200 as i16); V -= W;
        SU1.lo = V.lo; SU1.hi = V.hi;

        // === shl ===
        V.from_i16(1 as i16); V.shl(4);
        SL0.lo = V.lo; SL0.hi = V.hi;

        // === sar ===
        V.from_i16(16 as i16); V.sar(2);
        SA0.lo = V.lo; SA0.hi = V.hi;

        V.from_i16(-16 as i16); V.sar(2);
        SA1.lo = V.lo; SA1.hi = V.hi;

        V.from_i16(-1 as i16); V.sar(1);
        SA2.lo = V.lo; SA2.hi = V.hi;

        // === comparison operators ===
        V.from_i16(100 as i16); W.from_i16(100 as i16);
        if V == W {{ CMP_EQ = 1; }} else {{ CMP_EQ = 0; }}
        if V != W {{ CMP_NE = 1; }} else {{ CMP_NE = 0; }}

        V.from_i16(100 as i16); W.from_i16(-100 as i16);
        if V > W  {{ CMP_GT = 1; }} else {{ CMP_GT = 0; }}
        if V >= W {{ CMP_GE = 1; }} else {{ CMP_GE = 0; }}

        V.from_i16(-100 as i16); W.from_i16(100 as i16);
        if V < W  {{ CMP_LT = 1; }} else {{ CMP_LT = 0; }}
        if V <= W {{ CMP_LE = 1; }} else {{ CMP_LE = 0; }}
    }}
'''

# ── Slow operations source ──────────────────────────────────────────────────

SLOW_SOURCE = COMMON_HEADER + f'''
    // Result slots: div
    #[lowram(0x0200)] static mut DV0: I32;
    #[lowram(0x0204)] static mut DV1: I32;
    #[lowram(0x0208)] static mut DV2: I32;
    #[lowram(0x020C)] static mut DV3: I32;
    #[lowram(0x0210)] static mut DV4: I32;
    #[lowram(0x0214)] static mut DV5: I32;

    // Result slots: mod
    #[lowram(0x0218)] static mut MD0: I32;
    #[lowram(0x021C)] static mut MD1: I32;
    #[lowram(0x0220)] static mut MD2: I32;

    // Result slots: mod_i16
    #[lowram(0x0224)] static mut MI0: I32;
    #[lowram(0x0228)] static mut MI1: I32;
    #[lowram(0x022C)] static mut MI2: I32;
    #[lowram(0x0230)] static mut MI3: I32;
    #[lowram(0x0234)] static mut MI4: I32;

    #[entry]
    fn main() {{
        // === div ===
        V.from_i16(1000 as i16); W.from_i16(10 as i16); V /= W;
        DV0.lo = V.lo; DV0.hi = V.hi;

        V.from_i16(-1000 as i16); W.from_i16(10 as i16); V /= W;
        DV1.lo = V.lo; DV1.hi = V.hi;

        V.from_i16(1000 as i16); W.from_i16(-10 as i16); V /= W;
        DV2.lo = V.lo; DV2.hi = V.hi;

        V.from_i16(-1000 as i16); W.from_i16(-10 as i16); V /= W;
        DV3.lo = V.lo; DV3.hi = V.hi;

        V.from_i16(7 as i16); W.from_i16(2 as i16); V /= W;
        DV4.lo = V.lo; DV4.hi = V.hi;

        V.from_i16(1000 as i16); W.from_i16(0 as i16); V /= W;
        DV5.lo = V.lo; DV5.hi = V.hi;

        // === mod ===
        V.from_i16(1000 as i16); W.from_i16(7 as i16); V.mod(&W);
        MD0.lo = V.lo; MD0.hi = V.hi;

        V.from_i16(-1000 as i16); W.from_i16(7 as i16); V.mod(&W);
        MD1.lo = V.lo; MD1.hi = V.hi;

        V.from_i16(1000 as i16); W.from_i16(10 as i16); V.mod(&W);
        MD2.lo = V.lo; MD2.hi = V.hi;

        // === mod_i16 ===
        V.from_i16(1000 as i16); V.mod_i16(7);
        MI0.lo = V.lo; MI0.hi = V.hi;

        V.from_i16(-1000 as i16); V.mod_i16(7);
        MI1.lo = V.lo; MI1.hi = V.hi;

        V.from_i16(1000 as i16); V.mod_i16(-7 as i16);
        MI2.lo = V.lo; MI2.hi = V.hi;

        V.from_i16(1000 as i16); V.mod_i16(10);
        MI3.lo = V.lo; MI3.hi = V.hi;

        V.from_i16(1000 as i16); V.mod_i16(0);
        MI4.lo = V.lo; MI4.hi = V.hi;
    }}
'''


# ── Test classes ─────────────────────────────────────────────────────────────

class TestI32FastOps1:
    """Batched tests: literals, from_i16, from_u16, neg, abs, is_negative."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(FAST_SOURCE_1)
        return e2e.execute(rom, max_instructions=200000)

    def test_literal_macro(self, cpu):
        """I32! literal initialization macro."""
        assert read_i32(cpu, 0x0200) == i32_bytes(0), "I32!(0)"
        assert read_i32(cpu, 0x0204) == i32_bytes(1000), "I32!(1000)"
        assert read_i32(cpu, 0x0208) == i32_bytes(100000), "I32!(100000)"
        assert read_i32(cpu, 0x020C) == i32_bytes(-1), "I32!(-1)"
        assert read_i32(cpu, 0x0210) == i32_bytes(-42), "I32!(-42)"
        assert read_i32(cpu, 0x0214) == i32_bytes(-100000), "I32!(-100000)"

    def test_from_i16(self, cpu):
        """from_i16 sign-extends i16 to I32."""
        assert read_i32(cpu, 0x0218) == i32_bytes(1000), "from_i16(1000)"
        assert read_i32(cpu, 0x021C) == i32_bytes(-100), "from_i16(-100)"
        assert read_i32(cpu, 0x0220) == i32_bytes(0), "from_i16(0)"
        assert read_i32(cpu, 0x0224) == i32_bytes(-1), "from_i16(-1)"

    def test_from_u16(self, cpu):
        """from_u16 zero-extends u16 to I32."""
        assert read_i32(cpu, 0x0228) == i32_bytes(1000), "from_u16(1000)"

    def test_neg(self, cpu):
        """Two's complement negation."""
        assert read_i32(cpu, 0x022C) == i32_bytes(-100), "neg(100)"
        assert read_i32(cpu, 0x0230) == i32_bytes(100), "neg(-100)"
        assert read_i32(cpu, 0x0234) == i32_bytes(0), "neg(0)"

    def test_abs(self, cpu):
        """Absolute value."""
        assert read_i32(cpu, 0x0238) == i32_bytes(100), "abs(100)"
        assert read_i32(cpu, 0x023C) == i32_bytes(100), "abs(-100)"
        assert read_i32(cpu, 0x0240) == i32_bytes(0), "abs(0)"

    def test_is_negative(self, cpu):
        """is_negative returns correct boolean."""
        assert read_u8(cpu, 0x20) == 0, "is_negative(100) -> false"
        assert read_u8(cpu, 0x21) == 1, "is_negative(-100) -> true"


class TestI32FastOps2:
    """Batched tests: add, sub, mul, shl, sar, cmp."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(FAST_SOURCE_2)
        return e2e.execute(rom, max_instructions=600000)

    def test_mul(self, cpu):
        """I32 multiplication with sign handling."""
        assert read_i32(cpu, 0x0200) == i32_bytes(20000), "100*200"
        assert read_i32(cpu, 0x0204) == i32_bytes(-500), "100*(-5)"
        assert read_i32(cpu, 0x0208) == i32_bytes(200), "(-10)*(-20)"
        assert read_i32(cpu, 0x020C) == i32_bytes(0), "100*0"

    def test_add(self, cpu):
        """I32 addition."""
        assert read_i32(cpu, 0x0210) == i32_bytes(300), "100+200"
        assert read_i32(cpu, 0x0214) == i32_bytes(50), "100+(-50)"
        assert read_i32(cpu, 0x0218) == i32_bytes(-300), "-100+(-200)"

    def test_sub(self, cpu):
        """I32 subtraction."""
        assert read_i32(cpu, 0x021C) == i32_bytes(200), "300-100"
        assert read_i32(cpu, 0x0220) == i32_bytes(-100), "100-200"

    def test_shl(self, cpu):
        """I32 shift left."""
        assert read_i32(cpu, 0x0224) == i32_bytes(16), "1<<4"

    def test_sar(self, cpu):
        """I32 arithmetic shift right (preserves sign)."""
        assert read_i32(cpu, 0x0228) == i32_bytes(4), "16>>2"
        assert read_i32(cpu, 0x022C) == i32_bytes(-4), "-16>>2 (sign preserved)"
        assert read_i32(cpu, 0x0230) == i32_bytes(-1), "-1>>1 (all bits set)"

    def test_comparison_operators(self, cpu):
        """I32 comparison operators (==, !=, <, <=, >, >=), signed."""
        assert read_u8(cpu, 0x20) == 1, "100 == 100"
        assert read_u8(cpu, 0x21) == 0, "100 != 100 is false"
        assert read_u8(cpu, 0x22) == 1, "100 > -100"
        assert read_u8(cpu, 0x23) == 1, "100 >= -100"
        assert read_u8(cpu, 0x24) == 1, "-100 < 100"
        assert read_u8(cpu, 0x25) == 1, "-100 <= 100"


class TestI32SlowOps:
    """Batched tests for slow I32 operations (div, mod)."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(SLOW_SOURCE)
        return e2e.execute(rom, max_instructions=2000000)

    def test_div(self, cpu):
        """I32 division with sign handling."""
        assert read_i32(cpu, 0x0200) == i32_bytes(100), "1000/10"
        assert read_i32(cpu, 0x0204) == i32_bytes(-100), "-1000/10"
        assert read_i32(cpu, 0x0208) == i32_bytes(-100), "1000/(-10)"
        assert read_i32(cpu, 0x020C) == i32_bytes(100), "-1000/(-10)"
        assert read_i32(cpu, 0x0210) == i32_bytes(3), "7/2 (truncation)"
        assert read_i32(cpu, 0x0214) == i32_bytes(-2147483648), "1000/0 (MIN_I32)"

    def test_mod(self, cpu):
        """I32 modulo with sign handling."""
        assert read_i32(cpu, 0x0218) == i32_bytes(6), "1000%7"
        assert read_i32(cpu, 0x021C) == i32_bytes(-6), "-1000%7"
        assert read_i32(cpu, 0x0220) == i32_bytes(0), "1000%10"

    def test_mod_i16(self, cpu):
        """I32 scalar modulo (mod_i16)."""
        assert read_i32(cpu, 0x0224) == i32_bytes(6), "1000%i16(7)"
        assert read_i32(cpu, 0x0228) == i32_bytes(-6), "-1000%i16(7)"
        assert read_i32(cpu, 0x022C) == i32_bytes(6), "1000%i16(-7)"
        assert read_i32(cpu, 0x0230) == i32_bytes(0), "1000%i16(10)"
        assert read_i32(cpu, 0x0234) == i32_bytes(1000), "1000%i16(0) unchanged"
