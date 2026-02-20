"""
End-to-end tests for U32 (32-bit unsigned integer) stdlib.

Batched: multiple operations compiled into a single ROM per test class
to reduce compile+assemble+link overhead from ~46 compilations to 4.

Result slots use lowram addresses at $0200+ to avoid stack overlap
(stack starts at $01FF and grows downward).
"""

import pytest
from pathlib import Path
from r65.tests.e2e import E2ETest

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
U32_PATH = STDLIB_DIR / "U32.r65"


def u32_bytes(value):
    """Convert a 32-bit unsigned value to 4-byte little-endian list."""
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def read_u32(cpu, lowram_addr):
    """Read 4 bytes (LE) from lowram via SNES address."""
    snes = 0x7E0000 + lowram_addr
    return [cpu.memory.read(snes + i) for i in range(4)]


def read_u8(cpu, zp_addr):
    """Read 1 byte from zeropage via SNES address."""
    return cpu.memory.read(0x7E0000 + zp_addr)


# ── Common header ────────────────────────────────────────────────────────────

COMMON_HEADER = f'''
    include!("{SNESLIB_PATH}")
    include!("{U32_PATH}")

    #[zeropage(0x02, register)]
    static mut SCRATCH0: u8;
    #[zeropage(0x04, register)]
    static mut SCRATCH1: u16;

    #[zeropage(0x10)]
    static mut V: U32;
    #[ram]
    static mut W: U32;
'''

# ── Fast operations batch 1: literals, from_u16, to_u16, copy, add ──────────

FAST_SOURCE_1 = COMMON_HEADER + f'''
    // ROM-initialized sources for literal tests
    #[ram] static mut SRC_0: U32 = U32!(0);
    #[ram] static mut SRC_1000: U32 = U32!(1000);
    #[ram] static mut SRC_65535: U32 = U32!(65535);
    #[ram] static mut SRC_65536: U32 = U32!(65536);
    #[ram] static mut SRC_100000: U32 = U32!(100000);
    #[ram] static mut SRC_MAX: U32 = U32!(4294967295);

    // Result slots at $0200+ (above stack)
    #[lowram(0x0200)] static mut LIT0: U32;
    #[lowram(0x0204)] static mut LIT1: U32;
    #[lowram(0x0208)] static mut LIT2: U32;
    #[lowram(0x020C)] static mut LIT3: U32;
    #[lowram(0x0210)] static mut LIT4: U32;
    #[lowram(0x0214)] static mut LIT5: U32;

    #[lowram(0x0218)] static mut FU0: U32;
    #[lowram(0x021C)] static mut FU1: U32;
    #[lowram(0x0220)] static mut FU2: U32;

    #[lowram(0x0224)] static mut CP0: U32;

    #[lowram(0x0228)] static mut AD0: U32;
    #[lowram(0x022C)] static mut AD1: U32;
    #[lowram(0x0230)] static mut AD2: U32;
    #[lowram(0x0234)] static mut AD3: U32;

    #[zeropage(0x20)] static mut T0: u16;

    #[entry]
    fn main() {{
        // === Literals ===
        V.lo = SRC_0.lo; V.hi = SRC_0.hi;
        LIT0.lo = V.lo; LIT0.hi = V.hi;

        V.lo = SRC_1000.lo; V.hi = SRC_1000.hi;
        LIT1.lo = V.lo; LIT1.hi = V.hi;

        V.lo = SRC_65535.lo; V.hi = SRC_65535.hi;
        LIT2.lo = V.lo; LIT2.hi = V.hi;

        V.lo = SRC_65536.lo; V.hi = SRC_65536.hi;
        LIT3.lo = V.lo; LIT3.hi = V.hi;

        V.lo = SRC_100000.lo; V.hi = SRC_100000.hi;
        LIT4.lo = V.lo; LIT4.hi = V.hi;

        V.lo = SRC_MAX.lo; V.hi = SRC_MAX.hi;
        LIT5.lo = V.lo; LIT5.hi = V.hi;

        // === from_u16 ===
        V.from_u16(0);
        FU0.lo = V.lo; FU0.hi = V.hi;

        V.from_u16(1000);
        FU1.lo = V.lo; FU1.hi = V.hi;

        V.from_u16(65535);
        FU2.lo = V.lo; FU2.hi = V.hi;

        // === to_u16 ===
        V.from_u16(1234);
        T0 = V.to_u16();

        // === copy ===
        W.from_u16(42);
        V.copy(&W);
        CP0.lo = V.lo; CP0.hi = V.hi;

        // === add ===
        V.from_u16(100); W.from_u16(200); V.add(&W);
        AD0.lo = V.lo; AD0.hi = V.hi;

        V.from_u16(65535); W.from_u16(1); V.add(&W);
        AD1.lo = V.lo; AD1.hi = V.hi;

        V.from_u16(40000); W.from_u16(30000); V.add(&W);
        AD2.lo = V.lo; AD2.hi = V.hi;

        V.from_u16(1000); W.from_u16(0); V.add(&W);
        AD3.lo = V.lo; AD3.hi = V.hi;
    }}
'''

# ── Fast operations batch 2: sub, mul, shl, shr, cmp, combined ──────────────

FAST_SOURCE_2 = COMMON_HEADER + f'''
    // Result slots at $0200+ (above stack)
    #[lowram(0x0200)] static mut SU0: U32;
    #[lowram(0x0204)] static mut SU1: U32;
    #[lowram(0x0208)] static mut SU2: U32;

    #[lowram(0x020C)] static mut MU0: U32;
    #[lowram(0x0210)] static mut MU1: U32;
    #[lowram(0x0214)] static mut MU2: U32;
    #[lowram(0x0218)] static mut MU3: U32;

    #[lowram(0x021C)] static mut SL0: U32;
    #[lowram(0x0220)] static mut SL1: U32;
    #[lowram(0x0224)] static mut SL2: U32;

    #[lowram(0x0228)] static mut SR0: U32;
    #[lowram(0x022C)] static mut SR1: U32;
    #[lowram(0x0230)] static mut SR2: U32;

    #[lowram(0x0234)] static mut CO0: U32;

    #[zeropage(0x20)] static mut CMP_EQ: u8;
    #[zeropage(0x21)] static mut CMP_GT: u8;
    #[zeropage(0x22)] static mut CMP_LT: u8;

    #[entry]
    fn main() {{
        // === sub ===
        V.from_u16(300); W.from_u16(100); V.sub(&W);
        SU0.lo = V.lo; SU0.hi = V.hi;

        V.from_u16(100); W.from_u16(100); V.sub(&W);
        SU1.lo = V.lo; SU1.hi = V.hi;

        // 65536 - 1 = 65535 (borrow from hi to lo)
        V.from_u16(65535); W.from_u16(1); V.add(&W);
        W.from_u16(1); V.sub(&W);
        SU2.lo = V.lo; SU2.hi = V.hi;

        // === mul ===
        V.from_u16(100); W.from_u16(200); V.mul(&W);
        MU0.lo = V.lo; MU0.hi = V.hi;

        V.from_u16(12345); W.from_u16(0); V.mul(&W);
        MU1.lo = V.lo; MU1.hi = V.hi;

        V.from_u16(12345); W.from_u16(1); V.mul(&W);
        MU2.lo = V.lo; MU2.hi = V.hi;

        V.from_u16(1000); W.from_u16(100); V.mul(&W);
        MU3.lo = V.lo; MU3.hi = V.hi;

        // === shl ===
        V.from_u16(1); V.shl(4);
        SL0.lo = V.lo; SL0.hi = V.hi;

        V.from_u16(1); V.shl(16);
        SL1.lo = V.lo; SL1.hi = V.hi;

        V.from_u16(42); V.shl(0);
        SL2.lo = V.lo; SL2.hi = V.hi;

        // === shr ===
        V.from_u16(1024); V.shr(2);
        SR0.lo = V.lo; SR0.hi = V.hi;

        V.from_u16(1); V.shr(1);
        SR1.lo = V.lo; SR1.hi = V.hi;

        V.from_u16(1); V.shl(16); V.shr(16);
        SR2.lo = V.lo; SR2.hi = V.hi;

        // === cmp ===
        V.from_u16(100); W.from_u16(100);
        A = V.cmp(&W); CMP_EQ = A;

        V.from_u16(200); W.from_u16(100);
        A = V.cmp(&W); CMP_GT = A;

        V.from_u16(100); W.from_u16(200);
        A = V.cmp(&W); CMP_LT = A;

        // === combined: (100+200)-200 = 100 ===
        V.from_u16(100); W.from_u16(200);
        V.add(&W); V.sub(&W);
        CO0.lo = V.lo; CO0.hi = V.hi;
    }}
'''

# ── Slow operations source ──────────────────────────────────────────────────

SLOW_SOURCE = COMMON_HEADER + f'''
    #[ram] static mut SRC_100007: U32 = U32!(100007);

    #[lowram(0x0200)] static mut DV0: U32;
    #[lowram(0x0204)] static mut DV1: U32;
    #[lowram(0x0208)] static mut DV2: U32;
    #[lowram(0x020C)] static mut DV3: U32;
    #[lowram(0x0210)] static mut DV4: U32;

    #[lowram(0x0214)] static mut MD0: U32;
    #[lowram(0x0218)] static mut MD1: U32;
    #[lowram(0x021C)] static mut MD2: U32;

    #[lowram(0x0220)] static mut MU0: U32;
    #[lowram(0x0224)] static mut MU1: U32;
    #[lowram(0x0228)] static mut MU2: U32;
    #[lowram(0x022C)] static mut MU3: U32;
    #[lowram(0x0230)] static mut MU4: U32;

    #[lowram(0x0234)] static mut CO0: U32;

    #[entry]
    fn main() {{
        // === div ===
        V.from_u16(1000); W.from_u16(10); V.div(&W);
        DV0.lo = V.lo; DV0.hi = V.hi;

        V.from_u16(1000); W.from_u16(7); V.div(&W);
        DV1.lo = V.lo; DV1.hi = V.hi;

        V.from_u16(42); W.from_u16(1); V.div(&W);
        DV2.lo = V.lo; DV2.hi = V.hi;

        V.from_u16(0); W.from_u16(5); V.div(&W);
        DV3.lo = V.lo; DV3.hi = V.hi;

        V.from_u16(1000); W.from_u16(0); V.div(&W);
        DV4.lo = V.lo; DV4.hi = V.hi;

        // === mod ===
        V.from_u16(1000); W.from_u16(7); V.mod(&W);
        MD0.lo = V.lo; MD0.hi = V.hi;

        V.from_u16(1000); W.from_u16(10); V.mod(&W);
        MD1.lo = V.lo; MD1.hi = V.hi;

        V.from_u16(3); W.from_u16(5); V.mod(&W);
        MD2.lo = V.lo; MD2.hi = V.hi;

        // === mod_u16 ===
        V.from_u16(1000); V.mod_u16(7);
        MU0.lo = V.lo; MU0.hi = V.hi;

        V.from_u16(1000); V.mod_u16(10);
        MU1.lo = V.lo; MU1.hi = V.hi;

        V.from_u16(3); V.mod_u16(5);
        MU2.lo = V.lo; MU2.hi = V.hi;

        V.lo = SRC_100007.lo; V.hi = SRC_100007.hi;
        V.mod_u16(7);
        MU3.lo = V.lo; MU3.hi = V.hi;

        V.from_u16(1000); V.mod_u16(0);
        MU4.lo = V.lo; MU4.hi = V.hi;

        // === combined: (100*10)/10 = 100 ===
        V.from_u16(100); W.from_u16(10);
        V.mul(&W); V.div(&W);
        CO0.lo = V.lo; CO0.hi = V.hi;
    }}
'''


# ── Test classes ─────────────────────────────────────────────────────────────

class TestU32FastOps1:
    """Batched tests: literals, from_u16, to_u16, copy, add."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(FAST_SOURCE_1)
        return e2e.execute(rom, max_instructions=200000)

    def test_literal_macro(self, cpu):
        """U32! literal initialization macro."""
        assert read_u32(cpu, 0x0200) == u32_bytes(0), "U32!(0)"
        assert read_u32(cpu, 0x0204) == u32_bytes(1000), "U32!(1000)"
        assert read_u32(cpu, 0x0208) == u32_bytes(65535), "U32!(65535)"
        assert read_u32(cpu, 0x020C) == u32_bytes(65536), "U32!(65536)"
        assert read_u32(cpu, 0x0210) == u32_bytes(100000), "U32!(100000)"
        assert read_u32(cpu, 0x0214) == u32_bytes(0xFFFFFFFF), "U32!(0xFFFFFFFF)"

    def test_from_u16(self, cpu):
        """from_u16 zero-extends u16 to U32."""
        assert read_u32(cpu, 0x0218) == u32_bytes(0), "from_u16(0)"
        assert read_u32(cpu, 0x021C) == u32_bytes(1000), "from_u16(1000)"
        assert read_u32(cpu, 0x0220) == u32_bytes(65535), "from_u16(65535)"

    def test_to_u16(self, cpu):
        """to_u16 returns low 16 bits."""
        snes = 0x7E0020
        assert [cpu.memory.read(snes), cpu.memory.read(snes + 1)] == [0xD2, 0x04], \
            "to_u16(1234) = 0x04D2"

    def test_copy(self, cpu):
        """copy transfers both lo and hi words."""
        assert read_u32(cpu, 0x0224) == u32_bytes(42), "copy(42)"

    def test_add(self, cpu):
        """U32 addition with carry propagation."""
        assert read_u32(cpu, 0x0228) == u32_bytes(300), "100+200"
        assert read_u32(cpu, 0x022C) == u32_bytes(65536), "65535+1 (carry)"
        assert read_u32(cpu, 0x0230) == u32_bytes(70000), "40000+30000"
        assert read_u32(cpu, 0x0234) == u32_bytes(1000), "1000+0 (identity)"


class TestU32FastOps2:
    """Batched tests: sub, mul, shl, shr, cmp, combined."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(FAST_SOURCE_2)
        return e2e.execute(rom, max_instructions=500000)

    def test_sub(self, cpu):
        """U32 subtraction with borrow propagation."""
        assert read_u32(cpu, 0x0200) == u32_bytes(200), "300-100"
        assert read_u32(cpu, 0x0204) == u32_bytes(0), "100-100"
        assert read_u32(cpu, 0x0208) == u32_bytes(65535), "65536-1 (borrow)"

    def test_mul(self, cpu):
        """U32 multiplication."""
        assert read_u32(cpu, 0x020C) == u32_bytes(20000), "100*200"
        assert read_u32(cpu, 0x0210) == u32_bytes(0), "12345*0"
        assert read_u32(cpu, 0x0214) == u32_bytes(12345), "12345*1"
        assert read_u32(cpu, 0x0218) == u32_bytes(100000), "1000*100"

    def test_shl(self, cpu):
        """U32 shift left."""
        assert read_u32(cpu, 0x021C) == u32_bytes(16), "1<<4"
        assert read_u32(cpu, 0x0220) == u32_bytes(65536), "1<<16 (cross-word)"
        assert read_u32(cpu, 0x0224) == u32_bytes(42), "42<<0 (no shift)"

    def test_shr(self, cpu):
        """U32 shift right."""
        assert read_u32(cpu, 0x0228) == u32_bytes(256), "1024>>2"
        assert read_u32(cpu, 0x022C) == u32_bytes(0), "1>>1"
        assert read_u32(cpu, 0x0230) == u32_bytes(1), "65536>>16 (cross-word)"

    def test_cmp(self, cpu):
        """U32 comparison returns 0/1/0xFF."""
        assert read_u8(cpu, 0x20) == 0, "100 == 100 -> 0"
        assert read_u8(cpu, 0x21) == 1, "200 > 100 -> 1"
        assert read_u8(cpu, 0x22) == 0xFF, "100 < 200 -> 0xFF"

    def test_combined_add_sub(self, cpu):
        """(100+200)-200 = 100."""
        assert read_u32(cpu, 0x0234) == u32_bytes(100), "(100+200)-200"


class TestU32SlowOps:
    """Batched tests for slow U32 operations (div, mod)."""

    @pytest.fixture(scope="class")
    def cpu(self):
        e2e = E2ETest()
        rom = e2e.compile(SLOW_SOURCE)
        return e2e.execute(rom, max_instructions=2000000)

    def test_div(self, cpu):
        """U32 division."""
        assert read_u32(cpu, 0x0200) == u32_bytes(100), "1000/10"
        assert read_u32(cpu, 0x0204) == u32_bytes(142), "1000/7 (truncation)"
        assert read_u32(cpu, 0x0208) == u32_bytes(42), "42/1"
        assert read_u32(cpu, 0x020C) == u32_bytes(0), "0/5"
        assert read_u32(cpu, 0x0210) == u32_bytes(0xFFFFFFFF), "1000/0 (sentinel)"

    def test_mod(self, cpu):
        """U32 modulo."""
        assert read_u32(cpu, 0x0214) == u32_bytes(6), "1000%7"
        assert read_u32(cpu, 0x0218) == u32_bytes(0), "1000%10"
        assert read_u32(cpu, 0x021C) == u32_bytes(3), "3%5"

    def test_mod_u16(self, cpu):
        """U32 scalar modulo (mod_u16)."""
        assert read_u32(cpu, 0x0220) == u32_bytes(6), "1000%u16(7)"
        assert read_u32(cpu, 0x0224) == u32_bytes(0), "1000%u16(10)"
        assert read_u32(cpu, 0x0228) == u32_bytes(3), "3%u16(5)"
        assert read_u32(cpu, 0x022C) == u32_bytes(5), "100007%u16(7)"
        assert read_u32(cpu, 0x0230) == u32_bytes(1000), "1000%u16(0) unchanged"

    def test_combined_mul_div(self, cpu):
        """(100*10)/10 = 100."""
        assert read_u32(cpu, 0x0234) == u32_bytes(100), "(100*10)/10"
