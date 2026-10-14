#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for #[snesrom(..., fastrom)] code placement.

MEMSEL ($420D) only speeds up banks $80-$FF, so `fastrom` has to do more than
set the ROM header's speed bit: code must be assembled with `.BASE $80` (LoROM)
so its bank bytes land in the fast mirror, and the reset/interrupt vectors --
which the CPU always fetches with PBR=$00 -- must JML into that mirror.
"""

import pytest

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder
from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.errors import CodegenError
from r65.compiler.main import compile_string
from r65.compiler.codegen.abi_model import ABI_FIXED_STACK


def compile_source(source: str) -> str:
    """Compile R65 source to WLA-DX assembly text."""
    ast = parse(source)
    hir_program = HIRBuilder().build_program(ast)
    TypeChecker(hir_program).check()
    mir_program = MIRBuilder().build_program(hir_program)
    return ProgramCodeGenerator().generate(mir_program)


def program(header: str, extra: str = "", body: str = "COUNTER++;") -> str:
    """Build a minimal SNES program.

    `extra` goes *after* main because `#[bank(n)]` sets the bank context for
    every declaration that follows it - putting it first would drag main out
    of bank 0.
    """
    return f"""
{header}

#[zeropage]
static mut COUNTER: u8;

#[interrupt(nmi)]
fn nmi_handler() {{
    COUNTER++;
}}

#[entry]
fn main() -> ! {{
    loop {{
        {body}
    }}
}}

{extra}
"""


LOROM_FAST = '#[snesrom(name = "FAST", lorom, fastrom)]'
LOROM_SLOW = '#[snesrom(name = "SLOW", lorom)]'
HIROM_FAST = '#[snesrom(name = "HIFAST", hirom, fastrom)]'
HIROM_SLOW = '#[snesrom(name = "HI", hirom)]'


class TestLoRomFastRom:
    """LoROM + fastrom relocates code into the $80-$BF mirror."""

    def test_base_80_before_banks(self):
        asm = compile_source(program(LOROM_FAST))
        assert ".BASE $80" in asm
        # and it precedes the bank it applies to
        assert asm.index(".BASE $80") < asm.index(".BANK 0 SLOT 0")

    def test_memsel_written_at_reset(self):
        asm = compile_source(program(LOROM_FAST))
        assert "LDA #$01" in asm
        assert "STA $420D" in asm

    def test_reset_trampoline_jmls_into_mirror(self):
        asm = compile_source(program(LOROM_FAST))
        assert "__fast_reset:" in asm
        assert "JML main" in asm

    def test_interrupt_trampoline(self):
        asm = compile_source(program(LOROM_FAST))
        assert "__fast_nmi:" in asm
        assert "JML nmi_handler" in asm

    def test_vectors_point_at_trampolines(self):
        asm = compile_source(program(LOROM_FAST))
        assert "RESET __fast_reset" in asm
        assert "NMI __fast_nmi" in asm
        # the raw handler must NOT be the vector target - it runs in bank $80
        assert "RESET main" not in asm

    def test_every_handled_vector_gets_a_trampoline(self):
        """BRK and COP ride the same stubs as NMI, each to its own handler."""
        asm = compile_source(program(LOROM_FAST, extra="""
#[interrupt(brk)]
fn brk_handler() { COUNTER = 1; }

#[interrupt(cop)]
fn cop_handler() { COUNTER = 2; }
"""))
        assert "JML brk_handler" in asm and "JML cop_handler" in asm
        assert "BRK __fast_brk" in asm
        assert "COP __fast_cop" in asm
        assert "ABORT __empty_handler" in asm

    def test_entry_sets_dbr_to_code_bank(self):
        """PHK/PLB so absolute ROM reads use the fast mirror too."""
        asm = compile_source(program(LOROM_FAST))
        assert "PHK" in asm
        assert "PLB" in asm

    def test_header_declares_fastrom(self):
        asm = compile_source(program(LOROM_FAST))
        assert "FASTROM" in asm
        assert "SLOWROM" not in asm


class TestSlowRomUnchanged:
    """Without the flag, none of the FastROM machinery appears."""

    def test_no_base_directive(self):
        asm = compile_source(program(LOROM_SLOW))
        assert ".BASE $80" not in asm
        assert ".BASE $C0" not in asm

    def test_no_memsel_write(self):
        asm = compile_source(program(LOROM_SLOW))
        assert "$420D" not in asm

    def test_no_trampolines(self):
        asm = compile_source(program(LOROM_SLOW))
        assert "__fast_reset" not in asm
        assert "__fast_nmi" not in asm

    def test_vectors_point_directly_at_handlers(self):
        asm = compile_source(program(LOROM_SLOW))
        assert "RESET main" in asm
        assert "NMI nmi_handler" in asm

    def test_entry_does_not_touch_dbr(self):
        asm = compile_source(program(LOROM_SLOW))
        assert "PHK" not in asm
        assert "PLB" not in asm

    def test_header_declares_slowrom(self):
        asm = compile_source(program(LOROM_SLOW))
        assert "SLOWROM" in asm


class TestHiRomFastRom:
    """HiROM already assembles at $C0+, which is FastROM-capable."""

    def test_keeps_c0_base(self):
        asm = compile_source(program(HIROM_FAST))
        assert ".BASE $C0" in asm
        assert ".BASE $80" not in asm

    def test_cop_vector_gets_a_trampoline(self):
        """Every handled vector needs one: a bank-$00 vector can't reach $C0."""
        asm = compile_source(program(HIROM_FAST, extra="""
#[interrupt(cop)]
fn cop_handler() { COUNTER = 2; }
"""))
        assert "__hirom_cop:" in asm and "JML cop_handler" in asm
        assert "COP __hirom_cop" in asm

    def test_uses_hirom_trampolines_with_memsel(self):
        asm = compile_source(program(HIROM_FAST))
        assert "__hirom_reset:" in asm
        assert "STA $420D" in asm
        assert "__fast_reset" not in asm


class TestBankByteOffsets:
    """A WLA bank *index* is not a 65816 bank *byte* once .BASE moves."""

    # The function must touch DBR-dependent memory (a #[hw] register here),
    # or _analyze_dbr_inline_needed elides the DBR setup entirely.
    SRC_EXTRA = """
#[hw(0x2100)]
static mut INIDISP: u8;

#[bank(1)]
#[mode(databank=inline)]
far fn in_bank_one() {
    INIDISP = 0x0F;
}
"""

    CALL = "in_bank_one();"

    def test_inline_databank_adds_rom_base(self):
        asm = compile_source(program(LOROM_FAST, self.SRC_EXTRA, self.CALL))
        assert "LDA #$81" in asm

    def test_inline_databank_unshifted_when_slow(self):
        asm = compile_source(program(LOROM_SLOW, self.SRC_EXTRA, self.CALL))
        assert "LDA #$81" not in asm
        assert "LDA #$01" in asm


class TestMirrorBounds:
    """The $80-$BF LoROM mirror only covers the first 4MB."""

    def test_bank_past_mirror_is_rejected(self):
        extra = """
#[bank(128)]
#[inline(never)]
far fn way_out_there() {
    COUNTER++;
}
"""
        with pytest.raises(CodegenError, match="fastrom"):
            compile_source(program(LOROM_FAST, extra, "way_out_there();"))


@pytest.mark.parametrize("header", [LOROM_FAST, HIROM_FAST])
def test_dbg_function_symbols_get_their_segment(header, tmp_path):
    """Segments carry the CPU bank ($80/$C0 + n); a function in WLA bank 0
    must still find its segment in the .dbg."""
    ast = parse(program(header))
    hir_program = HIRBuilder().build_program(ast)
    TypeChecker(hir_program).check()
    mir_program = MIRBuilder().build_program(hir_program)
    gen = ProgramCodeGenerator()
    gen.generate(mir_program, debug=True)
    gen.write_outputs(str(tmp_path / "game.asm"), debug=True)
    dbg = (tmp_path / "game.dbg").read_text()
    main_sym = next(l for l in dbg.splitlines()
                    if l.startswith("sym") and 'name="main"' in l)
    assert "seg=0" in main_sym


class TestDataBank:
    """DBR outside a far-pointer bracket is the data bank ($00, or $80 under
    FastROM), not PBR: HiROM code runs in $C0-$FF, which maps no I/O or low
    RAM, and LoROM code past bank 0 sees its own bank's ROM at $8000."""

    # A far-self trait method that writes a hardware register needs DBR set
    # for its body.
    FAR_SELF = """
#[zeropage(0x10, register)] static mut SCRATCH0: u16;
#[zeropage(0x12, register)] static mut SCRATCH1: u16;
#[zeropage(0x14, register)] static mut SCRATCH2: u16;
#[zeropage(0x16, register)] static mut SCRATCH3: u16;
#[hw(0x4202)] static mut WRMPYA: u8;
struct P { x: u8 }
trait T { far fn put(far *self); }
impl T for P { far fn put(far *self) { WRMPYA = self.x; } }
#[ram] static mut OBJ: P = P { x: 1 };
#[inline(never)]
fn go(p: far *dyn T) { p.put(); }
#[entry] fn main() { go(&OBJ); }
"""

    @staticmethod
    def _body(asm: str, label: str) -> str:
        return asm.split(f"\n{label}:", 1)[1].split("RTL", 1)[0]

    @pytest.mark.parametrize("abi", [None, ABI_FIXED_STACK],
                             ids=["d_equals_s", "fixedstack_scratch"])
    def test_hirom_far_self_body_gets_data_bank(self, abi):
        asm = compile_string(HIROM_SLOW + self.FAR_SELF, abi_model=abi)
        body = self._body(asm, "P__put")
        assert ("Push self address" in body) == (abi is None)
        assert "PHK" not in body
        assert "PEA $00\n" in body

    def test_far_self_body_reading_lowram_gets_data_bank(self):
        """A folded low-RAM operand (`ADC $0200`) is absolute too: through
        self's bank it would miss when self lives in ROM or $7F."""
        asm = compile_string("""
struct P { n: u8 }
trait Bump { fn bump(far *self); }
impl Bump for P { fn bump(far *self) { self.n = self.n + LR; } }
#[lowram] static mut LR: u8;
#[ram] static mut PP: P;
#[entry] fn main() -> ! { LR = 2; PP.n = 1; let b: far *dyn Bump = &PP; b.bump(); loop {} }
""")
        body = self._body(asm, "P__bump")
        assert "Set DBR to the data bank" in body

    def test_lorom_bank0_far_self_keeps_phk(self):
        """In LoROM bank 0, PBR is the data bank: PHK/PLB is the cheap load."""
        asm = compile_string(LOROM_FAST + self.FAR_SELF)
        body = self._body(asm, "P__put")
        assert "PHK" in body and "PEA" not in body

    def test_lorom_call_from_bank1_restores_data_bank(self):
        """A SET_DBR function in bank 1 hands its callee DBR=$00, not $01,
        so a near pointer to bank-0 ROM still reads bank 0."""
        asm = compile_string("""
#[lowram] static mut RESULT: u8;
#[zeropage] static mut Z0: u8;
#[zeropage] static mut Z1: u8;
#[zeropage] static mut Z2: u8;
#[zeropage] static mut Z3: u8;
static ARR: [u8; 4] = [5, 6, 7, 8];
#[ram] static mut BUF: [u8; 4] = [1, 2, 3, 4];
#[inline(never)]
far fn rdn(q: *u8) { RESULT = *q; }
#[entry] fn main() { f(&BUF[0]); }
#[bank(1)]
#[inline(never)]
far fn f(p: far *u8) {
    Z0 = p[0]; Z1 = p[1]; Z2 = Z0 + Z1; Z3 = Z2 + Z0; Z0 = Z3 + Z1; Z1 = Z0 + Z2;
    Z2 = Z1 + Z3; Z3 = Z2 + Z1; Z0 = Z0 + Z3; Z1 = Z1 + Z0; Z2 = Z2 + Z1;
    rdn(&ARR[1]);
    Z3 = Z3 + Z2;
}
""", cfg_options=["snes"])
        call = asm.split("Save ptr-bank DBR before call", 1)[1].split("JSL", 1)[0]
        assert "PHK" not in call
        assert "PEA $00" in call

    def test_hirom_fastrom_entry_sets_data_bank(self):
        asm = compile_source(program(HIROM_FAST))
        main = asm.split("\nmain:", 1)[1]
        assert "PEA $8080" in main
        assert "PHK" not in main
