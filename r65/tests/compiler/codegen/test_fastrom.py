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
far fn way_out_there() {
    COUNTER++;
}
"""
        with pytest.raises(CodegenError, match="fastrom"):
            compile_source(program(LOROM_FAST, extra, "way_out_there();"))


class TestBankZeroAlwaysExists:
    """The vector table needs bank-0 labels to exist.

    A program whose functions all sit in #[bank(n>0)] used to emit no bank 0
    at all, leaving `__empty_handler` referenced but undefined - wlalink
    rejects that with "Reference to an unknown label". The fastrom
    trampolines live in the same window and have the same requirement.
    """

    # #[bank(1)] sets the bank context for every declaration that follows,
    # so this drags main out of bank 0 too.
    SRC = """
#[snesrom(name = "NOBANK0"%s)]

#[zeropage]
static mut COUNTER: u8;

#[bank(1)]
#[interrupt(nmi)]
fn nmi_handler() {
    COUNTER++;
}

#[entry]
fn main() -> ! {
    loop {
        COUNTER++;
    }
}
"""

    def test_empty_handler_is_defined(self):
        asm = compile_source(self.SRC % "")
        assert "__empty_handler" in asm, "vector table references it"
        assert "__empty_handler:" in asm, "but nothing defines it"

    def test_fast_trampolines_are_defined(self):
        asm = compile_source(self.SRC % ", lorom, fastrom")
        assert "RESET __fast_reset" in asm
        assert "__fast_reset:" in asm
        assert "__fast_nmi:" in asm
