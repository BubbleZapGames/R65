# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A narrowing cast should keep its result in A instead of spilling.

`slot_allocator.py` listed `TypeConvert` in `_ALU_TYPES` — "this op routes
through A and clobbers it" — but in neither frozenset that makes a vreg
eligible to *live* in A. So a conversion result could never be coalesced
and always got a stack slot, which meant every narrowing cast allocated a
frame and spilled twice:

    b_cast:
        PHX / PHY      ; frame, needed only for the spills
        REP #$21
        ADC #$01
        STA $01,S      ; spill of the u16 operand
        SEP #$20
        STA $03,S      ; spill of the u8 result
        ...

Codegen was already ready for it: `_emit_narrowing_conversion` has an
A-destination path that emits a bare `SEP #$20` for `A = A as u8`.

An unsigned widening joined later, once `_emit_widening_conversion` had an
A-destination path (`REP #$20 / AND #$00FF`). A signed one stays excluded:
its sign extension branches and writes the high byte through memory.

A knock-on effect worth pinning: with the spill gone, `AND #$00FF` and
`SEP #$20` become adjacent, so the existing
`_eliminate_redundant_and_before_sep` peephole finally removes the mask
from `(x >> 8) as u8` with no new code.
"""

import re

from r65.compiler.main import compile_string


def _func_asm(source: str, name: str) -> str:
    asm = compile_string(source, cfg_options=["snes"])
    return asm.split(f"\n{name}:", 1)[1].split("RTL", 1)[0].split("RTS", 1)[0]


# Real functions, not inlined, so their own bodies can be inspected.
SOURCE = """
#[zeropage(0x10)] static mut R: u8;
#[zeropage(0x12)] static mut W: u16;

#[inline(never)] far fn nocast(v @ A: u16) -> u16 { return v + 1; }
#[inline(never)] far fn narrowed(v @ A: u16) -> u8 { return (v + 1) as u8; }
#[inline(never)] far fn shifted(v @ A: u16) -> u8 { return ((v + 1) >> 8) as u8; }
#[inline(never)] far fn widened(v @ A: u8) -> u16 { return v as u16; }
#[inline(never)] far fn signed(v @ A: i8) -> i16 { return v as i16; }

#[entry] fn main() {
    W = nocast(1); R = narrowed(2); R = shifted(3); W = widened(4);
    W = nocast(5); R = narrowed(6); R = shifted(7); W = widened(8);
    W = signed(9) as u16; W = signed(10) as u16;
}
"""


def test_narrowing_cast_needs_no_frame():
    """The cast result lives in A, so there is nothing to allocate a frame for."""
    asm = _func_asm(SOURCE, "narrowed")
    assert "Allocate frame" not in asm
    assert not re.search(r"^\s*STA \$[0-9A-F]+,S", asm, re.M)


def test_narrowing_cast_is_just_a_mode_switch():
    """`A = A as u8` truncates by switching width — no instruction of its own.

    (`v @ A: u16` enters in m16, so `v + 1` is a bare `INC A`.)"""
    asm = _func_asm(SOURCE, "narrowed")
    ops = re.findall(r"^\s*([A-Z]{3})\b", asm, re.M)
    assert ops == ["INC", "SEP"]


def test_shift_then_narrow_drops_the_mask():
    """With the spill gone, the existing AND-before-SEP peephole can fire."""
    asm = _func_asm(SOURCE, "shifted")
    ops = re.findall(r"^\s*([A-Z]{3})\b", asm, re.M)
    assert ops == ["INC", "XBA", "SEP"]
    assert "AND" not in asm


def test_unrelated_arithmetic_is_unaffected():
    """The no-cast baseline this was measured against must not move."""
    asm = _func_asm(SOURCE, "nocast")
    ops = re.findall(r"^\s*([A-Z]{3})\b", asm, re.M)
    assert ops == ["INC"]


def test_unsigned_widening_is_a_mask_in_a():
    """`v as u16` with v in A widens in place and returns in A."""
    asm = _func_asm(SOURCE, "widened")
    ops = re.findall(r"^\s*([A-Z]{3})\b", asm, re.M)
    assert ops == ["REP", "AND"]


def test_signed_widening_still_stores_to_its_destination():
    """Sign extension has no A-destination path; it must keep spilling."""
    asm = _func_asm(SOURCE, "signed")
    assert re.search(r"^\s*STA \$[0-9A-F]+,S", asm, re.M)


def test_a_computed_u8_widens_into_x_without_a_spill():
    """`K + 1` into an `@X: u16` parameter, implicit or spelled `as u16`."""
    asm = compile_string("""
        #[zeropage(0x10)] static mut K: u8;
        #[zeropage(0x12)] static mut O: u8;
        #[inline(never)]
        fn md(v @ A: u8, d @ X: u16) -> u8 { return v; }
        #[entry] fn main() { O = md(3, K + 1); O = md(3, (K + 1) as u16); }
    """, cfg_options=["snes"])
    main = asm.split("\nmain:", 1)[1].split("__SCMP", 1)[0]
    ops = re.findall(r"^\s*([A-Z]{3})\b", main, re.M)
    assert ops.count("TAX") == 2
    assert "Allocate frame" not in main
    assert not re.search(r",S", main)


def test_a_computed_u8_widens_into_memory_with_one_store():
    asm = compile_string("""
        #[zeropage(0x10)] static mut K: u8;
        #[zeropage(0x12)] static mut W: u16;
        #[entry] fn main() { W = K + 1; }
    """, cfg_options=["snes"])
    main = asm.split("\nmain:", 1)[1].split("__SCMP", 1)[0]
    assert re.search(r"AND #\$FF.*\n\s*STA \$12\b", main)
    assert not re.search(r",S", main)


def test_source_in_a_with_uncoalesced_destination():
    """The cast may consume an A-resident value even when its own result
    cannot stay in A. The safe-use rule permits that pairing, so pin it:
    the conversion still emits nothing, and both consumers read A.
    """
    asm = _func_asm("""
        #[zeropage(0x10)] static mut A1: u8;
        #[zeropage(0x11)] static mut B1: u8;
        far fn f(v @ A: u16) -> u8 { let b: u8 = (v + 1) as u8; A1 = b; return b + 1; }
        #[entry] fn main() { B1 = f(0x1122); B1 = f(0x3344); }
    """, "f")
    ops = re.findall(r"^\s*([A-Z]{3})\b", asm, re.M)
    assert ops == ["INC", "SEP", "STA", "INC"]
    assert not re.search(r"^\s*STA \$[0-9A-F]+,S", asm, re.M)
