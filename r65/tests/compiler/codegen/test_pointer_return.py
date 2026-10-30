# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A function returning a near pointer dropped the pointer's high byte.

Two independent halves, both of which had to be wrong for the value to
survive a round trip:

1. `infer_exit_mode` (hir/builder.py) asked whether the return type was
   *named* `u16`/`i16`. A `*T` is two bytes and rides in A like any other
   16-bit value, but answered "m8", so the callee's return path did
   `SEP #$20` and then moved a single byte — the high half never left the
   function. The same test governs the caller through
   `Call.callee_exit_m_mode`, so the caller then read one byte back.

2. The pointer spill in `memory_select._spill_pointer_to_scratch` emitted
   `STA dp` in whatever mode was current. X and Y are 16-bit unconditionally
   (x16), so STX/STY were fine; A in m8 wrote the low byte and left the
   scratch's high byte stale, which turns the pointer into a wild address.
   A pointer returned from a call arrives in A with the caller back in m8,
   so this was the common path.

Symptom in rotor_raiders: `UNIT_BASES[k].base()` returned a pointer whose
high byte was garbage, and the field reads through it hit zero page.
"""

import re

from r65.compiler.main import compile_string


SRC = """
#[zeropage(0x16, register)] static mut SCRATCH0: u16;
#[zeropage(0x18, register)] static mut SCRATCH1: u16;

struct Base { speed: u16, hp: u8, pad: u8 }
static B1: Base = Base { speed: 0x1234, hp: 5, pad: 0 };
static B2: Base = Base { speed: 0x5678, hp: 9, pad: 0 };

#[zeropage(0x40)] static mut o: u16;

fn pick(k: u16) -> *Base { if k == 0 { return &B1 as *Base; } return &B2 as *Base; }
fn use_it(k: u16) { let b: *Base = pick(k); o = b.speed; }
#[entry]
fn main() { use_it(1); }
"""


def _fn(asm: str, label: str) -> str:
    return asm[asm.index(f"\n{label}:"):].split("; ---")[0]


def test_callee_keeps_the_pointer_16_bit():
    """The epilogue may drop to m8 to pop the frame — that is fine, because B
    holds the high byte across `PLA` — but the value must never be *materialised*
    8-bit, which is what an m8 `LDA` of the spilled address did."""
    code = _fn(compile_string(SRC, "test.r65"), "pick")
    body = code.split("epilogue")[0]
    assert "SEP #$20" not in body, code


def test_caller_spills_both_pointer_bytes():
    code = _fn(compile_string(SRC, "test.r65"), "use_it")
    spill = code[:code.index("Spill A pointer to scratch")]
    # The store must run in m16, or STA writes one byte. `pick` returns in
    # m16, so the call itself may be what widened A; `.ACCU` follows the
    # mode dataflow either way.
    widths = re.findall(r"\.ACCU (8|16)", spill)
    assert widths and widths[-1] == "16", code


def test_indirect_read_follows_the_spill():
    code = _fn(compile_string(SRC, "test.r65"), "use_it")
    assert "LDA ($16),Y" in code or "LDA ($18),Y" in code, code
