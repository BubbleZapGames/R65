# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A `let x @ A/B/X/Y` alias whose register is overwritten before its last use is
a compile error (mir/register_alias_check.py).

An alias names the register; nothing preserves it. These all compiled
silently to the wrong value before: `case_a` returned `~PPR` instead of
`~PPR & cur`, the X alias stored the array index J, the Y alias the pointer
index - the register held whatever overwrote it.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import MIRLoweringError

PRELUDE = """
#[hw(0x4218)] static mut JOY1L: u8;
#[hw(0x4219)] static mut JOY1H: u8;
#[lowram(0x0200)] static mut ARR: [u8; 16];
#[lowram(0x0220)] static mut J: u16;
#[lowram(0x0222)] static mut OUT: u16;
#[lowram(0x0224)] static mut V: u8;
#[lowram(0x0226)] static mut PST: u16;
#[lowram(0x0228)] static mut PPR: u16;
#[lowram(0x022A)] static mut FLAG: u8;
#[zeropage(0x30)] static mut P: *u8;

#[inline(never)]
fn g() { V = 1; }
"""


def _compile(body: str):
    return compile_string(PRELUDE + body + """
#[entry]
fn main() -> ! { f(); g(); loop {} }
""", cfg_options=['snes'])


def _rejects(body: str, name: str, reg: str):
    with pytest.raises(MIRLoweringError, match=f"'{name}' is bound to {reg}") as e:
        _compile(body)
    return e.value


# --- rejected -------------------------------------------------------------

def test_a_alias_overwritten_by_a_later_expression():
    """The memory note's case_a: `~PPR` is evaluated in A over `cur`."""
    _rejects("""
fn f() -> u16 { let cur @ A = PST; let trig: u16 = (~PPR) & cur; return trig; }
""", 'cur', 'A')


def test_a_second_alias_on_a_overwrites_the_first():
    _rejects("""
fn f() -> u16 { let x @ A = PST; let y @ A = PPR; return x + y; }
""", 'x', 'A')


def test_the_pad_idiom_loses_its_low_byte():
    """`lo @ A` then a load of the high byte: the load is through A."""
    _rejects("""
fn f() { let lo @ A = JOY1L; let hi: u8 = JOY1H;
         OUT = ((hi as u16) << 8) | (lo as u16); }
""", 'lo', 'A')


def test_x_alias_overwritten_by_an_array_index():
    _rejects("fn f() { let i @ X = 3; V = ARR[J]; OUT = i; }", 'i', 'X')


def test_x_alias_overwritten_by_a_call():
    _rejects("fn f() { let i @ X = 3; g(); OUT = i; }", 'i', 'X')


def test_y_alias_overwritten_by_a_pointer_index():
    _rejects("fn f() { let k @ Y = 7; V = P[J]; OUT = k; }", 'k', 'Y')


@pytest.mark.parametrize("between", ["OUT = J + 1;", "OUT = V as u16;"])
def test_b_alias_overwritten_by_16_bit_accumulator_work(between):
    """B is A's high byte: u16 arithmetic or a widening cast in A writes it."""
    _rejects(f"fn f() {{ let h @ B = 0x12; {between} FLAG = h; }}", 'h', 'B')


def test_a_clobber_on_one_branch_reaches_a_use_after_the_join():
    _rejects("""
fn f() { let i @ X = 3; if FLAG != 0 { V = ARR[J]; } OUT = i; }
""", 'i', 'X')


def test_a_clobber_late_in_a_loop_reaches_a_use_on_the_next_pass():
    _rejects("""
fn f() { let i @ X = 3; while FLAG != 0 { OUT = i; V = ARR[J]; } }
""", 'i', 'X')


def test_the_error_names_the_overwriting_line():
    err = _rejects("""
fn f() {
    let i @ X = 3;
    V = ARR[J];
    OUT = i;
}
""", 'i', 'X')
    line = PRELUDE.count('\n') + 4      # `V = ARR[J];`
    assert err.source_loc.line == line


# --- accepted -------------------------------------------------------------

def test_an_alias_updated_through_itself_is_fine():
    _compile("fn f() { let mut acc @ A = 10; acc = acc + 5; V = acc; }")


def test_a_u16_alias_for_m16_work_is_fine():
    _compile("fn f() { let w @ A: u16 = 0x1234; OUT = A; }")


def test_an_x_alias_used_as_the_index_is_fine():
    _compile("fn f() { let i @ X = 3; V = ARR[i]; OUT = i; }")


def test_an_alias_whose_last_use_precedes_the_clobber_is_fine():
    _compile("fn f() { let i @ X = 3; OUT = i; V = ARR[J]; }")


def test_incrementing_an_x_alias_is_fine():
    _compile("fn f() { let mut i @ X = 3; i++; OUT = i; }")


def test_a_b_alias_across_8_bit_accumulator_work_is_fine():
    _compile("fn f() { let h @ B = 0x12; V = V + 1; FLAG = h; }")


def test_reassigning_an_alias_after_the_clobber_is_fine():
    _compile("fn f() { let mut i @ X = 3; V = ARR[J]; i = 4; OUT = i; }")


# --- loop promotion -------------------------------------------------------

def test_no_loop_counter_is_promoted_over_a_y_alias():
    """The outer counter used to be promoted into Y right over the alias
    (`LDY #$03` / `LDY #$00`), and `OUT = i` stored the counter's final value;
    MIR shows no write to Y for the check to see, so loop promotion keeps its
    counters off a register an alias names (_enforce_alias_register_safety)."""
    asm = _compile("""
fn f() { let i @ Y = 3; for k in 0..4 { for m in 0..4 { V = V + 1; } } OUT = i; }
""")
    body = asm.split('\nf:\n', 1)[1].split('RTS', 1)[0]
    writes = [l.split(';')[0].strip() for l in body.split('\n')
              if l.strip().split(' ')[0] in ('LDY', 'INY', 'DEY', 'TAY', 'TXY', 'PLY')]
    assert writes == ['LDY #$03']
