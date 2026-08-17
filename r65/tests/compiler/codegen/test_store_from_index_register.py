# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Storing an X/Y-resident value to memory must write exactly the destination width.

R65 is unconditionally x16, so `STX`/`STY` store **two** bytes. Nothing checked
the destination width before selecting them, so a 1-byte destination had its
neighbour overwritten:

    #[zeropage(0x10)] static mut OUT: u8;
    #[zeropage(0x11)] static mut GUARD: u8;
    let mut t: u8 = 0;
    while t < 10 { t = t + 1; }   // loop promotion puts the counter in Y
    OUT = t as u8;                // STY $10 -- writes $10 AND $11

A 1-byte destination therefore cannot use `STX`/`STY` at all and has to route
through A. The same routing covers the addressing modes those two mnemonics
lack, including long addressing, which used to be a hard compile error for a
`#[ram]` destination.

`r65/tests/e2e/test_index_store_width.py` runs the same shapes on the emulator.
"""

import pytest
from r65.compiler.main import compile_string


def _get_function_asm(full_asm: str, func_name: str) -> str:
    lines = full_asm.split('\n')
    in_func = False
    func_lines: list[str] = []
    for line in lines:
        if line.strip() == f'{func_name}:':
            in_func = True
        elif in_func:
            if line.startswith('; ---') and func_lines:
                break
            func_lines.append(line)
    return '\n'.join(func_lines)


def _instruction_lines(asm: str) -> list[str]:
    result = []
    for line in asm.split('\n'):
        stripped = line.strip()
        if (not stripped
                or stripped.startswith(';')
                or stripped.startswith('.')
                or stripped.endswith(':')):
            continue
        result.append(stripped.split(';')[0].strip())
    return result


def instrs(source: str, func: str = "main") -> list[str]:
    return _instruction_lines(_get_function_asm(compile_string(source), func))


COUNTER_LOOP = """
#[zeropage(0x10)]
static mut OUT: u8;
#[zeropage(0x11)]
static mut GUARD: u8;
#[entry]
fn main() {
    GUARD = 0xEE;
    let mut t: u8 = 0;
    while t < 10 { t = t + 1; }
    OUT = t as u8;
}
"""


class TestOneByteDestination:
    """The reported bug: a u8 destination must not be written with STY."""

    def test_no_wide_store_to_a_one_byte_destination(self):
        emitted = instrs(COUNTER_LOOP)
        assert not any(i.startswith(('STY', 'STX')) for i in emitted), (
            f"STX/STY writes 2 bytes under x16 and would clobber GUARD: {emitted}")

    def test_routes_through_the_accumulator(self):
        emitted = instrs(COUNTER_LOOP)
        assert 'TYA' in emitted or 'TXA' in emitted, emitted
        assert any(i.startswith('STA') for i in emitted), emitted

    def test_ram_destination_compiles(self):
        """A #[ram] destination has no STY long form, so this used to be a hard
        codegen error rather than a silent 2-byte write."""
        source = COUNTER_LOOP.replace("#[zeropage(0x10)]\nstatic mut OUT: u8;",
                                      "#[ram]\nstatic mut OUT: u8;")
        assert instrs(source)


class TestTwoByteDestinationUnaffected:
    """The anti-pessimization guard. A genuine 2-byte value in an index register
    going to a 2-byte destination must keep storing directly -- routing it
    through A would cost a transfer for nothing.

    Nothing in the repository asserted this before, which is what let the width
    question go unasked in three separate selectors."""

    DIRECT = """
#[zeropage(0x10)]
static mut W: u16;
#[zeropage(0x14)]
static mut V: u16;
far fn storex(p @ X: u16) { W = p; }
far fn storey(p @ Y: u16) { V = p; }
#[entry]
fn main() { storex(0x1234); storey(0x5678); }
"""

    @pytest.mark.parametrize("func,mnemonic", [("storex", "STX"), ("storey", "STY")])
    def test_direct_store_kept(self, func, mnemonic):
        emitted = instrs(self.DIRECT, func)
        assert any(i.startswith(mnemonic) for i in emitted), (
            f"a 2-byte destination should store directly: {emitted}")
        assert not any(i in ('TXA', 'TYA') for i in emitted), (
            f"no transfer needed here: {emitted}")


# The remaining hazard -- `TYA`/`TXA` executed in m16 copying 16 bits, which both
# widens the following store and clobbers B with the index register's high byte
# -- needs an index-resident value with a memory destination while the emitter is
# in m16. That combination resists synthesis (loop promotion declines every shape
# tried here), but occurs in pacman's `draw_lives_icons`, so it is covered by the
# byte-identical gate on that project and by the live-`@ B` case in
# `r65/tests/e2e/test_index_store_width.py`.
