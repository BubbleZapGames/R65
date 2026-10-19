# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Block entry widths chosen from what each block's code wants.

`WidthAssignment` (analysis/width_demand.py) is computed from MIR as codegen
receives it, after loop register promotion, so the tests capture each
function at `generate_function`.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.analysis.width_demand import WidthAssignment
from r65.compiler.codegen import function_gen


@pytest.fixture
def assign(monkeypatch):
    """compile(src, name) -> the WidthAssignment of function *name*."""
    def run(src: str, name: str) -> WidthAssignment:
        found = {}
        orig = function_gen.FunctionCodeGenerator.generate_function

        def capture(self, mir_func, *args, **kwargs):
            if mir_func.name == name:
                found['wa'] = WidthAssignment(mir_func)
            return orig(self, mir_func, *args, **kwargs)

        monkeypatch.setattr(function_gen.FunctionCodeGenerator,
                            'generate_function', capture)
        compile_string(src, 't.r65')
        return found['wa']
    return run


CRC = '''
#[lowram] static mut BUF: [u8; 256];
#[lowram] static mut OUT: u16;

#[inline(never)]
fn t() -> u16 {
    let mut crc: u16 = 0xFFFF;
    let mut i: u16 = 0;
    while i < 256 {
        crc = crc ^ ((BUF[i] as u16) << 8);
        let mut b: u8 = 0;
        while b < 8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc = crc << 1;
            }
            b = b + 1;
        }
        i = i + 1;
    }
    return crc;
}

#[entry]
fn main() { OUT = t(); }
'''


def test_inner_loop_stays_in_the_width_its_work_wants(assign):
    # The bit loop is all u16 work, and its counter's step and test run in
    # Y: no block of it is entered in m8, so nothing in it switches.
    wa = assign(CRC, 't')
    inner = [bid for bid, w in wa.weight.items() if w == 100]
    assert inner
    assert all(wa.entry[bid] == 16 for bid in inner)
    assert all(wa.exit_width(bid) == 16 for bid in inner)


def test_index_only_blocks_take_no_width(assign):
    # The latches step and test Y and X only.
    wa = assign(CRC, 't')
    assert sum(1 for first, exit_width in wa.summary.values()
               if first is None and exit_width is None) == 2


def test_m8_function_keeps_m8(assign):
    wa = assign('''
        #[zeropage] static mut G: u8;
        #[lowram] static mut OUT: u8;
        #[inline(never)]
        fn t() -> u8 {
            let mut i: u16 = 0;
            while i < 8 {
                G = G + 2;
                i = i + 1;
            }
            return G;
        }
        #[entry]
        fn main() { OUT = t(); }
    ''', 't')
    assert set(wa.entry.values()) == {8}
    assert wa.cost() == 0


def test_bit_loop_compiles_without_a_mode_switch():
    # Entered in the assigned m16, the loop switches nowhere: the m8 the
    # outer loop's byte load wants is set on its own edge.
    asm = compile_string(CRC, 't.r65')
    body = asm[asm.index('\nt:'):asm.index('\nmain:')]
    lines = [l.split(';')[0].strip() for l in body.splitlines()]
    loop = lines[lines.index('LDY #$08') + 1:
                 next(k for k, l in enumerate(lines) if l.startswith('BNE '))]
    assert loop
    assert not [l for l in loop if l.startswith(('SEP', 'REP'))]
