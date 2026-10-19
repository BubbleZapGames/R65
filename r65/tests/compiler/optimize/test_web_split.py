# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A reassigned local gets a vreg per def-use web (optimize/web_split.py).

Each case counts the vregs a named local is spread over once
`WebSplitter` has run. Values: tests/e2e/test_web_split.py.
"""

from r65.compiler.main import compile_string
from r65.compiler.mir.nodes import VirtualRegister, iter_operands
from r65.compiler.optimize.web_split import WebSplitter

GLOBALS = """
#[lowram] static mut FLAGS: [u8; 64];
#[zeropage] static mut W: u16;
#[zeropage] static mut O: u16;
#[inline(never)] {fn}
#[entry] fn main() {{ O = t(W); }}
"""


def _vregs(monkeypatch, fn: str, name: str) -> int:
    seen = {}
    run = WebSplitter.optimize

    def spy(self, program):
        count = run(self, program)
        func = next(f for f in program.functions if f.name == 't')
        seen['ids'] = {v.id for b in func.blocks.values() for ins in b.instructions
                       for _, v in iter_operands(ins)
                       if type(v) is VirtualRegister and v.hint == name}
        return count

    monkeypatch.setattr(WebSplitter, 'optimize', spy)
    compile_string(GLOBALS.format(fn=fn), 't.r65')
    return len(seen['ids'])


def test_counter_reused_by_a_second_loop_splits(monkeypatch):
    assert _vregs(monkeypatch, '''fn t(n: u16) -> u16 {
        let mut i: u16 = 0;
        while i < 64 { FLAGS[i] = 0; i = i + 1; }
        i = n;
        while i < 64 { FLAGS[i] = 1; i = i + 2; }
        return 0;
    }''', 'i') == 2


def test_value_carried_around_a_loop_stays_one(monkeypatch):
    assert _vregs(monkeypatch, '''fn t(n: u16) -> u16 {
        let mut s: u16 = n;
        let mut i: u16 = 0;
        while i < 64 { s = s + (FLAGS[i] as u16); i = i + 1; }
        return s;
    }''', 's') == 1


def test_value_set_in_either_arm_stays_one(monkeypatch):
    assert _vregs(monkeypatch, '''fn t(n: u16) -> u16 {
        let mut s: u16 = 0;
        if n > 3 { s = W; } else { s = n; }
        return s;
    }''', 's') == 1


def test_parameter_keeps_its_vreg(monkeypatch):
    assert _vregs(monkeypatch, '''fn t(n: u16) -> u16 {
        O = n;
        n = W;
        return n;
    }''', 'param_n') == 1
