# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""W006: a static at a fixed address inside the stack region
(codegen/memory_alloc.py) - the declared one, or the default $0100-$01FF."""

import pytest

from r65.compiler.codegen.codegen import ProgramCodeGenerator
from r65.compiler.main import run_frontend
from r65.compiler.mir.builder import MIRBuilder


def _warnings(decls: str):
    src = decls + '\n#[entry]\nfn main() -> ! { loop {} }\n'
    gen = ProgramCodeGenerator()
    gen.generate(MIRBuilder().build_program(run_frontend(src, "t.r65").hir_program))
    return [w for w in gen.warnings if "W006" in w]


STACK = '#[stack(0x1F00, 0x1FFF)]\n'


def test_lowram_in_declared_stack():
    (w,) = _warnings(STACK + '#[lowram(0x1FF0)]\nstatic mut BUF: [u8; 4];\n')
    assert "t.r65:2:" in w and "'BUF' at $1FF0-$1FF3" in w and "$1F00-$1FFF" in w


@pytest.mark.parametrize("decl, warns", [
    ('#[zeropage(0x10)]\nstatic mut Z: u8;', False),
    ('#[lowram(0x0180)]\nstatic mut L: u8;', True),        # default $0100-$01FF
])
def test_default_stack(decl, warns):
    assert bool(_warnings(decl + '\n')) == warns


@pytest.mark.parametrize("addr, warns", [
    ("0x1F80", True), ("0x7E1F80", True), ("0x801F80", True),
    ("0x2100", False), ("0x7E2000", False), ("0x7F1F80", False),
])
def test_hw_in_low_ram_or_a_mirror(addr, warns):
    assert bool(_warnings(STACK + f'#[hw({addr})]\nstatic mut R: u8;\n')) == warns
