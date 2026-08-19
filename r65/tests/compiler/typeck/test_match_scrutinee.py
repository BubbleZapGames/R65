# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A match scrutinee has to be something patterns can compare against.

Root cause (fixed): `_check_pattern_type` decided whether a literal or range
pattern could apply by reading `scrutinee_type.name`. Only scalars have one, so
a pointer, struct, or array scrutinee raised `AttributeError` out of the type
checker — a compiler crash reading

    Compilation error: 'PointerTypeInfo' object has no attribute 'name'

rather than any diagnostic. A bare pointer could always reach it; newtypes over
near pointers (`struct Handle(*Sprite);`) added a second route, which is how it
surfaced.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import TypeCheckError


def compile_source(source: str):
    return compile_string(source, "test.r65")


HW = '#[hw(0x2100)]\nstatic mut INIDISP: u8;\n'
MATCH = 'match {scrutinee} {{ 0 => {{ INIDISP = 1; }} _ => {{ }} }};'


def program(decls: str, scrutinee: str) -> str:
    body = MATCH.format(scrutinee=scrutinee)
    return f"{HW}{decls}#[entry]\nfn main() -> ! {{ {body} loop {{ }} }}"


class TestNonScalarScrutineeIsRejected:
    """Each of these used to crash; each should now name the type."""

    def test_bare_pointer(self):
        src = program('#[zeropage(0x10)]\nstatic mut P: *u8;\n', 'P')
        with pytest.raises(TypeCheckError, match=r"cannot match on \*u8"):
            compile_source(src)

    def test_newtype_over_a_pointer(self):
        src = program(
            'struct Spr { x: u8, y: u8 }\nstruct H(*Spr);\n'
            '#[zeropage(0x10)]\nstatic mut HH: H;\n', 'HH')
        with pytest.raises(TypeCheckError, match="cannot match on H"):
            compile_source(src)

    def test_struct(self):
        src = program('struct P { x: u8 }\n#[ram]\nstatic mut PP: P;\n', 'PP')
        with pytest.raises(TypeCheckError, match="cannot match on P"):
            compile_source(src)

    def test_hint_offers_a_way_forward(self):
        src = program('#[zeropage(0x10)]\nstatic mut P: *u8;\n', 'P')
        with pytest.raises(TypeCheckError) as exc:
            compile_source(src)
        assert "'=='" in (exc.value.hint or ""), exc.value.hint


class TestScalarScrutineesStillWork:
    """The guard must not narrow what already matched."""

    def test_integer(self):
        compile_source(program('#[zeropage(0x10)]\nstatic mut V: u8;\n', 'V'))

    def test_range_pattern(self):
        src = (HW + '#[zeropage(0x10)]\nstatic mut V: u8;\n#[entry]\n'
               'fn main() -> ! { match V { 1..5 => { INIDISP = 1; } _ => { } };'
               ' loop { } }')
        compile_source(src)

    def test_newtype_over_an_integer(self):
        compile_source(program(
            'struct Tid(u8);\n#[zeropage(0x10)]\nstatic mut T: Tid;\n', 'T'))

    def test_enum(self):
        src = (HW + 'enum Dir { North, East }\n#[zeropage(0x10)]\n'
               'static mut FACING: Dir;\n#[entry]\n'
               'fn main() -> ! { match FACING { Dir::North => { INIDISP = 1; }'
               ' _ => { } }; loop { } }')
        compile_source(src)

    def test_newtype_over_an_enum(self):
        src = (HW + 'enum Dir { North, East }\nstruct Facing(Dir);\n'
               '#[zeropage(0x10)]\nstatic mut F: Facing;\n#[entry]\n'
               'fn main() -> ! { match F { Dir::East => { INIDISP = 1; }'
               ' _ => { } }; loop { } }')
        compile_source(src)

    def test_newtype_over_a_bool(self):
        src = (HW + 'struct Flag(bool);\n#[zeropage(0x10)]\n'
               'static mut F: Flag;\n#[entry]\n'
               'fn main() -> ! { match F { true => { INIDISP = 1; } false => { } };'
               ' loop { } }')
        compile_source(src)
