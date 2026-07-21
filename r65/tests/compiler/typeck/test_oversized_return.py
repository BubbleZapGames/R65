# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Return values must fit in a return register.

Root cause (fixed): the return ABI is register-count based - each returned
value rides back in A, B/X, or Y - but nothing checked that the value fits.
A `far *T` return (3 bytes) was classified as "one register": the callee
materialised the pointer in its stack frame, kept only the low byte in A, and
tore the frame down. The caller then read all three bytes back out of the
deallocated frame, so

    far fn area_ptr(id: u8) -> far *Area { return &AREAS[id as u16] as far *Area; }
    let a: far *Area = area_ptr(CUR);
    take(a.name);

worked or not depending on whether an unrelated later call happened to
overwrite the dead frame. Now it is a type error.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import TypeCheckError


FAR_POINTER_RETURN = """
struct Area { name: far *u8, pad: u8 }

static NAME: [u8; 4] = "HI\\0";
static AREAS: [Area; 2] = [
    Area { name: &NAME as far *u8, pad: 0 },
    Area { name: &NAME as far *u8, pad: 0 }
];

far fn area_ptr(id: u8) -> far *Area {
    return &AREAS[id as u16] as far *Area;
}

#[entry]
fn main() {
    let a: far *Area = area_ptr(1);
}
"""


def test_far_pointer_return_is_rejected():
    with pytest.raises(TypeCheckError, match='does not fit in a return register'):
        compile_string(FAR_POINTER_RETURN, cfg_options=['snes'])


NEAR_POINTER_RETURN = """
#[zeropage(0x20, register)]
static mut SCRATCH0: u16;

#[zeropage(0x30)]
static mut OUT: u8;

static NAME: [u8; 4] = "HI\\0";

fn name_ptr() -> *u8 {
    return &NAME as *u8;
}

#[entry]
fn main() {
    let p: *u8 = name_ptr();
    OUT = *p;
}
"""


def test_near_pointer_return_still_allowed():
    """Two bytes fits in A, so this stays legal."""
    asm = compile_string(NEAR_POINTER_RETURN, cfg_options=['snes'])
    assert 'name_ptr:' in asm


MULTI_RETURN_OK = """
fn pair() -> u8, u16 {
    return 1, 2;
}

#[entry]
fn main() {
    let a, b = pair();
}
"""


def test_multi_return_of_register_sized_values_still_allowed():
    asm = compile_string(MULTI_RETURN_OK, cfg_options=['snes'])
    assert 'pair:' in asm
