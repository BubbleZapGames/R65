# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A function that binds a parameter cannot be a function pointer.

A call through a function pointer passes every argument on the stack, and
the pointer type does not record where the callee looks, so `F = dbl` with
`dbl(v @ A)` compiled and `dbl` read whatever A held.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.typeck.errors import TypeCheckError


def compile_prog(decls: str, body: str):
    return compile_string(f"""
        #[lowram(0x0300)] static mut OUT: u8;
        #[lowram(0x0310)] static mut T: u8;
        {decls}
        #[entry] fn main() {{ {body} }}
    """, cfg_options=['snes'])


def test_register_bound_parameter_is_rejected():
    with pytest.raises(TypeCheckError, match="binds parameter 'v' to 'A'"):
        compile_prog("fn dbl(v @ A: u8) -> u8 { return v + v; }\n"
                     "#[lowram(0x0320)] static mut F: fn(u8) -> u8;",
                     "F = dbl; OUT = F(3);")


def test_variable_bound_parameter_is_rejected():
    with pytest.raises(TypeCheckError, match="binds parameter 'v' to 'T'"):
        compile_prog("fn dbl(v @ T: u8) -> u8 { return v + v; }\n"
                     "#[lowram(0x0320)] static mut F: fn(u8) -> u8;",
                     "F = dbl; OUT = F(3);")


def test_rom_initializer_is_rejected():
    with pytest.raises(TypeCheckError, match="cannot be used as a function pointer"):
        compile_prog("fn dbl(v @ A: u8) -> u8 { return v + v; }\n"
                     "static F: fn(u8) -> u8 = dbl;",
                     "OUT = F(3);")


def test_unbound_parameters_still_convert():
    compile_prog("fn dbl(v: u8) -> u8 { return v + v; }\n"
                 "#[lowram(0x0320)] static mut F: fn(u8) -> u8;",
                 "F = dbl; OUT = F(3);")
