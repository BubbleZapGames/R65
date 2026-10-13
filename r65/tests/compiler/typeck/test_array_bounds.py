# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A constant index is bounds-checked at compile time; a dynamic one is not.

The plain-array check raised its error inside its own `except Exception`, so
an out-of-bounds constant on an array compiled; only `(*p)[k]` was checked.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.typeck.errors import TypeCheckError


def compile_body(body: str):
    return compile_string(f"""
        #[ram] static mut BUFFER: [u8; 256];
        #[zeropage] static mut I: u16;
        #[entry] fn main() {{ {body} }}
    """, cfg_options=['snes'])


def test_constant_index_in_bounds():
    compile_body("BUFFER[0] = 42; BUFFER[255] = 1;")


def test_dynamic_index_is_not_checked():
    compile_body("I = 300; BUFFER[I] = 1;")


def test_constant_index_past_the_end():
    with pytest.raises(TypeCheckError, match="out of bounds for array of size 256"):
        compile_body("BUFFER[256] = 99;")


def test_constant_expression_index_past_the_end():
    with pytest.raises(TypeCheckError, match="index 300 is out of bounds"):
        compile_body("BUFFER[100 * 3] = 99;")


def test_pointer_to_array_index_past_the_end():
    with pytest.raises(TypeCheckError, match="out of bounds for array of size 4"):
        compile_string("""
            #[ram] static mut ARR: [u8; 4];
            #[zeropage] static mut O: u8;
            fn f(p: *[u8; 4]) -> u8 { return (*p)[4]; }
            #[entry] fn main() { O = f(&ARR); }
        """, cfg_options=['snes'])
