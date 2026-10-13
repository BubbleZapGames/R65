# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Every array access scales its index through compute_scaled_index."""

from r65.compiler.main import compile_string


def test_address_of_three_byte_element_uses_shift_and_add():
    """`&T[i]` of a 3-byte element is (i << 1) + i, not a `*` codegen rejects."""
    asm = compile_string('''
        struct V3 { x: u8, y: u8, z: u8 }
        #[lowram] static mut TBL: [V3; 8];
        #[lowram] static mut OUT: u16;
        #[inline(never)] fn addr(i: u8) -> *V3 { return &TBL[i]; }
        #[entry] fn main() { OUT = addr(5) as u16; }
    ''')
    body = asm.split('addr:', 1)[1].split('RTS', 1)[0]
    assert 'ASL A' in body


def test_power_of_two_struct_over_16_bytes_shifts():
    """A 32-byte struct element scales by a shift, without mul16."""
    asm = compile_string('''
        struct Big { a: [u8; 31], b: u8 }
        #[lowram] static mut ARR: [Big; 4];
        #[inline(never)] fn set(i: u8) { ARR[i].b = 1; }
        #[entry] fn main() { set(2); }
    ''')
    body = asm.split('set:', 1)[1].split('RTS', 1)[0]
    assert 'mul16' not in body and body.count('ASL A') == 5
