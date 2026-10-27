# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Call lowering (mir/lowerers/call.py) at the MIR/codegen seam."""

from r65.compiler.main import compile_string


def test_type_id_through_far_dyn_reads_far():
    """`p.type_id()` on a `far *dyn` reads the TypeId byte through the bank."""
    asm = compile_string('''
        #[zeropage(0x10, register)] static mut S0: u8;
        #[zeropage(0x12, register)] static mut S1: u16;
        struct P { x: u8 }
        trait Draw { fn draw(*self); }
        impl Draw for P { fn draw(*self) {} }
        #[ram] static mut PL: P;
        #[lowram] static mut TID: u8;
        #[used]
        #[inline(never)] fn tid(p: far *dyn Draw) { TID = p.type_id(); }
        #[entry] fn main() { tid(&PL as far *dyn Draw); }
    ''')
    body = asm.split('tid:', 1)[1].split('RTS', 1)[0]
    assert 'through far pointer' in body
