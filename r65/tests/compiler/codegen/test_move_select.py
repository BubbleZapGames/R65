# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Move instruction selection."""

from r65.compiler.main import compile_string


def test_u8_immediate_store_after_u16_work_runs_in_m8():
    """`let mut b: u8 = 5;` right after u16 work must not store 2 bytes."""
    asm = compile_string('''
        #[lowram] static mut W: u16;
        #[zeropage(0x10)] static mut R: u8;
        #[zeropage(0x12)] static mut Q: u16;
        #[entry] fn main() {
            let mut a: u16 = W + 1;
            let mut b: u8 = 5;
            a = a + 1;
            b = b + 1;
            Q = a;
            R = b;
        }
    ''', cfg_options=["snes"])
    lines = [l.split(";")[0].strip() for l in asm.splitlines()]
    lines = [l for l in lines if l and not l.startswith(".")]
    store = lines.index("LDA #$05")
    mode = next(l for l in reversed(lines[:store]) if l.split()[0] in ("SEP", "REP"))
    assert mode == "SEP #$20"
