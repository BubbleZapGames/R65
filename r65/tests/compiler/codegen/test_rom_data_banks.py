# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""ROM static data is placed in its static's own bank."""

from r65.compiler.main import compile_string


def _bank_of(asm: str, label: str) -> int:
    bank = None
    for line in asm.splitlines():
        if line.startswith(".BANK "):
            bank = int(line.split()[1])
        elif line.strip() == f"{label}:":
            return bank
    raise AssertionError(f"{label} not emitted")


def test_rom_static_data_follows_its_bank():
    """A bank-0 static read through a near pointer must not land in the
    bank the last function was emitted in."""
    asm = compile_string('''
        static Z: [u8; 4] = [1, 2, 3, 4];
        #[zeropage(0x10)] static mut R: u8;
        #[zeropage(0x12)] static mut P: *u8;
        #[entry] fn main() { P = &Z[0]; R = helper(); }
        #[bank(1)]
        static T: [u8; 4] = [9, 8, 7, 6];
        far fn helper() -> u8 { let i: u16 = 1; return T[i]; }
    ''', cfg_options=["snes"])
    assert _bank_of(asm, "__Z_data") == 0
    assert _bank_of(asm, "__T_data") == 1
