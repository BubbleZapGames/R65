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


FIELD_READER = '''
    struct P { x: u8, y: u8 }
    #[zeropage(0x10)] static mut R: u8;
    #[inline(never)]
    fn rd(q: far *u8) -> u8 { return *q; }
    #[entry] fn main() { R = rd(&ROMP.y); }
'''


def test_far_address_with_offset_splits_the_sum():
    """`>label + 1` is `(>label) + 1` to WLA-DX: the byte prefixes need the
    sum parenthesised, and the bank byte is the label's own."""
    asm = compile_string('''
        struct P { x: u8, y: u8 }
        #[zeropage(0x10)] static mut R: u8;
        #[zeropage(0x20)] static mut FP: far *u8;
        #[entry] fn main() { FP = &ROMP.y; R = *FP; }
        #[bank(1)]
        static ROMP: P = P { x: 1, y: 2 };
    ''', cfg_options=["snes"])
    assert "LDA #<(__ROMP_data + 1)" in asm
    assert "LDA #>(__ROMP_data + 1)" in asm
    assert "#:__ROMP_data + 1" not in asm


def test_far_address_argument_is_pushed_whole():
    """A constant far address passed on the stack is its bank byte and a PEA
    of the label plus offset, not three frame bytes pushed one at a time."""
    asm = compile_string(FIELD_READER + '''
        #[bank(1)]
        static ROMP: P = P { x: 1, y: 2 };
    ''', cfg_options=["snes"])
    assert "LDA #:__ROMP_data" in asm
    assert "PEA __ROMP_data+1" in asm


def test_hirom_rom_static_address_is_far():
    """HiROM ROM bank 0 is $C0:0000-$FFFF; the data bank a near pointer reads
    through maps only its upper half, so `&ROMP.y` carries its bank byte."""
    asm = compile_string('#[snesrom(name = "HI", hirom)]' + FIELD_READER + '''
        static ROMP: P = P { x: 1, y: 2 };
    ''', cfg_options=["snes"])
    assert "#:__ROMP_data" in asm
    assert "Zero-extend bank byte" not in asm
