# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A run of four or more constant bytes stored to consecutive addresses
becomes word stores (`ConstantStoreMerger`): `LDA #$5048 / STA BUF` for
"HP". A shorter run stays byte stores, which are cheaper than a pair plus
its REP/SEP, and a `#[hw]` register is never written two bytes at a time.
"""

import re

from r65.compiler.main import compile_string


def _main_asm(body: str) -> str:
    source = f"""
    #[ram] static mut BUF: [u8; 8];
    #[hw(0x2118)] static mut VMDATAL: u8;
    #[hw(0x2119)] static mut VMDATAH: u8;
    #[hw(0x211A)] static mut VMAIN2: u8;
    #[hw(0x211B)] static mut M7A: u8;
    #[entry] fn main() {{ {body} }}
    """
    return compile_string(source, cfg_options=["snes"]).split("\nmain:", 1)[1]


def test_four_bytes_are_two_word_stores():
    asm = _main_asm("BUF[0] = 0x48; BUF[1] = 0x50; BUF[2] = 0x3A; BUF[3] = 0x20;")
    assert re.search(r"LDA #\$5048\s*\n\s*STA \S+", asm)
    assert re.search(r"LDA #\$203A\s*\n\s*STA \S+", asm)


def test_three_bytes_stay_byte_stores():
    asm = _main_asm("BUF[0] = 0x48; BUF[1] = 0x50; BUF[2] = 0x3A;")
    assert not re.search(r"LDA #\$[0-9A-F]{4}\b", asm)


def test_hardware_registers_are_left_alone():
    asm = _main_asm("VMDATAL = 1; VMDATAH = 2; VMAIN2 = 3; M7A = 4;")
    assert not re.search(r"LDA #\$[0-9A-F]{4}\b", asm)
