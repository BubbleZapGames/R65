# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Dead-static elimination drops a static no live code refers to, with its
`__init_start` code and ROM data, and keeps the ones it must."""

import re

from r65.compiler.main import compile_string

PROGRAM = """
#[zeropage(0x10)] static mut OUT: u8;
#[zeropage(0x11)] static mut OUTP: *u8;
#[zeropage(0x14, register)] static mut SCRATCH: u8;
#[zeropage] static mut DEAD_ZP: u16;
#[ram] static mut DEAD_BUF: [u8; 64] = [7; 64];
static DEAD_TABLE: [u8; 4] = [1, 2, 3, 4];
#[ram] static mut LIVE: u8 = 3;
#[used] #[ram] static mut KEPT: u8 = 9;
#[ram(0x7E3000)] static mut PINNED: u8;
#[hw(0x4200)] static mut NMITIMEN: u8;
static PTRS: [*u8; 1] = [&TARGET as *u8];
static TARGET: [u8; 2] = [5, 6];
#[ram] static mut IN_ASM: u8;
fn dead() { DEAD_ZP = 1; }

#[entry]
fn main() {
    OUT = LIVE;
    OUTP = PTRS[0];
    asm!("LDA IN_ASM");
}
"""


def _defined(asm: str, name: str) -> bool:
    return re.search(rf"^\s*(\.DEFINE\s+{name}\b|{name}\s+dsb|{name}:|__{name}_data:)",
                     asm, re.M) is not None


def test_unreferenced_statics_go_and_the_rest_stay():
    asm = compile_string(PROGRAM, "t.r65")
    for name in ("DEAD_ZP", "DEAD_BUF", "DEAD_TABLE"):
        assert not _defined(asm, name), name
    # Read by code, by a live ROM table, by asm text; or pinned.
    for name in ("LIVE", "PTRS", "TARGET", "IN_ASM", "KEPT", "PINNED", "SCRATCH", "NMITIMEN"):
        assert _defined(asm, name), name


def test_dead_initializer_code_goes():
    asm = compile_string(PROGRAM, "t.r65")
    init = asm[asm.index("\n__init_start:"):]
    init = init[:init.index("RTS")]
    assert "#$07" not in init           # DEAD_BUF's fill
    assert "#$03" in init and "#$09" in init    # LIVE's and KEPT's


def test_soa_planes_live_or_die_together():
    asm = compile_string("""
        #[soa] struct E { hp: u8, x: u8 }
        #[lowram] static mut POOL: [E; 4];
        #[zeropage(0x10)] static mut OUT: u8;
        #[entry] fn main() { OUT = POOL[1].hp; }
    """, "t.r65")
    assert _defined(asm, "POOL__hp") and _defined(asm, "POOL__x")
