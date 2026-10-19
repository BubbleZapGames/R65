# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A frame holds only the locals that stayed on the stack.

Slot allocation sizes the frame before `allocate_vreg` moves the locals not
live across a call into scratch (`RegisterAllocator._compact_frame`). Under
FixedStack, with the stdlib scratch block, a function whose locals all fit
there used to keep that frame for nothing but the return value's
`STA $n,S` and the pulls. Values: tests/e2e/test_frame_compaction.py.
"""

from pathlib import Path

from r65.compiler.codegen.abi_model import ABI_FIXED_STACK
from r65.compiler.main import compile_string

SCRATCH = Path(__file__).resolve().parents[4] / 'stdlib' / 'scratch_regs.r65'


def _fn(asm: str, name: str) -> list:
    start = asm.index(f"\n{name}:")
    end = asm.find("; ---", start)
    body = asm[start:end if end != -1 else None]
    return [l.split(';')[0].strip() for l in body.splitlines()
            if l.startswith('    ') and not l.strip().startswith(('.', ';'))
            and l.split(';')[0].strip()]


def test_locals_all_in_scratch_need_no_frame():
    src = f'include!("{SCRATCH}")\n' + '''
        #[lowram] static mut BUF: [u8; 8];
        #[zeropage(0x20)] static mut OUT: u16;
        #[inline(never)]
        fn t() -> u16 {
            let mut s: u16 = 0;
            let mut t: u16 = 0;
            let mut i: u16 = 0;
            while i < 8 { s = s + (BUF[i] as u16); t = t ^ s; i = i + 1; }
            return s + t;
        }
        #[entry]
        fn main() { OUT = t(); }
    '''
    code = _fn(compile_string(src, str(SCRATCH.parent / 't.r65'), abi_model=ABI_FIXED_STACK,
                              cfg_options=['snes']), 't')
    assert not [l for l in code if l in ('PHB', 'PHX', 'PHY', 'PLA') or ',S' in l], code
