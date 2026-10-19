# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end OAM entries written by `OamBuffer.set!` / `set_full!`.

The macros bind `chr` once (`let oam_chr`), constant propagation turns a
constant `chr` back into two immediate byte stores, and store merging writes
a computed one's tile and attr bytes as one 16-bit store. `chr` with a side
effect must run once.
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).resolve().parents[3] / "stdlib"


class TestOamSet:
    def test_entries(self, e2e):
        result = e2e.run(f'''
            include!("{STDLIB_DIR / "65816.r65"}")
            include!("{STDLIB_DIR / "sneslib.r65"}")
            #[lowram(0x0200)] static mut oam_buffer: OamBuffer;
            #[lowram(0x0500)] static mut CHR: u16;
            #[lowram(0x0502)] static mut N: u8;
            #[lowram(0x0503)] static mut FLIP: u16;

            #[inline(never)] fn next_chr() -> u16 {{ N = N + 1; return 0x2100 | (N as u16); }}

            #[inline(never)] fn go(i: u16) {{
                oam_buffer.set!(i, 0x11, 0x22, 0);                     // entry 1: constant
                oam_buffer.set!(i + 1, 0x33, 0x44, CHR | FLIP);        // entry 2: computed
                oam_buffer.set!(i + 2, 0x55, 0x66, next_chr());        // entry 3: called once
                oam_buffer.set_full!(i + 3, 0x0177, 0x88, CHR);        // entry 4: X bit 8 set
            }}

            #[entry]
            fn main() {{
                CHR = 0x3A4B; FLIP = 0x4000; N = 6;
                oam_buffer.ext[1] = 0;
                go(1);
            }}
        ''', ExpectedState(memory={
            0x0204: [0x11, 0x22, 0x00, 0x00,
                     0x33, 0x44, 0x4B, 0x7A,
                     0x55, 0x66, 0x07, 0x21,
                     0x77, 0x88, 0x4B, 0x3A],
            0x0502: [0x07],                    # next_chr ran once
            0x0401: [0x01],                    # entry 4's X bit 8: ext[4 >> 2], bit (4 & 3) * 2
        }))
        assert result.success, f"Failures: {result.failures}"
