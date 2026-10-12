# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
The bank a stdlib macro takes with WLA-DX's `:` operator.

`dma_set_src!` and `block_move!` get a symbol's bank from `:{name}`. A
`#[ram]` static is a label in a `.RAMSECTION` under `.BASE $7E`, so `:` is
$7E; a `#[lowram]` static is a `.DEFINE` - a plain number - for which WLA
gives bank 0, which is right for low RAM (and mirrored in every bank a
program runs from). Projects used to hand-write the DMA bank for low-RAM
buffers on the belief that `:` was wrong for them; these pin that it is not.
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
CPU_PATH = STDLIB_DIR / "65816.r65"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"

PRELUDE = f'''
    include!("{CPU_PATH}")
    include!("{SNESLIB_PATH}")
'''


def test_dma_source_bank_of_a_lowram_buffer_is_zero(e2e):
    result = e2e.run(PRELUDE + '''
        #[lowram(0x0300)]
        static mut BUF: [u8; 16];

        #[entry]
        fn main() {
            A1B3 = 0x55;
            dma_set_src!(3, BUF);
        }
    ''', ExpectedState(memory={0x004332: [0x00, 0x03, 0x00]}))
    assert result.success, f"Failures: {result.failures}"


def test_dma_source_bank_of_a_ram_buffer_is_7e(e2e):
    result = e2e.run(PRELUDE + '''
        #[ram]
        static mut BUF: [u8; 16];

        #[entry]
        fn main() {
            dma_set_src!(3, BUF);
        }
    ''', ExpectedState(memory={0x004334: 0x7E}))
    assert result.success, f"Failures: {result.failures}"


def test_block_move_between_lowram_buffers(e2e):
    result = e2e.run(PRELUDE + '''
        #[lowram(0x0300)]
        static mut SRC: [u8; 4];
        #[lowram(0x0310)]
        static mut DST: [u8; 4];

        #[entry]
        fn main() {
            SRC[0] = 1; SRC[1] = 2; SRC[2] = 3; SRC[3] = 4;
            block_move!(SRC, DST, 4);
        }
    ''', ExpectedState(memory={0x7E0310: [1, 2, 3, 4]}))
    assert result.success, f"Failures: {result.failures}"
    assert result.cpu.DBR == 0
