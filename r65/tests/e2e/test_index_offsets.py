# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for constant index offset folding (optimize/index_offsets.py).

`walkmap[t + 32]` loads X with t and addresses `walkmap+32,X`, so the
neighbouring cells share one X load. These check the values in the shapes
it folds - unscaled, through a struct stride, negative, a ROM table - and
the one it must not: a u8 index, whose `+ 1` wraps at 256.
"""

from r65.tests.e2e import ExpectedState


class TestIndexOffsets:
    def test_neighbouring_cells(self, e2e):
        result = e2e.run('''
            #[ram] static mut MAP: [u8; 128];
            #[lowram(0x0300)] static mut OUT: [u8; 4];
            #[zeropage(0x30)] static mut T: u16;

            #[entry]
            fn main() {
                T = 33;
                let t = T;
                MAP[t] = 1;
                MAP[t + 1] = 2;
                MAP[t + 32] = 3;
                MAP[t - 1] = 4;
                OUT[0] = MAP[33];
                OUT[1] = MAP[34];
                OUT[2] = MAP[65];
                OUT[3] = MAP[32];
            }
        ''', ExpectedState(memory={0x0300: [1, 2, 3, 4]}))
        assert result.success, f"Failures: {result.failures}"

    def test_struct_stride(self, e2e):
        """`ROWS[i + 1].b`: the constant comes out through the << scale."""
        result = e2e.run('''
            struct Row { a: u8, b: u8, c: u8, d: u8 }
            #[ram] static mut ROWS: [Row; 8];
            #[lowram(0x0300)] static mut OUT: [u8; 2];
            #[zeropage(0x30)] static mut I: u16;

            #[entry]
            fn main() {
                I = 2;
                let i = I;
                ROWS[i + 1].b = 9;
                ROWS[i - 1].d = 8;
                OUT[0] = ROWS[3].b;
                OUT[1] = ROWS[1].d;
            }
        ''', ExpectedState(memory={0x0300: [9, 8]}))
        assert result.success, f"Failures: {result.failures}"

    def test_rom_table_below(self, e2e):
        """A negative offset on a ROM label: `TABLE-1,X`."""
        result = e2e.run('''
            static TABLE: [u8; 4] = [10, 20, 30, 40];
            #[lowram(0x0300)] static mut OUT: [u8; 2];
            #[zeropage(0x30)] static mut I: u16;

            #[entry]
            fn main() {
                I = 2;
                let i = I;
                OUT[0] = TABLE[i - 1];
                OUT[1] = TABLE[i + 1];
            }
        ''', ExpectedState(memory={0x0300: [20, 40]}))
        assert result.success, f"Failures: {result.failures}"

    def test_u8_ring_wraps(self, e2e):
        """A u8 index wraps at 256: `RING[head + 1]` with head 255 is RING[0]."""
        result = e2e.run('''
            #[ram] static mut RING: [u8; 256];
            #[lowram(0x0300)] static mut OUT: [u8; 1];
            #[zeropage(0x30)] static mut HEAD: u8;

            #[entry]
            fn main() {
                HEAD = 255;
                let head = HEAD;
                RING[head + 1] = 7;
                OUT[0] = RING[0];
            }
        ''', ExpectedState(memory={0x0300: [7]}))
        assert result.success, f"Failures: {result.failures}"
