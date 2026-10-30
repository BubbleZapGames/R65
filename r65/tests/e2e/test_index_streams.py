# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for index-stream splitting (optimize/index_streams.py).

A statement touching two arrays at two indexes used to reload X for every
access; one stream now takes Y when every access of it can be abs,Y and Y
is free. These check the values come out right in the shapes it rewrites -
the OAM copy loop `oam[spr].f = pool[i].f`, lowram to lowram, a zeropage
destination (abs,Y on a direct page address), a zero store (no STZ abs,Y) -
and in the shapes it must leave alone: a `@ Y` parameter, and a pointer
access (`LDY #offset`) inside the run.
"""

from r65.tests.e2e import ExpectedState


class TestIndexStreams:
    def test_pool_to_oam_loop(self, e2e):
        """The oam_buffer.set! shape: #[ram] source by i, lowram OAM by spr."""
        result = e2e.run('''
            struct Spr { x: u8, y: u8, tile: u8, attr: u8 }
            #[ram] static mut PX: [u8; 4];
            #[ram] static mut PY: [u8; 4];
            #[lowram(0x0300)] static mut OAM: [Spr; 8];
            #[zeropage(0xB0)] static mut N: u8;

            fn draw() {
                let mut spr: u16 = 2;
                for i in 0..N {
                    OAM[spr].x = PX[i];
                    OAM[spr].y = PY[i];
                    OAM[spr].tile = 0;
                    OAM[spr].attr = 0x30;
                    spr = spr + 1;
                }
            }

            #[entry]
            fn main() {
                PX[0] = 10; PX[1] = 11; PX[2] = 12;
                PY[0] = 20; PY[1] = 21; PY[2] = 22;
                OAM[2].tile = 0xEE; OAM[3].tile = 0xEE; OAM[4].tile = 0xEE;
                N = 3;
                draw();
            }
        ''', ExpectedState(memory={
            0x0308: [10, 20, 0, 0x30, 11, 21, 0, 0x30, 12, 22, 0, 0x30],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_lowram_to_lowram_two_indexes(self, e2e):
        result = e2e.run('''
            struct Pair { a: u8, b: u8 }
            #[lowram(0x0300)] static mut SRC: [Pair; 4];
            #[lowram(0x0340)] static mut DST: [Pair; 4];
            #[zeropage(0xB0)] static mut I: u16;
            #[zeropage(0xB2)] static mut J: u16;

            #[entry]
            fn main() {
                SRC[1].a = 7; SRC[1].b = 9;
                I = 1; J = 3;
                let i = I;
                let j = J;
                DST[j].a = SRC[i].a;
                DST[j].b = SRC[i].b + DST[j].a;
            }
        ''', ExpectedState(memory={
            0x0346: [7, 16],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_zeropage_destination(self, e2e):
        """A direct page static indexed by Y goes through abs,Y."""
        result = e2e.run('''
            #[ram] static mut SRC: [u8; 4];
            #[zeropage(0xC0)] static mut DST: [u8; 8];
            #[zeropage(0xB0)] static mut I: u16;
            #[zeropage(0xB2)] static mut J: u16;

            #[entry]
            fn main() {
                SRC[2] = 5; SRC[3] = 6;
                I = 2; J = 4;
                let i = I;
                let j = J * 2;
                DST[j] = SRC[i];
                DST[j + 1] = SRC[i + 1];
                DST[j] = DST[j] + SRC[i];
            }
        ''', ExpectedState(memory={
            0x7E00C8: [10, 6],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_y_parameter_survives(self, e2e):
        """A function with a `@ Y` parameter keeps it; the pass skips it."""
        result = e2e.run('''
            struct Spr { x: u8, y: u8 }
            #[ram] static mut PX: [u8; 4];
            #[lowram(0x0300)] static mut OAM: [Spr; 4];
            #[zeropage(0xB0)] static mut OUT: u16;

            fn copy(i @ X: u16, keep @ Y: u16) {
                OAM[1].x = PX[i];
                OAM[1].y = PX[i + 1];
                OUT = keep;
            }

            #[entry]
            fn main() {
                PX[0] = 3; PX[1] = 4;
                copy(0, 0x1234);
            }
        ''', ExpectedState(memory={
            0x0302: [3, 4],
            0x7E00B0: [0x34, 0x12],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_pointer_access_in_the_run(self, e2e):
        """A pointer read sets Y for its offset; the copy must not lose it."""
        result = e2e.run('''
            struct Spr { x: u8, y: u8 }
            struct Pt { a: u8, b: u8 }
            #[ram] static mut PX: [u8; 4];
            #[lowram(0x0300)] static mut OAM: [Spr; 4];
            #[lowram(0x0320)] static mut PT: Pt;
            #[zeropage(0xB0)] static mut I: u16;
            #[zeropage(0xB2)] static mut SP_IN: u16;

            fn copy(p: *Pt) {
                let i = I;
                let s = SP_IN;
                OAM[s].x = PX[i];
                OAM[s].y = (*p).b;
                OAM[s].x = OAM[s].x + PX[i + 1];
            }

            #[entry]
            fn main() {
                PX[0] = 3; PX[1] = 4;
                PT.b = 9;
                I = 0; SP_IN = 2;
                copy(&PT);
            }
        ''', ExpectedState(memory={
            0x0304: [7, 9],
        }))
        assert result.success, f"Failures: {result.failures}"
