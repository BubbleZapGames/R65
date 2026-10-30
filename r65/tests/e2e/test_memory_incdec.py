# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""INC/DEC on memory, checked on the hardware.

`m = m ± 1` on a zeropage/lowram static or an #[soa] plane is one
read-modify-write (see tests/compiler/codegen/test_memory_incdec.py for the
emitted shape). These pin the values: wrap-around at both widths, the carry
into a u16's high byte, and both outcomes of the zero branch that reuses the
INC/DEC flags.
"""

from r65.tests.e2e import ExpectedState


class TestIncDecMemory:

    def test_wraparound_both_widths(self, e2e):
        result = e2e.run('''
            #[zeropage(0xC0)] static mut B0: u8;
            #[lowram] static mut B1: u8;
            #[zeropage(0xC2)] static mut W0: u16;
            #[zeropage(0xC4)] static mut W1: u16;
            #[zeropage(0xC8)] static mut OUT: u8;

            #[entry]
            fn main() {
                B0 = 0;     B0 = B0 - 1;      // 0 - 1 = 255
                B1 = 255;   B1 = B1 + 1;      // 255 + 1 = 0
                W0 = 0x00FF; W0 = W0 + 1;     // carry into the high byte
                W1 = 0x0100; W1 = W1 - 1;     // borrow out of it
                OUT = B1;
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0xFF],
            0x7E00C2: [0x00, 0x01, 0xFF, 0x00],
            0x7E00C8: [0x00],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_plane_countdown_branches_on_both_outcomes(self, e2e):
        result = e2e.run('''
            #[soa] struct Spark { heat: u8 }
            #[soa] struct Flare { pos: u16 }
            enum Fx { Off, S(Spark), F(Flare) }
            #[soa] struct Particle { x: u8, fx: Fx }
            #[zeropage(0x10)]
            static mut PARTS: [Particle; 4];
            #[zeropage(0x60)]
            static mut HITS: [u8; 4];

            #[entry]
            fn main() {
                PARTS[0].fx = Fx::S(Spark { heat: 1 });   // reaches 0 on the first tick
                PARTS[1].fx = Fx::S(Spark { heat: 3 });   // does not
                PARTS[2].fx = Fx::F(Flare { pos: 0x00FF });
                PARTS[3].fx = Fx::S(Spark { heat: 0 });   // wraps to 255, not 0
                for i in 0..4 {
                    HITS[i] = 0;
                    match PARTS[i].fx {
                        Fx::S(s) => { s.heat = s.heat - 1; if s.heat == 0 { HITS[i] = 1; } }
                        Fx::F(f) => { f.pos = f.pos + 1; }
                        Fx::Off => {}
                    };
                }
            }
        ''', ExpectedState(memory={0x7E0060: [1, 0, 0, 0]}), max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_shifts_by_one_on_memory(self, e2e):
        result = e2e.run('''
            #[zeropage(0xC0)] static mut BY: u8;
            #[zeropage(0xC2)] static mut W: u16;
            #[zeropage(0xC4)] static mut SG: i8;
            #[lowram] static mut ARR: [u8; 4];
            #[zeropage(0xC6)] static mut OUT: [u8; 2];

            #[inline(never)]
            fn step(i @ X: u16) {
                BY = BY << 1;
                W = W >> 1;
                SG = SG >> 1;
                ARR[i] = ARR[i] >> 1;
            }

            #[entry]
            fn main() {
                BY = 0xC1;  W = 0x0301;  SG = -6;
                ARR[2] = 0x81;
                step(2);
                OUT[0] = ARR[2];
                OUT[1] = ARR[3];
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0x82],          # 0xC1 << 1, bit 7 out
            0x7E00C2: [0x80, 0x01],    # 0x0301 >> 1: bit 8 moves into the low byte
            0x7E00C4: [0xFD],          # -6 >> 1 = -3
            0x7E00C6: [0x40, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"


def test_direct_page_locals_shift_in_place():
    """Under FixedStack the locals are in direct page: the peephole folds
    `LDA v / LSR A / STA v` to `LSR v` once A is dead (`_fold_memory_rmw`)."""
    from r65.tests.e2e.framework import E2ETest
    result = E2ETest(abi="FixedStack").run('''
        #[lowram] static mut BUF: [u8; 4];
        #[zeropage(0x20)] static mut BITS: u16;
        #[zeropage(0x22)] static mut CRC: u16;

        #[inline(never)]
        fn popcount() -> u16 {
            let mut c: u16 = 0;
            let mut i: u16 = 0;
            while i < 4 {
                let mut v: u8 = BUF[i];
                while v != 0 {
                    c = c + ((v & 1) as u16);
                    v = v >> 1;
                }
                i = i + 1;
            }
            return c;
        }

        #[inline(never)]
        fn halve(n: u16) -> u16 {
            let mut w: u16 = 0xF0F1;
            let mut k: u16 = 0;
            while k < n {
                w = w >> 1;
                k = k + 1;
            }
            return w;
        }

        #[entry]
        fn main() {
            BUF[0] = 0xFF; BUF[1] = 0x10; BUF[2] = 0; BUF[3] = 0x81;
            BITS = popcount();      // 8 + 1 + 0 + 2
            CRC = halve(4);         // 0xF0F1 >> 4
        }
    ''', ExpectedState(memory={
        0x7E0020: [11, 0],
        0x7E0022: [0x0F, 0x0F],
    }), max_instructions=100000)
    assert result.success, f"Failures: {result.failures}"
