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
            #[zeropage(0x40)] static mut B0: u8;
            #[lowram] static mut B1: u8;
            #[zeropage(0x42)] static mut W0: u16;
            #[zeropage(0x44)] static mut W1: u16;
            #[zeropage(0x48)] static mut OUT: u8;

            #[entry]
            fn main() {
                B0 = 0;     B0 = B0 - 1;      // 0 - 1 = 255
                B1 = 255;   B1 = B1 + 1;      // 255 + 1 = 0
                W0 = 0x00FF; W0 = W0 + 1;     // carry into the high byte
                W1 = 0x0100; W1 = W1 - 1;     // borrow out of it
                OUT = B1;
            }
        ''', ExpectedState(memory={
            0x7E0040: [0xFF],
            0x7E0042: [0x00, 0x01, 0xFF, 0x00],
            0x7E0048: [0x00],
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
