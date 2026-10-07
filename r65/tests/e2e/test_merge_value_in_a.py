# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Values kept in A by the slot allocator, checked on the hardware.

A value loaded from a plane stays in A across the `TYX` that re-points X at
the loop counter in Y. These pin that the values are still right — the
codegen shape is pinned in tests/compiler/codegen/test_hw_coalescing.py.
"""

from r65.tests.e2e import ExpectedState


class TestValueKeptInAAcrossTyx:

    def test_counter_in_y_plane_arithmetic(self, e2e):
        """The u16 `pos` plane takes X for its scaled index, so the loop counter
        lives in Y and the byte planes are reached through TYX."""
        result = e2e.run('''
            #[soa] struct Spark { heat: u8 }
            #[soa] struct Ember { heat: u8, glow: u8 }
            enum Fx { Off, S(Spark), E(Ember) }
            #[soa] struct Particle { x: u8, pos: u16, fx: Fx }
            #[zeropage(0x10)]
            static mut PARTS: [Particle; 4];
            #[zeropage(0x60)]
            static mut OUT: [u8; 4];

            #[entry]
            fn main() {
                PARTS[0].fx = Fx::E(Ember { heat: 3, glow: 10 });
                PARTS[1].fx = Fx::Off;
                PARTS[1].pos = 0x00FF;
                PARTS[2].fx = Fx::S(Spark { heat: 5 });
                PARTS[3].fx = Fx::E(Ember { heat: 7, glow: 20 });
                for i in 0..4 {
                    match PARTS[i].fx {
                        Fx::S(s) => { s.heat = s.heat - 1; }
                        Fx::E(e) => { e.glow = e.glow + e.heat; }
                        Fx::Off => { PARTS[i].pos = PARTS[i].pos + 1; }
                    };
                }
                OUT[0] = match PARTS[0].fx { Fx::E(e) => e.glow, _ => 0 };
                OUT[1] = match PARTS[3].fx { Fx::E(e) => e.glow, _ => 0 };
                OUT[2] = match PARTS[2].fx { Fx::S(s) => s.heat, _ => 0 };
                OUT[3] = (PARTS[1].pos >> 8) as u8;
            }
        ''', ExpectedState(memory={0x7E0060: [13, 27, 4, 0x01]}))
        assert result.success, f"Failures: {result.failures}"
