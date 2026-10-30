# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Values kept in A by the slot allocator, checked on the hardware.

A `match` / `if` result is assigned at the end of each arm and read at the
join; it now stays in A instead of a stack slot. A value loaded from a plane
stays in A across the `TYX` that re-points X at the loop counter in Y. These
pin that the values are still right — the codegen shape is pinned in
tests/compiler/codegen/test_hw_coalescing.py.
"""

from r65.tests.e2e import ExpectedState


SHAPES = '''
    struct Circle { r: u8 }
    struct Rect   { w: u8, h: u8 }
    enum Shape { Empty, C(Circle), R(Rect) }

    #[zeropage(0x10)]
    static mut POOL: [Shape; 4];
    #[zeropage(0xC0)]
    static mut OUT: [u8; 4];
    #[zeropage(0xC8)]
    static mut OUT2: u8;
'''


class TestMergeValueInA:

    def test_every_arm_of_a_match(self, e2e):
        result = e2e.run(SHAPES + '''
            fn get(i: u16) {
                OUT[i] = match POOL[i] {
                    Shape::C(c) => c.r,
                    Shape::R(r) => r.h,
                    Shape::Empty => 0x77,
                };
            }

            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0x11 });
                POOL[1] = Shape::R(Rect { w: 0x22, h: 0x23 });
                POOL[2] = Shape::Empty;
                POOL[3] = Shape::C(Circle { r: 0x44 });
                get(0); get(1); get(2); get(3);
            }
        ''', ExpectedState(memory={0x7E00C0: [0x11, 0x23, 0x77, 0x44]}))
        assert result.success, f"Failures: {result.failures}"

    def test_if_expression(self, e2e):
        result = e2e.run(SHAPES + '''
            fn pick(c: u8, i: u16) {
                OUT[i] = if c != 0 { POOL[0] as u8 + 5 } else { 0x66 };
            }

            #[entry]
            fn main() {
                POOL[0] = Shape::R(Rect { w: 1, h: 2 });
                pick(1, 0);
                pick(0, 1);
            }
        ''', ExpectedState(memory={0x7E00C0: [0x07, 0x66]}))
        assert result.success, f"Failures: {result.failures}"

    def test_join_that_uses_a_before_the_merge(self, e2e):
        result = e2e.run(SHAPES + '''
            fn get(i: u16) {
                let v: u8 = match POOL[i] {
                    Shape::C(c) => c.r,
                    _ => 9,
                };
                OUT2 = 7;
                OUT[i] = v;
            }

            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0x31 });
                POOL[1] = Shape::Empty;
                get(0); get(1);
            }
        ''', ExpectedState(memory={0x7E00C0: [0x31, 0x09], 0x7E00C8: [0x07]}))
        assert result.success, f"Failures: {result.failures}"


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
