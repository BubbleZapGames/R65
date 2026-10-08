# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Values of a loop whose counter lives in Y and indexes via X.

Correctness guard for skipping a TYX when X already holds Y's value (shape
pinned in tests/compiler/codegen/test_redundant_index_transfer.py): every arm
must still address its own element.
"""

from r65.tests.e2e import ExpectedState


def test_every_arm_addresses_its_own_element(e2e):
    result = e2e.run('''
        #[soa] struct Spark { heat: u8 }
        #[soa] struct Ember { heat: u8, glow: u8 }
        enum Fx { Off, S(Spark), E(Ember) }
        #[soa] struct Particle { x: u8, pos: u16, fx: Fx }
        #[zeropage(0x10)]
        static mut PARTS: [Particle; 4];
        #[zeropage(0x60)]
        static mut OUT: [u8; 8];

        #[entry]
        fn main() {
            PARTS[0].fx = Fx::E(Ember { heat: 3, glow: 10 });
            PARTS[1].fx = Fx::Off;
            PARTS[1].pos = 0x00FF;
            PARTS[2].fx = Fx::S(Spark { heat: 5 });
            PARTS[2].x = 0;
            PARTS[3].fx = Fx::E(Ember { heat: 7, glow: 20 });
            for i in 0..4 {
                match PARTS[i].fx {
                    Fx::S(s) => { s.heat = s.heat - 1; }
                    Fx::E(e) => { e.glow = e.glow + e.heat; PARTS[i].x = e.glow; }
                    Fx::Off => { PARTS[i].pos = PARTS[i].pos + 1; PARTS[i].x = 0xEE; }
                };
            }
            for i in 0..4 { OUT[i] = PARTS[i].x; }
            OUT[4] = match PARTS[2].fx { Fx::S(s) => s.heat, _ => 0 };
            OUT[5] = (PARTS[1].pos >> 8) as u8;
        }
    ''', ExpectedState(memory={0x7E0060: [13, 0xEE, 0, 27, 4, 0x01]}), max_instructions=100000)
    assert result.success, f"Failures: {result.failures}"
