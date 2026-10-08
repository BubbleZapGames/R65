# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Values of a loop whose index lives in a stack slot and is reused from X.

Correctness guard for skipping an index reload X already holds (shape pinned
in tests/compiler/codegen/test_index_slot_reuse.py).
"""

from r65.tests.e2e import ExpectedState


def test_nearest_opposing_unit(e2e):
    result = e2e.run('''
        struct Tank { reload: u8 }
        enum UnitType { Free, Tank(Tank), Van }
        #[soa]
        struct Unit { x_px: u16, owner: u8, kind: UnitType }
        #[lowram] static mut units: [Unit; 8];
        #[zeropage(0x60)] static mut OUT: [u16; 2];

        #[inline(never)]
        fn target(self_i: u16, range: u16) -> u16 {
            let x: u16 = units[self_i].x_px;
            let mine: u8 = units[self_i].owner;
            let mut best: u16 = 8;
            let mut best_d: u16 = range;
            for j in 0..8 {
                match units[j].kind {
                    UnitType::Free => {}
                    _ => {
                        if units[j].owner != mine {
                            let mut d: u16 = 0;
                            if units[j].x_px > x { d = units[j].x_px - x; }
                            if d < best_d { best_d = d; best = j; }
                        }
                    }
                };
            }
            return best;
        }

        #[entry]
        fn main() {
            for i in 0..8 { units[i].kind = UnitType::Free; }
            units[0].kind = UnitType::Van;  units[0].owner = 0; units[0].x_px = 1000;
            units[2].kind = UnitType::Van;  units[2].owner = 1; units[2].x_px = 1300;   // d 300
            units[3].kind = UnitType::Van;  units[3].owner = 0; units[3].x_px = 1010;   // same side
            units[5].kind = UnitType::Tank(Tank { reload: 0 }); units[5].owner = 1; units[5].x_px = 1100; // d 100
            units[6].kind = UnitType::Van;  units[6].owner = 1; units[6].x_px = 900;    // behind: d 0
            OUT[0] = target(0, 500);
            units[6].kind = UnitType::Free;
            OUT[1] = target(0, 500);
        }
    ''', ExpectedState(memory={0x7E0060: [6, 0, 5, 0]}), max_instructions=200000)
    assert result.success, f"Failures: {result.failures}"
