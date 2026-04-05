# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
E2E test for slot allocator incorrectly merging live variables.

Bug: When a local `sy` is computed from another local `oy` (sy = oy - offset),
the slot allocator may assign them the same stack slot because liveness analysis
fails to detect their overlap. Writing to sy then corrupts oy.

Reproduction: `oy` is defined early and used late (after function calls).
`sy` is computed from `oy` in between. The slot allocator must keep them
in separate slots.
"""

from r65.tests.e2e import ExpectedState


class TestSlotReuseClobber:
    """Test that slot reuse doesn't merge live variables."""

    def test_derived_local_does_not_clobber_source(self, e2e):
        """A local computed from another must not share its slot."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[ram]
            static mut data: [u8; 32];
            #[lowram]
            static mut out_floor: u8;
            #[lowram]
            static mut out_oy: u8;

            far fn lookup(x: u8, y: u8) -> u8 {
                return x + y;
            }

            far fn draw(x: u8, y: u8, tile: u8, slot: u16) {
                // Side-effectful function call that forces register spills
                data[0] = x;
            }

            far fn process(idx: u8, spr: u16) {
                let ox: u8 = data[idx];
                let oy: u8 = data[idx + 1];
                let mut sy: u8;

                // sy = oy - offset (derived from oy)
                if data[idx + 2] < 19 {
                    sy = oy - data[idx + 2];
                    data[idx + 2] = data[idx + 2] + 1;
                } else {
                    sy = oy;
                }

                // Use sy in a function call (forces it to be materialized)
                draw(ox, sy, 0x10, spr);

                // Use oy AFTER sy — must still have original value
                let floor: u8 = lookup(ox + 6, oy + 14);

                out_floor = floor;
                out_oy = oy;
            }

            #[entry]
            fn main() {
                data[0] = 48;   // ox
                data[1] = 194;  // oy
                data[2] = 5;    // offset (< 19)
                process(0, 100);
            }
        ''', ExpectedState(memory={
            # out_oy must be 194 (original oy), NOT 194-5=189 (sy)
            0x7E0201: 194,
            # out_floor = lookup(48+6, 194+14) = 54 + 208 = 6 (u8 wraps)
            0x7E0200: 6,
        }))
        assert result.success, f"Failures: {result.failures}"
