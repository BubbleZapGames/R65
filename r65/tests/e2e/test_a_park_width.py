# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for the width of an A value parked across stack-argument
setup.

Bug: rotor_raiders' `heli_fire` computed a u16 `y` into A for a
scratch-promoted parameter, and the call also had a u16 stack argument
the location tracker reported as A-resident. That took the stack path of
`_compute_a_save_for_call`, which parked A in Y but never set the restore
width, so the value came back through an 8-bit TYA with B's stale high
byte: 0x6700 arrived as 0x0200 and every bullet spawned on the HUD row.
"""

from r65.tests.e2e import ExpectedState


class TestAParkWidth:
    def test_u16_scratch_arg_survives_stack_arg_setup(self, e2e):
        """The shape of heli_fire's muni_spawn call: three scratch-promoted
        parameters, four on the stack, the u16 stack argument chosen by a
        branch just before the call, and the u16 scratch argument computed
        into A last."""
        result = e2e.run('''
            #[zeropage(0x16, register)] static mut SCRATCH0: u16;
            #[zeropage(0x18, register)] static mut SCRATCH1: u16;
            #[zeropage(0x1A, register)] static mut SCRATCH2: u16;

            #[zeropage(0x30)] static mut base_x: u16;
            #[zeropage(0x32)] static mut base_y: u16;
            #[zeropage(0x34)] static mut face_left: u8;
            #[zeropage(0x36)] static mut out_y: u16;
            #[zeropage(0x38)] static mut out_vx: i16;

            fn spawn(kind: u8, x: u16, y: u16, vx: i16, vy: i16, life: u8, owner: u8) {
                out_y = y;
                out_vx = vx;
            }

            #[entry]
            fn main() {
                base_x = 0x0100;
                base_y = 0x6000;
                face_left = 0;
                let mut mx: u16 = base_x + 32;
                let mut v: i16 = 0x0200;
                if face_left != 0 {
                    mx = base_x;
                    v = 0 - 0x0200;
                }
                spawn(1, mx, base_y + (7 << 8), v, 0, 40, 0);
            }
        ''', ExpectedState(memory={
            0x7E0036: [0x00, 0x67],      # y = 0x6000 + 0x0700, high byte intact
            0x7E0038: [0x00, 0x02],      # vx
        }))
        assert result.success, f"Failures: {result.failures}"
