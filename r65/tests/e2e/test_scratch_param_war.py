# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for scratch parameter WAR (Write-After-Read) hazard
when an A-resident scratch param's target overlaps another param's source.

Bug: In game_show_princess, calling oam_spr1(princess_x, princess_y,
princess_anim[i] | flip, OAM_PRINCESS) generates code where the chr
scratch param (u16 at $02-$03) is written before the y param's vreg
(allocated at scratch $02) is read. The A-resident chr clobbers y.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestScratchParamWARHazard:
    """Test scratch param WAR hazard with A-resident args."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_array_expr_clobbers_global_param(self, e2e):
        """Array expression as u16 scratch param clobbers u8 param.

        Minimal reproduction of the princess sprite bug pattern:
        - Function takes (u8, u8, u16, u16) with scratch params at $00,$01,$02-$03
        - Caller passes globals for u8 params and array[idx]|mask for u16
        - The u16 expression result in A targets scratch $02-$03
        - If the register allocator puts a u8 global in scratch $02,
          the u16 STA clobbers it before it's read for the u8 param
        """
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[lowram]
            static mut sprite_x: u8;
            #[lowram]
            static mut sprite_y: u8;

            static anim_table: [u16; 4] = [
                0x1E00, 0x1E02, 0x1E04, 0x1E06
            ];

            #[lowram]
            static mut out_x: u8;
            #[lowram]
            static mut out_y: u8;
            #[lowram]
            static mut out_chr_lo: u8;
            #[lowram]
            static mut out_chr_hi: u8;
            #[lowram]
            static mut out_off_lo: u8;
            #[lowram]
            static mut out_off_hi: u8;

            fn draw_sprite(x: u8, y: u8, chr: u16, off: u16) {
                out_x = x;
                out_y = y;
                out_chr_lo = chr as u8;
                out_chr_hi = (chr >> 8) as u8;
                out_off_lo = off as u8;
                out_off_hi = (off >> 8) as u8;
            }

            fn show_sprite(frame: u8) {
                let i: u16 = (frame as u16) << 1;
                draw_sprite(sprite_x, sprite_y, anim_table[i], 0x00D8);
            }

            #[entry]
            fn main() {
                sprite_x = 80;
                sprite_y = 120;
                show_sprite(0);
            }
        ''', ExpectedState(memory={
            0x7E0202: 80,   # out_x = sprite_x
            0x7E0203: 120,  # out_y = sprite_y (BUG: gets clobbered by chr write)
            0x7E0204: 0x00, # out_chr_lo
            0x7E0205: 0x1E, # out_chr_hi
            0x7E0206: 0xD8, # out_off_lo
            0x7E0207: 0x00, # out_off_hi
        }))
        assert result.success, f"Failures: {result.failures}"
