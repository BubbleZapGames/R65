# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for unions.

Layout is unit-tested in language/types/test_unions.py. What can only be checked
by running the code is that the fields genuinely alias the same bytes: a write
through one field has to be visible through another.
"""

from r65.tests.e2e import ExpectedState


class TestUnionsE2E:
    def test_fields_alias_the_same_bytes(self, e2e):
        """Writes through `raw` are readable through `bytes` and vice versa."""
        result = e2e.run('''
            union Pixel { raw: u16, bytes: [u8; 2] }

            #[zeropage(0x20)]
            static mut P: Pixel;
            #[zeropage(0x30)]
            static mut LO: u8;
            #[zeropage(0x31)]
            static mut HI: u8;
            #[zeropage(0x32)]
            static mut ROUNDTRIP: u16;

            #[entry]
            fn main() {
                P.raw = 0x1234;
                LO = P.bytes[0];      // low byte of raw
                HI = P.bytes[1];      // high byte of raw
                P.bytes[1] = 0xFF;    // write back through the other field
                ROUNDTRIP = P.raw;
            }
        ''', ExpectedState(memory={
            0x7E0030: 0x34,
            0x7E0031: 0x12,
            0x7E0032: [0x34, 0xFF],  # 0xFF34 little-endian
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_static_initializer_zero_fills(self, e2e):
        """A one-field initializer clears the rest of the union's bytes."""
        result = e2e.run('''
            union State { flag: u8, buffer: [u8; 4] }

            #[ram]
            static mut SC: State = State { flag: 0xAB };

            #[zeropage(0x20)]
            static mut B0: u8;
            #[zeropage(0x21)]
            static mut B1: u8;
            #[zeropage(0x22)]
            static mut B3: u8;

            #[entry]
            fn main() {
                B0 = SC.buffer[0];
                B1 = SC.buffer[1];
                B3 = SC.buffer[3];
            }
        ''', ExpectedState(memory={
            0x7E0020: 0xAB,  # the initialized field
            0x7E0021: 0x00,  # zero-filled remainder
            0x7E0022: 0x00,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_overlay_reuses_one_region(self, e2e):
        """Two struct views of a shared scratch region occupy the same RAM."""
        result = e2e.run('''
            struct Menu   { cursor: u8, page: u8 }
            struct Battle { hp: u16, mp: u16 }

            // One scratch region, read as either a menu state or a battle state
            union Scene { menu: Menu, battle: Battle }

            #[zeropage(0x40)]
            static mut SCENE: Scene;
            #[zeropage(0x20)]
            static mut CURSOR: u8;
            #[zeropage(0x22)]
            static mut HP: u16;

            #[entry]
            fn main() {
                SCENE.battle.hp = 0x0100;
                HP = SCENE.battle.hp;
                SCENE.menu.cursor = 9;    // same bytes, viewed as the menu
                CURSOR = SCENE.menu.cursor;
            }
        ''', ExpectedState(memory={
            0x7E0020: 9,
            0x7E0022: [0x00, 0x01],
            # menu.cursor and battle.hp's low byte are the same byte
            0x7E0040: 9,
        }))
        assert result.success, f"Failures: {result.failures}"
