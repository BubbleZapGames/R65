# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for Clone (Slice 1).

Verifies, on a real ROM via Mesen-GDB, that:
- an auto (bitwise) struct clone is an independent copy of its source,
- array clone copies every byte,
- in-place `clone_from` between statics copies bytes,
- a manual `clone_from` body actually runs (override semantics).
"""

from r65.tests.e2e import ExpectedState


class TestCloneE2E:
    def test_struct_auto_clone_is_independent(self, e2e):
        """`let b = a.clone()` copies a's bytes; mutating a does not affect b."""
        result = e2e.run('''
            struct Vec2 { x: u8, y: u8 }
            impl Clone for Vec2 {}

            #[zeropage(0x20)]
            static mut RX: u8;
            #[zeropage(0x21)]
            static mut RY: u8;

            #[entry]
            fn main() {
                let mut a = Vec2 { x: 7, y: 9 };
                let b = a.clone();
                a.x = 0;          // mutate source after cloning
                a.y = 0;
                RX = b.x;         // clone must still hold 7
                RY = b.y;         // clone must still hold 9
            }
        ''', ExpectedState(memory={
            0x7E0020: 7,
            0x7E0021: 9,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_array_clone_copies_all_bytes(self, e2e):
        """`let dst = src.clone()` on an array copies every element (built-in)."""
        result = e2e.run('''
            #[zeropage(0x20)]
            static mut R0: u8;
            #[zeropage(0x21)]
            static mut R1: u8;
            #[zeropage(0x22)]
            static mut R3: u8;

            #[entry]
            fn main() {
                let src: [u8; 4] = [3, 6, 9, 12];
                let dst = src.clone();
                R0 = dst[0];
                R1 = dst[1];
                R3 = dst[3];
            }
        ''', ExpectedState(memory={
            0x7E0020: 3,
            0x7E0021: 6,
            0x7E0022: 12,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_clone_from_static_inplace(self, e2e):
        """`dst.clone_from(&src)` between statics copies the representation."""
        result = e2e.run('''
            struct Pair { a: u8, b: u8 }
            impl Clone for Pair {}

            #[lowram] static mut PSRC: Pair;
            #[lowram] static mut PDST: Pair;
            #[zeropage(0x20)] static mut RA: u8;
            #[zeropage(0x21)] static mut RB: u8;

            #[entry]
            fn main() {
                PSRC.a = 55;
                PSRC.b = 66;
                PDST.a = 0;
                PDST.b = 0;
                PDST.clone_from(&PSRC);
                RA = PDST.a;
                RB = PDST.b;
            }
        ''', ExpectedState(memory={
            0x7E0020: 55,
            0x7E0021: 66,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_manual_clone_from_override_runs(self, e2e):
        """A custom `clone_from` body runs instead of a bitwise copy."""
        result = e2e.run('''
            struct Tag { id: u8, flag: u8 }
            impl Clone for Tag {
                fn clone_from(*self, src: *Tag) {
                    self.id = src.id;
                    self.flag = 0;       // override: always clear flag
                }
            }

            #[lowram] static mut MSRC: Tag;
            #[lowram] static mut MDST: Tag;
            #[zeropage(0x20)] static mut RID: u8;
            #[zeropage(0x21)] static mut RFLAG: u8;

            #[entry]
            fn main() {
                MSRC.id = 42;
                MSRC.flag = 1;
                MDST.id = 0;
                MDST.flag = 9;
                MDST.clone_from(&MSRC);
                RID = MDST.id;       // copied -> 42
                RFLAG = MDST.flag;   // override -> 0
            }
        ''', ExpectedState(memory={
            0x7E0020: 42,
            0x7E0021: 0,
        }))
        assert result.success, f"Failures: {result.failures}"
