# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""End-to-end tests for multi-byte fields read out of an *indexed* aggregate.

Byte `n` of an indexed location is still indexed: `arr,X` offset by 2 is
`arr+2,X`, not `arr+2`. `_offset_location` used to drop the index register when
it built the location for bytes 1..n, so a multi-byte value at `arr[i]` was
assembled from one byte of element `i` and the rest of element 0 — silently, and
only for a non-zero index.

Two things these tests must do to have any value:

1. **Run the ROM.** The bug produced perfectly plausible-looking code; only the
   value proves it.
2. **Index with a runtime value.** A literal index folds to an absolute address
   with no index register, so `ARR[1]` never reaches the bug — the first draft
   of these tests passed with the fix reverted for exactly that reason. The
   index therefore comes from a static written at run time.
"""

from r65.tests.e2e import ExpectedState

# Explicit zeropage addresses here stay below $30: under the FixedStack ABI the
# harness injects stdlib/scratch_regs.r65, which reserves $30-$53. A static
# placed there is silently stomped by scratch traffic, and the ROM runs away
# rather than reporting a wrong value.


class TestFarPointerFieldInArray:
    """A 3-byte `far *T` field is the case that surfaced this."""

    def test_far_pointer_field_read_at_runtime_index(self, e2e):
        """Element 1's pointer must be read whole from element 1.

        With the index dropped, byte 0 came from element 1 and bytes 1-2 from
        element 0, producing a pointer to neither sprite.

        The index walks a loop rather than being assigned a literal: under the
        FixedStack ABI a constant assigned to a promoted scratch is stored with
        `STZ` and never loaded into A, so a literal index makes this fail for an
        unrelated reason (see the note in the module docstring).
        """
        result = e2e.run('''
            struct Sprite { x: u8 }
            struct Ref { p: far *Sprite }

            // Deliberately 0x400 apart: the two pointers must differ in a byte
            // the bug actually drops. Adjacent sprites differ only in the low
            // byte, which is the one byte the buggy code got right — so the
            // corrupted pointer came out correct and the test proved nothing.
            #[ram(0x7E2000)]
            static mut SPR_A: Sprite;
            #[ram(0x7E2400)]
            static mut SPR_B: Sprite;

            #[lowram(0x20)]
            static mut ARR: [Ref; 2];
            #[zeropage(0x10)]
            static mut OUT: [u8; 2];
            #[zeropage(0x14)]
            static mut i: u16;

            #[entry]
            fn main() {
                SPR_A.x = 0xA1;
                SPR_B.x = 0xB2;
                ARR[0].p = &SPR_A;
                ARR[1].p = &SPR_B;
                i = 0;
                while i < 2 {
                    OUT[i] = ARR[i].p.x;
                    i = i + 1;
                }
            }
        ''', ExpectedState(memory={0x7E0010: [0xA1, 0xB2]}))
        assert result.success, f"Failures: {result.failures}"

    def test_far_pointer_field_written_at_runtime_index(self, e2e):
        """The store side indexes every byte too.

        Asserted on the raw bytes rather than by reading back through the
        pointer: a buggy write and a buggy read drop the index the same way, so
        the round trip cancels out and reports success while element 2 holds a
        third of a pointer.
        """
        result = e2e.run('''
            struct Sprite { x: u8 }
            struct Ref { p: far *Sprite }

            #[ram(0x7E2400)]
            static mut SPR_B: Sprite;

            #[lowram(0x20)]
            static mut ARR: [Ref; 3];
            #[zeropage(0x14)]
            static mut idx: u16;

            #[entry]
            fn main() {
                idx = 2;
                ARR[idx].p = &SPR_B;
            }
        ''', ExpectedState(memory={
            # ARR[2] spans $7E0026..$7E0028 and must hold all of $7E2400.
            0x7E0026: [0x00, 0x24, 0x7E],
        }))
        assert result.success, f"Failures: {result.failures}"


class TestTaggedEnumFarPayload:
    """The route that led here: a far pointer inside a tagged enum payload."""

    def test_far_pointer_payload_at_runtime_index(self, e2e):
        result = e2e.run('''
            struct Sprite { x: u8 }
            struct Ref { p: far *Sprite }
            enum Slot { Empty, S(Ref) }

            #[ram(0x7E2000)]
            static mut SPR_A: Sprite;
            #[ram(0x7E2400)]
            static mut SPR_B: Sprite;

            #[lowram(0x20)]
            static mut POOL: [Slot; 3];
            #[zeropage(0x10)]
            static mut OUT: [u8; 2];
            #[zeropage(0x14)]
            static mut idx: u16;

            fn read(i: u16) -> u8 {
                return match POOL[i] {
                    Slot::S(s) => s.p.x,
                    Slot::Empty => 0,
                };
            }

            #[entry]
            fn main() {
                SPR_A.x = 0x7A;
                SPR_B.x = 0x8B;
                POOL[0] = Slot::S(Ref { p: &SPR_A });
                POOL[2] = Slot::S(Ref { p: &SPR_B });
                idx = 0;
                OUT[0] = read(idx);
                idx = 2;
                OUT[1] = read(idx);
            }
        ''', ExpectedState(memory={0x7E0010: [0x7A, 0x8B]}))
        assert result.success, f"Failures: {result.failures}"
