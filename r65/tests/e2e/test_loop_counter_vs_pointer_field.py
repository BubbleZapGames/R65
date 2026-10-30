# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A loop counter in Y does not survive a `p.field` read in the same loop.

`(zp),Y` and `[zp],Y` put the field offset in Y — codegen emits `LDY #offset`
immediately before the access. Loop register promotion classified a loop body
as clobbering X (jump/lookup tables) or both X and Y (block moves, fills), but
had no class for "clobbers Y", so a counter was still pinned to Y across a
pointer field read. The region epilogue faithfully reloaded it:

    PLY  ; Reload Y (region end)
    LDY #$00  ; Load field offset 0     <-- and the counter is gone

The counter then stopped advancing and the loop never terminated. This is the
"pointer field offset" entry in the known Y-clobber miscompiles; it surfaced in
rotor_raiders once unit state was read through a pointer inside the draw loop.

`PTR[i] = i`, where the index the access wants in Y *is* the counter, is
deliberately still promoted — that case is covered by
`TestCounterAsIndirectStoreSource` in tests/compiler/codegen/test_loop_promotion.py.
"""

from r65.tests.e2e import ExpectedState


class TestLoopCounterVersusPointerField:

    def test_loop_terminates_and_visits_every_element(self, e2e):
        result = e2e.run('''
            #[zeropage(0x16, register)]
            static mut SCRATCH0: u16;

            struct Rec { a: u16, b: u16 }

            #[lowram]
            static mut RECS: [Rec; 8];
            #[zeropage(0xC0)]
            static mut TOTAL: u16;
            #[zeropage(0xC2)]
            static mut VISITS: u8;

            fn rec(i: u16) -> *Rec { return &RECS[i] as *Rec; }

            #[entry]
            fn main() {
                for i in 0..8 {
                    RECS[i].a = 1;
                    RECS[i].b = 2;
                }
                TOTAL = 0;
                VISITS = 0;
                for i in 0..8 {
                    let p: *Rec = rec(i);
                    TOTAL = TOTAL + p.a + p.b;
                    VISITS = VISITS + 1;
                }
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0x18, 0x00],     # 8 * (1 + 2)
            0x7E00C2: [0x08],           # every element visited exactly once
        }))
        assert result.success, f"Failures: {result.failures}"


class TestCounterOutlivesItsLoop:
    """A counter still live after its loop meets a pointer store's `LDY #offset`.

    The loop-body check above cannot see this: the store is after the loop,
    but a promoted counter holds its register for its whole life, so the
    `LDY #$00` of `p.a = v` replaced `e` before `p.b = e` read it. F32's
    `from_i16` (exponent counted down by the normalize loop, written through
    `self` after it) had this shape; it was hidden while nested `if`s were
    misread as loops spanning the whole function.
    """

    def test_counter_survives_the_store(self, e2e):
        result = e2e.run('''
            struct Rec { a: u16, b: u16 }

            #[lowram]
            static mut R: Rec;
            #[zeropage(0xC0)]
            static mut OUT: u16;

            fn norm(p: *Rec, start: u16) {
                let mut v: u16 = start;
                let mut e: u16 = 151;
                loop {
                    if (v & 0x4000) != 0 { break; }
                    v = v << 1;
                    e = e - 1;
                }
                p.a = v;
                p.b = e;
            }

            #[entry]
            fn main() {
                norm(&R as *Rec, 1);
                OUT = R.b;
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0x89, 0x00],     # 151 - 14 shifts
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_counter_does_not_share_x_with_a_bound_param(self, e2e):
        """`v` copies a parameter that arrives in X; the counter moved off Y
        (the stores' `LDY`) is pinned to X. `v` then must not stay in X too:
        `e = 151` would be an `LDX` over it."""
        result = e2e.run('''
            struct Rec { a: u16, b: u16 }

            #[lowram(0x0300)]
            static mut R: Rec;

            #[inline(never)]
            fn norm(p: far *Rec, start @ X: u16) {
                let mut v: u16 = start;
                let mut e: u16 = 151;
                loop {
                    if (v & 0x4000) != 0 { break; }
                    v = v + v;
                    e = e - 1;
                }
                p.a = v;
                p.b = e;
            }

            #[entry]
            fn main() { norm(&R, 7); }
        ''', ExpectedState(memory={
            0x7E0300: [0x00, 0x70, 0x8B, 0x00],     # 7 << 12, 151 - 12
        }))
        assert result.success, f"Failures: {result.failures}"
