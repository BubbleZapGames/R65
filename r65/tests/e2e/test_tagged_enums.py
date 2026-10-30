# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""End-to-end tests for tagged enums.

Runs the ROM. These cover what the assembly-level tests cannot: that the two
planes really are laid out and addressed as intended at runtime, that a payload
binding reads and writes the pool rather than a copy, and that an index `self`
survives a real call.

See docs/enums.md.
"""

from r65.tests.e2e import ExpectedState


SHAPES = '''
    struct Circle { r: u8 }
    struct Rect { w: u8, h: u8 }
    enum Shape { Empty, C(Circle), R(Rect) }

    #[zeropage(0x10)]
    static mut POOL: [Shape; 4];
'''


class TestLayout:
    """Storage splits into a tag plane then a payload plane."""

    def test_tag_and_payload_planes_are_separate(self, e2e):
        """POOL is 4 elements: tags at 0x10..0x13, payloads (2 bytes each) from
        0x14. An array-of-structs layout would interleave them instead.
        """
        result = e2e.run(SHAPES + '''
            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0xB0 });
                POOL[1] = Shape::R(Rect { w: 0xC0, h: 0xC1 });
                POOL[2] = Shape::Empty;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x01, 0x02, 0x00],        # tags: C, R, Empty
            0x7E0014: [0xB0],                    # POOL__data[0].C.r
            0x7E0016: [0xC0, 0xC1],              # POOL__data[1].R.w, .h
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_variable_index(self, e2e):
        result = e2e.run(SHAPES + '''
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() {
                i = 3;
                POOL[i] = Shape::C(Circle { r: 0x5A });
            }
        ''', ExpectedState(memory={
            0x7E0013: [0x01],    # tag of element 3
            0x7E001A: [0x5A],    # payload of element 3
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMatch:
    """Dispatch on the tag, and read the payload through a binding."""

    def test_each_variant_takes_its_own_arm(self, e2e):
        result = e2e.run(SHAPES + '''
            #[zeropage(0xC0)]
            static mut OUT: [u8; 3];

            fn area(i: u16) -> u8 {
                return match POOL[i] {
                    Shape::C(c) => c.r,
                    Shape::R(r) => r.w,
                    Shape::Empty => 0xFF,
                };
            }

            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0x11 });
                POOL[1] = Shape::R(Rect { w: 0x22, h: 0x33 });
                POOL[2] = Shape::Empty;
                OUT[0] = area(0);
                OUT[1] = area(1);
                OUT[2] = area(2);
            }
        ''', ExpectedState(memory={0x7E00C0: [0x11, 0x22, 0xFF]}))
        assert result.success, f"Failures: {result.failures}"

    def test_binding_writes_through_to_the_pool(self, e2e):
        """The binding is a view: assigning to it must change the pool itself,
        not a copy that is then discarded."""
        result = e2e.run(SHAPES + '''
            fn bump(i: u16) {
                match POOL[i] {
                    Shape::C(c) => { c.r = 0x7E; }
                    _ => {}
                }
            }

            #[entry]
            fn main() {
                POOL[1] = Shape::C(Circle { r: 0x01 });
                bump(1);
            }
        ''', ExpectedState(memory={0x7E0016: [0x7E]}))
        assert result.success, f"Failures: {result.failures}"

    def test_binding_keeps_its_element_when_the_index_is_shadowed(self, e2e):
        """A `let i` after the match must not move the binding: it views the
        element that was matched, in a match arm and after a `let ... else`."""
        result = e2e.run(SHAPES + '''
            #[zeropage(0xC0)]
            static mut OUT: [u8; 2];

            fn read(i: u16) {
                match POOL[i] {
                    Shape::C(c) => { let i: u16 = 3; OUT[0] = c.r; }
                    _ => {}
                };
                let Shape::C(c) = POOL[i] else { return; };
                let i: u16 = 3;
                OUT[1] = c.r;
            }

            #[entry]
            fn main() {
                POOL[1] = Shape::C(Circle { r: 0x11 });
                POOL[3] = Shape::C(Circle { r: 0x33 });
                read(1);
            }
        ''', ExpectedState(memory={0x7E00C0: [0x11, 0x11]}))
        assert result.success, f"Failures: {result.failures}"

    def test_second_payload_field(self, e2e):
        result = e2e.run(SHAPES + '''
            #[zeropage(0xC0)]
            static mut OUT: u8;

            #[entry]
            fn main() {
                POOL[0] = Shape::R(Rect { w: 0x40, h: 0x9C });
                OUT = match POOL[0] {
                    Shape::R(r) => r.h,
                    _ => 0,
                };
            }
        ''', ExpectedState(memory={0x7E00C0: [0x9C]}))
        assert result.success, f"Failures: {result.failures}"


class TestMethods:
    """`self` is the element index, passed in X."""

    def test_index_self_method(self, e2e):
        result = e2e.run(SHAPES + '''
            impl Shape {
                fn area(self) -> u8 {
                    return match self {
                        Shape::C(c) => c.r,
                        Shape::R(r) => r.w,
                        Shape::Empty => 0xFF,
                    };
                }
            }

            #[zeropage(0xC0)]
            static mut OUT: [u8; 2];

            #[entry]
            fn main() {
                POOL[0] = Shape::C(Circle { r: 0x44 });
                POOL[3] = Shape::Empty;
                OUT[0] = POOL[0].area();
                OUT[1] = POOL[3].area();
            }
        ''', ExpectedState(memory={0x7E00C0: [0x44, 0xFF]}))
        assert result.success, f"Failures: {result.failures}"

    def test_method_mutates_the_pool(self, e2e):
        result = e2e.run(SHAPES + '''
            impl Shape {
                fn grow(self) {
                    match self {
                        Shape::C(c) => { c.r = c.r + 1; }
                        _ => {}
                    }
                }
            }

            #[entry]
            fn main() {
                POOL[2] = Shape::C(Circle { r: 0x10 });
                POOL[2].grow();
                POOL[2].grow();
            }
        ''', ExpectedState(memory={0x7E0018: [0x12]}))
        assert result.success, f"Failures: {result.failures}"


class TestSingleInstance:
    """A lone static is a pool of one."""

    def test_single_instance_round_trip(self, e2e):
        result = e2e.run('''
            struct Circle { r: u8 }
            enum Shape { Empty, C(Circle) }

            #[zeropage(0x10)]
            static mut SHAPE: Shape;

            #[zeropage(0xC0)]
            static mut OUT: u8;

            #[entry]
            fn main() {
                SHAPE = Shape::C(Circle { r: 0x6B });
                OUT = match SHAPE {
                    Shape::C(c) => c.r,
                    Shape::Empty => 0,
                };
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x01],    # tag plane, one element
            0x7E0011: [0x6B],    # payload plane
            0x7E00C0: [0x6B],
        }))
        assert result.success, f"Failures: {result.failures}"


class TestDiscriminantCast:
    """The tag read back as an integer, and used to index a per-kind table.

    This is the roster pattern: one row of constants per variant, addressed by
    the element's own discriminant, with no dispatch and no per-column match.
    """

    def test_ids_round_trip(self, e2e):
        result = e2e.run(SHAPES + '''
            #[zeropage(0xC0)]
            static mut OUT: [u8; 3];

            #[entry]
            fn main() {
                POOL[0] = Shape::R(Rect { w: 1, h: 2 });
                POOL[1] = Shape::Empty;
                POOL[2] = Shape::C(Circle { r: 3 });
                OUT[0] = POOL[0] as u8;
                OUT[1] = POOL[1] as u8;
                OUT[2] = POOL[2] as u8;
            }
        ''', ExpectedState(memory={0x7E00C0: [0x02, 0x00, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_indexes_a_per_kind_table(self, e2e):
        """A row per variant in ROM, read through the tag. The `Rect` row is at
        index 2 because `Shape::R` is the third variant."""
        result = e2e.run(SHAPES + '''
            struct Row { speed: u16, hp: u8, pad: u8 }
            static ROWS: [Row; 3] = [
                Row { speed: 0x1111, hp: 0x10, pad: 0 },
                Row { speed: 0x2222, hp: 0x20, pad: 0 },
                Row { speed: 0x3333, hp: 0x30, pad: 0 },
            ];

            #[zeropage(0xC0)]
            static mut SPEED: u16;
            #[zeropage(0xC2)]
            static mut HP: u8;

            fn look(i: u16) {
                let k: u16 = POOL[i] as u16;
                SPEED = ROWS[k].speed;
                HP = ROWS[k].hp;
            }

            #[entry]
            fn main() {
                POOL[0] = Shape::Empty;
                POOL[1] = Shape::R(Rect { w: 0, h: 0 });
                look(0);
                look(1);
            }
        ''', ExpectedState(memory={
            0x7E00C0: [0x33, 0x33],    # ROWS[2].speed — the R row
            0x7E00C2: [0x30],
        }))
        assert result.success, f"Failures: {result.failures}"


class TestNestedInSoaStruct:
    """A tagged enum as a field of an `#[soa]` element, on the hardware.

    This is the layout that lets shared state sit beside the union instead of
    inside every variant: `things[i].body.hp` is a plain plane access needing no
    match, while `things[i].shape` still reaches the tag and payload planes.
    """

    SOURCE = '''
        struct Circle { r: u8 }
        struct Rect { w: u8, h: u8 }
        enum Shape { Empty, C(Circle), R(Rect) }

        struct Body { x: u16, hp: u8, pad: u8 }

        #[soa]
        struct Thing { body: Body, shape: Shape }

        #[zeropage(0x10)]
        static mut things: [Thing; 4];
    '''

    def test_both_halves_round_trip(self, e2e):
        result = e2e.run(self.SOURCE + '''
            #[zeropage(0xC0)]
            static mut OUT: [u8; 4];

            #[entry]
            fn main() {
                things[1].body.x = 0x1234;
                things[1].body.hp = 25;
                things[1].shape = Shape::R(Rect { w: 0x40, h: 0x41 });

                things[2].body.hp = 12;
                things[2].shape = Shape::C(Circle { r: 0x5A });

                OUT[0] = things[1].body.hp;              // no match needed
                OUT[1] = match things[1].shape { Shape::R(r) => r.h, _ => 0 };
                OUT[2] = things[2].body.hp;
                OUT[3] = things[2].shape as u8;          // the discriminant
            }
        ''', ExpectedState(memory={0x7E00C0: [25, 0x41, 12, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_a_payload_binding_writes_through_to_the_pool(self, e2e):
        result = e2e.run(self.SOURCE + '''
            #[zeropage(0xC0)]
            static mut OUT: u8;

            fn bump(i: u16) {
                match things[i].shape {
                    Shape::C(c) => { c.r = c.r + 1; }
                    _ => {}
                };
            }

            #[entry]
            fn main() {
                things[3].shape = Shape::C(Circle { r: 0x10 });
                bump(3);
                bump(3);
                OUT = match things[3].shape { Shape::C(c) => c.r, _ => 0 };
            }
        ''', ExpectedState(memory={0x7E00C0: [0x12]}))
        assert result.success, f"Failures: {result.failures}"

    def test_method_over_the_nested_pool(self, e2e):
        result = e2e.run(self.SOURCE + '''
            impl Shape {
                fn area(self) -> u8 {
                    return match self {
                        Shape::C(c) => c.r,
                        Shape::R(r) => r.w,
                        Shape::Empty => 0xFF,
                    };
                }
            }

            #[zeropage(0xC0)]
            static mut OUT: [u8; 2];

            #[entry]
            fn main() {
                things[0].shape = Shape::C(Circle { r: 0x44 });
                things[2].shape = Shape::Empty;
                OUT[0] = things[0].shape.area();
                OUT[1] = things[2].shape.area();
            }
        ''', ExpectedState(memory={0x7E00C0: [0x44, 0xFF]}))
        assert result.success, f"Failures: {result.failures}"


SOA_PAYLOADS = '''
    #[soa] struct Barrel   { rolling: u8 }
    #[soa] struct Fireball { heat: u8, phase: u8 }
    #[soa] struct Boss     { stage: u8, cooldown: u8, target: far *u8 }
    enum Kind { None, Barrel(Barrel), Fireball(Fireball), Spring, Boss(Boss) }

    #[soa]
    struct Enemy { kind: Kind, x: u8 }

    #[zeropage(0x10)]
    static mut POOL: [Enemy; 4];
'''
# Explicit layout, 4 elements. Tag plane at 0x10. The data region starts at
# 0x14 and is sized by Boss: 1 + 1 + 4 (far pointer padded to a stride of 4)
# = 6 bytes per element, 24 bytes, to 0x2B. `x` follows at 0x2C.
#
#   Barrel::rolling    0x14   (region + 0)
#   Fireball::heat     0x14   (region + 0)
#   Fireball::phase    0x18   (region + 4*1)
#   Boss::stage        0x14
#   Boss::cooldown     0x18
#   Boss::target       0x1C   (region + 4*2), stride 4


class TestSoaPayloadLayout:
    """Every variant's field planes overlay the data region from its base."""

    def test_planes_overlay_one_region(self, e2e):
        result = e2e.run(SOA_PAYLOADS + '''
            #[entry]
            fn main() {
                POOL[0].kind = Kind::Fireball(Fireball { heat: 0xA0, phase: 0xA1 });
                POOL[1].kind = Kind::Boss(Boss { stage: 0xB0, cooldown: 0xB1, target: 0x7E1234 as far *u8 });
                POOL[2].kind = Kind::Barrel(Barrel { rolling: 0xC0 });
                POOL[3].kind = Kind::Spring;
                POOL[3].x = 0xD3;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x02, 0x04, 0x01, 0x03],          # tags
            0x7E0014: [0xA0, 0xB0, 0xC0],                # plane 0: heat / stage / rolling
            0x7E0018: [0xA1, 0xB1],                      # plane 1: phase / cooldown
            0x7E0020: [0x34, 0x12, 0x7E],                # Boss::target[1] at 0x1C + 4
            0x7E002F: [0xD3],                            # POOL__x[3]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_rewriting_a_variant_reuses_the_bytes(self, e2e):
        """Morphing element 0 from Fireball to Boss lands the new payload on the
        same bytes — the region is a union, transposed."""
        result = e2e.run(SOA_PAYLOADS + '''
            #[entry]
            fn main() {
                POOL[0].kind = Kind::Fireball(Fireball { heat: 0xA0, phase: 0xA1 });
                POOL[0].kind = Kind::Boss(Boss { stage: 0x55, cooldown: 0x66, target: 0 as far *u8 });
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x04],
            0x7E0014: [0x55],
            0x7E0018: [0x66],
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSoaPayloadAccess:
    """A binding and a payload method both index the planes by the pool index."""

    def test_binding_reads_and_writes_the_planes(self, e2e):
        result = e2e.run(SOA_PAYLOADS + '''
            #[zeropage(0xC0)]
            static mut OUT: u8;

            fn tick(i: u16) {
                match POOL[i].kind {
                    Kind::Fireball(f) => { f.phase = f.phase + 1; OUT = f.heat; }
                    Kind::Boss(b)     => { b.cooldown = b.cooldown - 1; }
                    _ => {}
                };
            }

            #[entry]
            fn main() {
                POOL[1].kind = Kind::Fireball(Fireball { heat: 0x77, phase: 0x10 });
                POOL[2].kind = Kind::Boss(Boss { stage: 1, cooldown: 0x20, target: 0 as far *u8 });
                tick(1);
                tick(2);
            }
        ''', ExpectedState(memory={
            0x7E0019: [0x11],    # Fireball::phase[1]
            0x7E001A: [0x1F],    # Boss::cooldown[2]
            0x7E00C0: [0x77],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_payload_method_and_trait_take_the_pool_index(self, e2e):
        result = e2e.run(SOA_PAYLOADS + '''
            trait Hazard { fn burn(*self) -> u8; }
            impl Hazard for Fireball {
                fn burn(self) -> u8 { self.heat = self.heat + 1; return self.heat; }
            }
            impl Fireball {
                fn flicker(self) { self.phase = self.phase + 2; self.burn(); }
            }

            #[zeropage(0xC0)]
            static mut OUT: u8;

            #[entry]
            fn main() {
                POOL[2].kind = Kind::Fireball(Fireball { heat: 0x30, phase: 0x40 });
                match POOL[2].kind {
                    Kind::Fireball(f) => { OUT = f.burn(); f.flicker(); }
                    _ => {}
                };
            }
        ''', ExpectedState(memory={
            0x7E0016: [0x32],    # heat[2]: two burns
            0x7E001A: [0x42],    # phase[2]: one flicker
            0x7E00C0: [0x31],    # first burn's return
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_standalone_pool(self, e2e):
        """The enum as the pool's own element, not a field of an #[soa] struct."""
        result = e2e.run('''
            #[soa] struct P { a: u8, b: u8 }
            #[soa] struct Q { c: u8 }
            enum K { Z, P(P), Q(Q) }

            #[zeropage(0x10)]
            static mut KS: [K; 4];

            impl P { fn bump(self) { self.b = self.b + self.a; } }

            #[entry]
            fn main() {
                KS[3] = K::P(P { a: 2, b: 5 });
                match KS[3] {
                    K::P(p) => { p.bump(); }
                    _ => {}
                };
            }
        ''', ExpectedState(memory={
            0x7E0013: [0x01],    # tag[3]: K::P
            0x7E0017: [0x02],    # P::a[3] = region 0x14 + 3
            0x7E001B: [0x07],    # P::b[3] = region 0x14 + 4 + 3
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMorph:
    """Rewriting an element as another variant built from its old payload.

    The new payload's stores land on the bytes the old one's operands are read
    from — union offset 0 for plain payloads, the overlaid planes for #[soa] —
    so every operand must be read before the first store.
    """

    def test_plain_payload_reads_before_it_writes(self, e2e):
        result = e2e.run(SHAPES + '''
            #[entry]
            fn main() {
                POOL[1] = Shape::C(Circle { r: 0x42 });
                match POOL[1] {
                    Shape::C(c) => { POOL[1] = Shape::R(Rect { w: 0x11, h: c.r }); }
                    _ => {}
                };
            }
        ''', ExpectedState(memory={
            0x7E0011: [0x02],            # tag[1]: R
            0x7E0016: [0x11, 0x42],      # R.w, R.h = the old C.r
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_soa_payload_swaps_across_planes(self, e2e):
        """Boss::stage overlays Fireball::heat and Boss::cooldown overlays
        Fireball::phase, and each reads the other: one operand needs a
        temporary whichever store goes first."""
        result = e2e.run(SOA_PAYLOADS + '''
            #[entry]
            fn main() {
                POOL[2].kind = Kind::Fireball(Fireball { heat: 0xA0, phase: 0xA1 });
                match POOL[2].kind {
                    Kind::Fireball(f) => {
                        POOL[2].kind = Kind::Boss(Boss { stage: f.phase, cooldown: f.heat,
                                                         target: 0 as far *u8 });
                    }
                    _ => {}
                };
            }
        ''', ExpectedState(memory={
            0x7E0012: [0x04],    # tag[2]: Boss
            0x7E0016: [0xA1],    # Boss::stage[2] = the old phase
            0x7E001A: [0xA0],    # Boss::cooldown[2] = the old heat
        }))
        assert result.success, f"Failures: {result.failures}"


INDEX_ONCE = SOA_PAYLOADS + '''
    #[zeropage(0xC0)]
    static mut CNT: u16;
    #[zeropage(0xC2)]
    static mut OUT: u8;
    #[zeropage(0xC8)]
    static mut ARR: [u8; 4];

    fn next() -> u16 { CNT = CNT + 1; return CNT; }
'''


class TestIndexEvaluatedOnce:
    """A pool index is read once, and a binding stays on the matched element.

    The planes are separate arrays, so an index used for several of them must
    not be re-evaluated per plane: a side effect would run again and each plane
    could land on a different element. `next()` returns 1 on its first call.
    """

    def test_construction(self, e2e):
        result = e2e.run(INDEX_ONCE + '''
            #[entry]
            fn main() {
                CNT = 0;
                POOL[next()].kind = Kind::Fireball(Fireball { heat: 0xA0, phase: 0xA1 });
            }
        ''', ExpectedState(memory={
            0x7E0011: [0x02],          # tag[1]: Fireball
            0x7E0015: [0xA0],          # Fireball::heat[1]
            0x7E0019: [0xA1],          # Fireball::phase[1]
            0x7E00C0: [0x01, 0x00],    # next() ran once
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_match_binding(self, e2e):
        result = e2e.run(INDEX_ONCE + '''
            #[entry]
            fn main() {
                POOL[1].kind = Kind::Fireball(Fireball { heat: 3, phase: 4 });
                POOL[2].kind = Kind::Fireball(Fireball { heat: 10, phase: 20 });
                CNT = 0;
                OUT = match POOL[next()].kind {
                    Kind::Fireball(f) => f.heat + f.phase,
                    _ => 0,
                };
            }
        ''', ExpectedState(memory={
            0x7E00C2: [7],             # both fields from element 1
            0x7E00C0: [0x01, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_binding_survives_index_reassignment(self, e2e):
        result = e2e.run(INDEX_ONCE + '''
            #[entry]
            fn main() {
                POOL[1].kind = Kind::Fireball(Fireball { heat: 3, phase: 4 });
                POOL[2].kind = Kind::Fireball(Fireball { heat: 10, phase: 20 });
                let mut i: u16 = 1;
                match POOL[i].kind {
                    Kind::Fireball(f) => { i = i + 1; OUT = f.heat; }
                    _ => {}
                };
                CNT = i;
            }
        ''', ExpectedState(memory={
            0x7E00C2: [3],             # element 1, the one matched
            0x7E00C0: [0x02, 0x00],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_compound_assignment(self, e2e):
        """Not SoA-specific: `a[f()] += n` evaluates the index once."""
        result = e2e.run(INDEX_ONCE + '''
            #[entry]
            fn main() {
                ARR[1] = 5;
                ARR[2] = 9;
                POOL[1].x = 7;
                CNT = 0;
                ARR[next()] += 1;
                CNT = 0;
                POOL[next()].x += 1;
            }
        ''', ExpectedState(memory={
            0x7E00C9: [6, 9],          # ARR[1] bumped, ARR[2] untouched
            0x7E002D: [8],             # POOL__x[1]
            0x7E00C0: [0x01, 0x00],    # the second next() ran once too
        }))
        assert result.success, f"Failures: {result.failures}"


class TestMixedWidthPayloadsArePrivate:
    """An element's payload bytes belong to that element alone.

    Variants overlay one data region, but a u16 plane strides by 2: laid from
    the region base like the u8 planes, its element i covered the bytes the u8
    planes use for elements 2i and 2i+1. Writing one element's payload then
    rewrote other elements' payloads. Planes are now grouped by stride, so
    only planes of the same stride overlay one another.
    """

    def test_wide_and_narrow_payloads_on_neighbours(self, e2e):
        result = e2e.run('''
            #[soa] struct Spark { heat: u8 }
            #[soa] struct Ember { heat: u8, glow: u8 }
            #[soa] struct Flare { pos: u16 }
            enum Fx { Off, S(Spark), E(Ember), F(Flare) }
            #[zeropage(0x10)]
            static mut FX: [Fx; 4];
            #[zeropage(0x60)]
            static mut OUT: [u8; 4];

            #[entry]
            fn main() {
                FX[1] = Fx::F(Flare { pos: 0x1234 });
                FX[2] = Fx::S(Spark { heat: 5 });
                FX[3] = Fx::E(Ember { heat: 7, glow: 9 });
                OUT[0] = match FX[1] { Fx::F(f) => (f.pos >> 8) as u8, _ => 0 };
                OUT[1] = match FX[1] { Fx::F(f) => f.pos as u8, _ => 0 };
                OUT[2] = match FX[2] { Fx::S(s) => s.heat, _ => 0 };
                OUT[3] = match FX[3] { Fx::E(e) => e.heat + e.glow, _ => 0 };
            }
        ''', ExpectedState(memory={0x7E0060: [0x12, 0x34, 5, 16]}))
        assert result.success, f"Failures: {result.failures}"
