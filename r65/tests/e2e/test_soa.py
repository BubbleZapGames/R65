# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""End-to-end tests for `#[soa]` struct arrays.

Covers the layout flip (one plane per field) and index-`self` methods,
both at the emitted-assembly level and by running the ROM.

See docs/soa.md.
"""

from r65.tests.e2e import ExpectedState


class TestSoaLayout:
    """Field writes land in per-field planes, not an interleaved array."""

    def test_planes_are_separate_arrays(self, e2e):
        """Two elements of a 2-field pool: planes, not AoS interleaving.

        AoS at 0x10 would be [k0,h0,k1,h1]; SoA is [k0,k1] then [h0,h1].
        """
        result = e2e.run('''
            #[soa] struct Enemy { kind: u8, hp: u8 }

            #[zeropage(0x10)]
            static mut POOL: [Enemy; 2];

            #[entry]
            fn main() {
                POOL[0].kind = 0xA0;
                POOL[1].kind = 0xA1;
                POOL[0].hp = 0xB0;
                POOL[1].hp = 0xB1;
            }
        ''', ExpectedState(memory={
            0x7E0010: [0xA0, 0xA1],   # POOL__kind
            0x7E0012: [0xB0, 0xB1],   # POOL__hp
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_variable_index(self, e2e):
        result = e2e.run('''
            #[soa] struct Enemy { kind: u8, hp: u8 }

            #[zeropage(0x10)]
            static mut POOL: [Enemy; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() {
                i = 3;
                POOL[i].hp = 0x5A;
                POOL[i].kind = 0x77;
            }
        ''', ExpectedState(memory={
            0x7E0013: 0x77,   # POOL__kind[3]
            0x7E0017: 0x5A,   # POOL__hp[3]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_field_widths(self, e2e):
        """A u16 plane strides by 2; the u8 planes either side stride by 1.

        The X-index reuse cache is a single slot keyed by stride, so an
        interleaved u8/u16/u8 sequence must recompute rather than reuse — this
        is the case that reads the wrong element if the key ignores width.
        """
        result = e2e.run('''
            #[soa] struct E { hp: u8, pos: u16, kind: u8 }

            #[zeropage(0x10)]
            static mut POOL: [E; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() {
                i = 2;
                POOL[i].hp = 0x11;
                POOL[i].pos = 0x2233;
                POOL[i].kind = 0x44;
            }
        ''', ExpectedState(memory={
            0x7E0012: 0x11,            # hp plane  (0x10 + 2)
            0x7E0018: [0x33, 0x22],    # pos plane (0x14 + 2*2)
            0x7E001E: 0x44,            # kind plane (0x1C + 2)
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_single_instance(self, e2e):
        """A bare `#[soa]` static is a pool of one."""
        result = e2e.run('''
            #[soa] struct Player { hp: u8, lives: u8 }

            #[zeropage(0x10)]
            static mut PLAYER: Player;

            #[entry]
            fn main() {
                PLAYER.hp = 0x63;
                PLAYER.lives = 0x03;
            }
        ''', ExpectedState(memory={
            0x7E0010: 0x63,
            0x7E0011: 0x03,
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSoaMethods:
    """`self` is an element index in X, so `self.f` is one indexed access."""

    def test_method_mutates_through_the_index(self, e2e):
        result = e2e.run('''
            #[soa] struct Enemy { kind: u8, hp: u8 }

            impl Enemy {
                fn hurt(self) { self.hp = self.hp - 1; }
            }

            #[zeropage(0x10)]
            static mut POOL: [Enemy; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() {
                POOL[1].hp = 10;
                POOL[2].hp = 20;
                i = 1;
                POOL[i].hurt();
                i = 2;
                POOL[i].hurt();
                POOL[i].hurt();
            }
        ''', ExpectedState(memory={
            0x7E0015: 9,    # POOL__hp[1], hurt once
            0x7E0016: 18,   # POOL__hp[2], hurt twice
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_method_calls_another_method_on_self(self, e2e):
        result = e2e.run('''
            #[soa] struct Enemy { hp: u8, dead: u8 }

            impl Enemy {
                fn kill(self) { self.hp = 0; self.mark(); }
                fn mark(self) { self.dead = 1; }
            }

            #[zeropage(0x10)]
            static mut POOL: [Enemy; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() {
                POOL[2].hp = 55;
                i = 2;
                POOL[i].kill();
            }
        ''', ExpectedState(memory={
            0x7E0012: 0,   # hp[2]
            0x7E0016: 1,   # dead[2]
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_method_on_a_single_instance(self, e2e):
        result = e2e.run('''
            #[soa] struct Player { hp: u8 }

            impl Player {
                fn heal(self) { self.hp = self.hp + 5; }
            }

            #[zeropage(0x10)]
            static mut PLAYER: Player;

            #[entry]
            fn main() { PLAYER.hp = 10; PLAYER.heal(); }
        ''', ExpectedState(memory={0x7E0010: 15}))
        assert result.success, f"Failures: {result.failures}"

    def test_method_takes_arguments(self, e2e):
        result = e2e.run('''
            #[soa] struct Enemy { hp: u8 }

            impl Enemy {
                fn damage(self, amount: u8) { self.hp = self.hp - amount; }
            }

            #[zeropage(0x10)]
            static mut POOL: [Enemy; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() { POOL[3].hp = 50; i = 3; POOL[i].damage(8); }
        ''', ExpectedState(memory={0x7E0013: 42}))
        assert result.success, f"Failures: {result.failures}"


class TestSoaInlinedConstantElement:
    """An #[inline(always)] index method called on a constant element is
    folded to absolute accesses (optimize/const_index.py); each element's
    fields must still land in their own place, at every field width."""

    def test_each_element_gets_its_own_fields(self, e2e):
        result = e2e.run('''
            #[soa] struct Air { x: u16, hp: u8, landed: u8 }

            #[zeropage(0x10)]
            static mut P: [Air; 2];

            impl Air {
                #[inline(always)]
                fn set(self, x: u16, hp: u8) { self.x = x; self.hp = hp; self.landed = 0; }
                #[inline(always)]
                fn step(self) {
                    if self.landed == 0 {
                        self.x = self.x + 1;
                        self.hp = self.hp - 1;
                    }
                }
            }

            #[entry]
            fn main() {
                P[0].set(0x1234, 0x50);
                P[1].set(0x5678, 0x60);
                P[1].landed = 1;
                P[0].step();
                P[1].step();
            }
        ''', ExpectedState(memory={
            0x7E0010: [0x35, 0x12, 0x78, 0x56],   # P__x: [0x1235, 0x5678]
            0x7E0014: [0x4F, 0x60],               # P__hp: element 1 landed, untouched
            0x7E0016: [0x00, 0x01],               # P__landed
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSoaNestedFields:
    """A nested aggregate field becomes one plane of that aggregate.

    `peel_field_chain` folds `SPR[i].pos.y` into a single summed offset, so
    the funnel recovers the owning plane by range and keeps the residual as
    the offset inside that plane's element.
    """

    def test_nested_struct_field(self, e2e):
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }
            #[soa] struct Sprite { pos: Pos, tile: u8 }

            #[zeropage(0x10)]
            static mut SPR: [Sprite; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() {
                i = 2;
                SPR[i].pos.x = 0x11;
                SPR[i].pos.y = 0x22;
                SPR[i].tile = 0x33;
            }
        ''', ExpectedState(memory={
            # pos plane is [Pos; 4] at 0x10, stride 2 -> element 2 at 0x14
            0x7E0014: [0x11, 0x22],
            # tile plane starts after 4*2 bytes
            0x7E001A: 0x33,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_nested_field_in_a_method(self, e2e):
        result = e2e.run('''
            struct Pos { x: u8, y: u8 }
            #[soa] struct Sprite { pos: Pos, tile: u8 }

            impl Sprite {
                fn shift(self) { self.pos.x = self.pos.x + 1; }
            }

            #[zeropage(0x10)]
            static mut SPR: [Sprite; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() { SPR[3].pos.x = 5; i = 3; SPR[i].shift(); }
        ''', ExpectedState(memory={0x7E0016: 6}))
        assert result.success, f"Failures: {result.failures}"


class TestSoaControlFlow:
    """The X-index cache must not survive a block switch.

    X holds `index * stride(field)`, which differs per plane width. For an
    array of structs X held `index * sizeof(element)` — the same value for
    every field — so a cache leaking across blocks was harmless. Under SoA
    it is not: an arm entered with a scaled X that then indexes a
    differently-strided plane addresses the wrong element.
    """

    def test_match_arms_do_not_share_a_scaled_index(self, e2e):
        result = e2e.run('''
            #[soa] struct E { a: u16, b: u8 }

            impl E {
                fn set(self) {
                    match self.b {
                        0 => { self.a = 0x1111; }
                        _ => { self.a = 0x2222; }
                    }
                }
            }

            #[zeropage(0x10)]
            static mut P: [E; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() {
                P[2].b = 9;      // non-zero -> takes the `_` arm
                i = 2;
                P[i].set();
            }
        ''', ExpectedState(memory={
            # `a` plane at 0x10, stride 2 -> element 2 at 0x14
            0x7E0014: [0x22, 0x22],
            0x7E0010: [0x00, 0x00],   # element 0 untouched
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_if_arms_do_not_share_a_scaled_index(self, e2e):
        result = e2e.run('''
            #[soa] struct E { a: u16, b: u8 }

            impl E {
                fn set(self) {
                    if self.b == 0 { self.a = 0x1111; } else { self.a = 0x2222; }
                }
            }

            #[zeropage(0x10)]
            static mut P: [E; 4];
            #[zeropage(0xB0)]
            static mut i: u16;

            #[entry]
            fn main() { P[3].b = 1; i = 3; P[i].set(); }
        ''', ExpectedState(memory={
            0x7E0016: [0x22, 0x22],   # a[3] at 0x10 + 3*2
        }))
        assert result.success, f"Failures: {result.failures}"


class TestSoaInitializer:
    """An array literal on an `#[soa]` static becomes one table per plane."""

    def test_rom_table_reads_by_plane(self, e2e):
        """Immutable: each column is a ROM table, read with the raw index
        scaled by the *field* width — 2 for `exits`, 1 for the others."""
        result = e2e.run('''
            #[soa] struct Area { id: u8, exits: u16, flags: u8 }
            static AREAS: [Area; 3] = [
                Area { id: 0x10, exits: 3,      flags: 0x01 },
                Area { id: 0x11, exits: 0x1234, flags: 0x00 },
                Area { id: 0x12, exits: 2,      flags: 0x04 },
            ];
            #[zeropage(0xB0)] static mut i: u16;
            #[zeropage(0xB2)] static mut exits: u16;
            #[zeropage(0xB4)] static mut id: u8;
            #[zeropage(0xB5)] static mut flags: u8;

            #[entry]
            fn main() {
                i = 1;
                exits = AREAS[i].exits;
                id = AREAS[i].id;
                flags = AREAS[2].flags;
            }
        ''', ExpectedState(memory={
            0x7E00B2: [0x34, 0x12],
            0x7E00B4: 0x11,
            0x7E00B5: 0x04,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_padded_plane_is_padded_in_rom(self, e2e):
        """A 3-byte field strides by 4, so its table must too or the index
        walks off the data: element 1 is at +4, not +3."""
        result = e2e.run('''
            static NAME_A: [u8; 1] = [0xAA];
            static NAME_B: [u8; 1] = [0xBB];
            #[soa] struct Area { name: far *u8, id: u8 }
            static AREAS: [Area; 2] = [
                Area { name: &NAME_A as far *u8, id: 1 },
                Area { name: &NAME_B as far *u8, id: 2 },
            ];
            #[zeropage(0xB0)] static mut i: u16;
            #[zeropage(0xB2)] static mut out: u8;

            #[entry]
            fn main() {
                i = 1;
                let p: far *u8 = AREAS[i].name;
                out = *p;
            }
        ''', ExpectedState(memory={0x7E00B2: 0xBB}))
        assert result.success, f"Failures: {result.failures}"

    def test_ram_pool_is_copied_per_plane(self, e2e):
        """Mutable: each column is block-copied into its plane at startup,
        so the planes hold the transposed data before main runs."""
        result = e2e.run('''
            #[soa] struct E { hp: u8, pos: u16 }
            #[zeropage(0x10)]
            static mut POOL: [E; 2] = [E { hp: 0xA0, pos: 0x1122 },
                                       E { hp: 0xA1, pos: 0x3344 }];

            #[entry]
            fn main() { }
        ''', ExpectedState(memory={
            0x7E0010: [0xA0, 0xA1],                 # POOL__hp
            0x7E0012: [0x22, 0x11, 0x44, 0x33],     # POOL__pos
        }))
        assert result.success, f"Failures: {result.failures}"
