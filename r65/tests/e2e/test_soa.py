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
            #[zeropage(0x30)]
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
            #[zeropage(0x30)]
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
            #[zeropage(0x30)]
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
            #[zeropage(0x30)]
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
            #[zeropage(0x30)]
            static mut i: u16;

            #[entry]
            fn main() { POOL[3].hp = 50; i = 3; POOL[i].damage(8); }
        ''', ExpectedState(memory={0x7E0013: 42}))
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
            #[zeropage(0x30)]
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
            #[zeropage(0x30)]
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
            #[zeropage(0x30)]
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
            #[zeropage(0x30)]
            static mut i: u16;

            #[entry]
            fn main() { P[3].b = 1; i = 3; P[i].set(); }
        ''', ExpectedState(memory={
            0x7E0016: [0x22, 0x22],   # a[3] at 0x10 + 3*2
        }))
        assert result.success, f"Failures: {result.failures}"
