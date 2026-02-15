"""
End-to-end tests for trait dynamic dispatch.

Tests that trait method calls correctly dispatch through jump tables
to the right implementation at runtime.
"""

import pytest
from r65.tests.e2e import E2ETest, ExpectedState


class TestTraitDispatch:
    """Test trait dynamic dispatch compiles and executes correctly."""

    @pytest.fixture
    def e2e(self):
        return E2ETest()

    def test_basic_trait_dispatch(self, e2e):
        """Trait pointer dispatches to correct implementation."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }
            struct Enemy { x: u8, y: u8, hp: u8 }

            trait Drawable { fn draw(*self); }

            impl Drawable for Player {
                fn draw(*self) { RESULT = 42; }
            }
            impl Drawable for Enemy {
                fn draw(*self) { RESULT = 99; }
            }

            #[lowram]
            static mut PLAYER: Player = Player { x: 10, y: 20 };

            #[entry]
            fn main() {
                let p: *dyn Drawable = &PLAYER;
                p.draw();
            }
        ''', ExpectedState(
            memory={0x7E0200: 42}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_trait_dispatch_second_impl(self, e2e):
        """Dispatch to second implementor works correctly."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8 }
            struct Enemy { x: u8 }

            trait Drawable { fn draw(*self); }

            impl Drawable for Player {
                fn draw(*self) { RESULT = 42; }
            }
            impl Drawable for Enemy {
                fn draw(*self) { RESULT = 99; }
            }

            #[lowram]
            static mut ENEMY: Enemy = Enemy { x: 5 };

            #[entry]
            fn main() {
                let e: *dyn Drawable = &ENEMY;
                e.draw();
            }
        ''', ExpectedState(
            memory={0x7E0200: 99}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_trait_type_id_in_struct(self, e2e):
        """TypeId byte is correctly stored at offset 0 and readable via type_id()."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut PLAYER_TID: u8;
            #[lowram]
            static mut ENEMY_TID: u8;

            struct Player { x: u8, y: u8 }
            struct Enemy { x: u8, y: u8 }

            trait Drawable { fn draw(*self); }

            impl Drawable for Player {
                fn draw(*self) { }
            }
            impl Drawable for Enemy {
                fn draw(*self) { }
            }

            #[lowram]
            static mut PLAYER: Player = Player { x: 0xAA, y: 0xBB };
            #[lowram]
            static mut ENEMY: Enemy = Enemy { x: 0xCC, y: 0xDD };

            #[entry]
            fn main() {
                let p: *dyn Drawable = &PLAYER;
                let e: *dyn Drawable = &ENEMY;
                PLAYER_TID = p.type_id();
                ENEMY_TID = e.type_id();
            }
        ''', ExpectedState(
            memory={
                # type_id() results
                0x7E0200: 1,  # Player TypeId
                0x7E0201: 2,  # Enemy TypeId
                # PLAYER: TypeId=1, x=0xAA, y=0xBB
                0x7E0202: 1, 0x7E0203: 0xAA, 0x7E0204: 0xBB,
                # ENEMY: TypeId=2, x=0xCC, y=0xDD
                0x7E0205: 2, 0x7E0206: 0xCC, 0x7E0207: 0xDD,
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_trait_method_reads_self_fields(self, e2e):
        """Trait method can read fields through self pointer."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }

            trait HasPosition {
                fn get_x(*self) -> u8;
            }

            impl HasPosition for Player {
                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }

            #[lowram]
            static mut PLAYER: Player = Player { x: 77, y: 88 };

            #[entry]
            fn main() {
                let p: *dyn HasPosition = &PLAYER;
                RESULT = p.get_x();
            }
        ''', ExpectedState(
            memory={0x7E0200: 77}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_trait_method_reads_multiple_fields(self, e2e):
        """Trait method can read multiple fields through self pointer via Y-register."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT_X: u8;
            #[lowram]
            static mut RESULT_Y: u8;

            struct Entity { x: u8, y: u8, hp: u8 }

            trait Positionable {
                fn store_pos(*self);
            }

            impl Positionable for Entity {
                fn store_pos(*self) {
                    RESULT_X = self.x;
                    RESULT_Y = self.y;
                }
            }

            #[lowram]
            static mut ENT: Entity = Entity { x: 42, y: 99, hp: 200 };

            #[entry]
            fn main() {
                let e: *dyn Positionable = &ENT;
                e.store_pos();
            }
        ''', ExpectedState(
            memory={
                0x7E0200: 42,   # RESULT_X = self.x
                0x7E0201: 99,   # RESULT_Y = self.y
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_rect_intersection_via_trait_list(self, e2e):
        """Loop through array of *dyn Collidable rects, check pairwise AABB intersections."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            // Results: 1 if rect intersects any other rect, 0 otherwise
            #[lowram]
            static mut RESULTS: [u8; 4] = [0; 4];

            // Scratch globals used by store_bounds
            #[lowram]
            static mut BOUND_X: u8;
            #[lowram]
            static mut BOUND_Y: u8;
            #[lowram]
            static mut BOUND_W: u8;
            #[lowram]
            static mut BOUND_H: u8;

            struct Rect { x: u8, y: u8, w: u8, h: u8 }

            trait Collidable {
                fn store_bounds(*self);
                fn collides(*self) -> u8;
            }

            impl Collidable for Rect {
                fn store_bounds(*self) {
                    BOUND_X = self.x;
                    BOUND_Y = self.y;
                    BOUND_W = self.w;
                    BOUND_H = self.h;
                }

                // AABB intersection: check if self overlaps with BOUND_*
                fn collides(*self) -> u8 {
                    if self.x < BOUND_X + BOUND_W {
                        if BOUND_X < self.x + self.w {
                            if self.y < BOUND_Y + BOUND_H {
                                if BOUND_Y < self.y + self.h {
                                    return 1;
                                }
                            }
                        }
                    }
                    return 0;
                }
            }

            // R0: [10,30) x [10,30) -- overlaps R1 and R3
            #[lowram]
            static mut R0: Rect = Rect { x: 10, y: 10, w: 20, h: 20 };
            // R1: [25,40) x [15,25) -- overlaps R0 only
            #[lowram]
            static mut R1: Rect = Rect { x: 25, y: 15, w: 15, h: 10 };
            // R2: [50,60) x [50,60) -- isolated, no overlaps
            #[lowram]
            static mut R2: Rect = Rect { x: 50, y: 50, w: 10, h: 10 };
            // R3: [15,25) x [15,25) -- overlaps R0 only
            #[lowram]
            static mut R3: Rect = Rect { x: 15, y: 15, w: 10, h: 10 };

            #[lowram]
            static mut RECTS: [*dyn Collidable; 4];

            #[entry]
            fn main() {
                // Initialize trait pointer array
                let p0: *dyn Collidable = &R0;
                let p1: *dyn Collidable = &R1;
                let p2: *dyn Collidable = &R2;
                let p3: *dyn Collidable = &R3;
                RECTS[0] = p0;
                RECTS[1] = p1;
                RECTS[2] = p2;
                RECTS[3] = p3;

                // Check each rect against all others
                for i in 0..4 {
                    // Store rect i bounds into BOUND_* globals
                    let pi: *dyn Collidable = RECTS[i];
                    pi.store_bounds();

                    for j in 0..4 {
                        if i != j {
                            // Check if rect j collides with the stored bounds
                            let pj: *dyn Collidable = RECTS[j];
                            if pj.collides() != 0 {
                                RESULTS[i] = 1;
                            }
                        }
                    }
                }
            }
        ''', ExpectedState(
            memory={
                0x7E0200: [1, 1, 0, 1],  # R0=yes, R1=yes, R2=no, R3=yes
            }
        ), max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"

    def test_collides_direct_loop(self, e2e):
        """Direct collision check without loops - verifies collides method works."""
        result = e2e.run('''
            #[zeropage(0x0, register)]
            static mut RESULT: [u8; 3] = [0,0,0];

            struct Rect { x: u8, y: u8, w: u8, h: u8 }

            trait Collidable {
                fn collides(*self, rect: *Rect) -> u8;
            }

            impl Collidable for Rect {
                fn collides(*self, rect: *Rect) -> u8 {
                    if self.x < *rect.x + *rect.w {
                        if *rect.x < self.x + self.w {
                            if self.y < *rect.y + *rect.h {
                                if rect.y < self.y + self.h {
                                    return 1;
                                }
                            }
                        }
                    }
                    return 0;
                }
            }

            #[lowram]
            static mut rects: [Rect; 3] = [
              // overlaps R1
              Rect { x: 10, y: 10, w: 20, h: 20 },
              // no overlap with R0
              Rect { x: 25, y: 15, w: 15, h: 10 },
              // overlaps R0
              Rect { x: 50, y: 50, w: 10, h: 10 },
            ];


            #[entry]
            fn main() {
                for i in 0..rects.len() {
                    let r = &rects[i]
                    for j + 1 in i..rects.len() {
                        let collides: u8 = r.collides(&rects[j])
                        if collides != 0 {
                             RESULT[i] = 1;
                             break;
                        }
                    }
                }
            }
        ''', ExpectedState(
            memory={
                0x0: 1,
                0x1: 0,
                0x2: 0, 
            }
        ))
        assert result.success, f"Failures: {result.failures}"

