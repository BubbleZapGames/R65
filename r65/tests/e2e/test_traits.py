# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for trait dynamic dispatch.

Tests that trait method calls correctly dispatch through jump tables
to the right implementation at runtime.
"""

from r65.tests.e2e import ExpectedState


class TestTraitDispatch:
    """Test trait dynamic dispatch compiles and executes correctly."""

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

    def test_uninitialized_static_dispatches(self, e2e):
        """A *dyn target declared without an initializer still gets its TypeId."""
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
            static mut PLAYER: Player;
            #[lowram]
            static mut ENEMY: Enemy;

            #[entry]
            fn main() {
                let p: *dyn Drawable = &ENEMY;
                p.draw();
            }
        ''', ExpectedState(
            memory={0x7E0200: 99, 0x7E0201: 1, 0x7E0203: 2}
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

    def test_collides_direct_loop(self, e2e):
        """Pairwise AABB collision via trait dispatch with type_id check and cast."""
        result = e2e.run('''
            #[lowram]
            static mut RESULT: [u8; 3] = [0, 0, 0];

            struct Rect { x: u8, y: u8, w: u8, h: u8 }

            trait Collidable {
                fn collides(*self, other: *dyn Collidable) -> u8;
            }

            fn collides_with_rect(a: *Rect, b: *Rect) -> u8 {
                if a.x < b.x + b.w {
                    if b.x < a.x + a.w {
                        if a.y < b.y + b.h {
                            if b.y < a.y + a.h {
                                return 1;
                            }
                        }
                    }
                }
                return 0;
            }

            impl Collidable for Rect {
                fn collides(*self, other: *dyn Collidable) -> u8 {
                    if other.type_id() == Rect::TYPE_ID {
                        let me: *Rect = self as *Rect;
                        return collides_with_rect(me, other as *Rect);
                    }
                    return 0;
                }
            }

            #[lowram]
            static mut rects: [Rect; 3] = [
                Rect { x: 10, y: 10, w: 20, h: 20 },
                Rect { x: 25, y: 15, w: 15, h: 10 },
                Rect { x: 50, y: 50, w: 10, h: 10 }
            ];

            #[lowram]
            static mut ptrs: [*dyn Collidable; 3];

            #[entry]
            fn main() {
                ptrs[0] = &rects[0] as *dyn Collidable;
                ptrs[1] = &rects[1] as *dyn Collidable;
                ptrs[2] = &rects[2] as *dyn Collidable;

                for i in 0..ptrs.len() {
                    let pi: *dyn Collidable = ptrs[i];
                    for j in i+1..ptrs.len() {
                        let pj: *dyn Collidable = ptrs[j];
                        if pi.collides(pj) != 0 {
                            RESULT[i] = 1;
                            RESULT[j] = 1;
                            break;
                        }
                    }
                }
            }
        ''', ExpectedState(
            memory={
                0x7E0200: [1, 1, 0],
            }
        ), max_instructions=100000)
        assert result.success, f"Failures: {result.failures}"


class TestFarTraitDispatch:
    """Test trait dynamic dispatch with far functions (JSL/RTL)."""

    def test_far_trait_basic_dispatch(self, e2e):
        """Far trait pointer dispatches to correct implementation via JML trampoline."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }
            struct Enemy { x: u8, y: u8 }

            trait Drawable { far fn draw(*self); }

            impl Drawable for Player {
                far fn draw(*self) { RESULT = 42; }
            }
            impl Drawable for Enemy {
                far fn draw(*self) { RESULT = 99; }
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

    def test_far_trait_second_impl(self, e2e):
        """Far dispatch to second implementor works correctly."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8 }
            struct Enemy { x: u8 }

            trait Drawable { far fn draw(*self); }

            impl Drawable for Player {
                far fn draw(*self) { RESULT = 42; }
            }
            impl Drawable for Enemy {
                far fn draw(*self) { RESULT = 99; }
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

    def test_far_trait_method_reads_self_fields(self, e2e):
        """Far trait method can read fields through self pointer."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }

            trait HasPosition {
                far fn get_x(*self) -> u8;
            }

            impl HasPosition for Player {
                far fn get_x(*self) -> u8 {
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

    def test_far_trait_multiple_methods(self, e2e):
        """Far trait with multiple methods dispatches each correctly."""
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
                far fn get_x(*self) -> u8;
                far fn get_y(*self) -> u8;
            }

            impl Positionable for Entity {
                far fn get_x(*self) -> u8 {
                    return self.x;
                }
                far fn get_y(*self) -> u8 {
                    return self.y;
                }
            }

            #[lowram]
            static mut ENT: Entity = Entity { x: 42, y: 99, hp: 200 };

            #[entry]
            fn main() {
                let e: *dyn Positionable = &ENT;
                RESULT_X = e.get_x();
                RESULT_Y = e.get_y();
            }
        ''', ExpectedState(
            memory={
                0x7E0200: 42,   # RESULT_X = self.x
                0x7E0201: 99,   # RESULT_Y = self.y
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_far_trait_type_id(self, e2e):
        """TypeId works correctly with far traits."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut PLAYER_TID: u8;
            #[lowram]
            static mut ENEMY_TID: u8;

            struct Player { x: u8 }
            struct Enemy { x: u8 }

            trait Drawable { far fn draw(*self); }

            impl Drawable for Player {
                far fn draw(*self) { }
            }
            impl Drawable for Enemy {
                far fn draw(*self) { }
            }

            #[lowram]
            static mut PLAYER: Player = Player { x: 0xAA };
            #[lowram]
            static mut ENEMY: Enemy = Enemy { x: 0xCC };

            #[entry]
            fn main() {
                let p: *dyn Drawable = &PLAYER;
                let e: *dyn Drawable = &ENEMY;
                PLAYER_TID = p.type_id();
                ENEMY_TID = e.type_id();
            }
        ''', ExpectedState(
            memory={
                0x7E0200: 1,  # Player TypeId
                0x7E0201: 2,  # Enemy TypeId
            }
        ))
        assert result.success, f"Failures: {result.failures}"


class TestFarSelfTraitDispatch:
    """Test trait dispatch with far *self (24-bit self pointer).

    When objects are in a different bank (e.g., #[ram] in bank $7E) from the
    code (bank $00), near *self can't reach them. far *self uses a 24-bit
    pointer and the caller sets DBR to the object's bank before dispatch.
    """

    def test_far_self_leaf_method_reads_field(self, e2e):
        """Far self leaf method reads struct field via DBR:Y path.

        Leaf method (no calls, no ROM/HW access) uses fast DBR:Y path:
        caller sets DBR to object's bank, Y to object's address,
        field access via LDA $offset,Y.
        """
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }

            trait HasPosition {
                far fn get_x(far *self) -> u8;
            }

            impl HasPosition for Player {
                far fn get_x(far *self) -> u8 {
                    return self.x;
                }
            }

            #[ram]
            static mut PLAYER: Player = Player { x: 42, y: 99 };

            #[entry]
            fn main() {
                let p: far *dyn HasPosition = &PLAYER;
                RESULT = p.get_x();
            }
        ''', ExpectedState(
            memory={0x7E0200: 42}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_far_self_leaf_method_second_field(self, e2e):
        """Far self leaf method reads second field (non-zero offset)."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }

            trait HasPosition {
                far fn get_y(far *self) -> u8;
            }

            impl HasPosition for Player {
                far fn get_y(far *self) -> u8 {
                    return self.y;
                }
            }

            #[ram]
            static mut PLAYER: Player = Player { x: 42, y: 99 };

            #[entry]
            fn main() {
                let p: far *dyn HasPosition = &PLAYER;
                RESULT = p.get_y();
            }
        ''', ExpectedState(
            memory={0x7E0200: 99}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_far_self_method_with_call(self, e2e):
        """Far self method with function call uses D=S path.

        Method contains a function call, so it needs D=S prologue to
        preserve DBR for the callee while still accessing self fields.
        """
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;
            #[lowram]
            static mut RESULT2: u8;

            struct Player { x: u8, y: u8 }

            fn helper(val @ A: u8) {
                RESULT2 = A;
            }

            trait Drawable {
                far fn draw(far *self);
            }

            impl Drawable for Player {
                far fn draw(far *self) {
                    RESULT = self.x;
                    helper(self.y);
                }
            }

            #[ram]
            static mut PLAYER: Player = Player { x: 55, y: 77 };

            #[entry]
            fn main() {
                let p: far *dyn Drawable = &PLAYER;
                p.draw();
            }
        ''', ExpectedState(
            memory={
                0x7E0200: 55,  # RESULT = self.x
                0x7E0201: 77,  # RESULT2 = self.y (via helper)
            }
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_far_self_dispatch_selects_correct_impl(self, e2e):
        """Far self dispatch routes to correct implementation based on TypeId."""
        result = e2e.run('''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8 }
            struct Enemy { x: u8 }

            trait Identifiable {
                far fn identify(far *self) -> u8;
            }

            impl Identifiable for Player {
                far fn identify(far *self) -> u8 {
                    return self.x;
                }
            }
            impl Identifiable for Enemy {
                far fn identify(far *self) -> u8 {
                    return self.x;
                }
            }

            #[ram]
            static mut PLAYER: Player = Player { x: 10 };
            #[ram]
            static mut ENEMY: Enemy = Enemy { x: 20 };

            #[entry]
            fn main() {
                let p: far *dyn Identifiable = &PLAYER;
                let e: far *dyn Identifiable = &ENEMY;
                RESULT = p.identify();
            }
        ''', ExpectedState(
            memory={0x7E0200: 10}
        ))
        assert result.success, f"Failures: {result.failures}"



class TestTraitInheritance:
    """End-to-end tests for trait inheritance (supertraits)."""

    # Inherited dispatch selects the concrete impl by TypeId. Tested with one
    # trait pointer per program (two simultaneous live trait pointers trip an
    # unrelated, pre-existing dispatch bug).
    _INHERIT_NEAR = '''
        #[zeropage(0x10, register)]
        static mut SCRATCH0: u8;
        #[zeropage(0x12, register)]
        static mut SCRATCH1: u16;

        #[lowram]
        static mut RESULT: u8;

        struct Player {{ x: u8, y: u8 }}
        struct Enemy  {{ x: u8, y: u8 }}

        trait Tag {{ fn tag(*self); }}
        trait Drawable: Tag {{ fn draw(*self); }}

        impl Tag for Player {{ fn tag(*self) {{ RESULT = 42; }} }}
        impl Drawable for Player {{ fn draw(*self) {{ }} }}
        impl Tag for Enemy {{ fn tag(*self) {{ RESULT = 99; }} }}
        impl Drawable for Enemy {{ fn draw(*self) {{ }} }}

        #[lowram]
        static mut PLAYER: Player = Player {{ x: 1, y: 2 }};
        #[lowram]
        static mut ENEMY: Enemy = Enemy {{ x: 3, y: 4 }};

        #[entry]
        fn main() {{
            let d: *dyn Drawable = &{obj};
            d.tag();
        }}
    '''

    def test_inherited_dispatch_selects_first_impl(self, e2e):
        """Inherited Tag::tag on *dyn Drawable dispatches to Player."""
        result = e2e.run(self._INHERIT_NEAR.format(obj="PLAYER"),
                         ExpectedState(memory={0x7E0200: 42}))
        assert result.success, f"Failures: {result.failures}"

    def test_inherited_dispatch_selects_second_impl(self, e2e):
        """Inherited Tag::tag on *dyn Drawable dispatches to Enemy by its TypeId."""
        result = e2e.run(self._INHERIT_NEAR.format(obj="ENEMY"),
                         ExpectedState(memory={0x7E0200: 99}))
        assert result.success, f"Failures: {result.failures}"

    def test_upcast_then_inherited_dispatch(self, e2e):
        """*dyn Sub upcast to *dyn Super still dispatches to the concrete impl."""
        src = '''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }
            struct Enemy  { x: u8, y: u8 }

            trait Tag { fn tag(*self); }
            trait Drawable: Tag { fn draw(*self); }

            impl Tag for Player { fn tag(*self) { RESULT = 42; } }
            impl Drawable for Player { fn draw(*self) { } }
            impl Tag for Enemy { fn tag(*self) { RESULT = 99; } }
            impl Drawable for Enemy { fn draw(*self) { } }

            #[lowram]
            static mut ENEMY: Enemy = Enemy { x: 3, y: 4 };

            #[entry]
            fn main() {
                let d: *dyn Drawable = &ENEMY;
                let t: *dyn Tag = d;   // upcast (no-op)
                t.tag();               // -> Enemy -> 99
            }
        '''
        result = e2e.run(src, ExpectedState(memory={0x7E0200: 99}))
        assert result.success, f"Failures: {result.failures}"

    def test_own_and_inherited_on_same_pointer(self, e2e):
        """Both the subtrait's own method and an inherited method work on one pointer."""
        src = '''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut OWN: u8;
            #[lowram]
            static mut INHERITED: u8;

            struct Player { x: u8, y: u8 }

            trait Tag { fn tag(*self); }
            trait Drawable: Tag { fn draw(*self); }

            impl Tag for Player { fn tag(*self) { INHERITED = 7; } }
            impl Drawable for Player { fn draw(*self) { OWN = 5; } }

            #[lowram]
            static mut PLAYER: Player = Player { x: 1, y: 2 };

            #[entry]
            fn main() {
                let d: *dyn Drawable = &PLAYER;
                d.draw();   // own -> OWN = 5
                d.tag();    // inherited -> INHERITED = 7
            }
        '''
        result = e2e.run(src, ExpectedState(memory={0x7E0200: 5, 0x7E0201: 7}))
        assert result.success, f"Failures: {result.failures}"

    def test_inherited_method_dispatch_far(self, e2e):
        """Inherited dispatch works for a far trait hierarchy (JML trampolines)."""
        src = '''
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;

            #[lowram]
            static mut RESULT: u8;

            struct Player { x: u8, y: u8 }
            struct Enemy  { x: u8, y: u8 }

            trait Tag { far fn tag(*self); }
            trait Drawable: Tag { far fn draw(*self); }

            impl Tag for Player { far fn tag(*self) { RESULT = 42; } }
            impl Drawable for Player { far fn draw(*self) { } }
            impl Tag for Enemy { far fn tag(*self) { RESULT = 99; } }
            impl Drawable for Enemy { far fn draw(*self) { } }

            #[lowram]
            static mut ENEMY: Enemy = Enemy { x: 3, y: 4 };

            #[entry]
            fn main() {
                let d: *dyn Drawable = &ENEMY;
                d.tag();   // inherited far Tag::tag -> Enemy -> 99
            }
        '''
        result = e2e.run(src, ExpectedState(memory={0x7E0200: 99}))
        assert result.success, f"Failures: {result.failures}"

    def test_dispatch_in_loop_preserves_counter(self, e2e):
        """A trait dispatch in a loop body must not corrupt a hardware-pinned loop
        counter. The dispatch clobbers X (TAX into the jump table) and Y (self),
        but the clobber-region analysis treats TraitDispatch as a call and spills
        X/Y around it. depth-1 for-loops pin the counter to X, depth-2 to Y."""
        result = e2e.run('''
            #[zeropage(0x10, register)] static mut SCRATCH0: u8;
            #[zeropage(0x12, register)] static mut SCRATCH1: u16;

            struct Counter { v: u8 }
            trait Tick { fn tick(*self); }
            impl Tick for Counter { fn tick(*self) { self.v = self.v + 1; } }

            #[lowram] static mut OBJ: Counter = Counter { v: 0 };
            #[lowram(0x0280)] static mut BUFX: [u8; 8];
            #[lowram(0x0290)] static mut BUFY: [u8; 8];

            #[entry]
            fn main() {
                let p: *dyn Tick = &OBJ;
                for i in 0..5 {                 // i pinned to X
                    p.tick();
                    BUFX[i] = (i as u8) + 1;
                }
                for a in 0..1 {
                    for j in 0..5 {             // j pinned to Y
                        p.tick();
                        BUFY[j] = (j as u8) + 1;
                    }
                }
            }
        ''', ExpectedState(memory={
            0x7E0280: [1, 2, 3, 4, 5, 0, 0, 0],
            0x7E0290: [1, 2, 3, 4, 5, 0, 0, 0],
        }))
        assert result.success, f"Failures: {result.failures}"


class TestTraitMethodArguments:
    """A dispatched trait method must read its arguments where the dispatcher
    put them.

    `analyze_scratch_params` promoted a callee's stack parameter into a
    zero-page scratch register, but `_update_call_sites` cannot rewrite a
    `TraitDispatch`: the dispatch has no static callee, so it does not know
    which implementation it will land in. The callee's ABI changed without the
    call site — the dispatcher pushed the argument and the callee read a
    scratch address nobody had written, so the method ran on whatever happened
    to be in that register.
    """

    def test_dispatched_method_receives_its_argument(self, e2e):
        result = e2e.run('''
            #[zeropage(0x16, register)]
            static mut SCRATCH0: u16;
            #[zeropage(0x18, register)]
            static mut SCRATCH1: u16;

            trait Pick { fn at(*self, i: u16) -> u16; }

            struct Low { r: u8 }
            struct High { r: u8 }

            static TBL: [u16; 4] = [10, 11, 12, 13];

            impl Pick for Low  { fn at(*self, i: u16) -> u16 { return TBL[i]; } }
            impl Pick for High { fn at(*self, i: u16) -> u16 { return TBL[i] + 100; } }

            static LOW: Low = Low { r: 0 };
            static HIGH: High = High { r: 0 };
            static TAB: [*dyn Pick; 2] = [&LOW as *dyn Pick, &HIGH as *dyn Pick];

            #[zeropage(0x40)]
            static mut OUT: [u16; 2];
            #[zeropage(0x50)]
            static mut k: u16;

            #[entry]
            fn main() {
                k = 0;
                OUT[0] = TAB[k].at(3);      // 13
                k = 1;
                OUT[1] = TAB[k].at(2);      // 112
            }
        ''', ExpectedState(memory={0x7E0040: [0x0D, 0x00, 0x70, 0x00]}))
        assert result.success, f"Failures: {result.failures}"

    def test_argument_survives_an_unrelated_scratch_user(self, e2e):
        """The old bug read whatever the caller had last left in the scratch
        register, so it only showed up when that was not the argument."""
        result = e2e.run('''
            #[zeropage(0x16, register)]
            static mut SCRATCH0: u16;
            #[zeropage(0x18, register)]
            static mut SCRATCH1: u16;

            trait Pick { fn at(*self, i: u16) -> u16; }
            struct Low { r: u8 }
            static TBL: [u16; 4] = [10, 11, 12, 13];
            impl Pick for Low { fn at(*self, i: u16) -> u16 { return TBL[i]; } }
            static LOW: Low = Low { r: 0 };
            static TAB: [*dyn Pick; 1] = [&LOW as *dyn Pick];

            #[zeropage(0x40)]
            static mut OUT: u16;

            fn noise(a: u16) -> u16 { return a + 1; }

            #[entry]
            fn main() {
                let n: u16 = noise(0x0777);     // leaves 0x0777 in a scratch slot
                OUT = TAB[0].at(1) + n;         // 11 + 0x0778
            }
        ''', ExpectedState(memory={0x7E0040: [0x83, 0x07]}))
        assert result.success, f"Failures: {result.failures}"


class TestDirectTraitMethodCall:
    """A trait method called directly takes `self` in Y, like the dispatched one.

    `is_trait_method` means the callee reads its fields off Y — that is the
    trait ABI, near pointer or far. The call lowering only produced the `SELF_Y`
    argument when `*self` was *far*, so a directly called near trait method had
    its receiver pushed on the stack while the callee read Y, and it ran on
    whatever Y happened to hold. Dispatching through `*dyn` hid it, because that
    dispatcher loads Y itself.

    In rotor_raiders this surfaced as a unit's record pointer coming back wrong
    and the ROM reading uninitialised zero page.
    """

    def test_near_self_reaches_the_callee(self, e2e):
        result = e2e.run('''
            #[zeropage(0x16, register)]
            static mut SCRATCH0: u16;

            struct Point { x: u16, y: u16 }
            trait Get { fn get(*self) -> u16; }
            impl Get for Point { fn get(*self) -> u16 { return self.y; } }

            #[lowram]
            static mut P: Point;
            #[zeropage(0x40)]
            static mut OUT: u16;

            #[entry]
            fn main() {
                P.x = 0x1111;
                P.y = 0x2222;
                let q: *Point = &P as *Point;
                OUT = q.get();
            }
        ''', ExpectedState(memory={0x7E0040: [0x22, 0x22]}))
        assert result.success, f"Failures: {result.failures}"

    def test_receiver_is_not_a_stale_y(self, e2e):
        """Y is loaded with something else first, so a caller that forgets to
        set it reads that instead of the receiver."""
        result = e2e.run('''
            #[zeropage(0x16, register)]
            static mut SCRATCH0: u16;

            struct Point { x: u16, y: u16 }
            trait Get { fn get(*self) -> u16; }
            impl Get for Point { fn get(*self) -> u16 { return self.y; } }

            #[lowram]
            static mut P: Point;
            #[zeropage(0x40)]
            static mut OUT: u16;

            #[entry]
            fn main() {
                P.x = 0x1111;
                P.y = 0x3344;
                Y = 0x0700;                 // a plausible-looking wrong address
                let q: *Point = &P as *Point;
                OUT = q.get();
            }
        ''', ExpectedState(memory={0x7E0040: [0x44, 0x33]}))
        assert result.success, f"Failures: {result.failures}"
