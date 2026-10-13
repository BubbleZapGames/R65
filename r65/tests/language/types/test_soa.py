# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for the `#[soa]` struct attribute.

`#[soa]` marks a *struct*: every array of that struct is laid out as one
plane per field, and its methods take `self` as an element index rather
than a pointer. Slice S1 lands the attribute and the restrictions that are
answerable from the declaration alone; the layout flip and index-`self`
methods come later.

See docs/soa.md.
"""

import pytest

from r65.compiler.errors import HIRError
from r65.compiler.errors import ParseError
from r65.tests.language.common import build_hir, parse_program


def build_fails(source: str) -> str:
    """Build HIR expecting an HIRError; return its message."""
    with pytest.raises(HIRError) as exc:
        build_hir(source)
    return str(exc.value)


class TestSoaAccepted:
    def test_marks_the_struct(self):
        program = build_hir("""
            #[soa] struct Enemy { kind: u8, hp: u8 }
            #[lowram] static mut POOL: [Enemy; 4];
        """)
        struct = next(d for d in program.declarations
                      if getattr(d, 'name', None) == 'Enemy')
        assert struct.is_soa

    def test_plain_struct_is_not_soa(self):
        program = build_hir("""
            struct Enemy { kind: u8, hp: u8 }
            #[lowram] static mut POOL: [Enemy; 4];
        """)
        struct = next(d for d in program.declarations
                      if getattr(d, 'name', None) == 'Enemy')
        assert not struct.is_soa

    def test_field_access_still_compiles(self):
        """S1 is inert — an accepted program keeps the array-of-structs layout."""
        build_hir("""
            #[soa] struct Enemy { kind: u8, hp: u8 }
            #[lowram] static mut POOL: [Enemy; 4];
            fn f() { POOL[1].hp = 3; }
        """)


class TestSoaRejected:
    def test_union(self):
        msg = build_fails("#[soa] union Pixel { raw: u16, bytes: [u8; 2] }")
        assert "cannot apply to union 'Pixel'" in msg

    def test_struct_with_no_fields(self):
        msg = build_fails("#[soa] struct Empty { }")
        assert "has no fields" in msg

    def test_on_a_newtype(self):
        msg = build_fails("#[soa] struct TileId(u8);")
        assert "only valid on struct declarations" in msg

    def test_on_a_static(self):
        """The attribute moved to the struct; on a static it is an error."""
        msg = build_fails("""
            struct Enemy { kind: u8, hp: u8 }
            #[soa] #[lowram] static mut POOL: [Enemy; 4];
        """)
        assert "only valid on struct declarations" in msg

    def test_on_a_function(self):
        msg = build_fails("#[soa] fn f() { }")
        assert "only valid on struct declarations" in msg

    def test_takes_no_arguments(self):
        msg = build_fails("#[soa(4)] struct Enemy { kind: u8 }")
        assert "takes no arguments" in msg


class TestTypeDeclarationAttributes:
    """Types accept attributes now, so `#[derive]` is caught in the parser.

    Previously `#[derive(...)] struct P` failed to *parse* and a heuristic
    recovered the message. Now it parses, so the rejection has to be explicit
    — and it can name the type, which the heuristic could only do by regex.
    """

    def test_derive_on_struct_names_the_type(self):
        with pytest.raises(ParseError) as exc:
            parse_program("#[derive(Clone)] struct Point { x: u8 }")
        assert "#[derive(...)] is not supported" in str(exc.value)
        assert "Point" in exc.value.hint

    def test_derive_on_union_names_the_type(self):
        with pytest.raises(ParseError) as exc:
            parse_program("#[derive(Clone)] union Word { a: u8, b: u8 }")
        assert "Word" in exc.value.hint

    def test_derive_on_newtype_names_the_type(self):
        with pytest.raises(ParseError) as exc:
            parse_program("#[derive(Clone)] struct TileId(u8);")
        assert "TileId" in exc.value.hint

    def test_proc_macro_on_struct(self):
        with pytest.raises(ParseError) as exc:
            parse_program("#[proc_macro] struct Point { x: u8 }")
        assert "procedural macros are not supported" in exc.value.hint

    def test_cfg_on_struct_is_honoured(self):
        """A `#[cfg]`d-out struct must actually be excluded, not silently kept."""
        program = build_hir("#[cfg(never)] struct Gone { x: u8 }")
        assert not any(getattr(d, 'name', None) == 'Gone'
                       for d in program.declarations)

    def test_cfg_on_struct_kept_when_true(self):
        program = build_hir("struct Kept { x: u8 }")
        assert any(getattr(d, 'name', None) == 'Kept'
                   for d in program.declarations)

    def test_newtype_arity_error_survives_attributes(self):
        with pytest.raises(ParseError) as exc:
            parse_program("struct Pair(u8, u8);")
        assert "wraps exactly one field" in str(exc.value)


class TestSoaLayout:
    """The layout flip: one plane per field, parent allocated no bytes."""

    def _planes(self, source):
        program = build_hir(source)
        return {d.name: d for d in program.declarations
                if getattr(d, 'is_synthetic', False)}

    def test_one_plane_per_field(self):
        planes = self._planes("""
            #[soa] struct Enemy { kind: u8, hp: u8 }
            #[lowram] static mut POOL: [Enemy; 4];
        """)
        assert set(planes) == {'POOL__kind', 'POOL__hp'}
        assert all(p.var_type.size == 4 for p in planes.values())

    def test_parent_records_its_planes(self):
        program = build_hir("""
            #[soa] struct Enemy { kind: u8, hp: u8 }
            #[lowram] static mut POOL: [Enemy; 4];
        """)
        parent = next(d for d in program.declarations if d.name == 'POOL')
        assert [p.name for p in parent.soa_planes] == ['kind', 'hp']
        assert [p.aos_offset for p in parent.soa_planes] == [0, 1]

    def test_explicit_addresses_are_consecutive(self):
        planes = self._planes("""
            #[soa] struct E { hp: u8, x: u16 }
            #[zeropage(0x40)] static mut POOL: [E; 4];
        """)
        assert planes['POOL__hp'].storage_attr.address == 0x40
        # hp is 4 * 1 byte, so x starts 4 later
        assert planes['POOL__x'].storage_attr.address == 0x44

    def test_three_byte_field_is_padded_to_four(self):
        """One byte of waste per element buys a shift instead of a shift-and-add."""
        planes = self._planes("""
            #[soa] struct Area { name: far *u8, exits: u8 }
            #[zeropage(0x40)] static mut AREAS: [Area; 4];
        """)
        assert planes['AREAS__name'].soa_stride == 4
        assert planes['AREAS__exits'].soa_stride == 1
        # 4 elements * stride 4 = 16, so exits starts at 0x40 + 16
        assert planes['AREAS__exits'].storage_attr.address == 0x50

    def test_five_byte_field_is_not_padded(self):
        """Reaching 8 would cost 3 bytes per element — past the one-byte rule."""
        planes = self._planes("""
            struct Five { a: u8, b: u16, c: u16 }
            #[soa] struct Holder { five: Five, tag: u8 }
            #[lowram] static mut H: [Holder; 4];
        """)
        assert planes['H__five'].soa_stride == 5

    def test_single_instance_is_a_pool_of_one(self):
        planes = self._planes("""
            #[soa] struct Player { hp: u8, lives: u8 }
            #[lowram] static mut PLAYER: Player;
        """)
        assert set(planes) == {'PLAYER__hp', 'PLAYER__lives'}
        assert all(p.var_type.size == 1 for p in planes.values())


class TestSoaSelfRegisterConflict:
    """`self` occupies X for the whole body, so no parameter may bind it.

    An *associated* function has no `self` at all, so the register is free
    and nothing should be rejected.
    """

    POOL = """
        #[soa] struct E { hp: u8 }
        #[lowram] static mut POOL: [E; 4];
    """

    def test_method_param_binding_x_is_rejected(self):
        msg = build_fails(self.POOL + """
            impl E { fn f(self, s @ X: u16) { self.hp = 1; } }
        """)
        assert "binds X, which holds 'self'" in msg

    def test_method_second_param_binding_x_is_rejected(self):
        msg = build_fails(self.POOL + """
            impl E { fn f(self, a @ A: u8, s @ X: u16) { self.hp = a; } }
        """)
        assert "binds X, which holds 'self'" in msg

    def test_associated_fn_may_bind_x(self):
        """No `self` exists, so X is not spoken for."""
        build_hir(self.POOL + """
            impl E { fn spawn(slot @ X: u16) { POOL[slot].hp = 1; } }
        """)

    def test_associated_fn_may_bind_x_as_a_later_param(self):
        """The check must not assume params[0] is `self` when there is none."""
        build_hir(self.POOL + """
            impl E { fn spawn(kind @ A: u8, slot @ X: u16) { POOL[slot].hp = kind; } }
        """)

    def test_hint_names_the_right_feature(self):
        """The hint explains which construct put `self` in that register."""
        with pytest.raises(HIRError) as exc:
            build_hir("""
                struct TileId(u8);
                impl TileId { fn f(self, other @ A: u8) -> u8 { self.0 } }
            """)
        assert "newtype" in exc.value.hint

        with pytest.raises(HIRError) as exc:
            build_hir(self.POOL + """
                impl E { fn f(self, s @ X: u16) { self.hp = 1; } }
            """)
        assert "#[soa] method receives 'self' as an element index" in exc.value.hint
        assert "newtype" not in exc.value.hint
        # X is taken by self, so the hint must not offer it
        assert "A or Y" in exc.value.hint


class TestSoaStorageMustBeMutable:
    """An immutable `#[soa]` static with nothing in it could only ever be
    undefined ROM bytes; with an initializer it is a ROM table per plane."""

    def test_immutable_soa_static_is_rejected(self):
        msg = build_fails("""
            #[soa] struct E { hp: u8 }
            static POOL: [E; 4];
        """)
        assert "must be `static mut`" in msg

    def test_mutable_soa_static_is_accepted(self):
        build_hir("""
            #[soa] struct E { hp: u8 }
            #[lowram] static mut POOL: [E; 4];
        """)

    def test_immutable_soa_static_with_initializer_is_accepted(self):
        build_hir("""
            #[soa] struct E { hp: u8 }
            static POOL: [E; 2] = [E { hp: 1 }, E { hp: 2 }];
        """)


class TestSoaInitializer:
    """An array literal on an `#[soa]` static transposes into one literal per
    plane, so the ordinary array-literal path emits a table (or a block copy)
    per field. The struct-literal checks it would otherwise miss run here."""

    def _planes(self, source):
        program = build_hir(source)
        return {d.name: d for d in program.declarations
                if getattr(d, 'is_synthetic', False)}

    TABLE = """
        #[soa] struct Area { id: u8, exits: u16, flags: u8 }
        static AREAS: [Area; 3] = [
            Area { id: 0, exits: 3, flags: 0x01 },
            Area { id: 1, exits: 1, flags: 0x00 },
            Area { id: 2, exits: 2, flags: 0x04 },
        ];
    """

    def test_each_plane_gets_its_column(self):
        planes = self._planes(self.TABLE)
        ids = planes['AREAS__id'].initializer
        assert [e.value for e in ids.elements] == [0, 1, 2]
        exits = planes['AREAS__exits'].initializer
        assert [e.value for e in exits.elements] == [3, 1, 2]
        flags = planes['AREAS__flags'].initializer
        assert [e.value for e in flags.elements] == [1, 0, 4]

    def test_parent_keeps_no_initializer(self):
        """The parent is allocated no bytes, so nothing may initialize it."""
        program = build_hir(self.TABLE)
        parent = next(d for d in program.declarations
                      if getattr(d, 'name', None) == 'AREAS')
        assert parent.initializer is None

    def test_planes_inherit_rom_storage(self):
        planes = self._planes(self.TABLE)
        assert all(not p.is_mutable and p.storage_attr is None
                   for p in planes.values())

    def test_mutable_static_may_carry_one_too(self):
        planes = self._planes("""
            #[soa] struct E { hp: u8, x: u16 }
            #[lowram] static mut POOL: [E; 2] = [E { hp: 1, x: 10 }, E { hp: 2, x: 20 }];
        """)
        assert [e.value for e in planes['POOL__x'].initializer.elements] == [10, 20]
        assert all(p.is_mutable for p in planes.values())

    def test_element_count_must_match(self):
        msg = build_fails("""
            #[soa] struct E { hp: u8 }
            static POOL: [E; 3] = [E { hp: 1 }, E { hp: 2 }];
        """)
        assert "has 3 element(s) but its initializer has 2" in msg

    def test_fill_is_rejected(self):
        """A struct fill `[E { .. }; N]` does not parse; a scalar fill does,
        and has no per-plane form."""
        msg = build_fails("""
            #[soa] struct E { hp: u8 }
            static POOL: [E; 3] = [0; 3];
        """)
        assert "must be initialized with an array literal" in msg

    def test_element_must_be_the_element_struct(self):
        msg = build_fails("""
            struct Other { hp: u8 }
            #[soa] struct E { hp: u8 }
            static POOL: [E; 1] = [Other { hp: 1 }];
        """)
        assert "element 0" in msg and "`E { .. }` literal" in msg

    def test_unknown_field(self):
        msg = build_fails("""
            #[soa] struct E { hp: u8 }
            static POOL: [E; 1] = [E { hp: 1, mp: 2 }];
        """)
        assert "Unknown field: E.mp" in msg

    def test_missing_field(self):
        msg = build_fails("""
            #[soa] struct E { hp: u8, mp: u8 }
            static POOL: [E; 1] = [E { hp: 1 }];
        """)
        assert "missing field(s) in struct literal: mp" in msg

    def test_duplicate_field(self):
        msg = build_fails("""
            #[soa] struct E { hp: u8 }
            static POOL: [E; 1] = [E { hp: 1, hp: 2 }];
        """)
        assert "field 'hp' initialized multiple times" in msg


class TestSoaFieldAddressRejected:
    """`&POOL[i].field` has no meaningful address: the fields of consecutive
    elements are adjacent, so a pointer to one would walk the plane, not the
    element. It is rejected like `&POOL[i]` — it used to crash in codegen."""

    SOURCE = """
        #[soa] struct Fireball { heat: u8, phase: u8 }
        enum Kind { None, Fireball(Fireball) }
        #[soa] struct Enemy { x: u8, kind: Kind }
        #[lowram] static mut POOL: [Enemy; 8];
    """

    @pytest.mark.parametrize("body", [
        "fn t() { let p: *u8 = &POOL[1].x as *u8; *p = 4; }",
        "fn t() { match POOL[1].kind { Kind::Fireball(f) => { let p: *u8 = &f.heat as *u8; *p = 1; } _ => {} }; }",
        "impl Enemy { fn m(self) { let p: *u8 = &self.x as *u8; *p = 1; } }\nfn t() { POOL[1].m(); }",
    ])
    def test_rejected(self, body):
        from r65.compiler.errors import TypeCheckError
        from r65.compiler.main import compile_string
        with pytest.raises(TypeCheckError,
                           match=r"cannot take the address of a field of #\[soa\] array"):
            compile_string(self.SOURCE + body + "\n#[entry]\nfn main() { t(); }\n", "t.r65")
