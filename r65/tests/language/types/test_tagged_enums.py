# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for tagged enum declaration and layout.

A tagged enum — one with at least one payload variant — is stored the way an
`#[soa]` struct is: split into planes, one static per type, no contiguous
element. These tests pin the split itself (a tag plane and a payload plane, with
the tag first so a tag read needs no index scaling) and the restrictions that
follow from an element having no address.

See docs/enums.md.
"""

import pytest

from r65.compiler.errors import HIRError
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.hir.types import StructTypeInfo


SHAPES = """
struct Circle { r: u8 }
struct Rect { w: u8, h: u8 }
enum Shape { Empty, C(Circle), R(Rect) }
"""

POOL = SHAPES + """
#[lowram]
static mut POOL: [Shape; 8];
"""


def build(source: str):
    return HIRBuilder(source_file="test.r65").build_program(parse(source, "test.r65"))


def build_fails(source: str, match: str):
    with pytest.raises(HIRError, match=match):
        build(source)


def statics(hir_prog):
    return {d.name: d for d in hir_prog.statics}


class TestDeclaration:
    """What makes an enum tagged, and what a payload may be."""

    def test_payload_variant_makes_the_enum_tagged(self):
        enum = build(SHAPES + "fn main() {}").symbol_table.lookup("Shape").definition
        assert enum.is_tagged
        assert [(v.name, v.value) for v in enum.variants] == [
            ("Empty", 0), ("C", 1), ("R", 2)]

    def test_fieldless_enum_is_not_tagged(self):
        enum = build("enum Dir { North, East }\nfn main() {}"
                     ).symbol_table.lookup("Dir").definition
        assert not enum.is_tagged
        assert enum.payload_union is None

    def test_unit_variant_carries_no_payload(self):
        enum = build(SHAPES + "fn main() {}").symbol_table.lookup("Shape").definition
        by_name = {v.name: v for v in enum.variants}
        assert by_name["Empty"].payload_type is None
        assert by_name["C"].payload_type.name == "Circle"

    @pytest.mark.parametrize("payload,message", [
        ("u8", r"is 'u8', not a struct"),
        ("bool", r"is 'bool', not a struct"),
        ("Missing", r"names undefined type 'Missing'"),
    ])
    def test_payload_must_be_a_declared_struct(self, payload, message):
        build_fails(f"enum E {{ X({payload}) }}\nfn main() {{}}", message)

    def test_union_payload_rejected(self):
        build_fails("union U { a: u8 }\nenum E { X(U) }\nfn main() {}",
                    r"is a union, not a struct")

    def test_newtype_payload_rejected(self):
        build_fails("struct N(u8);\nenum E { X(N) }\nfn main() {}",
                    r"is a newtype, not a struct")

    def test_enum_payload_rejected(self):
        build_fails("enum Inner { A }\nenum E { X(Inner) }\nfn main() {}",
                    r"is an enum, not a struct")

    def test_soa_payload_rejected(self):
        """An #[soa] element's bytes are spread across planes, so it cannot sit
        contiguously inside another plane."""
        build_fails(
            "#[soa]\nstruct Sp { a: u8 }\n"
            "#[lowram]\nstatic mut SPOOL: [Sp; 4];\n"
            "enum E { X(Sp) }\nfn main() {}",
            r"is #\[soa\]")

    def test_explicit_discriminant_rejected(self):
        """A tagged enum numbers densely from 0 so the tag indexes a jump table."""
        build_fails("struct C { r: u8 }\nenum E { A = 5, X(C) }\nfn main() {}",
                    r"cannot have an explicit discriminant")

    def test_fieldless_enum_keeps_explicit_discriminants(self):
        enum = build("enum E { A = 5, B, C = 10 }\nfn main() {}"
                     ).symbol_table.lookup("E").definition
        assert [(v.name, v.value) for v in enum.variants] == [
            ("A", 5), ("B", 6), ("C", 10)]


class TestPayloadUnion:
    """The synthetic union that gives the data plane its element type."""

    def test_every_payload_is_a_field_at_offset_zero(self):
        enum = build(SHAPES + "fn main() {}").symbol_table.lookup("Shape").definition
        union = enum.payload_union
        assert union.name == "Shape__Payload"
        assert union.is_union
        assert [(f.name, f.offset) for f in union.fields] == [("C", 0), ("R", 0)]

    def test_size_is_the_largest_payload(self):
        enum = build(SHAPES + "fn main() {}").symbol_table.lookup("Shape").definition
        union = enum.payload_union
        assert StructTypeInfo(name=union.name, definition=union).size_bytes == 2

    def test_three_byte_payload_pads_to_a_power_of_two(self):
        """A 3-byte element scales by two ASLs; 3 itself needs a shift-and-add.
        The same rounding `_soa_stride` applies to an #[soa] field plane."""
        enum = build(
            "struct Three { a: u8, b: u8, c: u8 }\n"
            "enum E { Empty, T(Three) }\nfn main() {}"
        ).symbol_table.lookup("E").definition
        union = enum.payload_union
        assert "__pad" in [f.name for f in union.fields]
        assert StructTypeInfo(name=union.name, definition=union).size_bytes == 4

    def test_no_padding_when_already_a_good_stride(self):
        enum = build(SHAPES + "fn main() {}").symbol_table.lookup("Shape").definition
        assert "__pad" not in [f.name for f in enum.payload_union.fields]


class TestPlanes:
    """Storage splits into a tag plane and a payload plane."""

    def test_two_planes_tag_first(self):
        parent = statics(build(POOL + "fn main() {}"))["POOL"]
        assert [(name, offset, stride, child.name)
                for name, offset, stride, child in parent.soa_planes] == [
            ("tag", 0, 1, "POOL__tag"),
            ("data", 1, 2, "POOL__data"),
        ]

    def test_tag_plane_is_one_byte_per_element(self):
        """Stride 1 is the point: `LDA POOL__tag,X` with the raw index in X."""
        tag = statics(build(POOL + "fn main() {}"))["POOL__tag"]
        assert str(tag.var_type) == "[u8; 8]"
        assert tag.soa_stride == 1

    def test_data_plane_strides_by_the_payload_size(self):
        data = statics(build(POOL + "fn main() {}"))["POOL__data"]
        assert str(data.var_type) == "[Shape__Payload; 8]"
        assert data.soa_stride == 2

    def test_single_instance_is_a_pool_of_one(self):
        """`static mut SHAPE: Shape;` wraps to `[Shape; 1]`, as #[soa] does, so
        every downstream path is the array path with index 0."""
        decls = statics(build(SHAPES + "#[lowram]\nstatic mut SHAPE: Shape;\nfn main() {}"))
        assert str(decls["SHAPE"].var_type) == "[Shape; 1]"
        assert str(decls["SHAPE__tag"].var_type) == "[u8; 1]"

    def test_fieldless_enum_static_is_not_planed(self):
        decls = statics(build(
            "enum Dir { North }\n#[lowram]\nstatic mut CURRENT: Dir;\nfn main() {}"))
        assert decls["CURRENT"].soa_planes is None
        assert "CURRENT__tag" not in decls


class TestStorageRestrictions:
    """One static per type, and no storage form that leaves planes unplaced."""

    def test_second_static_rejected(self):
        build_fails(POOL + "#[lowram]\nstatic mut OTHER: [Shape; 2];\nfn main() {}",
                    r"tagged enum 'Shape' is already stored by static 'POOL'")

    def test_must_be_mutable(self):
        build_fails(SHAPES + "#[lowram]\nstatic POOL: [Shape; 4];\nfn main() {}",
                    r"must be `static mut`")

    def test_initializer_rejected(self):
        build_fails(
            SHAPES + "#[lowram]\nstatic mut POOL: [Shape; 1] = [Shape::Empty; 1];\n"
            "fn main() {}",
            r"cannot have an initializer yet")

    def test_hw_storage_rejected(self):
        build_fails(SHAPES + "#[hw(0x2100)]\nstatic mut POOL: [Shape; 4];\nfn main() {}",
                    r"cannot use #\[hw\] storage")


class TestNoContiguousElement:
    """A tagged enum value exists only as a pool element."""

    def test_local_rejected(self):
        build_fails(POOL + "fn main() { let s: Shape = Shape::Empty; }",
                    r"local 's' cannot have tagged enum 'Shape'")

    def test_parameter_rejected(self):
        build_fails(POOL + "fn f(s: Shape) {}\nfn main() {}",
                    r"parameter 's' cannot have tagged enum 'Shape'")

    def test_struct_field_rejected(self):
        build_fails(POOL + "struct Holder { s: Shape }\nfn main() {}",
                    r"field 's' cannot have tagged enum 'Shape'")

    def test_declared_after_the_code_that_uses_it(self):
        """Source order must not matter.

        Whether a value is a tagged enum is answered from state filled before
        any function body is built. Reading the enum's `definition` instead would
        still see the AST node here, and the match would silently fail to
        recognise its own scrutinee.
        """
        build("""
            fn area(i: u16) -> u8 {
                return match POOL[i] { Shape::C(c) => c.r, Shape::Empty => 0 };
            }
            fn spawn(i: u16) { POOL[i] = Shape::C(Circle { r: 7 }); }

            struct Circle { r: u8 }
            enum Shape { Empty, C(Circle) }
            #[lowram]
            static mut POOL: [Shape; 4];
            fn main() {}
        """)

    def test_fieldless_enum_is_unaffected(self):
        """The restriction is on planed types, so an ordinary enum still works
        as a local, a parameter and a field."""
        build("enum Dir { North, East }\n"
              "struct Holder { d: Dir }\n"
              "fn f(d: Dir) {}\n"
              "fn main() { let x: Dir = Dir::North; }")


class TestConstruction:
    """`POOL[i] = Shape::C(Circle { r: 5 });` writes the planes in place."""

    def _asm(self, body: str) -> str:
        from r65.compiler.main import compile_string
        return compile_string(POOL + body + "\nfn main() {}", "test.r65")

    def _fn(self, body: str, label: str) -> str:
        asm = self._asm(body)
        return asm[asm.index(f"\n{label}:"):].split("; ---")[0]

    def test_payload_variant_writes_tag_then_payload(self):
        """No temporary and no block copy: one store for the tag, one per
        payload field."""
        code = self._fn("fn spawn(i: u16) { POOL[i] = Shape::C(Circle { r: 5 }); }",
                        "spawn")
        assert "LDA #$01" in code and "STA $0200,X" in code, code   # tag
        assert "LDA #$05" in code and "STA $0208,X" in code, code   # payload

    def test_unit_variant_writes_only_the_tag(self):
        code = self._fn("fn clear(i: u16) { POOL[i] = Shape::Empty; }", "clear")
        assert "$0200,X" in code, code
        assert "$0208,X" not in code, code

    def test_multi_field_payload(self):
        build(POOL + "fn f(i: u16) { POOL[i] = Shape::R(Rect { w: 1, h: 2 }); }\n"
                     "fn main() {}")

    def test_payload_on_a_unit_variant_rejected(self):
        build_fails(POOL + "fn f(i: u16) { POOL[i] = Shape::Empty(Circle { r: 1 }); }\n"
                           "fn main() {}",
                    r"variant 'Shape::Empty' takes no payload")

    def test_missing_payload_rejected(self):
        build_fails(POOL + "fn f(i: u16) { POOL[i] = Shape::C; }\nfn main() {}",
                    r"variant 'Shape::C' needs a payload")

    def test_two_payloads_rejected(self):
        build_fails(
            POOL + "fn f(i: u16) { POOL[i] = Shape::C(Circle { r: 1 }, Circle { r: 2 }); }\n"
                   "fn main() {}",
            r"takes one payload, got 2")

    def test_assigning_a_plain_value_rejected(self):
        build_fails(POOL + "fn f(i: u16) { POOL[i] = 5; }\nfn main() {}",
                    r"cannot assign to a 'Shape' element")

    def test_assigning_another_enum_rejected(self):
        build_fails(POOL + "enum Other { A }\nfn f(i: u16) { POOL[i] = Other::A; }\n"
                           "fn main() {}",
                    r"cannot assign a 'Other' to a 'Shape' element")
