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
        """In a plain array-of-structs the enum would need a contiguous form —
        a tag byte with the payload inline — which puts the element stride
        multiply back on every tag read."""
        build_fails(POOL + "struct Holder { s: Shape }\nfn main() {}",
                    r"field 's' cannot have tagged enum 'Shape'")


class TestNestedInSoaStruct:
    """A tagged enum may be a field of an `#[soa]` struct.

    The outer static splits into planes anyway, so the enum contributes its own
    two rather than needing a contiguous form: `units__kind__tag` keeps its
    stride of 1 and the tag read stays `LDA plane,X` off a raw index. This is
    what lets the common state sit *beside* the union instead of inside every
    variant, which is the layout both Rust and R65 push you toward.
    """

    SOURCE = """
    struct Circle { r: u8 }
    enum Shape { Empty, C(Circle) }

    struct Body { x: u16, hp: u8, pad: u8 }

    #[soa]
    struct Thing { body: Body, shape: Shape }

    #[lowram]
    static mut things: [Thing; 8];
    """

    def test_the_field_is_accepted(self):
        build(self.SOURCE + "fn main() {}")

    def test_the_enum_contributes_two_planes(self):
        prog = build(self.SOURCE + "fn main() {}")
        names = [name for name, _off, _stride, _child
                 in statics(prog)['things'].soa_planes]
        assert names == ['body', 'shape__tag', 'shape__data']

    def test_plane_offsets_stay_ordered_and_disjoint(self):
        """`_soa_plane_for` picks a plane by which range contains the summed
        field offset, so the pair must not sit on top of the field after it."""
        prog = build(self.SOURCE + "fn main() {}")
        offsets = [off for _n, off, _s, _c in statics(prog)['things'].soa_planes]
        assert offsets == sorted(offsets)
        assert len(set(offsets)) == len(offsets)
        assert offsets[1] + 1 == offsets[2]        # tag is one byte, payload next

    def test_tag_plane_strides_by_one(self):
        prog = build(self.SOURCE + "fn main() {}")
        planes = {name: (stride, child)
                  for name, _off, stride, child in statics(prog)['things'].soa_planes}
        assert planes['shape__tag'][0] == 1

    def test_a_standalone_static_as_well_is_rejected(self):
        build_fails(self.SOURCE + "#[lowram] static mut more: [Shape; 4];\nfn main() {}",
                    r"backs both 'Thing.shape' and a static of its own")


class TestNestedAccess:
    """Reaching the nested element: match, construct, cast and method call."""

    def _asm(self, body: str) -> str:
        from r65.compiler.main import compile_string
        return compile_string(TestNestedInSoaStruct.SOURCE + body, "test.r65")

    def test_construction_writes_both_planes(self):
        asm = self._asm("""
            #[zeropage(0x40)] static mut i: u16;
            #[entry]
            fn main() { i = 2; things[i].shape = Shape::C(Circle { r: 7 }); }
        """)
        assert "STA $0220,X" in asm, asm      # things__shape__tag
        assert "STA $0228,X" in asm, asm      # things__shape__data

    def test_match_reads_the_nested_tag_plane(self):
        asm = self._asm("""
            #[zeropage(0x40)] static mut o: u8;
            #[zeropage(0x42)] static mut i: u16;
            #[entry]
            fn main() {
                i = 1;
                things[i].shape = Shape::Empty;
                o = match things[i].shape { Shape::C(c) => c.r, Shape::Empty => 9 };
            }
        """)
        assert "LDA $0220,X" in asm, asm

    def test_the_sibling_field_needs_no_match(self):
        """The whole point: shared state is reachable without discriminating."""
        asm = self._asm("""
            #[zeropage(0x40)] static mut o: u8;
            #[zeropage(0x42)] static mut i: u16;
            #[entry]
            fn main() { i = 3; things[i].body.hp = 25; o = things[i].body.hp; }
        """)
        assert "CMP" not in asm.split("\nmain:")[1].split("__SCMP")[0], asm

    def test_method_takes_the_outer_index(self):
        asm = self._asm("""
            impl Shape {
                fn radius(self) -> u8 {
                    return match self { Shape::C(c) => c.r, Shape::Empty => 0 };
                }
            }
            #[zeropage(0x40)] static mut o: u8;
            #[entry]
            fn main() { things[2].shape = Shape::Empty; o = things[2].shape.radius(); }
        """)
        assert "LDX #$02" in asm, asm
        assert "LDA $0220,X" in asm, asm

    def test_discriminant_cast(self):
        asm = self._asm("""
            #[zeropage(0x40)] static mut o: u16;
            #[entry]
            fn main() { things[0].shape = Shape::Empty; o = things[0].shape as u16; }
        """)
        assert "LDA $0220" in asm, asm

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


class TestDiscriminantCast:
    """`POOL[i] as u8` is the tag plane's byte.

    An element has no contiguous value, so a cast of one is not a conversion —
    it is a read of the discriminant. That makes a tagged enum usable as the
    index into a per-kind table, which is what the roster pattern needs.
    """

    def _fn(self, body: str, label: str, call: str) -> str:
        from r65.compiler.main import compile_string
        asm = compile_string(POOL + body + f"\nfn main() {{ {call} }}", "test.r65")
        return asm[asm.index(f"\n{label}:"):].split("; ---")[0]

    def test_u8_cast_is_a_bare_tag_read(self):
        """One load off the tag plane, and nothing else — no bounds check and no
        lookup table, which is what a `match` would have added."""
        code = self._fn("#[zeropage(0x40)] static mut o: u8;\n"
                        "fn id(i: u16) { o = POOL[i] as u8; }", "id", "id(0);")
        assert "LDA $0200,X" in code, code       # POOL__tag,X
        assert "CMP" not in code, code           # no match-style bounds check
        assert "LUT" not in code, code

    def test_u16_cast_widens_the_tag(self):
        code = self._fn("#[zeropage(0x40)] static mut o: u16;\n"
                        "fn id(i: u16) { o = POOL[i] as u16; }", "id", "id(0);")
        assert "LDA $0200,X" in code, code
        assert "AND #$FF" in code, code          # explicit zero-extension

    def test_indexes_a_per_kind_table(self):
        """The point of the cast: a table row per variant, addressed by the tag.

        Read from the whole listing rather than one function, and give it a real
        `#[entry]`: an uncalled function that reads a ROM array is dropped by
        dead-function elimination, and a called-once one is inlined into main.
        """
        from r65.compiler.main import compile_string
        asm = compile_string(POOL + "static SPEED: [u16; 3] = [0, 128, 192];\n"
                             "#[zeropage(0x40)] static mut o: u16;\n"
                             "fn look(i: u16) { o = SPEED[POOL[i] as u16]; }\n"
                             "#[entry]\nfn main() { look(0); look(1); }", "test.r65")
        assert "LDA $0200,X" in asm, asm
        assert "__SPEED_data" in asm, asm

    def test_cast_inside_a_method(self):
        """`self` is the element index, so the tag still comes from the pool."""
        code = self._fn("#[zeropage(0x40)] static mut o: u8;\n"
                        "impl Shape { fn id(self) -> u8 { return POOL[self] as u8; } }",
                        "Shape__id", "o = POOL[1].id();")
        assert "LDA $0200,X" in code, code

    def test_single_instance_reads_index_zero(self):
        from r65.compiler.main import compile_string
        asm = compile_string(SHAPES + "#[lowram] static mut SHAPE: Shape;\n"
                             "#[zeropage(0x40)] static mut o: u8;\n"
                             "fn main() { o = SHAPE as u8; }", "test.r65")
        assert "LDA $0200\n" in asm or "LDA $0200 " in asm, asm

    def test_non_integer_target_is_still_rejected(self):
        """The cast is a tag read only where a tag is a sensible value; anything
        else keeps the planed-value rejection."""
        from r65.compiler.errors import TypeCheckError
        from r65.compiler.main import compile_string
        with pytest.raises(TypeCheckError,
                           match=r"whole element of tagged enum array 'POOL'"):
            compile_string(POOL + "fn f(i: u16) { let p: *u8 = POOL[i] as *u8; }\n"
                                  "fn main() { f(0); }", "test.r65")

    def test_rejection_names_the_cast(self):
        """The hint for a planed enum cannot talk about fields — it has none."""
        from r65.compiler.errors import TypeCheckError
        from r65.compiler.main import compile_string
        with pytest.raises(TypeCheckError) as exc:
            compile_string(POOL + "#[zeropage(0x40)] static mut o: u8;\n"
                                  "fn f(i: u16) { o = POOL[i]; }\nfn main() { f(0); }",
                           "test.r65")
        assert "POOL[i] as u8" in exc.value.hint
        assert "match POOL[i]" in exc.value.hint
