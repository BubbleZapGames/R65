# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for `match` on a tagged enum.

A tagged enum has no contiguous element, so matching one is really matching its
tag plane at an index, and a payload pattern binds a *view* of the payload plane
rather than a copy. Both facts are invisible at the HIR layer once the desugar
has run — which is why the codegen tests here assert on instructions.

See docs/enums.md.
"""

import pytest

from r65.compiler.errors import HIRError, TypeCheckError
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.main import compile_string
from r65.compiler.typeck import TypeChecker


SHAPES = """
struct Circle { r: u8 }
struct Rect { w: u8, h: u8 }
enum Shape { Empty, C(Circle), R(Rect) }
#[lowram]
static mut POOL: [Shape; 4];
"""

AREA = SHAPES + """
fn area(i: u16) -> u8 {
    return match POOL[i] {
        Shape::C(c) => c.r,
        Shape::R(r) => r.w,
        Shape::Empty => 0,
    };
}
"""


def build_and_check(source: str):
    program = parse(source, "test.r65")
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


def fails(source: str, exc, match: str):
    with pytest.raises(exc, match=match):
        build_and_check(source)


def body_of(source: str, label: str) -> str:
    asm = compile_string(source, "test.r65")
    return asm[asm.index(f"\n{label}:"):]


class TestScrutinee:
    """Every spelling that names an element reduces to an index into the planes."""

    def test_pool_element_reads_the_tag_plane(self):
        hir_prog = build_and_check(AREA + "fn main() {}")
        fn = next(f for f in hir_prog.functions if f.name == "area")
        match = fn.body.statements[0].values[0]
        assert match.tagged_enum == "Shape"
        assert match.scrutinee.array.name == "POOL__tag"

    def test_single_instance_matches_at_index_zero(self):
        """`static mut SHAPE: Shape;` is a pool of one, so `match SHAPE` is
        `match SHAPE[0]` with no special case below the desugar."""
        build_and_check("""
            struct Circle { r: u8 }
            enum Shape { Empty, C(Circle) }
            #[lowram]
            static mut SHAPE: Shape;
            fn f() -> u8 {
                return match SHAPE { Shape::C(c) => c.r, Shape::Empty => 0 };
            }
            fn main() {}
        """)

    def test_fieldless_enum_match_is_unchanged(self):
        """A fieldless enum is an ordinary value: no planes, no rewrite."""
        hir_prog = build_and_check("""
            enum Dir { North, East }
            fn f(d: Dir) -> u8 {
                return match d { Dir::North => 0, Dir::East => 1 };
            }
            fn main() {}
        """)
        fn = next(f for f in hir_prog.functions if f.name == "f")
        assert fn.body.statements[0].values[0].tagged_enum is None


class TestExhaustiveness:
    """Coverage is checked against the variants, not the tag's integer range."""

    def test_missing_variant_rejected(self):
        fails(SHAPES + """
            fn f(i: u16) -> u8 {
                return match POOL[i] { Shape::C(c) => c.r, Shape::Empty => 0 };
            }
            fn main() {}
        """, TypeCheckError, r"Non-exhaustive match on Shape: missing patterns for R")

    def test_all_variants_covered_needs_no_wildcard(self):
        """Without the tagged-enum branch this would demand a wildcard, since the
        rewritten scrutinee types as a plain u8."""
        build_and_check(AREA + "fn main() {}")

    def test_wildcard_covers_the_rest(self):
        build_and_check(SHAPES + """
            fn f(i: u16) -> u8 {
                return match POOL[i] { Shape::C(c) => c.r, _ => 0 };
            }
            fn main() {}
        """)

    def test_arm_naming_another_enum_rejected(self):
        """Pre-existing gap: an enum pattern was accepted unconditionally, so a
        foreign variant compiled to a comparison against an unrelated number."""
        fails(SHAPES + """
            enum Other { A, B }
            fn f(i: u16) -> u8 {
                return match POOL[i] { Other::A => 1, _ => 0 };
            }
            fn main() {}
        """, TypeCheckError, r"pattern names enum 'Other'.*matched is a 'Shape'")


class TestBindings:
    """A payload binding is a view of the pool."""

    def test_binding_on_a_unit_variant_rejected(self):
        fails(SHAPES + """
            fn f(i: u16) -> u8 {
                return match POOL[i] { Shape::Empty(x) => 0, _ => 1 };
            }
            fn main() {}
        """, HIRError, r"variant 'Shape::Empty' has no payload to bind")

    def test_binding_outside_a_tagged_match_rejected(self):
        fails("""
            enum Dir { North, East }
            fn f(d: Dir) -> u8 {
                return match d { Dir::North(x) => 1, _ => 0 };
            }
            fn main() {}
        """, HIRError, r"binds a payload, but the value being matched is not a "
                       r"tagged enum element")

    def test_binding_inside_an_or_pattern_rejected(self):
        """`Shape::C(c) | Shape::R(r) => c.r` would read Rect's bytes as a
        Circle whenever the arm is reached through R. Two payload variants never
        share a type, so no binding can be valid for both alternatives."""
        fails(SHAPES + """
            fn f(i: u16) -> u8 {
                return match POOL[i] {
                    Shape::C(c) | Shape::R(r) => c.r,
                    Shape::Empty => 0,
                };
            }
            fn main() {}
        """, HIRError, r"binds a payload inside an or-pattern")

    def test_or_pattern_without_bindings_is_fine(self):
        build_and_check(SHAPES + """
            fn f(i: u16) -> u8 {
                return match POOL[i] {
                    Shape::C | Shape::Empty => 1,
                    Shape::R(r) => r.w,
                };
            }
            fn main() {}
        """)

    def test_binding_reads_the_payload_plane(self):
        """`c.r` is one absolute-indexed load off the data plane — the pointer
        the binding stands for is never materialized."""
        body = body_of(AREA + "fn main() {}", "area")
        arm = body.split("CMP #$01")[1].split("CMP #$02")[0]
        assert "LDA $0204,X" in arm, arm

    def test_writing_through_a_binding_writes_the_pool(self):
        """The binding is a view, not a copy: assigning to it stores into the
        payload plane rather than a local that is then discarded."""
        body = body_of(SHAPES + """
            fn bump(i: u16) {
                match POOL[i] { Shape::C(c) => { c.r = 9; } _ => {} }
            }
            fn main() {}
        """, "bump")
        assert "STA $0204,X" in body, body


class TestMatchCodegen:
    """The claim the layout exists for: a tag read needs no index scaling."""

    def test_tag_read_has_no_multiply(self):
        """`LDA POOL__tag,X` with the raw index. An array-of-structs layout would
        need a stride multiply here — that is the whole reason for the split."""
        body = body_of(AREA + "fn main() {}", "area")
        dispatch = body.split("CMP #$01")[0]
        assert "LDA $0200,X" in dispatch, dispatch
        assert "ASL" not in dispatch, dispatch

    def test_payload_read_scales_by_the_payload_stride(self):
        """The data plane strides by the payload size (2 here), so its index
        does scale — independently of the tag's."""
        body = body_of(AREA + "fn main() {}", "area")
        arm = body.split("CMP #$01")[1].split("CMP #$02")[0]
        assert "ASL A" in arm, arm
