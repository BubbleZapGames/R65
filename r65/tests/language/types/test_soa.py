# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for the `#[soa]` struct-array attribute (slice S1).

S1 lands the attribute and its restrictions only. `#[soa]` is otherwise
inert: an accepted program still compiles to the array-of-structs layout,
so these tests assert on acceptance and on diagnostics, not on codegen.

See docs/soa.md.
"""

import pytest

from r65.compiler.errors import HIRError
from r65.tests.language.common import build_hir


def build_fails(source: str) -> str:
    """Build HIR expecting an HIRError; return its message."""
    with pytest.raises(HIRError) as exc:
        build_hir(source)
    return str(exc.value)


class TestSoaAccepted:
    """Shapes `#[soa]` allows."""

    def test_plain_struct_array(self):
        program = build_hir("""
            struct Enemy { kind: u8, hp: u8 }
            #[lowram]
            #[soa]
            static mut POOL: [Enemy; 4];
        """)
        static = next(s for s in program.statics if s.name == "POOL")
        assert static.is_soa
        assert not static.is_synthetic

    def test_attribute_order_does_not_matter(self):
        program = build_hir("""
            struct Enemy { kind: u8, hp: u8 }
            #[soa]
            #[lowram]
            static mut POOL: [Enemy; 4];
        """)
        assert next(s for s in program.statics if s.name == "POOL").is_soa

    def test_absent_attribute_leaves_flag_clear(self):
        program = build_hir("""
            struct Enemy { kind: u8, hp: u8 }
            #[lowram]
            static mut POOL: [Enemy; 4];
        """)
        assert not next(s for s in program.statics if s.name == "POOL").is_soa

    def test_static_dispatch_trait_impl_is_allowed(self):
        """A trait never used as `*dyn` injects no TypeId, so layout is unchanged."""
        program = build_hir("""
            trait Tick { fn tick(*self); }
            struct Enemy { kind: u8, hp: u8 }
            impl Tick for Enemy { fn tick(*self) { self.hp = 0; } }
            #[lowram]
            #[soa]
            static mut POOL: [Enemy; 4];
        """)
        assert next(s for s in program.statics if s.name == "POOL").is_soa

    def test_impl_clone_is_allowed(self):
        """Clone is a lang item; it never populates the TypeId table."""
        program = build_hir("""
            struct Enemy { kind: u8, hp: u8 }
            impl Clone for Enemy {}
            #[lowram]
            #[soa]
            static mut POOL: [Enemy; 4];
        """)
        assert next(s for s in program.statics if s.name == "POOL").is_soa


class TestSoaRejected:
    """Shapes that have no per-field layout, each with its own diagnostic."""

    def test_scalar_static(self):
        msg = build_fails("#[soa] static mut POOL: u8;")
        assert "must be an array of structs" in msg

    def test_non_struct_element(self):
        msg = build_fails("#[soa] static mut POOL: [u8; 4];")
        assert "must have a struct element type" in msg

    def test_union_element(self):
        msg = build_fails("""
            union Pixel { raw: u16, bytes: [u8; 2] }
            #[soa] static mut POOL: [Pixel; 4];
        """)
        assert "cannot have a union element type" in msg

    def test_dyn_used_trait_element(self):
        """A `*dyn` trait puts a TypeId byte at the element's offset 0."""
        msg = build_fails("""
            trait Tick { fn tick(*self); }
            struct Enemy { kind: u8, hp: u8 }
            impl Tick for Enemy { fn tick(*self) { self.hp = 0; } }
            #[lowram] static mut E: Enemy;
            #[soa] static mut POOL: [Enemy; 4];
            fn use_it() { let d: *dyn Tick = &E as *dyn Tick; d.tick(); }
        """)
        assert "implements a trait" in msg

    def test_hw_storage(self):
        msg = build_fails("""
            struct Enemy { kind: u8, hp: u8 }
            #[hw(0x2100)] #[soa] static mut POOL: [Enemy; 4];
        """)
        assert "cannot use #[hw] storage" in msg

    def test_initializer_not_yet_supported(self):
        msg = build_fails("""
            struct Enemy { kind: u8, hp: u8 }
            #[soa] static POOL: [Enemy; 2] = [
                Enemy { kind: 1, hp: 2 },
                Enemy { kind: 3, hp: 4 },
            ];
        """)
        assert "cannot have an initializer" in msg

    def test_extern_static(self):
        msg = build_fails("""
            struct Enemy { kind: u8, hp: u8 }
            #[soa] extern static POOL: [Enemy; 4];
        """)
        assert "cannot have storage attributes" in msg

    def test_on_a_function(self):
        msg = build_fails("#[soa] fn f() { }")
        assert "only valid on statics" in msg

    def test_takes_no_arguments(self):
        msg = build_fails("""
            struct Enemy { kind: u8, hp: u8 }
            #[soa(4)] static mut POOL: [Enemy; 4];
        """)
        assert "takes no arguments" in msg
