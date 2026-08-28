# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for the `power_of_two_arrays` lint rule kind.

The rule measures two different costs and must not conflate them: element
*stride* is what `arr[i]` scales by, *length* is what lets an index wrap with
`& (N - 1)`. An array can be fine on one and not the other.
"""

import pytest

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.lint.rule import LintContext
from r65.compiler.lint.rule_kinds import KINDS
from r65.compiler.lint.rule_kinds.power_of_two_arrays import KIND_NAME, from_config


def lint(source: str, **spec):
    """Run the rule over *source*, returning its diagnostic messages."""
    program = HIRBuilder(source_file="test.r65").build_program(parse(source, "test.r65"))
    full = {"code": "C030", "kind": KIND_NAME, "message": "not a power of two"}
    full.update(spec)
    ctx = LintContext(program=program)
    from_config(full).setup(ctx)
    return [d.message for d in ctx.diagnostics.diagnostics]


SHAPES = """
struct Three { a: u8, b: u8, c: u8 }
struct Four  { a: u8, b: u8, c: u8, d: u8 }
"""


class TestRegistration:
    def test_kind_is_registered(self):
        """Unregistered, the kind is unusable from r65-lint.toml however
        correct the rule itself is."""
        assert KIND_NAME in KINDS


class TestStride:
    def test_odd_stride_flagged(self):
        out = lint(SHAPES + "#[lowram]\nstatic mut POOL: [Three; 16];\nfn main() {}")
        assert len(out) == 1
        assert "'POOL' stride is 3" in out[0]

    def test_power_of_two_stride_clean(self):
        assert lint(SHAPES + "#[lowram]\nstatic mut POOL: [Four; 16];\nfn main() {}") == []

    def test_single_byte_stride_clean(self):
        """1 is 2^0 — a byte array must not be flagged."""
        assert lint("#[lowram]\nstatic mut BUF: [u8; 16];\nfn main() {}") == []

    def test_length_not_checked_by_default(self):
        """`check` defaults to stride, so an awkward length passes silently."""
        assert lint("#[lowram]\nstatic mut BUF: [u8; 40];\nfn main() {}") == []


class TestLength:
    def test_odd_length_flagged(self):
        out = lint("#[lowram]\nstatic mut BUF: [u8; 40];\nfn main() {}", check="length")
        assert len(out) == 1
        assert "'BUF' length is 40" in out[0]

    def test_power_of_two_length_clean(self):
        assert lint("#[lowram]\nstatic mut BUF: [u8; 64];\nfn main() {}",
                    check="length") == []

    def test_stride_not_checked(self):
        assert lint(SHAPES + "#[lowram]\nstatic mut POOL: [Three; 16];\nfn main() {}",
                    check="length") == []


class TestBoth:
    def test_reports_each_axis_separately(self):
        out = lint(SHAPES + "#[lowram]\nstatic mut POOL: [Three; 12];\nfn main() {}",
                   check="both")
        assert len(out) == 2
        assert any("stride is 3" in m for m in out)
        assert any("length is 12" in m for m in out)


class TestMinElements:
    def test_short_array_skipped(self):
        """A one-element array is never indexed by a runtime value."""
        assert lint(SHAPES + "#[lowram]\nstatic mut ONE: [Three; 1];\nfn main() {}") == []

    def test_threshold_is_configurable(self):
        src = SHAPES + "#[lowram]\nstatic mut POOL: [Three; 4];\nfn main() {}"
        assert lint(src) != []
        assert lint(src, min_elements=8) == []


class TestPlanedStatics:
    """`#[soa]` structs and tagged enums split into planes."""

    def test_parent_is_not_measured(self):
        """The parent holds no bytes; measuring its element type would report a
        stride the emitted code never uses."""
        out = lint("""
            #[soa]
            struct Enemy { kind: u8, hp: u8, x: u8 }
            #[lowram]
            static mut POOL: [Enemy; 16];
            fn main() {}
        """, check="both")
        assert not any("'POOL'" in m for m in out), out

    def test_plane_stride_is_the_padded_one(self):
        """A 3-byte field plane is padded to stride 4 by `_soa_stride`, so it is
        clean — reading the element type instead would wrongly flag it."""
        out = lint("""
            struct Vec3 { x: u8, y: u8, z: u8 }
            #[soa]
            struct Body { pos: Vec3, hp: u8 }
            #[lowram]
            static mut BODIES: [Body; 16];
            fn main() {}
        """)
        assert out == [], out

    def test_tagged_enum_payload_plane_padded(self):
        out = lint("""
            struct Three { a: u8, b: u8, c: u8 }
            enum Shape { Empty, T(Three) }
            #[lowram]
            static mut POOL: [Shape; 16];
            fn main() {}
        """, check="both")
        assert out == [], out


class TestConfig:
    def test_rejects_unknown_check(self):
        with pytest.raises(ValueError, match=r"`check` must be one of"):
            lint("fn main() {}", check="size")

    def test_rejects_negative_min_elements(self):
        with pytest.raises(ValueError, match=r"`min_elements` must be >= 0"):
            lint("fn main() {}", min_elements=-1)

    def test_rejects_non_integer_min_elements(self):
        with pytest.raises(ValueError, match=r"`min_elements` must be an integer"):
            lint("fn main() {}", min_elements="4")

    def test_custom_hint_overrides(self):
        program = HIRBuilder(source_file="test.r65").build_program(
            parse(SHAPES + "#[lowram]\nstatic mut POOL: [Three; 16];\nfn main() {}",
                  "test.r65"))
        ctx = LintContext(program=program)
        from_config({"code": "C030", "kind": KIND_NAME, "message": "m",
                     "hint": "pad it"}).setup(ctx)
        assert ctx.diagnostics.diagnostics[0].hint == "pad it"

    def test_severity_error(self):
        from r65.compiler.errors import DiagnosticSeverity
        program = HIRBuilder(source_file="test.r65").build_program(
            parse(SHAPES + "#[lowram]\nstatic mut POOL: [Three; 16];\nfn main() {}",
                  "test.r65"))
        ctx = LintContext(program=program)
        from_config({"code": "C030", "kind": KIND_NAME, "message": "m",
                     "severity": "error"}).setup(ctx)
        assert ctx.diagnostics.diagnostics[0].severity == DiagnosticSeverity.ERROR
