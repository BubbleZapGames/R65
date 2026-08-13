# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for union types.

A union is a struct whose fields all sit at offset 0 and whose size is that of
its largest field. Most of the surface is shared with structs, so these tests
concentrate on the layout difference and the union-specific restrictions.
"""

import pytest

from r65.compiler.errors import TypeCheckError, HIRError
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.tests.language.common import parse_struct


def build_and_check(source: str):
    """Parse, build HIR, and type check source."""
    program = parse(source, "test.r65")
    hir_builder = HIRBuilder(source_file="test.r65")
    hir_prog = hir_builder.build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


def find_decl(hir_prog, name: str):
    """Find a struct/union declaration in the built HIR program."""
    return hir_prog.symbol_table.lookup(name).definition


class TestUnionParsing:
    """Tests for parsing union declarations."""

    def test_basic_union(self):
        union = parse_struct("union Pixel { raw: u16, bytes: [u8; 2] }")
        assert union.name == "Pixel"
        assert union.is_union
        assert [f.name for f in union.fields] == ["raw", "bytes"]

    def test_struct_is_not_a_union(self):
        assert not parse_struct("struct Point { x: u8, y: u8 }").is_union

    def test_union_keyword_does_not_break_identifiers(self):
        """Adding the keyword is backward compatible.

        R65 keywords are contextual: `struct`, `trait` and `enum` all still parse
        as identifiers where no keyword is expected. `union` must behave the same,
        so existing code using `union` as a variable name keeps working.
        """
        program = parse("fn f() { let union: u8 = 1; }", "test.r65")
        assert program.items[0].body.statements[0].name == "union"


class TestUnionLayout:
    """Tests for union field offsets and size."""

    def test_all_fields_at_offset_zero(self):
        hir_prog = build_and_check("""
            union Word { w: u16, b: [u8; 2] }
            #[lowram]
            static mut W: Word;
            #[entry]
            fn main() { W.w = 1; }
        """)
        decl = find_decl(hir_prog, "Word")
        assert decl.is_union
        assert [f.offset for f in decl.fields] == [0, 0]

    def test_size_is_largest_field(self):
        """Union size is max(field sizes), not their sum."""
        hir_prog = build_and_check("""
            struct Vec2 { x: u8, y: u8 }
            union State { flag: u8, coords: Vec2, buffer: [u8; 8] }
            #[lowram]
            static mut ST: State;
            #[lowram]
            static mut AFTER: u8;
            #[entry]
            fn main() { ST.flag = 1; }
        """)
        state = hir_prog.symbol_table.lookup("State")
        from r65.compiler.hir.types import StructTypeInfo
        assert StructTypeInfo(name="State", symbol=state).size_bytes == 8

    def test_union_does_not_displace_following_static(self):
        """The next static starts at union_base + max field size."""
        asm = compile_to_asm("""
            union State { flag: u8, buffer: [u8; 8] }
            #[lowram]
            static mut ST: State;
            #[lowram]
            static mut AFTER: u8;
            #[entry]
            fn main() { ST.flag = 1; AFTER = 2; }
        """)
        base = _defined_address(asm, "ST")
        assert _defined_address(asm, "AFTER") == base + 8

    def test_array_of_unions_strides_by_union_size(self):
        hir_prog = build_and_check("""
            union Word { w: u16, b: [u8; 2] }
            #[lowram]
            static mut ARR: [Word; 4];
            #[entry]
            fn main() { ARR[2].w = 1; }
        """)
        from r65.compiler.hir.types import StructTypeInfo
        sym = hir_prog.symbol_table.lookup("Word")
        assert StructTypeInfo(name="Word", symbol=sym).size_bytes == 2


class TestUnionLiterals:
    """Tests for union literal initialization."""

    def test_single_field_literal_accepted(self):
        build_and_check("""
            union Pixel { raw: u16, bytes: [u8; 2] }
            #[lowram]
            static mut P: Pixel = Pixel { raw: 0x1234 };
            #[entry]
            fn main() { P.raw = 1; }
        """)

    def test_multi_field_literal_rejected(self):
        with pytest.raises(TypeCheckError, match="exactly one field"):
            build_and_check("""
                union Pixel { raw: u16, bytes: [u8; 2] }
                #[lowram]
                static mut P: Pixel = Pixel { raw: 0, bytes: [1, 1] };
                #[entry]
                fn main() { P.raw = 1; }
            """)

    def test_struct_literal_still_requires_all_fields(self):
        """The relaxed rule must not leak into structs."""
        with pytest.raises(TypeCheckError, match="missing field"):
            build_and_check("""
                struct Point { x: u8, y: u8 }
                #[lowram]
                static mut P: Point = Point { x: 1 };
                #[entry]
                fn main() { P.x = 1; }
            """)

    def test_literal_zero_fills_to_union_size(self):
        """A 1-byte initializer for an 8-byte union emits 8 bytes of ROM data."""
        asm = compile_to_asm("""
            union State { flag: u8, buffer: [u8; 8] }
            #[lowram]
            static mut ST: State = State { flag: 0xAB };
            #[entry]
            fn main() { ST.flag = 1; }
        """)
        data = _rom_data_bytes(asm, "__ST_data")
        assert data == ["$AB", "$00", "$00", "$00", "$00", "$00", "$00", "$00"]


class TestUnionRestrictions:
    """Tests for what unions may not do."""

    def test_empty_union_rejected(self):
        with pytest.raises(HIRError, match="at least one field"):
            build_and_check("union Empty { }")

    def test_trait_impl_rejected(self):
        with pytest.raises(HIRError, match="cannot implement trait"):
            build_and_check("""
                union Pixel { raw: u16, bytes: [u8; 2] }
                trait Show { fn show(*self) -> u8; }
                impl Show for Pixel { fn show(*self) -> u8 { return self.bytes[0]; } }
                #[lowram]
                static mut P: Pixel;
                #[entry]
                fn main() { P.raw = 1; }
            """)

    def test_operator_trait_impl_rejected(self):
        with pytest.raises(HIRError, match="cannot implement trait"):
            build_and_check("""
                union Pixel { raw: u16, bytes: [u8; 2] }
                impl AddAssign for Pixel {
                    fn add_assign(*self, other: *Pixel) { }
                }
                #[lowram]
                static mut P: Pixel;
                #[entry]
                fn main() { P.raw = 1; }
            """)

    def test_inherent_impl_accepted(self):
        hir_prog = build_and_check("""
            union Pixel { raw: u16, bytes: [u8; 2] }
            impl Pixel {
                fn lo(*self) -> u8 { return self.bytes[0]; }
            }
            #[lowram]
            static mut P: Pixel;
            #[lowram]
            static mut R: u8;
            #[entry]
            fn main() { R = P.lo(); }
        """)
        assert hir_prog.symbol_table.lookup("Pixel__lo") is not None

    def test_clone_impl_accepted(self):
        """Clone is the one exception: a bitwise copy cannot violate layout."""
        build_and_check("""
            union Pixel { raw: u16, bytes: [u8; 2] }
            impl Clone for Pixel { }
            #[lowram]
            static mut P: Pixel = Pixel { raw: 0x1234 };
            #[lowram]
            static mut Q: Pixel;
            #[entry]
            fn main() { Q.clone_from(&P); }
        """)


class TestUnionCodegen:
    """Tests for generated code."""

    def test_field_access_matches_struct_at_offset_zero(self):
        """The zero-cost claim: union access emits what a struct field at 0 does."""
        union_asm = compile_to_asm("""
            union U { a: u8, b: u8 }
            #[lowram]
            static mut V: U;
            #[lowram]
            static mut R: u8;
            #[entry]
            fn main() { V.a = 5; R = V.b; }
        """)
        struct_asm = compile_to_asm("""
            struct U { a: u8 }
            #[lowram]
            static mut V: U;
            #[lowram]
            static mut R: u8;
            #[entry]
            fn main() { V.a = 5; R = V.a; }
        """)
        assert _main_body(union_asm) == _main_body(struct_asm)

    def test_pointer_params(self):
        """Near and far pointers to a union resolve field access."""
        asm = compile_to_asm("""
            union Word { w: u16, b: [u8; 2] }
            fn hi_of(p: *Word) -> u8 { return p.b[1]; }
            far fn far_hi(p: far *Word) -> u8 { return p.b[1]; }
            #[lowram]
            static mut W: Word;
            #[lowram]
            static mut R: u8;
            #[entry]
            fn main() { W.w = 0x1234; R = hi_of(&W) + far_hi(&W); }
        """)
        assert "hi_of:" in asm
        assert "far_hi:" in asm


# =============================================================================
# Helpers
# =============================================================================

def compile_to_asm(source: str) -> str:
    """Compile R65 source to assembly string."""
    from r65.compiler.frontend.preprocessor import preprocess
    from r65.compiler.frontend.macros import expand_macros
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.codegen.codegen import ProgramCodeGenerator

    program = parse(source, "test.r65")
    program = preprocess(program, "test.r65")
    program = expand_macros(program)
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    mir_prog = MIRBuilder().build_program(hir_prog)
    return ProgramCodeGenerator().generate(mir_prog)


def _defined_address(asm: str, name: str) -> int:
    """Read the address from a `.DEFINE <name> $xxxx` line."""
    import re
    match = re.search(rf'^\.DEFINE {name}\s+\$([0-9A-Fa-f]+)', asm, re.MULTILINE)
    assert match, f"no .DEFINE for {name}"
    return int(match.group(1), 16)


def _rom_data_bytes(asm: str, label: str):
    """Read the `.db` bytes emitted under *label*."""
    import re
    match = re.search(rf'^{re.escape(label)}:\s*\n\.db (.+)$', asm, re.MULTILINE)
    assert match, f"no ROM data for {label}"
    return [b.strip() for b in match.group(1).split(',')]


def _main_body(asm: str):
    """Extract main's instruction lines for codegen comparison.

    Compiler-generated label numbers (`__SCMP1` vs `__SCMP2`) come from a global
    counter and differ between two independent compiles, so they are normalized.
    """
    import re
    lines = []
    in_main = False
    for line in asm.splitlines():
        if line.startswith('main:'):
            in_main = True
            continue
        if in_main:
            if 'BRA' in line:
                break
            lines.append(re.sub(r'__SCMP\d+', '__SCMP', line.strip()))
    return lines
