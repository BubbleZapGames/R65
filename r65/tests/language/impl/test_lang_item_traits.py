# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for compiler lang-item traits (Clone + operator traits).

Slice 0 (foundation): `impl <LangItem> for T` is accepted WITHOUT a source
`trait` declaration, registers its methods through the normal static-dispatch
path, never injects a TypeId, and cannot be used as `*dyn`. See
docs/operator-overloading.md.
"""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.frontend.preprocessor import preprocess
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import HIRError


def build_hir(source: str):
    """Parse + build HIR; return the HIRBuilder (exposes _clone_impls)."""
    program = parse(source, "test.r65")
    hb = HIRBuilder(source_file="test.r65")
    hb.build_program(program)
    return hb


def compile_to_asm(source: str) -> str:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.codegen.codegen import ProgramCodeGenerator

    program = parse(source, "test.r65")
    program = preprocess(program, "test.r65")
    program = expand_macros(program)
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    mir_prog = MIRBuilder().build_program(hir_prog)
    return ProgramCodeGenerator().generate(mir_prog)


class TestLangItemImplAcceptance:
    """impl <LangItem> for T is accepted with no declared trait."""

    def test_empty_clone_impl_is_auto(self):
        hb = build_hir("struct V { x: u8 }\nimpl Clone for V {}\n")
        assert hb._clone_impls["V"]["auto"] is True
        assert hb._clone_impls["V"]["mangled"] is None

    def test_manual_clone_impl_recorded(self):
        src = """
            struct V { x: u8 }
            impl Clone for V {
                fn clone_from(*self, src: *V) { self.x = src.x; }
            }
        """
        hb = build_hir(src)
        assert hb._clone_impls["V"]["auto"] is False
        assert hb._clone_impls["V"]["mangled"] == "V__clone_from"

    def test_operator_trait_impl_accepted_without_declaration(self):
        """Operator lang-item traits also need no `trait` declaration (Slice 0 foundation)."""
        src = """
            struct V { x: u8 }
            impl AddAssign for V {
                fn add_assign(*self, other: *V) { self.x = self.x + other.x; }
            }
        """
        # Must not raise "impl block for undefined trait".
        build_hir(src)

    def test_clone_impl_adds_no_type_id(self):
        """Lang-item impls are static-only: no TypeId injection, layout unchanged."""
        hb = build_hir("struct V { x: u8 }\nimpl Clone for V {}\n")
        assert "V" not in hb._struct_type_ids


class TestManualCloneCall:
    """A manual clone_from resolves and lowers as an ordinary static call."""

    def test_clone_from_compiles_to_call(self):
        src = """
            struct Vec2 { x: u8, y: u8 }
            impl Clone for Vec2 {
                fn clone_from(*self, src: *Vec2) {
                    self.x = src.x;
                    self.y = src.y;
                }
            }
            #[lowram]
            static mut SRC: Vec2;
            #[lowram]
            static mut DST: Vec2;
            fn main() {
                SRC.x = 1;
                SRC.y = 2;
                DST.clone_from(&SRC);
            }
        """
        asm = compile_to_asm(src)
        assert "Vec2__clone_from" in asm
        assert ("JSR Vec2__clone_from" in asm) or ("JSL Vec2__clone_from" in asm)


class TestLangItemRejections:
    """Guard rails."""

    def test_dyn_clone_rejected(self):
        src = "struct V { x: u8 }\nimpl Clone for V {}\nfn f(p: *dyn Clone) {}\n"
        with pytest.raises(HIRError, match="cannot be .*used as a trait object"):
            build_hir(src)

    def test_clone_wrong_method_name_rejected(self):
        src = "struct V { x: u8 }\nimpl Clone for V { fn frob(*self, s: *V) {} }\n"
        with pytest.raises(HIRError, match="only a 'clone_from' method"):
            build_hir(src)

    def test_clone_wrong_arity_rejected(self):
        src = "struct V { x: u8 }\nimpl Clone for V { fn clone_from(*self) {} }\n"
        with pytest.raises(HIRError, match="exactly one parameter"):
            build_hir(src)
