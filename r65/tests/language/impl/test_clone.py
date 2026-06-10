# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for Clone (Slice 1): auto/array intrinsic copy, .clone() sugar, errors."""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.frontend.preprocessor import preprocess
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError, HIRError


def build_and_check(source: str):
    program = parse(source, "test.r65")
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


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


class TestCloneIntrinsic:
    def test_auto_struct_clone_emits_block_move(self):
        asm = compile_to_asm('''
            struct Vec2 { x: u8, y: u8 }
            impl Clone for Vec2 {}
            #[lowram] static mut SRC: Vec2;
            #[lowram] static mut DST: Vec2;
            fn main() { DST.clone_from(&SRC); }
        ''')
        assert "MVN" in asm
        # Auto impl generates no method function.
        assert "Vec2__clone_from" not in asm

    def test_clone_sugar_resolves(self):
        asm = compile_to_asm('''
            struct Vec2 { x: u8, y: u8 }
            impl Clone for Vec2 {}
            #[lowram] static mut SRC: Vec2;
            fn main() { let c = SRC.clone(); let unused = c.x; }
        ''')
        assert "MVN" in asm

    def test_array_clone_needs_no_impl(self):
        asm = compile_to_asm('''
            #[lowram] static mut SRC: [u8; 8];
            #[lowram] static mut DST: [u8; 8];
            fn main() {
                DST.clone_from(&SRC);
                let c = SRC.clone();
                let unused = c[0];
            }
        ''')
        assert asm.count("MVN") >= 2

    def test_manual_clone_from_calls_method(self):
        asm = compile_to_asm('''
            struct Tag { id: u8, flag: u8 }
            impl Clone for Tag {
                fn clone_from(*self, src: *Tag) { self.id = src.id; }
            }
            #[lowram] static mut SRC: Tag;
            #[lowram] static mut DST: Tag;
            fn main() { DST.clone_from(&SRC); }
        ''')
        # Manual impl runs the user body (field copy), not the intrinsic block move.
        # (The trivial body may be inlined, so don't require a JSR.)
        assert "Tag__clone_from" in asm
        assert "MVN" not in asm


class TestCloneErrors:
    def test_no_clone_impl_rejected(self):
        with pytest.raises(TypeCheckError, match="does not implement Clone"):
            build_and_check('''
                struct V { x: u8 }
                #[lowram] static mut P: V;
                #[lowram] static mut Q: V;
                fn main() { Q.clone_from(&P); }
            ''')

    def test_clone_sugar_only_as_initializer(self):
        with pytest.raises(TypeCheckError, match="only allowed as the direct initializer"):
            build_and_check('''
                struct V { x: u8 }
                impl Clone for V {}
                fn sink(p: *V) {}
                #[lowram] static mut P: V;
                fn main() { sink(&P.clone()); }
            ''')

    def test_clone_through_runtime_pointer_rejected(self):
        with pytest.raises(TypeCheckError, match="runtime pointer is not yet supported"):
            build_and_check('''
                struct V { x: u8 }
                impl Clone for V {}
                fn copy(dst: *V, srcp: *V) { dst.clone_from(srcp); }
                fn main() {}
            ''')

    def test_manual_clone_sugar_rejected(self):
        with pytest.raises(TypeCheckError, match="not yet supported.*custom Clone impl"):
            build_and_check('''
                struct V { x: u8 }
                impl Clone for V { fn clone_from(*self, src: *V) { self.x = src.x; } }
                #[lowram] static mut P: V;
                fn main() { let c = P.clone(); let u = c.x; }
            ''')
