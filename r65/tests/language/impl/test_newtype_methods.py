# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for newtype methods, which take `self` by value in a register.

One self form per type: a newtype is a value and always takes bare `self`;
a struct or union is addressed and always takes `*self`. Mixing them is an error
in both directions.
"""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder, RegisterBinding
from r65.compiler.hir.types import NewtypeTypeInfo
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError, HIRError


def build_and_check(source: str):
    """Parse, build HIR, and type check source."""
    program = parse(source, "test.r65")
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


def compile_to_asm(source: str) -> str:
    """Compile R65 source to assembly string."""
    from r65.compiler.frontend.preprocessor import preprocess
    from r65.compiler.frontend.macros import expand_macros
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.codegen.codegen import ProgramCodeGenerator
    from r65.compiler.analysis import RecursionChecker

    program = expand_macros(preprocess(parse(source, "test.r65"), "test.r65"))
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    mir_prog = MIRBuilder().build_program(hir_prog)
    RecursionChecker(mir_prog).check()
    return ProgramCodeGenerator().generate(mir_prog)


def find_method(hir_prog, mangled_name):
    for decl in hir_prog.declarations:
        for method in getattr(decl, "methods", []) or []:
            if getattr(method, "name", None) == mangled_name:
                return method
        if getattr(decl, "name", None) == mangled_name:
            return decl
    raise AssertionError(f"method {mangled_name} not found")


TILE = """
struct TileId(u8);

impl TileId {
    fn raw(self) -> u8 { return self.0; }
    fn bumped(self) -> TileId { return TileId(self.0 + 1); }
    fn is_solid(self) -> bool { return self.0 >= 0x80; }
    fn plus(self, n: u8) -> TileId { return TileId(self.0 + n); }
}
"""


class TestByValueSelf:
    """`self` arrives in the accumulator, by value."""

    def test_methods_compile(self):
        build_and_check(TILE + "fn main() { let t: TileId = 5; let n: u8 = t.raw(); }")

    def test_self_is_the_newtype_not_a_pointer(self):
        hir_prog = build_and_check(TILE + "fn main() {}")
        method = find_method(hir_prog, "TileId__raw")
        self_param = method.parameters[0]
        assert self_param.name == "self"
        assert isinstance(self_param.param_type, NewtypeTypeInfo)
        assert self_param.param_type.newtype_name == "TileId"

    def test_self_is_bound_to_the_accumulator(self):
        hir_prog = build_and_check(TILE + "fn main() {}")
        method = find_method(hir_prog, "TileId__raw")
        binding = method.parameters[0].binding
        assert isinstance(binding, RegisterBinding)
        assert binding.register_name == "A"

    def test_not_a_trait_method(self):
        """Trait methods take self in Y and are never inlined; a newtype method
        must stay on the ordinary static-dispatch path."""
        hir_prog = build_and_check(TILE + "fn main() {}")
        assert not find_method(hir_prog, "TileId__raw").is_trait_method

    def test_method_with_extra_argument(self):
        build_and_check(TILE + "fn main() { let t: TileId = 5; let u: TileId = t.plus(3); }")

    def test_returning_the_newtype(self):
        build_and_check(TILE + "fn main() { let t: TileId = 5; let u: TileId = t.bumped(); }")

    def test_returning_bool(self):
        build_and_check(TILE + "fn main() { let t: TileId = 5; let b: bool = t.is_solid(); }")

    def test_chained_calls(self):
        build_and_check(TILE + "fn main() { let t: TileId = 5; let u: TileId = t.bumped().bumped(); }")

    def test_receiver_may_be_a_static(self):
        src = TILE + "#[zeropage(0x10)]\nstatic mut T: TileId;\nfn main() { T = T.bumped(); }"
        build_and_check(src)

    def test_associated_function_without_self_declares(self):
        """An associated fn writes no self, so neither self-form check applies."""
        src = ("struct TileId(u8);\n"
               "impl TileId { fn zero() -> TileId { return TileId(0); } }\n"
               "fn main() {}")
        build_and_check(src)

    def test_associated_function_self_claims_no_register(self):
        """A synthetic self is still injected (pre-existing), but binding it to A
        would make a 2-byte payload infer m16 — emitting a REP into a handler
        that takes no arguments."""
        from r65.compiler.typeck.processor_mode import ModeState
        src = ("struct Addr(u16);\n"
               "impl Addr { fn zero() -> Addr { return Addr(0); } }\n"
               "fn main() {}")
        hir_prog = build_and_check(src)
        method = find_method(hir_prog, "Addr__zero")
        assert method.parameters[0].binding is None
        assert method.entry_m_mode == ModeState.M8

    def test_interrupt_handler_has_no_spurious_mode_switch(self):
        from r65.compiler.typeck.processor_mode import ModeState
        src = ("struct Addr(u16);\n"
               "impl Addr { #[interrupt(nmi)] fn h() { } }\n"
               "#[entry]\nfn main() {}")
        hir_prog = build_and_check(src)
        assert find_method(hir_prog, "Addr__h").entry_m_mode == ModeState.M8

    def test_u16_payload_self_enters_m16(self):
        """A 2-byte self in A must put the function in m16, exactly as `@ A: i16`."""
        from r65.compiler.typeck.processor_mode import ModeState
        src = ("struct Q10(i16);\n"
               "impl Q10 { fn doubled(self) -> Q10 { return Q10(self.0 + self.0); } }\n"
               "fn main() {}")
        hir_prog = build_and_check(src)
        assert find_method(hir_prog, "Q10__doubled").entry_m_mode == ModeState.M16


class TestSelfRegisterConflict:
    """`self` occupies A, so a parameter cannot also bind it.

    Without this check the two share the register and the method silently
    computes with the wrong operand.
    """

    def test_parameter_binding_a_is_rejected(self):
        src = ("struct TileId(u8);\n"
               "impl TileId { fn m(self, x @ A: u8) -> u8 { return x; } }\nfn main() {}")
        with pytest.raises(HIRError, match="binds A, which holds 'self'"):
            build_and_check(src)

    def test_hint_offers_the_alternatives(self):
        src = ("struct TileId(u8);\n"
               "impl TileId { fn m(self, x @ A: u8) -> u8 { return x; } }\nfn main() {}")
        with pytest.raises(HIRError) as exc:
            build_and_check(src)
        assert "bind 'x' to X or Y, or pass it on the stack" in exc.value.hint

    def test_index_register_binding_is_fine(self):
        build_and_check("struct TileId(u8);\n"
                        "impl TileId { fn m(self, x @ X: u16) -> u8 { return self.0; } }\n"
                        "fn main() {}")

    def test_stack_parameter_is_fine(self):
        build_and_check("struct TileId(u8);\n"
                        "impl TileId { fn m(self, x: u8) -> u8 { return self.0 + x; } }\n"
                        "fn main() {}")

    def test_duplicate_bindings_between_ordinary_params(self):
        """Impl methods never ran the duplicate-binding check at all; a pointer
        self hid it because it carries no binding."""
        src = ("struct P { x: u8 }\n"
               "impl P { fn m(*self, a @ A: u8, b @ A: u8) -> u8 { return a; } }\nfn main() {}")
        with pytest.raises(HIRError, match="bound to multiple parameters"):
            build_and_check(src)


class TestSelfFormMismatch:
    """One self form per type, enforced in both directions."""

    def test_pointer_self_on_a_newtype_rejected(self):
        src = "struct TileId(u8);\nimpl TileId { fn bump(*self) {} }\nfn main() {}"
        with pytest.raises(HIRError, match="is a newtype"):
            build_and_check(src)

    def test_pointer_self_hint_shows_the_by_value_form(self):
        src = "struct TileId(u8);\nimpl TileId { fn bump(*self) {} }\nfn main() {}"
        with pytest.raises(HIRError) as exc:
            build_and_check(src)
        assert "fn bump(self)" in exc.value.hint

    def test_far_pointer_self_on_a_newtype_rejected(self):
        src = "struct TileId(u8);\nimpl TileId { far fn bump(far *self) {} }\nfn main() {}"
        with pytest.raises(HIRError, match="is a newtype"):
            build_and_check(src)

    def test_by_value_self_on_a_struct_rejected(self):
        src = ("struct P { x: u8 }\n"
               "impl P { fn get(self) -> u8 { return self.x; } }\nfn main() {}")
        with pytest.raises(HIRError, match="is not a newtype"):
            build_and_check(src)

    def test_by_value_self_on_a_struct_hints_pointer_form(self):
        src = ("struct P { x: u8 }\n"
               "impl P { fn get(self) -> u8 { return self.x; } }\nfn main() {}")
        with pytest.raises(HIRError) as exc:
            build_and_check(src)
        assert "*self" in exc.value.hint

    def test_by_value_self_on_a_union_rejected(self):
        src = ("union U { a: u8, b: u8 }\n"
               "impl U { fn get(self) -> u8 { return self.a; } }\nfn main() {}")
        with pytest.raises(HIRError, match="is not a newtype"):
            build_and_check(src)


class TestNewtypeTraitImpls:
    """A newtype has no spare byte for a TypeId, so it cannot be a dispatch target."""

    def test_trait_impl_rejected(self):
        src = ("struct TileId(u8);\n"
               "trait Drawable { fn draw(*self); }\n"
               "impl Drawable for TileId { fn draw(self) {} }\n"
               "fn main() {}")
        with pytest.raises(HIRError, match="cannot implement trait 'Drawable'"):
            build_and_check(src)

    def test_trait_impl_hint_explains_the_layout_conflict(self):
        src = ("struct TileId(u8);\n"
               "trait Drawable { fn draw(*self); }\n"
               "impl Drawable for TileId { fn draw(self) {} }\n"
               "fn main() {}")
        with pytest.raises(HIRError) as exc:
            build_and_check(src)
        assert "TypeId byte at offset 0" in exc.value.hint

    def test_clone_impl_rejected(self):
        src = "struct TileId(u8);\nimpl Clone for TileId {}\nfn main() {}"
        with pytest.raises(HIRError, match="cannot implement Clone"):
            build_and_check(src)

    def test_clone_hint_points_at_plain_assignment(self):
        src = "struct TileId(u8);\nimpl Clone for TileId {}\nfn main() {}"
        with pytest.raises(HIRError) as exc:
            build_and_check(src)
        assert "plain assignment" in exc.value.hint

    def test_newtype_copies_by_assignment(self):
        """The reason Clone is redundant: a newtype is a scalar."""
        src = ("struct TileId(u8);\n"
               "fn main() { let a: TileId = 5; let b: TileId = a; }")
        build_and_check(src)


class TestZeroCost:
    """The point of by-value self: the generated code is what you would write."""

    def test_trivial_accessor_is_inlined_away(self):
        """`t.raw()` is a retype of a value already in A, so nothing should
        survive — no JSR, and no function body to jump to."""
        asm = compile_to_asm('''
            struct TileId(u8);
            impl TileId { fn raw(self) -> u8 { return self.0; } }

            #[zeropage(0x10)]
            static mut TILE: TileId;
            #[zeropage(0x11)]
            static mut OUT: u8;

            #[entry]
            fn main() { TILE = TileId(5); OUT = TILE.raw(); }
        ''')
        assert "JSR TileId__raw" not in asm
        assert "TileId__raw:" not in asm

    def test_self_arrives_in_a_without_a_pointer(self):
        """A non-inlined method takes self in A — no address is ever formed, so
        nothing is pushed for it and the body reads the accumulator directly."""
        asm = compile_to_asm('''
            struct TileId(u8);
            impl TileId { fn bumped(self) -> TileId { return TileId(self.0 + 1); } }

            #[zeropage(0x10)]
            static mut TILE: TileId;

            #[entry]
            fn main() { TILE = TileId(5); TILE = TILE.bumped(); }
        ''')
        body = asm.split("TileId__bumped:")[1].split("RTS")[0]
        # The whole method is an increment of the accumulator.
        assert "INC A" in body
        # No stack traffic for self, and no pointer dereference.
        for pointer_ish in ("PHA", "PHX", "PHY", "LDA $00", "STA $00"):
            assert pointer_ish not in body, f"unexpected {pointer_ish} in:\n{body}"


class TestMethodCallErrors:
    def test_unknown_method(self):
        src = "struct TileId(u8);\nfn main() { let t: TileId = 5; let n: u8 = t.nope(); }"
        with pytest.raises(TypeCheckError):
            build_and_check(src)

    def test_wrong_argument_count(self):
        src = TILE + "fn main() { let t: TileId = 5; let u: TileId = t.plus(1, 2); }"
        with pytest.raises(TypeCheckError, match="expects 1 argument"):
            build_and_check(src)

    def test_argument_type_still_checked(self):
        src = TILE + ("struct Q10(i16);\n"
                      "fn main() { let t: TileId = 5; let q: Q10 = 1; let u: TileId = t.plus(q); }")
        with pytest.raises(TypeCheckError, match="wrong type"):
            build_and_check(src)
