# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for methods on fieldless enums, which take `self` by value in A.

An `impl` block on an enum used to be rejected outright ("is not a struct").
A fieldless enum is a 1-byte scalar, so it takes the same receiver form a
newtype does — `SelfForm.VALUE_IN_A` — and these tests pin that it really is A
and not a stack slot, since the two are indistinguishable at the HIR layer.

See docs/enums.md.
"""

import pytest
from r65.compiler.errors import HIRError, TypeCheckError
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder, RegisterBinding
from r65.compiler.hir.types import EnumTypeInfo
from r65.compiler.typeck import TypeChecker
from r65.compiler.main import compile_string


def build_and_check(source: str):
    """Parse, build HIR, and type check source."""
    program = parse(source, "test.r65")
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


def find_method(hir_prog, mangled: str):
    """Impl methods live inside HIRImplDecl, not among the top-level functions."""
    from r65.compiler.hir.nodes import HIRImplDecl
    for impl in hir_prog.get_declarations_by_type(HIRImplDecl):
        for method in impl.methods:
            if method.name == mangled:
                return method
    raise AssertionError(f"no method {mangled}")


DIR = """
enum Dir { North, East, South, West }

impl Dir {
    fn opposite(self) -> Dir {
        match self {
            Dir::North => Dir::South,
            Dir::East  => Dir::West,
            Dir::South => Dir::North,
            Dir::West  => Dir::East,
        }
    }
}
"""


class TestEnumSelf:
    """The receiver form of a fieldless enum method."""

    def test_self_is_bound_to_a(self):
        hir_prog = build_and_check(DIR + "fn main() {}")
        self_param = find_method(hir_prog, "Dir__opposite").parameters[0]
        assert self_param.name == "self"
        assert isinstance(self_param.binding, RegisterBinding)
        assert self_param.binding.register_name == "A"

    def test_self_has_the_enum_type(self):
        """Not the underlying u8 — `self` keeps the nominal type, so
        `match self { Dir::North => ... }` type checks against the variants."""
        hir_prog = build_and_check(DIR + "fn main() {}")
        self_param = find_method(hir_prog, "Dir__opposite").parameters[0]
        assert isinstance(self_param.param_type, EnumTypeInfo)
        assert self_param.param_type.name == "Dir"

    def test_pointer_self_is_rejected(self):
        """The self form follows the type, so `*self` on an enum is an error the
        way it is on a newtype."""
        with pytest.raises(HIRError, match="takes '\\*self', but 'Dir' is an enum"):
            build_and_check(
                "enum Dir { North }\n"
                "impl Dir { fn f(*self) -> u8 { return 0; } }\n"
                "fn main() {}"
            )

    def test_parameter_may_not_bind_a(self):
        """`self` occupies A, so a parameter cannot also claim it."""
        with pytest.raises(HIRError, match="binds A, which holds 'self'"):
            build_and_check(
                "enum Dir { North }\n"
                "impl Dir { fn f(self, x @ A: u8) -> u8 { return x; } }\n"
                "fn main() {}"
            )

    def test_parameter_may_not_bind_b(self):
        """B is A's high byte, so it is spoken for too."""
        with pytest.raises(HIRError, match="binds B"):
            build_and_check(
                "enum Dir { North }\n"
                "impl Dir { fn f(self, x @ B: u8) -> u8 { return x; } }\n"
                "fn main() {}"
            )

    def test_parameter_may_bind_x(self):
        build_and_check(
            "enum Dir { North }\n"
            "impl Dir { fn f(self, i @ X: u16) -> u16 { return i; } }\n"
            "fn main() {}"
        )


class TestAssociatedFunctions:
    """An enum associated fn declares no self and reserves no register."""

    def test_associated_fn_has_no_self(self):
        hir_prog = build_and_check(
            "enum Dir { North, East }\n"
            "impl Dir { fn start() -> Dir { return Dir::North; } }\n"
            "fn main() { let d: Dir = Dir::start(); }"
        )
        assert find_method(hir_prog, "Dir__start").parameters == []

    def test_associated_fn_may_bind_a(self):
        """No receiver means A is free — the guard that blames `self` must not
        fire on an ordinary first parameter."""
        build_and_check(
            "enum Dir { North }\n"
            "impl Dir { fn from_raw(x @ A: u8) -> u8 { return x; } }\n"
            "fn main() {}"
        )


class TestEnumMethodCodegen:
    """What only the assembly layer can prove: `self` arrives in A."""

    def _asm(self, source: str) -> str:
        return compile_string(source, "test.r65")

    def test_self_is_read_from_a_not_the_stack(self):
        """The method reads its receiver straight out of A. A stack-passed self
        would show a `LDA $nn,S` before the first use instead."""
        asm = self._asm(DIR + """
            #[zeropage]
            static mut OUT: u8;
            fn main() { OUT = Dir::North.opposite() as u8; }
        """)
        body = asm.split("Dir__opposite:")[1].split("Dir__opposite__L1")[0]
        assert "CMP #$04" in body, body

    def test_caller_passes_self_in_a(self):
        asm = self._asm(DIR + """
            #[zeropage]
            static mut OUT: u8;
            fn main() { OUT = Dir::East.opposite() as u8; }
        """)
        main = asm.split("\nmain:")[1]
        call = main.split("JSR Dir__opposite")[0]
        assert "LDA #$01" in call, main


class TestEnumWidth:
    """An enum is an integer of its own width.

    `EnumTypeInfo.size_bytes` was a hardcoded 1 while `HIREnumDecl.underlying_type`
    — which infers u16 — was written and never read, so a wide enum was silently
    truncated to a byte.
    """

    def test_small_enum_is_one_byte(self):
        hir_prog = build_and_check("enum Small { A, B }\nfn main() {}")
        assert hir_prog.symbol_table.lookup("Small").definition.underlying_type.name == "u8"
        from r65.compiler.hir.types import TypeResolver
        t = TypeResolver(hir_prog.symbol_table).resolve_named_type("Small")
        assert t.size_bytes == 1

    def test_wide_enum_is_two_bytes(self):
        hir_prog = build_and_check("enum Wide { A = 0, B = 256 }\nfn main() {}")
        from r65.compiler.hir.types import TypeResolver
        t = TypeResolver(hir_prog.symbol_table).resolve_named_type("Wide")
        assert t.size_bytes == 2

    def test_wide_enum_binds_x(self):
        """A 2-byte enum is u16-shaped, so it fits an index register; a 1-byte
        one does not."""
        build_and_check(
            "enum Wide { A = 0, B = 256 }\n"
            "fn f(w @ X: Wide) {}\n"
            "fn main() {}"
        )

    def test_narrow_enum_does_not_bind_x(self):
        with pytest.raises(HIRError, match="X"):
            build_and_check(
                "enum Small { A, B }\n"
                "fn f(s @ X: Small) {}\n"
                "fn main() {}"
            )


TAGGED = """
struct Circle { r: u8 }
struct Rect { w: u8, h: u8 }
enum Shape { Empty, C(Circle), R(Rect) }
#[lowram]
static mut POOL: [Shape; 4];

impl Shape {
    fn area(self) -> u8 {
        return match self {
            Shape::C(c) => c.r,
            Shape::R(r) => r.w,
            Shape::Empty => 0,
        };
    }
}
"""


class TestTaggedEnumMethods:
    """A tagged enum element has no address, so `self` is its index in X.

    Same receiver form as `#[soa]`, for the same reason and through the same
    code path.
    """

    def test_self_is_an_index_in_x(self):
        hir_prog = build_and_check(TAGGED + "fn main() {}")
        self_param = find_method(hir_prog, "Shape__area").parameters[0]
        assert isinstance(self_param.binding, RegisterBinding)
        assert self_param.binding.register_name == "X"
        assert str(self_param.param_type) == "u16"

    def test_pointer_self_is_rejected(self):
        with pytest.raises(HIRError,
                           match=r"takes '\*self', but 'Shape' is a tagged enum"):
            build_and_check("""
                struct Circle { r: u8 }
                enum Shape { Empty, C(Circle) }
                #[lowram]
                static mut POOL: [Shape; 4];
                impl Shape { fn f(*self) -> u8 { return 0; } }
                fn main() {}
            """)

    def test_method_without_a_static_is_rejected(self):
        """Methods hard-code their plane addresses, so there must be a pool."""
        with pytest.raises(HIRError, match=r"has no static, so its methods have no "
                                           r"array to index"):
            build_and_check("""
                struct Circle { r: u8 }
                enum Shape { Empty, C(Circle) }
                impl Shape { fn f(self) -> u8 { return 0; } }
                fn main() {}
            """)

    def test_trait_impl_rejected(self):
        """The TypeId byte and the discriminant both want offset 0."""
        with pytest.raises(HIRError, match=r"tagged enum 'Shape' cannot implement "
                                           r"trait 'Draw'"):
            build_and_check("""
                struct Circle { r: u8 }
                enum Shape { Empty, C(Circle) }
                #[lowram]
                static mut POOL: [Shape; 4];
                trait Draw { fn draw(*self); }
                impl Draw for Shape { fn draw(*self) {} }
                fn main() {}
            """)

    def test_clone_impl_rejected(self):
        """Clone copies a contiguous block; the element's bytes are in two planes."""
        with pytest.raises(HIRError, match=r"tagged enum 'Shape' cannot implement Clone"):
            build_and_check("""
                struct Circle { r: u8 }
                enum Shape { Empty, C(Circle) }
                #[lowram]
                static mut POOL: [Shape; 4];
                impl Clone for Shape {}
                fn main() {}
            """)


class TestTaggedEnumMethodCodegen:
    """The receiver really is an index, and the tag read really is unscaled."""

    def _asm(self, source: str) -> str:
        return compile_string(source, "test.r65")

    def test_caller_passes_the_index_in_x(self):
        asm = self._asm(TAGGED + """
            #[zeropage]
            static mut OUT: u8;
            fn main() { OUT = POOL[2].area(); }
        """)
        main = asm[asm.index("\nmain:"):]
        call = main.split("JSR Shape__area")[0]
        assert "LDX #$02" in call, main

    def test_tag_read_uses_the_index_unscaled(self):
        """`LDA POOL__tag,X` straight off the receiver — the whole point of
        keeping the tag in its own stride-1 plane."""
        asm = self._asm(TAGGED + """
            #[zeropage]
            static mut OUT: u8;
            fn main() { OUT = POOL[2].area(); }
        """)
        body = asm[asm.index("\nShape__area:"):]
        dispatch = body.split("CMP #$01")[0]
        assert "LDA $0200,X" in dispatch, dispatch
        assert "ASL" not in dispatch, dispatch
