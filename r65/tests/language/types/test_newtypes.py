# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for newtypes: `struct TileId(u8);`.

A newtype is nominal at compile time and its payload at runtime. The rules it
has to obey are transparent in, opaque out — payload values flow into it
implicitly, nothing flows back out without an explicit unwrap.
"""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder, HIRNewtypeDecl
from r65.compiler.hir.types import NewtypeTypeInfo, BasicTypeInfo
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError, HIRError


NEWTYPES = "struct TileId(u8);\nstruct Q10(i16);\n"


def build_and_check(source: str):
    """Parse, build HIR, and type check source."""
    program = parse(source, "test.r65")
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


def in_main(body: str, prelude: str = NEWTYPES):
    return f"{prelude}\nfn main() {{ {body} }}"


class TestNewtypeDeclaration:
    """Declaration, and the payload types a newtype may wrap."""

    def test_declares_a_newtype(self):
        hir_prog = build_and_check("struct TileId(u8);")
        decls = [d for d in hir_prog.declarations if isinstance(d, HIRNewtypeDecl)]
        assert len(decls) == 1
        assert decls[0].name == "TileId"
        assert decls[0].inner_type == BasicTypeInfo("u8")

    def test_is_not_an_aggregate(self):
        """The whole design rests on this: a newtype never meets the
        pass-by-reference machinery."""
        from r65.compiler.typeck.type_utils import TypeUtils
        ty = NewtypeTypeInfo(newtype_name="TileId", inner=BasicTypeInfo("u8"))
        assert not TypeUtils.is_aggregate_type(ty)

    def test_size_is_the_payload_size(self):
        assert NewtypeTypeInfo("T", BasicTypeInfo("u8")).size_bytes == 1
        assert NewtypeTypeInfo("T", BasicTypeInfo("i16")).size_bytes == 2

    @pytest.mark.parametrize("decl", [
        "struct W(u8);", "struct W(i8);", "struct W(bool);",
        "struct W(u16);", "struct W(i16);",
        "struct W(*u8);", "struct W(fn());",
    ])
    def test_scalar_payloads_accepted(self, decl):
        build_and_check(decl)

    def test_enum_payload_accepted(self):
        build_and_check("enum Direction { North, East }\nstruct Facing(Direction);")

    def test_far_pointer_payload_rejected(self):
        with pytest.raises(HIRError, match="which is 3 bytes"):
            build_and_check("struct Ref(far *u8);")

    def test_struct_payload_rejected(self):
        with pytest.raises(HIRError, match="is not a scalar"):
            build_and_check("struct P { x: u8, y: u8 }\nstruct Wrap(P);")

    def test_array_payload_rejected(self):
        with pytest.raises(HIRError, match="is not a scalar"):
            build_and_check("struct Buf([u8; 4]);")

    def test_nested_newtype_rejected(self):
        with pytest.raises(HIRError, match="cannot wrap another newtype"):
            build_and_check("struct TileId(u8);\nstruct Wrap(TileId);")


class TestTransparentIn:
    """Payload values flow into a newtype without a cast."""

    def test_literal_initializer(self):
        build_and_check(in_main("let t: TileId = 5;"))

    def test_payload_typed_variable(self):
        build_and_check(in_main("let n: u8 = 3; let t: TileId = n;"))

    def test_construction(self):
        build_and_check(in_main("let t: TileId = TileId(5);"))

    def test_static_initializer(self):
        build_and_check(NEWTYPES + "static T: TileId = TileId(7);\nfn main() {}")

    def test_argument(self):
        build_and_check(NEWTYPES + "fn f(t: TileId) {}\nfn main() { f(5); }")

    def test_construction_takes_exactly_one_value(self):
        with pytest.raises(HIRError, match="takes exactly 1 value"):
            build_and_check(in_main("let t: TileId = TileId(1, 2);"))


class TestPayloadRangeIsInherited:
    """Transparency carries the payload's constraints, not just its operations —
    a literal too large for `u8` is too large for a newtype over `u8`."""

    def test_let_binding(self):
        with pytest.raises(TypeCheckError, match="300 does not fit in type u8"):
            build_and_check(in_main("let t: TileId = 300;"))

    def test_static_initializer(self):
        with pytest.raises(TypeCheckError, match="300 does not fit in type u8"):
            build_and_check(NEWTYPES + "static mut T: TileId = 300;\nfn main() {}")

    def test_const_declaration(self):
        with pytest.raises(TypeCheckError, match="300 does not fit in type u8"):
            build_and_check(NEWTYPES + "const T: TileId = 300;\nfn main() {}")

    def test_argument(self):
        with pytest.raises(TypeCheckError, match="300 does not fit in type u8"):
            build_and_check(NEWTYPES + "fn f(t: TileId) {}\nfn main() { f(300); }")

    def test_in_range_still_accepted(self):
        build_and_check(in_main("let t: TileId = 255;"))

    def test_wider_payload_accepts_wider_literal(self):
        build_and_check(in_main("let q: Q10 = 300;"))

    def test_most_negative_literal_of_a_signed_payload(self):
        """`-128` parses as `-(128)` and needs the payload's escape hatch; without
        it the newtype rejects a literal its payload accepts."""
        build_and_check("struct Q8(i8);\nfn main() { let q: Q8 = -128; }")
        build_and_check(in_main("let q: Q10 = -32768;"))

    def test_negative_overflow_still_rejected(self):
        with pytest.raises(TypeCheckError, match="does not fit in type i8"):
            build_and_check("struct Q8(i8);\nfn main() { let q: Q8 = -129; }")

    def test_unary_operators_are_inherited(self):
        build_and_check(in_main("let q: Q10 = 5; let a = -q; let t: TileId = 3; let b = ~t;"))

    def test_construction_is_checked_like_an_assignment(self):
        """`TileId(x)` rejects what `let t: TileId = x;` rejects — the two
        spellings of the same operation must not disagree."""
        with pytest.raises(TypeCheckError, match="300 does not fit in type u8"):
            build_and_check(in_main("let t: TileId = TileId(300);"))

    def test_cast_is_the_truncating_spelling(self):
        """`as` truncates for newtypes exactly as it does for their payloads."""
        build_and_check(in_main("let t: TileId = 300 as TileId;"))

    def test_construction_from_a_foreign_nominal_type_is_rejected(self):
        src = ("enum Dir { North }\nenum Col { Red }\nstruct Facing(Dir);\n"
               "fn main() { let f: Facing = Facing(Col::Red); }")
        with pytest.raises(TypeCheckError, match="cannot make a 'Facing' from 'Col'"):
            build_and_check(src)


class TestOpaqueOut:
    """A newtype never flows back into its payload implicitly."""

    def test_let_binding_rejected(self):
        with pytest.raises(TypeCheckError, match="expected u8, found TileId"):
            build_and_check(in_main("let t: TileId = 5; let n: u8 = t;"))

    def test_argument_rejected(self):
        with pytest.raises(TypeCheckError, match="expected u8, found TileId"):
            build_and_check(NEWTYPES + "fn g(n: u8) {}\nfn main() { let t: TileId = 5; g(t); }")

    def test_assignment_rejected(self):
        src = NEWTYPES + "static mut N: u8;\nfn main() { let t: TileId = 5; N = t; }"
        with pytest.raises(TypeCheckError, match="found TileId"):
            build_and_check(src)

    def test_distinct_newtypes_do_not_mix(self):
        with pytest.raises(TypeCheckError, match="expected TileId, found Q10"):
            build_and_check(in_main("let q: Q10 = 5; let t: TileId = q;"))


class TestUnwrapping:
    """`.0` and `as` are the two explicit escape hatches."""

    def test_field_access(self):
        build_and_check(in_main("let t: TileId = 5; let n: u8 = t.0;"))

    def test_cast(self):
        build_and_check(in_main("let t: TileId = 5; let n: u8 = t as u8;"))

    def test_cast_into_newtype(self):
        build_and_check(in_main("let n: u8 = 5; let t: TileId = n as TileId;"))

    def test_widening_cast_from_payload(self):
        build_and_check(in_main("let t: TileId = 5; let n: u16 = t as u16;"))

    def test_index_other_than_zero_rejected(self):
        with pytest.raises(TypeCheckError, match="has only field '.0'"):
            build_and_check(in_main("let t: TileId = 5; let n: u8 = t.1;"))

    def test_field_access_on_struct_rejected(self):
        src = ("struct P { x: u8 }\nstatic mut PP: P;\n"
               "fn main() { let n: u8 = PP.0; }")
        with pytest.raises(TypeCheckError, match="is not a newtype"):
            build_and_check(src)


class TestInheritedOperators:
    """Operators come from the payload; the result stays nominal."""

    def _expr_type(self, body, expr_name="r"):
        hir_prog = build_and_check(in_main(body))
        fn = [d for d in hir_prog.declarations
              if getattr(d, "name", None) == "main"][0]
        for stmt in fn.body.statements:
            if getattr(stmt, "name", None) == expr_name:
                return stmt.initializer.expr_type
        raise AssertionError(f"no binding named {expr_name}")

    def test_arithmetic_result_is_the_newtype(self):
        ty = self._expr_type("let t: TileId = 5; let r = t + 1;")
        assert isinstance(ty, NewtypeTypeInfo) and ty.newtype_name == "TileId"

    def test_newtype_on_the_right(self):
        ty = self._expr_type("let t: TileId = 5; let r = 1 + t;")
        assert isinstance(ty, NewtypeTypeInfo) and ty.newtype_name == "TileId"

    def test_both_operands(self):
        ty = self._expr_type("let a: TileId = 5; let b: TileId = 6; let r = a & b;")
        assert isinstance(ty, NewtypeTypeInfo) and ty.newtype_name == "TileId"

    def test_shift_result_is_the_newtype(self):
        ty = self._expr_type("let t: TileId = 5; let r = t << 1;")
        assert isinstance(ty, NewtypeTypeInfo) and ty.newtype_name == "TileId"

    def test_comparison_is_bool(self):
        ty = self._expr_type("let a: TileId = 5; let b: TileId = 6; let r = a < b;")
        assert ty == BasicTypeInfo("bool")

    def test_mismatched_newtypes_rejected(self):
        with pytest.raises(TypeCheckError, match="mismatched types 'TileId' and 'Q10'"):
            build_and_check(in_main("let a: TileId = 5; let b: Q10 = 6; let c = a + b;"))


class TestComposition:
    """Newtypes compose as ordinary scalars."""

    def test_struct_field(self):
        build_and_check(NEWTYPES + "struct Holder { a: u8, t: TileId }\n"
                        "static mut H: Holder;\nfn main() { H.t = TileId(3); }")

    def test_array_element(self):
        build_and_check(NEWTYPES + "#[ram]\nstatic mut ARR: [TileId; 8];\n"
                        "fn main() { ARR[2] = TileId(3); }")

    def test_register_bound_parameter(self):
        build_and_check(NEWTYPES + "fn f(t @ A: TileId) -> TileId { return t + 1; }\n"
                        "fn main() { let r: TileId = f(TileId(2)); }")

    def test_return_type(self):
        build_and_check(NEWTYPES + "fn f() -> TileId { return TileId(1); }\n"
                        "fn main() { let t: TileId = f(); }")

    def test_enum_payload_round_trip(self):
        """An enum payload wraps, unwraps, and binds a register — unlike a bare
        enum, which cannot bind one."""
        build_and_check('''
            enum Direction { North, East }
            struct Facing(Direction);
            fn turn(d @ A: Facing) -> Facing { return d; }
            fn main() {
                let f: Facing = Facing(Direction::East);
                let d: Direction = f.0;
                let n: u8 = f as u8;
                let g: Facing = turn(f);
            }
        ''')

    def test_bare_enum_still_cannot_bind_a_register(self):
        """Unchanged behaviour — the newtype carve-out must not leak."""
        with pytest.raises(HIRError, match="must have a primitive type"):
            build_and_check("enum Direction { North }\nfn f(d @ A: Direction) {}\nfn main() {}")

    def test_pointer_payload_derefs_through_zero(self):
        build_and_check('''
            struct Sprite { x: u8 }
            struct Handle(*Sprite);
            impl Handle { fn get(self) -> u8 { return self.0.x; } }
            fn main() {}
        ''')


class TestNewtypeToStringViaFormat:
    """`format!("{s}", x)` resolves `to_string` by name, not by trait dispatch.

    A newtype cannot `impl ToString` — the TypeId byte at offset 0 would overlap
    its value — but it can carry an inherent `to_string`, and static resolution
    finds it. That is strictly better here: no vtable, and `self` by value.
    """

    # `format!` itself lives in stdlib/string.r65; the type checker's dispatch
    # happens on the `__fmt_str` call the macro expands to, so drive that
    # directly and keep this a unit test. The stdlib path is covered by
    # r65/tests/e2e/test_Q10_to_string.py.
    DECL = ("struct Tag(u8);\n"
            "#[ram]\nstatic mut BUF: [u8; 16];\n"
            "#[zeropage(0x30)]\nstatic mut N: u16;\n")

    def check(self, extra: str):
        src = (self.DECL + extra
               + '#[entry]\nfn main() { let t: Tag = 5;'
                 ' N = __fmt_str(&BUF as far *u8, t); }')
        TypeChecker(HIRBuilder(source_file="t.r65").build_program(
            parse(src, "t.r65"))).check()

    def test_inherent_to_string_is_accepted(self):
        self.check("impl Tag {\n"
                   "    fn to_string(self, buf: far *u8) -> u16 { return 0; }\n"
                   "}\n")

    def test_error_names_the_inherent_form(self):
        """Without the method, the hint must not suggest `impl ToString for` —
        that is rejected outright for a newtype."""
        with pytest.raises(TypeCheckError) as exc:
            self.check("")
        assert "newtype 'Tag' has no 'to_string'" in str(exc.value), str(exc.value)
        hint = exc.value.hint or ""
        assert "impl Tag {" in hint, f"hint should show the inherent form: {hint!r}"
        assert "impl ToString for" not in hint, (
            f"a newtype cannot implement the trait: {hint!r}")


class TestNewtypeInMatch:
    """`match` on a newtype, which used to crash the compiler.

    `match_validator` had no newtype awareness at all: it read
    `scrutinee_type.name` bare to decide whether a literal pattern could apply,
    and `NewtypeTypeInfo.name` raises by design — so matching a newtype over a
    primitive produced a `TypeError` out of the type checker rather than any
    diagnostic. Matching over an *enum* payload was unaffected, taking a
    different branch.

    Exhaustiveness was a quieter second gap: its `isinstance` checks simply
    missed `NewtypeTypeInfo`, so once the crash was gone a newtype match would
    have been accepted without the wildcard its payload requires.
    """

    DECL = "struct Tid(u8);\nstruct W(u16);\nstruct Flag(bool);\nstruct Rot(i16);\n"

    def check(self, body: str, decl: str = ""):
        src = self.DECL + decl + "#[entry]\nfn main() { " + body + " }"
        TypeChecker(HIRBuilder(source_file="t.r65").build_program(
            parse(src, "t.r65"))).check()

    @pytest.mark.parametrize("ty,val,pat", [
        ("Tid", "1", "1"),
        ("W", "1", "1"),
        ("Rot", "0 - 1", "-1"),     # a negative literal *is* a pattern; only an
                                    # expression like `0 - 1` is not
    ])
    def test_integer_literal_patterns(self, ty, val, pat):
        """Literals reach a newtype's payload transparently here, as they do in
        every other position that accepts one."""
        self.check(f"let v: {ty} = {val}; match v {{ {pat} => {{ }}, _ => {{ }} }};")

    def test_bool_payload(self):
        self.check("let f: Flag = true; match f { true => { }, false => { } };")

    def test_range_pattern(self):
        self.check("let t: Tid = 1; match t { 1..5 => { }, _ => { } };")

    def test_enum_payload_still_works(self):
        """Unaffected by the bug — it never reached the bare `.name` read."""
        self.check("let d: Dir = Way::N; match d { Way::N => { }, _ => { } };",
                   decl="enum Way { N, S }\nstruct Dir(Way);\n")

    def test_exhaustiveness_is_enforced(self):
        """A newtype over an integer needs a wildcard, exactly as u8 does."""
        with pytest.raises(TypeCheckError, match="Non-exhaustive"):
            self.check("let t: Tid = 1; match t { 1 => { } };")

    def test_exhaustiveness_names_the_newtype(self):
        """The message should say what the reader wrote, not the payload."""
        with pytest.raises(TypeCheckError) as exc:
            self.check("let t: Tid = 1; match t { 1 => { } };")
        assert "on Tid" in str(exc.value), str(exc.value)

    def test_bool_payload_exhaustiveness(self):
        with pytest.raises(TypeCheckError, match="missing patterns for false"):
            self.check("let f: Flag = true; match f { true => { } };")

    def test_mismatched_literal_still_rejected(self):
        """The check must still reject, not merely stop crashing — and name the
        newtype when it does."""
        with pytest.raises(TypeCheckError) as exc:
            self.check("let f: Flag = true; match f { 1 => { }, _ => { } };")
        assert "Flag" in str(exc.value), str(exc.value)


class TestNewtypeTraitImpls:
    """A newtype may implement a trait, but only for static dispatch.

    The blanket rejection this replaces cited the TypeId byte that dynamic
    dispatch stores at offset 0 — but that byte is only injected for traits
    actually used with `*dyn` (`builder.py`: "Traits used only for static
    dispatch must not alter layout"), so it never applied to a statically
    dispatched trait. The restriction now sits at the `*dyn` itself.

    The two rarely meet in practice: a trait whose methods take `self` by value
    can only be implemented by a newtype, and one taking `*self` only by a
    struct, so a trait is naturally either dyn-able or newtype-able.
    """

    BASE = ("struct Tid(u8);\n"
            "trait Bump { fn bump(self) -> u8; }\n"
            "impl Bump for Tid { fn bump(self) -> u8 { return self.0 + 1; } }\n"
            "#[lowram]\nstatic mut V: Tid;\n")

    def check(self, body: str, decl: str = ""):
        src = self.BASE + decl + "#[entry]\nfn main() { " + body + " }"
        TypeChecker(HIRBuilder(source_file="t.r65").build_program(
            parse(src, "t.r65"))).check()

    def test_impl_is_accepted(self):
        self.check("")

    def test_method_is_callable(self):
        self.check("let t: Tid = 1; OUT = t.bump();",
                   decl="#[zeropage(0x30)]\nstatic mut OUT: u8;\n")

    @pytest.mark.parametrize("cast", [
        "&V as *dyn Bump",
        "&V as far *dyn Bump",
    ])
    def test_dyn_cast_is_rejected(self, cast):
        """An explicit cast must not be a way around the restriction — the
        implicit coercion already declines, requiring a struct pointee."""
        with pytest.raises(TypeCheckError) as exc:
            self.check(f"let d: far *dyn Bump = {cast};")
        assert "cannot form a '*dyn Bump' over newtype 'Tid'" in str(exc.value), str(exc.value)

    def test_dyn_coercion_still_rejected(self):
        with pytest.raises(TypeCheckError):
            self.check("let d: *dyn Bump = &V;")

    def test_clone_is_still_rejected(self):
        """Redundant rather than impossible — a newtype copies by assignment."""
        src = ("struct T2(u8);\nimpl Clone for T2 {}\n#[entry]\nfn main() { }")
        with pytest.raises(HIRError, match="cannot implement Clone"):
            TypeChecker(HIRBuilder(source_file="t.r65").build_program(
                parse(src, "t.r65"))).check()

    def test_pointer_self_still_rejected(self):
        """A `*self` trait cannot be implemented by a newtype — that is an ABI
        mismatch, not a dispatch one, and was already caught."""
        src = ("struct T3(u8);\ntrait B3 { fn b(*self) -> u8; }\n"
               "impl B3 for T3 { fn b(*self) -> u8 { return 1; } }\n"
               "#[entry]\nfn main() { }")
        with pytest.raises(HIRError, match="is a newtype"):
            TypeChecker(HIRBuilder(source_file="t.r65").build_program(
                parse(src, "t.r65"))).check()

    def test_structs_still_dyn_dispatch(self):
        """The relaxation must not disturb ordinary dynamic dispatch."""
        src = ("trait Sh { fn go(*self) -> u8; }\nstruct Bx { x: u8 }\n"
               "impl Sh for Bx { fn go(*self) -> u8 { return self.x; } }\n"
               "#[lowram]\nstatic mut BB: Bx;\n"
               "#[zeropage(0x30)]\nstatic mut OUT: u8;\n"
               "#[entry]\nfn main() { let d: *dyn Sh = &BB; OUT = d.go(); }")
        TypeChecker(HIRBuilder(source_file="t.r65").build_program(
            parse(src, "t.r65"))).check()


class TestNewtypeInConstContext:
    """Construction and casts inside a `const` initializer.

    The const evaluator runs on the AST, before the HIR builder desugars
    `TileId(x)` into a retype, so both spellings reached the const-fn check and
    were rejected — construction as "Function 'TileId' is not a const fn",
    naming a function nobody wrote, and the cast as an unknown target type. A
    `static` accepted both all along, as did the bare-payload form.
    """

    def test_construction(self):
        build_and_check(NEWTYPES + "const T: TileId = TileId(7);\nfn main() {}")

    def test_cast(self):
        build_and_check(NEWTYPES + "const T: TileId = 7 as TileId;\nfn main() {}")

    def test_payload_still_flows_in(self):
        build_and_check(NEWTYPES + "const T: TileId = 7;\nfn main() {}")

    def test_arithmetic_on_constructed_values(self):
        build_and_check(
            NEWTYPES + "const T: TileId = TileId(0x0F) | TileId(0x10);\nfn main() {}")

    def test_construction_is_still_range_checked(self):
        with pytest.raises(TypeCheckError, match="does not fit in type u8"):
            build_and_check(NEWTYPES + "const T: TileId = TileId(300);\nfn main() {}")

    def test_cast_truncates(self):
        """`as` is the truncating spelling here too: 300 & 0xFF."""
        hir_prog = build_and_check(
            NEWTYPES + "const T: TileId = 300 as TileId;\nfn main() {}")
        const = next(d for d in hir_prog.declarations
                     if getattr(d, 'name', None) == 'T')
        assert const.evaluated_value == 44, const.evaluated_value

    def test_wrong_arity_is_rejected(self):
        with pytest.raises(HIRError, match="takes exactly 1 value"):
            build_and_check(NEWTYPES + "const T: TileId = TileId(1, 2);\nfn main() {}")


class TestImplConstantKeepsTheNewtype:
    """An associated constant folds to a literal, which used to drop its type.

    `Color::WHITE` came back a bare u16: it flowed into any u16, mixed with
    other newtypes, and did not answer to `.0`. The declared type is nominal, so
    the fold has to carry it.
    """

    DECL = ("struct Color(u16);\nstruct Other(u16);\n"
            "impl Color { const WHITE: Color = 0x7FFF; const RAW: u16 = 5; }\n")

    def test_binds_to_the_newtype(self):
        build_and_check(self.DECL + "fn main() { let c: Color = Color::WHITE; }")

    def test_does_not_flow_out_to_the_payload(self):
        with pytest.raises(TypeCheckError, match="expected u16, found Color"):
            build_and_check(self.DECL + "fn main() { let n: u16 = Color::WHITE; }")

    def test_does_not_mix_with_another_newtype(self):
        with pytest.raises(TypeCheckError, match="expected Other, found Color"):
            build_and_check(self.DECL + "fn main() { let o: Other = Color::WHITE; }")

    def test_unwraps_with_field_access(self):
        build_and_check(self.DECL + "fn main() { let n: u16 = Color::WHITE.0; }")

    def test_inherits_operators(self):
        build_and_check(self.DECL + "fn main() { let c: Color = Color::WHITE + 1; }")

    def test_plain_constant_is_unaffected(self):
        build_and_check(self.DECL + "fn main() { let n: u16 = Color::RAW; }")


