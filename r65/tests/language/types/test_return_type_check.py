# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""`return` is checked against the declared return type.

Return values used to be checked only for internal consistency — the value's own
type was computed and then discarded, never compared against the function's
declared return type. That let a newtype launder itself back into its payload
(`fn f() -> i16 { return some_q10; }`), which is the one direction newtype
opacity is supposed to close.

Return is assignment-shaped, so it uses the same rule as `let`: `assignable`,
plus the declared type as literal context. `let r: i8 = 0xFF;` was already an
error; `return 0xFF;` from a `-> i8` function now is too.
"""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError


def check(source: str):
    program = parse(source, "test.r65")
    TypeChecker(HIRBuilder(source_file="test.r65").build_program(program)).check()


NEWTYPES = "struct Q(i16);\nstruct R(i16);\n"


class TestNewtypeOpacity:
    """The hole this closes: a newtype returned as its payload, or as a sibling."""

    def test_newtype_cannot_return_as_its_payload(self):
        with pytest.raises(TypeCheckError, match=r"returning 'Q'.*'-> i16'"):
            check(NEWTYPES + "fn f() -> i16 { let q: Q = 5; return q; }\nfn main(){}")

    def test_newtype_cannot_return_as_a_sibling_newtype(self):
        with pytest.raises(TypeCheckError, match=r"returning 'R'.*'-> Q'"):
            check(NEWTYPES + "fn f() -> Q { let r: R = 5; return r; }\nfn main(){}")

    def test_payload_may_still_return_as_the_newtype(self):
        """Transparent in — construction from the payload stays implicit."""
        check(NEWTYPES + "fn f() -> Q { let n: i16 = 5; return n; }\nfn main(){}")

    def test_newtype_returns_as_itself(self):
        check(NEWTYPES + "fn f() -> Q { let q: Q = 5; return q; }\nfn main(){}")

    def test_explicit_cast_out_is_still_allowed(self):
        check(NEWTYPES + "fn f() -> i16 { let q: Q = 5; return q.0; }\nfn main(){}")

    def test_multi_return_checks_each_position(self):
        """Value 1 maps to A, value 2 to X — each gets its own declared type."""
        with pytest.raises(TypeCheckError, match=r"value 2"):
            check(NEWTYPES + "fn f() -> u8, i16 { let q: Q = 5; return 1, q; }\n"
                             "fn main(){}")

    def test_multi_return_accepts_matching_positions(self):
        check(NEWTYPES + "fn f() -> u8, Q { let q: Q = 5; return 1, q; }\n"
                         "fn main(){}")


class TestLiteralRange:
    """The declared type is literal context, exactly as in a `let`."""

    def test_out_of_range_literal_rejected(self):
        with pytest.raises(TypeCheckError, match="does not fit in type i8"):
            check("fn f() -> i8 { return 0xFF; }\nfn main(){}")

    def test_matching_signed_literal_accepted(self):
        check("fn f() -> i8 { return -1; }\nfn main(){}")

    def test_same_literal_fits_an_unsigned_return(self):
        check("fn f() -> u8 { return 0xFF; }\nfn main(){}")


class TestStillAccepted:
    """Shapes that must keep compiling — this check is not meant to tighten
    integer conversion, only to apply the existing assignment rule to `return`."""

    @pytest.mark.parametrize("src", [
        "fn f() -> u8 { let n: u8 = 5; return n; }",
        "fn f() -> u8 { return 5; }",
        "fn f() -> u16 { let n: u8 = 5; return n as u16; }",
        "fn f() -> u8 { let w: u16 = 300; return w as u8; }",
        "fn f() -> u8, u16 { return 3, 300; }",
        "fn f() { return; }",
        "fn f() -> ! { loop { } }",
        "fn f() -> u16 { return X; }",
        "fn f() -> bool { let a: u8 = 1; return a > 0; }",
    ])
    def test_accepted(self, src):
        check(src + "\nfn main(){}")



class TestReturnArity:
    """`return` must hand back as many values as the signature promises.

    Only the counted forms. A bare `return;` is the implicit-A form — the value
    is already in the register, so there is nothing to count — and a `-> !`
    function has no declared type to count against.

    Unchecked, the caller believes the signature: `let a, b = f();` against a
    `return 1;` reads a register the callee never wrote. The caller side already
    rejects the mirror image (binding two names to a single-value call).
    """

    def test_too_few_values(self):
        with pytest.raises(TypeCheckError, match="which returns 2"):
            check("fn f() -> u8, u8 { return 1; }\nfn main() { }")

    def test_too_many_values(self):
        with pytest.raises(TypeCheckError, match="which returns 1"):
            check("fn f() -> u8 { return 1, 2; }\nfn main() { }")

    def test_error_quotes_a_spelling_that_parses(self):
        """`-> u8, u8`, not the parenthesized form the language does not accept."""
        with pytest.raises(TypeCheckError) as exc:
            check("fn f() -> u8, u8 { return 1; }\nfn main() { }")
        assert "'-> u8, u8'" in str(exc.value), str(exc.value)

    def test_hint_points_at_the_implicit_a_form(self):
        with pytest.raises(TypeCheckError) as exc:
            check("fn f() -> u8, u8 { return 1; }\nfn main() { }")
        assert "bare 'return;'" in (exc.value.hint or ""), exc.value.hint

    def test_matching_arity_accepted(self):
        check("fn f() -> u8, u16 { return 1, 2; }\nfn main() { }")

    def test_bare_return_is_the_implicit_a_form(self):
        check("fn f() -> u8 { A = 5; return; }\nfn main() { }")

    def test_bare_return_from_a_multi_return_signature(self):
        check("fn f() -> u8, u8 { A = 5; B = 6; return; }\nfn main() { }")

    def test_never_returning_function_is_unchecked(self):
        check("fn f() -> ! { loop { } }\nfn main() { }")
