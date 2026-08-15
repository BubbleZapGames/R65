# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for function attributes."""

import pytest

from r65.compiler.typeck.type_checker import TypeChecker
from r65.tests.language.common import parse_function, get_attr, build_hir


class TestPreservesAttribute:
    """Tests for #[preserves(...)] attribute."""

    def test_preserves_single(self):
        """Test preserves with single register."""
        func = parse_function("#[preserves(X)] fn test() { }")
        attr = get_attr(func, "preserves")
        assert attr is not None
        assert len(attr.args) == 1

    def test_preserves_multiple(self):
        """Test preserves with multiple registers."""
        func = parse_function("#[preserves(X, Y, A)] fn test() { }")
        attr = get_attr(func, "preserves")
        assert len(attr.args) == 3


class TestEntryAttribute:
    """Tests for #[entry] attribute."""

    def test_entry_attribute(self):
        """Test entry point function."""
        func = parse_function("#[entry] fn main() { }")
        attr = get_attr(func, "entry")
        assert attr is not None


class TestMultipleAttributes:
    """Tests for multiple attributes on functions."""

    def test_combined_attributes(self):
        """Test function with multiple attributes."""
        func = parse_function("""
            #[preserves(X, Y)]
            fn complex() { }
        """)
        # In the simplified mode system, mode is inferred from parameters,
        # not specified via attribute. Only preserves attribute should be present.
        assert get_attr(func, "mode") is None  # Mode is no longer an attribute
        assert get_attr(func, "preserves") is not None




class TestPreservesAccumulatorConflict:
    """`#[preserves(A)]` on a function that returns a value is rejected.

    Both want the same register. The save/restore is a bracket around the whole
    body, so the restore runs after the result is already in A:

        bump:  PHA        ; preserve A
               INC A      ; the result
               PLA        ; restore A -- overwrites the result
               RTS

    That compiled silently and returned the argument unchanged.
    """

    def check(self, source: str):
        TypeChecker(build_hir(source + "\nfn main() { }")).check()

    @pytest.mark.parametrize("signature", [
        "#[preserves(A)] fn f() -> u8 { return 1; }",
        "#[preserves(A)] fn f() -> u16 { return 1; }",
        "#[preserves(A)] fn f() -> bool { return true; }",
        "#[preserves(A, X)] fn f() -> u8 { X = 1; return 1; }",
        "#[preserves(X, A, Y)] fn f() -> u8 { X = 1; return 1; }",
        "#[preserves(A)] fn f() -> u8, u16 { return 1, 2; }",
    ])
    def test_rejected_when_a_value_is_returned(self, signature):
        with pytest.raises(Exception) as exc:
            self.check(signature)
        assert "A cannot be in #[preserves" in str(exc.value)

    def test_rejected_for_a_newtype_return(self):
        with pytest.raises(Exception) as exc:
            self.check("struct Q(i16);\n#[preserves(A)] fn f() -> Q { return 1; }")
        assert "A cannot be in #[preserves" in str(exc.value)

    @pytest.mark.parametrize("signature", [
        "#[preserves(A)] fn f() { A = 1; }",
        "#[preserves(A)] fn f() -> ! { loop { } }",
        "#[preserves(X)] fn f() -> u8 { X = 1; return 1; }",
        "#[preserves(X, Y)] fn f() -> u8 { X = 1; return 1; }",
        "#[preserves(STATUS)] fn f() -> u8 { return 1; }",
        "#[preserves(D, DBR)] fn f() -> u8 { return 1; }",
        "fn f() -> u8 { return 1; }",
    ])
    def test_accepted(self, signature):
        """A void function has no result to lose, a `-> !` function never
        reaches its epilogue, and the other registers do not carry the result."""
        self.check(signature)
