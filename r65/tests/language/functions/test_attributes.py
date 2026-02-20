"""Tests for function attributes."""

from r65.tests.language.common import parse_function, get_attr


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


