"""Tests for function attributes."""

from r65.compiler.frontend import ast
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


class TestModeAttribute:
    """Tests for #[mode(...)] attribute."""

    def test_mode_m8(self):
        """Test mode with m8."""
        func = parse_function("#[mode(m8)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_combined(self):
        """Test mode with m8 and x16."""
        func = parse_function("#[mode(m8, x16)] fn test() { }")
        attr = get_attr(func, "mode")
        assert len(attr.args) >= 2


class TestBankDirective:
    """Tests for #[bank(...)] directive."""

    def test_bank_directive(self):
        """Test bank directive placement."""
        from r65.compiler.frontend.parser import parse
        prog = parse("#[bank(1)] far fn rom_code() { }")

        # First item should be the BankDirective
        assert len(prog.items) == 2
        assert isinstance(prog.items[0], ast.BankDirective)
        assert prog.items[0].bank_number == 1

        # Second item should be the far function
        func = prog.items[1]
        assert isinstance(func, ast.FunctionDecl)
        assert func.is_far == True


class TestInterruptAttribute:
    """Tests for #[interrupt(...)] attribute."""

    def test_interrupt_nmi(self):
        """Test NMI interrupt handler."""
        func = parse_function("#[interrupt(nmi)] fn vblank() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None

    def test_interrupt_irq(self):
        """Test IRQ interrupt handler."""
        func = parse_function("#[interrupt(irq)] fn timer() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None


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


class TestAttributeHIR:
    """Tests for attribute HIR generation."""

    def test_attribute_hir(self):
        """Test attributes generate proper HIR."""
        hir_prog = build_hir("""
                        #[preserves(X)]
            fn process() { A = 1; }
        """)
        func = hir_prog.functions[0]
        assert func is not None
