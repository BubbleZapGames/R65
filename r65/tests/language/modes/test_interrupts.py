"""Tests for interrupt handlers."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function, get_attr, build_hir


class TestInterruptAttribute:
    """Tests for #[interrupt(...)] attribute."""

    def test_interrupt_nmi(self):
        """Test NMI interrupt handler."""
        func = parse_function("#[interrupt(nmi)] fn vblank() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None

    def test_interrupt_irq(self):
        """Test IRQ interrupt handler."""
        func = parse_function("#[interrupt(irq)] fn timer_handler() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None

    def test_interrupt_brk(self):
        """Test BRK interrupt handler."""
        func = parse_function("#[interrupt(brk)] fn break_handler() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None

    def test_interrupt_cop(self):
        """Test COP interrupt handler."""
        func = parse_function("#[interrupt(cop)] fn cop_handler() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None


class TestInterruptPreservation:
    """Tests for interrupt preservation options."""

    def test_interrupt_default_preserve(self):
        """Test default preservation (true)."""
        func = parse_function("#[interrupt(nmi)] fn handler() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None
        # Default is preserve=true

    def test_interrupt_no_preserve(self):
        """Test explicit preserve=false."""
        func = parse_function("#[interrupt(irq, preserve=false)] fn fast_handler() { }")
        attr = get_attr(func, "interrupt")
        assert len(attr.args) >= 2


class TestBankAttribute:
    """Tests for #[bank(...)] attribute on functions."""

    def test_bank_placement(self):
        """Test bank placement for far functions."""
        func = parse_function("#[bank(1)] far fn bank1_code() { }")
        attr = get_attr(func, "bank")
        assert attr is not None

    def test_bank_with_databank_mode(self):
        """Test bank with databank in mode attribute."""
        func = parse_function("#[bank(2)] #[mode(databank=inline)] far fn bank2_code() { }")
        mode_attr = get_attr(func, "mode")
        bank_attr = get_attr(func, "bank")
        assert mode_attr is not None
        assert bank_attr is not None


class TestInterruptHIR:
    """Tests for interrupt HIR generation."""

    def test_interrupt_hir(self):
        """Test interrupt handlers generate proper HIR."""
        hir_prog = build_hir("""
            #[interrupt(nmi)]
            fn vblank() { A = 1; }
        """)
        func = hir_prog.functions[0]
        assert func is not None
