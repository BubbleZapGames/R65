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


class TestBankDirective:
    """Tests for #[bank(...)] directive."""

    def test_bank_placement(self):
        """Test bank directive for far functions."""
        from r65.compiler.frontend.parser import parse
        prog = parse("#[bank(1)] far fn bank1_code() { }")

        # First item is BankDirective, second is function
        assert len(prog.items) == 2
        assert isinstance(prog.items[0], ast.BankDirective)
        assert prog.items[0].bank_number == 1
        assert isinstance(prog.items[1], ast.FunctionDecl)
        assert prog.items[1].is_far == True

    def test_bank_with_databank_mode(self):
        """Test bank directive with databank in mode attribute."""
        from r65.compiler.frontend.parser import parse
        prog = parse("#[bank(2)] #[mode(databank=inline)] far fn bank2_code() { }")

        # First item is BankDirective
        assert len(prog.items) == 2
        assert isinstance(prog.items[0], ast.BankDirective)
        assert prog.items[0].bank_number == 2

        # Second item is function with mode attribute
        func = prog.items[1]
        assert isinstance(func, ast.FunctionDecl)
        mode_attr = get_attr(func, "mode")
        assert mode_attr is not None


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
