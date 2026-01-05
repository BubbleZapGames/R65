"""Tests for storage class attributes."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_static, parse_program, get_attr, build_hir


class TestZeropage:
    """Tests for #[zeropage] attribute."""

    def test_zeropage_explicit_address(self):
        """Test zeropage with explicit address."""
        static = parse_static("#[zeropage(0x10)] static mut VAR: u8;")
        attr = get_attr(static, "zeropage")
        assert attr is not None
        assert len(attr.args) == 1

    def test_zeropage_auto_address(self):
        """Test zeropage without address (auto-allocated)."""
        static = parse_static("#[zeropage] static mut VAR: u8;")
        attr = get_attr(static, "zeropage")
        assert attr is not None


class TestLowram:
    """Tests for #[lowram] attribute."""

    def test_lowram_explicit_address(self):
        """Test lowram with explicit address."""
        static = parse_static("#[lowram(0x0200)] static mut BUFFER: [u8; 256];")
        attr = get_attr(static, "lowram")
        assert attr is not None

    def test_lowram_auto_address(self):
        """Test lowram without address."""
        static = parse_static("#[lowram] static mut VAR: u16;")
        attr = get_attr(static, "lowram")
        assert attr is not None


class TestRam:
    """Tests for #[ram] attribute."""

    def test_ram_declaration(self):
        """Test ram static declaration."""
        static = parse_static("#[ram] static mut WORK_RAM: [u8; 1024];")
        attr = get_attr(static, "ram")
        assert attr is not None

    def test_ram_with_initializer(self):
        """Test ram with initializer."""
        static = parse_static("#[ram] static mut DATA: u8 = 42;")
        assert static.initializer is not None


class TestRom:
    """Tests for #[rom] attribute."""

    def test_rom_declaration(self):
        """Test rom static declaration."""
        static = parse_static("#[rom(0x8000)] static GRAPHICS: [u8; 256];")
        attr = get_attr(static, "rom")
        assert attr is not None


class TestHardware:
    """Tests for #[hw] attribute."""

    def test_hw_register(self):
        """Test hardware register declaration."""
        static = parse_static("#[hw(0x2100)] static mut INIDISP: u8;")
        attr = get_attr(static, "hw")
        assert attr is not None

    def test_hw_multiple_registers(self):
        """Test multiple hardware registers."""
        prog = parse_program("""
            #[hw(0x2100)] static mut INIDISP: u8;
            #[hw(0x4212)] static mut HVBJOY: u8;
        """)
        assert len(prog.items) == 2


class TestStackDirective:
    """Tests for #[stack] directive."""

    def test_stack_directive(self):
        """Test stack reservation directive."""
        prog = parse_program("#[stack(0x1F00, 0x1FFF)]")
        assert len(prog.items) == 1


class TestStorageHIR:
    """Tests for storage HIR generation."""

    def test_storage_hir(self):
        """Test storage classes generate proper HIR."""
        hir_prog = build_hir("""
            #[zeropage(0x10)] static mut ZP_VAR: u8;
            #[ram] static mut RAM_VAR: u16;
            #[hw(0x2100)] static mut HW_REG: u8;
        """)
        assert len(hir_prog.statics) >= 3
