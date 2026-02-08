"""Tests for storage class attributes."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.hir.errors import HIRError
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


class TestImplicitRom:
    """Tests for implicit ROM storage (immutable statics without storage attr)."""

    def test_immutable_static_is_rom(self):
        """Immutable static without attribute is ROM."""
        hir_prog = build_hir("static MESSAGE: [u8; 5] = [1, 2, 3, 4, 5];")
        assert len(hir_prog.statics) == 1
        static = hir_prog.statics[0]
        # storage_attr is None for ROM
        assert static.storage_attr is None
        # Should have bank_attr since it's ROM
        assert static.bank_attr is not None

    def test_mutable_static_requires_attribute(self):
        """Mutable static without attribute should error."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("static mut VAR: u8;")
        assert "requires explicit storage attribute" in str(exc_info.value)

    def test_immutable_static_ram_error(self):
        """Immutable static with #[ram] should error."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[ram] static DATA: u8 = 42;")
        assert "cannot use #ram storage" in str(exc_info.value)

    def test_immutable_static_zeropage_error(self):
        """Immutable static with #[zeropage] should error."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[zeropage] static DATA: u8 = 42;")
        assert "cannot use #zeropage storage" in str(exc_info.value)

    def test_immutable_static_hw_allowed(self):
        """Immutable static with #[hw] is OK for read-only hardware registers."""
        hir_prog = build_hir("#[hw(0x4212)] static HVBJOY: u8;")
        assert len(hir_prog.statics) == 1
        static = hir_prog.statics[0]
        assert static.storage_attr is not None
        assert static.storage_attr.storage_kind.value == "hw"

    def test_rom_attribute_unknown(self):
        """#[rom] attribute should be rejected as unknown."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[rom] static DATA: u8 = 42;")
        assert "Unknown attribute" in str(exc_info.value)


class TestConstTypeRestrictions:
    """Tests for const type restrictions (only primitives allowed)."""

    def test_const_array_rejected(self):
        """Const cannot have array type."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("const TABLE: [u8; 3] = [1, 2, 3];")
        assert "cannot have array type" in str(exc_info.value)

    def test_const_struct_rejected(self):
        """Const cannot have struct type."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("struct Point { x: u8, y: u8 } const ORIGIN: Point = 0;")
        assert "cannot have struct type" in str(exc_info.value)

    def test_const_primitive_allowed(self):
        """Const with primitive type should work."""
        hir_prog = build_hir("const MAX_VALUE: u8 = 255;")
        assert len(hir_prog.constants) == 1
        assert hir_prog.constants[0].evaluated_value == 255


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


