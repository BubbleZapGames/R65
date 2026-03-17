# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for storage class attributes."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.hir.errors import HIRError
from r65.tests.language.common import parse_static, parse_program, get_attr, build_hir, build_hir_with_warnings


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
    """Tests for const type restrictions."""

    def test_const_array_allowed(self):
        """Const can have array type (evaluated at compile time)."""
        hir_prog = build_hir("const TABLE: [u8; 3] = [1, 2, 3];")
        assert len(hir_prog.constants) == 1
        assert hir_prog.constants[0].evaluated_value == [1, 2, 3]

    def test_const_struct_allowed(self):
        """Const can have struct type (evaluated to dict at compile time)."""
        hir_prog = build_hir(
            "struct Point { x: u8, y: u8 } "
            "const ORIGIN: Point = Point { x: 0, y: 0 };"
        )
        assert len(hir_prog.constants) == 1
        assert hir_prog.constants[0].evaluated_value == {'x': 0, 'y': 0}

    def test_const_primitive_allowed(self):
        """Const with primitive type should work."""
        hir_prog = build_hir("const MAX_VALUE: u8 = 255;")
        assert len(hir_prog.constants) == 1
        assert hir_prog.constants[0].evaluated_value == 255

    def test_const_struct_field_access_folds(self):
        """Const struct field access folds to integer literal at HIR level."""
        from r65.compiler.hir.nodes import HIRIntegerLiteral
        hir_prog = build_hir(
            "struct Point { x: u8, y: u8 } "
            "const ORIGIN: Point = Point { x: 10, y: 20 }; "
            "fn test() { let val: u8 = ORIGIN.x; }"
        )
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, HIRIntegerLiteral)
        assert let_stmt.initializer.value == 10

    def test_const_array_of_structs(self):
        """Const array of struct literals evaluates to list of dicts."""
        hir_prog = build_hir(
            "struct Rect { x: u8, y: u8, w: u8, h: u8 } "
            "const RECTS: [Rect; 2] = [Rect { x: 0, y: 0, w: 8, h: 8 }, "
            "Rect { x: 10, y: 20, w: 16, h: 16 }];"
        )
        assert len(hir_prog.constants) == 1
        val = hir_prog.constants[0].evaluated_value
        assert isinstance(val, list)
        assert len(val) == 2
        assert val[0] == {'x': 0, 'y': 0, 'w': 8, 'h': 8}
        assert val[1] == {'x': 10, 'y': 20, 'w': 16, 'h': 16}

    def test_const_array_of_structs_field_access_folds(self):
        """CONST_ARRAY[0].field folds to integer literal."""
        from r65.compiler.hir.nodes import HIRIntegerLiteral
        hir_prog = build_hir(
            "struct Rect { x: u8, y: u8, w: u8, h: u8 } "
            "const RECTS: [Rect; 2] = [Rect { x: 0, y: 0, w: 8, h: 8 }, "
            "Rect { x: 10, y: 20, w: 16, h: 16 }]; "
            "fn test() { let val: u8 = RECTS[1].w; }"
        )
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, HIRIntegerLiteral)
        assert let_stmt.initializer.value == 16

    def test_const_nested_struct(self):
        """Nested const struct field access folds correctly."""
        hir_prog = build_hir(
            "struct Vec2 { x: u8, y: u8 } "
            "struct Sprite { pos: Vec2, tile: u8 } "
            "const PLAYER: Sprite = Sprite { pos: Vec2 { x: 5, y: 10 }, tile: 42 }; "
            "fn test() { let t: u8 = PLAYER.tile; }"
        )
        from r65.compiler.hir.nodes import HIRIntegerLiteral
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, HIRIntegerLiteral)
        assert let_stmt.initializer.value == 42

    def test_const_struct_in_const_expr(self):
        """Const struct field can be used in another const definition."""
        hir_prog = build_hir(
            "struct Config { width: u8, height: u8 } "
            "const CFG: Config = Config { width: 32, height: 28 }; "
            "const AREA: u16 = (CFG.width as u16) * (CFG.height as u16);"
        )
        consts = {c.name: c for c in hir_prog.constants}
        assert consts['AREA'].evaluated_value == 32 * 28


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

    def test_hw_initializer_warning(self):
        """Initializer on #[hw] static should warn and be stripped."""
        hir_prog, warnings = build_hir_with_warnings(
            "#[hw(0x2100)] static mut INIDISP: u8 = 0x0F;"
        )
        assert len(hir_prog.statics) == 1
        static = hir_prog.statics[0]
        assert static.initializer is None
        assert len(warnings) == 1
        assert "volatile hardware register" in warnings[0]
        assert "initializer ignored" in warnings[0]

    def test_hw_no_initializer_no_warning(self):
        """#[hw] static without initializer should produce no warnings."""
        hir_prog, warnings = build_hir_with_warnings(
            "#[hw(0x2100)] static mut INIDISP: u8;"
        )
        assert len(hir_prog.statics) == 1
        assert len(warnings) == 0


class TestStackDirective:
    """Tests for #[stack] directive."""

    def test_stack_directive(self):
        """Test stack reservation directive."""
        prog = parse_program("#[stack(0x1F00, 0x1FFF)]")
        assert len(prog.items) == 1


