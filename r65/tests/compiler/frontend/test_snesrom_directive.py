# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for #[snesrom(...)] directive parsing."""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend import ast
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.errors import ParseError


class TestSnesRomDirectiveParsing:
    """Tests for parsing the #[snesrom(...)] directive."""

    def test_minimal_snesrom(self):
        """Test minimal snesrom with just required name."""
        source = '#[snesrom(name="MY GAME")]'
        prog = parse(source)

        assert len(prog.items) == 1
        directive = prog.items[0]
        assert isinstance(directive, ast.SnesRomDirective)
        assert directive.name == "MY GAME"
        # Check defaults
        assert directive.id == "SNES"
        assert directive.cartridge_type == 0x00
        assert directive.sram_size == 0x00
        assert directive.country == 0x01
        assert directive.version == 0x00
        assert directive.lorom == True  # Default
        assert directive.hirom == False
        assert directive.slowrom == True  # Default
        assert directive.fastrom == False

    def test_snesrom_with_all_options(self):
        """Test snesrom with all options specified."""
        source = '''#[snesrom(
            name="SUPER GAME",
            id="ABCD",
            cartridge_type=0x02,
            sram_size=0x03,
            country=0x00,
            version=0x10,
            hirom,
            fastrom
        )]'''
        prog = parse(source)

        directive = prog.items[0]
        assert isinstance(directive, ast.SnesRomDirective)
        assert directive.name == "SUPER GAME"
        assert directive.id == "ABCD"
        assert directive.cartridge_type == 0x02
        assert directive.sram_size == 0x03
        assert directive.country == 0x00
        assert directive.version == 0x10
        assert directive.lorom == False  # Not specified, hirom is
        assert directive.hirom == True
        assert directive.slowrom == False  # Not specified, fastrom is
        assert directive.fastrom == True

    def test_snesrom_lorom_slowrom(self):
        """Test explicit lorom and slowrom flags."""
        source = '#[snesrom(name="TEST", lorom, slowrom)]'
        prog = parse(source)

        directive = prog.items[0]
        assert directive.lorom == True
        assert directive.hirom == False
        assert directive.slowrom == True
        assert directive.fastrom == False

    def test_snesrom_exhirom(self):
        """Test exhirom flag."""
        source = '#[snesrom(name="BIG GAME", exhirom)]'
        prog = parse(source)

        directive = prog.items[0]
        assert directive.lorom == False
        assert directive.hirom == False
        assert directive.exhirom == True

    def test_snesrom_missing_name_error(self):
        """Test that missing name raises error."""
        source = '#[snesrom(lorom)]'
        with pytest.raises(ParseError, match="requires 'name' parameter"):
            parse(source)


class TestSnesRomDirectiveHIR:
    """Tests for HIR building with #[snesrom(...)] directive."""

    def test_hir_snesrom_config(self):
        """Test that snesrom config is passed to HIR program."""
        source = '''
        #[snesrom(name="MY GAME", version=0x05, fastrom)]
                fn main() { A = 1; }
        '''
        prog = parse(source)
        builder = HIRBuilder()
        hir_prog = builder.build_program(prog)

        assert hir_prog.snesrom_config is not None
        cfg = hir_prog.snesrom_config
        assert cfg.name == "MY GAME"
        assert cfg.version == 0x05
        assert cfg.fastrom == True
        assert cfg.slowrom == False
        assert cfg.lorom == True  # Default

    def test_hir_no_snesrom_config(self):
        """Test that snesrom_config is None when not specified."""
        source = '''
                fn main() { A = 1; }
        '''
        prog = parse(source)
        builder = HIRBuilder()
        hir_prog = builder.build_program(prog)

        assert hir_prog.snesrom_config is None

    def test_hir_snesrom_with_other_directives(self):
        """Test snesrom works alongside other directives."""
        source = '''
        #[snesrom(name="GAME", hirom)]
        #[stack(0x1F00, 0x1FFF)]
        #[bank(1)]
                far fn main() { A = 1; }
        '''
        prog = parse(source)
        builder = HIRBuilder()
        hir_prog = builder.build_program(prog)

        assert hir_prog.snesrom_config is not None
        assert hir_prog.snesrom_config.hirom == True
        assert hir_prog.stack_attr is not None
        assert len(hir_prog.functions) == 1
