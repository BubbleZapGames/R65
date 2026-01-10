"""Tests for #[bank(auto)] feature."""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.errors import HIRError


class TestAutoBankParsing:
    """Tests for parsing #[bank(auto)] directive."""

    def test_parse_bank_auto(self):
        """#[bank(auto)] should parse correctly."""
        code = '''
#[bank(auto)]
far fn test() {}
'''
        ast = parse(code)
        assert len(ast.items) == 2
        bank_dir = ast.items[0]
        assert bank_dir.is_auto
        assert bank_dir.bank_number is None

    def test_parse_bank_explicit(self):
        """#[bank(n)] should parse correctly."""
        code = '''
#[bank(2)]
fn test() {}
'''
        ast = parse(code)
        bank_dir = ast.items[0]
        assert not bank_dir.is_auto
        assert bank_dir.bank_number == 2


class TestAutoBankValidation:
    """Tests for auto-bank validation rules."""

    def test_auto_bank_requires_far_function(self):
        """Functions in auto-bank mode must be far."""
        code = '''
#[bank(auto)]
fn test() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        with pytest.raises(HIRError) as exc_info:
            builder.build_program(ast)

        assert "auto-bank mode must be declared as 'far fn'" in str(exc_info.value)

    def test_auto_bank_far_function_succeeds(self):
        """Far functions in auto-bank mode should succeed."""
        code = '''
#[bank(auto)]
far fn test() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Function should have auto-bank attribute
        func = hir.declarations[0]
        assert func.bank_attr is not None
        assert func.bank_attr.is_auto

    def test_auto_bank_rom_static_requires_far(self):
        """#[rom] statics in auto-bank mode must be far."""
        code = '''
#[bank(auto)]
#[rom]
static DATA: u8 = 0;
'''
        ast = parse(code)
        builder = HIRBuilder()
        with pytest.raises(HIRError) as exc_info:
            builder.build_program(ast)

        assert "auto-bank mode must be declared as 'far static'" in str(exc_info.value)

    def test_auto_bank_far_rom_static_succeeds(self):
        """Far #[rom] statics in auto-bank mode should succeed."""
        code = '''
#[bank(auto)]
#[rom]
far static DATA: u8 = 0;
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Static should have auto-bank attribute
        static = hir.declarations[0]
        assert static.bank_attr is not None
        assert static.bank_attr.is_auto

    def test_auto_bank_ram_static_no_far_needed(self):
        """RAM statics in auto-bank mode don't need far (they're not in ROM)."""
        code = '''
#[bank(auto)]
#[ram]
static mut DATA: u8 = 0;
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # RAM static should not have bank attribute
        static = hir.declarations[0]
        assert static.bank_attr is None


class TestExplicitBankMode:
    """Tests for explicit #[bank(n)] mode."""

    def test_explicit_bank_near_function(self):
        """Near functions in explicit bank mode should succeed."""
        code = '''
#[bank(0)]
fn test() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        func = hir.declarations[0]
        assert func.bank_attr is not None
        assert func.bank_attr.bank_number == 0
        assert not func.bank_attr.is_auto

    def test_explicit_bank_far_function(self):
        """Far functions in explicit bank mode should succeed."""
        code = '''
#[bank(1)]
far fn test() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        func = hir.declarations[0]
        assert func.bank_attr is not None
        assert func.bank_attr.bank_number == 1

    def test_explicit_bank_rom_static(self):
        """#[rom] statics in explicit bank mode don't need far."""
        code = '''
#[bank(2)]
#[rom]
static DATA: u8 = 0;
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        static = hir.declarations[0]
        assert static.bank_attr is not None
        assert static.bank_attr.bank_number == 2


class TestDefaultBankMode:
    """Tests for default behavior (no #[bank] directive)."""

    def test_default_is_bank_0(self):
        """Default (no #[bank]) should be explicit bank 0."""
        code = '''
fn test() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        func = hir.declarations[0]
        assert func.bank_attr is not None
        assert func.bank_attr.bank_number == 0
        assert not func.bank_attr.is_auto

    def test_default_allows_near_functions(self):
        """Near functions should work with default bank (backward compatibility)."""
        code = '''
fn near_func() {}
fn another_func() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Both functions should compile without error
        assert len(hir.declarations) == 2


class TestBankModeTransitions:
    """Tests for switching between bank modes."""

    def test_auto_then_explicit(self):
        """Can switch from auto to explicit bank mode."""
        code = '''
#[bank(auto)]
far fn auto_fn() {}

#[bank(1)]
fn explicit_fn() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # First function should be auto
        func1 = hir.declarations[0]
        assert func1.bank_attr.is_auto

        # Second function should be explicit bank 1
        func2 = hir.declarations[1]
        assert func2.bank_attr.bank_number == 1
        assert not func2.bank_attr.is_auto

    def test_explicit_then_auto(self):
        """Can switch from explicit to auto bank mode."""
        code = '''
#[bank(0)]
fn explicit_fn() {}

#[bank(auto)]
far fn auto_fn() {}
'''
        ast = parse(code)
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # First function should be explicit bank 0
        func1 = hir.declarations[0]
        assert func1.bank_attr.bank_number == 0
        assert not func1.bank_attr.is_auto

        # Second function should be auto
        func2 = hir.declarations[1]
        assert func2.bank_attr.is_auto


class TestFarStaticParsing:
    """Tests for parsing 'far static' syntax."""

    def test_parse_far_static(self):
        """far static should parse correctly."""
        code = '''
#[rom]
far static DATA: u8 = 0;
'''
        ast = parse(code)
        static = ast.items[0]
        assert static.is_far

    def test_parse_near_static(self):
        """static without far should not be far."""
        code = '''
#[ram]
static mut DATA: u8 = 0;
'''
        ast = parse(code)
        static = ast.items[0]
        assert not static.is_far
