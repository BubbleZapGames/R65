# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for include_bytes! file validation, type inference, and edge cases.
"""

import pytest
import tempfile
import os

from r65.compiler.frontend import parse
from r65.compiler.frontend.parser import ParseError
from r65.compiler.hir import HIRBuilder, HIRError


@pytest.fixture
def bin_file_256():
    """Create a 256-byte temp binary file, yield (dir, name), cleanup."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
        f.write(b'\xAB' * 256)
        path = f.name
    yield os.path.dirname(path), os.path.basename(path)
    os.unlink(path)


@pytest.fixture
def bin_file_100():
    """Create a 100-byte temp binary file, yield (dir, name), cleanup."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
        f.write(b'\x00' * 100)
        path = f.name
    yield os.path.dirname(path), os.path.basename(path)
    os.unlink(path)


@pytest.fixture
def empty_bin_file():
    """Create an empty temp binary file, yield (dir, name), cleanup."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
        path = f.name
    yield os.path.dirname(path), os.path.basename(path)
    os.unlink(path)


def _build(source, source_dir, filename="test.r65"):
    program = parse(source, filename)
    builder = HIRBuilder(source_file=os.path.join(source_dir, filename))
    return builder.build_program(program)


class TestFileValidation:
    """Tests for include_bytes! file resolution and error reporting."""

    def test_file_not_found(self):
        source = 'static DATA: [u8; 100] = include_bytes!("nonexistent_file.bin");'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file="/tmp/test.r65")

        with pytest.raises(HIRError, match="file not found"):
            builder.build_program(program)

    def test_existing_file(self, bin_file_100):
        tmp_dir, tmp_name = bin_file_100
        source = f'static DATA: [u8; 100] = include_bytes!("{tmp_name}");'
        hir = _build(source, tmp_dir)
        assert hir is not None

    def test_directory_path_rejected(self):
        source = 'static DATA: [u8; 100] = include_bytes!(".");'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file="/tmp/test.r65")

        with pytest.raises(HIRError, match="file not found"):
            builder.build_program(program)

    def test_error_has_source_location(self):
        source = 'static DATA: [u8; 100] = include_bytes!("missing.bin");'
        program = parse(source, "myfile.r65")
        builder = HIRBuilder(source_file="/tmp/myfile.r65")

        with pytest.raises(HIRError) as exc_info:
            builder.build_program(program)

        assert exc_info.value.source_loc is not None
        assert "myfile.r65" in str(exc_info.value.source_loc)


class TestTypeInference:
    """Tests for static type inference from include_bytes!."""

    def test_inferred_type(self, bin_file_256):
        tmp_dir, tmp_name = bin_file_256
        source = f'static DATA = include_bytes!("{tmp_name}");'
        hir = _build(source, tmp_dir)
        static_decl = hir.declarations[0]
        assert static_decl.name == 'DATA'
        assert str(static_decl.var_type) == '[u8; 256]'

    def test_inferred_type_file_not_found(self):
        source = 'static DATA = include_bytes!("nonexistent.bin");'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file="/tmp/test.r65")

        with pytest.raises(HIRError, match="file not found"):
            builder.build_program(program)

    def test_inferred_type_empty_file(self, empty_bin_file):
        tmp_dir, tmp_name = empty_bin_file
        source = f'static DATA = include_bytes!("{tmp_name}");'
        hir = _build(source, tmp_dir)
        static_decl = hir.declarations[0]
        assert str(static_decl.var_type) == '[u8; 0]'

    def test_explicit_type_still_works(self, bin_file_100):
        tmp_dir, tmp_name = bin_file_100
        source = f'static DATA: [u8; 100] = include_bytes!("{tmp_name}");'
        hir = _build(source, tmp_dir)
        static_decl = hir.declarations[0]
        assert str(static_decl.var_type) == '[u8; 100]'

    def test_far_static_inferred_type(self, bin_file_256):
        tmp_dir, tmp_name = bin_file_256
        source = f'far static DATA = include_bytes!("{tmp_name}");'
        program = parse(source, "test.r65")
        # far static requires auto-bank mode or explicit bank, so just check parsing
        decl = program.items[0]
        assert decl.is_far is True
        assert decl.var_type is None

    def test_non_include_bytes_requires_type(self):
        source = 'static DATA = 42;'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file="/tmp/test.r65")

        with pytest.raises(HIRError, match="requires a type annotation"):
            builder.build_program(program)

    def test_string_literal_requires_type(self):
        source = 'static DATA = "hello";'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file="/tmp/test.r65")

        with pytest.raises(HIRError, match="requires a type annotation"):
            builder.build_program(program)

    def test_array_fill_requires_type(self):
        source = 'static DATA = [0; 256];'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file="/tmp/test.r65")

        with pytest.raises(HIRError, match="requires a type annotation"):
            builder.build_program(program)


class TestMutableStaticRejection:
    """Tests that include_bytes! is rejected on mutable (RAM) statics."""

    def test_ram_static_mut_rejected(self, bin_file_256):
        tmp_dir, tmp_name = bin_file_256
        source = f'#[ram] static mut DATA: [u8; 256] = include_bytes!("{tmp_name}");'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file=os.path.join(tmp_dir, "test.r65"))

        with pytest.raises(HIRError, match="cannot be used with 'static mut'"):
            builder.build_program(program)

    def test_zeropage_static_mut_rejected(self, bin_file_100):
        tmp_dir, tmp_name = bin_file_100
        source = f'#[zeropage] static mut DATA: [u8; 100] = include_bytes!("{tmp_name}");'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file=os.path.join(tmp_dir, "test.r65"))

        with pytest.raises(HIRError, match="cannot be used with 'static mut'"):
            builder.build_program(program)

    def test_lowram_static_mut_rejected(self, bin_file_100):
        tmp_dir, tmp_name = bin_file_100
        source = f'#[lowram] static mut DATA: [u8; 100] = include_bytes!("{tmp_name}");'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file=os.path.join(tmp_dir, "test.r65"))

        with pytest.raises(HIRError, match="cannot be used with 'static mut'"):
            builder.build_program(program)

    def test_inferred_type_static_mut_rejected(self, bin_file_256):
        tmp_dir, tmp_name = bin_file_256
        source = f'#[ram] static mut DATA = include_bytes!("{tmp_name}");'
        program = parse(source, "test.r65")
        builder = HIRBuilder(source_file=os.path.join(tmp_dir, "test.r65"))

        with pytest.raises(HIRError, match="cannot be used with 'static mut'"):
            builder.build_program(program)


class TestParsingEdgeCases:
    """Tests for grammar edge cases around type-omitted statics."""

    def test_static_no_type_no_init_rejected(self):
        with pytest.raises(ParseError):
            parse("#[ram] static mut X;")

    def test_pointer_static_requires_type(self):
        with pytest.raises(ParseError):
            parse("static *PTR = 0x2000;")

    def test_bank_directive_with_inferred_type(self, bin_file_256):
        """#[bank(2)] is a directive (not an attribute on the static), followed by static."""
        tmp_dir, tmp_name = bin_file_256
        source = f'#[bank(2)]\nstatic DATA = include_bytes!("{tmp_name}");'
        program = parse(source, "test.r65")
        # bank(2) is parsed as a BankDirective, static is a separate item
        static_decl = program.items[1]
        assert static_decl.var_type is None
        assert static_decl.name == 'DATA'
