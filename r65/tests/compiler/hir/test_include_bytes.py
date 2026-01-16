"""
Tests for include_bytes! file validation.
"""

import pytest
import tempfile
import os

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder, HIRError


def test_include_bytes_file_not_found():
    """Test that include_bytes! with nonexistent file raises HIRError."""
    source = '''
static DATA: [u8; 100] = include_bytes!("nonexistent_file.bin");
'''
    program = parse(source, "test.r65")
    builder = HIRBuilder(source_file="/tmp/test.r65")

    with pytest.raises(HIRError) as exc_info:
        builder.build_program(program)

    assert "include_bytes!" in str(exc_info.value)
    assert "file not found" in str(exc_info.value)
    assert "nonexistent_file.bin" in str(exc_info.value)


def test_include_bytes_with_existing_file():
    """Test that include_bytes! with existing file succeeds."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
        f.write(b'\x00' * 100)
        temp_path = f.name

    try:
        temp_dir = os.path.dirname(temp_path)
        temp_name = os.path.basename(temp_path)

        source = f'''
static DATA: [u8; 100] = include_bytes!("{temp_name}");
'''
        program = parse(source, "test.r65")
        # Use temp_dir as the source file location
        builder = HIRBuilder(source_file=os.path.join(temp_dir, "test.r65"))

        # Should not raise
        hir_program = builder.build_program(program)
        assert hir_program is not None
    finally:
        os.unlink(temp_path)


def test_include_bytes_directory_error():
    """Test that include_bytes! with directory path raises HIRError."""
    source = '''
static DATA: [u8; 100] = include_bytes!(".");
'''
    program = parse(source, "test.r65")
    builder = HIRBuilder(source_file="/tmp/test.r65")

    with pytest.raises(HIRError) as exc_info:
        builder.build_program(program)

    assert "include_bytes!" in str(exc_info.value)
    assert "not a file" in str(exc_info.value)


def test_include_bytes_error_has_source_location():
    """Test that include_bytes! error includes source location."""
    source = '''
static DATA: [u8; 100] = include_bytes!("missing.bin");
'''
    program = parse(source, "myfile.r65")
    builder = HIRBuilder(source_file="/tmp/myfile.r65")

    with pytest.raises(HIRError) as exc_info:
        builder.build_program(program)

    error = exc_info.value
    # Error should have source location
    assert error.source_loc is not None
    assert "myfile.r65" in str(error.source_loc)
