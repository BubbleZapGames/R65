"""
Tests for the include! preprocessor.
"""

import pytest
import tempfile
import os
from pathlib import Path

from r65.compiler.frontend import parse, preprocess, PreprocessorError


class TestIncludeBasic:
    """Test basic include! functionality."""

    def test_include_file_not_found(self):
        """Test that include! with nonexistent file raises PreprocessorError."""
        source = '''
include!("nonexistent_file.r65")
'''
        program = parse(source, "test.r65")

        with pytest.raises(PreprocessorError) as exc_info:
            preprocess(program, "/tmp/test.r65")

        assert "include!" in str(exc_info.value)
        assert "file not found" in str(exc_info.value)
        assert "nonexistent_file.r65" in str(exc_info.value)

    def test_include_directory_error(self):
        """Test that include! with directory path raises PreprocessorError."""
        source = '''
include!(".")
'''
        program = parse(source, "test.r65")

        with pytest.raises(PreprocessorError) as exc_info:
            preprocess(program, "/tmp/test.r65")

        assert "include!" in str(exc_info.value)
        assert "not a file" in str(exc_info.value)

    def test_include_with_existing_file(self):
        """Test that include! with existing file succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the included file
            included_path = Path(tmpdir) / "included.r65"
            included_path.write_text('''
fn helper() {
    A = 42;
}
''')

            # Create main source that includes it
            main_source = '''
include!("included.r65")

fn main() {
    helper();
}
'''
            program = parse(main_source, "main.r65")
            main_file = str(Path(tmpdir) / "main.r65")

            # Should not raise
            processed = preprocess(program, main_file)

            # Should have merged declarations (include! replaced with included content)
            # The include statement is replaced, so we should have 2 functions
            assert len(processed.items) == 2


class TestIncludeNested:
    """Test nested include! functionality."""

    def test_nested_includes(self):
        """Test that nested includes work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deeply nested includes
            # level2.r65 -> has a const
            level2_path = Path(tmpdir) / "level2.r65"
            level2_path.write_text('''
const DEEP_VALUE: u8 = 99;
''')

            # level1.r65 -> includes level2.r65
            level1_path = Path(tmpdir) / "level1.r65"
            level1_path.write_text('''
include!("level2.r65")

const LEVEL1_VALUE: u8 = 50;
''')

            # main.r65 -> includes level1.r65
            main_source = '''
include!("level1.r65")

fn main() {
    A = DEEP_VALUE;
}
'''
            program = parse(main_source, "main.r65")
            main_file = str(Path(tmpdir) / "main.r65")

            processed = preprocess(program, main_file)

            # Should have: DEEP_VALUE const, LEVEL1_VALUE const, main function
            assert len(processed.items) == 3


class TestIncludeCircular:
    """Test circular include detection."""

    def test_direct_circular_include(self):
        """Test that direct circular include is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file that includes itself
            self_include_path = Path(tmpdir) / "self.r65"
            self_include_path.write_text('''
include!("self.r65")
''')

            program = parse('include!("self.r65")', "self.r65")

            with pytest.raises(PreprocessorError) as exc_info:
                preprocess(program, str(self_include_path))

            assert "circular include" in str(exc_info.value)

    def test_indirect_circular_include(self):
        """Test that indirect circular include is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create A -> B -> A cycle
            a_path = Path(tmpdir) / "a.r65"
            b_path = Path(tmpdir) / "b.r65"

            a_path.write_text('include!("b.r65")')
            b_path.write_text('include!("a.r65")')

            program = parse('include!("a.r65")', "main.r65")
            main_file = str(Path(tmpdir) / "main.r65")

            with pytest.raises(PreprocessorError) as exc_info:
                preprocess(program, main_file)

            assert "circular include" in str(exc_info.value)


class TestIncludeDuplicates:
    """Test duplicate include handling."""

    def test_duplicate_includes_ignored(self):
        """Test that duplicate includes of the same file are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a shared file
            shared_path = Path(tmpdir) / "shared.r65"
            shared_path.write_text('''
const SHARED: u8 = 1;
''')

            # Main includes shared twice
            main_source = '''
include!("shared.r65")
include!("shared.r65")

fn main() {
    A = SHARED;
}
'''
            program = parse(main_source, "main.r65")
            main_file = str(Path(tmpdir) / "main.r65")

            processed = preprocess(program, main_file)

            # Should only have SHARED once (duplicate ignored) + main
            assert len(processed.items) == 2


class TestIncludeRelativePaths:
    """Test relative path resolution."""

    def test_include_from_subdirectory(self):
        """Test include from a subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory structure
            subdir = Path(tmpdir) / "lib"
            subdir.mkdir()

            # Create file in subdirectory
            lib_file = subdir / "utils.r65"
            lib_file.write_text('''
fn util_func() {
    A = 1;
}
''')

            # Main includes from subdirectory
            main_source = '''
include!("lib/utils.r65")

fn main() {
    util_func();
}
'''
            program = parse(main_source, "main.r65")
            main_file = str(Path(tmpdir) / "main.r65")

            processed = preprocess(program, main_file)

            # Should have util_func and main
            assert len(processed.items) == 2

    def test_nested_include_relative_to_includer(self):
        """Test that nested includes are relative to the including file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure:
            # tmpdir/
            #   main.r65 -> includes lib/a.r65
            #   lib/
            #     a.r65 -> includes b.r65 (sibling, not ../b.r65)
            #     b.r65

            lib_dir = Path(tmpdir) / "lib"
            lib_dir.mkdir()

            b_path = lib_dir / "b.r65"
            b_path.write_text('const B_VAL: u8 = 2;')

            a_path = lib_dir / "a.r65"
            a_path.write_text('''
include!("b.r65")
const A_VAL: u8 = 1;
''')

            main_source = '''
include!("lib/a.r65")

fn main() {
    A = A_VAL;
}
'''
            program = parse(main_source, "main.r65")
            main_file = str(Path(tmpdir) / "main.r65")

            processed = preprocess(program, main_file)

            # Should have B_VAL, A_VAL, main
            assert len(processed.items) == 3


class TestIncludeSourceLocation:
    """Test source location tracking through includes."""

    def test_error_has_source_location(self):
        """Test that include! errors include source location."""
        source = '''
include!("missing.r65")
'''
        program = parse(source, "myfile.r65")

        with pytest.raises(PreprocessorError) as exc_info:
            preprocess(program, "/tmp/myfile.r65")

        error = exc_info.value
        assert error.source_loc is not None
        assert "myfile.r65" in str(error.source_loc)
