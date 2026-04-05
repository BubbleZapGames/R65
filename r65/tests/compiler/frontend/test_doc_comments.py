# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for doc comment parsing (///, //!, /** */, /*! */).

Validates:
- Outer doc comments (///) attach to the following declaration
- Inner doc comments (//!) attach to the program/file
- Block doc comments (/** */, /*! */) are handled
- Multi-line doc comments are concatenated
- Regular comments are still ignored
- Edge cases: ////, /**/, etc.
"""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend import ast


class TestOuterDocComments:
    """Tests for /// outer doc comments."""

    def test_function_doc(self):
        """/// before fn attaches to function."""
        program = parse('/// Adds two values\nfn add(a @ A: u8) -> u8 { return A; }', '<test>')
        assert program.items[0].doc == 'Adds two values'

    def test_multi_line_doc(self):
        """Multiple /// lines are concatenated."""
        source = '/// Line 1\n/// Line 2\n/// Line 3\nfn test() { }'
        program = parse(source, '<test>')
        assert 'Line 1' in program.items[0].doc
        assert 'Line 2' in program.items[0].doc
        assert 'Line 3' in program.items[0].doc

    def test_struct_doc(self):
        """/// before struct attaches to struct."""
        program = parse('/// A player entity\nstruct Player { x: u8 }', '<test>')
        assert program.items[0].doc == 'A player entity'

    def test_enum_doc(self):
        """/// before enum attaches to enum."""
        program = parse('/// Cardinal directions\nenum Dir { N = 0, E, S, W }', '<test>')
        assert program.items[0].doc == 'Cardinal directions'

    def test_const_doc(self):
        """/// before const attaches to const."""
        program = parse('/// Maximum health\nconst MAX_HP: u8 = 100;', '<test>')
        assert program.items[0].doc == 'Maximum health'

    def test_static_doc(self):
        """/// before static attaches to static."""
        program = parse('/// Player buffer\n#[ram]\nstatic mut BUF: [u8; 4];', '<test>')
        assert program.items[0].doc == 'Player buffer'

    def test_trait_doc(self):
        """/// before trait attaches to trait."""
        source = '/// Renderable objects\ntrait Drawable { fn draw(*self); }'
        program = parse(source, '<test>')
        assert program.items[0].doc == 'Renderable objects'

    def test_impl_doc(self):
        """/// before impl attaches to impl."""
        source = '/// Player methods\nimpl Player { fn tick(*self) { } }'
        program = parse(source, '<test>')
        assert program.items[0].doc == 'Player methods'

    def test_no_doc(self):
        """Declarations without doc comments have doc=None."""
        program = parse('fn test() { }', '<test>')
        assert program.items[0].doc is None

    def test_regular_comment_not_doc(self):
        """// comments are NOT doc comments."""
        program = parse('// Regular comment\nfn test() { }', '<test>')
        assert program.items[0].doc is None


class TestInnerDocComments:
    """Tests for //! inner doc comments."""

    def test_inner_doc_on_program(self):
        """//! at file start attaches to program."""
        program = parse('//! This is a module\nfn test() { }', '<test>')
        assert program.doc == 'This is a module'

    def test_multi_line_inner_doc(self):
        """Multiple //! lines are concatenated."""
        source = '//! Line 1\n//! Line 2\nfn test() { }'
        program = parse(source, '<test>')
        assert 'Line 1' in program.doc
        assert 'Line 2' in program.doc

    def test_no_inner_doc(self):
        """Programs without inner doc have doc=None."""
        program = parse('fn test() { }', '<test>')
        assert program.doc is None


class TestBlockDocComments:
    """Tests for /** */ and /*! */ block doc comments."""

    def test_block_outer_doc(self):
        """/** */ before fn attaches to function."""
        program = parse('/** Block doc */\nfn test() { }', '<test>')
        assert 'Block doc' in program.items[0].doc

    def test_block_inner_doc(self):
        """/*! */ at file start attaches to program."""
        program = parse('/*! Module block doc */\nfn test() { }', '<test>')
        assert 'Module block doc' in program.doc

    def test_multiline_block_doc(self):
        """Multi-line /** */ preserves content."""
        source = '/**\n * Line 1\n * Line 2\n */\nfn test() { }'
        program = parse(source, '<test>')
        doc = program.items[0].doc
        assert 'Line 1' in doc
        assert 'Line 2' in doc


class TestEdgeCases:
    """Edge cases for comment parsing."""

    def test_four_slashes_is_regular(self):
        """//// is a regular comment, not a doc comment."""
        program = parse('//// Not a doc comment\nfn test() { }', '<test>')
        assert program.items[0].doc is None

    def test_empty_block_comment(self):
        """/**/ is a regular empty block comment."""
        program = parse('/**/\nfn test() { }', '<test>')
        assert program.items[0].doc is None

    def test_doc_with_attributes(self):
        """Doc comments work alongside attributes."""
        source = '/// Documented entry\n#[entry]\nfn main() { }'
        program = parse(source, '<test>')
        assert program.items[0].doc == 'Documented entry'
        assert len(program.items[0].attributes) == 1

    def test_empty_doc_comment(self):
        """/// with no text produces empty doc."""
        program = parse('///\nfn test() { }', '<test>')
        assert program.items[0].doc == ''

    def test_mixed_regular_and_doc_comments(self):
        """Regular comments before doc comments are ignored."""
        source = '// Regular\n/// Doc\nfn test() { }'
        program = parse(source, '<test>')
        assert program.items[0].doc == 'Doc'
