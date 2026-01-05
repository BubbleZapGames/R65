"""Tests for array types and literals."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_type, parse_static, parse_function, build_hir


class TestArrayTypes:
    """Tests for array type parsing."""

    def test_basic_array_types(self):
        """Test array type with various element types."""
        for elem_type in ["u8", "u16", "bool"]:
            t = parse_type(f"[{elem_type}; 10]")
            assert isinstance(t, ast.ArrayType)
            assert isinstance(t.element_type, ast.BasicType)
            assert t.element_type.name == elem_type

    def test_array_sizes(self):
        """Test array types with various sizes."""
        sizes = [1, 10, 100, 256, 1024]
        for size in sizes:
            t = parse_type(f"[u8; {size}]")
            assert isinstance(t, ast.ArrayType)
            assert isinstance(t.size, ast.IntegerLiteral)
            assert t.size.value == size

    def test_array_hex_size(self):
        """Test array with hex size."""
        t = parse_type("[u8; 0x100]")
        assert isinstance(t, ast.ArrayType)
        assert t.size.value == 256

    def test_nested_array_types(self):
        """Test nested array types."""
        t = parse_type("[[u8; 8]; 4]")
        assert isinstance(t, ast.ArrayType)
        assert isinstance(t.element_type, ast.ArrayType)


class TestArrayLiterals:
    """Tests for array literals."""

    def test_array_literal_initialization(self):
        """Test array literal in static declaration."""
        static = parse_static("#[ram] static mut ARR: [u8; 3] = [1, 2, 3];")
        assert isinstance(static.initializer, ast.ArrayLiteralExpr)
        assert len(static.initializer.elements) == 3

    def test_array_repeat_literal(self):
        """Test array repeat literal [value; count]."""
        static = parse_static("#[ram] static mut ARR: [u8; 10] = [0; 10];")
        # Repeat literals may be parsed as ArrayLiteralExpr with special handling
        assert static.initializer is not None

    def test_string_literal_for_array(self):
        """Test string literal initializing byte array."""
        static = parse_static('#[ram] static mut MSG: [u8; 16] = "Hello";')
        assert isinstance(static.initializer, ast.StringLiteral)


class TestArrayAccess:
    """Tests for array indexing."""

    def test_array_index_access(self):
        """Test array index access."""
        func = parse_function("fn test() { let x: u8 = ARR[0]; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.ArrayIndex)
        assert isinstance(let_stmt.initializer.index, ast.IntegerLiteral)

    def test_array_index_with_variable(self):
        """Test array index with variable."""
        func = parse_function("fn test() { let x: u8 = ARR[Y]; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.ArrayIndex)


class TestArrayHIR:
    """Tests for array HIR generation."""

    def test_array_hir(self):
        """Test array declarations generate proper HIR."""
        hir_prog = build_hir("""
            #[ram] static mut BUFFER: [u8; 256] = [0; 256];
            fn test() { let x: u8 = BUFFER[0]; }
        """)
        assert len(hir_prog.statics) >= 1
        assert len(hir_prog.functions) >= 1
