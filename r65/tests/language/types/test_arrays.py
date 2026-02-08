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



class TestArrayLen:
    """Tests for array len() method."""

    def test_len_method_parsing(self):
        """Test array.len() method parses correctly."""
        func = parse_function("""
            fn test() {
                let size: u16 = BUFFER.len();
            }
        """)
        let_stmt = func.body.statements[0]
        # len() is parsed as a FunctionCall with FieldAccess
        assert isinstance(let_stmt.initializer, ast.FunctionCall)
        assert isinstance(let_stmt.initializer.func, ast.FieldAccess)
        assert let_stmt.initializer.func.field == "len"

    def test_len_hir_const_evaluation(self):
        """Test len() is const-evaluated to HIRIntegerLiteral in HIR."""
        from r65.compiler.hir import nodes as hir
        hir_prog = build_hir("""
            #[ram] static mut BUFFER: [u8; 256] = [0; 256];
            fn test() {
                let size: u16 = BUFFER.len();
            }
        """)
        # Find the let statement in the function
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        # len() should be const-evaluated to an integer literal
        assert isinstance(let_stmt.initializer, hir.HIRIntegerLiteral)
        assert let_stmt.initializer.value == 256

    def test_len_different_sizes(self):
        """Test len() returns correct size for different array sizes."""
        from r65.compiler.hir import nodes as hir
        for size in [1, 10, 100, 256, 1000]:
            hir_prog = build_hir(f"""
                #[ram] static mut ARR: [u8; {size}] = [0; {size}];
                fn test() {{
                    let sz: u16 = ARR.len();
                }}
            """)
            func = hir_prog.functions[0]
            let_stmt = func.body.statements[0]
            assert isinstance(let_stmt.initializer, hir.HIRIntegerLiteral)
            assert let_stmt.initializer.value == size

    def test_len_in_const_expression(self):
        """Test len() can be used in const expressions."""
        from r65.compiler.hir import nodes as hir
        hir_prog = build_hir("""
            #[ram] static mut SOURCE: [u8; 100] = [0; 100];
            const SIZE: u16 = SOURCE.len();
            fn test() {
                A = SIZE as u8;
            }
        """)
        # Find the const and verify its value
        consts = [c for c in hir_prog.constants if c.name == "SIZE"]
        assert len(consts) == 1
        assert consts[0].evaluated_value == 100

    def test_len_no_arguments(self):
        """Test len() with arguments fails."""
        import pytest
        with pytest.raises(Exception) as exc_info:
            build_hir("""
                #[ram] static mut BUFFER: [u8; 10] = [0; 10];
                fn test() {
                    let size: u16 = BUFFER.len(1);
                }
            """)
        assert "no arguments" in str(exc_info.value).lower() or "0 argument" in str(exc_info.value).lower()

    def test_len_on_non_array_fails(self):
        """Test len() on non-array types fails type checking."""
        import pytest
        from r65.compiler.typeck.type_checker import TypeChecker
        from r65.compiler.hir.builder import HIRBuilder
        from r65.compiler.frontend.parser import parse

        # Build HIR first
        program = parse("""
            #[zeropage] static mut VALUE: u8;
            fn test() {
                let size: u16 = VALUE.len();
            }
        """)
        builder = HIRBuilder()
        hir_prog = builder.build_program(program)

        # Type checking should fail
        checker = TypeChecker(hir_prog)
        with pytest.raises(Exception) as exc_info:
            checker.check()
        assert "array" in str(exc_info.value).lower()
