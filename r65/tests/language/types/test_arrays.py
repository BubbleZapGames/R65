# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for array types and literals."""

import pytest

from r65.compiler.frontend import ast
from r65.compiler.typeck.type_checker import TypeChecker
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


class TestArrayLiteralElementTypes:
    """Every element of an array literal is checked against the element type.

    The first element used to only *infer* the element type. When the type came
    from context that inference result was thrown away and the element was never
    compared against it, so element 1 accepted anything — the one position a
    newtype could launder itself through into an array of its payload.

    Initializing from an array literal is assignment, so it now uses the same
    rule as `let`. That closes the hole and, in the other direction, lets a
    payload flow implicitly into an array of a newtype the way it already does
    in a `let`.
    """

    NEWTYPES = "struct Q(i16);\nstruct Rot(i16);\n"

    def check(self, body: str, decl: str = ""):
        source = self.NEWTYPES + decl + "fn main() { " + body + " }"
        TypeChecker(build_hir(source)).check()

    @pytest.mark.parametrize("body,bad", [
        ("let q: Q = 5; let a: [i16; 2] = [q, 2];", "Q"),
        ("let q: Q = 5; let a: [i16; 2] = [q, q];", "Q"),
        ("let r: Rot = 5; let a: [Q; 2] = [r, r];", "Rot"),
    ])
    def test_newtype_rejected_in_first_position(self, body, bad):
        with pytest.raises(Exception) as exc:
            self.check(body)
        assert "Array element 1" in str(exc.value)
        assert bad in str(exc.value)

    def test_newtype_still_rejected_in_later_positions(self):
        with pytest.raises(Exception) as exc:
            self.check("let q: Q = 5; let a: [i16; 2] = [1, q];")
        assert "Array element 2" in str(exc.value)

    @pytest.mark.parametrize("body", [
        "let a: [Q; 2] = [1, 2];",
        "let q: Q = 5; let a: [Q; 2] = [q, 1];",
        "let q: Q = 5; let a: [Q; 2] = [1, q];",
        "let q: Q = 5; let a: [Q; 2] = [q, q];",
    ])
    def test_payload_flows_into_a_newtype_array(self, body):
        """Transparent in — and `[1, 2]` used to fail on the *second* element."""
        self.check(body)

    @pytest.mark.parametrize("body", [
        "let x: i8 = 1; let a: [u8; 2] = [x, 2];",
        "let x: u8 = 1; let a: [i8; 2] = [x, 2];",
        "let x: i16 = 1; let a: [u16; 2] = [x, 2];",
        "let x: i8 = 1; let y: u8 = 2; let a = [x, y];",
        "let a: [u8; 3] = [1, 2, 3];",
    ])
    def test_same_size_signed_unsigned_still_mix(self, body):
        """The leniency the old element predicate existed to provide."""
        self.check(body)

    @pytest.mark.parametrize("body", [
        "let a: [u8; 2] = [300, 2];",
        "let a: [u8; 2] = [1, 300];",
    ])
    def test_literal_range_still_checked_in_both_positions(self, body):
        with pytest.raises(Exception) as exc:
            self.check(body)
        assert "does not fit" in str(exc.value)
