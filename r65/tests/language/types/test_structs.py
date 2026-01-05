"""Tests for struct types."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_struct, parse_static, parse_function, build_hir


class TestStructDeclaration:
    """Tests for struct declaration parsing."""

    def test_basic_struct(self):
        """Test basic struct with multiple fields."""
        struct = parse_struct("struct Point { x: u8, y: u8 }")
        assert struct.name == "Point"
        assert len(struct.fields) == 2
        assert struct.fields[0].name == "x"
        assert struct.fields[1].name == "y"

    def test_struct_with_various_types(self):
        """Test struct with different field types."""
        struct = parse_struct("""
            struct Entity {
                x: u16,
                y: u16,
                health: u8,
                alive: bool
            }
        """)
        assert len(struct.fields) == 4
        assert struct.fields[0].field_type.name == "u16"
        assert struct.fields[3].field_type.name == "bool"

    def test_empty_struct(self):
        """Test empty struct."""
        struct = parse_struct("struct Empty { }")
        assert struct.name == "Empty"
        assert len(struct.fields) == 0


class TestStructInstances:
    """Tests for struct instances and access."""

    def test_struct_static_declaration(self):
        """Test struct as static variable type."""
        prog_source = """
            struct Player { x: u8, y: u8 }
            #[ram] static mut P: Player;
        """
        from r65.tests.language.common import parse_program
        prog = parse_program(prog_source)
        static = prog.items[1]
        # Named types (like struct names) are parsed as BasicType
        assert isinstance(static.var_type, ast.BasicType)
        assert static.var_type.name == "Player"

    def test_struct_field_access(self):
        """Test struct field access with dot operator."""
        func = parse_function("fn test() { let x: u8 = player.x; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.FieldAccess)
        assert let_stmt.initializer.field == "x"

    def test_nested_field_access(self):
        """Test nested struct field access."""
        func = parse_function("fn test() { let v: u8 = outer.inner.value; }")
        let_stmt = func.body.statements[0]
        # outer.inner.value is FieldAccess(FieldAccess(outer, inner), value)
        assert isinstance(let_stmt.initializer, ast.FieldAccess)
        assert let_stmt.initializer.field == "value"


class TestStructLiterals:
    """Tests for struct literal initialization."""

    def test_struct_literal(self):
        """Test struct literal syntax."""
        func = parse_function("fn test() { let p = Point { x: 10, y: 20 }; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.StructLiteralExpr)
        assert let_stmt.initializer.struct_name == "Point"
        assert len(let_stmt.initializer.fields) == 2


class TestStructHIR:
    """Tests for struct HIR generation."""

    def test_struct_hir(self):
        """Test struct generates proper HIR."""
        hir_prog = build_hir("""
            struct Player { x: u8, y: u8, health: u16 }
            #[ram] static mut P: Player;
            fn test() { let h: u16 = P.health; }
        """)
        assert len(hir_prog.structs) >= 1
        assert len(hir_prog.statics) >= 1
