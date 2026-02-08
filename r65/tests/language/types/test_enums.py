"""Tests for enum types."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_enum, parse_function


class TestEnumDeclaration:
    """Tests for enum declaration parsing."""

    def test_basic_enum(self):
        """Test basic enum with variants."""
        enum = parse_enum("enum Direction { North, East, South, West }")
        assert enum.name == "Direction"
        assert len(enum.variants) == 4
        assert enum.variants[0].name == "North"
        assert enum.variants[3].name == "West"

    def test_enum_with_explicit_values(self):
        """Test enum with explicit discriminant values."""
        enum = parse_enum("enum Status { Idle = 0, Running = 1, Stopped = 2 }")
        assert len(enum.variants) == 3
        assert enum.variants[0].value.value == 0
        assert enum.variants[1].value.value == 1
        assert enum.variants[2].value.value == 2

    def test_enum_mixed_values(self):
        """Test enum with mixed explicit and auto values."""
        enum = parse_enum("enum Mix { A = 10, B, C, D = 20 }")
        assert len(enum.variants) == 4
        assert enum.variants[0].value.value == 10
        # B, C are auto-increment (handled by type checker)
        assert enum.variants[3].value.value == 20


class TestEnumUsage:
    """Tests for enum usage in code."""

    def test_enum_variant_access(self):
        """Test enum variant access with :: syntax."""
        func = parse_function("fn test() { let d: u8 = Direction::North; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.EnumVariantExpr)
        assert let_stmt.initializer.enum_name == "Direction"
        assert let_stmt.initializer.variant_name == "North"

    def test_enum_in_comparison(self):
        """Test enum variant in comparison."""
        func = parse_function("fn test() { if dir == Direction::North { A = 1; } }")
        if_stmt = func.body.statements[0]
        assert isinstance(if_stmt.condition, ast.BinaryOp)


