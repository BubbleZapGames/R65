"""Tests for basic types: u8, i8, u16, i16, bool."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_type, parse_static, parse_function


class TestBasicTypes:
    """Tests for basic type parsing."""

    def test_all_basic_types(self):
        """Test u8, i8, u16, i16, bool type parsing."""
        types = ["u8", "i8", "u16", "i16", "bool"]
        for type_name in types:
            t = parse_type(type_name)
            assert isinstance(t, ast.BasicType)
            assert t.name == type_name

    def test_basic_type_in_static(self):
        """Test basic types in static declarations."""
        for type_name in ["u8", "i8", "u16", "i16", "bool"]:
            static = parse_static(f"#[ram] static mut X: {type_name};")
            assert isinstance(static.var_type, ast.BasicType)
            assert static.var_type.name == type_name

    def test_basic_type_in_function_params(self):
        """Test basic types in function parameters."""
        func = parse_function("fn test(a: u8, b: u16, c: bool) { }")
        assert len(func.params) == 3
        assert func.params[0].param_type.name == "u8"
        assert func.params[1].param_type.name == "u16"
        assert func.params[2].param_type.name == "bool"

    def test_basic_type_return(self):
        """Test basic types in return types."""
        func = parse_function("fn test() -> u16 { return A; }")
        assert isinstance(func.return_type, ast.BasicType)
        assert func.return_type.name == "u16"


class TestTypeLiterals:
    """Tests for typed literals."""

    def test_integer_literals(self):
        """Test integer literals in let statements."""
        func = parse_function("fn test() { let x: u8 = 42; let y: u16 = 0x1000; }")
        let1 = func.body.statements[0]
        let2 = func.body.statements[1]
        assert isinstance(let1, ast.LetStmt)
        assert let1.var_type.name == "u8"
        assert isinstance(let2, ast.LetStmt)
        assert let2.var_type.name == "u16"

    def test_bool_literals(self):
        """Test true and false literals."""
        func = parse_function("fn test() { let t: bool = true; let f: bool = false; }")
        let_t = func.body.statements[0]
        let_f = func.body.statements[1]
        assert isinstance(let_t.initializer, ast.BooleanLiteral)
        assert let_t.initializer.value is True
        assert isinstance(let_f.initializer, ast.BooleanLiteral)
        assert let_f.initializer.value is False


