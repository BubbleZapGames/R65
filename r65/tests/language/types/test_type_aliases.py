"""Tests for type aliases."""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.hir import nodes as hir
from r65.compiler.hir import types as hir_types
from r65.tests.language.common import (
    parse_type_alias,
    parse_program,
    build_hir,
)


# =============================================================================
# AST-level parsing tests
# =============================================================================

class TestTypeAliasParsing:
    """Tests for type alias declaration parsing."""

    def test_basic_type_alias(self):
        """Test basic type alias: type Word = u16;"""
        alias = parse_type_alias("type Word = u16;")
        assert alias.name == "Word"
        assert isinstance(alias.aliased_type, ast.BasicType)
        assert alias.aliased_type.name == "u16"

    def test_u8_alias(self):
        """Test alias to u8."""
        alias = parse_type_alias("type Byte = u8;")
        assert alias.name == "Byte"
        assert isinstance(alias.aliased_type, ast.BasicType)
        assert alias.aliased_type.name == "u8"

    def test_bool_alias(self):
        """Test alias to bool."""
        alias = parse_type_alias("type Flag = bool;")
        assert alias.name == "Flag"
        assert isinstance(alias.aliased_type, ast.BasicType)
        assert alias.aliased_type.name == "bool"

    def test_pointer_type_alias(self):
        """Test pointer type alias: type Ptr = *u8;"""
        alias = parse_type_alias("type Ptr = *u8;")
        assert alias.name == "Ptr"
        assert isinstance(alias.aliased_type, ast.PointerType)
        assert not alias.aliased_type.is_far
        assert isinstance(alias.aliased_type.pointee_type, ast.BasicType)
        assert alias.aliased_type.pointee_type.name == "u8"

    def test_far_pointer_alias(self):
        """Test far pointer alias: type FarPtr = far *u8;"""
        alias = parse_type_alias("type FarPtr = far *u8;")
        assert alias.name == "FarPtr"
        assert isinstance(alias.aliased_type, ast.PointerType)
        assert alias.aliased_type.is_far
        assert alias.aliased_type.pointee_type.name == "u8"

    def test_array_type_alias(self):
        """Test array type alias: type Buffer = [u8; 256];"""
        alias = parse_type_alias("type Buffer = [u8; 256];")
        assert alias.name == "Buffer"
        assert isinstance(alias.aliased_type, ast.ArrayType)
        assert isinstance(alias.aliased_type.element_type, ast.BasicType)
        assert alias.aliased_type.element_type.name == "u8"
        assert alias.aliased_type.size.value == 256

    def test_function_type_alias(self):
        """Test function type alias: type Callback = fn(u8) -> u8;"""
        alias = parse_type_alias("type Callback = fn(u8) -> u8;")
        assert alias.name == "Callback"
        assert isinstance(alias.aliased_type, ast.FunctionType)
        assert not alias.aliased_type.is_far
        assert len(alias.aliased_type.param_types) == 1
        assert alias.aliased_type.param_types[0].name == "u8"
        assert alias.aliased_type.return_type.name == "u8"

    def test_far_function_type_alias(self):
        """Test far function type alias: type FarCallback = far fn() -> u8;"""
        alias = parse_type_alias("type FarCallback = far fn() -> u8;")
        assert alias.name == "FarCallback"
        assert isinstance(alias.aliased_type, ast.FunctionType)
        assert alias.aliased_type.is_far
        assert len(alias.aliased_type.param_types) == 0
        assert alias.aliased_type.return_type.name == "u8"

    def test_struct_name_alias(self):
        """Test alias to user-defined type name: type Player = PlayerData;"""
        alias = parse_type_alias("type Player = PlayerData;")
        assert alias.name == "Player"
        assert isinstance(alias.aliased_type, ast.BasicType)
        assert alias.aliased_type.name == "PlayerData"

    def test_type_alias_in_program(self):
        """Test that type alias appears as a program item."""
        prog = parse_program("type Word = u16;")
        assert len(prog.items) == 1
        assert isinstance(prog.items[0], ast.TypeAlias)


# =============================================================================
# HIR-level resolution tests
# =============================================================================

def _get_type_aliases(hir_prog):
    """Helper to get HIRTypeAlias declarations from an HIR program."""
    return hir_prog.get_declarations_by_type(hir.HIRTypeAlias)


class TestTypeAliasHIRResolution:
    """Tests for type alias resolution in HIR."""

    def test_alias_resolves_to_basic_type(self):
        """Type alias to u16 resolves to BasicTypeInfo."""
        hir_prog = build_hir("type Word = u16;")
        aliases = _get_type_aliases(hir_prog)
        assert len(aliases) == 1
        assert isinstance(aliases[0].aliased_type, hir_types.BasicTypeInfo)
        assert aliases[0].aliased_type.name == "u16"

    def test_alias_resolves_to_pointer_type(self):
        """Type alias to *u8 resolves to PointerTypeInfo."""
        hir_prog = build_hir("type Ptr = *u8;")
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.PointerTypeInfo)
        assert not alias.aliased_type.is_far
        assert isinstance(alias.aliased_type.pointee_type, hir_types.BasicTypeInfo)
        assert alias.aliased_type.pointee_type.name == "u8"

    def test_alias_resolves_to_far_pointer(self):
        """Type alias to far *u8 resolves to PointerTypeInfo with is_far."""
        hir_prog = build_hir("type FarPtr = far *u8;")
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.PointerTypeInfo)
        assert alias.aliased_type.is_far

    def test_alias_resolves_to_array_type(self):
        """Type alias to [u8; 256] resolves to ArrayTypeInfo."""
        hir_prog = build_hir("type Buffer = [u8; 256];")
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.ArrayTypeInfo)
        assert alias.aliased_type.size == 256
        assert isinstance(alias.aliased_type.element_type, hir_types.BasicTypeInfo)
        assert alias.aliased_type.element_type.name == "u8"

    def test_alias_resolves_to_function_type(self):
        """Type alias to fn(u8) -> u8 resolves to FunctionTypeInfo."""
        hir_prog = build_hir("type Callback = fn(u8) -> u8;")
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.FunctionTypeInfo)
        assert not alias.aliased_type.is_far
        assert len(alias.aliased_type.param_types) == 1
        assert alias.aliased_type.return_type.name == "u8"

    def test_chained_alias(self):
        """Chained alias: type Word16 = u16; type Size = Word16; resolves to u16."""
        hir_prog = build_hir("type Word16 = u16; type Size = Word16;")
        aliases = _get_type_aliases(hir_prog)
        assert len(aliases) == 2
        # Second alias should resolve through to the underlying u16
        assert isinstance(aliases[1].aliased_type, hir_types.BasicTypeInfo)
        assert aliases[1].aliased_type.name == "u16"

    def test_alias_to_struct(self):
        """Type alias to struct resolves to StructTypeInfo."""
        source = """
            struct PlayerData { x: u8, y: u8 }
            type Player = PlayerData;
        """
        hir_prog = build_hir(source)
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.StructTypeInfo)
        assert alias.aliased_type.name == "PlayerData"

    def test_alias_to_enum(self):
        """Type alias to enum resolves to EnumTypeInfo."""
        source = """
            enum Direction { North, East, South, West }
            type Dir = Direction;
        """
        hir_prog = build_hir(source)
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.EnumTypeInfo)
        assert alias.aliased_type.name == "Direction"

    def test_alias_preserves_name(self):
        """HIR type alias preserves its declared name."""
        hir_prog = build_hir("type Word = u16;")
        alias = _get_type_aliases(hir_prog)[0]
        assert alias.name == "Word"

    def test_alias_has_symbol(self):
        """HIR type alias has a symbol reference."""
        hir_prog = build_hir("type Word = u16;")
        alias = _get_type_aliases(hir_prog)[0]
        assert alias.symbol is not None
        assert alias.symbol.name == "Word"


# =============================================================================
# Usage in declarations (HIR + type checker)
# =============================================================================

class TestTypeAliasUsage:
    """Tests for using type aliases in variable and function declarations."""

    def test_alias_as_static_var_type(self):
        """Use alias as static variable type."""
        from r65.compiler.frontend import Parser
        from r65.compiler.hir import HIRBuilder
        from r65.compiler.typeck import TypeChecker

        source = """
            type Word = u16;
            #[ram] static mut VAL: Word;
            fn test() { VAL = 42; }
        """
        parser = Parser()
        ast_prog = parser.parse(source)
        hir_builder = HIRBuilder()
        hir_prog = hir_builder.build_program(ast_prog)
        type_checker = TypeChecker(hir_prog)
        type_checker.check()

        static = hir_prog.statics[0]
        assert isinstance(static.var_type, hir_types.BasicTypeInfo)
        assert static.var_type.name == "u16"

    def test_alias_as_function_parameter_type(self):
        """Use alias as function parameter type."""
        source = """
            type Byte = u8;
            fn process(val: Byte) { }
        """
        hir_prog = build_hir(source)
        func = hir_prog.functions[0]
        param = func.parameters[0]
        assert isinstance(param.param_type, hir_types.BasicTypeInfo)
        assert param.param_type.name == "u8"

    def test_alias_as_return_type(self):
        """Use alias as function return type."""
        source = """
            type Byte = u8;
            fn get_value() -> Byte { return 0; }
        """
        hir_prog = build_hir(source)
        func = hir_prog.functions[0]
        assert isinstance(func.return_type, hir_types.BasicTypeInfo)
        assert func.return_type.name == "u8"

    def test_alias_as_array_element_type(self):
        """Use alias as array element type."""
        source = """
            type Byte = u8;
            #[ram] static mut BUF: [Byte; 16];
        """
        hir_prog = build_hir(source)
        static = hir_prog.statics[0]
        assert isinstance(static.var_type, hir_types.ArrayTypeInfo)
        assert static.var_type.size == 16
        assert isinstance(static.var_type.element_type, hir_types.BasicTypeInfo)
        assert static.var_type.element_type.name == "u8"

    def test_alias_in_let_binding(self):
        """Use alias in let binding type annotation."""
        source = """
            type Word = u16;
            fn test() { let x: Word = 0; }
        """
        hir_prog = build_hir(source)
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.var_type, hir_types.BasicTypeInfo)
        assert let_stmt.var_type.name == "u16"

    def test_alias_struct_as_static_type(self):
        """Use struct alias as static variable type."""
        source = """
            struct PlayerData { x: u8, y: u8 }
            type Player = PlayerData;
            #[ram] static mut P: Player;
        """
        hir_prog = build_hir(source)
        static = hir_prog.statics[0]
        assert isinstance(static.var_type, hir_types.StructTypeInfo)
        assert static.var_type.name == "PlayerData"

    def test_chained_alias_in_declaration(self):
        """Chained alias used in a declaration resolves to the underlying type."""
        source = """
            type Word16 = u16;
            type Size = Word16;
            #[ram] static mut VAL: Size;
        """
        hir_prog = build_hir(source)
        static = hir_prog.statics[0]
        assert isinstance(static.var_type, hir_types.BasicTypeInfo)
        assert static.var_type.name == "u16"

    def test_pointer_to_struct_alias_as_param(self):
        """Pointer-to-struct alias used as function parameter resolves correctly."""
        source = """
            struct Sprite { x: u8, y: u8 }
            type SpritePtr = *Sprite;
            fn move_sprite(ptr: SpritePtr) { }
        """
        hir_prog = build_hir(source)
        func = hir_prog.functions[0]
        param = func.parameters[0]
        assert isinstance(param.param_type, hir_types.PointerTypeInfo)
        assert not param.param_type.is_far
        assert isinstance(param.param_type.pointee_type, hir_types.StructTypeInfo)
        assert param.param_type.pointee_type.name == "Sprite"

    def test_fn_pointer_alias_as_param(self):
        """Function pointer alias used as function parameter resolves correctly."""
        source = """
            type Callback = fn(u8) -> u8;
            fn apply(cb: Callback) { }
        """
        hir_prog = build_hir(source)
        func = hir_prog.functions[0]
        param = func.parameters[0]
        assert isinstance(param.param_type, hir_types.FunctionTypeInfo)
        assert not param.param_type.is_far
        assert len(param.param_type.param_types) == 1
        assert param.param_type.param_types[0].name == "u8"
        assert param.param_type.return_type.name == "u8"

    def test_far_fn_pointer_alias_as_param(self):
        """Far function pointer alias used as function parameter resolves correctly."""
        source = """
            type FarCallback = far fn() -> u8;
            fn apply(cb: FarCallback) { }
        """
        hir_prog = build_hir(source)
        func = hir_prog.functions[0]
        param = func.parameters[0]
        assert isinstance(param.param_type, hir_types.FunctionTypeInfo)
        assert param.param_type.is_far
        assert len(param.param_type.param_types) == 0
        assert param.param_type.return_type.name == "u8"


# =============================================================================
# HIR resolution tests for pointer/fn-pointer aliases
# =============================================================================

class TestTypeAliasPointerResolution:
    """Tests for pointer and function pointer type alias resolution in HIR."""

    def test_pointer_to_struct_resolves(self):
        """type SpritePtr = *Sprite; resolves to PointerTypeInfo(StructTypeInfo)."""
        source = """
            struct Sprite { x: u8, y: u8 }
            type SpritePtr = *Sprite;
        """
        hir_prog = build_hir(source)
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.PointerTypeInfo)
        assert not alias.aliased_type.is_far
        assert isinstance(alias.aliased_type.pointee_type, hir_types.StructTypeInfo)
        assert alias.aliased_type.pointee_type.name == "Sprite"

    def test_far_pointer_to_struct_resolves(self):
        """type FarSpritePtr = far *Sprite; resolves with is_far=True."""
        source = """
            struct Sprite { x: u8, y: u8 }
            type FarSpritePtr = far *Sprite;
        """
        hir_prog = build_hir(source)
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.PointerTypeInfo)
        assert alias.aliased_type.is_far
        assert isinstance(alias.aliased_type.pointee_type, hir_types.StructTypeInfo)

    def test_fn_pointer_with_multiple_params_resolves(self):
        """type BinOp = fn(u8, u8) -> u8; resolves with two param types."""
        hir_prog = build_hir("type BinOp = fn(u8, u8) -> u8;")
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.FunctionTypeInfo)
        assert len(alias.aliased_type.param_types) == 2
        assert alias.aliased_type.param_types[0].name == "u8"
        assert alias.aliased_type.param_types[1].name == "u8"
        assert alias.aliased_type.return_type.name == "u8"

    def test_fn_pointer_no_return_resolves(self):
        """type Action = fn(u8); resolves with no return type."""
        hir_prog = build_hir("type Action = fn(u8);")
        alias = _get_type_aliases(hir_prog)[0]
        assert isinstance(alias.aliased_type, hir_types.FunctionTypeInfo)
        assert len(alias.aliased_type.param_types) == 1
        assert alias.aliased_type.return_type is None


# =============================================================================
# Error cases
# =============================================================================

class TestTypeAliasErrors:
    """Tests for type alias error handling."""

    def test_undefined_type_in_alias(self):
        """Alias to undefined type should raise an error."""
        with pytest.raises(Exception):
            build_hir("type Bad = Nonexistent;")

    def test_undefined_alias_in_declaration(self):
        """Using an undefined alias in a declaration should raise an error."""
        with pytest.raises(Exception):
            build_hir("#[ram] static mut VAL: Undefined;")
