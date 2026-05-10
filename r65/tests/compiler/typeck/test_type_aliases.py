# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for type aliases (`type Foo = Bar;`).

Aliases are pure typeck — they collapse to their underlying type at HIR build
time and are resolvable in any type position (params, returns, statics, casts).
Runtime behavior follows from the resolved type, so we only test resolution.
"""
from r65.compiler.frontend import Parser
from r65.compiler.hir import HIRBuilder
from r65.compiler.hir.nodes import HIRTypeAlias, HIRFunctionDecl, HIRStaticDecl
from r65.compiler.hir.types import (
    BasicTypeInfo, PointerTypeInfo, FunctionTypeInfo, StructTypeInfo,
    EnumTypeInfo,
)
from r65.compiler.typeck import TypeChecker


def build(source: str):
    parser = Parser()
    ast = parser.parse(source)
    hir = HIRBuilder().build_program(ast)
    TypeChecker(hir).check()
    return hir


def find_decl(hir, kind, name):
    for d in hir.declarations:
        if isinstance(d, kind) and getattr(d, 'name', None) == name:
            return d
    raise AssertionError(f"{kind.__name__} {name!r} not found")


def find_alias(hir, name):
    return find_decl(hir, HIRTypeAlias, name)


class TestPointerTypeAliases:
    """Pointer-typed aliases resolve to PointerTypeInfo."""

    def test_pointer_to_struct_alias(self):
        """type SpritePtr = *Sprite; — alias resolves; param uses *Sprite."""
        hir = build('''
            struct Sprite { x: u8, y: u8 }
            type SpritePtr = *Sprite;
            fn set_pos(ptr: SpritePtr, xval @ A: u8) {
                ptr.x = xval;
            }
        ''')
        alias = find_alias(hir, 'SpritePtr')
        assert isinstance(alias.aliased_type, PointerTypeInfo)
        assert isinstance(alias.aliased_type.pointee_type, StructTypeInfo)
        assert alias.aliased_type.pointee_type.name == 'Sprite'

        fn = find_decl(hir, HIRFunctionDecl, 'set_pos')
        ptr_param = fn.parameters[0]
        assert isinstance(ptr_param.param_type, PointerTypeInfo)
        assert ptr_param.param_type.pointee_type.name == 'Sprite'

    def test_pointer_to_u8_alias(self):
        """type BytePtr = *u8; — alias resolves to *u8."""
        hir = build('''
            type BytePtr = *u8;
            #[zeropage(0x10)]
            static mut VAL: u8;
            fn write_byte(ptr: BytePtr, val @ A: u8) { *ptr = val; }
        ''')
        alias = find_alias(hir, 'BytePtr')
        assert isinstance(alias.aliased_type, PointerTypeInfo)
        assert alias.aliased_type.pointee_type.name == 'u8'

        fn = find_decl(hir, HIRFunctionDecl, 'write_byte')
        assert isinstance(fn.parameters[0].param_type, PointerTypeInfo)


class TestFunctionPointerTypeAliases:
    """Function-pointer aliases resolve to FunctionTypeInfo."""

    def test_fn_pointer_alias(self):
        """type Callback = fn() -> u8; — aliased type is a FunctionTypeInfo."""
        hir = build('''
            type Callback = fn() -> u8;
            #[zeropage(0x10)]
            static mut CB: Callback;
            fn get_answer() -> u8 { return 42; }
        ''')
        alias = find_alias(hir, 'Callback')
        assert isinstance(alias.aliased_type, FunctionTypeInfo)
        assert alias.aliased_type.return_type.name == 'u8'
        assert alias.aliased_type.param_types == []

        cb = find_decl(hir, HIRStaticDecl, 'CB')
        assert isinstance(cb.var_type, FunctionTypeInfo)

    def test_fn_pointer_alias_param(self):
        """Function param typed by alias resolves to FunctionTypeInfo."""
        hir = build('''
            type Transform = fn() -> u8;
            fn apply(cb: Transform) -> u8 { return cb(); }
            fn get_value() -> u8 { return 99; }
        ''')
        fn = find_decl(hir, HIRFunctionDecl, 'apply')
        cb_param = fn.parameters[0]
        assert isinstance(cb_param.param_type, FunctionTypeInfo)
        assert cb_param.param_type.return_type.name == 'u8'


class TestTypeAliasInDeclarations:
    """Aliases work in static-var, parameter, return, and chained positions."""

    def test_alias_as_static_var_type(self):
        """type Word = u16; static VAL: Word; — VAL.var_type is u16."""
        hir = build('''
            type Word = u16;
            #[zeropage(0x10)]
            static mut VAL: Word;
        ''')
        val = find_decl(hir, HIRStaticDecl, 'VAL')
        assert isinstance(val.var_type, BasicTypeInfo)
        assert val.var_type.name == 'u16'

    def test_alias_as_parameter_type(self):
        """type Byte = u8; fn(a: Byte) — param type is u8."""
        hir = build('''
            type Byte = u8;
            fn add_bytes(a @ A: Byte, b: Byte) -> Byte { return a + b; }
        ''')
        fn = find_decl(hir, HIRFunctionDecl, 'add_bytes')
        for p in fn.parameters:
            assert isinstance(p.param_type, BasicTypeInfo)
            assert p.param_type.name == 'u8'

    def test_alias_as_return_type(self):
        """fn() -> Byte (alias) — return_type is u8."""
        hir = build('''
            type Byte = u8;
            fn get_value() -> Byte { return 99; }
        ''')
        fn = find_decl(hir, HIRFunctionDecl, 'get_value')
        assert isinstance(fn.return_type, BasicTypeInfo)
        assert fn.return_type.name == 'u8'

    def test_chained_alias_collapses(self):
        """type Byte = u8; type Octet = Byte; — Octet.aliased_type is u8."""
        hir = build('''
            type Byte = u8;
            type Octet = Byte;
            fn get_val() -> Octet { return 77; }
        ''')
        byte = find_alias(hir, 'Byte')
        octet = find_alias(hir, 'Octet')
        assert byte.aliased_type.name == 'u8'
        assert octet.aliased_type.name == 'u8'  # collapsed through Byte
        fn = find_decl(hir, HIRFunctionDecl, 'get_val')
        assert fn.return_type.name == 'u8'

    def test_alias_to_struct(self):
        """type Player = PlayerData; — alias to a struct resolves."""
        hir = build('''
            struct PlayerData { x: u8, y: u8 }
            type Player = PlayerData;
            #[zeropage(0x10)]
            static mut P: Player;
        ''')
        alias = find_alias(hir, 'Player')
        assert isinstance(alias.aliased_type, StructTypeInfo)
        assert alias.aliased_type.name == 'PlayerData'

        p = find_decl(hir, HIRStaticDecl, 'P')
        assert isinstance(p.var_type, StructTypeInfo)
        assert p.var_type.name == 'PlayerData'

    def test_alias_to_enum(self):
        """type Dir = Direction; — alias to an enum resolves."""
        hir = build('''
            enum Direction { North = 0, East = 1, South = 2, West = 3 }
            type Dir = Direction;
            #[zeropage(0x10)]
            static mut FACING: Dir;
        ''')
        alias = find_alias(hir, 'Dir')
        assert isinstance(alias.aliased_type, EnumTypeInfo)
        assert alias.aliased_type.name == 'Direction'

        facing = find_decl(hir, HIRStaticDecl, 'FACING')
        assert isinstance(facing.var_type, EnumTypeInfo)
