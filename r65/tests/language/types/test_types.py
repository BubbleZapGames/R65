"""
Comprehensive type system tests for R65.

Tests all type constructs:
- Basic types: u8, i8, u16, i16, bool
- Array types: [T; N]
- Pointer types: near<T>, far<T>
- Function types: fn(...) -> T, far fn(...) -> T
- Never type: !
- Struct declarations and types
- Enum declarations and types
- Type aliases

Each test validates:
1. Parsing succeeds and produces correct AST
2. Type annotations are correctly represented
3. HIR is built correctly (where applicable)
"""

import pytest
from r65.compiler.frontend import parse, ParseError, ast
from r65.compiler.hir import HIRBuilder
from r65.compiler.hir import nodes as hir


# ============================================================================
# Test Helpers
# ============================================================================

def parse_program(source: str) -> ast.Program:
    """Parse source and return the program."""
    return parse(source)


def parse_function(source: str) -> ast.FunctionDecl:
    """Parse source and return the first function declaration."""
    program = parse(source)
    assert len(program.items) >= 1
    func = program.items[0]
    assert isinstance(func, ast.FunctionDecl)
    return func


def parse_static(source: str) -> ast.StaticDecl:
    """Parse source and return the first static declaration."""
    program = parse(source)
    assert len(program.items) >= 1
    static = program.items[0]
    assert isinstance(static, ast.StaticDecl)
    return static


def parse_struct(source: str) -> ast.StructDecl:
    """Parse source and return the first struct declaration."""
    program = parse(source)
    assert len(program.items) >= 1
    struct = program.items[0]
    assert isinstance(struct, ast.StructDecl)
    return struct


def parse_enum(source: str) -> ast.EnumDecl:
    """Parse source and return the first enum declaration."""
    program = parse(source)
    assert len(program.items) >= 1
    enum = program.items[0]
    assert isinstance(enum, ast.EnumDecl)
    return enum


def build_hir(source: str) -> hir.HIRProgram:
    """Parse and build HIR from source."""
    program = parse(source)
    builder = HIRBuilder()
    return builder.build_program(program)


def get_hir_function(hir_prog: hir.HIRProgram, name: str) -> hir.HIRFunctionDecl:
    """Get a function by name from HIR program."""
    for func in hir_prog.functions:
        if func.name == name:
            return func
    raise KeyError(f"Function '{name}' not found")


# ============================================================================
# Basic Type Tests
# ============================================================================

class TestBasicTypes:
    """Tests for basic types: u8, i8, u16, i16, bool"""

    def test_u8_type(self):
        """Test u8 type annotation."""
        static = parse_static("static X: u8;")

        assert isinstance(static.var_type, ast.BasicType)
        assert static.var_type.name == 'u8'

    def test_i8_type(self):
        """Test i8 type annotation."""
        static = parse_static("static X: i8;")

        assert isinstance(static.var_type, ast.BasicType)
        assert static.var_type.name == 'i8'

    def test_u16_type(self):
        """Test u16 type annotation."""
        static = parse_static("static X: u16;")

        assert isinstance(static.var_type, ast.BasicType)
        assert static.var_type.name == 'u16'

    def test_i16_type(self):
        """Test i16 type annotation."""
        static = parse_static("static X: i16;")

        assert isinstance(static.var_type, ast.BasicType)
        assert static.var_type.name == 'i16'

    def test_bool_type(self):
        """Test bool type annotation."""
        static = parse_static("static FLAG: bool;")

        assert isinstance(static.var_type, ast.BasicType)
        assert static.var_type.name == 'bool'

    def test_u8_in_let(self):
        """Test u8 in let statement."""
        func = parse_function("fn test() { let x: u8 = 0; }")
        let_stmt = func.body.statements[0]

        assert isinstance(let_stmt, ast.LetStmt)
        assert isinstance(let_stmt.var_type, ast.BasicType)
        assert let_stmt.var_type.name == 'u8'

    def test_u16_in_let(self):
        """Test u16 in let statement."""
        func = parse_function("fn test() { let x: u16 = 0; }")
        let_stmt = func.body.statements[0]

        assert let_stmt.var_type.name == 'u16'

    def test_bool_in_let(self):
        """Test bool in let statement."""
        func = parse_function("fn test() { let flag: bool = true; }")
        let_stmt = func.body.statements[0]

        assert let_stmt.var_type.name == 'bool'

    def test_type_in_parameter(self):
        """Test type in function parameter."""
        func = parse_function("fn test(x: u8, y: u16) { }")

        assert len(func.params) == 2
        assert func.params[0].param_type.name == 'u8'
        assert func.params[1].param_type.name == 'u16'

    def test_type_in_return(self):
        """Test type in function return."""
        func = parse_function("fn test() -> u8 { return 0; }")

        assert isinstance(func.return_type, ast.BasicType)
        assert func.return_type.name == 'u8'

    def test_signed_types_in_params(self):
        """Test signed types in parameters."""
        func = parse_function("fn test(a: i8, b: i16) -> i8 { return a; }")

        assert func.params[0].param_type.name == 'i8'
        assert func.params[1].param_type.name == 'i16'
        assert func.return_type.name == 'i8'


# ============================================================================
# Array Type Tests
# ============================================================================

class TestArrayTypes:
    """Tests for array types: [T; N]"""

    def test_u8_array(self):
        """Test [u8; N] array type."""
        static = parse_static("static BUFFER: [u8; 256];")

        assert isinstance(static.var_type, ast.ArrayType)
        assert isinstance(static.var_type.element_type, ast.BasicType)
        assert static.var_type.element_type.name == 'u8'
        assert isinstance(static.var_type.size, ast.IntegerLiteral)
        assert static.var_type.size.value == 256

    def test_u16_array(self):
        """Test [u16; N] array type."""
        static = parse_static("static DATA: [u16; 128];")

        assert isinstance(static.var_type, ast.ArrayType)
        assert static.var_type.element_type.name == 'u16'
        assert static.var_type.size.value == 128

    def test_bool_array(self):
        """Test [bool; N] array type."""
        static = parse_static("static FLAGS: [bool; 8];")

        assert isinstance(static.var_type, ast.ArrayType)
        assert static.var_type.element_type.name == 'bool'

    def test_small_array(self):
        """Test small array size."""
        static = parse_static("static SMALL: [u8; 4];")

        assert static.var_type.size.value == 4

    def test_large_array(self):
        """Test large array size."""
        static = parse_static("static LARGE: [u8; 65535];")

        assert static.var_type.size.value == 65535

    def test_array_size_one(self):
        """Test array size of 1."""
        static = parse_static("static SINGLE: [u8; 1];")

        assert static.var_type.size.value == 1

    def test_array_with_hex_size(self):
        """Test array with hex size."""
        static = parse_static("static HEX: [u8; 0x100];")

        assert static.var_type.size.value == 0x100

    def test_array_in_let(self):
        """Test array type in let statement."""
        func = parse_function("fn test() { let arr: [u8; 10] = [0; 10]; }")
        let_stmt = func.body.statements[0]

        assert isinstance(let_stmt.var_type, ast.ArrayType)
        assert let_stmt.var_type.size.value == 10

    def test_nested_array(self):
        """Test nested array type [[T; M]; N]."""
        static = parse_static("static GRID: [[u8; 8]; 8];")

        assert isinstance(static.var_type, ast.ArrayType)
        assert isinstance(static.var_type.element_type, ast.ArrayType)
        inner = static.var_type.element_type
        assert inner.element_type.name == 'u8'
        assert inner.size.value == 8
        assert static.var_type.size.value == 8

    def test_array_of_i16(self):
        """Test array of signed type."""
        static = parse_static("static SIGNED: [i16; 32];")

        assert static.var_type.element_type.name == 'i16'

    def test_array_with_const_size(self):
        """Test array with const identifier as size."""
        program = parse_program("""
            const SIZE: u8 = 16;
            static BUFFER: [u8; SIZE];
        """)

        static = program.items[1]
        assert isinstance(static.var_type, ast.ArrayType)
        assert isinstance(static.var_type.size, ast.Identifier)
        assert static.var_type.size.name == 'SIZE'


# ============================================================================
# Pointer Type Tests
# ============================================================================

class TestPointerTypes:
    """Tests for pointer types: near<T>, far<T>"""

    def test_near_u8_pointer(self):
        """Test near<u8> pointer type."""
        static = parse_static("static PTR: near<u8>;")

        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far == False
        assert isinstance(static.var_type.pointee_type, ast.BasicType)
        assert static.var_type.pointee_type.name == 'u8'

    def test_near_u16_pointer(self):
        """Test near<u16> pointer type."""
        static = parse_static("static PTR: near<u16>;")

        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far == False
        assert static.var_type.pointee_type.name == 'u16'

    def test_far_u8_pointer(self):
        """Test far<u8> pointer type."""
        static = parse_static("static PTR: far<u8>;")

        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far == True
        assert static.var_type.pointee_type.name == 'u8'

    def test_far_u16_pointer(self):
        """Test far<u16> pointer type."""
        static = parse_static("static PTR: far<u16>;")

        assert isinstance(static.var_type, ast.PointerType)
        assert static.var_type.is_far == True

    def test_near_pointer_in_param(self):
        """Test near pointer in function parameter."""
        func = parse_function("fn test(ptr: near<u8>) { }")

        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far == False

    def test_far_pointer_in_param(self):
        """Test far pointer in function parameter."""
        func = parse_function("fn test(ptr: far<u8>) { }")

        assert isinstance(func.params[0].param_type, ast.PointerType)
        assert func.params[0].param_type.is_far == True

    def test_pointer_return_type(self):
        """Test pointer as return type."""
        func = parse_function("fn get_ptr() -> near<u8> { return ptr; }")

        assert isinstance(func.return_type, ast.PointerType)
        assert func.return_type.is_far == False

    def test_pointer_to_array(self):
        """Test pointer to array type."""
        static = parse_static("static PTR: near<[u8; 16]>;")

        assert isinstance(static.var_type, ast.PointerType)
        assert isinstance(static.var_type.pointee_type, ast.ArrayType)

    def test_near_bool_pointer(self):
        """Test near<bool> pointer type."""
        static = parse_static("static PTR: near<bool>;")

        assert static.var_type.pointee_type.name == 'bool'

    def test_pointer_in_let(self):
        """Test pointer in let statement."""
        func = parse_function("fn test() { let p: near<u8> = ptr; }")
        let_stmt = func.body.statements[0]

        assert isinstance(let_stmt.var_type, ast.PointerType)


# ============================================================================
# Function Type Tests
# ============================================================================

class TestFunctionTypes:
    """Tests for function types: fn(...) -> T, far fn(...) -> T"""

    def test_near_fn_no_params_no_return(self):
        """Test fn() function type."""
        static = parse_static("static CALLBACK: fn();")

        assert isinstance(static.var_type, ast.FunctionType)
        assert static.var_type.is_far == False
        assert len(static.var_type.param_types) == 0
        assert static.var_type.return_type is None

    def test_near_fn_with_return(self):
        """Test fn() -> T function type."""
        static = parse_static("static CALLBACK: fn() -> u8;")

        assert isinstance(static.var_type, ast.FunctionType)
        assert static.var_type.is_far == False
        assert isinstance(static.var_type.return_type, ast.BasicType)
        assert static.var_type.return_type.name == 'u8'

    def test_near_fn_with_params(self):
        """Test fn(T, T) function type."""
        static = parse_static("static CALLBACK: fn(u8, u16);")

        assert isinstance(static.var_type, ast.FunctionType)
        assert len(static.var_type.param_types) == 2
        assert static.var_type.param_types[0].name == 'u8'
        assert static.var_type.param_types[1].name == 'u16'

    def test_near_fn_with_params_and_return(self):
        """Test fn(T, T) -> T function type."""
        static = parse_static("static CALLBACK: fn(u8, u8) -> u8;")

        assert isinstance(static.var_type, ast.FunctionType)
        assert len(static.var_type.param_types) == 2
        assert static.var_type.return_type.name == 'u8'

    def test_far_fn_no_params(self):
        """Test far fn() function type."""
        static = parse_static("static CALLBACK: far fn();")

        assert isinstance(static.var_type, ast.FunctionType)
        assert static.var_type.is_far == True

    def test_far_fn_with_return(self):
        """Test far fn() -> T function type."""
        static = parse_static("static CALLBACK: far fn() -> u16;")

        assert isinstance(static.var_type, ast.FunctionType)
        assert static.var_type.is_far == True
        assert static.var_type.return_type.name == 'u16'

    def test_far_fn_with_params(self):
        """Test far fn(T) function type."""
        static = parse_static("static CALLBACK: far fn(u8);")

        assert isinstance(static.var_type, ast.FunctionType)
        assert static.var_type.is_far == True
        assert len(static.var_type.param_types) == 1

    def test_fn_type_in_param(self):
        """Test function type as parameter."""
        func = parse_function("fn call_it(callback: fn()) { }")

        assert isinstance(func.params[0].param_type, ast.FunctionType)

    def test_fn_single_param(self):
        """Test function type with single parameter."""
        static = parse_static("static HANDLER: fn(u8);")

        assert len(static.var_type.param_types) == 1
        assert static.var_type.param_types[0].name == 'u8'

    def test_fn_many_params(self):
        """Test function type with many parameters."""
        static = parse_static("static HANDLER: fn(u8, u8, u16, bool);")

        assert len(static.var_type.param_types) == 4


# ============================================================================
# Never Type Tests
# ============================================================================

class TestNeverType:
    """Tests for never type: !"""

    def test_never_return_type(self):
        """Test ! as return type."""
        func = parse_function("fn main() -> ! { loop { } }")

        assert isinstance(func.return_type, ast.NeverType)

    def test_never_in_entry_function(self):
        """Test ! in entry function."""
        func = parse_function("""
            #[entry]
            fn main() -> ! {
                loop { }
            }
        """)

        assert isinstance(func.return_type, ast.NeverType)

    def test_never_with_infinite_loop(self):
        """Test ! with infinite loop body."""
        func = parse_function("""
            fn endless() -> ! {
                loop {
                    process();
                }
            }
        """)

        assert isinstance(func.return_type, ast.NeverType)

    def test_never_with_halt(self):
        """Test ! with halt pattern."""
        func = parse_function("""
            fn fatal() -> ! {
                loop { }
            }
        """)

        assert isinstance(func.return_type, ast.NeverType)


# ============================================================================
# Struct Type Tests
# ============================================================================

class TestStructTypes:
    """Tests for struct declarations and types."""

    def test_simple_struct(self):
        """Test simple struct declaration."""
        struct = parse_struct("""
            struct Point {
                x: u8,
                y: u8,
            }
        """)

        assert struct.name == 'Point'
        assert len(struct.fields) == 2
        assert struct.fields[0].name == 'x'
        assert struct.fields[0].field_type.name == 'u8'
        assert struct.fields[1].name == 'y'

    def test_struct_mixed_types(self):
        """Test struct with mixed types."""
        struct = parse_struct("""
            struct Player {
                x: u8,
                y: u8,
                health: u16,
                alive: bool,
            }
        """)

        assert len(struct.fields) == 4
        assert struct.fields[0].field_type.name == 'u8'
        assert struct.fields[2].field_type.name == 'u16'
        assert struct.fields[3].field_type.name == 'bool'

    def test_struct_single_field(self):
        """Test struct with single field."""
        struct = parse_struct("""
            struct Wrapper {
                value: u16,
            }
        """)

        assert len(struct.fields) == 1
        assert struct.fields[0].name == 'value'

    def test_struct_many_fields(self):
        """Test struct with many fields."""
        struct = parse_struct("""
            struct Entity {
                x: u8,
                y: u8,
                vx: i8,
                vy: i8,
                sprite: u8,
                flags: u8,
                timer: u8,
                state: u8,
            }
        """)

        assert len(struct.fields) == 8

    def test_struct_with_array_field(self):
        """Test struct with array field."""
        struct = parse_struct("""
            struct Buffer {
                data: [u8; 16],
                length: u8,
            }
        """)

        assert len(struct.fields) == 2
        assert isinstance(struct.fields[0].field_type, ast.ArrayType)

    def test_struct_with_pointer_field(self):
        """Test struct with pointer field."""
        struct = parse_struct("""
            struct Node {
                value: u8,
                next: near<u8>,
            }
        """)

        assert isinstance(struct.fields[1].field_type, ast.PointerType)

    def test_struct_as_static_type(self):
        """Test struct used as static type."""
        program = parse_program("""
            struct Point { x: u8, y: u8 }
            static POS: Point;
        """)

        static = program.items[1]
        assert isinstance(static.var_type, ast.BasicType)  # Struct names are parsed as BasicType
        assert static.var_type.name == 'Point'

    def test_struct_in_let(self):
        """Test struct type in let statement."""
        program = parse_program("""
            struct Point { x: u8, y: u8 }
            fn test() {
                let p: Point = Point { x: 0, y: 0 };
            }
        """)

        func = program.items[1]
        let_stmt = func.body.statements[0]
        assert let_stmt.var_type.name == 'Point'

    def test_struct_no_trailing_comma(self):
        """Test struct without trailing comma."""
        struct = parse_struct("""
            struct Point {
                x: u8,
                y: u8
            }
        """)

        assert len(struct.fields) == 2

    def test_struct_all_same_type(self):
        """Test struct with all same type fields."""
        struct = parse_struct("""
            struct RGB {
                r: u8,
                g: u8,
                b: u8,
            }
        """)

        for field in struct.fields:
            assert field.field_type.name == 'u8'


# ============================================================================
# Enum Type Tests
# ============================================================================

class TestEnumTypes:
    """Tests for enum declarations and types."""

    def test_simple_enum(self):
        """Test simple enum declaration."""
        enum = parse_enum("""
            enum Direction {
                North,
                East,
                South,
                West
            }
        """)

        assert enum.name == 'Direction'
        assert len(enum.variants) == 4
        assert enum.variants[0].name == 'North'
        assert enum.variants[1].name == 'East'
        assert enum.variants[2].name == 'South'
        assert enum.variants[3].name == 'West'

    def test_enum_with_explicit_values(self):
        """Test enum with explicit values."""
        enum = parse_enum("""
            enum State {
                Idle = 0,
                Running = 1,
                Paused = 2,
                GameOver = 3
            }
        """)

        assert enum.variants[0].value.value == 0
        assert enum.variants[1].value.value == 1
        assert enum.variants[2].value.value == 2
        assert enum.variants[3].value.value == 3

    def test_enum_mixed_explicit_implicit(self):
        """Test enum with mixed explicit and implicit values."""
        enum = parse_enum("""
            enum Priority {
                Low = 0,
                Medium,
                High,
                Critical = 10
            }
        """)

        assert enum.variants[0].value.value == 0
        assert enum.variants[1].value is None  # Auto-increment
        assert enum.variants[2].value is None
        assert enum.variants[3].value.value == 10

    def test_enum_hex_values(self):
        """Test enum with hex values."""
        enum = parse_enum("""
            enum Flags {
                None = 0x00,
                Read = 0x01,
                Write = 0x02,
                Execute = 0x04
            }
        """)

        assert enum.variants[0].value.value == 0x00
        assert enum.variants[1].value.value == 0x01
        assert enum.variants[2].value.value == 0x02
        assert enum.variants[3].value.value == 0x04

    def test_enum_single_variant(self):
        """Test enum with single variant."""
        enum = parse_enum("""
            enum Single {
                Only
            }
        """)

        assert len(enum.variants) == 1
        assert enum.variants[0].name == 'Only'

    def test_enum_two_variants(self):
        """Test enum with two variants."""
        enum = parse_enum("""
            enum Boolean {
                False = 0,
                True = 1
            }
        """)

        assert len(enum.variants) == 2

    def test_enum_many_variants(self):
        """Test enum with many variants."""
        enum = parse_enum("""
            enum Key {
                None,
                Up,
                Down,
                Left,
                Right,
                A,
                B,
                Start,
                Select
            }
        """)

        assert len(enum.variants) == 9

    def test_enum_as_static_type(self):
        """Test enum used as static type."""
        program = parse_program("""
            enum State { Idle, Running }
            static CURRENT: State;
        """)

        static = program.items[1]
        assert static.var_type.name == 'State'

    def test_enum_in_param(self):
        """Test enum type in function parameter."""
        program = parse_program("""
            enum State { Idle, Running }
            fn update(state: State) { }
        """)

        func = program.items[1]
        assert func.params[0].param_type.name == 'State'

    def test_enum_with_trailing_comma(self):
        """Test enum with trailing comma."""
        enum = parse_enum("""
            enum Direction {
                North,
                East,
                South,
                West,
            }
        """)

        assert len(enum.variants) == 4


# ============================================================================
# Type Alias Tests
# ============================================================================

class TestTypeAliases:
    """Tests for type aliases."""

    def test_basic_type_alias(self):
        """Test basic type alias."""
        program = parse_program("type Byte = u8;")

        alias = program.items[0]
        assert isinstance(alias, ast.TypeAlias)
        assert alias.name == 'Byte'
        assert isinstance(alias.aliased_type, ast.BasicType)
        assert alias.aliased_type.name == 'u8'

    def test_u16_alias(self):
        """Test u16 type alias."""
        program = parse_program("type Word = u16;")

        alias = program.items[0]
        assert alias.name == 'Word'
        assert alias.aliased_type.name == 'u16'

    def test_array_type_alias(self):
        """Test array type alias."""
        program = parse_program("type Buffer = [u8; 256];")

        alias = program.items[0]
        assert isinstance(alias.aliased_type, ast.ArrayType)
        assert alias.aliased_type.size.value == 256

    def test_pointer_type_alias(self):
        """Test pointer type alias."""
        program = parse_program("type Ptr = near<u8>;")

        alias = program.items[0]
        assert isinstance(alias.aliased_type, ast.PointerType)
        assert alias.aliased_type.is_far == False

    def test_far_pointer_alias(self):
        """Test far pointer type alias."""
        program = parse_program("type FarPtr = far<u8>;")

        alias = program.items[0]
        assert alias.aliased_type.is_far == True

    def test_function_type_alias(self):
        """Test function type alias."""
        program = parse_program("type Callback = fn(u8) -> u8;")

        alias = program.items[0]
        assert isinstance(alias.aliased_type, ast.FunctionType)
        assert len(alias.aliased_type.param_types) == 1

    def test_far_function_alias(self):
        """Test far function type alias."""
        program = parse_program("type FarCallback = far fn();")

        alias = program.items[0]
        assert alias.aliased_type.is_far == True

    def test_alias_used_in_static(self):
        """Test type alias used in static declaration."""
        program = parse_program("""
            type Byte = u8;
            static X: Byte;
        """)

        static = program.items[1]
        assert static.var_type.name == 'Byte'

    def test_alias_used_in_param(self):
        """Test type alias used in function parameter."""
        program = parse_program("""
            type Byte = u8;
            fn process(x: Byte) { }
        """)

        func = program.items[1]
        assert func.params[0].param_type.name == 'Byte'

    def test_multiple_aliases(self):
        """Test multiple type aliases."""
        program = parse_program("""
            type Byte = u8;
            type Word = u16;
            type Flag = bool;
        """)

        assert len(program.items) == 3
        assert program.items[0].name == 'Byte'
        assert program.items[1].name == 'Word'
        assert program.items[2].name == 'Flag'


# ============================================================================
# Const Declaration Tests
# ============================================================================

class TestConstDeclarations:
    """Tests for const declarations with types."""

    def test_const_u8(self):
        """Test const with u8 type."""
        program = parse_program("const VALUE: u8 = 42;")

        const = program.items[0]
        assert isinstance(const, ast.ConstDecl)
        assert const.name == 'VALUE'
        assert const.const_type.name == 'u8'
        assert const.value.value == 42

    def test_const_u16(self):
        """Test const with u16 type."""
        program = parse_program("const SIZE: u16 = 1024;")

        const = program.items[0]
        assert const.const_type.name == 'u16'
        assert const.value.value == 1024

    def test_const_bool(self):
        """Test const with bool type."""
        program = parse_program("const DEBUG: bool = false;")

        const = program.items[0]
        assert const.const_type.name == 'bool'
        assert isinstance(const.value, ast.BooleanLiteral)

    def test_const_hex_value(self):
        """Test const with hex value."""
        program = parse_program("const MASK: u8 = 0xFF;")

        const = program.items[0]
        assert const.value.value == 0xFF

    def test_const_binary_value(self):
        """Test const with binary value."""
        program = parse_program("const FLAGS: u8 = 0b10101010;")

        const = program.items[0]
        assert const.value.value == 0b10101010

    def test_const_expression(self):
        """Test const with expression value."""
        program = parse_program("const DOUBLE: u8 = 21 * 2;")

        const = program.items[0]
        assert isinstance(const.value, ast.BinaryOp)


# ============================================================================
# Static Declaration Type Tests
# ============================================================================

class TestStaticDeclarationTypes:
    """Tests for static declarations with various types."""

    def test_static_mut_u8(self):
        """Test static mut with u8."""
        static = parse_static("static mut COUNTER: u8;")

        assert static.is_mut == True
        assert static.var_type.name == 'u8'

    def test_static_immutable_u8(self):
        """Test static (immutable) with u8."""
        static = parse_static("static VALUE: u8 = 42;")

        assert static.is_mut == False

    def test_static_with_zeropage(self):
        """Test static with zeropage attribute."""
        program = parse_program("""
            #[zeropage(0x10)]
            static mut TEMP: u8;
        """)

        static = program.items[0]
        assert len(static.attributes) == 1
        assert static.attributes[0].name == 'zeropage'

    def test_static_with_ram(self):
        """Test static with ram attribute."""
        program = parse_program("""
            #[ram]
            static mut BUFFER: [u8; 256];
        """)

        static = program.items[0]
        assert static.attributes[0].name == 'ram'

    def test_static_with_hw(self):
        """Test static with hw (hardware) attribute."""
        program = parse_program("""
            #[hw(0x2100)]
            static mut INIDISP: u8;
        """)

        static = program.items[0]
        assert static.attributes[0].name == 'hw'

    def test_static_with_initializer(self):
        """Test static with initializer."""
        static = parse_static("static mut X: u8 = 0;")

        assert static.initializer is not None
        assert isinstance(static.initializer, ast.IntegerLiteral)

    def test_static_array_with_fill(self):
        """Test static array with fill initializer."""
        static = parse_static("static DATA: [u8; 16] = [0; 16];")

        assert isinstance(static.initializer, ast.ArrayFillExpr)

    def test_static_struct_type(self):
        """Test static with struct type."""
        program = parse_program("""
            struct Point { x: u8, y: u8 }
            static mut POS: Point;
        """)

        static = program.items[1]
        assert static.var_type.name == 'Point'


# ============================================================================
# HIR Building Tests
# ============================================================================

class TestTypeHIR:
    """Tests for type HIR building."""

    def test_basic_type_hir(self):
        """Test basic type HIR building."""
        hir_prog = build_hir("""
            #[zeropage]
            static mut x: u8;
        """)

        assert len(hir_prog.statics) == 1
        static = hir_prog.statics[0]
        assert static.name == 'x'

    def test_array_type_hir(self):
        """Test array type HIR building."""
        hir_prog = build_hir("""
            #[ram]
            static BUFFER: [u8; 16];
        """)

        static = hir_prog.statics[0]
        assert static.name == 'BUFFER'

    def test_struct_hir(self):
        """Test struct HIR building."""
        hir_prog = build_hir("""
            struct Point { x: u8, y: u8 }
        """)

        # Struct should be registered in symbol table
        assert hir_prog.symbol_table is not None
        # Look up the struct symbol
        symbol = hir_prog.symbol_table.lookup('Point')
        assert symbol is not None

    def test_enum_hir(self):
        """Test enum HIR building."""
        hir_prog = build_hir("""
            enum State { Idle, Running }
        """)

        # Enum should be registered in symbol table
        symbol = hir_prog.symbol_table.lookup('State')
        assert symbol is not None

    def test_function_param_types_hir(self):
        """Test function parameter types in HIR."""
        hir_prog = build_hir("""
            fn add(a: u8, b: u8) -> u8 {
                return A;
            }
        """)

        func = get_hir_function(hir_prog, 'add')
        assert len(func.parameters) == 2


# ============================================================================
# Edge Cases
# ============================================================================

class TestTypeEdgeCases:
    """Tests for type edge cases."""

    def test_type_same_as_keyword(self):
        """Test that type names don't conflict with keywords."""
        # 'bool' is both a keyword and a type - should work
        static = parse_static("static FLAG: bool;")
        assert static.var_type.name == 'bool'

    def test_custom_type_name(self):
        """Test custom type names."""
        program = parse_program("""
            struct MyCustomType { value: u8 }
            static X: MyCustomType;
        """)

        static = program.items[1]
        assert static.var_type.name == 'MyCustomType'

    def test_type_with_underscore(self):
        """Test type name with underscore."""
        program = parse_program("""
            struct my_type { value: u8 }
            static X: my_type;
        """)

        static = program.items[1]
        assert static.var_type.name == 'my_type'

    def test_type_starting_with_underscore(self):
        """Test type name starting with underscore."""
        program = parse_program("""
            struct _Internal { value: u8 }
            static X: _Internal;
        """)

        static = program.items[1]
        assert static.var_type.name == '_Internal'

    def test_array_of_pointers(self):
        """Test array of pointers."""
        static = parse_static("static PTRS: [near<u8>; 4];")

        assert isinstance(static.var_type, ast.ArrayType)
        assert isinstance(static.var_type.element_type, ast.PointerType)


# ============================================================================
# Parse Error Tests
# ============================================================================

class TestTypeParseErrors:
    """Tests for type parse errors."""

    def test_missing_type_annotation(self):
        """Test missing type annotation fails."""
        with pytest.raises(Exception):
            parse("static X;")

    def test_invalid_type_name(self):
        """Test invalid type name fails."""
        with pytest.raises(Exception):
            parse("static X: 123;")

    def test_array_missing_size(self):
        """Test array missing size fails."""
        with pytest.raises(Exception):
            parse("static X: [u8];")

    def test_array_missing_element_type(self):
        """Test array missing element type fails."""
        with pytest.raises(Exception):
            parse("static X: [; 10];")

    def test_pointer_missing_pointee(self):
        """Test pointer missing pointee type fails."""
        with pytest.raises(Exception):
            parse("static X: near<>;")

    def test_function_type_missing_parens(self):
        """Test function type missing parens fails."""
        with pytest.raises(Exception):
            parse("static X: fn;")

    def test_struct_missing_braces(self):
        """Test struct missing braces fails."""
        with pytest.raises(Exception):
            parse("struct Point x: u8, y: u8")

    def test_enum_missing_braces(self):
        """Test enum missing braces fails."""
        with pytest.raises(Exception):
            parse("enum State Idle, Running")


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
