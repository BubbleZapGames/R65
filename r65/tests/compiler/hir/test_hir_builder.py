"""
Tests for HIR (High-level Intermediate Representation) building.

Tests the conversion from AST to HIR, including:
- Symbol table construction
- Type resolution
- Attribute processing
- Name resolution
- Const evaluation
"""
import pytest

from r65.compiler.frontend import Parser
from r65.compiler.hir import (
    HIRBuilder, HIRError,
    HIRProgram, HIRFunctionDecl, HIRStaticDecl, HIRConstDecl,
    HIRStructDecl, HIREnumDecl,
    HIRIntegerLiteral, HIRBooleanLiteral, HIRIdentifier, HIRRegister,
    HIRBinaryOp, HIRUnaryOp, HIRFunctionCall, HIRArrayIndex,
    HIRLetStmt, HIRIfStmt, HIRWhileStmt, HIRReturnStmt,
    HIRStringLiteral,
    SymbolKind, BasicTypeInfo, ArrayTypeInfo, StructTypeInfo,
)
from r65.compiler.hir.attributes import (
    StorageKind, MMode, XMode, ModeTransition,
    InterruptVector,
)


def build_hir(source: str) -> HIRProgram:
    """Helper to parse and build HIR from source."""
    parser = Parser()
    ast = parser.parse(source)
    builder = HIRBuilder()
    return builder.build_program(ast)


class TestBasicHIRBuilding:
    """Basic HIR building from AST."""

    def test_empty_program(self):
        """Test building HIR for empty program."""
        hir = build_hir("")
        assert isinstance(hir, HIRProgram)
        assert len(hir.declarations) == 0

    def test_simple_function(self):
        """Test building HIR for simple function."""
        source = """
fn foo() {
}
"""
        hir = build_hir(source)
        assert len(hir.declarations) == 1
        func = hir.declarations[0]
        assert isinstance(func, HIRFunctionDecl)
        assert func.name == "foo"
        assert len(func.parameters) == 0
        assert func.return_type is None

    def test_function_with_parameters(self):
        """Test function with typed parameters."""
        source = """
fn add(a: u8, b: u16) -> u8 {
    return A;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert len(func.parameters) == 2
        assert func.parameters[0].name == "a"
        assert isinstance(func.parameters[0].param_type, BasicTypeInfo)
        assert func.parameters[0].param_type.name == "u8"
        assert func.parameters[1].name == "b"
        assert func.parameters[1].param_type.name == "u16"

    def test_function_with_register_binding(self):
        """Test function parameter with register binding."""
        source = """
fn process(value @ A: u8) -> u8 {
    return A;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert len(func.parameters) == 1
        param = func.parameters[0]
        assert param.name == "value"
        assert param.binding is not None
        assert param.binding.register_name == "A"


class TestStaticDeclarations:
    """Test static variable declarations."""

    def test_static_with_zeropage(self):
        """Test static with zeropage attribute."""
        source = """
#[zeropage(0x10)]
static mut TEMP: u8;
"""
        hir = build_hir(source)
        assert len(hir.declarations) == 1
        static = hir.declarations[0]
        assert isinstance(static, HIRStaticDecl)
        assert static.name == "TEMP"
        assert static.is_mutable == True
        assert static.storage_attr is not None
        assert static.storage_attr.storage_kind == StorageKind.ZEROPAGE
        assert static.storage_attr.address == 0x10

    def test_static_with_ram(self):
        """Test static with ram attribute."""
        source = """
#[ram]
static mut BUFFER: [u8; 256];
"""
        hir = build_hir(source)
        static = hir.declarations[0]
        assert static.storage_attr.storage_kind == StorageKind.RAM
        assert isinstance(static.var_type, ArrayTypeInfo)
        assert static.var_type.size == 256

    def test_static_with_initializer(self):
        """Test static with initializer."""
        source = """
#[ram]
static mut COUNT: u8 = 42;
"""
        hir = build_hir(source)
        static = hir.declarations[0]
        assert static.initializer is not None
        assert isinstance(static.initializer, HIRIntegerLiteral)
        assert static.initializer.value == 42

    def test_immutable_static(self):
        """Test immutable static (no mut)."""
        source = """
#[rom]
static DATA: u8 = 0xFF;
"""
        hir = build_hir(source)
        static = hir.declarations[0]
        assert static.is_mutable == False
        assert static.storage_attr.storage_kind == StorageKind.ROM


class TestConstDeclarations:
    """Test const declarations."""

    def test_simple_const(self):
        """Test simple const declaration."""
        source = """
const MAX_SIZE: u8 = 255;
"""
        hir = build_hir(source)
        assert len(hir.declarations) == 1
        const = hir.declarations[0]
        assert isinstance(const, HIRConstDecl)
        assert const.name == "MAX_SIZE"
        assert const.value is not None

    def test_const_expression(self):
        """Test const with expression value."""
        source = """
const TILE_SIZE: u8 = 8;
const ROW_SIZE: u16 = 256 / 8;
"""
        hir = build_hir(source)
        assert len(hir.declarations) == 2


class TestStructDeclarations:
    """Test struct declarations."""

    def test_simple_struct(self):
        """Test simple struct declaration."""
        source = """
struct Player {
    x: u8,
    y: u8,
    health: u16,
}
"""
        hir = build_hir(source)
        struct = hir.declarations[0]
        assert isinstance(struct, HIRStructDecl)
        assert struct.name == "Player"
        assert len(struct.fields) == 3
        assert struct.fields[0].name == "x"
        assert struct.fields[1].name == "y"
        assert struct.fields[2].name == "health"

    def test_struct_field_types(self):
        """Test struct field type resolution."""
        source = """
struct Point {
    x: u16,
    y: u16,
}
"""
        hir = build_hir(source)
        struct = hir.declarations[0]
        assert struct.fields[0].field_type.name == "u16"
        assert struct.fields[1].field_type.name == "u16"


class TestEnumDeclarations:
    """Test enum declarations."""

    def test_simple_enum(self):
        """Test simple enum declaration."""
        source = """
enum Direction {
    North = 0,
    East,
    South,
    West,
}
"""
        hir = build_hir(source)
        enum = hir.declarations[0]
        assert isinstance(enum, HIREnumDecl)
        assert enum.name == "Direction"
        assert len(enum.variants) == 4
        assert enum.variants[0].name == "North"
        assert enum.variants[1].name == "East"

    def test_enum_explicit_values(self):
        """Test enum with explicit values."""
        source = """
enum Flags {
    None = 0,
    Read = 1,
    Write = 2,
    Execute = 4,
}
"""
        hir = build_hir(source)
        enum = hir.declarations[0]
        assert len(enum.variants) == 4


class TestSymbolTable:
    """Test symbol table construction."""

    def test_function_symbol(self):
        """Test function added to symbol table."""
        source = """
fn my_func() {
}
"""
        hir = build_hir(source)
        symbol = hir.symbol_table.lookup("my_func")
        assert symbol is not None
        assert symbol.kind == SymbolKind.FUNCTION

    def test_static_symbol(self):
        """Test static variable added to symbol table."""
        source = """
#[ram]
static mut COUNTER: u8;
"""
        hir = build_hir(source)
        symbol = hir.symbol_table.lookup("COUNTER")
        assert symbol is not None
        assert symbol.kind == SymbolKind.STATIC_VAR

    def test_const_symbol(self):
        """Test const added to symbol table."""
        source = """
const SIZE: u8 = 10;
"""
        hir = build_hir(source)
        symbol = hir.symbol_table.lookup("SIZE")
        assert symbol is not None
        assert symbol.kind == SymbolKind.CONST

    def test_struct_symbol(self):
        """Test struct added to symbol table."""
        source = """
struct Point { x: u8, y: u8 }
"""
        hir = build_hir(source)
        symbol = hir.symbol_table.lookup("Point")
        assert symbol is not None
        assert symbol.kind == SymbolKind.STRUCT

    def test_enum_symbol(self):
        """Test enum added to symbol table."""
        source = """
enum State { Idle, Running }
"""
        hir = build_hir(source)
        symbol = hir.symbol_table.lookup("State")
        assert symbol is not None
        assert symbol.kind == SymbolKind.ENUM

    def test_register_symbols(self):
        """Test hardware registers in symbol table."""
        hir = build_hir("")
        # Hardware registers should be pre-populated
        for reg in ["A", "X", "Y", "STATUS", "D", "DBR", "PBR", "S"]:
            symbol = hir.symbol_table.lookup(reg)
            assert symbol is not None, f"Register {reg} not found"
            assert symbol.kind == SymbolKind.REGISTER


class TestModeAttributes:
    """Test processor mode attribute processing."""

    def test_mode_m8_x8(self):
        """Test #[mode(m8, x8)] attribute."""
        source = """
#[mode(m8, x8)]
fn process() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.mode_attr is not None
        assert func.mode_attr.m_mode == MMode.M8
        assert func.mode_attr.x_mode == XMode.X8

    def test_mode_m16_x16(self):
        """Test #[mode(m16, x16)] attribute."""
        source = """
#[mode(m16, x16)]
fn wide_mode() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.mode_attr.m_mode == MMode.M16
        assert func.mode_attr.x_mode == XMode.X16

    def test_mode_with_transition(self):
        """Test mode with transition attribute."""
        source = """
#[mode(m8, x8, transition=inline)]
fn safe_func() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.mode_attr.transition == ModeTransition.INLINE


class TestPreservesAttribute:
    """Test #[preserves(...)] attribute processing."""

    def test_preserves_registers(self):
        """Test preserves attribute with registers."""
        source = """
#[preserves(X, Y)]
fn preserve_xy() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.preserves_attr is not None
        assert "X" in func.preserves_attr.registers
        assert "Y" in func.preserves_attr.registers


class TestInterruptAttribute:
    """Test #[interrupt(...)] attribute processing."""

    def test_interrupt_nmi(self):
        """Test NMI interrupt handler."""
        source = """
#[interrupt(nmi)]
fn vblank() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.interrupt_attr is not None
        assert func.interrupt_attr.vector == InterruptVector.NMI

    def test_interrupt_irq(self):
        """Test IRQ interrupt handler."""
        source = """
#[interrupt(irq)]
fn timer() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.interrupt_attr.vector == InterruptVector.IRQ


class TestExpressionBuilding:
    """Test expression HIR building."""

    def test_integer_literal(self):
        """Test integer literal expression."""
        source = """
#[ram]
static mut VAL: u8 = 42;
"""
        hir = build_hir(source)
        init = hir.declarations[0].initializer
        assert isinstance(init, HIRIntegerLiteral)
        assert init.value == 42

    def test_hex_literal(self):
        """Test hex literal expression."""
        source = """
#[ram]
static mut VAL: u8 = 0xFF;
"""
        hir = build_hir(source)
        init = hir.declarations[0].initializer
        assert init.value == 0xFF

    def test_binary_literal(self):
        """Test binary literal expression."""
        source = """
#[ram]
static mut VAL: u8 = 0b10101010;
"""
        hir = build_hir(source)
        init = hir.declarations[0].initializer
        assert init.value == 0b10101010

    def test_boolean_literal(self):
        """Test boolean literal expression."""
        source = """
#[ram]
static mut FLAG: bool = true;
"""
        hir = build_hir(source)
        init = hir.declarations[0].initializer
        assert isinstance(init, HIRBooleanLiteral)
        assert init.value == True

    def test_string_literal(self):
        """Test string literal for array init."""
        source = """
#[ram]
static mut MSG: [u8; 8] = "Hello";
"""
        hir = build_hir(source)
        init = hir.declarations[0].initializer
        assert isinstance(init, HIRStringLiteral)
        assert init.value == "Hello"


class TestStatementBuilding:
    """Test statement HIR building."""

    def test_let_statement(self):
        """Test let statement building."""
        source = """
#[mode(m8, x8)]
fn test() {
    let x: u8 = 10;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt, HIRLetStmt)
        assert stmt.name == "x"
        assert stmt.is_mutable == False

    def test_let_mut_statement(self):
        """Test let mut statement."""
        source = """
#[mode(m8, x8)]
fn test() {
    let mut count: u8 = 0;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        stmt = func.body.statements[0]
        assert stmt.is_mutable == True

    def test_if_statement(self):
        """Test if statement building."""
        source = """
#[mode(m8, x8)]
fn test() {
    if A == 0 {
        X = 1;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt, HIRIfStmt)
        assert stmt.condition is not None
        assert stmt.then_block is not None

    def test_if_else_statement(self):
        """Test if-else statement."""
        source = """
#[mode(m8, x8)]
fn test() {
    if A == 0 {
        X = 1;
    } else {
        X = 2;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        stmt = func.body.statements[0]
        assert stmt.else_block is not None

    def test_while_statement(self):
        """Test while statement building."""
        source = """
#[mode(m8, x8)]
fn test() {
    while A != 0 {
        A = A - 1;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt, HIRWhileStmt)

    def test_return_statement(self):
        """Test return statement."""
        source = """
#[mode(m8, x8)]
fn get_value() -> u8 {
    return A;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt, HIRReturnStmt)


class TestNameResolution:
    """Test identifier resolution in HIR."""

    def test_static_reference(self):
        """Test reference to static variable resolves."""
        source = """
#[ram]
static mut COUNT: u8 = 0;

#[mode(m8, x8)]
fn increment() {
    COUNT = COUNT + 1;
}
"""
        hir = build_hir(source)
        # Should build without error - name resolution worked

    def test_const_reference(self):
        """Test reference to const resolves."""
        source = """
const MAX: u8 = 100;

#[mode(m8, x8)]
fn check() {
    if A == MAX {
        X = 1;
    }
}
"""
        hir = build_hir(source)
        # Should build without error - const reference resolved

    def test_undefined_identifier_error(self):
        """Test error on undefined identifier."""
        source = """
#[mode(m8, x8)]
fn test() {
    A = UNDEFINED;
}
"""
        with pytest.raises(HIRError) as excinfo:
            build_hir(source)
        assert "Undefined" in str(excinfo.value) or "undefined" in str(excinfo.value).lower()


class TestFarFunctions:
    """Test far function handling."""

    def test_far_function(self):
        """Test far function declaration."""
        source = """
far fn cross_bank() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.is_far == True

    def test_near_function(self):
        """Test regular (near) function."""
        source = """
fn local_func() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.is_far == False
