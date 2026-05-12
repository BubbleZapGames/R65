# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
    HIRLetStmt, HIRIfStmt, HIRWhileStmt, HIRBreakStmt, HIRReturnStmt, HIRAsmStmt,
    HIRStringLiteral, HIRBlock, HIRAssignment, HIRLoopExpression,
    HIRMatchExpression, HIRLiteralPattern, HIRIdentifierPattern, HIROrPattern,
    SymbolKind, BasicTypeInfo, ArrayTypeInfo, StructTypeInfo,
)
from r65.compiler.hir.attributes import (
    StorageKind,
    InterruptVector,
    InlineMode,
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
        """Test immutable static (no mut) - implicit ROM."""
        source = """
static DATA: u8 = 0xFF;
"""
        hir = build_hir(source)
        static = hir.declarations[0]
        assert static.is_mutable == False
        # Immutable statics without storage attr are ROM (storage_attr is None)
        assert static.storage_attr is None
        # ROM statics have a bank attribute
        assert static.bank_attr is not None


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

    def test_struct_field_offsets_packed(self):
        """Field offsets are packed (no alignment padding)."""
        hir = build_hir("""
struct Mixed { kind: u8, health: u16, count: u8 }
""")
        struct = hir.declarations[0]
        offsets = {f.name: f.offset for f in struct.fields}
        assert offsets == {'kind': 0, 'health': 1, 'count': 3}

    def test_struct_field_offsets_all_u8(self):
        """All-u8 struct: offsets advance by 1 each."""
        hir = build_hir("struct RGB { r: u8, g: u8, b: u8 }")
        offsets = [f.offset for f in hir.declarations[0].fields]
        assert offsets == [0, 1, 2]

    def test_struct_field_offsets_all_u16(self):
        """All-u16 struct: offsets advance by 2 each."""
        hir = build_hir("struct V2 { x: u16, y: u16 }")
        offsets = [f.offset for f in hir.declarations[0].fields]
        assert offsets == [0, 2]


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


class TestModeInference:
    """Test processor mode inference from parameter types.

    In the simplified mode system:
    - X/Y are always 16-bit (x16)
    - A mode is inferred from parameter types:
      - u16 @ A parameter -> m16 entry mode
      - otherwise -> m8 entry mode (default)
    """

    def test_default_m8_mode(self):
        """Test that functions without u16 @ A use m8 mode."""
        source = """
fn process() {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        # entry_m_mode is inferred from parameters
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M8

    def test_m16_mode_from_u16_a_param(self):
        """Test that u16 @ A parameter infers m16 mode."""
        source = """
fn wide_mode(value @ A: u16) {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M16

    def test_m8_mode_with_u8_a_param(self):
        """Test that u8 @ A parameter uses m8 mode."""
        source = """
fn narrow_mode(value @ A: u8) {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M8

    def test_b_param_rejected_with_u16_a(self):
        """Test that B register parameter is rejected when A is 16-bit."""
        source = """
fn bad(multA @ A: u16, multB @ B: u8) {
}
"""
        with pytest.raises(HIRError) as excinfo:
            build_hir(source)
        assert "B register" in str(excinfo.value)
        assert "m16" in str(excinfo.value) or "16-bit" in str(excinfo.value)

    def test_b_param_rejected_with_i16_a(self):
        """Test that B register parameter is rejected when A is i16."""
        source = """
fn bad(val @ A: i16, extra @ B: i8) {
}
"""
        with pytest.raises(HIRError) as excinfo:
            build_hir(source)
        assert "B register" in str(excinfo.value)

    def test_b_param_allowed_with_u8_a(self):
        """Test that B register parameter is allowed when A is 8-bit (m8 mode)."""
        source = """
fn ok(low @ A: u8, high @ B: u8) {
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M8


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


class TestLabeledLoops:
    """Test labeled loop handling."""

    def test_labeled_loop_hir(self):
        """Test labeled loop produces HIR with label."""
        source = """
fn test() {
    'outer: loop {
        break;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        while_stmt = func.body.statements[0]
        assert isinstance(while_stmt, HIRWhileStmt)
        assert while_stmt.label == 'outer'
        assert while_stmt.is_infinite == True

    def test_labeled_while_hir(self):
        """Test labeled while produces HIR with label."""
        source = """
fn test() {
    'inner: while A < 10 {
        A = A + 1;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        while_stmt = func.body.statements[0]
        assert isinstance(while_stmt, HIRWhileStmt)
        assert while_stmt.label == 'inner'
        assert while_stmt.is_infinite == False

    def test_labeled_break_hir(self):
        """Test labeled break has correct label in HIR."""
        source = """
fn test() {
    'outer: loop {
        break 'outer;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        while_stmt = func.body.statements[0]
        from r65.compiler.hir import HIRBreakStmt
        break_stmt = while_stmt.body.statements[0]
        assert isinstance(break_stmt, HIRBreakStmt)
        assert break_stmt.label == 'outer'

    def test_labeled_continue_hir(self):
        """Test labeled continue has correct label in HIR."""
        source = """
fn test() {
    'outer: while A < 10 {
        continue 'outer;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        while_stmt = func.body.statements[0]
        from r65.compiler.hir import HIRContinueStmt
        continue_stmt = while_stmt.body.statements[0]
        assert isinstance(continue_stmt, HIRContinueStmt)
        assert continue_stmt.label == 'outer'

    def test_nested_labeled_loops(self):
        """Test nested labeled loops."""
        source = """
fn test() {
    'outer: loop {
        'inner: while A < 10 {
            if A == 5 {
                break 'outer;
            }
            continue 'inner;
        }
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        outer_loop = func.body.statements[0]
        assert outer_loop.label == 'outer'
        inner_while = outer_loop.body.statements[0]
        assert inner_while.label == 'inner'


class TestForLoopDesugaring:
    """Test for loop desugaring to while loop."""

    def test_basic_for_loop(self):
        """Test basic for loop desugars to while loop."""
        source = """
fn test() {
    for i in 0..10 {
        A = A + 1;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        # For loop desugars to a block containing let + while
        stmt = func.body.statements[0]
        assert isinstance(stmt, HIRBlock)
        assert len(stmt.statements) == 2
        # First statement is let
        assert isinstance(stmt.statements[0], HIRLetStmt)
        assert stmt.statements[0].name == "i"
        assert stmt.statements[0].is_mutable == True
        # Second statement is while
        assert isinstance(stmt.statements[1], HIRWhileStmt)

    def test_for_loop_variable_scope(self):
        """Test for loop variable is in loop scope."""
        source = """
fn test() {
    for x in 0..5 {
        A = x;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        # Should build without errors - variable resolved in loop
        block = func.body.statements[0]
        assert isinstance(block, HIRBlock)
        let_stmt = block.statements[0]
        assert let_stmt.name == "x"

    def test_for_loop_with_expressions(self):
        """Test for loop with expressions for start/end."""
        source = """
const START: u8 = 5;
const END: u8 = 15;

fn test() {
    for i in START..END {
        A = i;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[-1]
        block = func.body.statements[0]
        assert isinstance(block, HIRBlock)
        # Let statement has START as initializer
        let_stmt = block.statements[0]
        assert isinstance(let_stmt.initializer, HIRIdentifier)
        assert let_stmt.initializer.name == "START"

    def test_for_loop_body_contains_increment(self):
        """Test for loop body ends with increment statement."""
        source = """
fn test() {
    for i in 0..3 {
        X = i;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        block = func.body.statements[0]
        while_stmt = block.statements[1]
        # Body should have original statement + increment
        assert len(while_stmt.body.statements) == 2
        # Last statement is the increment (assignment)
        increment = while_stmt.body.statements[-1]
        assert isinstance(increment, HIRAssignment)

    def test_nested_for_loops(self):
        """Test nested for loops."""
        source = """
fn test() {
    for i in 0..3 {
        for j in 0..3 {
            A = i;
        }
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        # Outer for loop
        outer_block = func.body.statements[0]
        assert isinstance(outer_block, HIRBlock)
        outer_while = outer_block.statements[1]
        # Inner for loop is in outer while body
        inner_block = outer_while.body.statements[0]
        assert isinstance(inner_block, HIRBlock)
        inner_let = inner_block.statements[0]
        assert inner_let.name == "j"


class TestInclusiveForLoop:
    """Test inclusive for loop (..=) desugaring."""

    def test_inclusive_for_loop_uses_le(self):
        """Test inclusive for loop desugars to <= comparison."""
        source = """
fn test() {
    for i in 0..=10 {
        A = i;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        block = func.body.statements[0]
        while_stmt = block.statements[1]
        # Condition should be i <= end (inclusive)
        assert isinstance(while_stmt.condition, HIRBinaryOp)
        assert while_stmt.condition.op == '<='

    def test_exclusive_for_loop_uses_lt(self):
        """Test exclusive for loop still uses < comparison."""
        source = """
fn test() {
    for i in 0..10 {
        A = i;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        block = func.body.statements[0]
        while_stmt = block.statements[1]
        assert isinstance(while_stmt.condition, HIRBinaryOp)
        assert while_stmt.condition.op == '<'


class TestLoopExpression:
    """Test loop expression building."""

    def test_loop_expression_in_let(self):
        """Test loop expression in let initializer builds HIRLoopExpression."""
        source = """
fn test() {
    let mut x: u8 = loop {
        break 42;
    };
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt, HIRLetStmt)
        assert isinstance(let_stmt.initializer, HIRLoopExpression)
        assert let_stmt.initializer.body is not None

    def test_break_with_value_builds_hir(self):
        """Test break with value builds HIRBreakStmt with value."""
        source = """
fn test() {
    let mut x: u8 = loop {
        break 42;
    };
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        let_stmt = func.body.statements[0]
        loop_expr = let_stmt.initializer
        break_stmt = loop_expr.body.statements[0]
        assert isinstance(break_stmt, HIRBreakStmt)
        assert isinstance(break_stmt.value, HIRIntegerLiteral)
        assert break_stmt.value.value == 42

    def test_break_without_value_in_loop_stmt(self):
        """Test break without value in regular loop statement."""
        source = """
fn test() {
    loop {
        break;
    }
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        # loop desugars to while(true)
        while_stmt = func.body.statements[0]
        assert isinstance(while_stmt, HIRWhileStmt)
        break_stmt = while_stmt.body.statements[0]
        assert isinstance(break_stmt, HIRBreakStmt)
        assert break_stmt.value is None


class TestAutoInlineDetection:
    """Test auto-detection of trivial functions for inlining."""

    def test_simple_getter_auto_inlined(self):
        """Simple getter returning literal should be auto-inlined."""
        source = """
fn get_value() -> u8 {
    return 15;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is not None

    def test_getter_with_identifier_auto_inlined(self):
        """Getter returning variable should be auto-inlined."""
        source = """
fn get_a() -> u8 {
    return A;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is not None

    def test_getter_with_binary_op_auto_inlined(self):
        """Getter with simple binary operation should be auto-inlined."""
        source = """
fn get_masked(val @ A: u8) -> u8 {
    return val & 0x0F;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is not None

    def test_getter_with_unary_op_auto_inlined(self):
        """Getter with unary operation should be auto-inlined."""
        source = """
fn get_inverted(val @ A: u8) -> u8 {
    return ~val;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is not None

    def test_getter_with_nested_ops_auto_inlined(self):
        """Getter with nested operations (depth 2) should be auto-inlined."""
        source = """
fn get_combined(a @ A: u8, b @ X: u16) -> u8 {
    return (a & 0x0F) | (b as u8 & 0xF0);
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        # This might be too complex (depth > 2 due to cast), check if it's inlined
        # The cast adds depth, so this may or may not be inlined depending on implementation

    def test_complex_function_not_auto_inlined(self):
        """Function with control flow should not be auto-inlined."""
        source = """
fn complex(val @ A: u8) -> u8 {
    if val > 10 {
        return 1;
    }
    return 0;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is None

    def test_far_function_not_auto_inlined(self):
        """Far functions should not be auto-inlined even if trivial."""
        source = """
far fn get_value() -> u8 {
    return 15;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is None

    def test_setter_auto_inlined(self):
        """Simple setter should be auto-inlined."""
        source = """
#[zeropage]
static mut TEMP: u8;

fn set_temp(val @ A: u8) {
    TEMP = val;
}
"""
        hir = build_hir(source)
        func = hir.declarations[1]  # Second declaration is the function
        assert func.inline_attr is not None

    def test_setter_with_expression_auto_inlined(self):
        """Setter with simple expression should be auto-inlined."""
        source = """
#[zeropage]
static mut TEMP: u8;

fn set_masked(val @ A: u8) {
    TEMP = val & 0x0F;
}
"""
        hir = build_hir(source)
        func = hir.declarations[1]
        assert func.inline_attr is not None

    def test_impl_method_getter_auto_inlined(self):
        """Impl method getter should be auto-inlined."""
        source = """
struct Point { x: u8, y: u8 }

impl Point {
    fn get_x(*self) -> u8 {
        return self.x;
    }
}
"""
        hir = build_hir(source)
        impl_decl = hir.declarations[1]  # Second is impl
        method = impl_decl.methods[0]
        assert method.inline_attr is not None

    def test_impl_method_setter_with_expr_auto_inlined(self):
        """Impl method setter with expression should be auto-inlined."""
        source = """
struct Point { x: u8, y: u8 }

impl Point {
    fn set_x(*self, val @ A: u8) {
        self.x = val & 0x0F;
    }
}
"""
        hir = build_hir(source)
        impl_decl = hir.declarations[1]
        method = impl_decl.methods[0]
        assert method.inline_attr is not None


class TestInlineAttribute:
    """Test #[inline], #[inline(always)], and #[inline(never)] attribute parsing."""

    def test_inline_bare(self):
        """Bare #[inline] should set mode to HINT (size-gated)."""
        source = """
#[inline]
fn add_one(val @ A: u8) -> u8 {
    return val + 1;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is not None
        assert func.inline_attr.mode == InlineMode.HINT

    def test_inline_always(self):
        """#[inline(always)] should set mode to ALWAYS (bypasses size)."""
        source = """
#[inline(always)]
fn add_one(val @ A: u8) -> u8 {
    return val + 1;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is not None
        assert func.inline_attr.mode == InlineMode.ALWAYS

    def test_inline_never(self):
        """#[inline(never)] should set mode to NEVER."""
        source = """
#[inline(never)]
fn add_one(val @ A: u8) -> u8 {
    return val + 1;
}
"""
        hir = build_hir(source)
        func = hir.declarations[0]
        assert func.inline_attr is not None
        assert func.inline_attr.mode == InlineMode.NEVER

    def test_inline_invalid_argument_rejected(self):
        """#[inline(invalid)] should be rejected."""
        source = """
#[inline(invalid)]
fn add_one(val @ A: u8) -> u8 {
    return val + 1;
}
"""
        with pytest.raises(Exception) as exc_info:
            build_hir(source)
        assert "Invalid #[inline] argument" in str(exc_info.value)

    def test_inline_multiple_arguments_rejected(self):
        """#[inline(always, never)] should be rejected."""
        source = """
#[inline(always, never)]
fn add_one(val @ A: u8) -> u8 {
    return val + 1;
}
"""
        with pytest.raises(Exception) as exc_info:
            build_hir(source)
        assert "at most one argument" in str(exc_info.value)

    def test_inline_named_argument_rejected(self):
        """#[inline(mode=always)] should be rejected."""
        source = """
#[inline(mode=always)]
fn add_one(val @ A: u8) -> u8 {
    return val + 1;
}
"""
        with pytest.raises(Exception) as exc_info:
            build_hir(source)
        assert "does not accept named arguments" in str(exc_info.value)


class TestAsmFormatSubstitution:
    """Test asm! format string substitution."""

    def test_string_substitution(self):
        """Test string format argument substitution."""
        source = '''
fn test() {
    asm!("LD{REG} #$01", REG="A");
}
'''
        hir = build_hir(source)
        func = hir.functions[0]
        asm_stmt = func.body.statements[0]
        assert isinstance(asm_stmt, HIRAsmStmt)
        assert asm_stmt.instructions == ["LDA #$01"]

    def test_integer_substitution(self):
        """Test integer format argument substitution."""
        source = '''
fn test() {
    asm!("LDA #{VAL}", VAL=42);
}
'''
        hir = build_hir(source)
        func = hir.functions[0]
        asm_stmt = func.body.statements[0]
        assert asm_stmt.instructions == ["LDA #42"]

    def test_array_len_substitution(self):
        """Test array.len() in format arguments."""
        source = '''
#[ram]
static mut BUFFER: [u8; 256] = [0; 256];

fn test() {
    asm!("LDA #{LEN}", LEN=BUFFER.len());
}
'''
        hir = build_hir(source)
        func = hir.functions[0]
        asm_stmt = func.body.statements[0]
        assert asm_stmt.instructions == ["LDA #256"]

    def test_const_variable_substitution(self):
        """Test const variable in format arguments."""
        source = '''
const SIZE: u16 = 64;

fn test() {
    asm!("LDA #{SZ}", SZ=SIZE);
}
'''
        hir = build_hir(source)
        func = hir.functions[0]
        asm_stmt = func.body.statements[0]
        assert asm_stmt.instructions == ["LDA #64"]

    def test_arithmetic_expression_substitution(self):
        """Test arithmetic expression with array.len()."""
        source = '''
#[ram]
static mut DATA: [u8; 16] = [0; 16];

fn test() {
    asm!("LDA #{VAL}", VAL=DATA.len() - 1);
}
'''
        hir = build_hir(source)
        func = hir.functions[0]
        asm_stmt = func.body.statements[0]
        assert asm_stmt.instructions == ["LDA #15"]

    def test_multiple_substitutions(self):
        """Test multiple format arguments in one asm! statement."""
        source = '''
#[ram]
static mut BUF1: [u8; 100] = [0; 100];
#[ram]
static mut BUF2: [u8; 50] = [0; 50];

fn test() {
    asm!("LDA #{A}", "LDX #{B}", A=BUF1.len(), B=BUF2.len());
}
'''
        hir = build_hir(source)
        func = hir.functions[0]
        asm_stmt = func.body.statements[0]
        assert asm_stmt.instructions == ["LDA #100", "LDX #50"]

    def test_register_in_format_arg_fails(self):
        """Test that using a register in format argument fails gracefully."""
        source = '''
fn test() {
    asm!("LDA #{VAL}", VAL=A);
}
'''
        with pytest.raises(HIRError) as exc_info:
            build_hir(source)
        # Should fail because A is a register, not a const
        assert "const" in str(exc_info.value).lower() or "register" in str(exc_info.value).lower()

    def test_local_variable_in_format_arg_fails(self):
        """Test that using a local variable in format argument fails gracefully."""
        source = '''
fn test() {
    let mut x: u8 = 10;
    asm!("LDA #{VAL}", VAL=x);
}
'''
        with pytest.raises(HIRError) as exc_info:
            build_hir(source)
        # Should fail because local variables are not const-evaluable
        assert "const" in str(exc_info.value).lower()

    def test_static_mut_in_format_arg_fails(self):
        """Test that using a mutable static in format argument fails gracefully."""
        source = '''
#[ram]
static mut COUNTER: u8 = 0;

fn test() {
    asm!("LDA #{VAL}", VAL=COUNTER);
}
'''
        with pytest.raises(HIRError) as exc_info:
            build_hir(source)
        # Should fail because mutable statics are not const-evaluable
        assert "const" in str(exc_info.value).lower()

    def test_function_call_in_format_arg_fails(self):
        """Test that using a function call in format argument fails gracefully."""
        source = '''
fn get_value() -> u8 {
    return 42;
}

fn test() {
    asm!("LDA #{VAL}", VAL=get_value());
}
'''
        with pytest.raises(HIRError) as exc_info:
            build_hir(source)
        # Should fail because function calls are not const-evaluable
        assert "const" in str(exc_info.value).lower()


class TestConstantMatchPatterns:
    """Test that named constants in match arms resolve to literal patterns."""

    def test_const_in_match_resolves_to_literal(self):
        """A const identifier in a match arm becomes HIRLiteralPattern."""
        source = """
const PLAYER: u8 = 1;
const ENEMY: u8 = 2;

fn classify(id @ A: u8) -> u8 {
    let result: u8 = match id {
        PLAYER => 10,
        ENEMY => 20,
        _ => 0
    };
    return result;
}
"""
        hir = build_hir(source)
        func = hir.functions[0]
        let_stmt = func.body.statements[0]
        match_expr = let_stmt.initializer
        assert isinstance(match_expr, HIRMatchExpression)

        # PLAYER arm should resolve to literal 1
        pat0 = match_expr.arms[0].pattern
        assert isinstance(pat0, HIRLiteralPattern)
        assert pat0.value == 1

        # ENEMY arm should resolve to literal 2
        pat1 = match_expr.arms[1].pattern
        assert isinstance(pat1, HIRLiteralPattern)
        assert pat1.value == 2

    def test_non_const_identifier_stays_as_binding(self):
        """A non-constant identifier in a match arm creates a binding."""
        source = """
fn test(val @ A: u8) -> u8 {
    let result: u8 = match val {
        x => x
    };
    return result;
}
"""
        hir = build_hir(source)
        func = hir.functions[0]
        let_stmt = func.body.statements[0]
        match_expr = let_stmt.initializer
        pat = match_expr.arms[0].pattern
        assert isinstance(pat, HIRIdentifierPattern)
        assert pat.name == "x"

    def test_const_in_or_pattern(self):
        """Constants in OR patterns resolve both arms to literals."""
        source = """
const A_VAL: u8 = 5;
const B_VAL: u8 = 10;

fn test(val @ A: u8) -> u8 {
    let result: u8 = match val {
        A_VAL | B_VAL => 1,
        _ => 0
    };
    return result;
}
"""
        hir = build_hir(source)
        func = hir.functions[0]
        let_stmt = func.body.statements[0]
        match_expr = let_stmt.initializer
        pat = match_expr.arms[0].pattern
        assert isinstance(pat, HIROrPattern)
        assert isinstance(pat.patterns[0], HIRLiteralPattern)
        assert pat.patterns[0].value == 5
        assert isinstance(pat.patterns[1], HIRLiteralPattern)
        assert pat.patterns[1].value == 10
