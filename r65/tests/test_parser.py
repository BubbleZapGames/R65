"""
Tests for the R65 parser.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compiler.frontend import parse, ParseError, ast


def test_simple_function():
    """Test parsing a simple function."""
    source = """
    fn add(a: u8, b: u8) -> u8 {
        return a + b;
    }
    """

    program = parse(source)

    assert isinstance(program, ast.Program)
    assert len(program.items) == 1

    func = program.items[0]
    assert isinstance(func, ast.FunctionDecl)
    assert func.name == 'add'
    assert len(func.params) == 2
    assert func.params[0].name == 'a'
    assert isinstance(func.params[0].param_type, ast.BasicType)
    assert func.params[0].param_type.name == 'u8'

    print("✓ Simple function test passed")


def test_function_with_attributes():
    """Test parsing function with attributes."""
    source = """
    #[mode(m8, x8)]
    #[preserves(X, Y)]
    fn process() {
        A = 42;
    }
    """

    program = parse(source)
    func = program.items[0]

    assert len(func.attributes) == 2
    assert func.attributes[0].name == 'mode'
    assert func.attributes[1].name == 'preserves'

    print("✓ Function with attributes test passed")


def test_register_aliasing():
    """Test parsing register aliasing."""
    source = """
    fn test() {
        let value @ A = 100;
    }
    """

    program = parse(source)
    func = program.items[0]
    block = func.body
    let_stmt = block.statements[0]

    assert isinstance(let_stmt, ast.LetStmt)
    assert let_stmt.name == 'value'
    assert isinstance(let_stmt.binding, ast.Register)
    assert let_stmt.binding.name == 'A'

    print("✓ Register aliasing test passed")


def test_static_declaration():
    """Test parsing static variable declaration."""
    source = """
    #[zeropage(0x20)]
    static mut COUNTER: u16 = 0;
    """

    program = parse(source)
    static = program.items[0]

    assert isinstance(static, ast.StaticDecl)
    assert static.is_mut == True
    assert static.name == 'COUNTER'
    assert isinstance(static.var_type, ast.BasicType)
    assert static.var_type.name == 'u16'
    assert isinstance(static.initializer, ast.IntegerLiteral)
    assert static.initializer.value == 0
    assert len(static.attributes) == 1
    assert static.attributes[0].name == 'zeropage'

    print("✓ Static declaration test passed")


def test_struct_declaration():
    """Test parsing struct declaration."""
    source = """
    struct Player {
        x: u8,
        y: u8,
        health: u16,
    }
    """

    program = parse(source)
    struct = program.items[0]

    assert isinstance(struct, ast.StructDecl)
    assert struct.name == 'Player'
    assert len(struct.fields) == 3
    assert struct.fields[0].name == 'x'
    assert struct.fields[0].field_type.name == 'u8'

    print("✓ Struct declaration test passed")


def test_enum_declaration():
    """Test parsing enum declaration."""
    source = """
    enum Direction {
        North = 0,
        East,
        South,
        West
    }
    """

    program = parse(source)
    enum = program.items[0]

    assert isinstance(enum, ast.EnumDecl)
    assert enum.name == 'Direction'
    assert len(enum.variants) == 4
    assert enum.variants[0].name == 'North'
    assert isinstance(enum.variants[0].value, ast.IntegerLiteral)
    assert enum.variants[0].value.value == 0
    assert enum.variants[1].value is None  # Auto-increment

    print("✓ Enum declaration test passed")


def test_if_statement():
    """Test parsing if statement."""
    source = """
    fn test() {
        if x > 10 {
            return 1;
        } else {
            return 0;
        }
    }
    """

    program = parse(source)
    func = program.items[0]
    if_stmt = func.body.statements[0]

    assert isinstance(if_stmt, ast.IfStmt)
    assert isinstance(if_stmt.condition, ast.BinaryOp)
    assert if_stmt.condition.op == '>'
    assert isinstance(if_stmt.then_block, ast.Block)
    assert isinstance(if_stmt.else_block, ast.Block)

    print("✓ If statement test passed")


def test_loop_statements():
    """Test parsing loop statements."""
    source = """
    fn test() {
        loop {
            break;
        }

        while x < 10 {
            x = x + 1;
            continue;
        }
    }
    """

    program = parse(source)
    func = program.items[0]

    loop_stmt = func.body.statements[0]
    assert isinstance(loop_stmt, ast.LoopStmt)
    assert isinstance(loop_stmt.body.statements[0], ast.BreakStmt)

    while_stmt = func.body.statements[1]
    assert isinstance(while_stmt, ast.WhileStmt)
    assert isinstance(while_stmt.condition, ast.BinaryOp)
    assert isinstance(while_stmt.body.statements[1], ast.ContinueStmt)

    print("✓ Loop statements test passed")


def test_binary_operations():
    """Test parsing binary operations."""
    source = """
    fn test() {
        let a = 1 + 2 * 3;
        let b = x << 2;
        let c = y & 0xFF;
        let d = flag && other;
    }
    """

    program = parse(source)
    func = program.items[0]

    # Check precedence: 1 + 2 * 3 should be 1 + (2 * 3)
    stmt1 = func.body.statements[0]
    expr = stmt1.initializer
    assert isinstance(expr, ast.BinaryOp)
    assert expr.op == '+'
    assert isinstance(expr.right, ast.BinaryOp)
    assert expr.right.op == '*'

    print("✓ Binary operations test passed")


def test_function_call():
    """Test parsing function calls."""
    source = """
    fn test() {
        let result = calculate(10, 20);
        process();
    }
    """

    program = parse(source)
    func = program.items[0]

    call1 = func.body.statements[0].initializer
    assert isinstance(call1, ast.FunctionCall)
    assert isinstance(call1.func, ast.Identifier)
    assert len(call1.args) == 2

    call2 = func.body.statements[1].expr
    assert isinstance(call2, ast.FunctionCall)
    assert len(call2.args) == 0

    print("✓ Function call test passed")


def test_array_and_field_access():
    """Test parsing array indexing and field access."""
    source = """
    fn test() {
        let x = array[0];
        let y = player.health;
        let z = enemies[i].x;
    }
    """

    program = parse(source)
    func = program.items[0]

    # Array access
    stmt1 = func.body.statements[0]
    assert isinstance(stmt1.initializer, ast.ArrayIndex)

    # Field access
    stmt2 = func.body.statements[1]
    assert isinstance(stmt2.initializer, ast.FieldAccess)
    assert stmt2.initializer.field == 'health'

    # Chained access
    stmt3 = func.body.statements[2]
    assert isinstance(stmt3.initializer, ast.FieldAccess)
    assert isinstance(stmt3.initializer.base, ast.ArrayIndex)

    print("✓ Array and field access test passed")


def test_type_cast():
    """Test parsing type casts."""
    source = """
    fn test() {
        let x: u16 = y as u16;
        let b: bool = flag as bool;
    }
    """

    program = parse(source)
    func = program.items[0]

    stmt1 = func.body.statements[0]
    assert isinstance(stmt1.initializer, ast.TypeCast)
    assert isinstance(stmt1.initializer.target_type, ast.BasicType)

    print("✓ Type cast test passed")


def test_assignment():
    """Test parsing assignments."""
    source = """
    fn test() {
        x = 10;
        array[0] = 20;
        player.health = 100;
    }
    """

    program = parse(source)
    func = program.items[0]

    # Simple assignment
    stmt1 = func.body.statements[0]
    assert isinstance(stmt1.expr, ast.Assignment)

    # Array assignment
    stmt2 = func.body.statements[1]
    assert isinstance(stmt2.expr, ast.Assignment)
    assert isinstance(stmt2.expr.target, ast.ArrayIndex)

    # Field assignment
    stmt3 = func.body.statements[2]
    assert isinstance(stmt3.expr, ast.Assignment)
    assert isinstance(stmt3.expr.target, ast.FieldAccess)

    print("✓ Assignment test passed")


def test_compound_assignment():
    """Test parsing compound assignments."""
    source = """
    fn test() {
        x += 5;
        y -= 3;
        z *= 2;
        w /= 4;
        a %= 8;
        b &= 0x0F;
        c |= 0x80;
        d ^= 0xFF;
        e <<= 2;
        f >>= 1;
        A += value;
        array[i] &= mask;
        player.health -= damage;
    }
    """

    program = parse(source)
    func = program.items[0]

    # Test arithmetic compound assignments
    stmt1 = func.body.statements[0]
    assert isinstance(stmt1.expr, ast.CompoundAssignment)
    assert stmt1.expr.operator == '+'
    assert isinstance(stmt1.expr.target, ast.Identifier)
    assert stmt1.expr.target.name == 'x'

    stmt2 = func.body.statements[1]
    assert isinstance(stmt2.expr, ast.CompoundAssignment)
    assert stmt2.expr.operator == '-'

    stmt3 = func.body.statements[2]
    assert isinstance(stmt3.expr, ast.CompoundAssignment)
    assert stmt3.expr.operator == '*'

    stmt4 = func.body.statements[3]
    assert isinstance(stmt4.expr, ast.CompoundAssignment)
    assert stmt4.expr.operator == '/'

    stmt5 = func.body.statements[4]
    assert isinstance(stmt5.expr, ast.CompoundAssignment)
    assert stmt5.expr.operator == '%'

    # Test bitwise compound assignments
    stmt6 = func.body.statements[5]
    assert isinstance(stmt6.expr, ast.CompoundAssignment)
    assert stmt6.expr.operator == '&'

    stmt7 = func.body.statements[6]
    assert isinstance(stmt7.expr, ast.CompoundAssignment)
    assert stmt7.expr.operator == '|'

    stmt8 = func.body.statements[7]
    assert isinstance(stmt8.expr, ast.CompoundAssignment)
    assert stmt8.expr.operator == '^'

    # Test shift compound assignments
    stmt9 = func.body.statements[8]
    assert isinstance(stmt9.expr, ast.CompoundAssignment)
    assert stmt9.expr.operator == '<<'

    stmt10 = func.body.statements[9]
    assert isinstance(stmt10.expr, ast.CompoundAssignment)
    assert stmt10.expr.operator == '>>'

    # Test register compound assignment
    stmt11 = func.body.statements[10]
    assert isinstance(stmt11.expr, ast.CompoundAssignment)
    assert isinstance(stmt11.expr.target, ast.Register)
    assert stmt11.expr.target.name == 'A'

    # Test array compound assignment
    stmt12 = func.body.statements[11]
    assert isinstance(stmt12.expr, ast.CompoundAssignment)
    assert isinstance(stmt12.expr.target, ast.ArrayIndex)

    # Test field compound assignment
    stmt13 = func.body.statements[12]
    assert isinstance(stmt13.expr, ast.CompoundAssignment)
    assert isinstance(stmt13.expr.target, ast.FieldAccess)

    print("✓ Compound assignment test passed")


def test_far_function():
    """Test parsing far functions."""
    source = """
    #[bank(1)]
    far fn cross_bank() -> u8 {
        return 42;
    }
    """

    program = parse(source)
    func = program.items[0]

    assert func.is_far == True
    assert len(func.attributes) == 1
    assert func.attributes[0].name == 'bank'

    print("✓ Far function test passed")


def test_never_type():
    """Test parsing never type (!)."""
    source = """
    fn main() -> ! {
        loop {
        }
    }
    """

    program = parse(source)
    func = program.items[0]

    assert isinstance(func.return_type, ast.NeverType)

    print("✓ Never type test passed")


def test_array_type():
    """Test parsing array types."""
    source = """
    static BUFFER: [u8; 256];
    """

    program = parse(source)
    static = program.items[0]

    assert isinstance(static.var_type, ast.ArrayType)
    assert isinstance(static.var_type.element_type, ast.BasicType)
    assert static.var_type.element_type.name == 'u8'
    assert isinstance(static.var_type.size, ast.IntegerLiteral)
    assert static.var_type.size.value == 256

    print("✓ Array type test passed")


def test_pointer_types():
    """Test parsing pointer types."""
    source = """
    static PTR: near<u8>;
    static FAR_PTR: far<u16>;
    """

    program = parse(source)

    static1 = program.items[0]
    assert isinstance(static1.var_type, ast.PointerType)
    assert static1.var_type.is_far == False

    static2 = program.items[1]
    assert isinstance(static2.var_type, ast.PointerType)
    assert static2.var_type.is_far == True

    print("✓ Pointer types test passed")


def test_complete_program():
    """Test parsing a complete program."""
    source = """
    #[hw(0x2100)]
    static mut INIDISP: u8;

    #[zeropage(0x20)]
    static mut COUNTER: u16 = 0;

    #[mode(m8, x8)]
    #[preserves(X, Y)]
    fn increment() -> u16 {
        let value @ A = COUNTER;
        value = value + 1;
        COUNTER = value;
        return value;
    }

    #[entry]
    fn main() -> ! {
        loop {
            increment();
        }
    }
    """

    program = parse(source)

    assert isinstance(program, ast.Program)
    assert len(program.items) == 4  # 2 statics, 2 functions

    # Check static declarations
    assert isinstance(program.items[0], ast.StaticDecl)
    assert isinstance(program.items[1], ast.StaticDecl)

    # Check functions
    assert isinstance(program.items[2], ast.FunctionDecl)
    assert isinstance(program.items[3], ast.FunctionDecl)

    print("✓ Complete program test passed")


if __name__ == '__main__':
    print("Running parser tests...\n")

    test_simple_function()
    test_function_with_attributes()
    test_register_aliasing()
    test_static_declaration()
    test_struct_declaration()
    test_enum_declaration()
    test_if_statement()
    test_loop_statements()
    test_binary_operations()
    test_function_call()
    test_array_and_field_access()
    test_type_cast()
    test_assignment()
    test_compound_assignment()
    test_far_function()
    test_never_type()
    test_array_type()
    test_pointer_types()
    test_complete_program()

    print("\n✅ All parser tests passed!")
