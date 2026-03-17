# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for the R65 parser.
"""


from r65.compiler.frontend import parse, ParseError, ast


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
        #[preserves(X, Y)]
    fn process() {
        A = 42;
    }
    """

    program = parse(source)
    func = program.items[0]

    # Since #[mode] attribute was removed (mode is now inferred from parameter types),
    # only the preserves attribute should be present
    assert len(func.attributes) == 1
    assert func.attributes[0].name == 'preserves'

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


def test_nested_struct_literal():
    """Test parsing nested struct literals in field initializers."""
    source = """
    struct Point { x: u8, y: u8 }
    struct Rect { top_left: Point, bottom_right: Point }

    #[ram]
    static mut R: Rect = Rect {
        top_left: Point { x: 0, y: 0 },
        bottom_right: Point { x: 10, y: 10 }
    };
    """

    program = parse(source)

    # Check the static declaration
    static = program.items[2]
    assert isinstance(static, ast.StaticDecl)
    assert static.name == 'R'

    # Check outer struct literal
    init = static.initializer
    assert isinstance(init, ast.StructLiteralExpr)
    assert init.struct_name == 'Rect'
    assert len(init.fields) == 2

    # Check nested struct literal in top_left field
    top_left = init.fields[0]
    assert top_left.name == 'top_left'
    assert isinstance(top_left.value, ast.StructLiteralExpr)
    assert top_left.value.struct_name == 'Point'

    # Check nested struct literal in bottom_right field
    bottom_right = init.fields[1]
    assert bottom_right.name == 'bottom_right'
    assert isinstance(bottom_right.value, ast.StructLiteralExpr)
    assert bottom_right.value.struct_name == 'Point'

    print("✓ Nested struct literal test passed")


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


def test_for_loop():
    """Test parsing for loop statements."""
    source = """
    fn test() {
        for i in 0..10 {
            x = x + i;
        }
    }
    """

    program = parse(source)
    func = program.items[0]

    for_stmt = func.body.statements[0]
    assert isinstance(for_stmt, ast.ForStmt)
    assert for_stmt.variable == 'i'
    assert isinstance(for_stmt.start, ast.IntegerLiteral)
    assert for_stmt.start.value == 0
    assert isinstance(for_stmt.end, ast.IntegerLiteral)
    assert for_stmt.end.value == 10
    assert isinstance(for_stmt.body, ast.Block)

    print("✓ For loop test passed")


def test_labeled_loops():
    """Test parsing labeled loops and break/continue."""
    source = """
    fn test() {
        'outer: loop {
            'inner: while A < 10 {
                if A == 5 {
                    break 'outer;
                }
                continue 'inner;
            }
            break;
        }
    }
    """

    program = parse(source)
    func = program.items[0]

    # Outer loop
    loop_stmt = func.body.statements[0]
    assert isinstance(loop_stmt, ast.LoopStmt)
    assert loop_stmt.label == 'outer'

    # Inner while
    while_stmt = loop_stmt.body.statements[0]
    assert isinstance(while_stmt, ast.WhileStmt)
    assert while_stmt.label == 'inner'

    # Break 'outer
    if_stmt = while_stmt.body.statements[0]
    break_outer = if_stmt.then_block.statements[0]
    assert isinstance(break_outer, ast.BreakStmt)
    assert break_outer.label == 'outer'

    # Continue 'inner
    continue_inner = while_stmt.body.statements[1]
    assert isinstance(continue_inner, ast.ContinueStmt)
    assert continue_inner.label == 'inner'

    # Plain break (no label)
    plain_break = loop_stmt.body.statements[1]
    assert isinstance(plain_break, ast.BreakStmt)
    assert plain_break.label is None

    print("✓ Labeled loops test passed")


def test_labeled_for_loop():
    """Test parsing labeled for loop."""
    source = """
    fn test() {
        'rows: for y in 0..8 {
            'cols: for x in 0..8 {
                if x == y {
                    continue 'rows;
                }
            }
        }
    }
    """

    program = parse(source)
    func = program.items[0]

    outer_for = func.body.statements[0]
    assert isinstance(outer_for, ast.ForStmt)
    assert outer_for.label == 'rows'
    assert outer_for.variable == 'y'

    inner_for = outer_for.body.statements[0]
    assert isinstance(inner_for, ast.ForStmt)
    assert inner_for.label == 'cols'
    assert inner_for.variable == 'x'

    print("✓ Labeled for loop test passed")


def test_inclusive_for_loop():
    """Test parsing inclusive for loop with ..= syntax."""
    source = """
    fn test() {
        for i in 0..=10 {
            A = i;
        }
    }
    """

    program = parse(source)
    func = program.items[0]

    for_stmt = func.body.statements[0]
    assert isinstance(for_stmt, ast.ForStmt)
    assert for_stmt.variable == 'i'
    assert for_stmt.start.value == 0
    assert for_stmt.end.value == 10
    assert for_stmt.inclusive == True

    # Exclusive range should have inclusive=False
    source2 = """
    fn test() {
        for i in 0..10 {
            A = i;
        }
    }
    """
    program2 = parse(source2)
    func2 = program2.items[0]
    for_stmt2 = func2.body.statements[0]
    assert for_stmt2.inclusive == False

    print("✓ Inclusive for loop test passed")


def test_break_with_value():
    """Test parsing break statement with a value expression."""
    source = """
    fn test() {
        loop {
            break 42;
        }
    }
    """

    program = parse(source)
    func = program.items[0]

    loop_stmt = func.body.statements[0]
    assert isinstance(loop_stmt, ast.LoopStmt)
    break_stmt = loop_stmt.body.statements[0]
    assert isinstance(break_stmt, ast.BreakStmt)
    assert isinstance(break_stmt.value, ast.IntegerLiteral)
    assert break_stmt.value.value == 42
    assert break_stmt.label is None

    # Break without value should have value=None
    source2 = """
    fn test() {
        loop {
            break;
        }
    }
    """
    program2 = parse(source2)
    func2 = program2.items[0]
    loop2 = func2.body.statements[0]
    break2 = loop2.body.statements[0]
    assert isinstance(break2, ast.BreakStmt)
    assert break2.value is None

    print("✓ Break with value test passed")


def test_loop_expression():
    """Test parsing loop expression in initializer context."""
    source = """
    fn test() {
        let x: u8 = loop {
            break 42;
        };
    }
    """

    program = parse(source)
    func = program.items[0]

    let_stmt = func.body.statements[0]
    assert isinstance(let_stmt, ast.LetStmt)
    assert let_stmt.name == 'x'
    assert isinstance(let_stmt.initializer, ast.LoopExpression)
    assert isinstance(let_stmt.initializer.body, ast.Block)

    # Body should contain a break with value
    break_stmt = let_stmt.initializer.body.statements[0]
    assert isinstance(break_stmt, ast.BreakStmt)
    assert break_stmt.value.value == 42

    print("✓ Loop expression test passed")


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

    # Test bitwise compound assignments
    stmt5 = func.body.statements[4]
    assert isinstance(stmt5.expr, ast.CompoundAssignment)
    assert stmt5.expr.operator == '&'

    stmt6 = func.body.statements[5]
    assert isinstance(stmt6.expr, ast.CompoundAssignment)
    assert stmt6.expr.operator == '|'

    stmt7 = func.body.statements[6]
    assert isinstance(stmt7.expr, ast.CompoundAssignment)
    assert stmt7.expr.operator == '^'

    # Test shift compound assignments
    stmt8 = func.body.statements[7]
    assert isinstance(stmt8.expr, ast.CompoundAssignment)
    assert stmt8.expr.operator == '<<'

    stmt9 = func.body.statements[8]
    assert isinstance(stmt9.expr, ast.CompoundAssignment)
    assert stmt9.expr.operator == '>>'

    # Test register compound assignment
    stmt10 = func.body.statements[9]
    assert isinstance(stmt10.expr, ast.CompoundAssignment)
    assert isinstance(stmt10.expr.target, ast.Register)
    assert stmt10.expr.target.name == 'A'

    # Test array compound assignment
    stmt11 = func.body.statements[10]
    assert isinstance(stmt11.expr, ast.CompoundAssignment)
    assert isinstance(stmt11.expr.target, ast.ArrayIndex)

    # Test field compound assignment
    stmt12 = func.body.statements[11]
    assert isinstance(stmt12.expr, ast.CompoundAssignment)
    assert isinstance(stmt12.expr.target, ast.FieldAccess)

    print("✓ Compound assignment test passed")


def test_increment_decrement():
    """Test parsing increment/decrement operators."""
    source = """
    fn test() {
        x++;
        y--;
        A++;
        X--;
        array[i]++;
        array[j]--;
        player.health++;
        player.score--;
    }
    """

    program = parse(source)
    func = program.items[0]

    # Test increment on variable
    stmt1 = func.body.statements[0]
    assert isinstance(stmt1.expr, ast.CompoundAssignment)
    assert stmt1.expr.operator == '+'
    assert isinstance(stmt1.expr.target, ast.Identifier)
    assert stmt1.expr.target.name == 'x'
    assert isinstance(stmt1.expr.value, ast.IntegerLiteral)
    assert stmt1.expr.value.value == 1

    # Test decrement on variable
    stmt2 = func.body.statements[1]
    assert isinstance(stmt2.expr, ast.CompoundAssignment)
    assert stmt2.expr.operator == '-'
    assert isinstance(stmt2.expr.target, ast.Identifier)
    assert stmt2.expr.target.name == 'y'
    assert isinstance(stmt2.expr.value, ast.IntegerLiteral)
    assert stmt2.expr.value.value == 1

    # Test increment on register
    stmt3 = func.body.statements[2]
    assert isinstance(stmt3.expr, ast.CompoundAssignment)
    assert isinstance(stmt3.expr.target, ast.Register)
    assert stmt3.expr.target.name == 'A'

    # Test decrement on register
    stmt4 = func.body.statements[3]
    assert isinstance(stmt4.expr, ast.CompoundAssignment)
    assert isinstance(stmt4.expr.target, ast.Register)
    assert stmt4.expr.target.name == 'X'

    # Test increment on array element
    stmt5 = func.body.statements[4]
    assert isinstance(stmt5.expr, ast.CompoundAssignment)
    assert isinstance(stmt5.expr.target, ast.ArrayIndex)

    # Test decrement on array element
    stmt6 = func.body.statements[5]
    assert isinstance(stmt6.expr, ast.CompoundAssignment)
    assert isinstance(stmt6.expr.target, ast.ArrayIndex)

    # Test increment on struct field
    stmt7 = func.body.statements[6]
    assert isinstance(stmt7.expr, ast.CompoundAssignment)
    assert isinstance(stmt7.expr.target, ast.FieldAccess)

    # Test decrement on struct field
    stmt8 = func.body.statements[7]
    assert isinstance(stmt8.expr, ast.CompoundAssignment)
    assert isinstance(stmt8.expr.target, ast.FieldAccess)

    print("✓ Increment/decrement test passed")


def test_far_function():
    """Test parsing far functions with bank directive."""
    source = """
    #[bank(1)]
    far fn cross_bank() -> u8 {
        return 42;
    }
    """

    program = parse(source)

    # #[bank(1)] is now a directive, not a function attribute
    assert len(program.items) == 2

    # First item is the BankDirective
    bank_dir = program.items[0]
    assert isinstance(bank_dir, ast.BankDirective)
    assert bank_dir.bank_number == 1

    # Second item is the far function
    func = program.items[1]
    assert isinstance(func, ast.FunctionDecl)
    assert func.is_far == True
    assert func.name == 'cross_bank'
    # Function no longer has bank attribute (bank comes from directive context)
    assert len(func.attributes) == 0

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
    """Test parsing pointer types with new syntax.

    Two uses of 'far' in static declarations:
    1. 'far static' - static itself is far (for auto-bank mode)
    2. 'far *T' - the pointer type is far (24-bit address)
    """
    source = """
    static *PTR: u8;
    static FAR_PTR: far *u16;
    far static *FAR_STATIC: u8;
    """

    program = parse(source)

    # static *PTR: u8 - near static, near pointer
    static1 = program.items[0]
    assert isinstance(static1.var_type, ast.PointerType)
    assert static1.var_type.is_far == False  # Pointer type is near
    assert static1.name == "PTR"
    assert static1.is_far == False  # Static itself is near

    # static FAR_PTR: far *u16 - near static, far pointer type
    static2 = program.items[1]
    assert isinstance(static2.var_type, ast.PointerType)
    assert static2.var_type.is_far == True  # Pointer type is far
    assert static2.name == "FAR_PTR"
    assert static2.is_far == False  # Static itself is near

    # far static *FAR_STATIC: u8 - far static (for auto-bank)
    static3 = program.items[2]
    assert isinstance(static3.var_type, ast.PointerType)
    assert static3.var_type.is_far == False  # Pointer type is near
    assert static3.name == "FAR_STATIC"
    assert static3.is_far == True  # Static itself is far (for auto-bank)

    print("✓ Pointer types test passed")


def test_integer_suffixes():
    """Test parsing integer literals with type suffixes."""
    source = """
    fn main() {
        let x = 255u8;
        let y = 1000u16;
        let z = 42;
    }
    """

    program = parse(source)
    func = program.items[0]
    stmts = func.body.statements

    # 255u8 - has suffix
    let_x = stmts[0]
    assert isinstance(let_x, ast.LetStmt)
    assert isinstance(let_x.initializer, ast.IntegerLiteral)
    assert let_x.initializer.value == 255
    assert let_x.initializer.suffix == 'u8'

    # 1000u16 - has suffix
    let_y = stmts[1]
    assert isinstance(let_y.initializer, ast.IntegerLiteral)
    assert let_y.initializer.value == 1000
    assert let_y.initializer.suffix == 'u16'

    # 42 - no suffix
    let_z = stmts[2]
    assert isinstance(let_z.initializer, ast.IntegerLiteral)
    assert let_z.initializer.value == 42
    assert let_z.initializer.suffix is None

    print("✓ Integer suffixes test passed")


if __name__ == '__main__':
    print("Running parser tests...\n")

    test_simple_function()
    test_function_with_attributes()
    test_register_aliasing()
    test_static_declaration()
    test_struct_declaration()
    test_nested_struct_literal()
    test_enum_declaration()
    test_if_statement()
    test_loop_statements()
    test_for_loop()
    test_labeled_loops()
    test_labeled_for_loop()
    test_inclusive_for_loop()
    test_break_with_value()
    test_loop_expression()
    test_binary_operations()
    test_function_call()
    test_array_and_field_access()
    test_type_cast()
    test_assignment()
    test_compound_assignment()
    test_increment_decrement()
    test_far_function()
    test_never_type()
    test_array_type()
    test_pointer_types()
    test_integer_suffixes()

    print("\n✅ All parser tests passed!")
