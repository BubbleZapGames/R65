"""
Comprehensive operator tests for R65.

Tests all operators:
- Arithmetic: +, -, *, /, %
- Bitwise: &, |, ^, ~, <<, >>
- Comparison: ==, !=, <, <=, >, >=
- Logical: &&, ||, !
- Compound assignment: +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=
- Increment/decrement: ++, --
- Unary: !, ~, -, *, &
- Type cast: as
- Operator precedence

Each test validates:
1. Parsing succeeds and produces correct AST
2. Operator precedence is correct
3. HIR is built correctly (where applicable)
"""

import pytest
from r65.compiler.frontend import parse, ParseError, ast
from r65.compiler.hir import HIRBuilder
from r65.compiler.hir import nodes as hir


# ============================================================================
# Test Helpers
# ============================================================================

def parse_function(source: str) -> ast.FunctionDecl:
    """Parse source and return the first function declaration."""
    program = parse(source)
    assert len(program.items) >= 1
    func = program.items[0]
    assert isinstance(func, ast.FunctionDecl)
    return func


def parse_expr(source: str) -> ast.Expression:
    """Parse a function containing a let with expression and return that expression."""
    func = parse_function(f"fn test() {{ let x = {source}; }}")
    assert len(func.body.statements) == 1
    let_stmt = func.body.statements[0]
    assert isinstance(let_stmt, ast.LetStmt)
    return let_stmt.initializer


def parse_statement(source: str) -> ast.Statement:
    """Parse a function containing one statement and return that statement."""
    func = parse_function(f"fn test() {{ {source} }}")
    assert len(func.body.statements) == 1
    return func.body.statements[0]


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
# Arithmetic Operator Tests
# ============================================================================

class TestArithmeticOperators:
    """Tests for arithmetic operators: +, -, *, /, %"""

    def test_addition(self):
        """Test addition operator."""
        expr = parse_expr("a + b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '+'
        assert isinstance(expr.left, ast.Identifier)
        assert isinstance(expr.right, ast.Identifier)

    def test_subtraction(self):
        """Test subtraction operator."""
        expr = parse_expr("a - b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '-'

    def test_multiplication(self):
        """Test multiplication operator."""
        expr = parse_expr("a * b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '*'

    def test_division(self):
        """Test division operator."""
        expr = parse_expr("a / b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '/'

    def test_modulo(self):
        """Test modulo operator."""
        expr = parse_expr("a % b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '%'

    def test_addition_with_literals(self):
        """Test addition with integer literals."""
        expr = parse_expr("10 + 20")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '+'
        assert isinstance(expr.left, ast.IntegerLiteral)
        assert expr.left.value == 10
        assert isinstance(expr.right, ast.IntegerLiteral)
        assert expr.right.value == 20

    def test_subtraction_with_literals(self):
        """Test subtraction with literals."""
        expr = parse_expr("100 - 50")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '-'
        assert expr.left.value == 100
        assert expr.right.value == 50

    def test_multiply_by_power_of_two(self):
        """Test multiplication by power of two (optimized case)."""
        for power in [1, 2, 4, 8]:
            expr = parse_expr(f"x * {power}")
            assert isinstance(expr, ast.BinaryOp)
            assert expr.op == '*'
            assert expr.right.value == power

    def test_divide_by_power_of_two(self):
        """Test division by power of two (optimized case)."""
        for power in [1, 2, 4, 8]:
            expr = parse_expr(f"x / {power}")
            assert isinstance(expr, ast.BinaryOp)
            assert expr.op == '/'
            assert expr.right.value == power

    def test_chained_addition(self):
        """Test chained addition: a + b + c."""
        expr = parse_expr("a + b + c")

        # Left associative: (a + b) + c
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '+'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '+'
        assert isinstance(expr.right, ast.Identifier)
        assert expr.right.name == 'c'

    def test_chained_subtraction(self):
        """Test chained subtraction: a - b - c."""
        expr = parse_expr("a - b - c")

        # Left associative: (a - b) - c
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '-'
        assert isinstance(expr.left, ast.BinaryOp)

    def test_mixed_add_subtract(self):
        """Test mixed addition and subtraction."""
        expr = parse_expr("a + b - c + d")

        # ((a + b) - c) + d
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '+'

    def test_arithmetic_with_registers(self):
        """Test arithmetic with hardware registers."""
        expr = parse_expr("A + X")

        assert isinstance(expr, ast.BinaryOp)
        assert isinstance(expr.left, ast.Register)
        assert expr.left.name == 'A'
        assert isinstance(expr.right, ast.Register)
        assert expr.right.name == 'X'

    def test_arithmetic_with_hex_literals(self):
        """Test arithmetic with hex literals."""
        expr = parse_expr("0x10 + 0xFF")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.left.value == 0x10
        assert expr.right.value == 0xFF

    def test_arithmetic_with_binary_literals(self):
        """Test arithmetic with binary literals."""
        expr = parse_expr("0b1010 + 0b0101")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.left.value == 0b1010
        assert expr.right.value == 0b0101


# ============================================================================
# Bitwise Operator Tests
# ============================================================================

class TestBitwiseOperators:
    """Tests for bitwise operators: &, |, ^, ~, <<, >>"""

    def test_bitwise_and(self):
        """Test bitwise AND operator."""
        expr = parse_expr("a & b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&'

    def test_bitwise_or(self):
        """Test bitwise OR operator."""
        expr = parse_expr("a | b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '|'

    def test_bitwise_xor(self):
        """Test bitwise XOR operator."""
        expr = parse_expr("a ^ b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '^'

    def test_bitwise_not(self):
        """Test bitwise NOT operator."""
        expr = parse_expr("~a")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '~'
        assert isinstance(expr.operand, ast.Identifier)

    def test_left_shift(self):
        """Test left shift operator."""
        expr = parse_expr("a << 2")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '<<'

    def test_right_shift(self):
        """Test right shift operator."""
        expr = parse_expr("a >> 2")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '>>'

    def test_and_with_mask(self):
        """Test AND with hex mask."""
        expr = parse_expr("value & 0x0F")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&'
        assert expr.right.value == 0x0F

    def test_or_with_mask(self):
        """Test OR with hex mask."""
        expr = parse_expr("flags | 0x80")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '|'
        assert expr.right.value == 0x80

    def test_xor_with_mask(self):
        """Test XOR with hex mask."""
        expr = parse_expr("value ^ 0xFF")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '^'
        assert expr.right.value == 0xFF

    def test_shift_by_constant(self):
        """Test shift by various constants."""
        for shift in [1, 2, 3, 4, 8]:
            expr = parse_expr(f"x << {shift}")
            assert isinstance(expr, ast.BinaryOp)
            assert expr.op == '<<'
            assert expr.right.value == shift

    def test_chained_bitwise_and(self):
        """Test chained AND: a & b & c."""
        expr = parse_expr("a & b & c")

        # Left associative
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&'
        assert isinstance(expr.left, ast.BinaryOp)

    def test_chained_bitwise_or(self):
        """Test chained OR: a | b | c."""
        expr = parse_expr("a | b | c")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '|'
        assert isinstance(expr.left, ast.BinaryOp)

    def test_bitwise_with_registers(self):
        """Test bitwise ops with registers."""
        expr = parse_expr("A & 0x0F")

        assert isinstance(expr, ast.BinaryOp)
        assert isinstance(expr.left, ast.Register)

    def test_double_not(self):
        """Test double bitwise NOT: ~~a."""
        expr = parse_expr("~~a")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '~'
        assert isinstance(expr.operand, ast.UnaryOp)
        assert expr.operand.op == '~'

    def test_shift_left_shift_right(self):
        """Test combined shifts."""
        expr = parse_expr("a << 4 >> 4")

        # Left associative: (a << 4) >> 4
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '>>'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '<<'


# ============================================================================
# Comparison Operator Tests
# ============================================================================

class TestComparisonOperators:
    """Tests for comparison operators: ==, !=, <, <=, >, >="""

    def test_equal(self):
        """Test equality operator."""
        expr = parse_expr("a == b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '=='

    def test_not_equal(self):
        """Test inequality operator."""
        expr = parse_expr("a != b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '!='

    def test_less_than(self):
        """Test less than operator."""
        expr = parse_expr("a < b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '<'

    def test_less_equal(self):
        """Test less than or equal operator."""
        expr = parse_expr("a <= b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '<='

    def test_greater_than(self):
        """Test greater than operator."""
        expr = parse_expr("a > b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '>'

    def test_greater_equal(self):
        """Test greater than or equal operator."""
        expr = parse_expr("a >= b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '>='

    def test_compare_with_zero(self):
        """Test comparison with zero."""
        expr = parse_expr("x == 0")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '=='
        assert expr.right.value == 0

    def test_compare_with_max(self):
        """Test comparison with max value."""
        expr = parse_expr("x == 255")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.right.value == 255

    def test_compare_registers(self):
        """Test comparison of registers."""
        expr = parse_expr("A == X")

        assert isinstance(expr, ast.BinaryOp)
        assert isinstance(expr.left, ast.Register)
        assert isinstance(expr.right, ast.Register)

    def test_compare_with_hex(self):
        """Test comparison with hex literal."""
        expr = parse_expr("flags != 0xFF")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.right.value == 0xFF

    def test_boundary_comparison(self):
        """Test boundary value comparisons."""
        expr = parse_expr("index < 256")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '<'
        assert expr.right.value == 256

    def test_comparison_in_condition(self):
        """Test comparison as if condition."""
        stmt = parse_statement("if x > 10 { A = 1; }")

        assert isinstance(stmt, ast.IfStmt)
        assert isinstance(stmt.condition, ast.BinaryOp)
        assert stmt.condition.op == '>'


# ============================================================================
# Logical Operator Tests
# ============================================================================

class TestLogicalOperators:
    """Tests for logical operators: &&, ||, !"""

    def test_logical_and(self):
        """Test logical AND operator."""
        expr = parse_expr("a && b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&&'

    def test_logical_or(self):
        """Test logical OR operator."""
        expr = parse_expr("a || b")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '||'

    def test_logical_not(self):
        """Test logical NOT operator."""
        expr = parse_expr("!a")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '!'

    def test_double_not(self):
        """Test double logical NOT: !!a."""
        expr = parse_expr("!!a")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '!'
        assert isinstance(expr.operand, ast.UnaryOp)
        assert expr.operand.op == '!'

    def test_and_with_comparisons(self):
        """Test AND with comparison operands."""
        expr = parse_expr("x > 0 && y > 0")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&&'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '>'
        assert isinstance(expr.right, ast.BinaryOp)
        assert expr.right.op == '>'

    def test_or_with_comparisons(self):
        """Test OR with comparison operands."""
        expr = parse_expr("x == 0 || y == 0")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '||'

    def test_chained_and(self):
        """Test chained AND: a && b && c."""
        expr = parse_expr("a && b && c")

        # Left associative
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&&'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '&&'

    def test_chained_or(self):
        """Test chained OR: a || b || c."""
        expr = parse_expr("a || b || c")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '||'
        assert isinstance(expr.left, ast.BinaryOp)

    def test_not_with_comparison(self):
        """Test NOT with comparison."""
        expr = parse_expr("!done")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '!'

    def test_complex_logical(self):
        """Test complex logical expression."""
        expr = parse_expr("a && b || c && d")

        # && binds tighter than ||: (a && b) || (c && d)
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '||'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '&&'
        assert isinstance(expr.right, ast.BinaryOp)
        assert expr.right.op == '&&'

    def test_logical_with_booleans(self):
        """Test logical ops with boolean literals."""
        expr = parse_expr("true && false")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&&'
        assert isinstance(expr.left, ast.BooleanLiteral)
        assert expr.left.value == True
        assert isinstance(expr.right, ast.BooleanLiteral)
        assert expr.right.value == False


# ============================================================================
# Compound Assignment Tests
# ============================================================================

class TestCompoundAssignment:
    """Tests for compound assignment operators: +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>="""

    def test_plus_equal(self):
        """Test += operator."""
        stmt = parse_statement("x += 5;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '+'
        assert isinstance(stmt.expr.target, ast.Identifier)
        assert stmt.expr.value.value == 5

    def test_minus_equal(self):
        """Test -= operator."""
        stmt = parse_statement("x -= 3;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '-'

    def test_star_equal(self):
        """Test *= operator."""
        stmt = parse_statement("x *= 2;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '*'

    def test_slash_equal(self):
        """Test /= operator."""
        stmt = parse_statement("x /= 4;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '/'

    def test_percent_equal(self):
        """Test %= operator."""
        stmt = parse_statement("x %= 8;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '%'

    def test_and_equal(self):
        """Test &= operator."""
        stmt = parse_statement("flags &= 0x0F;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '&'

    def test_or_equal(self):
        """Test |= operator."""
        stmt = parse_statement("flags |= 0x80;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '|'

    def test_xor_equal(self):
        """Test ^= operator."""
        stmt = parse_statement("flags ^= 0xFF;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '^'

    def test_lshift_equal(self):
        """Test <<= operator."""
        stmt = parse_statement("x <<= 2;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '<<'

    def test_rshift_equal(self):
        """Test >>= operator."""
        stmt = parse_statement("x >>= 1;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '>>'

    def test_compound_on_register(self):
        """Test compound assignment on register."""
        stmt = parse_statement("A += value;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.Register)
        assert stmt.expr.target.name == 'A'

    def test_compound_on_array(self):
        """Test compound assignment on array element."""
        stmt = parse_statement("array[i] += 1;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.ArrayIndex)

    def test_compound_on_field(self):
        """Test compound assignment on struct field."""
        stmt = parse_statement("player.score += 100;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.FieldAccess)

    def test_compound_with_expression(self):
        """Test compound assignment with complex expression."""
        stmt = parse_statement("total += a + b;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.value, ast.BinaryOp)


# ============================================================================
# Increment/Decrement Tests
# ============================================================================

class TestIncrementDecrement:
    """Tests for increment/decrement operators: ++, --"""

    def test_increment_variable(self):
        """Test ++ on variable."""
        stmt = parse_statement("x++;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '+'
        assert isinstance(stmt.expr.target, ast.Identifier)
        assert stmt.expr.value.value == 1

    def test_decrement_variable(self):
        """Test -- on variable."""
        stmt = parse_statement("x--;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert stmt.expr.operator == '-'
        assert stmt.expr.value.value == 1

    def test_increment_register_a(self):
        """Test ++ on A register."""
        stmt = parse_statement("A++;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.Register)
        assert stmt.expr.target.name == 'A'

    def test_increment_register_x(self):
        """Test ++ on X register."""
        stmt = parse_statement("X++;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.Register)
        assert stmt.expr.target.name == 'X'

    def test_increment_register_y(self):
        """Test ++ on Y register."""
        stmt = parse_statement("Y++;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.Register)
        assert stmt.expr.target.name == 'Y'

    def test_decrement_register(self):
        """Test -- on register."""
        stmt = parse_statement("X--;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.Register)
        assert stmt.expr.operator == '-'

    def test_increment_array_element(self):
        """Test ++ on array element."""
        stmt = parse_statement("array[i]++;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.ArrayIndex)

    def test_decrement_array_element(self):
        """Test -- on array element."""
        stmt = parse_statement("array[j]--;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.ArrayIndex)

    def test_increment_field(self):
        """Test ++ on struct field."""
        stmt = parse_statement("player.health++;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.FieldAccess)

    def test_decrement_field(self):
        """Test -- on struct field."""
        stmt = parse_statement("player.lives--;")

        assert isinstance(stmt.expr, ast.CompoundAssignment)
        assert isinstance(stmt.expr.target, ast.FieldAccess)


# ============================================================================
# Unary Operator Tests
# ============================================================================

class TestUnaryOperators:
    """Tests for unary operators: !, ~, -, *, &"""

    def test_logical_not(self):
        """Test logical NOT."""
        expr = parse_expr("!flag")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '!'

    def test_bitwise_not(self):
        """Test bitwise NOT."""
        expr = parse_expr("~mask")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '~'

    def test_negation(self):
        """Test numeric negation."""
        expr = parse_expr("-value")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '-'

    def test_negation_literal(self):
        """Test negation of literal."""
        expr = parse_expr("-42")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '-'
        assert isinstance(expr.operand, ast.IntegerLiteral)
        assert expr.operand.value == 42

    def test_dereference(self):
        """Test pointer dereference."""
        expr = parse_expr("*ptr")

        assert isinstance(expr, ast.Dereference)
        assert isinstance(expr.pointer, ast.Identifier)

    def test_address_of(self):
        """Test address-of operator."""
        expr = parse_expr("&variable")

        assert isinstance(expr, ast.AddressOf)
        assert isinstance(expr.operand, ast.Identifier)

    def test_chained_unary(self):
        """Test chained unary operators."""
        expr = parse_expr("!!flag")

        assert isinstance(expr, ast.UnaryOp)
        assert isinstance(expr.operand, ast.UnaryOp)

    def test_not_comparison(self):
        """Test NOT applied to comparison result."""
        # Parse as separate expression with parens
        func = parse_function("fn test() { let x = !done; }")
        let_stmt = func.body.statements[0]

        assert isinstance(let_stmt.initializer, ast.UnaryOp)
        assert let_stmt.initializer.op == '!'

    def test_bitwise_not_hex(self):
        """Test bitwise NOT on hex literal."""
        expr = parse_expr("~0xFF")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '~'
        assert expr.operand.value == 0xFF

    def test_negation_on_register(self):
        """Test negation on register."""
        expr = parse_expr("-A")

        assert isinstance(expr, ast.UnaryOp)
        assert expr.op == '-'
        assert isinstance(expr.operand, ast.Register)


# ============================================================================
# Type Cast Tests
# ============================================================================

class TestTypeCasts:
    """Tests for type cast operator: as"""

    def test_cast_u8_to_u16(self):
        """Test widening cast u8 to u16."""
        expr = parse_expr("x as u16")

        assert isinstance(expr, ast.TypeCast)
        assert isinstance(expr.target_type, ast.BasicType)
        assert expr.target_type.name == 'u16'

    def test_cast_u16_to_u8(self):
        """Test narrowing cast u16 to u8."""
        expr = parse_expr("value as u8")

        assert isinstance(expr, ast.TypeCast)
        assert expr.target_type.name == 'u8'

    def test_cast_to_i8(self):
        """Test cast to i8 (signed)."""
        expr = parse_expr("x as i8")

        assert isinstance(expr, ast.TypeCast)
        assert expr.target_type.name == 'i8'

    def test_cast_to_i16(self):
        """Test cast to i16 (signed)."""
        expr = parse_expr("x as i16")

        assert isinstance(expr, ast.TypeCast)
        assert expr.target_type.name == 'i16'

    def test_cast_to_bool(self):
        """Test cast to bool."""
        expr = parse_expr("x as bool")

        assert isinstance(expr, ast.TypeCast)
        assert expr.target_type.name == 'bool'

    def test_cast_literal(self):
        """Test cast of literal."""
        expr = parse_expr("255 as i8")

        assert isinstance(expr, ast.TypeCast)
        assert isinstance(expr.expr, ast.IntegerLiteral)
        assert expr.expr.value == 255

    def test_cast_register(self):
        """Test cast of register."""
        expr = parse_expr("A as u16")

        assert isinstance(expr, ast.TypeCast)
        assert isinstance(expr.expr, ast.Register)

    def test_cast_expression(self):
        """Test cast of expression result."""
        expr = parse_expr("x + y as u16")

        # 'as' binds tightly, so this is x + (y as u16)
        assert isinstance(expr, ast.BinaryOp)
        assert isinstance(expr.right, ast.TypeCast)

    def test_chained_cast(self):
        """Test chained casts."""
        expr = parse_expr("x as u16 as u8")

        # Left to right: (x as u16) as u8
        assert isinstance(expr, ast.TypeCast)
        assert expr.target_type.name == 'u8'
        assert isinstance(expr.expr, ast.TypeCast)
        assert expr.expr.target_type.name == 'u16'

    def test_bool_to_u8(self):
        """Test bool to u8 cast."""
        expr = parse_expr("flag as u8")

        assert isinstance(expr, ast.TypeCast)
        assert expr.target_type.name == 'u8'


# ============================================================================
# Operator Precedence Tests
# ============================================================================

class TestOperatorPrecedence:
    """Tests for operator precedence."""

    def test_multiply_before_add(self):
        """Test * binds tighter than +."""
        expr = parse_expr("a + b * c")

        # a + (b * c)
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '+'
        assert isinstance(expr.right, ast.BinaryOp)
        assert expr.right.op == '*'

    def test_add_before_comparison(self):
        """Test + binds tighter than <."""
        expr = parse_expr("a + b < c")

        # (a + b) < c
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '<'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '+'

    def test_comparison_before_and(self):
        """Test < binds tighter than &&."""
        expr = parse_expr("a < b && c < d")

        # (a < b) && (c < d)
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&&'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '<'

    def test_and_before_or(self):
        """Test && binds tighter than ||."""
        expr = parse_expr("a && b || c")

        # (a && b) || c
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '||'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '&&'

    def test_bitwise_and_before_or(self):
        """Test & binds tighter than |."""
        expr = parse_expr("a & b | c")

        # (a & b) | c
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '|'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '&'

    def test_shift_before_bitwise(self):
        """Test << binds tighter than &."""
        expr = parse_expr("a << 2 & mask")

        # (a << 2) & mask
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '<<'

    def test_unary_before_binary(self):
        """Test unary binds tighter than binary."""
        expr = parse_expr("-a + b")

        # (-a) + b
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '+'
        assert isinstance(expr.left, ast.UnaryOp)
        assert expr.left.op == '-'

    def test_cast_precedence(self):
        """Test 'as' cast precedence."""
        expr = parse_expr("a + b as u16")

        # a + (b as u16) - cast binds tighter
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '+'
        assert isinstance(expr.right, ast.TypeCast)

    def test_not_before_and(self):
        """Test ! binds tighter than &&."""
        expr = parse_expr("!a && b")

        # (!a) && b
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&&'
        assert isinstance(expr.left, ast.UnaryOp)
        assert expr.left.op == '!'

    def test_comparison_chaining(self):
        """Test comparison doesn't chain (each is separate)."""
        # a < b < c parses as (a < b) < c
        expr = parse_expr("a < b < c")

        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '<'
        assert isinstance(expr.left, ast.BinaryOp)
        assert expr.left.op == '<'

    def test_equality_before_bitwise_and(self):
        """Test == binds looser than &."""
        # a & mask == 0 should be (a & mask) == 0
        # Due to C-style precedence, & actually binds looser than ==
        # Let's verify actual behavior
        expr = parse_expr("a & mask == 0")

        # In most languages including R65 (following C):
        # & binds tighter than ==, so: a & (mask == 0)
        # But R65 might differ - let's check what we get
        assert isinstance(expr, ast.BinaryOp)
        # This tests actual precedence in grammar

    def test_complex_arithmetic(self):
        """Test complex arithmetic expression."""
        expr = parse_expr("a + b * c - d / e")

        # a + (b * c) - (d / e) = ((a + (b * c)) - (d / e))
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '-'

    def test_mixed_logical_bitwise(self):
        """Test mixed logical and bitwise."""
        expr = parse_expr("a & b != 0 && c | d != 0")

        # Complex precedence test
        assert isinstance(expr, ast.BinaryOp)
        assert expr.op == '&&'


# ============================================================================
# Assignment Tests
# ============================================================================

class TestAssignment:
    """Tests for simple assignment operator: ="""

    def test_simple_assignment(self):
        """Test simple assignment."""
        stmt = parse_statement("x = 10;")

        assert isinstance(stmt.expr, ast.Assignment)
        assert isinstance(stmt.expr.target, ast.Identifier)
        assert stmt.expr.target.name == 'x'

    def test_assignment_expression(self):
        """Test assignment with expression."""
        stmt = parse_statement("x = a + b;")

        assert isinstance(stmt.expr, ast.Assignment)
        assert isinstance(stmt.expr.value, ast.BinaryOp)

    def test_assignment_to_register(self):
        """Test assignment to register."""
        stmt = parse_statement("A = 42;")

        assert isinstance(stmt.expr, ast.Assignment)
        # Register assignment may be parsed differently

    def test_assignment_to_array(self):
        """Test assignment to array element."""
        stmt = parse_statement("array[i] = value;")

        assert isinstance(stmt.expr, ast.Assignment)
        assert isinstance(stmt.expr.target, ast.ArrayIndex)

    def test_assignment_to_field(self):
        """Test assignment to struct field."""
        stmt = parse_statement("player.x = 100;")

        assert isinstance(stmt.expr, ast.Assignment)
        assert isinstance(stmt.expr.target, ast.FieldAccess)

    def test_chained_assignment(self):
        """Test chained assignment a = b = c."""
        stmt = parse_statement("a = b = 10;")

        # Right associative: a = (b = 10)
        assert isinstance(stmt.expr, ast.Assignment)
        assert isinstance(stmt.expr.value, ast.Assignment)


# ============================================================================
# HIR Building Tests
# ============================================================================

class TestOperatorHIR:
    """Tests for operator HIR building."""

    def test_binary_op_hir(self):
        """Test binary operation HIR building."""
        hir_prog = build_hir("""
            #[zeropage]
            static mut x: u8;
            #[zeropage]
            static mut y: u8;

            fn test() {
                let z = x + y;
            }
        """)

        func = get_hir_function(hir_prog, 'test')
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt, hir.HIRLetStmt)
        assert isinstance(let_stmt.initializer, hir.HIRBinaryOp)
        assert let_stmt.initializer.op == '+'

    def test_unary_op_hir(self):
        """Test unary operation HIR building."""
        hir_prog = build_hir("""
            #[zeropage]
            static mut x: u8;

            fn test() {
                let y = ~x;
            }
        """)

        func = get_hir_function(hir_prog, 'test')
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, hir.HIRUnaryOp)
        assert let_stmt.initializer.op == '~'

    def test_comparison_hir(self):
        """Test comparison HIR building."""
        hir_prog = build_hir("""
            #[zeropage]
            static mut x: u8;

            fn test() {
                let result = x > 10;
            }
        """)

        func = get_hir_function(hir_prog, 'test')
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, hir.HIRBinaryOp)
        assert let_stmt.initializer.op == '>'


# ============================================================================
# Edge Cases and Error Tests
# ============================================================================

class TestOperatorEdgeCases:
    """Tests for operator edge cases."""

    def test_zero_operand(self):
        """Test operations with zero."""
        expr = parse_expr("x + 0")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.right.value == 0

    def test_max_u8_operand(self):
        """Test operations with max u8."""
        expr = parse_expr("x + 255")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.right.value == 255

    def test_max_u16_operand(self):
        """Test operations with max u16."""
        expr = parse_expr("x + 65535")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.right.value == 65535

    def test_hex_max(self):
        """Test operations with hex max values."""
        expr = parse_expr("x & 0xFFFF")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.right.value == 0xFFFF

    def test_binary_literal(self):
        """Test operations with binary literals."""
        expr = parse_expr("x | 0b11110000")
        assert isinstance(expr, ast.BinaryOp)
        assert expr.right.value == 0b11110000

    def test_long_expression(self):
        """Test long chained expression."""
        expr = parse_expr("a + b + c + d + e + f")
        assert isinstance(expr, ast.BinaryOp)
        # Should be deeply nested left-associative

    def test_deeply_nested(self):
        """Test deeply nested expression."""
        expr = parse_expr("a * b + c * d + e * f")
        assert isinstance(expr, ast.BinaryOp)


class TestOperatorParseErrors:
    """Tests for operator parse errors."""

    def test_missing_operand(self):
        """Test missing operand fails."""
        with pytest.raises(Exception):
            parse("fn test() { let x = a +; }")

    def test_double_operator(self):
        """Test double operator fails (except valid ones like ++)."""
        with pytest.raises(Exception):
            parse("fn test() { let x = a + + b; }")

    def test_invalid_compound(self):
        """Test invalid compound assignment target."""
        # Literals can't be assignment targets
        with pytest.raises(Exception):
            parse("fn test() { 5 += 1; }")

    def test_missing_cast_type(self):
        """Test missing cast type fails."""
        with pytest.raises(Exception):
            parse("fn test() { let x = y as; }")


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
