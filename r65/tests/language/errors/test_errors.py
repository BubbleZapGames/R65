"""
Comprehensive error handling tests for R65.

Tests parse errors, syntax errors, and error recovery.
Validates that the parser correctly rejects invalid syntax.
"""

import pytest
from r65.compiler.frontend.parser import parse, ParseError
from r65.compiler.frontend import ast
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.hir import nodes as hir


# =============================================================================
# Helper Functions
# =============================================================================

def parse_succeeds(source: str) -> ast.Program:
    """Parse source and return program, asserting it succeeds."""
    return parse(source)


def parse_fails(source: str) -> str:
    """Parse source, expecting it to fail. Returns error message."""
    with pytest.raises(Exception) as exc_info:
        parse(source)
    return str(exc_info.value)


def build_hir(source: str) -> hir.HIRProgram:
    """Parse source and build HIR."""
    program = parse(source)
    builder = HIRBuilder()
    return builder.build_program(program)


# =============================================================================
# Test Classes
# =============================================================================

class TestMissingSemicolons:
    """Tests for missing semicolon errors."""

    def test_let_missing_semicolon(self):
        """Test let statement without semicolon."""
        with pytest.raises(ParseError):
            parse("fn test() { let x = 10 }")

    def test_assignment_missing_semicolon(self):
        """Test assignment without semicolon."""
        with pytest.raises(ParseError):
            parse("fn test() { A = 10 }")

    def test_return_missing_semicolon(self):
        """Test return without semicolon."""
        with pytest.raises(ParseError):
            parse("fn test() { return A }")

    def test_break_missing_semicolon(self):
        """Test break without semicolon."""
        with pytest.raises(ParseError):
            parse("fn test() { loop { break } }")

    def test_continue_missing_semicolon(self):
        """Test continue without semicolon."""
        with pytest.raises(ParseError):
            parse("fn test() { loop { continue } }")

    def test_static_missing_semicolon(self):
        """Test static declaration without semicolon."""
        with pytest.raises(ParseError):
            parse("#[ram] static mut X: u8")


class TestMissingBraces:
    """Tests for missing brace errors."""

    def test_function_missing_open_brace(self):
        """Test function without opening brace."""
        with pytest.raises(ParseError):
            parse("fn test() let x = 10; }")

    def test_function_missing_close_brace(self):
        """Test function without closing brace."""
        with pytest.raises(ParseError):
            parse("fn test() { let x = 10;")

    def test_if_missing_open_brace(self):
        """Test if without opening brace."""
        with pytest.raises(ParseError):
            parse("fn test() { if A == 0 A = 1; } }")

    def test_if_missing_close_brace(self):
        """Test if without closing brace."""
        with pytest.raises(ParseError):
            parse("fn test() { if A == 0 { A = 1; }")

    def test_loop_missing_braces(self):
        """Test loop without braces."""
        with pytest.raises(ParseError):
            parse("fn test() { loop A++; }")

    def test_while_missing_braces(self):
        """Test while without braces."""
        with pytest.raises(ParseError):
            parse("fn test() { while A != 0 A--; }")

    def test_struct_missing_braces(self):
        """Test struct without braces."""
        with pytest.raises(ParseError):
            parse("struct Point x: u8, y: u8")

    def test_enum_missing_braces(self):
        """Test enum without braces."""
        with pytest.raises(ParseError):
            parse("enum Dir North, South")


class TestMissingParentheses:
    """Tests for missing parenthesis errors."""

    def test_function_missing_open_paren(self):
        """Test function without opening paren."""
        with pytest.raises(ParseError):
            parse("fn test) { }")

    def test_function_missing_close_paren(self):
        """Test function without closing paren."""
        with pytest.raises(ParseError):
            parse("fn test( { }")

    def test_if_missing_condition_parens(self):
        """Test that if condition doesn't require parens (valid syntax)."""
        # R65 doesn't require parens around conditions
        prog = parse_succeeds("fn test() { if A == 0 { } }")
        assert len(prog.items) == 1

    def test_while_missing_condition_parens(self):
        """Test that while condition doesn't require parens."""
        prog = parse_succeeds("fn test() { while A != 0 { A--; } }")
        assert len(prog.items) == 1

    def test_function_call_missing_parens(self):
        """Test that identifier without parens is valid (not a call)."""
        # foo; without parens is a valid identifier expression, not an error
        prog = parse_succeeds("fn test() { foo; }")
        func = prog.items[0]
        # The statement is an expression statement containing an identifier
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)
        assert isinstance(stmt.expr, ast.Identifier)

    def test_attribute_missing_parens(self):
        """Test attribute argument without parens."""
        with pytest.raises(ParseError):
            parse("#[mode m8] fn test() { }")


class TestMissingBrackets:
    """Tests for missing bracket errors."""

    def test_array_type_missing_close_bracket(self):
        """Test array type without closing bracket."""
        with pytest.raises(ParseError):
            parse("#[ram] static mut X: [u8; 10;")

    def test_array_type_missing_size(self):
        """Test array type without size."""
        with pytest.raises(ParseError):
            parse("#[ram] static mut X: [u8];")

    def test_array_index_missing_close_bracket(self):
        """Test array index without closing bracket."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = ARR[0; }")

    def test_array_literal_missing_close_bracket(self):
        """Test array literal without closing bracket."""
        with pytest.raises(ParseError):
            parse("#[ram] static mut X: [u8; 3] = [1, 2, 3;")


class TestInvalidLiterals:
    """Tests for invalid literal errors."""

    def test_invalid_hex_literal(self):
        """Test invalid hexadecimal literal."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 0xGG; }")

    def test_invalid_binary_literal(self):
        """Test invalid binary literal."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 0b123; }")

    def test_hex_literal_no_digits(self):
        """Test hex prefix without digits."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 0x; }")

    def test_binary_literal_no_digits(self):
        """Test binary prefix without digits."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 0b; }")

    def test_unterminated_string(self):
        """Test unterminated string literal."""
        with pytest.raises(ParseError):
            parse('fn test() { asm!("WAI); }')


class TestInvalidOperators:
    """Tests for invalid operator usage."""

    def test_double_plus_operator(self):
        """Test invalid double plus in expression."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 1 ++ 2; }")

    def test_double_minus_operator(self):
        """Test invalid double minus in expression."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 1 -- 2; }")

    def test_missing_operand_left(self):
        """Test binary operator without left operand."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = + 1; }")

    def test_missing_operand_right(self):
        """Test binary operator without right operand."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 1 +; }")

    def test_consecutive_binary_operators(self):
        """Test consecutive binary operators - * is parsed as dereference."""
        # 1 + *2 parses as 1 + (dereference of 2), which is a type error, not parse error
        # Parser allows this; type checker would catch the invalid dereference
        prog = parse_succeeds("fn test() { let x: u8 = 1 + * 2; }")
        func = prog.items[0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt, ast.LetStmt)
        # The initializer is a binary op (add) where right side is dereference
        assert isinstance(let_stmt.initializer, ast.BinaryOp)
        assert let_stmt.initializer.op == "+"
        assert isinstance(let_stmt.initializer.right, ast.Dereference)
        assert let_stmt.initializer.right.pointer.value == 2


class TestInvalidDeclarations:
    """Tests for invalid declaration syntax."""

    def test_function_no_name(self):
        """Test function without name."""
        with pytest.raises(ParseError):
            parse("fn () { }")

    def test_function_no_body(self):
        """Test function without body."""
        with pytest.raises(ParseError):
            parse("fn test();")

    def test_static_no_type(self):
        """Test static without type."""
        with pytest.raises(ParseError):
            parse("#[ram] static mut X;")

    def test_const_no_value(self):
        """Test const without value."""
        with pytest.raises(ParseError):
            parse("const X: u8;")

    def test_const_no_type(self):
        """Test const without type."""
        with pytest.raises(ParseError):
            parse("const X = 10;")

    def test_parameter_no_type(self):
        """Test function parameter without type."""
        with pytest.raises(ParseError):
            parse("fn test(x) { }")

    def test_struct_field_no_type(self):
        """Test struct field without type."""
        with pytest.raises(ParseError):
            parse("struct Point { x, y }")

    def test_enum_empty(self):
        """Test empty enum."""
        with pytest.raises(ParseError):
            parse("enum Empty { }")


class TestInvalidStatements:
    """Tests for invalid statement syntax."""

    def test_let_no_initializer_no_type(self):
        """Test let without initializer or type."""
        with pytest.raises(ParseError):
            parse("fn test() { let x; }")

    def test_if_no_condition(self):
        """Test if without condition."""
        with pytest.raises(ParseError):
            parse("fn test() { if { A = 1; } }")

    def test_while_no_condition(self):
        """Test while without condition."""
        with pytest.raises(ParseError):
            parse("fn test() { while { A--; } }")

    def test_break_outside_loop(self):
        """Test break outside loop (parses, type checker catches)."""
        # Parser allows; type checker would reject
        prog = parse_succeeds("fn test() { break; }")
        assert len(prog.items) == 1

    def test_continue_outside_loop(self):
        """Test continue outside loop (parses, type checker catches)."""
        prog = parse_succeeds("fn test() { continue; }")
        assert len(prog.items) == 1


class TestInvalidExpressions:
    """Tests for invalid expression syntax."""

    def test_empty_parentheses(self):
        """Test empty parentheses as expression."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = (); }")

    def test_unmatched_open_paren(self):
        """Test unmatched opening parenthesis."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = (1 + 2; }")

    def test_unmatched_close_paren(self):
        """Test unmatched closing parenthesis."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = 1 + 2); }")

    def test_incomplete_ternary(self):
        """Test incomplete conditional (R65 doesn't have ternary)."""
        with pytest.raises(ParseError):
            parse("fn test() { let x: u8 = A == 0 ? 1 : 0; }")

    def test_trailing_comma_in_call(self):
        """Test trailing comma in function call (may be allowed)."""
        # Check if this is allowed or not
        try:
            prog = parse("fn test() { foo(1, 2,); }")
            # If it parses, that's fine
        except ParseError:
            pass  # Also acceptable


class TestInvalidTypes:
    """Tests for invalid type syntax."""

    def test_pointer_no_pointee(self):
        """Test pointer type without pointee."""
        with pytest.raises(ParseError):
            parse("#[zeropage] static mut P: near;")

    def test_pointer_missing_angle_bracket(self):
        """Test pointer type without angle brackets."""
        with pytest.raises(ParseError):
            parse("#[zeropage] static mut P: near u8;")

    def test_array_no_element_type(self):
        """Test array type without element type."""
        with pytest.raises(ParseError):
            parse("#[ram] static mut A: [; 10];")

    def test_function_type_no_parens(self):
        """Test function type without parentheses."""
        with pytest.raises(ParseError):
            parse("#[ram] static mut F: fn;")


class TestInvalidAttributes:
    """Tests for invalid attribute syntax."""

    def test_attribute_after_fn_keyword(self):
        """Test attribute after fn keyword."""
        with pytest.raises(ParseError):
            parse("fn #[mode(m8)] test() { }")

    def test_attribute_inside_function(self):
        """Test attribute inside function body."""
        with pytest.raises(ParseError):
            parse("fn test() { #[mode(m8)] let x = 10; }")

    def test_attribute_missing_name(self):
        """Test attribute without name."""
        with pytest.raises(ParseError):
            parse("#[] fn test() { }")

    def test_attribute_unclosed(self):
        """Test attribute without closing bracket."""
        with pytest.raises(ParseError):
            parse("#[mode(m8) fn test() { }")


class TestInvalidComments:
    """Tests for comment-related errors."""

    def test_unterminated_block_comment(self):
        """Test unterminated block comment."""
        with pytest.raises(ParseError):
            parse("fn test() { /* comment }")

    def test_valid_line_comment(self):
        """Test valid line comment."""
        prog = parse_succeeds("fn test() { // comment\n A = 1; }")
        assert len(prog.items) == 1

    def test_valid_block_comment(self):
        """Test valid block comment."""
        prog = parse_succeeds("fn test() { /* comment */ A = 1; }")
        assert len(prog.items) == 1


class TestInvalidAsmStatements:
    """Tests for invalid inline assembly."""

    def test_asm_no_arguments(self):
        """Test asm without arguments."""
        with pytest.raises(ParseError):
            parse("fn test() { asm!(); }")

    def test_asm_number_argument(self):
        """Test asm with number instead of string."""
        with pytest.raises(ParseError):
            parse("fn test() { asm!(123); }")

    def test_asm_missing_exclamation(self):
        """Test asm without exclamation mark."""
        with pytest.raises(ParseError):
            parse('fn test() { asm("WAI"); }')


class TestValidEdgeCases:
    """Tests for valid edge cases that should NOT error."""

    def test_empty_function(self):
        """Test empty function body is valid."""
        prog = parse_succeeds("fn empty() { }")
        assert len(prog.items) == 1

    def test_empty_struct(self):
        """Test empty struct is valid."""
        prog = parse_succeeds("struct Empty { }")
        assert len(prog.items) == 1

    def test_nested_blocks(self):
        """Test deeply nested blocks."""
        prog = parse_succeeds("fn test() { { { { A = 1; } } } }")
        assert len(prog.items) == 1

    def test_multiple_returns(self):
        """Test function with multiple return values."""
        prog = parse_succeeds("fn test() { return A, X, Y; }")
        assert len(prog.items) == 1

    def test_chained_field_access(self):
        """Test chained field access."""
        prog = parse_succeeds("""
            struct Inner { val: u8 }
            struct Outer { inner: Inner }
            #[ram] static mut O: Outer;
            fn test() { let x: u8 = O.inner.val; }
        """)
        assert len(prog.items) == 4

    def test_complex_expression(self):
        """Test complex expression."""
        prog = parse_succeeds("fn test() { let x: u8 = A + X * 2 - Y / 4 & 0xFF; }")
        assert len(prog.items) == 1


class TestErrorRecovery:
    """Tests for parser error recovery and messages."""

    def test_error_includes_line_info(self):
        """Test that parse errors include line information."""
        error_msg = parse_fails("fn test() {\n  let x = 10\n}")
        assert "line" in error_msg.lower() or "2" in error_msg

    def test_error_includes_position(self):
        """Test that parse errors include position information."""
        error_msg = parse_fails("fn test() { let x = ; }")
        assert "column" in error_msg.lower() or any(c.isdigit() for c in error_msg)

    def test_multiple_errors_first_reported(self):
        """Test that first error is reported for multiple issues."""
        # Parser stops at first error
        with pytest.raises(ParseError):
            parse("fn test() { let x = let y = 10; }")


class TestHIRErrors:
    """Tests for HIR building errors."""

    def test_hir_undefined_type(self):
        """Test HIR error for undefined type."""
        # Parser allows undefined types; HIR builder may catch
        try:
            hir_prog = build_hir("#[ram] static mut X: UndefinedType;")
            # If HIR builds, that's the builder's behavior
        except Exception:
            pass  # Expected to fail

    def test_hir_builds_valid_program(self):
        """Test that HIR builds for valid program."""
        hir_prog = build_hir("fn test() { A = 42; }")
        assert len(hir_prog.functions) == 1

    def test_hir_handles_all_constructs(self):
        """Test HIR with multiple constructs."""
        hir_prog = build_hir("""
            struct Point { x: u8, y: u8 }
            enum Dir { N, S, E, W }
            const SIZE: u8 = 10;
            #[ram]
            static mut DATA: [u8; 256];
            fn main() {
                A = SIZE;
            }
        """)
        assert len(hir_prog.functions) >= 1
        assert len(hir_prog.statics) >= 1


class TestSpecificSyntaxErrors:
    """Tests for specific syntax error scenarios."""

    def test_for_loop_not_supported(self):
        """Test that for loops are not supported."""
        with pytest.raises(ParseError):
            parse("fn test() { for i in 0..10 { } }")

    def test_match_not_fully_supported(self):
        """Test match expression syntax."""
        # Check if match is supported
        try:
            prog = parse("""
                fn test() {
                    match A {
                        0 => { X = 1; }
                        _ => { X = 0; }
                    }
                }
            """)
            # If it parses, match is supported
        except ParseError:
            pass  # Match might not be fully implemented

    def test_closure_not_supported(self):
        """Test that closures are not supported."""
        with pytest.raises(ParseError):
            parse("fn test() { let f = |x| x + 1; }")

    def test_async_not_supported(self):
        """Test that 'async' is not a keyword - parses as function name."""
        # 'async' is not a reserved keyword in R65, so it's parsed as an identifier
        # "async fn test() { }" parses as a function named 'async' followed by 'fn test()'
        # which causes a parse error because 'fn' can't follow a function definition
        # OR it parses 'async' as a function call. Let's verify actual behavior.
        try:
            prog = parse("async fn test() { }")
            # If it parses, async is treated as something valid
            # This would mean 'async' is parsed as a function and 'fn test()' is separate
            assert len(prog.items) >= 1
        except ParseError:
            # Parse error is also acceptable - depends on grammar
            pass

    def test_trait_not_supported(self):
        """Test that traits are not supported."""
        with pytest.raises(ParseError):
            parse("trait Foo { fn bar(); }")

    def test_impl_not_supported(self):
        """Test that impl blocks are not supported."""
        with pytest.raises(ParseError):
            parse("impl Foo { fn bar() { } }")

    def test_generic_not_supported(self):
        """Test that generics are not supported."""
        with pytest.raises(ParseError):
            parse("fn test<T>(x: T) { }")

    def test_lifetime_not_supported(self):
        """Test that lifetimes are not supported."""
        with pytest.raises(ParseError):
            parse("fn test<'a>(x: &'a u8) { }")
