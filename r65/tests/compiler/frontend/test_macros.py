"""
Tests for the macro system.

Tests macro definition parsing, invocation parsing, and macro expansion.
"""
import pytest
from r65.compiler.frontend import parse, expand_macros, MacroError
from r65.compiler.frontend import ast


def get_macro_invocation(stmt):
    """Extract MacroInvocation from statement, handling both parser outputs.

    LALR parser returns MacroInvocationStmtInner directly.
    Earley parser returns ExprStmt(MacroInvocation).
    Both are semantically equivalent.
    """
    if isinstance(stmt, ast.MacroInvocationStmtInner):
        # Create a MacroInvocation-like object for uniform access
        class MacroInvocationWrapper:
            def __init__(self, stmt):
                self.name = stmt.name
                self.args = stmt.args
        return MacroInvocationWrapper(stmt)
    elif isinstance(stmt, ast.ExprStmt) and isinstance(stmt.expr, ast.MacroInvocation):
        return stmt.expr
    else:
        raise AssertionError(f"Expected macro invocation, got {type(stmt)}")


# ============================================================================
# Macro Definition Parsing Tests
# ============================================================================

class TestMacroDefinitionParsing:
    """Tests for parsing macro definitions."""

    def test_simple_macro_definition(self):
        """Test parsing a simple macro with one parameter."""
        source = """
        macro_rules! inc($reg:reg) {
            $reg++;
        }
        """
        program = parse(source, "<test>")

        assert len(program.items) == 1
        macro = program.items[0]
        assert isinstance(macro, ast.MacroDecl)
        assert macro.name == "inc"
        assert len(macro.params) == 1
        assert macro.params[0].name == "reg"
        assert macro.params[0].fragment_type == "reg"
        assert macro.params[0].is_repeated is False

    def test_macro_with_multiple_params(self):
        """Test parsing a macro with multiple parameters."""
        source = """
        macro_rules! add_to($dest:reg, $val:expr) {
            $dest = $dest + $val;
        }
        """
        program = parse(source, "<test>")

        macro = program.items[0]
        assert len(macro.params) == 2
        assert macro.params[0].name == "dest"
        assert macro.params[0].fragment_type == "reg"
        assert macro.params[1].name == "val"
        assert macro.params[1].fragment_type == "expr"

    def test_macro_with_repeated_param(self):
        """Test parsing a macro with repeated parameter."""
        source = """
        macro_rules! push_all($($reg:reg),*) {
            $($reg++;)*
        }
        """
        program = parse(source, "<test>")

        macro = program.items[0]
        assert len(macro.params) == 1
        assert macro.params[0].name == "reg"
        assert macro.params[0].fragment_type == "reg"
        assert macro.params[0].is_repeated is True

    def test_macro_with_ident_fragment(self):
        """Test macro with ident fragment type."""
        source = """
        macro_rules! use_var($name:ident) {
            $name = 0;
        }
        """
        program = parse(source, "<test>")

        macro = program.items[0]
        assert macro.params[0].fragment_type == "ident"

    def test_macro_with_literal_fragment(self):
        """Test macro with literal fragment type."""
        source = """
        macro_rules! set_const($val:literal) {
            A = $val;
        }
        """
        program = parse(source, "<test>")

        macro = program.items[0]
        assert macro.params[0].fragment_type == "literal"

    def test_macro_with_ty_fragment(self):
        """Test macro with ty (type) fragment type."""
        source = """
        macro_rules! cast_to($e:expr, $t:ty) {
            $e as $t;
        }
        """
        program = parse(source, "<test>")

        macro = program.items[0]
        assert macro.params[1].fragment_type == "ty"

    def test_macro_with_tt_fragment(self):
        """Test macro with tt (token tree) fragment type."""
        source = """
        macro_rules! wrap($t:tt) {
            { $t }
        }
        """
        program = parse(source, "<test>")

        macro = program.items[0]
        assert macro.params[0].fragment_type == "tt"


# ============================================================================
# Macro Invocation Parsing Tests
# ============================================================================

class TestMacroInvocationParsing:
    """Tests for parsing macro invocations."""

    def test_simple_invocation(self):
        """Test parsing simple macro invocation in function body."""
        source = """
        macro_rules! inc($reg:reg) { $reg++; }

        fn main() {
            inc!(X);
        }
        """
        program = parse(source, "<test>")

        func = program.items[1]
        assert isinstance(func, ast.FunctionDecl)
        stmt = func.body.statements[0]
        macro = get_macro_invocation(stmt)
        assert macro.name == "inc"
        assert macro.args == ["X"]

    def test_invocation_with_expr_arg(self):
        """Test macro invocation with expression argument."""
        source = """
        macro_rules! add($v:expr) { A = A + $v; }

        fn main() {
            add!(1 + 2);
        }
        """
        program = parse(source, "<test>")

        func = program.items[1]
        stmt = func.body.statements[0]
        macro = get_macro_invocation(stmt)
        assert macro.name == "add"
        # Args should contain the expression as a single string
        assert len(macro.args) == 1
        assert "1" in macro.args[0] and "2" in macro.args[0]

    def test_invocation_with_multiple_args(self):
        """Test macro invocation with multiple arguments."""
        source = """
        macro_rules! copy($src:reg, $dst:reg) { $dst = $src; }

        fn main() {
            copy!(X, Y);
        }
        """
        program = parse(source, "<test>")

        func = program.items[1]
        stmt = func.body.statements[0]
        macro = get_macro_invocation(stmt)
        assert len(macro.args) == 2
        assert macro.args[0] == "X"
        assert macro.args[1] == "Y"

# ============================================================================
# Macro Expansion Tests
# ============================================================================

class TestMacroExpansion:
    """Tests for macro expansion."""

    def test_simple_expansion(self):
        """Test simple macro expansion."""
        source = """
        macro_rules! inc($reg:reg) {
            $reg++;
        }

        fn main() {
            inc!(X);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Macro definition should be removed
        assert len(expanded.items) == 1
        func = expanded.items[0]
        assert isinstance(func, ast.FunctionDecl)

        # Should have the expanded statement
        assert len(func.body.statements) == 1
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)
        assert isinstance(stmt.expr, ast.CompoundAssignment)

    def test_expansion_multiple_statements(self):
        """Test macro that expands to multiple statements."""
        source = """
        macro_rules! inc_twice($reg:reg) {
            $reg++;
            $reg++;
        }

        fn main() {
            inc_twice!(X);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = expanded.items[0]
        # Should have 2 statements from the expansion
        assert len(func.body.statements) == 2

    def test_expansion_with_value_substitution(self):
        """Test macro expansion with value substitution."""
        source = """
        macro_rules! set($reg:reg, $val:expr) {
            $reg = $val;
        }

        fn main() {
            set!(A, 42);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = expanded.items[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)
        assert isinstance(stmt.expr, ast.Assignment)
        # Check the value is 42
        assert isinstance(stmt.expr.value, ast.IntegerLiteral)
        assert stmt.expr.value.value == 42

    def test_nested_macro_expansion(self):
        """Test nested macro calls."""
        source = """
        macro_rules! inc($reg:reg) {
            $reg++;
        }

        macro_rules! inc_twice($reg:reg) {
            inc!($reg);
            inc!($reg);
        }

        fn main() {
            inc_twice!(X);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = expanded.items[0]
        # Both inc! calls should be expanded
        assert len(func.body.statements) == 2

    def test_macro_in_if_block(self):
        """Test macro expansion inside if block."""
        source = """
        macro_rules! inc($reg:reg) { $reg++; }

        fn main() {
            if A == 0 {
                inc!(X);
            }
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = expanded.items[0]
        if_stmt = func.body.statements[0]
        assert isinstance(if_stmt, ast.IfStmt)
        # Check the body was expanded
        assert len(if_stmt.then_block.statements) == 1
        assert isinstance(if_stmt.then_block.statements[0], ast.ExprStmt)

    def test_macro_in_loop_block(self):
        """Test macro expansion inside loop block."""
        source = """
        macro_rules! dec($reg:reg) { $reg--; }

        fn main() {
            loop {
                dec!(X);
                if X == 0 {
                    break;
                }
            }
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = expanded.items[0]
        loop_stmt = func.body.statements[0]
        assert isinstance(loop_stmt, ast.LoopStmt)
        # First statement should be expanded macro
        assert isinstance(loop_stmt.body.statements[0], ast.ExprStmt)

    def test_macro_in_while_block(self):
        """Test macro expansion inside while block."""
        source = """
        macro_rules! inc($reg:reg) { $reg++; }

        fn main() {
            while X != 10 {
                inc!(X);
            }
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = expanded.items[0]
        while_stmt = func.body.statements[0]
        assert isinstance(while_stmt, ast.WhileStmt)
        assert isinstance(while_stmt.body.statements[0], ast.ExprStmt)


# ============================================================================
# Macro Error Tests
# ============================================================================

class TestMacroErrors:
    """Tests for macro error handling."""

    def test_rust_arrow_syntax_error(self):
        """Test helpful error for Rust's => syntax."""
        from r65.compiler.frontend import ParseError

        source = """
        macro_rules! test($x:expr) => {
            A = $x;
        }
        """
        with pytest.raises(ParseError) as excinfo:
            parse(source, "<test>")
        error_msg = str(excinfo.value)
        assert "=>" in error_msg or "simplified macro syntax" in error_msg
        assert "R65 syntax" in error_msg

    def test_unsupported_fragment_error(self):
        """Test helpful error for unsupported fragment types."""
        from r65.compiler.frontend import ParseError

        source = """
        macro_rules! test($x:stmt) {
            $x;
        }
        """
        with pytest.raises(ParseError) as excinfo:
            parse(source, "<test>")
        error_msg = str(excinfo.value)
        assert "stmt" in error_msg
        assert "Supported fragment types" in error_msg

    def test_plus_repetition_error(self):
        """Test helpful error for + repetition (not supported)."""
        from r65.compiler.frontend import ParseError

        source = """
        macro_rules! test($($x:expr),+) {
            A = 0;
        }
        """
        with pytest.raises(ParseError) as excinfo:
            parse(source, "<test>")
        error_msg = str(excinfo.value)
        assert "+" in error_msg
        assert "*" in error_msg

    def test_undefined_macro(self):
        """Test error when calling undefined macro."""
        source = """
        fn main() {
            undefined_macro!(X);
        }
        """
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as excinfo:
            expand_macros(program)
        assert "undefined macro" in str(excinfo.value)

    def test_wrong_arg_count(self):
        """Test error when macro called with wrong number of args."""
        source = """
        macro_rules! inc($reg:reg) { $reg++; }

        fn main() {
            inc!(X, Y);
        }
        """
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as excinfo:
            expand_macros(program)
        assert "expects" in str(excinfo.value)

    def test_recursive_macro_error(self):
        """Test error on recursive macro expansion."""
        source = """
        macro_rules! recursive($x:expr) {
            recursive!($x);
        }

        fn main() {
            recursive!(1);
        }
        """
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as excinfo:
            expand_macros(program)
        assert "recursive" in str(excinfo.value).lower()


# ============================================================================
# Built-in stringify! Macro Tests  
# ============================================================================

class TestStringifyMacro:
    """Tests for built-in stringify! macro."""

    def test_stringify_single_arg(self):
        """Test stringify! with a single argument."""
        source = """
        fn test() {
            stringify!(Hello);
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        # Should have one function with one statement
        assert len(expanded.items) == 1
        func = expanded.items[0]
        assert len(func.body.statements) == 1
        
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "Hello"

    def test_stringify_multiple_args(self):
        """Test stringify! with multiple arguments."""
        source = """
        fn test() {
            stringify!(Hello World 123);
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        func = expanded.items[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "Hello World 123"

    def test_stringify_empty_args(self):
        """Test stringify! with no arguments."""
        source = """
        fn test() {
            stringify!();
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        func = expanded.items[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == ""

    def test_stringify_special_chars(self):
        """Test stringify! with special characters that need escaping."""
        source = """
        fn test() {
            stringify!(Hello "World");
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        func = expanded.items[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        # Should properly escape quotes
        assert stmt.expr.value == 'Hello \\"World\\"'

    def test_stringify_error_undefined_macro(self):
        """Test that non-stringify macros in statement context give proper error."""
        from r65.compiler.frontend.macros import MacroError
        
        source = """
        fn test() {
            unknown_macro!(arg);
        }
        """
        
        program = parse(source, "<test>")
        
        with pytest.raises(MacroError) as exc_info:
            expand_macros(program)
        
        assert "undefined macro: 'unknown_macro'" in str(exc_info.value)


# ============================================================================
# End-to-End Compilation Tests
# ============================================================================

class TestMacroCompilation:
    """Tests for full compilation with macros."""

    def test_compile_with_macro(self):
        """Test that code with macros compiles to assembly."""
        from r65.compiler.main import compile_string

        source = """
        macro_rules! inc_twice($reg:reg) {
            $reg++;
            $reg++;
        }

                fn main() {
            X = 0;
            inc_twice!(X);
            A = X;
        }
        """
        assembly = compile_string(source)

        # Should have INX twice from macro expansion
        assert assembly.count("INX") == 2

    def test_compile_macro_with_expression(self):
        """Test compiling macro with expression argument."""
        from r65.compiler.main import compile_string

        source = """
        macro_rules! add_const($val:expr) {
            A = A + $val;
        }

                fn test() {
            A = 10;
            add_const!(5);
        }
        """
        assembly = compile_string(source)

        # Should have addition with constant 5
        assert "ADC #$05" in assembly or "CLC" in assembly

    def test_compile_nested_macros(self):
        """Test compiling nested macro invocations."""
        from r65.compiler.main import compile_string

        source = """
        macro_rules! inc($reg:reg) {
            $reg++;
        }

        macro_rules! inc_three_times($reg:reg) {
            inc!($reg);
            inc!($reg);
            inc!($reg);
        }

                fn test() {
            X = 0;
            inc_three_times!(X);
        }
        """
        assembly = compile_string(source)

        # Should have INX three times from nested expansion
        assert assembly.count("INX") == 3


# ============================================================================
# Built-in stringify! Macro Tests  
# ============================================================================

class TestStringifyMacro:
    """Tests for built-in stringify! macro."""

    def test_stringify_single_arg(self):
        """Test stringify! with a single argument."""
        source = """
        fn test() {
            stringify!(Hello);
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        # Should have one function with one statement
        assert len(expanded.items) == 1
        func = expanded.items[0]
        assert len(func.body.statements) == 1
        
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "Hello"

    def test_stringify_multiple_args(self):
        """Test stringify! with multiple arguments."""
        source = """
        fn test() {
            stringify!(Hello World 123);
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        func = expanded.items[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "Hello World 123"

    def test_stringify_empty_args(self):
        """Test stringify! with no arguments."""
        source = """
        fn test() {
            stringify!();
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        func = expanded.items[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == ""

    def test_stringify_special_chars(self):
        """Test stringify! with special characters that need escaping."""
        source = """
        fn test() {
            stringify!(Hello "World");
        }
        """
        
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        
        func = expanded.items[0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        # Should properly escape quotes
        assert stmt.expr.value == 'Hello \\"World\\"'

    def test_stringify_error_undefined_macro(self):
        """Test that non-stringify macros in statement context give proper error."""
        from r65.compiler.frontend.macros import MacroError
        
        source = """
        fn test() {
            unknown_macro!(arg);
        }
        """
        
        program = parse(source, "<test>")
        
        with pytest.raises(MacroError) as exc_info:
            expand_macros(program)
        
        assert "undefined macro: 'unknown_macro'" in str(exc_info.value)


# ============================================================================
# Top-Level Macro Invocation Tests
# ============================================================================

class TestTopLevelMacroInvocation:
    """Tests for top-level macro invocations that expand to declarations."""

    def test_top_level_static_declaration(self):
        """Test macro that expands to a static variable declaration."""
        source = """
        macro_rules! define_port($name:ident, $addr:literal) {
            #[hw($addr)]
            static mut $name: u8;
        }

        define_port!(INIDISP, 0x2100);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have one static declaration
        assert len(expanded.items) == 1
        static = expanded.items[0]
        assert isinstance(static, ast.StaticDecl)
        assert static.name == "INIDISP"

    def test_top_level_multiple_declarations(self):
        """Test macro that expands to multiple declarations."""
        source = """
        macro_rules! define_ports($name1:ident, $name2:ident) {
            #[zeropage]
            static mut $name1: u8;
            #[zeropage]
            static mut $name2: u8;
        }

        define_ports!(PORT_A, PORT_B);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have two static declarations
        assert len(expanded.items) == 2
        assert all(isinstance(item, ast.StaticDecl) for item in expanded.items)
        assert expanded.items[0].name == "PORT_A"
        assert expanded.items[1].name == "PORT_B"

    def test_top_level_function_declaration(self):
        """Test macro that expands to a function declaration."""
        source = """
        macro_rules! define_handler($name:ident) {
            fn $name() {
                A = 0;
            }
        }

        define_handler!(my_handler);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have one function declaration
        assert len(expanded.items) == 1
        func = expanded.items[0]
        assert isinstance(func, ast.FunctionDecl)
        assert func.name == "my_handler"

    def test_top_level_multiple_invocations(self):
        """Test multiple top-level macro invocations."""
        source = """
        macro_rules! define_var($name:ident) {
            #[zeropage]
            static mut $name: u8;
        }

        define_var!(VAR1);
        define_var!(VAR2);
        define_var!(VAR3);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have three static declarations
        assert len(expanded.items) == 3
        assert expanded.items[0].name == "VAR1"
        assert expanded.items[1].name == "VAR2"
        assert expanded.items[2].name == "VAR3"

    def test_top_level_with_repeated_params(self):
        """Test top-level macro with repeated parameters."""
        source = """
        macro_rules! define_vars($($name:ident),*) {
            $(#[zeropage]
            static mut $name: u8;)*
        }

        define_vars!(X, Y, Z);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have three static declarations
        assert len(expanded.items) == 3
        names = [item.name for item in expanded.items]
        assert "X" in names
        assert "Y" in names
        assert "Z" in names

    def test_top_level_nested_macro_invocation(self):
        """Test top-level macro that expands to another macro invocation."""
        source = """
        macro_rules! inner_macro($name:ident) {
            #[zeropage]
            static mut $name: u8;
        }

        macro_rules! outer_macro($name:ident) {
            inner_macro!($name);
        }

        outer_macro!(NESTED_VAR);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have one static declaration after nested expansion
        assert len(expanded.items) == 1
        assert isinstance(expanded.items[0], ast.StaticDecl)
        assert expanded.items[0].name == "NESTED_VAR"

    def test_top_level_struct_declaration(self):
        """Test macro that expands to a struct declaration."""
        source = """
        macro_rules! define_struct($name:ident) {
            struct $name {
                x: u8,
                y: u8,
            }
        }

        define_struct!(Point);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have one struct declaration
        assert len(expanded.items) == 1
        struct = expanded.items[0]
        assert isinstance(struct, ast.StructDecl)
        assert struct.name == "Point"

    def test_top_level_enum_declaration(self):
        """Test macro that expands to an enum declaration."""
        source = """
        macro_rules! define_enum($name:ident) {
            enum $name {
                A = 0,
                B = 1,
            }
        }

        define_enum!(State);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have one enum declaration
        assert len(expanded.items) == 1
        enum = expanded.items[0]
        assert isinstance(enum, ast.EnumDecl)
        assert enum.name == "State"

    def test_top_level_const_declaration(self):
        """Test macro that expands to a const declaration."""
        source = """
        macro_rules! define_const($name:ident, $val:literal) {
            const $name: u8 = $val;
        }

        define_const!(MAX_VALUE, 255);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have one const declaration
        assert len(expanded.items) == 1
        const = expanded.items[0]
        assert isinstance(const, ast.ConstDecl)
        assert const.name == "MAX_VALUE"

    def test_top_level_mixed_declarations(self):
        """Test macro that expands to mixed declaration types."""
        source = """
        macro_rules! define_component($name:ident) {
            struct $name {
                value: u8,
            }

            #[zeropage]
            static mut CURRENT: $name;

            fn init() {
                A = 0;
            }
        }

        define_component!(Component);
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        # Should have struct, static, and function
        assert len(expanded.items) == 3
        types = [type(item).__name__ for item in expanded.items]
        assert "StructDecl" in types
        assert "StaticDecl" in types
        assert "FunctionDecl" in types

    def test_top_level_undefined_macro_error(self):
        """Test that undefined macro at top level raises error."""
        source = """
        undefined_macro!(arg);
        """
        program = parse(source, "<test>")

        with pytest.raises(MacroError) as exc_info:
            expand_macros(program)

        assert "undefined macro: 'undefined_macro'" in str(exc_info.value)

    def test_top_level_recursive_macro_error(self):
        """Test that recursive macro at top level raises error."""
        source = """
        macro_rules! recursive($x:ident) {
            recursive!($x);
        }

        recursive!(test);
        """
        program = parse(source, "<test>")

        with pytest.raises(MacroError) as exc_info:
            expand_macros(program)

        assert "recursive macro expansion" in str(exc_info.value)
