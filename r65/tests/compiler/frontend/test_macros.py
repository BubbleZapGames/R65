"""
Tests for the macro system.

Tests macro definition parsing, invocation parsing, and macro expansion.
"""
import pytest
from r65.compiler.frontend import parse, expand_macros, MacroError
from r65.compiler.frontend.macros import MacroExpander
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

class TestCompileErrorMacro:
    """Tests for the built-in compile_error! macro."""

    def test_compile_error_in_statement_context(self):
        """Test compile_error! raises error in statement context."""
        source = '''
        fn test() {
            compile_error!("This feature is not implemented");
        }
        '''
        program = parse(source, "<test>")

        with pytest.raises(MacroError) as exc_info:
            expand_macros(program)

        assert "This feature is not implemented" in str(exc_info.value)

class TestConstAssertMacro:
    """Tests for the built-in const_assert! macro."""

    def test_const_assert_passing(self):
        """Test const_assert! with true condition passes."""
        from r65.compiler.hir import HIRBuilder
        source = '''
        fn test() {
            const_assert!(1 < 8, "Value must be less than 8");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()
        # Should not raise
        hir = builder.build_program(expanded)
        assert hir is not None

    def test_const_assert_failing(self):
        """Test const_assert! with false condition raises error."""
        from r65.compiler.hir import HIRBuilder, HIRError
        source = '''
        fn test() {
            const_assert!(10 < 8, "Value must be less than 8");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()

        with pytest.raises(HIRError) as exc_info:
            builder.build_program(expanded)

        assert "Value must be less than 8" in str(exc_info.value)

    def test_const_assert_with_const_variable(self):
        """Test const_assert! with const variable in condition."""
        from r65.compiler.hir import HIRBuilder
        source = '''
        const MAX_CHANNEL: u8 = 8;

        fn test() {
            const_assert!(3 < MAX_CHANNEL, "Channel out of range");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()
        # Should not raise
        hir = builder.build_program(expanded)
        assert hir is not None

    def test_const_assert_default_message(self):
        """Test const_assert! without message uses default."""
        from r65.compiler.hir import HIRBuilder, HIRError
        source = '''
        fn test() {
            const_assert!(false);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()

        with pytest.raises(HIRError) as exc_info:
            builder.build_program(expanded)

        assert "const assertion failed" in str(exc_info.value)

    def test_const_assert_in_user_macro(self):
        """Test const_assert! used inside user-defined macro."""
        from r65.compiler.hir import HIRBuilder, HIRError
        source = '''
        macro_rules! check_channel($ch:expr) {
            const_assert!($ch < 8, "DMA channel must be 0-7");
        }

        fn test() {
            check_channel!(9);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()

        with pytest.raises(HIRError) as exc_info:
            builder.build_program(expanded)

        assert "DMA channel must be 0-7" in str(exc_info.value)

    def test_const_assert_valid_channel_in_macro(self):
        """Test const_assert! passes with valid channel in macro."""
        from r65.compiler.hir import HIRBuilder
        source = '''
        macro_rules! check_channel($ch:expr) {
            const_assert!($ch < 8, "DMA channel must be 0-7");
        }

        fn test() {
            check_channel!(0);
            check_channel!(7);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()
        # Should not raise
        hir = builder.build_program(expanded)
        assert hir is not None

    def test_const_assert_non_const_fails(self):
        """Test const_assert! with non-const expression fails."""
        from r65.compiler.hir import HIRBuilder, HIRError
        source = '''
        fn test() {
            const_assert!(A < 8, "Register not allowed");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()

        with pytest.raises(HIRError) as exc_info:
            builder.build_program(expanded)

        assert "not const-evaluable" in str(exc_info.value).lower() or "register" in str(exc_info.value).lower()


# ============================================================================
# Built-in symbol! Macro Tests
# ============================================================================

class TestSymbolMacro:
    """Tests for built-in symbol! macro that resolves R65 names to WLA-DX labels."""

    def test_symbol_rom_static_array_literal(self):
        """Immutable static with array literal init -> __NAME_data."""
        source = """
        static TILE_DATA: [u8; 4] = [0, 1, 2, 3];

        fn test() {
            symbol!(TILE_DATA);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = [i for i in expanded.items if isinstance(i, ast.FunctionDecl)][0]
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "__TILE_DATA_data"

    def test_symbol_rom_static_string_literal(self):
        """Immutable static with string literal init -> __NAME_data."""
        source = """
        static MESSAGE: [u8; 6] = "Hello";

        fn test() {
            symbol!(MESSAGE);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = [i for i in expanded.items if isinstance(i, ast.FunctionDecl)][0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "__MESSAGE_data"

    def test_symbol_mutable_static(self):
        """Mutable static (RAM) -> name unchanged."""
        source = """
        #[ram]
        static mut BUFFER: [u8; 256];

        fn test() {
            symbol!(BUFFER);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = [i for i in expanded.items if isinstance(i, ast.FunctionDecl)][0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "BUFFER"

    def test_symbol_unknown_identifier(self):
        """Unknown identifier -> pass-through."""
        source = """
        fn test() {
            symbol!(UNKNOWN_THING);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = [i for i in expanded.items if isinstance(i, ast.FunctionDecl)][0]
        stmt = func.body.statements[0]
        assert isinstance(stmt.expr, ast.StringLiteral)
        assert stmt.expr.value == "UNKNOWN_THING"

    def test_symbol_expression_context(self):
        """symbol! in expression context returns StringLiteral."""
        source = """
        static PALETTE: [u8; 8] = [0; 8];

        fn test() {
            let x @ A = symbol!(PALETTE);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = [i for i in expanded.items if isinstance(i, ast.FunctionDecl)][0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt, ast.LetStmt)
        assert isinstance(let_stmt.initializer, ast.StringLiteral)
        assert let_stmt.initializer.value == "__PALETTE_data"

    def test_symbol_in_macro_body_token_level(self):
        """symbol! resolved at token level inside user macro body (asm! named arg)."""
        source = """
        static GFX: [u8; 16] = [0; 16];

        macro_rules! load_src($ptr:ident) {
            asm!(
                "LDA #<{PTR}",
                PTR=symbol!($ptr)
            );
        }

        fn test() {
            load_src!(GFX);
        }
        """
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        func = [i for i in expanded.items if isinstance(i, ast.FunctionDecl)][0]
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.AsmStmt)
        # The asm named arg should have the resolved label
        assert stmt.format_args.get("PTR") == "__GFX_data"


# ============================================================================
# Built-in __format! Macro Tests
# ============================================================================

class TestFormatStringParser:
    """Tests for format string parsing internals."""

    def _get_expander(self):
        from r65.compiler.frontend.macros import MacroExpander
        return MacroExpander()

    def test_parse_literal_only(self):
        """Format string with no specifiers."""
        exp = self._get_expander()
        segments = exp._parse_format_string("Hello World", None)
        assert len(segments) == 1
        assert segments[0] == ('literal', {'text': 'Hello World'})

    def test_parse_single_specifier(self):
        """Format string with one specifier."""
        exp = self._get_expander()
        segments = exp._parse_format_string("N:{u8}", None)
        assert len(segments) == 2
        assert segments[0] == ('literal', {'text': 'N:'})
        assert segments[1] == ('specifier', {'type': 'u8', 'format': 'd'})

    def test_parse_multiple_specifiers(self):
        """Format string with multiple specifiers."""
        exp = self._get_expander()
        segments = exp._parse_format_string("{u8} {u16:x}", None)
        assert len(segments) == 3
        assert segments[0] == ('specifier', {'type': 'u8', 'format': 'd'})
        assert segments[1] == ('literal', {'text': ' '})
        assert segments[2] == ('specifier', {'type': 'u16', 'format': 'x'})

    def test_parse_escaped_braces(self):
        """{{ and }} produce literal braces."""
        exp = self._get_expander()
        segments = exp._parse_format_string("{{braces}}", None)
        assert len(segments) == 1
        assert segments[0] == ('literal', {'text': '{braces}'})

    def test_parse_empty_string(self):
        """Empty format string produces no segments."""
        exp = self._get_expander()
        segments = exp._parse_format_string("", None)
        assert len(segments) == 0

    def test_parse_all_specifier_types(self):
        """All specifier types parse correctly."""
        exp = self._get_expander()
        for spec_str, expected in [
            ('u8', {'type': 'u8', 'format': 'd'}),
            ('u16', {'type': 'u16', 'format': 'd'}),
            ('i8', {'type': 'i8', 'format': 'd'}),
            ('i16', {'type': 'i16', 'format': 'd'}),
            ('bool', {'type': 'bool'}),
            ('u8:x', {'type': 'u8', 'format': 'x'}),
            ('u16:x', {'type': 'u16', 'format': 'x'}),
            ('u16:5d', {'type': 'u16', 'format': 'd', 'width': 5}),
            ('u8:3d', {'type': 'u8', 'format': 'd', 'width': 3}),
            ('u8:03d', {'type': 'u8', 'format': 'd', 'width': 3, 'zero_pad': True}),
            ('u16:05d', {'type': 'u16', 'format': 'd', 'width': 5, 'zero_pad': True}),
            ('s', {'type': 's'}),
            ('c', {'type': 'c'}),
        ]:
            result = exp._parse_specifier(spec_str, None)
            assert result == expected, f"Failed for {spec_str}: {result}"

    def test_parse_escape_sequences(self):
        """Escape sequences in literal text are preserved."""
        exp = self._get_expander()
        segments = exp._parse_format_string("A\\nB{u8}", None)
        assert len(segments) == 2
        assert segments[0] == ('literal', {'text': 'A\\nB'})

    def test_parse_unterminated_brace_error(self):
        """Unterminated { raises error."""
        exp = self._get_expander()
        with pytest.raises(MacroError) as exc:
            exp._parse_format_string("bad {u8", None)
        assert "unterminated" in str(exc.value)

    def test_parse_unmatched_close_brace_error(self):
        """Unmatched } raises error."""
        exp = self._get_expander()
        with pytest.raises(MacroError) as exc:
            exp._parse_format_string("bad } here", None)
        assert "unmatched" in str(exc.value)

    def test_parse_unknown_specifier_error(self):
        """Unknown specifier raises error."""
        exp = self._get_expander()
        with pytest.raises(MacroError) as exc:
            exp._parse_specifier("i32", None)
        assert "unknown format specifier" in str(exc.value)

    def test_parse_padded_width_bounds(self):
        """Padded width outside 1-10 raises error."""
        exp = self._get_expander()
        with pytest.raises(MacroError):
            exp._parse_specifier("u16:0d", None)
        with pytest.raises(MacroError):
            exp._parse_specifier("u16:11d", None)

    def test_compute_byte_length_simple(self):
        """Byte length of plain text."""
        exp = self._get_expander()
        assert exp._compute_literal_byte_length("Hello") == 5

    def test_compute_byte_length_escapes(self):
        """Byte length counts escape sequences as 1 byte each."""
        exp = self._get_expander()
        assert exp._compute_literal_byte_length("A\\nB") == 3  # A, \n, B
        assert exp._compute_literal_byte_length("\\x41") == 1  # \x41
        assert exp._compute_literal_byte_length("\\\\") == 1   # \\

    def test_compute_byte_length_empty(self):
        """Byte length of empty string is 0."""
        exp = self._get_expander()
        assert exp._compute_literal_byte_length("") == 0

    def test_adjacent_specifiers(self):
        """Adjacent specifiers with no literal between them."""
        exp = self._get_expander()
        segments = exp._parse_format_string("{u8}{u16}", None)
        assert len(segments) == 2
        assert segments[0] == ('specifier', {'type': 'u8', 'format': 'd'})
        assert segments[1] == ('specifier', {'type': 'u16', 'format': 'd'})


class TestFormatMacro:
    """Tests for format! macro expansion."""

    def test_format_literal_only(self):
        """format! with only literal text generates inline copy + null terminate."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "Hello");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        assert len(funcs) == 1
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, let __fmtlit0, for loop, ptr advance, null terminate
        assert len(stmts) >= 3

    def test_format_with_u8(self):
        """format! with {u8} specifier generates u8_to_dec call."""
        source = '''
        far fn u8_to_dec(buf: far *u8, value @ A: u8) -> u8 { return 0; }

        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "N:{u8}", 42);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, 2x inline byte write (N:), let __fmtn0 = u8_to_dec, advance, null term
        assert len(stmts) >= 5

    def test_format_wrong_arg_count(self):
        """format! with wrong number of args raises error."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "{u8} {u16}", 42);
        }
        '''
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as exc:
            expand_macros(program)
        assert "2 format specifier(s)" in str(exc.value)
        assert "1 argument(s)" in str(exc.value)

    def test_format_missing_format_string(self):
        """format! with only buffer raises error."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF);
        }
        '''
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as exc:
            expand_macros(program)
        assert "requires at least" in str(exc.value)

    def test_format_non_string_literal(self):
        """format! with non-string second arg raises error."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, 42);
        }
        '''
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as exc:
            expand_macros(program)
        assert "string literal" in str(exc.value)

    def test_format_unknown_specifier(self):
        """format! with unknown specifier raises error."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "{i32}", 42);
        }
        '''
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as exc:
            expand_macros(program)
        assert "unknown format specifier" in str(exc.value)

    def test_format_unterminated_brace(self):
        """format! with unterminated { raises error."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "bad {u8");
        }
        '''
        program = parse(source, "<test>")
        with pytest.raises(MacroError) as exc:
            expand_macros(program)
        assert "unterminated" in str(exc.value)

    def test_format_empty_string(self):
        """format! with empty format string generates just null terminate."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, *__fmtptr = 0
        assert len(stmts) == 2

    def test_format_compile_with_all_specifiers(self):
        """format! with all specifier types compiles without error."""
        from r65.compiler.main import compile_string

        source = '''
        far fn u8_to_dec(buf: far *u8, value @ A: u8) -> u8 { return 0; }
        far fn u16_to_dec(buf: far *u8, value @ A: u16) -> u8 { return 0; }
        far fn u8_to_hex(buf: far *u8, value @ A: u8) {}
        far fn u16_to_hex(buf: far *u8, value @ A: u16) {}
        far fn u16_to_dec_pad(buf: far *u8, value @ A: u16, width: u8, fill: u8) -> u8 { return 0; }
        far fn strcpy(dst: far *u8, src: far *u8) -> u16 { return 0; }

        #[ram]
        static mut BUF: [u8; 128] = [0; 128];
        static NAME: [u8; 6] = "World";

        fn test() {
            __format!(BUF, "{u8} {u16} {u8:x} {u16:x} {u16:5d} {s} {c}",
                42, 1000, 0xAB, 0xDEAD, 99, &NAME as far *u8, 0x58);
        }
        '''
        assembly = compile_string(source)
        assert assembly is not None
        assert len(assembly) > 0


class TestFormatBufferOverflow:
    """Tests for format! buffer overflow detection."""

    def _get_expander_with_program(self, source):
        """Parse source and return an expander with program items loaded."""
        program = parse(source, "<test>")
        expander = MacroExpander()
        expander._program_items = program.items
        return expander

    def test_no_warning_when_buffer_fits(self):
        """No warning when max output fits in buffer."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];
        '''
        exp = self._get_expander_with_program(source)
        segments = exp._parse_format_string("Hello", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 0

    def test_warning_when_literal_overflows(self):
        """Warning when literal text alone exceeds buffer."""
        source = '''
        #[ram]
        static mut BUF: [u8; 4] = [0; 4];
        '''
        exp = self._get_expander_with_program(source)
        # "Hello" = 5 bytes + 1 null = 6 > 4
        segments = exp._parse_format_string("Hello", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1
        assert "overflow" in exp.warnings[0]
        assert "BUF" in exp.warnings[0]

    def test_warning_with_u8_specifier_overflow(self):
        """Warning when u8 specifier max (3) causes overflow."""
        source = '''
        #[ram]
        static mut BUF: [u8; 4] = [0; 4];
        '''
        exp = self._get_expander_with_program(source)
        # "N:" = 2 + {u8} max 3 + null = 6 > 4
        segments = exp._parse_format_string("N:{u8}", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1

    def test_warning_with_u16_specifier_overflow(self):
        """Warning when u16 specifier max (5) causes overflow."""
        source = '''
        #[ram]
        static mut BUF: [u8; 4] = [0; 4];
        '''
        exp = self._get_expander_with_program(source)
        # {u16} max 5 + null = 6 > 4
        segments = exp._parse_format_string("{u16}", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1

    def test_no_warning_when_exactly_fits(self):
        """No warning when max output exactly fills buffer."""
        source = '''
        #[ram]
        static mut BUF: [u8; 3] = [0; 3];
        '''
        exp = self._get_expander_with_program(source)
        # "AB" = 2 + null = 3 == 3
        segments = exp._parse_format_string("AB", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 0

    def test_no_warning_for_unknown_buffer(self):
        """No warning when buffer is not a known static array."""
        source = '''
        #[ram]
        static mut OTHER: [u8; 4] = [0; 4];
        '''
        exp = self._get_expander_with_program(source)
        segments = exp._parse_format_string("Hello World", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 0

    def test_hex_specifier_sizes(self):
        """Hex specifiers have fixed sizes: u8:x=2, u16:x=4."""
        source = '''
        #[ram]
        static mut BUF: [u8; 6] = [0; 6];
        '''
        exp = self._get_expander_with_program(source)
        # {u8:x}=2 + {u16:x}=4 + null = 7 > 6
        segments = [
            ('specifier', {'type': 'u8', 'format': 'x'}),
            ('specifier', {'type': 'u16', 'format': 'x'}),
        ]
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1

    def test_padded_specifier_uses_width(self):
        """Padded u16:Nd uses width N for size calculation."""
        source = '''
        #[ram]
        static mut BUF: [u8; 6] = [0; 6];
        '''
        exp = self._get_expander_with_program(source)
        # {u16:8d} = 8 + null = 9 > 6
        segments = [
            ('specifier', {'type': 'u16', 'format': 'd', 'width': 8}),
        ]
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1

    def test_char_specifier_size(self):
        """Char specifier {c} contributes 1 byte."""
        source = '''
        #[ram]
        static mut BUF: [u8; 3] = [0; 3];
        '''
        exp = self._get_expander_with_program(source)
        # "AB" = 2 + {c} = 1 + null = 4 > 3
        segments = exp._parse_format_string("AB{c}", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1

    def test_string_specifier_unbounded_near_full(self):
        """Warning when known parts nearly fill buffer and {s} adds more."""
        source = '''
        #[ram]
        static mut BUF: [u8; 6] = [0; 6];
        '''
        exp = self._get_expander_with_program(source)
        # "Hello" = 5 + {s} = unknown + null = at least 6 == 6
        segments = exp._parse_format_string("Hello{s}", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1
        assert "{s}" in exp.warnings[0]

    def test_string_specifier_unbounded_with_room(self):
        """No warning when known parts leave room for {s}."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];
        '''
        exp = self._get_expander_with_program(source)
        # "Hi " = 3 + {s} = unknown + null = at least 4 < 32
        segments = exp._parse_format_string("Hi {s}", None)
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 0

    def test_i8_overflow_size(self):
        """i8 contributes max 4 bytes ('-128')."""
        source = '''
        #[ram]
        static mut BUF: [u8; 4] = [0; 4];
        '''
        exp = self._get_expander_with_program(source)
        # {i8} max 4 + null = 5 > 4
        segments = [('specifier', {'type': 'i8', 'format': 'd'})]
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1

    def test_i16_overflow_size(self):
        """i16 contributes max 6 bytes ('-32768')."""
        source = '''
        #[ram]
        static mut BUF: [u8; 6] = [0; 6];
        '''
        exp = self._get_expander_with_program(source)
        # {i16} max 6 + null = 7 > 6
        segments = [('specifier', {'type': 'i16', 'format': 'd'})]
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1

    def test_bool_overflow_size(self):
        """bool contributes 1 byte."""
        source = '''
        #[ram]
        static mut BUF: [u8; 2] = [0; 2];
        '''
        exp = self._get_expander_with_program(source)
        # {bool} = 1 + null = 2 == 2, should fit
        segments = [('specifier', {'type': 'bool'})]
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 0

    def test_u8_padded_overflow_size(self):
        """u8 with width uses width for overflow calculation."""
        source = '''
        #[ram]
        static mut BUF: [u8; 4] = [0; 4];
        '''
        exp = self._get_expander_with_program(source)
        # {u8:5d} = 5 + null = 6 > 4
        segments = [('specifier', {'type': 'u8', 'format': 'd', 'width': 5})]
        exp._check_format_buffer_overflow("BUF", segments, None)
        assert len(exp.warnings) == 1


class TestFormatSpecifierParsing:
    """Tests for new format specifier parsing."""

    def _get_expander(self):
        return MacroExpander()

    def test_parse_bool(self):
        """bool specifier parses correctly."""
        exp = self._get_expander()
        result = exp._parse_specifier("bool", None)
        assert result == {'type': 'bool'}

    def test_parse_i8(self):
        """i8 specifier parses correctly."""
        exp = self._get_expander()
        result = exp._parse_specifier("i8", None)
        assert result == {'type': 'i8', 'format': 'd'}

    def test_parse_i16(self):
        """i16 specifier parses correctly."""
        exp = self._get_expander()
        result = exp._parse_specifier("i16", None)
        assert result == {'type': 'i16', 'format': 'd'}

    def test_parse_u8_padded(self):
        """u8:3d space-padded specifier parses correctly."""
        exp = self._get_expander()
        result = exp._parse_specifier("u8:3d", None)
        assert result == {'type': 'u8', 'format': 'd', 'width': 3}

    def test_parse_u8_zero_padded(self):
        """u8:03d zero-padded specifier parses correctly."""
        exp = self._get_expander()
        result = exp._parse_specifier("u8:03d", None)
        assert result == {'type': 'u8', 'format': 'd', 'width': 3, 'zero_pad': True}

    def test_parse_u16_zero_padded(self):
        """u16:05d zero-padded specifier parses correctly."""
        exp = self._get_expander()
        result = exp._parse_specifier("u16:05d", None)
        assert result == {'type': 'u16', 'format': 'd', 'width': 5, 'zero_pad': True}

    def test_parse_padded_width_bounds_u8(self):
        """u8 padded width outside 1-10 raises error."""
        exp = self._get_expander()
        with pytest.raises(MacroError):
            exp._parse_specifier("u8:0d", None)
        with pytest.raises(MacroError):
            exp._parse_specifier("u8:11d", None)


class TestFormatLiteralInlining:
    """Tests for small literal inlining optimization."""

    def _get_expander(self):
        return MacroExpander()

    def test_literal_to_bytes_simple(self):
        """Plain ASCII text converts to byte values."""
        exp = self._get_expander()
        assert exp._literal_to_bytes("AB") == [0x41, 0x42]

    def test_literal_to_bytes_escapes(self):
        """Escape sequences convert correctly."""
        exp = self._get_expander()
        assert exp._literal_to_bytes("\\n") == [0x0A]
        assert exp._literal_to_bytes("\\t") == [0x09]
        assert exp._literal_to_bytes("\\r") == [0x0D]
        assert exp._literal_to_bytes("\\0") == [0x00]
        assert exp._literal_to_bytes("\\\\") == [0x5C]

    def test_literal_to_bytes_hex_escape(self):
        """\\xNN hex escapes convert correctly."""
        exp = self._get_expander()
        assert exp._literal_to_bytes("\\x41") == [0x41]
        assert exp._literal_to_bytes("\\xFF") == [0xFF]

    def test_literal_to_bytes_mixed(self):
        """Mixed text and escapes convert correctly."""
        exp = self._get_expander()
        assert exp._literal_to_bytes("A\\nB") == [0x41, 0x0A, 0x42]

    def test_small_literal_inlines(self):
        """Small literals emit inline byte writes."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "Hi");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, 2x (*__fmtptr=byte + ptr advance), null terminate
        # That's: 1 let + 2*(deref + advance) + 1 null = 6 stmts
        assert len(stmts) == 6

    def test_large_literal_uses_for_loop(self):
        """4+ byte literals use for loop copy from static ROM array."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "Hello");
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, for loop, ptr advance, null terminate = 4
        assert len(stmts) == 4

        # Should have injected a static declaration for the literal
        statics = [i for i in expanded.items if isinstance(i, ast.StaticDecl) and i.name.startswith('__fmtstr_')]
        assert len(statics) == 1
        assert statics[0].is_mut is False  # ROM (immutable)


class TestFormatNewSpecifiers:
    """Tests for format! with new specifier types."""

    def test_format_with_bool(self):
        """format! with {bool} generates inline if/else."""
        source = '''
        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "{bool}", true);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, if/else, ptr advance, null terminate = 4
        assert len(stmts) == 4

    def test_format_with_i8(self):
        """format! with {i8} generates inline sign check + u8_to_dec call."""
        source = '''
        far fn u8_to_dec(buf: far *u8, value @ A: u8) -> u8 { return 0; }

        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "{i8}", 42);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, let __fmts0, if sign check, let __fmtn0 = u8_to_dec,
        #              ptr advance, null terminate = 6
        assert len(stmts) == 6

    def test_format_with_i16(self):
        """format! with {i16} generates inline sign check + u16_to_dec call."""
        source = '''
        far fn u16_to_dec(buf: far *u8, value @ A: u16) -> u8 { return 0; }

        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "{i16}", 1000);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, let __fmts0, if sign check, let __fmtn0 = u16_to_dec,
        #              ptr advance, null terminate = 6
        assert len(stmts) == 6

    def test_format_with_u8_padded(self):
        """format! with {u8:3d} generates u8_to_dec_pad call."""
        source = '''
        far fn u8_to_dec_pad(buf: far *u8, value @ A: u8, width: u8, fill: u8) -> u8 { return 0; }

        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "{u8:3d}", 42);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, u8_to_dec_pad call, ptr advance, null terminate = 4
        assert len(stmts) == 4

    def test_format_with_u16_zero_padded(self):
        """format! with {u16:05d} generates u16_to_dec_pad with zero fill."""
        source = '''
        far fn u16_to_dec_pad(buf: far *u8, value @ A: u16, width: u8, fill: u8) -> u8 { return 0; }

        #[ram]
        static mut BUF: [u8; 32] = [0; 32];

        fn test() {
            __format!(BUF, "{u16:05d}", 42);
        }
        '''
        program = parse(source, "<test>")
        expanded = expand_macros(program)

        funcs = [i for i in expanded.items if isinstance(i, ast.FunctionDecl) and i.name == 'test']
        stmts = funcs[0].body.statements
        # Should have: let __fmtptr, u16_to_dec_pad call, ptr advance, null terminate = 4
        assert len(stmts) == 4
