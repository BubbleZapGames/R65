"""
Tests for the macro system.

Tests macro definition parsing, invocation parsing, and macro expansion.
"""
import pytest
from r65.compiler.frontend import parse, expand_macros, MacroError
from r65.compiler.frontend import ast


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

    def test_macro_no_params(self):
        """Test macro with no parameters."""
        source = """
        macro_rules! nop() {
            A = A;
        }
        """
        program = parse(source, "<test>")

        macro = program.items[0]
        assert len(macro.params) == 0


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
        assert isinstance(stmt, ast.MacroInvocationStmtInner)
        assert stmt.name == "inc"
        assert stmt.args == ["X"]

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
        assert stmt.name == "add"
        # Args should contain the expression as a single string
        assert len(stmt.args) == 1
        assert "1" in stmt.args[0] and "2" in stmt.args[0]

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
        assert len(stmt.args) == 2
        assert stmt.args[0] == "X"
        assert stmt.args[1] == "Y"

    def test_invocation_no_args(self):
        """Test macro invocation with no arguments."""
        source = """
        macro_rules! nop() { A = A; }

        fn main() {
            nop!();
        }
        """
        program = parse(source, "<test>")

        func = program.items[1]
        stmt = func.body.statements[0]
        assert len(stmt.args) == 0


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

        #[mode(m8, x8)]
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

        #[mode(m8, x8)]
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

        #[mode(m8, x8)]
        fn test() {
            X = 0;
            inc_three_times!(X);
        }
        """
        assembly = compile_string(source)

        # Should have INX three times from nested expansion
        assert assembly.count("INX") == 3
