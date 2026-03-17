# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for helpful error messages when users try unsupported Rust features.

Each test verifies that R65 produces a clear, actionable error message
instead of a cryptic parse error when a Rust programmer uses idioms
that R65 doesn't support.
"""

import pytest
from r65.compiler.frontend import parse, ParseError, expand_macros
from r65.compiler.frontend.macros import MacroExpander, MacroError


# =============================================================================
# Reserved Keyword Errors (pub, async, unsafe, use, etc.)
# =============================================================================

class TestReservedKeywordErrors:
    """Test that reserved Rust keywords produce targeted error messages."""

    def test_pub_fn(self):
        """pub fn should error, not be silently ignored."""
        with pytest.raises(ParseError) as exc:
            parse("pub fn test() { }")
        assert "'pub'" in str(exc.value)
        assert exc.value.hint is not None
        assert "remove" in exc.value.hint.lower() or "global" in exc.value.hint.lower()

    def test_pub_static(self):
        """pub static should error."""
        with pytest.raises(ParseError) as exc:
            parse("pub static X: u8 = 0;")
        assert "'pub'" in str(exc.value)

    def test_async_fn(self):
        """async fn should error, not be silently ignored."""
        with pytest.raises(ParseError) as exc:
            parse("async fn test() { }")
        assert "'async'" in str(exc.value)
        assert exc.value.hint is not None
        assert "interrupt" in exc.value.hint.lower()

    def test_await(self):
        """await should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("await fn test() { }")
        assert "'await'" in str(exc.value)

    def test_unsafe_block(self):
        """unsafe should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("unsafe fn test() { }")
        assert "'unsafe'" in str(exc.value)
        assert exc.value.hint is not None
        assert "hardware" in exc.value.hint.lower() or "unnecessary" in exc.value.hint.lower()

    def test_use_import(self):
        """use import should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("use foo;")
        assert "'use'" in str(exc.value)
        assert exc.value.hint is not None
        assert "include!" in exc.value.hint

    def test_extern(self):
        """extern should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("extern fn test() { }")
        assert "'extern'" in str(exc.value)

    def test_crate(self):
        """crate should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("crate fn test() { }")
        assert "'crate'" in str(exc.value)
        assert exc.value.hint is not None
        assert "include!" in exc.value.hint

    def test_move(self):
        """move should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("move fn test() { }")
        assert "'move'" in str(exc.value)

    def test_yield(self):
        """yield should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("yield fn test() { }")
        assert "'yield'" in str(exc.value)

    def test_where(self):
        """where should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("where fn test() { }")
        assert "'where'" in str(exc.value)

    def test_ref(self):
        """ref should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("ref fn test() { }")
        assert "'ref'" in str(exc.value)

    def test_try(self):
        """try should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("try fn test() { }")
        assert "'try'" in str(exc.value)

    def test_super(self):
        """super should produce a targeted error."""
        with pytest.raises(ParseError) as exc:
            parse("super fn test() { }")
        assert "'super'" in str(exc.value)


# =============================================================================
# Generics
# =============================================================================

class TestGenericsErrors:
    """Test that generic syntax produces helpful errors."""

    def test_generic_function(self):
        """fn foo<T>() should explain generics aren't supported."""
        with pytest.raises(ParseError) as exc:
            parse("fn foo<T>(x: u8) { }")
        assert "generic" in str(exc.value).lower()

    def test_generic_after_type(self):
        """Type<T> should explain generics aren't supported."""
        with pytest.raises(ParseError) as exc:
            parse("fn foo(x: Vec<u8>) { }")
        assert "generic" in str(exc.value).lower()


# =============================================================================
# Closures
# =============================================================================

class TestClosureErrors:
    """Test that closure syntax produces helpful errors."""

    def test_closure_assignment(self):
        """let f = |x| x + 1; should explain closures aren't supported."""
        with pytest.raises(ParseError) as exc:
            parse("fn test() { let f = |x| x + 1; }")
        assert "closure" in str(exc.value).lower()
        assert exc.value.hint is not None
        assert "fn()" in exc.value.hint or "function pointer" in exc.value.hint.lower()


# =============================================================================
# ? Operator
# =============================================================================

class TestQuestionMarkErrors:
    """Test that the ? operator produces a helpful error."""

    def test_question_mark_operator(self):
        """foo()? should explain ? isn't supported."""
        with pytest.raises(ParseError) as exc:
            parse("fn test() { let x: u8 = foo()?; }")
        assert "'?'" in str(exc.value)
        assert exc.value.hint is not None
        assert "return code" in exc.value.hint.lower() or "Result" in exc.value.hint


# =============================================================================
# if let / while let
# =============================================================================

class TestIfLetErrors:
    """Test that if let / while let produce helpful errors."""

    def test_if_let(self):
        """if let x = ... should explain if let isn't supported."""
        with pytest.raises(ParseError) as exc:
            parse("fn test() { if let x = foo() { } }")
        assert "if let" in str(exc.value).lower()

    def test_while_let(self):
        """while let x = ... should explain while let isn't supported."""
        with pytest.raises(ParseError) as exc:
            parse("fn test() { while let x = foo() { } }")
        assert "while let" in str(exc.value).lower()


# =============================================================================
# &self in impl methods
# =============================================================================

class TestSelfRefErrors:
    """Test that &self produces a helpful error pointing to *self."""

    def test_ampersand_self_in_impl(self):
        """impl Foo { fn bar(&self) } should suggest *self."""
        with pytest.raises(ParseError) as exc:
            parse("""
            struct Foo { x: u8 }
            impl Foo {
                fn bar(&self) { }
            }
            """)
        assert "*self" in str(exc.value) or "*self" in (exc.value.hint or "")

    def test_ampersand_self_in_trait(self):
        """trait Foo { fn bar(&self); } should suggest *self."""
        with pytest.raises(ParseError) as exc:
            parse("""
            trait Foo {
                fn bar(&self);
            }
            """)
        assert "*self" in str(exc.value) or "*self" in (exc.value.hint or "")


# =============================================================================
# Rust Types
# =============================================================================

class TestRustTypeErrors:
    """Test that common Rust types produce helpful hints."""

    def _compile_to_hir(self, source):
        """Helper to compile through HIR to trigger type resolution."""
        from r65.compiler.hir import HIRBuilder, HIRError
        program = parse(source, "<test>")
        expanded = expand_macros(program)
        builder = HIRBuilder()
        builder.build_program(expanded)

    def test_string_type(self):
        """String type should hint about [u8; N]."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: String = 0; }")
        assert "Undefined type: String" in str(exc.value)
        assert exc.value.hint is not None
        assert "[u8; N]" in exc.value.hint or "array" in exc.value.hint.lower()

    def test_usize_type(self):
        """usize should hint about u16."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: usize = 0; }")
        assert "Undefined type: usize" in str(exc.value)
        assert exc.value.hint is not None
        assert "u16" in exc.value.hint

    def test_isize_type(self):
        """isize should hint about i16."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: isize = 0; }")
        assert exc.value.hint is not None
        assert "i16" in exc.value.hint

    def test_u32_type(self):
        """u32 should hint about 16-bit limitation."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: u32 = 0; }")
        assert exc.value.hint is not None
        assert "16-bit" in exc.value.hint

    def test_f32_type(self):
        """f32 should hint about no FPU."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: f32 = 0; }")
        assert exc.value.hint is not None
        assert "FPU" in exc.value.hint or "fixed-point" in exc.value.hint

    def test_option_type(self):
        """Option should hint about sentinel values."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: Option = 0; }")
        assert exc.value.hint is not None
        assert "sentinel" in exc.value.hint.lower() or "flag" in exc.value.hint.lower()

    def test_result_type(self):
        """Result should hint about return codes."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: Result = 0; }")
        assert exc.value.hint is not None
        assert "return code" in exc.value.hint.lower() or "error flag" in exc.value.hint.lower()

    def test_vec_type(self):
        """Vec should hint about fixed-size arrays."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: Vec = 0; }")
        assert exc.value.hint is not None
        assert "array" in exc.value.hint.lower()

    def test_char_type(self):
        """char should hint about u8."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: char = 0; }")
        assert exc.value.hint is not None
        assert "u8" in exc.value.hint

    def test_unknown_type_no_hint(self):
        """Unknown non-Rust types should not have a hint."""
        from r65.compiler.hir import HIRError
        with pytest.raises(HIRError) as exc:
            self._compile_to_hir("fn test() { let x: MyCustomType = 0; }")
        assert "Undefined type: MyCustomType" in str(exc.value)
        assert exc.value.hint is None


# =============================================================================
# Rust Macros
# =============================================================================

class TestRustMacroErrors:
    """Test that common Rust macros produce helpful hints."""

    def _expand_with_macro(self, source):
        """Helper: parse and expand macros."""
        program = parse(source, "<test>")
        return expand_macros(program)

    def test_println_macro(self):
        """println!() should hint about format!()."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { println!("hello"); }')
        assert "undefined macro: 'println'" in str(exc.value)
        assert exc.value.hint is not None
        assert "format!" in exc.value.hint

    def test_print_macro(self):
        """print!() should hint about format!()."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { print!("hello"); }')
        assert exc.value.hint is not None
        assert "format!" in exc.value.hint

    def test_panic_macro(self):
        """panic!() should hint about BRK."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { panic!("oh no"); }')
        assert exc.value.hint is not None
        assert "BRK" in exc.value.hint

    def test_todo_macro(self):
        """todo!() should hint about BRK."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { todo!(); }')
        assert exc.value.hint is not None
        assert "BRK" in exc.value.hint

    def test_assert_macro(self):
        """assert!() should hint about if + BRK."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { assert!(true); }')
        assert exc.value.hint is not None
        assert "BRK" in exc.value.hint

    def test_vec_macro(self):
        """vec![] should hint about array literals."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { vec!(1, 2, 3); }')
        assert exc.value.hint is not None
        assert "array" in exc.value.hint.lower()

    def test_dbg_macro(self):
        """dbg!() should hint about format!()."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { dbg!(42); }')
        assert exc.value.hint is not None
        assert "format!" in exc.value.hint

    def test_unknown_macro_no_hint(self):
        """Unknown non-Rust macros should not have a hint."""
        with pytest.raises(MacroError) as exc:
            self._expand_with_macro('fn test() { my_custom_macro!(1); }')
        assert "undefined macro: 'my_custom_macro'" in str(exc.value)
        assert exc.value.hint is None
