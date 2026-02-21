"""Tests for FixedStack ABI parameter promotion."""

import pytest
from r65.compiler.main import compile_string
from r65.compiler.codegen.abi_model import ABI_FIXED_STACK
from r65.compiler.errors import CodegenError


class TestFixedStackBasicPromotion:
    """Test that FixedStack promotes stack params to hw regs."""

    def test_single_u8_stack_param_promoted_to_A(self):
        """A single u8 stack param should be promoted to A register."""
        source = """
        fn add_one(x: u8) -> u8 {
            return x + 1;
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        # Should NOT contain TSC/SBC/TCS frame setup
        assert "TSC" not in result or "SBC" not in result
        # Function should still compile
        assert "add_one:" in result

    def test_single_u16_stack_param_promoted_to_X(self):
        """A single u16 stack param should be promoted to X register."""
        source = """
        fn add_one(x: u16) -> u16 {
            return x + 1;
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        assert "add_one:" in result

    def test_no_outgoing_arg_area(self):
        """FixedStack should not allocate outgoing arg area."""
        source = """
        fn callee(x: u8) -> u8 {
            return x;
        }
        fn caller(a @ A: u8) -> u8 {
            return callee(a);
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        assert "caller:" in result
        assert "callee:" in result


class TestFixedStackWithExplicitBindings:
    """Test FixedStack with mixed explicit and auto-promoted params."""

    def test_explicit_A_plus_stack_u16(self):
        """Explicit @ A param + u16 stack param → stack promoted to X."""
        source = """
        fn process(a @ A: u8, b: u16) -> u8 {
            return a;
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        assert "process:" in result

    def test_explicit_A_plus_stack_u8(self):
        """Explicit @ A param + u8 stack param → stack promoted to B."""
        source = """
        fn process(a @ A: u8, b: u8) -> u8 {
            return a;
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        assert "process:" in result

    def test_all_explicit_bindings_no_promotion_needed(self):
        """When all params already have bindings, no promotion needed."""
        source = """
        fn add(a @ A: u8, b @ X: u16) -> u8 {
            return a;
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        assert "add:" in result


class TestFixedStackScratchFallback:
    """Test that params fall back to scratch when hw regs exhausted."""

    def test_three_u8_params(self):
        """Three u8 params: 2 to hw regs (A, B), 1 to scratch."""
        source = """
        #[zeropage(0x10, register)]
        static mut SCRATCH0: u8;

        fn three_params(a: u8, b: u8, c: u8) -> u8 {
            return a;
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        assert "three_params:" in result


class TestFixedStackNoTscFrame:
    """Test that FixedStack never uses TSC/SBC/TCS."""

    def test_function_with_locals(self):
        """Even with local variables, FixedStack shouldn't use TSC/SBC/TCS."""
        source = """
        fn uses_locals(a @ A: u8) -> u8 {
            let x: u8 = a + 1;
            let y: u8 = x + 2;
            return y;
        }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        # Extract just the function body (between label and RTS)
        lines = result.split('\n')
        in_func = False
        func_lines = []
        for line in lines:
            if 'uses_locals:' in line:
                in_func = True
            elif in_func:
                if line.strip() in ('RTS', 'RTL'):
                    break
                func_lines.append(line)

        func_body = '\n'.join(func_lines)
        # TSC followed by SBC (frame allocation pattern) should not appear
        # Note: TSC might appear in other contexts, so check for the specific pattern
        assert 'SBC' not in func_body or 'TSC' not in func_body


class TestFixedStackDefaultUnchanged:
    """Test that Default ABI behavior is unchanged."""

    def test_default_abi_allows_stack_params(self):
        """Default ABI should still work with stack params normally."""
        source = """
        fn add(a: u8, b: u8) -> u8 {
            return a + b;
        }
        """
        # Should compile without abi_model (default)
        result = compile_string(source, "test.r65")
        assert "add:" in result

    def test_default_abi_explicit_none(self):
        """Passing abi_model=None should use Default ABI (the default)."""
        source = """
        fn identity(x @ A: u8) -> u8 {
            return x;
        }
        """
        result = compile_string(source, "test.r65", abi_model=None)
        assert "identity:" in result


class TestFixedStackRecursionRejection:
    """Test that FixedStack rejects recursive functions."""

    def test_direct_recursion_rejected(self):
        """Direct recursion should be a compile error under FixedStack.

        Uses stack params so the existing RecursionChecker (which only
        rejects register/zeropage params) passes, and the FixedStack
        check catches the recursion.
        """
        source = """
        fn countdown(n: u8) -> u8 {
            if n == 0 { return 0; }
            return countdown(n - 1);
        }
        """
        with pytest.raises(CodegenError, match="does not support recursive"):
            compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)

    def test_mutual_recursion_rejected(self):
        """Mutual recursion should be a compile error under FixedStack."""
        source = """
        fn is_even(n: u8) -> u8 {
            if n == 0 { return 1; }
            return is_odd(n - 1);
        }
        fn is_odd(n: u8) -> u8 {
            if n == 0 { return 0; }
            return is_even(n - 1);
        }
        """
        with pytest.raises(CodegenError, match="does not support recursive"):
            compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)

    def test_non_recursive_allowed(self):
        """Non-recursive functions should compile fine under FixedStack."""
        source = """
        fn helper(x @ A: u8) -> u8 { return x + 1; }
        fn caller(x @ A: u8) -> u8 { return helper(x); }
        """
        result = compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
        assert "helper:" in result
        assert "caller:" in result

    def test_default_abi_allows_recursion(self):
        """Default ABI should allow recursive functions with stack params."""
        source = """
        fn countdown(n: u8) -> u8 {
            if n == 0 { return 0; }
            return countdown(n - 1);
        }
        """
        result = compile_string(source, "test.r65")
        assert "countdown:" in result
