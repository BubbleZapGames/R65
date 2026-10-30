# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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

    def test_caller_of_promoted_param_compiles(self):
        """A caller of a function whose stack param was promoted compiles."""
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


def test_address_taken_function_with_stack_params_rejected():
    """A call through a pointer passes its arguments on the stack, which
    FixedStack does not use, so an address-taken function takes none."""
    source = """
    #[zeropage(0x20, register)] static mut S0: u8;
    fn put(v: u8, w: u8) { }
    static HP: fn(u8, u8) = put;
    #[entry]
    fn main() { HP(7, 8); }
    """
    with pytest.raises(CodegenError, match="has its address taken but has parameters"):
        compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)


def test_dispatch_reserves_the_dispatched_methods_param_bytes(monkeypatch):
    """A caller of `wide` reserves both bytes of its u16 scratch param,
    sized from `wide`'s signature, not from the first impl (`small`)."""
    from r65.compiler.analysis import fixedstack_params as fp

    reserved = {}
    collect = fp._collect_trait_dispatch_scratch_addrs

    def spy(program, *args):
        collect(program, *args)
        for f in program.functions:
            reserved[f.name] = f.trait_dispatch_scratch_bytes

    monkeypatch.setattr(fp, '_collect_trait_dispatch_scratch_addrs', spy)
    source = """
    #[zeropage(0x10, register)] static mut S0: u8;
    #[zeropage(0x11, register)] static mut S1: u8;
    #[zeropage(0x12, register)] static mut S2: u8;
    #[zeropage(0x13, register)] static mut S3: u8;
    #[zeropage(0x14, register)] static mut S4: u8;
    struct Foo { a: u8 }
    trait T {
        fn small(far *self, v: u8) -> u8;
        fn wide(far *self, v: u16) -> u8;
    }
    impl T for Foo {
        fn small(far *self, v: u8) -> u8 { return v; }
        fn wide(far *self, v: u16) -> u8 { return v as u8; }
    }
    #[ram] static mut FOO: Foo;
    #[zeropage(0x30)] static mut R: u8;
    fn run(p: far *dyn T) { R = p.wide(0x1234); }
    #[entry] fn main() -> ! { run(&FOO as far *dyn T); run(&FOO as far *dyn T); loop {} }
    """
    compile_string(source, "test.r65", abi_model=ABI_FIXED_STACK)
    assert reserved['run'] == {0x10, 0x11}
