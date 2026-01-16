"""Tests for far pointer parameters using D = S technique."""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.mir import MIRBuilder
from r65.compiler.errors import MIRLoweringError


def build_mir(source: str):
    """Parse source and build MIR."""
    program = parse(source, "test.r65")
    hir_builder = HIRBuilder(source_file="test.r65")
    hir_prog = hir_builder.build_program(program)
    mir_builder = MIRBuilder()
    return mir_builder.build_program(hir_prog)


class TestFarPointerStackParamDetection:
    """Tests for detecting far pointer stack parameters."""

    def test_far_pointer_stack_param_detected(self):
        """Function with far pointer stack param should have flag set."""
        source = """
            #[mode(m8, x16)]
            fn process(far *data: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True
        assert 0 in func.far_ptr_param_indices

    def test_near_pointer_stack_param_not_detected(self):
        """Function with near pointer stack param should not have flag set."""
        source = """
            #[mode(m8, x16)]
            fn process(*data: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is False

    def test_register_bound_far_pointer_not_detected(self):
        """Register-bound far pointer should not set stack param flag."""
        source = """
            #[mode(m16, x16)]
            fn process(far *data @ A: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        # Register-bound params don't go on stack
        assert func.has_far_ptr_stack_params is False

    def test_multiple_far_pointer_params(self):
        """Multiple far pointer stack params should all be tracked."""
        source = """
            #[mode(m8, x16)]
            fn copy(far *src: u8, far *dst: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True
        assert 0 in func.far_ptr_param_indices
        assert 1 in func.far_ptr_param_indices


class TestFarPointerModeValidation:
    """Tests for x16 mode requirement with far pointer params."""

    def test_far_pointer_requires_x16_mode(self):
        """Function with far pointer stack param must use x16 mode."""
        source = """
            #[mode(m8, x8)]
            fn process(far *data: u8) {
            }
        """
        with pytest.raises(MIRLoweringError) as exc_info:
            build_mir(source)
        assert "x16" in str(exc_info.value)
        assert "far pointer" in str(exc_info.value).lower()

    def test_far_pointer_no_mode_attr_fails(self):
        """Function with far pointer stack param without mode attr fails."""
        source = """
            fn process(far *data: u8) {
            }
        """
        with pytest.raises(MIRLoweringError) as exc_info:
            build_mir(source)
        assert "x16" in str(exc_info.value)

    def test_far_pointer_with_x16_succeeds(self):
        """Function with far pointer and x16 mode should compile."""
        source = """
            #[mode(m8, x16)]
            fn process(far *data: u8) {
            }
        """
        # Should not raise
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_far_pointer_with_m16_x16_succeeds(self):
        """Function with far pointer and m16/x16 mode should compile."""
        source = """
            #[mode(m16, x16)]
            fn process(far *data: u16) {
            }
        """
        # Should not raise
        mir = build_mir(source)
        assert len(mir.functions) == 1


class TestFarPointerCodegen:
    """Tests for far pointer code generation."""

    def test_far_pointer_prologue_generated(self):
        """Function with far pointer should generate D = S prologue."""
        # This test verifies MIR is built correctly
        # Full codegen tests would need assembly output verification
        source = """
            #[mode(m8, x16)]
            fn process(far *data: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True


class TestMixedParams:
    """Tests for functions with mixed parameter types."""

    def test_far_and_near_pointers(self):
        """Function with both far and near pointer params."""
        source = """
            #[mode(m8, x16)]
            fn copy(far *src: u8, *dst: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True
        # Only the far pointer param should be in the set
        assert 0 in func.far_ptr_param_indices
        assert 1 not in func.far_ptr_param_indices

    def test_far_pointer_and_regular_params(self):
        """Function with far pointer and regular (non-pointer) params."""
        source = """
            #[mode(m8, x16)]
            fn process(far *data: u8, count: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True
        assert 0 in func.far_ptr_param_indices
        # count is not a far pointer
        assert 1 not in func.far_ptr_param_indices


class TestCallHandlingWithFarPointers:
    """Tests for D register management around function calls."""

    def test_function_with_call_and_far_pointer(self):
        """Function with far pointer param that makes a call."""
        source = """
            #[mode(m8, x16)]
            fn helper() {
            }

            #[mode(m8, x16)]
            fn process(far *data: u8) {
                helper();
            }
        """
        mir = build_mir(source)
        # Should have two functions: helper and process
        assert len(mir.functions) == 2
        # process should have far pointer flag set
        process_func = next(f for f in mir.functions if f.name == "process")
        assert process_func.has_far_ptr_stack_params is True

    def test_function_with_multiple_calls(self):
        """Function with far pointer param that makes multiple calls."""
        source = """
            #[mode(m8, x16)]
            fn helper1() {
            }

            #[mode(m8, x16)]
            fn helper2() {
            }

            #[mode(m8, x16)]
            fn process(far *data: u8) {
                helper1();
                helper2();
            }
        """
        mir = build_mir(source)
        # All functions should compile
        assert len(mir.functions) == 3
        process_func = next(f for f in mir.functions if f.name == "process")
        assert process_func.has_far_ptr_stack_params is True

    def test_nested_function_calls(self):
        """Far pointer function calling another far pointer function."""
        source = """
            #[mode(m8, x16)]
            fn inner(far *ptr: u8) {
            }

            #[mode(m8, x16)]
            fn outer(far *data: u8) {
                inner(data);
            }
        """
        # This should compile - both functions need far pointer handling
        mir = build_mir(source)
        assert len(mir.functions) == 2
        for func in mir.functions:
            assert func.has_far_ptr_stack_params is True
