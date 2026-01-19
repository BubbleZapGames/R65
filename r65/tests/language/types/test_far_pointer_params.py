"""Tests for far pointer parameters using D = S technique."""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder
from r65.compiler.errors import MIRLoweringError, TypeCheckError


def build_mir(source: str):
    """Parse source and build MIR."""
    program = parse(source, "test.r65")
    hir_builder = HIRBuilder(source_file="test.r65")
    hir_prog = hir_builder.build_program(program)
    type_checker = TypeChecker(hir_prog)
    type_checker.check()
    mir_builder = MIRBuilder()
    return mir_builder.build_program(hir_prog)


class TestFarPointerStackParamDetection:
    """Tests for detecting far pointer stack parameters."""

    def test_far_pointer_stack_param_detected(self):
        """Function with far pointer stack param should have flag set."""
        source = """
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
                        fn process(*data: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is False

    def test_register_bound_far_pointer_not_detected(self):
        """Register-bound far pointer should not set stack param flag."""
        source = """
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
                        fn copy(far *src: u8, far *dst: u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True
        assert 0 in func.far_ptr_param_indices
        assert 1 in func.far_ptr_param_indices


class TestFarPointerModeValidation:
    """Tests for far pointer params with automatic x16 mode.

    In the simplified mode system, X/Y registers are always 16-bit (x16 mode),
    so far pointer stack params always work without explicit mode attributes.
    """

    def test_far_pointer_always_succeeds(self):
        """Function with far pointer stack param succeeds (x16 is automatic)."""
        source = """
            fn process(far *data: u8) {
            }
        """
        # Should succeed - X/Y are always 16-bit in the new design
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_far_pointer_no_mode_attr_succeeds(self):
        """Function with far pointer stack param without mode attr succeeds."""
        source = """
            fn process(far *data: u8) {
            }
        """
        # Should succeed - X/Y are always 16-bit in the new design
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_far_pointer_with_x16_succeeds(self):
        """Function with far pointer and x16 mode should compile."""
        source = """
                        fn process(far *data: u8) {
            }
        """
        # Should not raise
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_far_pointer_with_m16_x16_succeeds(self):
        """Function with far pointer and m16/x16 mode should compile."""
        source = """
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
                        fn helper() {
            }

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
                        fn helper1() {
            }

                        fn helper2() {
            }

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
                        fn inner(far *ptr: u8) {
            }

                        fn outer(far *data: u8) {
                inner(data);
            }
        """
        # This should compile - both functions need far pointer handling
        mir = build_mir(source)
        assert len(mir.functions) == 2
        for func in mir.functions:
            assert func.has_far_ptr_stack_params is True
