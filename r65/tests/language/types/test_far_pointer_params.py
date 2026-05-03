# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for far pointer parameters using D = S technique."""

import pytest
from r65.compiler.errors import MIRLoweringError, TypeCheckError
from r65.tests.language.common import build_mir


class TestFarPointerStackParamDetection:
    """Tests for detecting far pointer stack parameters."""

    def test_far_pointer_stack_param_detected(self):
        """Function with far pointer stack param should have flag set."""
        source = """
                        fn process(data: far *u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True
        assert 0 in func.far_ptr_param_indices

    def test_near_pointer_stack_param_not_detected(self):
        """Function with near pointer stack param should not have flag set."""
        source = """
                        fn process(data: *u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is False

    def test_register_bound_far_pointer_rejected(self):
        """Far pointers cannot be bound to registers (they're 24-bit, registers are 8/16-bit)."""
        from r65.compiler.errors import HIRError
        source = """
                        fn process(data @ A: far *u8) {
            }
        """
        with pytest.raises(HIRError) as exc_info:
            build_mir(source)
        assert "far pointers are 24-bit and cannot fit in any register" in str(exc_info.value)

    def test_register_bound_near_pointer_x_accepted(self):
        """Near pointers are 16-bit and can be bound to X register."""
        source = """
            fn process(buffer @ X: *u8) {
            }
        """
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_register_bound_near_pointer_y_accepted(self):
        """Near pointers are 16-bit and can be bound to Y register."""
        source = """
            fn process(buffer @ Y: *u8) {
            }
        """
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_register_bound_near_pointer_a_accepted(self):
        """Near pointers are 16-bit and can be bound to A register (implies m16)."""
        source = """
            fn process(ptr @ A: *u8) {
            }
        """
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_multiple_far_pointer_params(self):
        """Multiple far pointer stack params should all be tracked."""
        source = """
                        fn copy(src: far *u8, dst: far *u8) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.has_far_ptr_stack_params is True
        assert 0 in func.far_ptr_param_indices
        assert 1 in func.far_ptr_param_indices


class TestFarFnPointerStackParamDetection:
    """Tests for tracking far function pointer stack parameters separately
    from far data pointer parameters. Far fn pointer params land in
    fn_ptr_param_indices; the cost-modeled strategy analysis decides whether
    to flip has_far_ptr_stack_params later."""

    def test_far_fn_pointer_stack_param_detected(self):
        """Function with far fn pointer stack param populates fn_ptr_param_indices."""
        source = """
            fn caller(handler: far fn(u8)) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert 0 in func.fn_ptr_param_indices
        # No far data pointer params, so this set stays empty.
        assert func.far_ptr_param_indices == set()
        # Strategy decision is deferred — the builder leaves the flag clear.
        assert func.has_far_ptr_stack_params is False

    def test_mixed_far_data_and_fn_pointer_params(self):
        """Mix of far *T and far fn(...) params populates both sets."""
        source = """
            fn caller(buf: far *u8, handler: far fn()) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert 0 in func.far_ptr_param_indices
        assert 1 in func.fn_ptr_param_indices
        # Data pointer param trips has_far_ptr_stack_params immediately
        assert func.has_far_ptr_stack_params is True

    def test_no_far_params_both_sets_empty(self):
        """Function with neither far pointer kind has both sets empty."""
        source = """
            fn caller(x: u8, y: u16) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.far_ptr_param_indices == set()
        assert func.fn_ptr_param_indices == set()
        assert func.has_far_ptr_stack_params is False

    def test_near_fn_pointer_not_tracked(self):
        """Near fn pointer params (no far) do not populate fn_ptr_param_indices."""
        source = """
            fn caller(handler: fn(u8)) {
            }
        """
        mir = build_mir(source)
        func = mir.functions[0]
        assert func.fn_ptr_param_indices == set()
        assert func.far_ptr_param_indices == set()


class TestFarPointerModeValidation:
    """Tests for far pointer params with automatic x16 mode.

    In the simplified mode system, X/Y registers are always 16-bit (x16 mode),
    so far pointer stack params always work without explicit mode attributes.
    """

    def test_far_pointer_always_succeeds(self):
        """Function with far pointer stack param succeeds (x16 is automatic)."""
        source = """
            fn process(data: far *u8) {
            }
        """
        mir = build_mir(source)
        assert len(mir.functions) == 1

    def test_far_pointer_u16_succeeds(self):
        """Function with far pointer to u16 should compile."""
        source = """
            fn process(data: far *u16) {
            }
        """
        mir = build_mir(source)
        assert len(mir.functions) == 1


class TestMixedParams:
    """Tests for functions with mixed parameter types."""

    def test_far_and_near_pointers(self):
        """Function with both far and near pointer params."""
        source = """
                        fn copy(src: far *u8, dst: *u8) {
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
                        fn process(data: far *u8, count: u8) {
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

                        fn process(data: far *u8) {
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

                        fn process(data: far *u8) {
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
                        fn inner(ptr: far *u8) {
            }

                        fn outer(data: far *u8) {
                inner(data);
            }
        """
        # This should compile - both functions need far pointer handling
        mir = build_mir(source)
        assert len(mir.functions) == 2
        for func in mir.functions:
            assert func.has_far_ptr_stack_params is True
