"""Tests for ABIModel abstraction."""

import pytest
from r65.compiler.codegen.abi_model import (
    ABIModel, ABIKind,
    ABI_DEFAULT, ABI_FIXED_STACK, abi_model_from_string,
)


class TestABIKind:
    def test_default_value(self):
        assert ABIKind.DEFAULT.value == "Default"

    def test_fixed_stack_value(self):
        assert ABIKind.FIXED_STACK.value == "FixedStack"


class TestABIModelDefault:
    def test_kind(self):
        assert ABI_DEFAULT.kind == ABIKind.DEFAULT

    def test_allows_stack_params(self):
        assert ABI_DEFAULT.allows_stack_params() is True

    def test_uses_tsc_frame(self):
        assert ABI_DEFAULT.uses_tsc_frame() is True

    def test_requires_mandatory_param_promotion(self):
        assert ABI_DEFAULT.requires_mandatory_param_promotion() is False

    def test_repr(self):
        assert "Default" in repr(ABI_DEFAULT)


class TestABIModelFixedStack:
    def test_kind(self):
        assert ABI_FIXED_STACK.kind == ABIKind.FIXED_STACK

    def test_allows_stack_params(self):
        assert ABI_FIXED_STACK.allows_stack_params() is False

    def test_uses_tsc_frame(self):
        assert ABI_FIXED_STACK.uses_tsc_frame() is False

    def test_requires_mandatory_param_promotion(self):
        assert ABI_FIXED_STACK.requires_mandatory_param_promotion() is True

    def test_repr(self):
        assert "FixedStack" in repr(ABI_FIXED_STACK)


class TestAbiModelFromString:
    def test_default(self):
        model = abi_model_from_string("Default")
        assert model.kind == ABIKind.DEFAULT

    def test_fixed_stack(self):
        model = abi_model_from_string("FixedStack")
        assert model.kind == ABIKind.FIXED_STACK

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown ABI model"):
            abi_model_from_string("invalid")

    def test_singleton_default(self):
        assert abi_model_from_string("Default") is ABI_DEFAULT

    def test_singleton_fixed_stack(self):
        assert abi_model_from_string("FixedStack") is ABI_FIXED_STACK
