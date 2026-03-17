# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for function parameters."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function, build_hir


class TestBasicParameters:
    """Tests for basic function parameters."""

    def test_single_parameter(self):
        """Test function with single parameter."""
        func = parse_function("fn process(x: u8) { }")
        assert len(func.params) == 1
        assert func.params[0].name == "x"
        assert func.params[0].param_type.name == "u8"

    def test_multiple_parameters(self):
        """Test function with multiple parameters."""
        func = parse_function("fn add(a: u8, b: u8, c: u16) { }")
        assert len(func.params) == 3
        assert func.params[0].param_type.name == "u8"
        assert func.params[2].param_type.name == "u16"


class TestRegisterParameters:
    """Tests for register-aliased parameters."""

    def test_register_parameter(self):
        """Test parameter with register alias."""
        func = parse_function("fn process(value @ A: u8) { }")
        assert len(func.params) == 1
        param = func.params[0]
        assert param.name == "value"
        assert param.binding is not None
        assert param.binding.name == "A"

    def test_multiple_register_params(self):
        """Test multiple register parameters."""
        func = parse_function("fn coords(x @ X: u16, y @ Y: u16) { }")
        assert len(func.params) == 2
        assert func.params[0].binding.name == "X"
        assert func.params[1].binding.name == "Y"

    def test_mixed_parameters(self):
        """Test mix of stack and register parameters."""
        func = parse_function("fn mixed(stack_param: u8, reg_param @ A: u8) { }")
        assert len(func.params) == 2
        assert func.params[0].binding is None  # Stack parameter
        assert func.params[1].binding is not None  # Register parameter


class TestParameterHIR:
    """Tests for parameter HIR generation."""

    def test_parameter_hir(self):
        """Test parameters generate proper HIR."""
        hir_prog = build_hir("""
            fn process(x: u8, y @ A: u8) -> u8 { return A; }
        """)
        func = hir_prog.functions[0]
        # HIR uses 'parameters' not 'params'
        assert len(func.parameters) == 2
