# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for processor mode annotations.

The #[mode] attribute now only supports the databank parameter.
CPU mode (m8/m16, x8/x16) is automatically inferred from parameter types:
- Default: m8 (8-bit A), x16 (16-bit X/Y)
- u16 @ A parameter -> m16 entry mode
- X/Y registers are always u16
"""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.hir.errors import HIRError
from r65.tests.language.common import parse_function, get_attr, get_attr_arg_by_name, build_hir


class TestModeAttributeParsing:
    """Tests for #[mode(...)] attribute parsing.

    Parsing still accepts the old syntax (for better error messages at HIR stage).
    """

    def test_mode_parses_m8(self):
        """Test m8 mode annotation parses (but will fail at HIR)."""
        func = parse_function("#[mode(m8)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_parses_databank(self):
        """Test databank=inline mode annotation parses."""
        func = parse_function("#[mode(databank=inline)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None


class TestModeAttributeRejection:
    """Tests that old m8/m16/x8/x16/transition syntax is rejected at HIR level."""

    def test_mode_m8_rejected(self):
        """Test m8 mode is rejected at HIR level."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[mode(m8)] fn test() { }")
        assert "no longer supported" in str(exc_info.value)
        assert "m8" in str(exc_info.value)

    def test_mode_m16_rejected(self):
        """Test m16 mode is rejected at HIR level."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[mode(m16)] fn test() { }")
        assert "no longer supported" in str(exc_info.value)
        assert "m16" in str(exc_info.value)

    def test_mode_x8_rejected(self):
        """Test x8 mode is rejected at HIR level."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[mode(x8)] fn test() { }")
        assert "no longer supported" in str(exc_info.value)

    def test_mode_x16_rejected(self):
        """Test x16 mode is rejected at HIR level."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[mode(x16)] fn test() { }")
        assert "no longer supported" in str(exc_info.value)

    def test_mode_combined_rejected(self):
        """Test combined m8, x16 is rejected at HIR level."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[mode(m8, x16)] fn test() { }")
        assert "no longer supported" in str(exc_info.value)

    def test_transition_rejected(self):
        """Test transition= parameter is rejected at HIR level."""
        with pytest.raises(HIRError) as exc_info:
            build_hir("#[mode(transition=inline)] fn test() { }")
        assert "no longer supported" in str(exc_info.value)
        assert "transition" in str(exc_info.value)


class TestModeDatabank:
    """Tests for the databank parameter (the only remaining mode parameter)."""

    def test_databank_none(self):
        """Test databank=none (default)."""
        hir_prog = build_hir("fn test() { }")
        func = hir_prog.functions[0]
        # No mode attribute means default databank=none
        assert func is not None

    def test_databank_inline(self):
        """Test databank=inline for far functions."""
        hir_prog = build_hir("#[mode(databank=inline)] far fn test() { }")
        func = hir_prog.functions[0]
        assert func.mode_attr is not None
        from r65.compiler.hir.attributes import DataBankMode
        assert func.mode_attr.databank == DataBankMode.INLINE

    def test_databank_caller(self):
        """Test databank=caller for far functions."""
        hir_prog = build_hir("#[mode(databank=caller)] far fn test() { }")
        func = hir_prog.functions[0]
        assert func.mode_attr is not None
        from r65.compiler.hir.attributes import DataBankMode
        assert func.mode_attr.databank == DataBankMode.CALLER


class TestModeInference:
    """Tests for automatic mode inference from parameter and return types."""

    def test_default_entry_mode_is_m8(self):
        """Test that default entry mode is m8 (no u16 A parameter)."""
        hir_prog = build_hir("fn test() { }")
        func = hir_prog.functions[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M8

    def test_u8_a_parameter_is_m8(self):
        """Test that u8 @ A parameter results in m8 entry mode."""
        hir_prog = build_hir("fn test(val @ A: u8) { }")
        func = hir_prog.functions[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M8

    def test_u16_a_parameter_is_m16(self):
        """Test that u16 @ A parameter results in m16 entry mode."""
        hir_prog = build_hir("fn test(val @ A: u16) { }")
        func = hir_prog.functions[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M16

    def test_i16_a_parameter_is_m16(self):
        """Test that i16 @ A parameter also results in m16 entry mode."""
        hir_prog = build_hir("fn test(val @ A: i16) { }")
        func = hir_prog.functions[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.entry_m_mode == ModeState.M16

    def test_x_parameter_u16_allowed(self):
        """Test that u16 @ X parameter is allowed."""
        hir_prog = build_hir("fn test(idx @ X: u16) { }")
        func = hir_prog.functions[0]
        assert func is not None  # Should compile without error

    def test_y_parameter_u16_allowed(self):
        """Test that u16 @ Y parameter is allowed."""
        hir_prog = build_hir("fn test(idx @ Y: u16) { }")
        func = hir_prog.functions[0]
        assert func is not None  # Should compile without error

    def test_x_parameter_u8_rejected(self):
        """Test that u8 @ X parameter is rejected (X is always 16-bit)."""
        from r65.compiler.hir.errors import HIRError
        with pytest.raises(HIRError) as exc_info:
            build_hir("fn test(idx @ X: u8) { }")
        assert "X only supports: u16, i16" in str(exc_info.value)

    def test_y_parameter_u8_rejected(self):
        """Test that u8 @ Y parameter is rejected (Y is always 16-bit)."""
        from r65.compiler.hir.errors import HIRError
        with pytest.raises(HIRError) as exc_info:
            build_hir("fn test(idx @ Y: u8) { }")
        assert "Y only supports: u16, i16" in str(exc_info.value)

    def test_exit_mode_u8_return_is_m8(self):
        """Test that u8 return type results in m8 exit mode."""
        hir_prog = build_hir("fn test() -> u8 { return 0; }")
        func = hir_prog.functions[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.exit_m_mode == ModeState.M8

    def test_exit_mode_u16_return_is_m16(self):
        """Test that u16 return type results in m16 exit mode."""
        hir_prog = build_hir("fn test() -> u16 { return 0; }")
        func = hir_prog.functions[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.exit_m_mode == ModeState.M16

    def test_exit_mode_void_return_is_m8(self):
        """Test that void (no return type) results in m8 exit mode."""
        hir_prog = build_hir("fn test() { }")
        func = hir_prog.functions[0]
        from r65.compiler.typeck.processor_mode import ModeState
        assert func.exit_m_mode == ModeState.M8


