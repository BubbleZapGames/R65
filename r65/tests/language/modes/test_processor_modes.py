"""Tests for processor mode annotations."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_function, get_attr, get_attr_arg_by_name, build_hir


class TestModeAttribute:
    """Tests for #[mode(...)] attribute."""

    def test_mode_m8(self):
        """Test m8 mode annotation."""
        func = parse_function("#[mode(m8)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_m16(self):
        """Test m16 mode annotation."""
        func = parse_function("#[mode(m16)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_x8(self):
        """Test x8 mode annotation."""
        func = parse_function("#[mode(x8)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_x16(self):
        """Test x16 mode annotation."""
        func = parse_function("#[mode(x16)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_combined(self):
        """Test combined m and x modes."""
        func = parse_function("#[mode(m8, x16)] fn test() { }")
        attr = get_attr(func, "mode")
        assert len(attr.args) >= 2


class TestModeTransition:
    """Tests for mode transition options."""

    def test_transition_none(self):
        """Test transition=none option."""
        func = parse_function("#[mode(m8, transition=none)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_transition_inline(self):
        """Test transition=inline option."""
        func = parse_function("#[mode(m8, transition=inline)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_transition_caller(self):
        """Test transition=caller option."""
        func = parse_function("#[mode(m16, transition=caller)] fn test() { }")
        attr = get_attr(func, "mode")
        assert attr is not None


class TestModeBuiltins:
    """Tests for mode control built-in functions."""

    def test_sep_builtin(self):
        """Test SEP() built-in."""
        func = parse_function("fn test() { SEP(0x30); }")
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)

    def test_rep_builtin(self):
        """Test REP() built-in."""
        func = parse_function("fn test() { REP(0x30); }")
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)

    def test_xba_builtin(self):
        """Test xba() built-in."""
        func = parse_function("fn test() { xba(); }")
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.ExprStmt)


class TestModeHIR:
    """Tests for mode HIR generation."""

    def test_mode_hir(self):
        """Test mode annotations generate proper HIR."""
        hir_prog = build_hir("""
            #[mode(m8, x8)]
            fn eight_bit() { A = 0xFF; }
        """)
        func = hir_prog.functions[0]
        assert func is not None
