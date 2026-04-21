# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for L009 xy16_mode lint rule."""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint


def lint_source(source: str):
    program = parse(source, '<test>')
    program = expand_macros(program)
    hir = HIRBuilder().build_program(program)
    # Built-in rules are opt-in — explicitly enable L009.
    config = LintConfig(enabled_codes={"L009"})
    return run_lint(hir, config)


def _codes(diagnostics):
    return [d.code for d in diagnostics.diagnostics]


class TestL009Xy16Mode:
    def test_xy16_false_emits_warning(self):
        # XY16 = false triggers L009; function also needs restore to avoid
        # E-XY16-REGION, but the lint pass runs before typeck so both paths
        # surface independently — lint just checks that the write is flagged.
        diags = lint_source('''
            fn leaf() {
                STATUS.XY16 = false;
                asm!("LDX #$10");
                STATUS.XY16 = true;
            }
        ''')
        assert "L009" in _codes(diags)

    def test_xy16_true_does_not_emit(self):
        diags = lint_source('''
            fn harmless() {
                STATUS.XY16 = true;
                asm!("NOP");
            }
        ''')
        assert "L009" not in _codes(diags)

    def test_no_xy16_write_does_not_emit(self):
        diags = lint_source('''
            fn normal(n: u8) -> u8 {
                return n;
            }
        ''')
        assert "L009" not in _codes(diags)
