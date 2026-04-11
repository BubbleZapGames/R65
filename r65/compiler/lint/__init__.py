"""
R65 linter package.

Provides a pluggable lint pass that runs after type checking and before MIR
construction. Rules walk the HIR once via :class:`R65Linter` and emit
:class:`Diagnostic` objects into a shared :class:`DiagnosticCollector`.

Public API:
    run_lint(program, enabled_codes=None) -> DiagnosticCollector
"""

from r65.compiler.lint.config import (
    LintConfig,
    LintConfigError,
    default_config,
    discover_config,
    load_config,
)
from r65.compiler.lint.linter import R65Linter, run_lint
from r65.compiler.lint.rule import LintContext, LintRule

__all__ = [
    "LintRule",
    "LintContext",
    "R65Linter",
    "run_lint",
    "LintConfig",
    "LintConfigError",
    "default_config",
    "discover_config",
    "load_config",
]
