"""
Rule kind: ``forbidden_identifier``.

Bans references to specific resolved symbol names from function bodies.
Optionally scoped via ``name_pattern`` (regex over the enclosing function
name) or ``allow_in`` (exempt list of function names).

    [[rule]]
    code = "C021"
    kind = "forbidden_identifier"
    message = "`legacy_state` is deprecated; use GAME_STATE"
    symbols = ["legacy_state"]
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from r65.compiler.hir import HIRFunctionDecl, HIRIdentifier
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import (
    optional_list_of_str,
    parse_severity,
    require_key,
    require_list_of_str,
)


KIND_NAME = "forbidden_identifier"


class ForbiddenIdentifier(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        symbols: List[str],
        allow_in: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if not symbols:
            raise ValueError(f"rule kind '{KIND_NAME}': `symbols` must not be empty")
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.symbols: Set[str] = set(symbols)
        self.allow_in: Set[str] = set(allow_in or ())
        self.pattern = None
        if name_pattern is not None:
            try:
                self.pattern = re.compile(name_pattern)
            except re.error as e:
                raise ValueError(
                    f"rule kind '{KIND_NAME}': `name_pattern` is not a valid regex: {e}"
                ) from e
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def visit_identifier(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        self._check(expr, ctx)

    def visit_identifier_write(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        self._check(expr, ctx)

    def _check(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        if expr.name not in self.symbols:
            return
        func = ctx.current_function
        if func is None:
            return
        if func.name in self.allow_in:
            return
        if self.pattern is not None and not self.pattern.fullmatch(func.name):
            return
        ctx.emit(
            code=self.code,
            message=f"{self.message} (`{expr.name}`)",
            source_loc=expr.source_loc,
            hint=self.custom_hint,
            severity=self.severity,
        )


def from_config(spec: Dict[str, Any]) -> ForbiddenIdentifier:
    code = spec["code"]
    message = spec["message"]
    symbols = require_list_of_str(
        require_key(spec, "symbols", KIND_NAME), "symbols", KIND_NAME
    )
    allow_in = optional_list_of_str(spec, "allow_in", KIND_NAME)
    name_pattern = spec.get("name_pattern")
    if name_pattern is not None and not isinstance(name_pattern, str):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `name_pattern` must be a string"
        )
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return ForbiddenIdentifier(
        code=code,
        message=message,
        symbols=symbols,
        allow_in=allow_in,
        name_pattern=name_pattern,
        severity_name=severity,
        hint=hint,
    )
