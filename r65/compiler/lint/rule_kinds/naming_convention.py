"""
Rule kind: ``naming_convention``.

Regex enforcement on identifier names grouped by declaration kind::

    [[rule]]
    code = "C022"
    kind = "naming_convention"
    message = "statics must be SCREAMING_SNAKE_CASE"
    target  = "statics"
    pattern = "^[A-Z][A-Z0-9_]*$"
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from r65.compiler.hir import (
    HIRConstDecl,
    HIREnumDecl,
    HIRFunctionDecl,
    HIRStaticDecl,
    HIRStructDecl,
)
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import parse_severity, require_key


KIND_NAME = "naming_convention"

_TARGET_TO_CLASS = {
    "statics": HIRStaticDecl,
    "functions": HIRFunctionDecl,
    "constants": HIRConstDecl,
    "enums": HIREnumDecl,
    "structs": HIRStructDecl,
}


class NamingConvention(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        target: str,
        pattern: str,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if target not in _TARGET_TO_CLASS:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `target` must be one of "
                f"{sorted(_TARGET_TO_CLASS)}, got `{target}`"
            )
        try:
            self.regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `pattern` is not a valid regex: {e}"
            ) from e
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.target = target
        self.target_class = _TARGET_TO_CLASS[target]
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def setup(self, ctx: LintContext) -> None:
        for decl in ctx.program.declarations:
            if not isinstance(decl, self.target_class):
                continue
            name = getattr(decl, "name", None)
            if not name:
                continue
            if self.regex.fullmatch(name):
                continue
            ctx.emit(
                code=self.code,
                message=f"{self.message} (`{name}` does not match `{self.regex.pattern}`)",
                source_loc=decl.source_loc,
                hint=self.custom_hint,
                severity=self.severity,
            )


def from_config(spec: Dict[str, Any]) -> NamingConvention:
    code = spec["code"]
    message = spec["message"]
    target = require_key(spec, "target", KIND_NAME)
    if not isinstance(target, str):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `target` must be a string"
        )
    pattern = require_key(spec, "pattern", KIND_NAME)
    if not isinstance(pattern, str):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `pattern` must be a string"
        )
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return NamingConvention(
        code=code,
        message=message,
        target=target,
        pattern=pattern,
        severity_name=severity,
        hint=hint,
    )
