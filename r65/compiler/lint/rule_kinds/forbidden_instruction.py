"""
Rule kind: ``forbidden_instruction``.

Walks every ``HIRAsmStmt`` and flags any occurrence of a configured mnemonic
(case-insensitive). Escape hatch via ``allow_in``:

    [[rule]]
    code = "C011"
    kind = "forbidden_instruction"
    message = "CLI is only allowed in the boot path"
    severity = "error"
    mnemonics = ["CLI"]
    allow_in  = ["boot_init", "late_init"]
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from r65.compiler.hir import HIRAsmStmt
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import (
    optional_list_of_str,
    parse_severity,
    require_key,
    require_list_of_str,
)


KIND_NAME = "forbidden_instruction"

# Split asm lines into tokens by whitespace/commas/semicolons (comments).
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ForbiddenInstruction(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        mnemonics: List[str],
        allow_in: Optional[List[str]] = None,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if not mnemonics:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `mnemonics` must not be empty"
            )
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.mnemonics: Set[str] = {m.upper() for m in mnemonics}
        self.allow_in: Set[str] = set(allow_in or ())
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def visit_asm(self, stmt: HIRAsmStmt, ctx: LintContext) -> None:
        if ctx.current_function is not None and ctx.current_function.name in self.allow_in:
            return
        # Strip line comments (; ...) then tokenize; the first identifier-like
        # token on the line is conventionally the mnemonic for 65816 asm.
        for raw_line in stmt.instructions:
            line = raw_line.split(";", 1)[0]
            match = _TOKEN_RE.search(line)
            if match is None:
                continue
            mnemonic = match.group(0).upper()
            if mnemonic in self.mnemonics:
                ctx.emit(
                    code=self.code,
                    message=f"{self.message} (`{mnemonic}`)",
                    source_loc=stmt.source_loc,
                    hint=self.custom_hint,
                    severity=self.severity,
                )


def from_config(spec: Dict[str, Any]) -> ForbiddenInstruction:
    code = spec["code"]
    message = spec["message"]
    mnemonics = require_list_of_str(
        require_key(spec, "mnemonics", KIND_NAME), "mnemonics", KIND_NAME
    )
    allow_in = optional_list_of_str(spec, "allow_in", KIND_NAME)
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return ForbiddenInstruction(
        code=code,
        message=message,
        mnemonics=mnemonics,
        allow_in=allow_in,
        severity_name=severity,
        hint=hint,
    )
