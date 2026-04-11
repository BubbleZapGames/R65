"""
Rule kind: ``zeropage_budget``.

Sums declared ``#[zeropage]`` usage (bytes) and emits a single diagnostic
when the total exceeds a budget. Optional sub-cap for scratch registers
(decls declared with the ``register`` flag).

    [[rule]]
    code = "C024"
    kind = "zeropage_budget"
    message = "zeropage footprint exceeds project budget"
    max_bytes          = 96
    max_register_bytes = 16   # optional
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from r65.compiler.hir import HIRStaticDecl
from r65.compiler.hir.attributes import StorageKind
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import parse_severity, require_key


KIND_NAME = "zeropage_budget"


class ZeropageBudget(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        max_bytes: int,
        max_register_bytes: Optional[int] = None,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if max_bytes < 0:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `max_bytes` must be >= 0"
            )
        if max_register_bytes is not None and max_register_bytes < 0:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `max_register_bytes` must be >= 0"
            )
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.max_bytes = max_bytes
        self.max_register_bytes = max_register_bytes
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def setup(self, ctx: LintContext) -> None:
        total = 0
        reg_total = 0
        contributors: List[Tuple[str, int]] = []
        reg_contributors: List[Tuple[str, int]] = []

        for decl in ctx.program.declarations:
            if not isinstance(decl, HIRStaticDecl):
                continue
            storage = decl.storage_attr
            if storage is None or storage.storage_kind != StorageKind.ZEROPAGE:
                continue
            size = self._type_size(decl)
            if size <= 0:
                continue
            total += size
            contributors.append((decl.name, size))
            if getattr(storage, "is_register", False):
                reg_total += size
                reg_contributors.append((decl.name, size))

        if total > self.max_bytes:
            top = sorted(contributors, key=lambda kv: kv[1], reverse=True)[:5]
            hint = self.custom_hint or (
                "top contributors: "
                + ", ".join(f"{name} ({size}B)" for name, size in top)
            )
            ctx.emit(
                code=self.code,
                message=(
                    f"{self.message} ({total} bytes declared, budget "
                    f"{self.max_bytes} bytes)"
                ),
                source_loc=None,
                hint=hint,
                severity=self.severity,
            )
        if self.max_register_bytes is not None and reg_total > self.max_register_bytes:
            top = sorted(reg_contributors, key=lambda kv: kv[1], reverse=True)[:5]
            hint = self.custom_hint or (
                "top scratch registers: "
                + ", ".join(f"{name} ({size}B)" for name, size in top)
            )
            ctx.emit(
                code=self.code,
                message=(
                    f"{self.message} (scratch registers: {reg_total} bytes "
                    f"declared, budget {self.max_register_bytes} bytes)"
                ),
                source_loc=None,
                hint=hint,
                severity=self.severity,
            )

    def _type_size(self, decl: HIRStaticDecl) -> int:
        try:
            return decl.var_type.size_bytes
        except (AttributeError, NotImplementedError):
            return 0


def from_config(spec: Dict[str, Any]) -> ZeropageBudget:
    code = spec["code"]
    message = spec["message"]
    max_bytes = require_key(spec, "max_bytes", KIND_NAME)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `max_bytes` must be an integer"
        )
    max_register_bytes = spec.get("max_register_bytes")
    if max_register_bytes is not None:
        if not isinstance(max_register_bytes, int) or isinstance(max_register_bytes, bool):
            raise ValueError(
                f"rule kind '{KIND_NAME}': `max_register_bytes` must be an integer"
            )
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return ZeropageBudget(
        code=code,
        message=message,
        max_bytes=max_bytes,
        max_register_bytes=max_register_bytes,
        severity_name=severity,
        hint=hint,
    )
