"""
Rule kind: ``enforce_storage_class``.

Catches ``static mut`` declarations whose explicit address falls in a
configured range but which don't carry the required storage attribute.
Typical use case: every decl in ``$2100-$43FF`` must be ``#[hw]`` so the
compiler knows to treat accesses as volatile::

    [[rule]]
    code = "C020"
    kind = "enforce_storage_class"
    message = "Hardware address space must use #[hw]"
    addr_range    = { start = 0x2100, end = 0x4400 }
    required_attr = "#[hw]"

Auto-allocated decls (no explicit address in the attribute) are skipped —
their final address isn't known until codegen.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from r65.compiler.hir import HIRStaticDecl
from r65.compiler.hir.attributes import StorageKind
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import (
    optional_addr_range,
    parse_severity,
    require_key,
)


KIND_NAME = "enforce_storage_class"


_STORAGE_KIND_BY_NAME = {
    "#[zeropage]": StorageKind.ZEROPAGE,
    "#[lowram]": StorageKind.LOWRAM,
    "#[ram]": StorageKind.RAM,
    "#[hw]": StorageKind.HW,
}


class EnforceStorageClass(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        addr_start: int,
        addr_end: int,
        required_kind: StorageKind,
        required_raw: str,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.addr_start = addr_start
        self.addr_end = addr_end
        self.required_kind = required_kind
        self.required_raw = required_raw
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def visit_static_decl(self, decl: HIRStaticDecl, ctx: LintContext) -> None:
        storage = decl.storage_attr
        if storage is None:
            return
        addr = getattr(storage, "address", None)
        if addr is None:
            return  # Auto-allocated; skip.
        if not (self.addr_start <= addr < self.addr_end):
            return
        if storage.storage_kind == self.required_kind:
            return
        ctx.emit(
            code=self.code,
            message=(
                f"{self.message} (`{decl.name}` at ${addr:04X} uses "
                f"#[{storage.storage_kind.value}], expected {self.required_raw})"
            ),
            source_loc=decl.source_loc,
            hint=self.custom_hint,
            severity=self.severity,
        )


def from_config(spec: Dict[str, Any]) -> EnforceStorageClass:
    code = spec["code"]
    message = spec["message"]
    addr_range = optional_addr_range(spec, "addr_range", KIND_NAME)
    if addr_range is None:
        raise ValueError(
            f"rule kind '{KIND_NAME}': `addr_range` is required"
        )
    required_raw = require_key(spec, "required_attr", KIND_NAME)
    if not isinstance(required_raw, str):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `required_attr` must be a string"
        )
    required_kind = _STORAGE_KIND_BY_NAME.get(required_raw.strip())
    if required_kind is None:
        raise ValueError(
            f"rule kind '{KIND_NAME}': `required_attr` must be one of "
            f"{sorted(_STORAGE_KIND_BY_NAME)}, got `{required_raw}`"
        )
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return EnforceStorageClass(
        code=code,
        message=message,
        addr_start=addr_range[0],
        addr_end=addr_range[1],
        required_kind=required_kind,
        required_raw=required_raw.strip(),
        severity_name=severity,
        hint=hint,
    )
