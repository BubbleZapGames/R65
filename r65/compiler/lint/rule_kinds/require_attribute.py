"""
Rule kind: ``require_attribute``.

Enforces that functions matching a regex pattern carry a required attribute::

    [[rule]]
    code = "C012"
    kind = "require_attribute"
    message = "NMI-named functions must carry #[interrupt(nmi)]"
    name_pattern = ".*_nmi"
    required     = ["#[interrupt(nmi)]"]

Supported attribute specs (matched against fields on ``HIRFunctionDecl``):

    #[entry]
    #[interrupt(nmi|irq|brk|cop|abort)]
    #[bank(N)] | #[bank(auto)]
    #[preserves(A, X, ...)]
    #[inline] | #[inline(always)] | #[inline(never)]
    #[mode(databank=none|inline|caller)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from r65.compiler.hir import HIRFunctionDecl
from r65.compiler.hir.attributes import (
    DataBankMode,
    InlineMode,
    InterruptVector,
)
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import (
    parse_severity,
    require_key,
    require_list_of_str,
)


KIND_NAME = "require_attribute"

_ATTR_RE = re.compile(r"^\s*#\[\s*(\w+)\s*(?:\(\s*(.*?)\s*\))?\s*\]\s*$")


@dataclass
class _AttrSpec:
    """A parsed attribute requirement."""

    name: str
    args: List[str]  # Empty for #[entry], one element for #[interrupt(nmi)], etc.
    raw: str

    def __str__(self) -> str:
        return self.raw


def _parse_attr_spec(raw: str) -> _AttrSpec:
    match = _ATTR_RE.match(raw)
    if match is None:
        raise ValueError(
            f"rule kind '{KIND_NAME}': attribute spec `{raw}` is not a valid "
            f"#[name] or #[name(args)] form"
        )
    name = match.group(1)
    args_raw = match.group(2) or ""
    args = [a.strip() for a in args_raw.split(",") if a.strip()] if args_raw else []
    return _AttrSpec(name=name, args=args, raw=raw.strip())


def _func_has_attr(func: HIRFunctionDecl, spec: _AttrSpec) -> bool:
    name = spec.name
    if name == "entry":
        return bool(getattr(func, "is_entry", False))

    if name == "interrupt":
        attr = func.interrupt_attr
        if attr is None:
            return False
        if not spec.args:
            return True  # Any interrupt vector
        want = spec.args[0]
        return attr.vector.value == want

    if name == "bank":
        attr = func.bank_attr
        if attr is None:
            return False
        if not spec.args:
            return True
        want = spec.args[0]
        if want == "auto":
            return attr.is_auto
        try:
            want_num = int(want, 0)
        except ValueError:
            return False
        return attr.bank_number == want_num

    if name == "preserves":
        attr = func.preserves_attr
        if attr is None:
            return False
        if not spec.args:
            return True
        required_regs = {a.upper() for a in spec.args}
        actual_regs = {r.upper() for r in attr.registers}
        return required_regs.issubset(actual_regs)

    if name == "inline":
        attr = func.inline_attr
        if attr is None:
            return False
        if not spec.args:
            return True
        want = spec.args[0]
        try:
            want_mode = InlineMode(want)
        except ValueError:
            return False
        return attr.mode == want_mode

    if name == "mode":
        attr = func.mode_attr
        if attr is None:
            return False
        for arg in spec.args:
            if "=" not in arg:
                continue
            key, value = [s.strip() for s in arg.split("=", 1)]
            if key == "databank":
                try:
                    want = DataBankMode(value)
                except ValueError:
                    return False
                if attr.databank != want:
                    return False
        return True

    raise ValueError(
        f"rule kind '{KIND_NAME}': unknown attribute `#[{name}]` in `required` "
        f"(supported: entry, interrupt, bank, preserves, inline, mode)"
    )


class RequireAttribute(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        name_pattern: str,
        required: List[_AttrSpec],
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        try:
            self.pattern = re.compile(name_pattern)
        except re.error as e:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `name_pattern` is not a valid regex: {e}"
            ) from e
        self.required = required
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def enter_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        if not self.pattern.fullmatch(func.name):
            return
        missing = [spec for spec in self.required if not _func_has_attr(func, spec)]
        if not missing:
            return
        missing_str = ", ".join(str(spec) for spec in missing)
        ctx.emit(
            code=self.code,
            message=f"{self.message} (`{func.name}` is missing: {missing_str})",
            source_loc=func.source_loc,
            hint=self.custom_hint,
            severity=self.severity,
        )


def from_config(spec: Dict[str, Any]) -> RequireAttribute:
    code = spec["code"]
    message = spec["message"]
    name_pattern = require_key(spec, "name_pattern", KIND_NAME)
    if not isinstance(name_pattern, str):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `name_pattern` must be a string"
        )
    required_raw = require_list_of_str(
        require_key(spec, "required", KIND_NAME), "required", KIND_NAME
    )
    required = [_parse_attr_spec(r) for r in required_raw]
    # Validate supported attribute names by probing each spec against a
    # throwaway decl — surfaces unknown names at config load time.
    probe = HIRFunctionDecl()
    for r in required:
        _func_has_attr(probe, r)
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return RequireAttribute(
        code=code,
        message=message,
        name_pattern=name_pattern,
        required=required,
        severity_name=severity,
        hint=hint,
    )
