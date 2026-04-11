"""
Rule kind: ``forbidden_value_at_register``.

Flags assignments to specific hardware register addresses where the constant
RHS matches a forbidden set. Built for the canonical SNES case: detecting
when a DMA channel's B-bus address (``BBADx`` at ``$43x1``) is configured to
target a PPU port like ``$2118`` (VMDATAL), which means the next ``MDMAEN``
trigger will write to PPU memory — unsafe outside vblank.

Config shape::

    [[rule]]
    code = "C002"
    kind = "forbidden_value_at_register"
    message = "DMA channel BBAD set to a PPU port"
    severity = "warning"
    entry_points     = ["main"]                # required
    exclude_subtrees = ["nmi_handler"]         # optional
    target_addrs = [0x4301, 0x4311, 0x4321, 0x4331,
                    0x4341, 0x4351, 0x4361, 0x4371]
    forbid_values = [0x18, 0x19, 0x22]         # PPU register low bytes

Limitations:
  - Only literal RHS values are checked. ``BBAD0 = some_var`` is silent.
  - HDMA setup uses the same ``BBADx`` registers as DMA, so HDMA-to-PPU
    setup will trigger false positives. Add HDMA helper functions to
    ``exclude_subtrees`` to silence them, same as ``snes_init`` is for
    boot init.
  - Compound assignments (``BBAD0 |= 0x18``) desugar to a binop and are
    silent. Use a plain ``=`` to land in the rule.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from r65.compiler.hir import (
    HIRAssignment,
    HIRFunctionDecl,
    HIRIdentifier,
    HIRIntegerLiteral,
    HIRStaticDecl,
)
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import (
    optional_list_of_int,
    optional_list_of_str,
    parse_severity,
    require_key,
    require_list_of_str,
)


KIND_NAME = "forbidden_value_at_register"


class ForbiddenValueAtRegister(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        entry_points: List[str],
        target_addrs: List[int],
        forbid_values: List[int],
        exclude_subtrees: Iterable[str] = (),
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if not entry_points:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `entry_points` must not be empty"
            )
        if not target_addrs:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `target_addrs` must not be empty"
            )
        if not forbid_values:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `forbid_values` must not be empty"
            )
        for v in forbid_values:
            if not (0 <= v <= 0xFF):
                raise ValueError(
                    f"rule kind '{KIND_NAME}': `forbid_values` entries must be "
                    f"byte-sized (0..255), got 0x{v:X}"
                )

        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.entry_points = list(entry_points)
        self.exclude_subtrees = list(exclude_subtrees)
        self.target_addrs: Set[int] = set(target_addrs)
        self.forbid_values: Set[int] = {v & 0xFF for v in forbid_values}
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

        # symbol id → address (filled in setup)
        self._target_sym_addr: Dict[int, int] = {}
        self._reachable: Set[str] = set()
        self._current_reachable = False

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: LintContext) -> None:
        for decl in ctx.program.declarations:
            if not isinstance(decl, HIRStaticDecl):
                continue
            if decl.symbol is None or decl.storage_attr is None:
                continue
            addr = getattr(decl.storage_attr, "address", None)
            if addr is None or addr not in self.target_addrs:
                continue
            self._target_sym_addr[id(decl.symbol)] = addr
        cg = ctx.call_graph
        self._reachable = cg.reachable_from(
            self.entry_points, exclude=self.exclude_subtrees
        )

    # -------------------------------------------------------------- per-func

    def enter_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        self._current_reachable = func.name in self._reachable

    def leave_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        self._current_reachable = False

    def visit_assignment(self, expr: HIRAssignment, ctx: LintContext) -> None:
        if not self._current_reachable:
            return
        target = expr.target
        if not isinstance(target, HIRIdentifier) or target.symbol is None:
            return
        addr = self._target_sym_addr.get(id(target.symbol))
        if addr is None:
            return
        value = expr.value
        if not isinstance(value, HIRIntegerLiteral):
            return
        masked = value.value & 0xFF
        if masked not in self.forbid_values:
            return
        ctx.emit(
            code=self.code,
            message=(
                f"{self.message} "
                f"(`{target.name}` (${addr:04X}) = ${masked:02X})"
            ),
            source_loc=expr.source_loc or target.source_loc,
            hint=self._build_hint(ctx),
            severity=self.severity,
        )

    def _build_hint(self, ctx: LintContext) -> Optional[str]:
        if self.custom_hint:
            return self.custom_hint
        if ctx.current_function is None:
            return None
        target = ctx.current_function.name
        cg = ctx.call_graph
        best_path = None
        for entry in self.entry_points:
            path = cg.shortest_path(entry, target)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path
        if best_path is None:
            return None
        return "reachable via: " + " → ".join(best_path)


# ----------------------------------------------------------------- factory

def from_config(spec: Dict[str, Any]) -> ForbiddenValueAtRegister:
    code = spec["code"]
    message = spec["message"]
    entry_points = require_list_of_str(
        require_key(spec, "entry_points", KIND_NAME),
        "entry_points",
        KIND_NAME,
    )
    exclude_subtrees = optional_list_of_str(spec, "exclude_subtrees", KIND_NAME)
    target_addrs = optional_list_of_int(spec, "target_addrs", KIND_NAME)
    if not target_addrs:
        raise ValueError(
            f"rule kind '{KIND_NAME}': `target_addrs` is required"
        )
    forbid_values = optional_list_of_int(spec, "forbid_values", KIND_NAME)
    if not forbid_values:
        raise ValueError(
            f"rule kind '{KIND_NAME}': `forbid_values` is required"
        )
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return ForbiddenValueAtRegister(
        code=code,
        message=message,
        entry_points=entry_points,
        target_addrs=target_addrs,
        forbid_values=forbid_values,
        exclude_subtrees=exclude_subtrees,
        severity_name=severity,
        hint=hint,
    )
