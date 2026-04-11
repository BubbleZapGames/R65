"""
Rule kind: ``reachability_forbidden_access``.

Flags accesses to a forbidden set of symbols (by name, by sparse address
list, or by ``#[hw]`` address range) from any function transitively
reachable from configured entry points, with subtree exclusions for legal
contexts.

Canonical use case: *"Writing ``$2118`` (VMDATAL) during active rendering
corrupts VRAM; flag any such write not reachable only from the NMI
handler."*

Config shape::

    [[rule]]
    code = "C001"
    kind = "reachability_forbidden_access"
    message = "PPU register write unsafe during rendering"
    severity = "warning"                       # optional, default "warning"
    entry_points     = ["main"]                # required
    exclude_subtrees = ["nmi_handler"]         # optional, default []
    forbid_symbols   = ["VMDATAL", "CGDATA"]   # optional; by symbol name
    forbid_addrs     = [0x2118, 0x2119]        # optional; sparse address list
    forbid_addr_range = { start = 0x2100, end = 0x2140 }  # optional; contiguous range
    # At least one of forbid_symbols / forbid_addrs / forbid_addr_range is required.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set, Tuple

from r65.compiler.hir import HIRFunctionDecl, HIRIdentifier, HIRStaticDecl
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import (
    optional_addr_range,
    optional_list_of_int,
    optional_list_of_str,
    parse_severity,
    require_key,
    require_list_of_str,
)


KIND_NAME = "reachability_forbidden_access"


class ReachabilityForbiddenAccess(LintRule):
    """Config-instantiated rule — one instance per ``[[rule]]`` table."""

    def __init__(
        self,
        code: str,
        message: str,
        entry_points,
        exclude_subtrees=(),
        forbid_symbols=(),
        forbid_addrs: Iterable[int] = (),
        forbid_addr_range: Optional[Tuple[int, int]] = None,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if not entry_points:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `entry_points` must not be empty"
            )
        if not forbid_symbols and not forbid_addrs and forbid_addr_range is None:
            raise ValueError(
                f"rule kind '{KIND_NAME}': at least one of `forbid_symbols`, "
                f"`forbid_addrs`, or `forbid_addr_range` must be specified"
            )

        super().__init__(
            code=code,
            name=KIND_NAME,
            description=message,
        )
        self.message = message
        self.entry_points = list(entry_points)
        self.exclude_subtrees = list(exclude_subtrees)
        self.forbid_symbols: Set[str] = set(forbid_symbols)
        self.forbid_addrs: Set[int] = set(forbid_addrs)
        self.forbid_addr_range = forbid_addr_range
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

        self._forbidden_sym_ids: Set[int] = set()
        self._reachable: Set[str] = set()
        self._current_reachable = False

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: LintContext) -> None:
        self._forbidden_sym_ids = self._collect_forbidden(ctx)
        cg = ctx.call_graph
        self._reachable = cg.reachable_from(
            self.entry_points, exclude=self.exclude_subtrees
        )

    def _collect_forbidden(self, ctx: LintContext) -> Set[int]:
        forbidden: Set[int] = set()
        for decl in ctx.program.declarations:
            if not isinstance(decl, HIRStaticDecl):
                continue
            if decl.symbol is None:
                continue
            if self._decl_matches(decl):
                forbidden.add(id(decl.symbol))
        return forbidden

    def _decl_matches(self, decl: HIRStaticDecl) -> bool:
        if decl.name in self.forbid_symbols:
            return True
        if decl.storage_attr is None:
            return False
        addr = getattr(decl.storage_attr, "address", None)
        if addr is None:
            return False
        if addr in self.forbid_addrs:
            return True
        if self.forbid_addr_range is not None:
            start, end = self.forbid_addr_range
            return start <= addr < end
        return False

    # -------------------------------------------------------------- per-func

    def enter_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        self._current_reachable = func.name in self._reachable

    def leave_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        self._current_reachable = False

    def visit_identifier(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        self._check(expr, ctx)

    def visit_identifier_write(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        self._check(expr, ctx)

    def _check(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        if not self._current_reachable:
            return
        if expr.symbol is None:
            return
        if id(expr.symbol) not in self._forbidden_sym_ids:
            return
        ctx.emit(
            code=self.code,
            message=f"{self.message} (`{expr.name}`)",
            source_loc=expr.source_loc,
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

def from_config(spec: Dict[str, Any]) -> ReachabilityForbiddenAccess:
    """Instantiate the rule from a ``[[rule]]`` TOML table.

    The loader has already shape-validated ``code``, ``kind``, ``message``.
    We only validate the kind-specific params here.
    """
    code = spec["code"]
    message = spec["message"]
    entry_points = require_list_of_str(
        require_key(spec, "entry_points", KIND_NAME),
        "entry_points",
        KIND_NAME,
    )
    exclude_subtrees = optional_list_of_str(spec, "exclude_subtrees", KIND_NAME)
    forbid_symbols = optional_list_of_str(spec, "forbid_symbols", KIND_NAME)
    forbid_addrs = optional_list_of_int(spec, "forbid_addrs", KIND_NAME)
    forbid_addr_range = optional_addr_range(spec, "forbid_addr_range", KIND_NAME)
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return ReachabilityForbiddenAccess(
        code=code,
        message=message,
        entry_points=entry_points,
        exclude_subtrees=exclude_subtrees,
        forbid_symbols=forbid_symbols,
        forbid_addrs=forbid_addrs,
        forbid_addr_range=forbid_addr_range,
        severity_name=severity,
        hint=hint,
    )
