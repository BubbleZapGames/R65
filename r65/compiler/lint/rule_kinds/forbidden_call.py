"""
Rule kind: ``forbidden_call``.

Flags calls from a set of ``callers`` to any function in ``callees``, either
directly or transitively. Complements ``reachability_forbidden_access``
for the "A must not call B" case:

    [[rule]]
    code = "C010"
    kind = "forbidden_call"
    message = "IRQ handlers must not call allocator routines"
    callers    = ["irq_handler", "nmi_handler"]
    callees    = ["malloc_far", "free_far"]
    transitive = true    # optional, default true
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from r65.compiler.hir import HIRFunctionDecl
from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import (
    optional_list_of_str,
    parse_severity,
    require_key,
    require_list_of_str,
)


KIND_NAME = "forbidden_call"


class ForbiddenCall(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        callers: List[str],
        callees: List[str],
        transitive: bool = True,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if not callers:
            raise ValueError(f"rule kind '{KIND_NAME}': `callers` must not be empty")
        if not callees:
            raise ValueError(f"rule kind '{KIND_NAME}': `callees` must not be empty")

        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.callers = list(callers)
        self.callees: Set[str] = set(callees)
        self.transitive = transitive
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

        # For each caller, map forbidden callee -> path list (or None if
        # direct-only and not in direct callees).
        self._violations: Dict[str, Dict[str, List[str]]] = {}

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: LintContext) -> None:
        cg = ctx.call_graph
        for caller in self.callers:
            if caller not in cg.nodes:
                continue
            hits: Dict[str, List[str]] = {}
            if self.transitive:
                reachable = cg.reachable_from([caller])
                for forbidden in self.callees:
                    if forbidden in reachable and forbidden != caller:
                        path = cg.shortest_path(caller, forbidden)
                        if path is not None:
                            hits[forbidden] = path
            else:
                direct = cg.get_callees(caller)
                for forbidden in self.callees:
                    if forbidden in direct:
                        hits[forbidden] = [caller, forbidden]
            if hits:
                self._violations[caller] = hits

    # -------------------------------------------------------------- per-func

    def enter_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        if func.name not in self._violations:
            return
        hits = self._violations[func.name]
        for callee, path in hits.items():
            hint = self.custom_hint
            if hint is None:
                hint = "call chain: " + " → ".join(path)
            ctx.emit(
                code=self.code,
                message=f"{self.message} (`{func.name}` → `{callee}`)",
                source_loc=func.source_loc,
                hint=hint,
                severity=self.severity,
            )


def from_config(spec: Dict[str, Any]) -> ForbiddenCall:
    code = spec["code"]
    message = spec["message"]
    callers = require_list_of_str(
        require_key(spec, "callers", KIND_NAME), "callers", KIND_NAME
    )
    callees = require_list_of_str(
        require_key(spec, "callees", KIND_NAME), "callees", KIND_NAME
    )
    transitive = spec.get("transitive", True)
    if not isinstance(transitive, bool):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `transitive` must be a boolean"
        )
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return ForbiddenCall(
        code=code,
        message=message,
        callers=callers,
        callees=callees,
        transitive=transitive,
        severity_name=severity,
        hint=hint,
    )
