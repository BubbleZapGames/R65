"""
Rule kind: ``call_depth_limit``.

BFS from an entry point; emit at the *first* edge that exceeds ``max_depth``.
Pointing users at the offending call rather than at the leaf keeps the
diagnostic actionable::

    [[rule]]
    code = "C023"
    kind = "call_depth_limit"
    message = "NMI handler call chain too deep — vblank is sacred"
    entry     = "nmi_handler"
    max_depth = 3
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from r65.compiler.lint.rule import LintContext, LintRule
from r65.compiler.lint.rule_kinds._common import parse_severity, require_key


KIND_NAME = "call_depth_limit"


class CallDepthLimit(LintRule):
    def __init__(
        self,
        code: str,
        message: str,
        entry: str,
        max_depth: int,
        severity_name: str = "warning",
        hint: Optional[str] = None,
    ):
        if max_depth < 0:
            raise ValueError(
                f"rule kind '{KIND_NAME}': `max_depth` must be >= 0"
            )
        super().__init__(code=code, name=KIND_NAME, description=message)
        self.message = message
        self.entry = entry
        self.max_depth = max_depth
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

    def setup(self, ctx: LintContext) -> None:
        cg = ctx.call_graph
        if self.entry not in cg.nodes:
            return
        distances = cg.depth_from(self.entry)
        # For each function beyond max_depth, find the edge (parent -> func)
        # that first crosses the threshold and emit one diagnostic per such
        # edge. A parent can emit multiple edges, but each edge is only
        # reported once by construction (its distance is unique per target).
        seen = set()
        for func_name, depth in sorted(distances.items(), key=lambda kv: kv[1]):
            if depth <= self.max_depth:
                continue
            if func_name in seen:
                continue
            seen.add(func_name)
            # Find a parent whose depth is exactly depth - 1.
            parent = None
            for candidate in cg.edges:
                if func_name in cg.get_callees(candidate) and distances.get(candidate) == depth - 1:
                    parent = candidate
                    break
            parent_decl = cg.nodes.get(parent) if parent else None
            loc = parent_decl.source_loc if parent_decl is not None else None
            path = cg.shortest_path(self.entry, func_name) or [func_name]
            hint = self.custom_hint or f"call chain: {' → '.join(path)} (depth {depth}, limit {self.max_depth})"
            ctx.emit(
                code=self.code,
                message=(
                    f"{self.message} (`{func_name}` at depth {depth} from `{self.entry}`)"
                ),
                source_loc=loc,
                hint=hint,
                severity=self.severity,
            )


def from_config(spec: Dict[str, Any]) -> CallDepthLimit:
    code = spec["code"]
    message = spec["message"]
    entry = require_key(spec, "entry", KIND_NAME)
    if not isinstance(entry, str):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `entry` must be a string"
        )
    max_depth = require_key(spec, "max_depth", KIND_NAME)
    if not isinstance(max_depth, int) or isinstance(max_depth, bool):
        raise ValueError(
            f"rule kind '{KIND_NAME}': `max_depth` must be an integer"
        )
    severity = spec.get("severity", "warning")
    hint = spec.get("hint")
    return CallDepthLimit(
        code=code,
        message=message,
        entry=entry,
        max_depth=max_depth,
        severity_name=severity,
        hint=hint,
    )
