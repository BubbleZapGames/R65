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
    exempt_on_write_addrs   = [0x2100]        # optional; see below
    exempt_on_write_symbols = ["INIDISP"]     # optional; see below

``exempt_on_write_*`` names guard registers. A function that *writes* one of
them anywhere in its body has its init prologue exempted — every access before
the first loop is skipped, including ones textually before the guard write
(the guard may sit anywhere in that straight-line prologue). Accesses from the
first loop onward are never exempt; see the limits below.

The canonical guard is ``INIDISP`` ($2100): a function that touches it is
managing forced blank deliberately, so its PPU writes are the programmer's
call. This is a heuristic, not an analysis — blanking is global mutable state
that any function can change, so tracking it soundly is not possible here, and
guessing would trade a sound rule for an unsound one. Exempting the writer is
predictable and keeps the rule free of false negatives elsewhere.

Two deliberate limits:

* The exemption covers the writing function only, not its callees. Helpers
  called during forced blank still need ``exclude_subtrees``.
* The exemption covers only the init prologue — accesses **before the first
  loop** in the function. Forced blank is established once during init, in
  straight-line code; the first loop is the game loop, running with the screen
  on. So neither the loop body nor anything textually after it is exempt: a
  ``main`` that blanks up in init cannot silence a PPU write down in its game
  loop, nor one placed after the loop. This is the position sensitivity the
  guard needs — "OK to write at the very top, not once the loop begins" — and
  it is a strict tightening (it can only add warnings, never hide one).

Init-reachable functions are also exempt, automatically — no config entry.
A function reachable from an entry point *only* through calls that precede the
entry's first loop (never through the game loop) runs entirely during init, so
its straight-line PPU writes are setup writes. This lifts the same
"before the first loop = init" boundary from inside a function up to the call
graph, so a boot-time video-setup helper need not be listed in
``exclude_subtrees`` by hand. The loop carve-out still applies: a write inside
a loop in such a function is *still* flagged — a VRAM-writing loop with the
display on is a real bug the rule must keep catching, and flagging a blanked
init loop is the safe direction to err (``exclude_subtrees`` covers that). A
helper reachable through the game loop as well (e.g. a level-load routine
called both at boot and on a mid-game transition) is *not* init-reachable and
stays fully checked.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set, Tuple

from r65.compiler.hir import (
    HIRFunctionCall,
    HIRFunctionDecl,
    HIRIdentifier,
    HIRLoopExpression,
    HIRMethodCall,
    HIRStaticDecl,
    HIRWhileStmt,
)
from r65.compiler.lint.call_graph import _resolve_direct_callee
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
        exempt_on_write_symbols=(),
        exempt_on_write_addrs: Iterable[int] = (),
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
        self.exempt_on_write_symbols: Set[str] = set(exempt_on_write_symbols)
        self.exempt_on_write_addrs: Set[int] = set(exempt_on_write_addrs)
        self.severity = parse_severity(severity_name)
        self.custom_hint = hint

        self._forbidden_sym_ids: Set[int] = set()
        self._guard_sym_ids: Set[int] = set()
        self._reachable: Set[str] = set()
        # Functions reachable only through an entry point's init prologue.
        self._init_reachable_funcs: Set[str] = set()
        self._current_reachable = False
        self._current_init_reachable = False
        # Diagnostics are buffered per function so a guard write anywhere in
        # the body can retract the ones already found.
        self._pending: list = []
        self._guard_written = False
        # Identifiers at or after the first loop — never covered by the guard
        # or init-reachable exemption (see module docstring).
        self._post_init_ids: Set[int] = set()

    # ------------------------------------------------------------------ setup

    def setup(self, ctx: LintContext) -> None:
        self._forbidden_sym_ids = self._collect_forbidden(ctx)
        self._guard_sym_ids = self._collect_matching(
            ctx, self.exempt_on_write_symbols, self.exempt_on_write_addrs
        )
        cg = ctx.call_graph
        self._reachable = cg.reachable_from(
            self.entry_points, exclude=self.exclude_subtrees
        )
        self._init_reachable_funcs = self._compute_init_reachable(cg)

    def _compute_init_reachable(self, cg) -> Set[str]:
        """Functions reachable from an entry point only through its prologue.

        The live region of an entry is everything from its first loop onward —
        the game loop and whatever it calls, directly or through function
        pointers. Anything reachable from the entry but not from that live
        region runs only during init, so its straight-line writes are exempt.

        If an entry cannot be found or has no body its game loop is unknown, so
        no auto-exemption is granted at all (conservative — the whole point is
        to never hide a live-display write).
        """
        live_roots: Set[str] = set()
        for entry in self.entry_points:
            func = cg.nodes.get(entry)
            if func is None or func.body is None:
                return set()
            live_direct, live_indirect = _live_region_callees(func)
            live_roots |= live_direct
            if live_indirect:
                # A function-pointer call in the game loop can reach any
                # address-taken function; none of those are init-only.
                live_roots |= cg.address_taken
        live_reachable = cg.reachable_from(
            live_roots, exclude=self.exclude_subtrees
        )
        return self._reachable - live_reachable - set(self.entry_points)

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

    def _collect_matching(self, ctx: LintContext, names: Set[str],
                          addrs: Set[int]) -> Set[int]:
        """Symbol ids for statics matching `names` or whose #[hw] address is in `addrs`."""
        if not names and not addrs:
            return set()
        matched: Set[int] = set()
        for decl in ctx.program.declarations:
            if not isinstance(decl, HIRStaticDecl) or decl.symbol is None:
                continue
            if decl.name in names:
                matched.add(id(decl.symbol))
                continue
            addr = getattr(decl.storage_attr, "address", None) if decl.storage_attr else None
            if addr is not None and addr in addrs:
                matched.add(id(decl.symbol))
        return matched

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
        self._current_init_reachable = func.name in self._init_reachable_funcs
        self._pending = []
        self._guard_written = False
        self._post_init_ids = _identifiers_from_first_loop(func)

    def leave_function(self, func: HIRFunctionDecl, ctx: LintContext) -> None:
        # Two exemptions share the same prologue carve-out: a guard write
        # anywhere in the body, or the function being reachable only through an
        # entry's init prologue. Either one exempts an access before the first
        # loop; neither exempts an access from the first loop onward. So the
        # buffered diagnostics are emitted once the body has been walked.
        prologue_exempt = self._guard_written or self._current_init_reachable
        for post_init, kwargs in self._pending:
            if prologue_exempt and not post_init:
                continue
            ctx.emit(**kwargs)
        self._pending = []
        self._guard_written = False
        self._current_reachable = False
        self._current_init_reachable = False

    def visit_identifier(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        self._check(expr, ctx)

    def visit_identifier_write(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        if expr.symbol is not None and id(expr.symbol) in self._guard_sym_ids:
            self._guard_written = True
        self._check(expr, ctx)

    def _check(self, expr: HIRIdentifier, ctx: LintContext) -> None:
        if not self._current_reachable:
            return
        if expr.symbol is None:
            return
        if id(expr.symbol) not in self._forbidden_sym_ids:
            return
        self._pending.append((id(expr) in self._post_init_ids, dict(
            code=self.code,
            message=f"{self.message} (`{expr.name}`)",
            source_loc=expr.source_loc,
            hint=self._build_hint(ctx),
            severity=self.severity,
        )))

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


def _identifiers_from_first_loop(func: HIRFunctionDecl) -> Set[int]:
    """ids of every HIRIdentifier at or after the first loop in ``func``.

    The forced-blank exemption covers only the init prologue — code before the
    game loop. This returns the complement: every identifier from the first
    loop onward, which includes the loop body *and* everything textually after
    it. Those are never covered by the ``exempt_on_write_*`` exemption, so a
    guard write up in init cannot silence a PPU access once the game loop has
    begun.

    The walk is pre-order in dataclass-field order, which is source order for
    the statement lists that matter here. ``passed_loop`` latches on at the
    first loop node and stays on, so siblings visited after the loop — not just
    its body — are included.
    """
    found: Set[int] = set()
    seen: Set[int] = set()
    passed_loop = False

    def walk(node) -> None:
        nonlocal passed_loop
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, (HIRWhileStmt, HIRLoopExpression)):
            passed_loop = True
        if isinstance(node, HIRIdentifier) and passed_loop:
            found.add(id(node))
        for name in getattr(node, "__dataclass_fields__", ()):
            value = getattr(node, name, None)
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, "__dataclass_fields__"):
                        walk(item)
            elif hasattr(value, "__dataclass_fields__"):
                walk(value)

    walk(func.body)
    return found


def _contains_loop(node) -> bool:
    """True if ``node``'s subtree contains any loop construct."""
    seen: Set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None or id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, (HIRWhileStmt, HIRLoopExpression)):
            return True
        for fname in getattr(n, "__dataclass_fields__", ()):
            value = getattr(n, fname, None)
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, "__dataclass_fields__"):
                        stack.append(item)
            elif hasattr(value, "__dataclass_fields__"):
                stack.append(value)
    return False


def _live_region_callees(func: HIRFunctionDecl) -> Tuple[Set[str], bool]:
    """Direct callees invoked in ``func``'s live region, and whether that region
    also makes an indirect (function-pointer / method) call.

    The live region is everything from the first loop onward — an entry point's
    game loop and any straight-line code following it. A function with no loop
    has no init/live split, so its whole body is treated as live (conservative:
    such an entry grants no init-reachable exemption to anything).

    Returns callee *names* (mangled for resolved method calls); the caller feeds
    them to ``reachable_from`` to get everything the game loop can reach.
    """
    direct: Set[str] = set()
    seen: Set[int] = set()
    # If there is no loop at all, everything is live from the start.
    state = {"live": not _contains_loop(func.body), "indirect": False}

    def record(call: HIRFunctionCall) -> None:
        name = _resolve_direct_callee(getattr(call, "func", None))
        if name is None:
            info = getattr(call, "method_call_info", None)
            if info:
                name = info.get("mangled_name")
        if name is not None:
            direct.add(name)
        else:
            state["indirect"] = True

    def walk(node) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, (HIRWhileStmt, HIRLoopExpression)):
            state["live"] = True
        if state["live"]:
            if isinstance(node, HIRFunctionCall):
                record(node)
            elif isinstance(node, HIRMethodCall):
                state["indirect"] = True
        for fname in getattr(node, "__dataclass_fields__", ()):
            value = getattr(node, fname, None)
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, "__dataclass_fields__"):
                        walk(item)
            elif hasattr(value, "__dataclass_fields__"):
                walk(value)

    walk(func.body)
    return direct, state["indirect"]


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
    exempt_on_write_symbols = optional_list_of_str(
        spec, "exempt_on_write_symbols", KIND_NAME)
    exempt_on_write_addrs = optional_list_of_int(
        spec, "exempt_on_write_addrs", KIND_NAME)
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
        exempt_on_write_symbols=exempt_on_write_symbols,
        exempt_on_write_addrs=exempt_on_write_addrs,
        severity_name=severity,
        hint=hint,
    )
