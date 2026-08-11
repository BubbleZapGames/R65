"""
HIR-level call graph for the linter.

Runs before MIR construction, so we cannot reuse
``r65.compiler.analysis.call_graph`` (which operates on MIR). This module
provides a lightweight directed call graph over :class:`HIRFunctionDecl`,
built by walking each function body and collecting ``HIRFunctionCall`` /
``HIRMethodCall`` / ``HIRFunctionAddress`` nodes.

Used by ``reachability_forbidden_access`` and any other reachability-based
rule kind.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from r65.compiler.errors import SourceLocation
from r65.compiler.hir import (
    HIRAddressOf,
    HIRArrayFillExpr,
    HIRArrayIndex,
    HIRArrayLiteralExpr,
    HIRAssignment,
    HIRBinaryOp,
    HIRBlock,
    HIRBreakStmt,
    HIRContinueStmt,
    HIRDereference,
    HIRExprStmt,
    HIRExpression,
    HIRFieldAccess,
    HIRFunctionAddress,
    HIRFunctionCall,
    HIRFunctionDecl,
    HIRImplDecl,
    HIRIdentifier,
    HIRIfStmt,
    HIRLetStmt,
    HIRMethodCall,
    HIRMultiAssignment,
    HIRProgram,
    HIRReturnStmt,
    HIRStatement,
    HIRStaticDecl,
    HIRStructLiteralExpr,
    HIRMultiLetStmt,
    HIRTupleLetStmt,
    HIRTypeCast,
    HIRUnaryOp,
    HIRWhileStmt,
)
from r65.compiler.hir.nodes import (
    HIRBlockExpression,
    HIRIfExpression,
    HIRLoopExpression,
    HIRMatchExpression,
)
from r65.compiler.hir.symbol_table import SymbolKind


@dataclass
class CallSite:
    """A specific call observed during call-graph construction."""

    caller: str
    callee: Optional[str]  # None for indirect calls
    source_loc: Optional[SourceLocation] = None
    is_indirect: bool = False


@dataclass
class CallGraph:
    """Directed HIR-level call graph.

    ``edges`` maps a caller's function name to the set of directly-called
    function names. ``indirect_call_sites`` records call sites that could not
    be statically resolved (function pointers, trait dispatch, method calls).
    ``address_taken`` lists functions whose address was referenced somewhere
    — the candidate targets of a function-pointer call. ``indirect_callers``
    lists functions that make a function-pointer call; :meth:`get_callees`
    widens each of them to the whole ``address_taken`` set.
    """

    nodes: Dict[str, HIRFunctionDecl] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    indirect_call_sites: Dict[str, List[CallSite]] = field(default_factory=dict)
    address_taken: Set[str] = field(default_factory=set)
    indirect_callers: Set[str] = field(default_factory=set)
    all_call_sites: Dict[str, List[CallSite]] = field(default_factory=dict)

    def add_edge(self, caller: str, callee: str) -> None:
        self.edges.setdefault(caller, set()).add(callee)

    def add_indirect(self, caller: str, site: CallSite) -> None:
        self.indirect_call_sites.setdefault(caller, []).append(site)

    def add_call_site(self, caller: str, site: CallSite) -> None:
        self.all_call_sites.setdefault(caller, []).append(site)

    def get_callees(self, func_name: str) -> Set[str]:
        """Functions ``func_name`` can call, direct edges plus indirect targets.

        A function that calls through a function pointer can reach any
        function whose address is taken. That is a sound over-approximation
        (the pointer's true value is not tracked here), and it mirrors the
        MIR-level call graph, so every reachability-based rule — C001
        included — sees function-pointer targets rather than treating them as
        dead ends. Trait/method dispatch is a separate, still-unresolved case.
        """
        callees = set(self.edges.get(func_name, ()))
        if func_name in self.indirect_callers:
            callees |= self.address_taken
        return callees

    def reachable_from(
        self,
        roots: Iterable[str],
        exclude: Iterable[str] = (),
    ) -> Set[str]:
        """BFS from ``roots`` collecting every transitively-called function.

        ``exclude`` names functions whose subtree should be pruned: the
        excluded function itself and everything reachable only through it
        are omitted from the result. This is the mechanism that lets
        ``reachability_forbidden_access`` say "code reachable from main
        EXCEPT the nmi_handler subtree".
        """
        exclude_set = set(exclude)
        visited: Set[str] = set()
        queue = deque()
        for root in roots:
            if root in exclude_set:
                continue
            if root not in visited:
                visited.add(root)
                queue.append(root)
        while queue:
            current = queue.popleft()
            for callee in self.get_callees(current):
                if callee in exclude_set:
                    continue
                if callee in visited:
                    continue
                visited.add(callee)
                queue.append(callee)
        return visited

    def depth_from(self, root: str, exclude: Iterable[str] = ()) -> Dict[str, int]:
        """BFS from ``root`` returning the minimum call distance to each
        reachable function. ``root`` itself is at depth 0."""
        exclude_set = set(exclude)
        distances: Dict[str, int] = {}
        if root in exclude_set:
            return distances
        distances[root] = 0
        queue = deque([root])
        while queue:
            current = queue.popleft()
            current_depth = distances[current]
            for callee in self.get_callees(current):
                if callee in exclude_set:
                    continue
                if callee in distances:
                    continue
                distances[callee] = current_depth + 1
                queue.append(callee)
        return distances

    def shortest_path(self, src: str, dst: str) -> Optional[List[str]]:
        """Return the shortest caller-chain ``[src, ..., dst]`` or ``None``."""
        if src == dst:
            return [src]
        parents: Dict[str, str] = {}
        visited: Set[str] = {src}
        queue = deque([src])
        while queue:
            current = queue.popleft()
            for callee in self.get_callees(current):
                if callee in visited:
                    continue
                visited.add(callee)
                parents[callee] = current
                if callee == dst:
                    path = [dst]
                    while path[-1] != src:
                        path.append(parents[path[-1]])
                    return list(reversed(path))
                queue.append(callee)
        return None


def build_call_graph(program: HIRProgram) -> CallGraph:
    """Build a :class:`CallGraph` by walking every function body in ``program``."""
    graph = CallGraph()

    for func in _all_functions(program):
        graph.nodes[func.name] = func

    builder = _CallGraphBuilder(graph)
    for func in _all_functions(program):
        builder.walk_function(func)

    # Static initializers can name functions too — a ROM jump table such as
    # `static TABLE: [fn(); N] = [h0, h1, ...]` is the canonical
    # function-pointer pattern on this hardware. Those references must count as
    # address-taken so an indirect call through the table resolves to them.
    for decl in program.declarations:
        if isinstance(decl, HIRStaticDecl) and decl.initializer is not None:
            builder.walk_static_initializer(decl)

    return graph


def _all_functions(program: HIRProgram):
    """Every function body in the program, impl methods included.

    Impl methods are desugared to HIRFunctionDecls named ``Struct__method``
    but live inside their HIRImplDecl rather than at top level. Skipping them
    would leave every method body outside the call graph, so nothing reachable
    only through a method call could be analysed.
    """
    for decl in program.declarations:
        if isinstance(decl, HIRFunctionDecl):
            yield decl
        elif isinstance(decl, HIRImplDecl):
            yield from decl.methods


class _CallGraphBuilder:
    """Walks HIR bodies collecting call sites and function-address references."""

    def __init__(self, graph: CallGraph):
        self.graph = graph
        self._current: Optional[str] = None

    def walk_function(self, func: HIRFunctionDecl) -> None:
        self._current = func.name
        if func.body is not None:
            self._walk_block(func.body)
        self._current = None

    def walk_static_initializer(self, decl: HIRStaticDecl) -> None:
        """Scan a static's initializer for function references (jump tables).

        Runs under a synthetic ``<static:NAME>`` caller so ``_walk_expr`` is
        satisfied; that node is never an entry point, so any edges it picks up
        (e.g. a const-fn call in the initializer) do not affect reachability
        from real entry points. The point is the ``address_taken`` side effect.
        """
        self._current = f"<static:{decl.name}>"
        self._walk_expr(decl.initializer)
        self._current = None

    # ------------------------------------------------------------------ stmts

    def _walk_block(self, block: HIRBlock) -> None:
        for stmt in block.statements:
            self._walk_stmt(stmt)

    def _walk_stmt(self, stmt: HIRStatement) -> None:
        if isinstance(stmt, HIRBlock):
            self._walk_block(stmt)
        elif isinstance(stmt, HIRLetStmt):
            self._walk_expr(stmt.initializer)
        elif isinstance(stmt, HIRMultiLetStmt):
            self._walk_expr(stmt.initializer)
        elif isinstance(stmt, HIRExprStmt):
            self._walk_expr(stmt.expr)
        elif isinstance(stmt, HIRReturnStmt):
            for v in stmt.values:
                self._walk_expr(v)
        elif isinstance(stmt, HIRIfStmt):
            self._walk_expr(stmt.condition)
            if stmt.then_block is not None:
                self._walk_block(stmt.then_block)
            if stmt.else_block is not None:
                if isinstance(stmt.else_block, HIRIfStmt):
                    self._walk_stmt(stmt.else_block)
                else:
                    self._walk_block(stmt.else_block)
        elif isinstance(stmt, HIRWhileStmt):
            self._walk_expr(stmt.condition)
            if stmt.body is not None:
                self._walk_block(stmt.body)
        elif isinstance(stmt, (HIRBreakStmt, HIRContinueStmt)):
            pass
        elif isinstance(stmt, HIRExpression):
            # See note in linter.py: for-loop desugar plants bare expressions
            # (HIRAssignment, HIRMultiAssignment) directly into statement lists.
            self._walk_expr(stmt)

    # ------------------------------------------------------------------ exprs

    def _walk_expr(self, expr: Optional[HIRExpression]) -> None:
        if expr is None or self._current is None:
            return

        if isinstance(expr, HIRFunctionCall):
            self._record_call(expr)
            # Descend into the callee expression only for indirect calls. A
            # direct callee identifier is already recorded as an edge; walking
            # it would hit the HIRIdentifier branch below and misclassify the
            # called function as address-taken.
            if _resolve_direct_callee(expr.func) is None:
                self._walk_expr(expr.func)
            for a in expr.args:
                self._walk_expr(a)
        elif isinstance(expr, HIRIdentifier):
            # A function name used as a value (not in call position) is a
            # function-address expression. At this pre-MIR stage it is a plain
            # HIRIdentifier resolving to a FUNCTION symbol, not a
            # HIRFunctionAddress — so record it as address-taken here, which is
            # what makes it a candidate target of function-pointer calls.
            symbol = expr.symbol
            if symbol is not None and getattr(symbol, "kind", None) in (
                SymbolKind.FUNCTION,
                SymbolKind.METHOD,
            ):
                self.graph.address_taken.add(symbol.name)
        elif isinstance(expr, HIRMethodCall):
            # Treat method calls as indirect by default — trait dispatch is
            # polymorphic, and impl-resolved calls are not yet rewritten to
            # HIRFunctionCall at this pipeline stage. Safer to record them as
            # reachability boundaries so rules can decide what to do.
            site = CallSite(
                caller=self._current,
                callee=None,
                source_loc=expr.source_loc,
                is_indirect=True,
            )
            self.graph.add_indirect(self._current, site)
            self.graph.add_call_site(self._current, site)
            self._walk_expr(expr.receiver)
            for a in expr.args:
                self._walk_expr(a)
        elif isinstance(expr, HIRFunctionAddress):
            if expr.function_name:
                self.graph.address_taken.add(expr.function_name)
        elif isinstance(expr, HIRBinaryOp):
            self._walk_expr(expr.left)
            self._walk_expr(expr.right)
        elif isinstance(expr, HIRUnaryOp):
            self._walk_expr(expr.operand)
        elif isinstance(expr, HIRAssignment):
            self._walk_expr(expr.target)
            self._walk_expr(expr.value)
        elif isinstance(expr, HIRMultiAssignment):
            for t in expr.targets:
                self._walk_expr(t)
            self._walk_expr(expr.value)
        elif isinstance(expr, HIRArrayIndex):
            self._walk_expr(expr.array)
            self._walk_expr(expr.index)
        elif isinstance(expr, HIRFieldAccess):
            self._walk_expr(expr.base)
        elif isinstance(expr, HIRDereference):
            self._walk_expr(expr.pointer)
        elif isinstance(expr, HIRAddressOf):
            self._walk_expr(expr.operand)
        elif isinstance(expr, HIRTypeCast):
            self._walk_expr(expr.expr)
        elif isinstance(expr, HIRArrayLiteralExpr):
            for e in expr.elements:
                self._walk_expr(e)
        elif isinstance(expr, HIRArrayFillExpr):
            self._walk_expr(expr.fill_value)
        elif isinstance(expr, HIRStructLiteralExpr):
            for f in expr.fields:
                self._walk_expr(f.value)
        elif isinstance(expr, HIRBlockExpression):
            for s in expr.statements:
                self._walk_stmt(s)
            self._walk_expr(expr.final_expr)
        elif isinstance(expr, HIRIfExpression):
            self._walk_expr(expr.condition)
            self._walk_expr(expr.then_block)
            self._walk_expr(expr.else_block)
        elif isinstance(expr, HIRLoopExpression):
            if expr.body is not None:
                self._walk_block(expr.body)
        elif isinstance(expr, HIRMatchExpression):
            self._walk_expr(expr.scrutinee)
            for arm in expr.arms:
                body = arm.body
                if isinstance(body, HIRExpression):
                    self._walk_expr(body)
                elif isinstance(body, HIRStatement):
                    self._walk_stmt(body)
        # Leaves (literals, identifiers, registers) have no sub-expressions.

    def _record_call(self, call: HIRFunctionCall) -> None:
        assert self._current is not None
        func_expr = call.func
        callee_name = _resolve_direct_callee(func_expr)
        if callee_name is None:
            # Method calls reach the linter already resolved: the type checker
            # rewrites them to HIRFunctionCall with func=None and the target's
            # mangled name in method_call_info. Without this they degrade to
            # "indirect" and the callee drops out of reachability entirely.
            info = getattr(call, "method_call_info", None)
            if info:
                callee_name = info.get("mangled_name")
        if callee_name is not None:
            self.graph.add_edge(self._current, callee_name)
            site = CallSite(
                caller=self._current,
                callee=callee_name,
                source_loc=call.source_loc,
                is_indirect=False,
            )
            self.graph.add_call_site(self._current, site)
        else:
            # A call whose callee is not a statically-known function: a call
            # through a function pointer. Mark the caller so get_callees widens
            # it to the address-taken set. (Trait/method dispatch takes the
            # HIRMethodCall path above and stays unresolved — a separate case.)
            self.graph.indirect_callers.add(self._current)
            site = CallSite(
                caller=self._current,
                callee=None,
                source_loc=call.source_loc,
                is_indirect=True,
            )
            self.graph.add_indirect(self._current, site)
            self.graph.add_call_site(self._current, site)


def _resolve_direct_callee(func_expr: Optional[HIRExpression]) -> Optional[str]:
    """Return the name of a direct callee, or ``None`` if the call is indirect."""
    if not isinstance(func_expr, HIRIdentifier):
        return None
    symbol = func_expr.symbol
    if symbol is None:
        return None
    if getattr(symbol, "kind", None) in (
        SymbolKind.FUNCTION,
        SymbolKind.METHOD,
        SymbolKind.BUILTIN_FUNC,
    ):
        return symbol.name
    return None
