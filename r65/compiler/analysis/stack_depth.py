# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Stack depth overflow analysis.

Walks the call graph from entry points and interrupt handlers, computes
worst-case stack depth (bytes), and emits warnings when it exceeds the
configured stack size.

Warning codes:
  W003 — worst-case call depth exceeds stack size
  W004 — recursive call chain detected (unbounded depth)
  W005 — worst-case depth with interrupts exceeds stack size
"""

from typing import Dict, List, Optional, Set, Tuple
from r65.compiler.mir.nodes import MIRProgram, MIRFunction
from r65.compiler.analysis.call_graph import CallGraphAnalyzer, CallGraph


# Bytes the 65816 hardware pushes on interrupt entry (native mode):
#   PBR (1) + PCH (1) + PCL (1) + P (1) = 4 bytes
INTERRUPT_HW_PUSH_BYTES = 4

# Maximum bytes an interrupt prologue auto-saves when preserve=True:
#   PHA (1-2) + PHX (2) + PHY (2) + PHD (2) + PHB (1) + PHP (1) = 9-10
# We use 10 (worst-case: A in m16 mode = 2 bytes)
INTERRUPT_AUTO_SAVE_BYTES = 10

# Small per-function allowance for spill slots created by codegen
# (region-based spilling for calls, etc.)
MAX_SPILL_BYTES = 6


class StackDepthAnalyzer:
    """
    Analyzes worst-case stack depth across all call paths from roots.

    Roots are entry functions (is_entry) and interrupt handlers (interrupt_attr).
    The analyzer builds a call graph, detects cycles, then does DFS to find
    the deepest call chain.
    """

    def __init__(self, mir_program: MIRProgram,
                 stack_lower: int, stack_upper: int):
        self.program = mir_program
        self.stack_lower = stack_lower
        self.stack_upper = stack_upper
        self.stack_size = stack_upper - stack_lower + 1

        # Built during analyze()
        self.func_map: Dict[str, MIRFunction] = {}
        self.graph: Optional[CallGraph] = None
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> List[str]:
        """
        Run the analysis and return a list of warning strings.
        """
        self.func_map = {f.name: f for f in self.program.functions}

        # Build call graph (resolves TraitDispatch to impl set)
        cga = CallGraphAnalyzer(
            self.program,
            trait_dispatch_info=getattr(self.program, 'trait_dispatch_info', None),
        )
        self.graph = cga.analyze()

        # Detect and warn about recursive cycles (W004)
        cycles = cga.find_cycles()
        cycle_funcs: Set[str] = set()
        for cycle in cycles:
            cycle_funcs.update(cycle)
            self._warn_recursion(cycle)

        # Identify roots
        entry_roots: List[MIRFunction] = []
        interrupt_roots: List[MIRFunction] = []
        for f in self.program.functions:
            if f.is_entry:
                entry_roots.append(f)
            if f.interrupt_attr:
                interrupt_roots.append(f)

        # Compute worst-case depth for normal (non-interrupt) call trees
        max_normal_depth = 0
        max_normal_chain: List[str] = []
        for root in entry_roots:
            depth, chain = self._max_depth(root.name, cycle_funcs)
            if depth > max_normal_depth:
                max_normal_depth = depth
                max_normal_chain = chain

        # Compute worst-case depth for each interrupt handler subtree
        max_interrupt_depth = 0
        max_interrupt_chain: List[str] = []
        for root in interrupt_roots:
            depth, chain = self._max_depth(root.name, cycle_funcs)
            # Add hardware push + auto-save for the interrupt entry
            preserve = True
            if root.interrupt_attr and hasattr(root.interrupt_attr, 'preserve'):
                preserve = root.interrupt_attr.preserve
            hw_bytes = INTERRUPT_HW_PUSH_BYTES
            if preserve:
                hw_bytes += INTERRUPT_AUTO_SAVE_BYTES
            depth += hw_bytes
            if depth > max_interrupt_depth:
                max_interrupt_depth = depth
                max_interrupt_chain = chain

        # W003: normal depth alone exceeds stack
        if max_normal_depth > self.stack_size:
            chain_str = " -> ".join(max_normal_chain)
            self.warnings.append(
                f"W003: Worst-case stack depth ({max_normal_depth} bytes) "
                f"exceeds stack size ({self.stack_size} bytes). "
                f"Call chain: {chain_str}"
            )

        # W005: normal + interrupt depth exceeds stack
        # (only if there are both entry roots and interrupt handlers)
        if entry_roots and interrupt_roots:
            combined = max_normal_depth + max_interrupt_depth
            if combined > self.stack_size:
                self.warnings.append(
                    f"W005: Worst-case stack depth with interrupts "
                    f"({combined} bytes = {max_normal_depth} normal + "
                    f"{max_interrupt_depth} interrupt) exceeds stack size "
                    f"({self.stack_size} bytes)."
                )

        return self.warnings

    # ------------------------------------------------------------------
    # Per-function cost
    # ------------------------------------------------------------------

    def _function_own_cost(self, func: MIRFunction) -> int:
        """
        Bytes this function adds to the stack (excluding the return address
        pushed by its caller, which is accounted for at the call edge).

        When partial frame deallocation is active (codegen_frame_dead_before_calls),
        only the live portion of the frame is counted for non-leaf functions,
        since the dead portion is reclaimed before calls.

        Spill slots are only needed for non-leaf functions (region-based
        spilling saves registers around call sites).
        """
        cost = func.codegen_prologue_bytes
        if func.codegen_frame_dead_before_calls and self.graph.get_callees(func.name):
            cost += func.codegen_max_live_frame_bytes_at_calls
        else:
            cost += func.codegen_frame_size
        if self.graph.get_callees(func.name):
            cost += MAX_SPILL_BYTES
        return cost

    @staticmethod
    def _call_edge_cost(callee: MIRFunction) -> int:
        """Bytes the call instruction itself pushes (return address)."""
        return 3 if callee.is_far else 2

    # ------------------------------------------------------------------
    # DFS for maximum depth
    # ------------------------------------------------------------------

    def _max_depth(self, root_name: str,
                   cycle_funcs: Set[str]) -> Tuple[int, List[str]]:
        """
        DFS from *root_name* returning (max_bytes, call_chain).

        - Entry functions have no call-edge cost (no caller pushes a return addr).
        - Cycles are skipped (warned separately as W004).
        - Diamond shapes are handled correctly: we visit each function freshly
          per path so the *worst* path wins (not a shared-visit cache).
        """
        root_func = self.func_map.get(root_name)
        if root_func is None:
            return (0, [root_name])

        # own_cost for the root (no return address — it is the root)
        root_cost = self._function_own_cost(root_func)

        best_sub_depth, best_sub_chain = self._dfs(
            root_name, set(), cycle_funcs
        )
        total = root_cost + best_sub_depth
        chain = [root_name] + best_sub_chain
        return (total, chain)

    def _dfs(self, func_name: str, visited: Set[str],
             cycle_funcs: Set[str]) -> Tuple[int, List[str]]:
        """
        Return (additional_bytes, sub_chain) for the worst callee path
        reachable from *func_name*.
        """
        callees = self.graph.get_callees(func_name)
        if not callees:
            return (0, [])

        best_depth = 0
        best_chain: List[str] = []

        for callee_name in callees:
            callee = self.func_map.get(callee_name)
            if callee is None:
                # External / unknown function — can't analyse, skip
                continue

            # Skip if already on the current DFS path (cycle)
            if callee_name in visited:
                continue

            # Skip functions that are part of a recursive cycle
            if callee_name in cycle_funcs:
                continue

            # Cost of calling this callee: return address + callee's own cost
            edge_cost = self._call_edge_cost(callee)
            callee_cost = self._function_own_cost(callee)

            visited.add(callee_name)
            sub_depth, sub_chain = self._dfs(callee_name, visited, cycle_funcs)
            visited.remove(callee_name)

            total = edge_cost + callee_cost + sub_depth
            if total > best_depth:
                best_depth = total
                best_chain = [callee_name] + sub_chain

        return (best_depth, best_chain)

    # ------------------------------------------------------------------
    # Warning helpers
    # ------------------------------------------------------------------

    def _warn_recursion(self, cycle: List[str]):
        """Emit W004 for a recursive cycle."""
        if len(cycle) == 1:
            self.warnings.append(
                f"W004: Function '{cycle[0]}' is recursive "
                f"(unbounded stack depth)."
            )
        else:
            cycle_str = " -> ".join(cycle + [cycle[0]])
            self.warnings.append(
                f"W004: Recursive call cycle detected "
                f"(unbounded stack depth): {cycle_str}"
            )
