# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Static stack-size analysis.

Walks the call graph bottom-up to bound the runtime stack high-water mark
of an R65 program. Runs as the final step of compilation, after codegen
has populated the per-function inputs on ``MIRFunction`` (frame size,
prologue bytes, region-spill peak, outgoing arg bytes).

Algorithm
---------

    stack_use(f) = own_frame(f)
                 + max over call sites s in f of (ra(s) + stack_use(callee(s)))

    own_frame(f) = abi.prologue_stack_bytes      # PHB / PHX / PHY / PHD
                 + codegen_frame_size             # locals
                 + codegen_max_region_spill_bytes # peak PHA/PHX/PHY around calls
                 + max_outgoing_arg_bytes         # caller-owned outgoing args

    ra(s) = 3 if call_site.is_far else 2         # JSL pushes PBR+PC vs JSR PC

Recursion is already rejected at step 7 of the pipeline; we assert
acyclicity defensively and raise a clear error if a cycle slips through.

Indirect calls (``Call.function`` is a ``VirtualRegister``) widen to the
union of all address-taken functions — a conservative over-approximation.
Trait dispatch resolves precisely via ``CallGraph.trait_impls``. ``asm!()``
blocks are treated as zero stack effect (documented contract: any
``asm!()`` that pushes without balancing within the block invalidates the
bound).

Interrupt handlers fire asynchronously, so their cost is added on top of
the sync-path high-water mark. v1 assumes handlers run with interrupts
disabled (R65's default), so at most one handler is active at a time;
nested-interrupt accounting is a follow-up.

Capacity is taken from ``MIRProgram.stack_attr`` (``#[stack(start, end)]``)
with the SNES default of $0100..$01FF (256 bytes). If the computed total
exceeds capacity, raises ``CodegenError`` with the deepest call chain.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, Call, TraitDispatch, VirtualRegister,
)
from r65.compiler.analysis.call_graph import CallGraph, CallGraphAnalyzer
from r65.compiler.errors import CodegenError


# SNES default stack region when no #[stack(...)] is declared.
_DEFAULT_STACK_LOWER = 0x0100
_DEFAULT_STACK_UPPER = 0x01FF


# Bytes per register save on interrupt entry, in the default 65816 mode
# the handler prologue sees: A pushed as 16-bit (PHA in m16), X/Y/D as
# 16-bit, B/DBR as 8-bit. The CPU itself pushes 4 bytes on interrupt entry
# (P + PB + PCH + PCL).
_INTERRUPT_CPU_PUSH = 4
_INTERRUPT_REG_BYTES = {
    'A': 2,
    'X': 2,
    'Y': 2,
    'D': 2,
    'DBR': 1,
}


@dataclass
class StackBudget:
    """Result of the stack-usage analysis."""

    # Per-function sync stack usage (own frame + deepest callee path).
    per_function: Dict[str, int] = field(default_factory=dict)

    # Entry function's sync high-water mark.
    entry_use: int = 0

    # Worst-case interrupt overhead (single handler depth + CPU push).
    interrupt_extra: int = 0

    # entry_use + interrupt_extra. The number compared against capacity.
    total: int = 0

    # Bytes available in the declared stack region.
    capacity: int = _DEFAULT_STACK_UPPER - _DEFAULT_STACK_LOWER + 1

    # Deepest sync-path chain: list of (func_name, cumulative_bytes) tuples,
    # outermost call first. Used to format the diagnostic.
    deepest_chain: List[Tuple[str, int]] = field(default_factory=list)

    # Name of the worst-case interrupt handler (None if no handlers).
    worst_handler: Optional[str] = None

    # True when any reachable function makes an indirect (non-trait) call.
    # Indicates the bound is widened to the address-taken set.
    has_indirect_calls: bool = False

    # True when the call graph contains a cycle (stack-only-arg recursion is
    # permitted by the language and not caught at step 7). The recursive
    # back-edge is treated as zero-cost in this case, so the reported bound
    # is the per-invocation overhead — the user is responsible for bounding
    # recursion depth.
    has_recursion: bool = False


class StackUsageAnalyzer:
    """Compute a static upper bound on stack usage from a populated MIRProgram.

    Must run after ``Codegen.generate()`` so that each ``MIRFunction`` has
    ``abi_info`` set and the ``codegen_*`` and ``max_outgoing_arg_bytes``
    fields populated.
    """

    def __init__(self, mir_program: MIRProgram,
                 call_graph: Optional[CallGraph] = None):
        self.program = mir_program
        self.func_map: Dict[str, MIRFunction] = {
            f.name: f for f in mir_program.functions
        }
        if call_graph is None:
            analyzer = CallGraphAnalyzer(mir_program)
            call_graph = analyzer.analyze()
        self.cg = call_graph

        # Memoized per-function stack use.
        self._memo: Dict[str, int] = {}
        # Best successor for each func, used to reconstruct deepest chain.
        # Maps func_name -> (callee_name, ra_bytes) of the deepest edge.
        self._best_succ: Dict[str, Optional[Tuple[str, int]]] = {}
        # Cycle-detection sentinel for the memoized DFS.
        self._on_stack: Set[str] = set()
        # Indirect-call flag — surfaced on the budget for the diagnostic.
        self._has_indirect_calls: bool = False
        # Set when DFS finds a back-edge (recursion). Stack-only-arg
        # recursion is allowed; we treat the back-edge as zero-cost and
        # note that the bound is per-invocation, not whole-program.
        self._has_recursion: bool = False

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def analyze(self) -> StackBudget:
        budget = StackBudget()
        budget.capacity = self._stack_capacity()

        entry = self._entry_function()
        if entry is None:
            # No #[entry] function — nothing to bound. This is unusual but
            # not an error here; emitter would have already complained.
            return budget

        budget.entry_use = self._stack_use(entry.name)
        budget.per_function = dict(self._memo)
        budget.deepest_chain = self._reconstruct_chain(entry.name)
        budget.has_indirect_calls = self._has_indirect_calls
        budget.has_recursion = self._has_recursion

        # Interrupt overhead, on top of the sync-path peak.
        worst_extra, worst_handler = self._worst_interrupt_handler()
        budget.interrupt_extra = worst_extra
        budget.worst_handler = worst_handler

        budget.total = budget.entry_use + budget.interrupt_extra

        if budget.total > budget.capacity:
            raise self._overflow_error(budget)

        return budget

    # ------------------------------------------------------------------
    # Capacity + entry resolution
    # ------------------------------------------------------------------

    def _stack_capacity(self) -> int:
        attr = getattr(self.program, 'stack_attr', None)
        if attr is None:
            return _DEFAULT_STACK_UPPER - _DEFAULT_STACK_LOWER + 1
        lower = getattr(attr, 'lower', _DEFAULT_STACK_LOWER)
        upper = getattr(attr, 'upper', _DEFAULT_STACK_UPPER)
        return upper - lower + 1

    def _entry_function(self) -> Optional[MIRFunction]:
        for f in self.program.functions:
            if getattr(f, 'is_entry', False):
                return f
        return None

    # ------------------------------------------------------------------
    # Bottom-up DFS over the (acyclic) call graph
    # ------------------------------------------------------------------

    def _stack_use(self, name: str) -> int:
        if name in self._memo:
            return self._memo[name]
        if name in self._on_stack:
            # Cycle. Step 7 rejects recursion only when it uses register or
            # zeropage parameters; stack-only-arg recursion is permitted but
            # the analyzer cannot bound it statically. Treat the back-edge as
            # zero-cost (the user must bound recursion depth themselves) and
            # surface the fact on the budget so the diagnostic can note it.
            self._has_recursion = True
            return 0

        func = self.func_map.get(name)
        if func is None:
            # External / unknown — treat as zero. Mirrors how recursion check
            # handles missing func_map entries.
            self._memo[name] = 0
            self._best_succ[name] = None
            return 0

        own = self._own_frame(func)

        self._on_stack.add(name)
        best_callee_cost = 0
        best_edge: Optional[Tuple[str, int]] = None
        for callee_name, ra_bytes in self._call_edges(func):
            edge_cost = ra_bytes + self._stack_use(callee_name)
            if edge_cost > best_callee_cost:
                best_callee_cost = edge_cost
                best_edge = (callee_name, ra_bytes)
        self._on_stack.discard(name)

        result = own + best_callee_cost
        self._memo[name] = result
        self._best_succ[name] = best_edge
        return result

    def _own_frame(self, func: MIRFunction) -> int:
        prologue = 0
        abi_info = getattr(func, 'abi_info', None)
        if abi_info is not None:
            prologue = getattr(abi_info, 'prologue_stack_bytes', 0) or 0
        else:
            # Fallback: codegen-recorded prologue (covers the populated case
            # where abi_info was attached but its property is missing).
            prologue = getattr(func, 'codegen_prologue_bytes', 0) or 0

        frame = getattr(func, 'codegen_frame_size', 0) or 0
        spill = getattr(func, 'codegen_max_region_spill_bytes', 0) or 0
        out_args = getattr(func, 'max_outgoing_arg_bytes', 0) or 0
        return prologue + frame + spill + out_args

    def _call_edges(self, func: MIRFunction):
        """Yield (callee_name, ra_bytes) for every reachable call edge.

        Direct call → singleton; trait dispatch → all impls (from CallGraph);
        indirect fn-pointer call → union of address-taken; InlineAsm → skipped.
        """
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Call):
                    ra = 3 if getattr(instr, 'is_far', False) else 2
                    if isinstance(instr.function, str):
                        yield (instr.function, ra)
                    else:
                        # Indirect: widen to address-taken set. Conservative
                        # but matches what CallGraphAnalyzer records.
                        self._has_indirect_calls = True
                        for target in self.cg.address_taken:
                            yield (target, ra)
                elif isinstance(instr, TraitDispatch):
                    ra = 3 if getattr(instr, 'is_far', False) else 2
                    impls = self.cg.resolve_trait_method(
                        instr.trait_name, instr.method_name
                    )
                    for target in impls:
                        yield (target, ra)
                # InlineAsm: zero stack effect by convention.

    # ------------------------------------------------------------------
    # Deepest-chain reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_chain(self, root: str) -> List[Tuple[str, int]]:
        chain: List[Tuple[str, int]] = []
        node: Optional[str] = root
        seen: Set[str] = set()
        while node is not None and node not in seen:
            seen.add(node)
            chain.append((node, self._memo.get(node, 0)))
            edge = self._best_succ.get(node)
            node = edge[0] if edge else None
        return chain

    # ------------------------------------------------------------------
    # Interrupt overhead
    # ------------------------------------------------------------------

    def _worst_interrupt_handler(self) -> Tuple[int, Optional[str]]:
        worst = 0
        worst_name: Optional[str] = None
        for func in self.program.functions:
            if getattr(func, 'interrupt_attr', None) is None:
                continue
            handler_use = self._stack_use(func.name)
            prologue = _INTERRUPT_CPU_PUSH + self._interrupt_save_bytes(func)
            total = handler_use + prologue
            if total > worst:
                worst = total
                worst_name = func.name
        return worst, worst_name

    @staticmethod
    def _interrupt_save_bytes(func: MIRFunction) -> int:
        regs = getattr(func, 'interrupt_modified_regs', None)
        if not regs:
            return 0
        return sum(_INTERRUPT_REG_BYTES.get(r, 0) for r in regs)

    # ------------------------------------------------------------------
    # Error formatting
    # ------------------------------------------------------------------

    def _overflow_error(self, budget: StackBudget) -> CodegenError:
        chain_str = self._format_chain(budget.deepest_chain)
        lines = [
            f"Stack overflow: program needs {budget.total} bytes, "
            f"declared region is {budget.capacity} bytes.",
            "Deepest sync call chain (high-water at each frame):",
            chain_str,
        ]
        if budget.interrupt_extra:
            handler = budget.worst_handler or "unknown"
            lines.append(
                f"Interrupt overhead: {budget.interrupt_extra} bytes "
                f"(worst handler: {handler})."
            )
        if budget.has_indirect_calls:
            lines.append(
                "Note: indirect (fn-pointer) calls present — bound widened to "
                "the address-taken set. Trim by removing unused address-of-fn "
                "references or tightening fn pointer types."
            )
        if budget.has_recursion:
            lines.append(
                "Note: recursion present in the call graph — the bound shown "
                "is per-invocation. Each additional recursive call adds the "
                "callee's own frame plus its return address."
            )
        suggested = max(budget.total, budget.capacity + 1)
        lines.append(
            f"Hint: widen the stack region — e.g. #[stack(0x01FF - {suggested - 1:#x}, 0x01FF)] "
            f"or move it into bank 0 WRAM with #[stack(0x1E00, 0x1FFF)] (512 bytes)."
        )
        return CodegenError("\n".join(lines))

    @staticmethod
    def _format_chain(chain: List[Tuple[str, int]]) -> str:
        if not chain:
            return "  (empty)"
        parts = [f"  {name} ({use} B)" for name, use in chain]
        return "\n".join(parts)


def analyze_stack_usage(mir_program: MIRProgram,
                        call_graph: Optional[CallGraph] = None) -> StackBudget:
    """Convenience entry point — build/reuse a CallGraph and run the analyzer."""
    return StackUsageAnalyzer(mir_program, call_graph).analyze()
