# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
The program's call graph, built once from MIR for every pass that asks who
calls whom.

It also detects recursion, to reject zero-page or register parameters on a
function that can re-enter itself.
"""

from typing import Dict, Iterable, List, Set, Tuple
from dataclasses import dataclass, field
from r65.compiler.mir.nodes import (
    CallBase,
    MIRProgram, Call, TraitDispatch, Move, FunctionPointer, SymbolByte,
)
from r65.compiler.errors import DiagnosticCollector
from r65.compiler.trait_dispatch import trait_impls_table


class RecursionError(Exception):
    """Raised when unsafe recursion is detected."""
    pass


@dataclass
class CallGraph:
    """Who calls whom, and whose address is taken.

    `calls` holds the direct calls. `edges` adds, for each TraitDispatch, an
    edge to every implementor it may reach. `refs` holds the functions each
    function takes the address of, through a `FunctionPointer` moved or passed
    as an argument; `rom_refs` those named by a `fn` static in ROM.
    `address_taken` is every function reachable by a pointer: all of these.
    """
    calls: Dict[str, Set[str]] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    refs: Dict[str, Set[str]] = field(default_factory=dict)
    rom_refs: Set[str] = field(default_factory=set)
    address_taken: Set[str] = field(default_factory=set)
    # (trait_name, method_name) -> the implementors a dispatch may invoke.
    trait_impls: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)

    def get_callees(self, func_name: str) -> Set[str]:
        """Every function `func_name` calls, directly or by trait dispatch."""
        return self.edges.get(func_name, set())

    def callers_of(self, func_name: str, direct: bool = False) -> Set[str]:
        """The functions that call `func_name`; only direct calls if `direct`."""
        graph = self.calls if direct else self.edges
        return {caller for caller, callees in graph.items() if func_name in callees}

    def transitive_callers(self, func_name: str) -> Set[str]:
        """Every function from which a chain of calls reaches `func_name`."""
        return self._closure(func_name, self.callers_of)

    def transitive_callees(self, func_name: str) -> Set[str]:
        """Every function a chain of calls from `func_name` reaches."""
        return self._closure(func_name, self.get_callees)

    @staticmethod
    def _closure(start: str, step) -> Set[str]:
        found: Set[str] = set()
        frontier = [start]
        while frontier:
            for name in step(frontier.pop()):
                if name not in found:
                    found.add(name)
                    frontier.append(name)
        return found

    def reachable_from(self, roots: Iterable[str]) -> Set[str]:
        """Every function `roots` can reach by a call or a taken address."""
        reachable: Set[str] = set()
        worklist = list(roots)
        while worklist:
            name = worklist.pop()
            if name in reachable:
                continue
            reachable.add(name)
            worklist.extend(self.edges.get(name, ()))
            worklist.extend(self.refs.get(name, ()))
        return reachable

    @property
    def dispatch_targets(self) -> Set[str]:
        """Every trait implementor, which a dispatch table reaches with no `Call`."""
        return set().union(*self.trait_impls.values())

    def resolve_trait_method(self, trait_name: str, method_name: str) -> Set[str]:
        """The implementors a dispatch on `trait_name::method_name` may invoke."""
        return self.trait_impls.get((trait_name, method_name), set())

    def find_cycles(self, direct: bool = False) -> List[List[str]]:
        """Each recursion cycle (strongly connected component), by Tarjan's
        algorithm. A one-function cycle is direct recursion. Trait dispatch
        counts as a call unless `direct`.
        """
        graph = self.calls if direct else self.edges
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        cycles = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            for successor in graph.get(node, ()):
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    lowlinks[node] = min(lowlinks[node], index[successor])

            if lowlinks[node] == index[node]:
                component = []
                while True:
                    successor = stack.pop()
                    on_stack.remove(successor)
                    component.append(successor)
                    if successor == node:
                        break
                if len(component) > 1 or node in graph.get(node, ()):
                    cycles.append(component)

        nodes = set(graph).union(*graph.values())
        for node in nodes:
            if node not in index:
                strongconnect(node)
        return cycles

    def recursive_functions(self, direct: bool = False) -> Set[str]:
        """Every function on a recursion cycle."""
        return {name for cycle in self.find_cycles(direct) for name in cycle}


def build_call_graph(program: MIRProgram) -> CallGraph:
    """Build the call graph of every function in `program`."""
    graph = CallGraph()
    graph.trait_impls = trait_impls_table(program.trait_dispatch_info)

    for func in program.functions:
        calls = graph.calls.setdefault(func.name, set())
        edges = graph.edges.setdefault(func.name, set())
        refs = graph.refs.setdefault(func.name, set())
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Call) and isinstance(instr.function, str):
                    calls.add(instr.function)
                    edges.add(instr.function)
                elif isinstance(instr, TraitDispatch):
                    edges |= graph.resolve_trait_method(instr.trait_name, instr.method_name)
                if isinstance(instr, Move) and isinstance(instr.source, FunctionPointer):
                    refs.add(instr.source.function_name)
                elif isinstance(instr, CallBase):
                    refs.update(arg.value.function_name for arg in instr.args
                                if isinstance(arg.value, FunctionPointer))
        graph.address_taken |= refs

    names = set(graph.calls)
    for rom_data in program.rom_data_sections:
        for byte in rom_data.data:
            if isinstance(byte, SymbolByte) and byte.label in names:
                graph.rom_refs.add(byte.label)
    graph.address_taken |= graph.rom_refs
    return graph


class RecursionChecker:
    """
    Checks for unsafe recursion with zero-page or register parameters.

    Errors raise; warnings (address-taken functions) collect in
    `self.diagnostics` for the caller to report.
    """

    def __init__(self, mir_program: MIRProgram):
        self.program = mir_program
        self.func_map = {func.name: func for func in mir_program.functions}
        self.diagnostics = DiagnosticCollector()

    def check(self):
        """
        Check for unsafe recursion.

        Raises:
            RecursionError: If unsafe recursion is detected
        """
        graph = build_call_graph(self.program)
        cycles = graph.find_cycles()

        # Mark recursive functions on MIRFunction nodes
        for cycle in cycles:
            for func_name in cycle:
                if func_name in self.func_map:
                    self.func_map[func_name].is_recursive = True

        # Check each cycle for unsafe parameters
        for cycle in cycles:
            self._check_cycle(cycle, graph)

        # Check functions with address taken
        self._check_address_taken(graph)

    def _check_cycle(self, cycle: List[str], graph: CallGraph):
        """
        Check if a cycle contains functions with unsafe parameters.

        Args:
            cycle: List of function names in the cycle
            graph: Call graph

        Raises:
            RecursionError: If unsafe recursion detected
        """
        for func_name in cycle:
            if func_name not in self.func_map:
                continue  # External function

            func = self.func_map[func_name]

            # Check each parameter
            for param in func.parameters:
                # Check HIRParameter.binding (should be RegisterBinding or VariableBinding)
                if hasattr(param, 'binding') and param.binding:
                    from r65.compiler.hir.nodes import RegisterBinding, VariableBinding

                    if isinstance(param.binding, RegisterBinding):
                        # Register parameter (e.g., param @ A)
                        self._raise_recursion_error(
                            func_name, param.name, cycle,
                            "register", param.binding.register_name
                        )
                    elif isinstance(param.binding, VariableBinding):
                        # Variable-bound parameter (e.g., param @ TEMP)
                        # Check if the variable is zero-page
                        var_symbol = param.binding.variable_symbol
                        if var_symbol and hasattr(var_symbol, 'definition'):
                            var_def = var_symbol.definition
                            if hasattr(var_def, 'storage_attr') and var_def.storage_attr:
                                from r65.compiler.hir.attributes import StorageKind
                                if var_def.storage_attr.storage_kind == StorageKind.ZEROPAGE:
                                    self._raise_recursion_error(
                                        func_name, param.name, cycle,
                                        "zero-page", param.binding.variable_name
                                    )

            # Check for promoted aggregate locals (static allocation is unsafe in recursion)
            if func.has_promoted_locals:
                self._raise_promoted_local_error(func_name, cycle)

    def _raise_promoted_local_error(self, func_name: str, cycle: List[str]):
        """
        Raise a RecursionError for functions with promoted aggregate locals.

        Promoted locals use static storage (only one copy exists), so recursive
        calls would corrupt the caller's data.
        """
        if len(cycle) == 1:
            msg = (
                f"Function '{func_name}' is directly recursive but has local "
                f"struct/array variables. Local aggregate variables are promoted "
                f"to static storage and would be corrupted by recursive calls.\n\n"
                f"Hint: Use static variables with explicit storage attributes, "
                f"or restructure to avoid recursion."
            )
        else:
            cycle_str = " -> ".join(cycle)
            msg = (
                f"Function '{func_name}' is part of a recursion cycle but has "
                f"local struct/array variables. Local aggregate variables are "
                f"promoted to static storage and would be corrupted by recursive "
                f"calls.\n\n"
                f"Recursion cycle: {cycle_str}"
            )
        raise RecursionError(msg)

    def _check_address_taken(self, graph: CallGraph):
        """
        Warn about functions with zero-page/register params that have address taken.

        Functions whose address is taken can be called indirectly, which means
        the caller cannot guarantee that parameters are set up correctly.

        Args:
            graph: Call graph
        """
        diagnostics = self.diagnostics

        for func_name in graph.address_taken:
            if func_name not in self.func_map:
                continue

            func = self.func_map[func_name]

            # Check for zero-page or register parameters
            for param in func.parameters:
                if hasattr(param, 'binding') and param.binding:
                    from r65.compiler.hir.nodes import RegisterBinding, VariableBinding

                    if isinstance(param.binding, RegisterBinding):
                        # Register parameter on function whose address is taken
                        diagnostics.warning(
                            f"Function '{func_name}' has register parameter "
                            f"'{param.name}' bound to '{param.binding.register_name}', "
                            f"but its address is taken for indirect calls",
                            code="W001",
                            hint=f"Indirect callers must ensure {param.binding.register_name} "
                                 f"is set before calling. Consider using stack parameters."
                        )
                    elif isinstance(param.binding, VariableBinding):
                        # Variable-bound parameter - check if zero-page
                        var_symbol = param.binding.variable_symbol
                        if var_symbol and hasattr(var_symbol, 'definition'):
                            var_def = var_symbol.definition
                            if hasattr(var_def, 'storage_attr') and var_def.storage_attr:
                                from r65.compiler.hir.attributes import StorageKind
                                if var_def.storage_attr.storage_kind == StorageKind.ZEROPAGE:
                                    diagnostics.warning(
                                        f"Function '{func_name}' has zero-page parameter "
                                        f"'{param.name}' bound to '{param.binding.variable_name}', "
                                        f"but its address is taken for indirect calls",
                                        code="W002",
                                        hint="Indirect callers must set the zero-page variable "
                                             "before calling. Consider using stack parameters."
                                    )

            # Warn about promoted aggregate locals
            if func.has_promoted_locals:
                diagnostics.warning(
                    f"Function '{func_name}' has local struct/array variables "
                    f"promoted to static storage, but its address is taken for "
                    f"indirect calls. Concurrent or recursive indirect calls "
                    f"would corrupt the static storage.",
                    code="W003",
                    hint="Consider using static variables with explicit storage "
                         "attributes if the function may be called indirectly."
                )

    def _raise_recursion_error(self, func_name: str, param_name: str,
                               cycle: List[str], param_type: str,
                               binding_name: str):
        """
        Raise a RecursionError with detailed message.

        Args:
            func_name: Name of the function with unsafe parameter
            param_name: Name of the parameter
            cycle: List of functions in the recursion cycle
            param_type: "zero-page" or "register"
            binding_name: Name of the zero-page variable or register
        """
        if len(cycle) == 1:
            # Direct recursion
            msg = (
                f"Function '{func_name}' is directly recursive but has "
                f"{param_type} parameter '{param_name}' bound to '{binding_name}'. "
                f"{param_type.capitalize()} parameters are not preserved across "
                f"function calls.\n\n"
                f"Hint: Use stack parameters instead:\n"
                f"  fn {func_name}({param_name}: type) -> ... {{"
            )
        else:
            # Indirect recursion
            cycle_str = " -> ".join(cycle)
            msg = (
                f"Function '{func_name}' is part of a recursion cycle but has "
                f"{param_type} parameter '{param_name}' bound to '{binding_name}'. "
                f"{param_type.capitalize()} parameters are not preserved across "
                f"function calls.\n\n"
                f"Recursion cycle: {cycle_str}\n\n"
                f"Hint: Use stack parameters for at least one function in the cycle."
            )

        raise RecursionError(msg)
