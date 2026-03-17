# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Call graph analysis for detecting recursion and function call patterns.

Builds a directed graph of function calls and detects cycles (recursion)
to check for unsafe use of zero-page or register parameters.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from r65.compiler.mir.nodes import MIRProgram, MIRFunction, Call, TraitDispatch
from r65.compiler.errors import get_diagnostics


class RecursionError(Exception):
    """Raised when unsafe recursion is detected."""
    pass


@dataclass
class CallGraph:
    """
    Directed graph of function calls.

    Nodes are function names, edges represent calls.
    """
    # Adjacency list: function_name -> set of called functions
    edges: Dict[str, Set[str]] = field(default_factory=dict)

    # Functions that have their address taken (potential indirect calls)
    address_taken: Set[str] = field(default_factory=set)

    # Functions that make indirect calls (via function pointers)
    indirect_callers: Set[str] = field(default_factory=set)

    def add_edge(self, caller: str, callee: str):
        """Add a call edge from caller to callee."""
        if caller not in self.edges:
            self.edges[caller] = set()
        self.edges[caller].add(callee)

    def add_address_taken(self, func_name: str):
        """Mark that a function's address is taken."""
        self.address_taken.add(func_name)

    def get_callees(self, func_name: str) -> Set[str]:
        """Get all functions called by the given function."""
        return self.edges.get(func_name, set())

    def get_all_functions(self) -> Set[str]:
        """Get all functions in the graph."""
        functions = set(self.edges.keys())
        for callees in self.edges.values():
            functions.update(callees)
        return functions


class CallGraphAnalyzer:
    """
    Analyzes function calls to build a call graph.
    """

    def __init__(self, mir_program: MIRProgram):
        """
        Initialize analyzer.

        Args:
            mir_program: MIR program to analyze
        """
        self.program = mir_program
        self.graph = CallGraph()

    def analyze(self) -> CallGraph:
        """
        Build call graph from MIR program.

        Returns:
            CallGraph with all function call relationships
        """
        # Build function name to function mapping
        func_map = {func.name: func for func in self.program.functions}

        # Analyze each function
        for func in self.program.functions:
            self._analyze_function(func)

        return self.graph

    def _analyze_function(self, func: MIRFunction):
        """Analyze a single function to find calls."""
        # Scan all blocks for Call instructions
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Call):
                    # Only track direct calls (function name is a string)
                    # Indirect calls (function pointers) are handled separately
                    if isinstance(instr.function, str):
                        self.graph.add_edge(func.name, instr.function)
                    else:
                        # Indirect call via function pointer - track caller for warnings
                        # The caller might be calling any function whose address is taken
                        self.graph.indirect_callers.add(func.name)
                elif isinstance(instr, TraitDispatch):
                    # Trait dispatch is an indirect call — track as indirect caller
                    self.graph.indirect_callers.add(func.name)

    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles (strongly connected components) in the call graph.

        Uses Tarjan's algorithm for finding SCCs.

        Returns:
            List of cycles, where each cycle is a list of function names.
            Single-node cycles indicate direct recursion.
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        cycles = []

        def strongconnect(node):
            # Set the depth index for this node to the smallest unused index
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            # Consider successors of node
            for successor in self.graph.get_callees(node):
                if successor not in index:
                    # Successor has not yet been visited; recurse on it
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    # Successor is in stack and hence in the current SCC
                    lowlinks[node] = min(lowlinks[node], index[successor])

            # If node is a root node, pop the stack and create an SCC
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    successor = stack.pop()
                    on_stack.remove(successor)
                    component.append(successor)
                    if successor == node:
                        break

                # Only add if it's actually a cycle (size > 1 or self-loop)
                if len(component) > 1:
                    cycles.append(component)
                elif node in self.graph.get_callees(node):
                    # Self-loop (direct recursion)
                    cycles.append(component)

        # Run algorithm on all nodes
        for node in self.graph.get_all_functions():
            if node not in index:
                strongconnect(node)

        return cycles


class RecursionChecker:
    """
    Checks for unsafe recursion with zero-page or register parameters.
    """

    def __init__(self, mir_program: MIRProgram):
        """
        Initialize recursion checker.

        Args:
            mir_program: MIR program to check
        """
        self.program = mir_program
        self.func_map = {func.name: func for func in mir_program.functions}

    def check(self):
        """
        Check for unsafe recursion.

        Raises:
            RecursionError: If unsafe recursion is detected
        """
        # Build call graph
        analyzer = CallGraphAnalyzer(self.program)
        graph = analyzer.analyze()

        # Find cycles
        cycles = analyzer.find_cycles()

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
        diagnostics = get_diagnostics()

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
