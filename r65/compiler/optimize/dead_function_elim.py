"""
Dead Function Elimination.

Removes functions that are never called from the program.
Entry points (marked with #[entry]) and interrupt handlers are always kept.
"""

from typing import Set, List
from r65.compiler.mir.nodes import MIRProgram, MIRFunction, Call, TraitDispatch


class DeadFunctionEliminator:
    """
    Eliminates unused functions from a MIR program.

    A function is considered "live" (not dead) if:
    1. It is an entry point (is_entry=True)
    2. It is an interrupt handler (interrupt_attr is not None)
    3. It is called by another live function
    4. It is the initialization function (__init_start)

    All other functions are removed from the program.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the dead function eliminator.

        Args:
            verbose: If True, print information about eliminated functions
        """
        self.verbose = verbose

    def eliminate(self, mir_program: MIRProgram) -> int:
        """
        Eliminate dead functions from the MIR program.

        Modifies mir_program.functions in place to remove unreachable functions.

        Args:
            mir_program: The MIR program to optimize

        Returns:
            Number of functions eliminated
        """
        # Build function name -> function mapping
        func_map = {func.name: func for func in mir_program.functions}

        # Find all entry points (roots for reachability analysis)
        roots = self._find_entry_points(mir_program.functions)

        # If no entry points exist, skip optimization (likely a unit test)
        if not roots:
            return 0

        # Mark all trait implementor methods as reachable (called via jump tables)
        if hasattr(mir_program, 'trait_dispatch_info') and mir_program.trait_dispatch_info:
            for trait_info in mir_program.trait_dispatch_info.values():
                for impl_info in trait_info.get('implementors', []):
                    for mangled_name in impl_info.get('mangled', []):
                        roots.add(mangled_name)

        # Build call graph
        call_graph = self._build_call_graph(mir_program.functions)

        # Find all reachable functions via transitive closure
        reachable = self._find_reachable(roots, call_graph)

        # Filter out unreachable functions
        original_count = len(mir_program.functions)
        mir_program.functions = [
            func for func in mir_program.functions
            if func.name in reachable
        ]
        eliminated_count = original_count - len(mir_program.functions)

        if self.verbose and eliminated_count > 0:
            eliminated_names = set(func_map.keys()) - reachable
            print(f"Dead function elimination: removed {eliminated_count} function(s)")
            for name in sorted(eliminated_names):
                print(f"  - {name}")

        return eliminated_count

    def _find_entry_points(self, functions: List[MIRFunction]) -> Set[str]:
        """
        Find all entry point functions.

        Entry points are:
        - Functions marked with #[entry]
        - Interrupt handlers (marked with #[interrupt(...)])
        - The __init_start initialization function

        Args:
            functions: List of MIR functions

        Returns:
            Set of function names that are entry points
        """
        entry_points = set()

        for func in functions:
            # Entry function
            if func.is_entry:
                entry_points.add(func.name)

            # Interrupt handler
            if func.interrupt_attr:
                entry_points.add(func.name)

            # Initialization function (always called by entry)
            if func.name == "__init_start":
                entry_points.add(func.name)

        return entry_points

    def _build_call_graph(self, functions: List[MIRFunction]) -> dict:
        """
        Build a call graph mapping function names to called functions.

        Args:
            functions: List of MIR functions

        Returns:
            Dict mapping function name to set of called function names
        """
        call_graph = {}

        for func in functions:
            called_functions = set()

            # Scan all blocks for Call instructions
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call):
                        # Direct calls have function name as string
                        if isinstance(instr.function, str):
                            called_functions.add(instr.function)
                        # Indirect calls through function pointers - can't determine
                        # statically, so we can't eliminate any function that might
                        # be called through a pointer. For now, we only handle direct calls.
                    elif isinstance(instr, TraitDispatch):
                        # Trait dispatch calls all implementor methods via jump table.
                        # Mark the dispatch wrapper and all implementors as called.
                        dispatch_name = f"{instr.trait_name}__{instr.method_name}__dispatch"
                        called_functions.add(dispatch_name)

            call_graph[func.name] = called_functions

        return call_graph

    def _find_reachable(self, roots: Set[str], call_graph: dict) -> Set[str]:
        """
        Find all functions reachable from the given roots.

        Uses breadth-first search to find transitive closure.

        Args:
            roots: Set of entry point function names
            call_graph: Dict mapping function name to called functions

        Returns:
            Set of all reachable function names
        """
        reachable = set()
        worklist = list(roots)

        while worklist:
            func_name = worklist.pop()

            if func_name in reachable:
                continue

            reachable.add(func_name)

            # Add all called functions to worklist
            if func_name in call_graph:
                for called in call_graph[func_name]:
                    if called not in reachable:
                        worklist.append(called)

        return reachable
