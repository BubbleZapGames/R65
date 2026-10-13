# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Dead Function Elimination.

Removes functions that are never called from the program.
Entry points (marked with #[entry]), interrupt handlers and #[used]
functions are always kept.
"""

from typing import Set, List
from r65.compiler.mir.nodes import MIRProgram, MIRFunction
from r65.compiler.analysis.call_graph import build_call_graph


class DeadFunctionEliminator:
    """
    Eliminates unused functions from a MIR program.

    A function is considered "live" (not dead) if:
    1. It is an entry point (is_entry=True) or #[used]
    2. It is an interrupt handler (interrupt_attr is not None)
    3. It is called by another live function
    4. It is the initialization function (__init_start)
    5. It is a trait implementor (reached via a dispatch table) or named by
       a `fn` static in ROM

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

        # Trait implementors are reached through dispatch tables, and a `fn`
        # static in ROM through code this pass cannot follow.
        graph = build_call_graph(mir_program)
        reachable = graph.reachable_from(roots | graph.dispatch_targets | graph.rom_refs)

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
        - Functions marked with #[entry] or #[used]
        - Interrupt handlers (marked with #[interrupt(...)])
        - The __init_start initialization function

        Args:
            functions: List of MIR functions

        Returns:
            Set of function names that are entry points
        """
        entry_points = set()

        for func in functions:
            # Entry function, or one kept for code the compiler cannot see
            if func.is_entry or func.is_used:
                entry_points.add(func.name)

            # Interrupt handler
            if func.interrupt_attr:
                entry_points.add(func.name)

            # Initialization function (always called by entry)
            if func.name == "__init_start":
                entry_points.add(func.name)

        return entry_points
