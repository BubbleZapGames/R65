"""
Scratch register usage analysis across call graph.

Analyzes which scratch registers each function uses (directly and transitively
through callees) to enable call-graph-aware register allocation.

Handles indirect calls (function pointers) conservatively by tracking which
functions have their address taken and assuming any of them could be called.
"""

from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, Call, VirtualRegister,
    Load, Store, Move, FunctionPointer
)


@dataclass
class ScratchUsageInfo:
    """Scratch usage information for a single function."""
    # Scratch addresses used directly by this function
    direct_usage: Set[int] = field(default_factory=set)
    # Scratch addresses used by this function and all its callees (transitively)
    transitive_usage: Set[int] = field(default_factory=set)


class ScratchUsageAnalyzer:
    """
    Analyzes scratch register usage across call graph.

    Computes which scratch registers each function uses, then propagates
    that usage up through the call graph so callers know which scratches
    are safe to use for variables that live across calls.

    Key insight: If a variable is live across a call, it cannot use any
    scratch register that the callee (or any of the callee's callees) might use.

    For indirect calls (function pointers), we conservatively assume that
    any function whose address is taken could be called. This means variables
    live across indirect calls must avoid scratches used by ALL address-taken
    functions.
    """

    def __init__(self, mir_program: MIRProgram, scratch_addresses: Set[int]):
        """
        Initialize scratch usage analyzer.

        Args:
            mir_program: MIR program to analyze
            scratch_addresses: Set of scratch register addresses available
        """
        self.program = mir_program
        self.all_scratches = scratch_addresses
        self.func_map: Dict[str, MIRFunction] = {
            func.name: func for func in mir_program.functions
        }

        # Call graph: func_name -> set of directly called function names
        self.call_graph: Dict[str, Set[str]] = {}

        # Reverse call graph for bottom-up propagation
        self.callers: Dict[str, Set[str]] = {}

        # Usage info per function
        self.usage: Dict[str, ScratchUsageInfo] = {}

        # Functions whose address is taken (could be called indirectly)
        self.address_taken_funcs: Set[str] = set()

        # Functions that have indirect calls
        self.has_indirect_call: Set[str] = set()

    def analyze(self) -> Dict[str, ScratchUsageInfo]:
        """
        Perform scratch usage analysis.

        Returns:
            Dictionary mapping function names to their scratch usage info
        """
        # Step 1: Build call graph (including indirect call tracking)
        self._build_call_graph()

        # Step 2: Find functions whose address is taken
        self._find_address_taken_functions()

        # Step 3: Compute direct scratch usage per function
        # For now, we can't know direct usage until after allocation,
        # so we initialize with empty sets. The actual usage will be
        # tracked during allocation.
        self._initialize_usage()

        # Step 4: Propagate usage through call graph (not needed initially
        # since direct usage is empty - will be done incrementally)

        return self.usage

    def _build_call_graph(self):
        """Build call graph from MIR program."""
        # Initialize all functions
        for func in self.program.functions:
            self.call_graph[func.name] = set()
            self.callers[func.name] = set()

        # Build edges
        for func in self.program.functions:
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call):
                        if isinstance(instr.function, str):
                            # Direct call
                            callee_name = instr.function
                            self.call_graph[func.name].add(callee_name)

                            # Add to reverse graph
                            if callee_name not in self.callers:
                                self.callers[callee_name] = set()
                            self.callers[callee_name].add(func.name)
                        else:
                            # Indirect call (function pointer)
                            self.has_indirect_call.add(func.name)

    def _find_address_taken_functions(self):
        """
        Find all functions whose address is taken.

        These functions could potentially be called through function pointers,
        so we need to be conservative about scratch allocation for indirect calls.
        """
        for func in self.program.functions:
            for block in func.blocks.values():
                for instr in block.instructions:
                    # Check for FunctionPointer operands
                    self._check_for_function_pointers(instr)

    def _check_for_function_pointers(self, instr):
        """Check an instruction for function pointer references."""
        # Check all attributes for FunctionPointer
        for attr_name in ['source', 'value', 'dest', 'left', 'right']:
            if hasattr(instr, attr_name):
                val = getattr(instr, attr_name)
                if isinstance(val, FunctionPointer):
                    self.address_taken_funcs.add(val.function_name)

    def _initialize_usage(self):
        """Initialize usage info for all functions."""
        for func_name in self.func_map:
            self.usage[func_name] = ScratchUsageInfo()

    def register_scratch_usage(self, func_name: str, scratch_addr: int):
        """
        Register that a function uses a scratch register.

        Called by the register allocator when allocating a scratch.
        Propagates usage to all callers.

        Args:
            func_name: Name of function using the scratch
            scratch_addr: Address of scratch register
        """
        if func_name not in self.usage:
            self.usage[func_name] = ScratchUsageInfo()

        info = self.usage[func_name]
        if scratch_addr in info.direct_usage:
            return  # Already tracked

        info.direct_usage.add(scratch_addr)
        info.transitive_usage.add(scratch_addr)

        # Propagate to all callers (transitively)
        self._propagate_to_callers(func_name, scratch_addr)

    def _propagate_to_callers(self, func_name: str, scratch_addr: int):
        """Propagate scratch usage to all callers transitively."""
        visited = set()
        to_process = [func_name]

        while to_process:
            current = to_process.pop()
            if current in visited:
                continue
            visited.add(current)

            for caller in self.callers.get(current, set()):
                if caller not in self.usage:
                    self.usage[caller] = ScratchUsageInfo()

                caller_info = self.usage[caller]
                if scratch_addr not in caller_info.transitive_usage:
                    caller_info.transitive_usage.add(scratch_addr)
                    to_process.append(caller)

    def get_available_scratches(self, func_name: str,
                                live_across_call: bool,
                                live_across_indirect_call: bool = False) -> Set[int]:
        """
        Get scratch registers available for allocation in a function.

        Args:
            func_name: Name of function doing the allocation
            live_across_call: True if the variable being allocated lives
                              across a call (needs to avoid callee scratches)
            live_across_indirect_call: True if the variable lives across an
                              indirect call (needs to avoid ALL address-taken
                              function scratches)

        Returns:
            Set of scratch addresses available for use
        """
        if not live_across_call:
            # Variable dies before any call - can use any scratch
            return self.all_scratches.copy()

        # Variable lives across a call - exclude scratches used by callees
        if func_name not in self.usage:
            return self.all_scratches.copy()

        # Get transitive usage of all direct callees
        callee_usage = set()
        for callee in self.call_graph.get(func_name, set()):
            if callee in self.usage:
                callee_usage |= self.usage[callee].transitive_usage

        # If live across indirect call, also exclude scratches from ALL
        # address-taken functions (conservative approach)
        if live_across_indirect_call or func_name in self.has_indirect_call:
            for addr_taken_func in self.address_taken_funcs:
                if addr_taken_func in self.usage:
                    callee_usage |= self.usage[addr_taken_func].transitive_usage

        return self.all_scratches - callee_usage

    def func_has_indirect_calls(self, func_name: str) -> bool:
        """Check if a function contains any indirect calls."""
        return func_name in self.has_indirect_call

    def get_address_taken_functions(self) -> Set[str]:
        """Get the set of functions whose address is taken."""
        return self.address_taken_funcs.copy()

    def get_transitive_usage(self, func_name: str) -> Set[int]:
        """
        Get all scratches used by a function and its callees.

        Args:
            func_name: Function name

        Returns:
            Set of scratch addresses used transitively
        """
        if func_name not in self.usage:
            return set()
        return self.usage[func_name].transitive_usage.copy()

    def propagate_all(self):
        """
        Propagate all scratch usage through the call graph.

        Should be called after all direct usage has been registered,
        to ensure transitive usage is fully computed.
        """
        # Topological sort (reverse) - process functions with no callees first
        # Then propagate upward through callers

        # Find functions with no callees (leaf functions)
        processed = set()
        to_process = []

        for func_name in self.func_map:
            callees = self.call_graph.get(func_name, set())
            # Only consider internal callees
            internal_callees = callees & set(self.func_map.keys())
            if not internal_callees:
                to_process.append(func_name)

        # BFS propagation from leaves upward
        while to_process:
            current = to_process.pop(0)
            if current in processed:
                continue
            processed.add(current)

            # Ensure this function's transitive usage includes all callee usage
            current_info = self.usage.get(current, ScratchUsageInfo())
            for callee in self.call_graph.get(current, set()):
                if callee in self.usage:
                    current_info.transitive_usage |= self.usage[callee].transitive_usage

            if current not in self.usage:
                self.usage[current] = current_info
            else:
                self.usage[current].transitive_usage = current_info.transitive_usage

            # Add callers to process
            for caller in self.callers.get(current, set()):
                if caller not in processed:
                    to_process.append(caller)
