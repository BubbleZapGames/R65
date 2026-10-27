# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Far-to-Near Call Optimization Pass.

Converts far function calls (JSL/RTL) to near function calls (JSR/RTS) when
call graph analysis shows that all callers and the callee are in the same bank.

This optimization saves:
- 1 byte per call (JSR is 3 bytes, JSL is 4 bytes)
- 2 cycles per call (JSR is 6 cycles, JSL is 8 cycles)
- 0 cycles on return (both RTS and RTL are 6 cycles)

The optimization is safe when:
1. All callers are in the same bank as the callee
2. The function's address is never taken (no indirect calls)
3. The function is not an interrupt handler (called by hardware from any context)
4. The function is not a `*dyn` dispatch target — a far trait's dispatch
   wrapper reaches its implementors through `JML` trampolines that
   expect an `RTL`, and those edges are not `Call` instructions.
5. The function has at least one direct caller.
6. The function has no `#[mode(databank=...)]`: DBR management is defined
   for far functions only, and codegen emits it only while `is_far` holds.
"""

from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass, field

from r65.compiler.mir.nodes import MIRProgram, MIRFunction, Call
from r65.compiler.analysis.call_graph import CallGraph, build_call_graph
# The bank #[bank(auto)] functions are placed in; codegen places them by it.
from r65.compiler.codegen.symbol_gen import auto_bank_number as compute_auto_bank_num


@dataclass
class FunctionBankInfo:
    """Information about a function's bank placement and call relationships."""
    name: str
    bank: int
    is_far: bool
    has_address_taken: bool = False
    is_interrupt: bool = False
    manages_dbr: bool = False
    callers: Set[str] = field(default_factory=set)


def effective_bank(func: MIRFunction, auto_bank_num: int) -> int:
    """The bank a function's code will be assembled into.

    - Explicit `#[bank(n)]` → bank n.
    - `#[bank(auto)]` (bank_attr present, bank_number=None) → `auto_bank_num`,
      the bank `_organize_functions_by_bank` parks auto-bank functions in.
      Treating these as bank 0 silently mis-categorizes cross-bank calls as
      same-bank.
    - No bank attribute → bank 0 (the default placement).
    """
    if func.bank_attr is None:
        return 0
    if func.bank_attr.bank_number is not None:
        return func.bank_attr.bank_number
    return auto_bank_num


def _manages_dbr(func: MIRFunction) -> bool:
    """True for a `#[mode(databank=inline|caller)]` function."""
    from r65.compiler.hir.attributes import DataBankMode
    return (func.mode_attr is not None
            and func.mode_attr.databank != DataBankMode.NONE)


class FarToNearOptimizer:
    """
    Optimizer that converts far function calls to near when safe.

    See the module docstring for the conditions.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the optimizer.

        Args:
            verbose: If True, print optimization details
        """
        self.verbose = verbose
        self.func_map: Dict[str, MIRFunction] = {}
        self.bank_info: Dict[str, FunctionBankInfo] = {}
        # Mangled names of every method reachable through a trait dispatch
        # table. Filled in at the start of `optimize()`.
        self._dispatch_targets: Set[str] = set()
        # The bank auto-bank functions will land in (`compute_auto_bank_num`).
        # Filled in at the start of `optimize()`.
        self._auto_bank_num: int = 0

    def optimize(self, mir_program: MIRProgram) -> int:
        """
        Optimize far calls to near calls where safe.

        Args:
            mir_program: The MIR program to optimize

        Returns:
            Number of functions converted from far to near
        """
        # Build function map
        self.func_map = {f.name: f for f in mir_program.functions}

        # The bank #[bank(auto)] functions will be placed in: treating it
        # as bank 0 would co-locate them with no-bank callers and turn a
        # cross-bank JSL into a JSR.
        self._auto_bank_num = compute_auto_bank_num(mir_program)
        graph = build_call_graph(mir_program)
        # A far trait's dispatch wrapper enters each implementor with JML,
        # so the implementor must still RTL even when every direct caller
        # sits in its own bank.
        self._dispatch_targets = graph.dispatch_targets

        # Analyze all functions
        self._analyze_functions(mir_program, graph)

        # Find functions that can be converted
        convertible = self._find_convertible_functions()

        if not convertible:
            return 0

        # Perform the conversion
        for func_name in convertible:
            self._convert_function(func_name, mir_program)

        return len(convertible)

    def _get_function_bank(self, func: MIRFunction) -> int:
        """Effective bank number for a function; see `effective_bank`."""
        return effective_bank(func, self._auto_bank_num)

    def _analyze_functions(self, mir_program: MIRProgram, graph: CallGraph):
        """
        Analyze all functions to build bank and call relationship info.

        Args:
            mir_program: The MIR program to analyze
        """
        for func in mir_program.functions:
            self.bank_info[func.name] = FunctionBankInfo(
                name=func.name,
                bank=self._get_function_bank(func),
                is_far=func.is_far,
                has_address_taken=func.name in graph.address_taken,
                is_interrupt=func.interrupt_attr is not None,
                manages_dbr=_manages_dbr(func),
                callers=graph.callers_of(func.name, direct=True),
            )

    def _find_convertible_functions(self) -> List[str]:
        """
        Find all far functions that can be safely converted to near.

        See the module docstring for the conditions.

        Returns:
            List of function names that can be converted
        """
        convertible = []

        for func_name, info in self.bank_info.items():
            # Skip if not a far function
            if not info.is_far:
                continue

            # Skip if address is taken (could be called indirectly from any bank)
            if info.has_address_taken:
                if self.verbose:
                    print(f"  Skip {func_name}: address taken")
                continue

            # Skip interrupt handlers (called by hardware from any context)
            if info.is_interrupt:
                if self.verbose:
                    print(f"  Skip {func_name}: interrupt handler")
                continue

            # Skip databank=inline/caller functions: their DBR handling is
            # only emitted for a far function
            if info.manages_dbr:
                continue

            # Skip *dyn dispatch targets (entered via JML from the far
            # dispatch wrapper, which needs the RTL to stay)
            if func_name in self._dispatch_targets:
                if self.verbose:
                    print(f"  Skip {func_name}: trait dispatch target")
                continue

            # Skip if no callers (might be externally referenced)
            if not info.callers:
                if self.verbose:
                    print(f"  Skip {func_name}: no callers (may be external)")
                continue

            # Check if all callers are in the same bank
            func_bank = info.bank
            all_same_bank = True

            for caller_name in info.callers:
                if caller_name not in self.bank_info:
                    # External caller - can't determine bank, be conservative
                    all_same_bank = False
                    if self.verbose:
                        print(f"  Skip {func_name}: external caller {caller_name}")
                    break

                caller_bank = self.bank_info[caller_name].bank
                if caller_bank != func_bank:
                    all_same_bank = False
                    if self.verbose:
                        print(f"  Skip {func_name}: caller {caller_name} in bank {caller_bank}, "
                              f"function in bank {func_bank}")
                    break

            if all_same_bank:
                convertible.append(func_name)
                if self.verbose:
                    print(f"  Convert {func_name}: all {len(info.callers)} callers in bank {func_bank}")

        return convertible

    def _convert_function(self, func_name: str, mir_program: MIRProgram):
        """
        Convert a far function to near.

        Updates:
        1. The function's is_far flag to False
        2. All Call instructions that call this function to is_far = False
        3. Stack parameter offsets (reduced by 1 since return address shrinks from 3 to 2 bytes)

        Args:
            func_name: Name of the function to convert
            mir_program: The MIR program containing the function
        """
        # Update the function declaration
        func = self.func_map[func_name]
        func.is_far = False

        # Adjust stack parameter offsets: far return address is 3 bytes, near is 2 bytes
        # So all stack parameters shift down by 1 byte
        if func.stack_param_offsets:
            for param_idx in func.stack_param_offsets:
                func.stack_param_offsets[param_idx] -= 1

        if self.verbose:
            print(f"  Converted function {func_name} from far to near")

        # Update all call sites
        for caller_func in mir_program.functions:
            for block in caller_func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call):
                        if isinstance(instr.function, str) and instr.function == func_name:
                            if instr.is_far:
                                instr.is_far = False
                                if self.verbose:
                                    print(f"    Updated call in {caller_func.name}")


def RunFarToNearOptimizer(mir_program: MIRProgram, verbose: bool = False) -> int:
    """
    Run the far-to-near call optimization pass.

    Args:
        mir_program: The MIR program to optimize
        verbose: If True, print optimization details

    Returns:
        Number of functions converted from far to near
    """
    optimizer = FarToNearOptimizer(verbose=verbose)
    return optimizer.optimize(mir_program)
