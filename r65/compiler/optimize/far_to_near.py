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
"""

from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass, field

from r65.compiler.mir.nodes import MIRProgram, MIRFunction, Call, TraitDispatch, BasicBlock


@dataclass
class FunctionBankInfo:
    """Information about a function's bank placement and call relationships."""
    name: str
    bank: int
    is_far: bool
    has_address_taken: bool = False
    is_interrupt: bool = False
    callers: Set[str] = field(default_factory=set)
    callees: Set[str] = field(default_factory=set)


class FarToNearOptimizer:
    """
    Optimizer that converts far function calls to near when safe.

    A far function can be converted to near if:
    1. All direct callers are in the same bank as the function
    2. The function's address is never taken (no indirect calls possible)
    3. The function is not an interrupt handler
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
        # Computed bank that auto-bank functions will land in. Filled in
        # at the start of `optimize()`. Mirrors the placement rule used
        # later by `ProgramCodeGenerator._organize_functions_by_bank`.
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

        # Pre-compute the bank that #[bank(auto)] functions will be
        # placed in. Without this we'd treat auto-bank like bank 0 and
        # incorrectly conclude that an auto-bank function and a no-bank
        # caller are co-located — converting their JSL to JSR. The real
        # placement (handled later by `_organize_functions_by_bank`)
        # parks auto-bank functions at `max(explicit_bank)+1`, floor 4,
        # so the JSR'd target ends up at a completely different address
        # from where the call points.
        self._auto_bank_num = self._compute_auto_bank_num(mir_program)

        # Analyze all functions
        self._analyze_functions(mir_program)

        # Find functions that can be converted
        convertible = self._find_convertible_functions()

        if not convertible:
            return 0

        # Perform the conversion
        for func_name in convertible:
            self._convert_function(func_name, mir_program)

        return len(convertible)

    def _compute_auto_bank_num(self, mir_program: MIRProgram) -> int:
        """Return the bank that auto-bank functions will be placed in.

        Mirrors the rule in
        `ProgramCodeGenerator._organize_functions_by_bank`:
        auto-bank functions live in the bank above the highest explicit
        bank, with a floor of bank 4. If the program has no auto-bank
        functions this value isn't actually used; we still compute a
        consistent number so the caller can compare freely.
        """
        explicit_max = -1
        for func in mir_program.functions:
            if func.bank_attr and func.bank_attr.bank_number is not None:
                if func.bank_attr.bank_number > explicit_max:
                    explicit_max = func.bank_attr.bank_number
        return max(explicit_max + 1, 4)

    def _get_function_bank(self, func: MIRFunction) -> int:
        """
        Get the effective bank number for a function.

        - Explicit `#[bank(n)]` → bank n.
        - `#[bank(auto)]` (bank_attr present, bank_number=None) → the
          bank that `_organize_functions_by_bank` will place auto-bank
          functions in (`max(explicit) + 1`, floor 4). Treating these
          as bank 0 — the previous behavior — silently
          mis-categorized cross-bank calls as same-bank and led the
          optimizer to emit JSR to a callee that the codegen later
          parks in a different bank.
        - No bank attribute → bank 0 (the default placement).

        Args:
            func: The MIR function

        Returns:
            Bank number (0-255)
        """
        if func.bank_attr is None:
            return 0
        if func.bank_attr.bank_number is not None:
            return func.bank_attr.bank_number
        return self._auto_bank_num

    def _analyze_functions(self, mir_program: MIRProgram):
        """
        Analyze all functions to build bank and call relationship info.

        Args:
            mir_program: The MIR program to analyze
        """
        # First pass: collect basic info for each function
        for func in mir_program.functions:
            bank = self._get_function_bank(func)
            is_interrupt = func.interrupt_attr is not None

            self.bank_info[func.name] = FunctionBankInfo(
                name=func.name,
                bank=bank,
                is_far=func.is_far,
                is_interrupt=is_interrupt,
            )

        # Second pass: analyze call relationships and address-taken
        for func in mir_program.functions:
            caller_name = func.name

            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call):
                        if isinstance(instr.function, str):
                            # Direct call
                            callee_name = instr.function

                            # Record caller -> callee relationship
                            if caller_name in self.bank_info:
                                self.bank_info[caller_name].callees.add(callee_name)

                            # Record callee <- caller relationship
                            if callee_name in self.bank_info:
                                self.bank_info[callee_name].callers.add(caller_name)
                        else:
                            # Indirect call through function pointer
                            # We can't know what functions might be called
                            pass

                    # Check for address-taken operations
                    # In MIR, this would be a Load of a function address or
                    # Store of a function address to create a function pointer
                    self._check_address_taken(instr)

    def _check_address_taken(self, instr):
        """
        Check if an instruction takes the address of a function.

        In MIR, function addresses can be taken via:
        - Move with function name as immediate source
        - Load of function address

        Args:
            instr: The MIR instruction to check
        """
        from r65.compiler.mir.nodes import Move, Immediate, Store, StoreIndirect

        # Check Move instructions for function address immediates
        if isinstance(instr, Move):
            if isinstance(instr.source, Immediate):
                # Check if the immediate is a function name (string label)
                if isinstance(instr.source.value, str):
                    func_name = instr.source.value
                    if func_name in self.bank_info:
                        self.bank_info[func_name].has_address_taken = True
                        if self.verbose:
                            print(f"  Address taken: {func_name}")

    def _find_convertible_functions(self) -> List[str]:
        """
        Find all far functions that can be safely converted to near.

        A function is convertible if:
        1. It is currently marked as far
        2. All its callers are in the same bank
        3. Its address is never taken
        4. It is not an interrupt handler

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
