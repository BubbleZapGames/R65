# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Shared utilities for parameter promotion analyses.

Contains helper functions used by both scratch_params.py (Default ABI)
and fixedstack_params.py (FixedStack ABI).
"""

from typing import Set, List, Optional, Tuple, Sequence
from r65.compiler.mir.nodes import (
    MIRProgram, Call, FunctionPointer, Move, TraitDispatch,
)


def find_address_taken_functions(mir_program: MIRProgram) -> Set[str]:
    """
    Find all functions whose address is taken (used as function pointer).

    These cannot have their calling convention changed because unknown
    callers may call them via the original stack-based ABI.

    Returns:
        Set of function names whose address is taken
    """
    address_taken: Set[str] = set()

    for func in mir_program.functions:
        for block in func.blocks.values():
            for instr in block.instructions:
                # Check Move instructions that load function pointers
                if isinstance(instr, Move) and isinstance(instr.source, FunctionPointer):
                    address_taken.add(instr.source.function_name)
                # Check Call args (function pointer passed as argument)
                if isinstance(instr, (Call, TraitDispatch)):
                    for arg in instr.args:
                        if isinstance(arg.value, FunctionPointer):
                            address_taken.add(arg.value.function_name)

    # Trait method implementations are called indirectly through dispatch tables,
    # so they cannot have scratch-promoted parameters
    if hasattr(mir_program, 'trait_dispatch_info') and mir_program.trait_dispatch_info:
        for trait_info in mir_program.trait_dispatch_info.values():
            for impl in trait_info.get('implementors', []):
                for mangled_name in impl.get('mangled', []):
                    address_taken.add(mangled_name)

    return address_taken


def find_composite_scratch(
    scratch_available: Sequence[Tuple[int, int, str]],
    used_scratches: Set[int],
    needed_size: int,
) -> Optional[List[Tuple[int, int, str]]]:
    """Find adjacent free scratches that together provide needed_size bytes."""
    free = sorted(
        [(addr, size, name) for addr, size, name in scratch_available
         if addr not in used_scratches],
        key=lambda x: x[0],
    )

    for i, (start_addr, start_size, start_name) in enumerate(free):
        total = start_size
        group = [(start_addr, start_size, start_name)]
        expected_next = start_addr + start_size

        for j in range(i + 1, len(free)):
            addr_j, size_j, name_j = free[j]
            if addr_j != expected_next:
                break
            group.append((addr_j, size_j, name_j))
            total += size_j
            if total >= needed_size:
                return group
            expected_next = addr_j + size_j

    return None
