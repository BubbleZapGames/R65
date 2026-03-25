# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Shared utilities for parameter promotion analyses.

Contains helper functions used by both scratch_params.py (Default ABI)
and fixedstack_params.py (FixedStack ABI).
"""

from typing import Dict, Set, List, Optional, Tuple, Sequence
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
    # but they CAN have scratch-promoted parameters as long as all impls of the
    # same trait method share the same scratch addresses. They are NOT marked
    # address-taken; instead, find_trait_impl_groups() coordinates their scratch
    # layout separately.

    return address_taken


def find_trait_impl_groups(mir_program: MIRProgram) -> Dict[Tuple[str, int], List[str]]:
    """
    Group trait method implementations by (trait_name, method_index).

    All impls of the same trait method share the same parameter signature,
    so they can be assigned shared scratch addresses for coordination.

    Returns:
        Dict mapping (trait_name, method_idx) -> list of impl mangled function names
    """
    groups: Dict[Tuple[str, int], List[str]] = {}

    if not hasattr(mir_program, 'trait_dispatch_info') or not mir_program.trait_dispatch_info:
        return groups

    for trait_name, trait_info in mir_program.trait_dispatch_info.items():
        methods = trait_info.get('methods', [])
        for method_idx in range(len(methods)):
            key = (trait_name, method_idx)
            impl_names: List[str] = []
            for impl in trait_info.get('implementors', []):
                mangled_list = impl.get('mangled', [])
                if method_idx < len(mangled_list):
                    impl_names.append(mangled_list[method_idx])
            if impl_names:
                groups[key] = impl_names

    return groups


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
