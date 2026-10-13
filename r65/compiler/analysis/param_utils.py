# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Shared utilities for parameter promotion analyses.

Contains helper functions used by both scratch_params.py (Default ABI)
and fixedstack_params.py (FixedStack ABI).
"""

from typing import Dict, Set, List, Optional, Tuple, Sequence
from r65.compiler.mir.nodes import MIRProgram, MIRFunction
from r65.compiler.codegen.type_utils import get_type_size


def param_bytes(func: MIRFunction, param_idx: int, base_addr: int) -> range:
    """The byte addresses occupied by `param_idx` placed at `base_addr`."""
    size = get_type_size(func.parameters[param_idx].param_type)
    return range(base_addr, base_addr + size)


def reserve_scratch_param_bytes(mir_program: MIRProgram,
                                skip: Set[str] = frozenset()) -> Set[int]:
    """Reserve every promoted scratch-param byte from all functions' locals.

    A caller's scratch params persist across its calls while the callee runs,
    so no function may put a local on them. The functions in `skip` keep
    theirs out of the set. Every function shares the returned set: bytes
    added to it later are reserved too.
    """
    reserved: Set[int] = set()
    for func in mir_program.functions:
        if func.name in skip:
            continue
        for param_idx, base_addr in func.scratch_param_addrs.items():
            reserved.update(param_bytes(func, param_idx, base_addr))
    for func in mir_program.functions:
        func.global_scratch_param_bytes = reserved
    return reserved


def clear_far_ptr_flag_if_all_promoted(func: MIRFunction, promoted_param_indices) -> None:
    """Clear the far-ptr-stack-param flag when every far-ptr param was promoted.

    Shared by both ABI promotion passes: once no far-pointer parameter remains
    on the stack, the function no longer needs far-ptr stack-param handling.
    """
    if func.far_ptr_param_indices:
        remaining_far = func.far_ptr_param_indices - set(promoted_param_indices)
        if not remaining_far:
            func.has_far_ptr_stack_params = False
            func.far_ptr_param_indices.clear()


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
