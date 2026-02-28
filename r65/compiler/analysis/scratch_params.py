"""
Scratch parameter promotion analysis.

Pre-pass that runs before codegen to determine which stack parameters can be
promoted to scratch (zero-page) register passing. Scratch params use faster
DP addressing and require no stack cleanup overhead.

A stack parameter is promoted if:
1. Scratch registers are available (matching size)
2. The parameter's vreg is NOT live across any call in the callee
3. The function's address is not taken (would break function pointer callers)
"""

from typing import Dict, Set, List, Optional, Tuple, Sequence
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, Call, Argument, ArgumentMechanism,
    FunctionPointer, Move, VirtualRegister, TraitDispatch,
)
from r65.compiler.codegen.register_alloc import ScratchRegisterPool
from r65.compiler.codegen.type_utils import get_type_size


def analyze_scratch_params(mir_program: MIRProgram, scratch_pool: ScratchRegisterPool):
    """
    Analyze and promote eligible stack parameters to scratch register passing.

    Mutates mir_program in place:
    - Sets func.scratch_param_addrs for promoted callee params
    - Recomputes func.stack_param_offsets for remaining stack params
    - Updates Call Argument entries at call sites to use SCRATCH_PARAM mechanism

    Args:
        mir_program: MIR program to analyze (mutated in place)
        scratch_pool: Pool of available scratch registers
    """
    if not scratch_pool.scratches:
        return  # No scratch registers available

    # Step 1: Find functions whose address is taken (not promotable)
    address_taken = _find_address_taken_functions(mir_program)

    # Step 2: For each function, determine which stack params can be promoted
    # Build map: function_name -> {param_idx: scratch_addr}
    promotions: Dict[str, Dict[int, int]] = {}

    for func in mir_program.functions:
        if func.name in address_taken:
            continue  # Skip functions whose address is taken

        func_promotions = _analyze_function(func, scratch_pool)
        if func_promotions:
            promotions[func.name] = func_promotions

    if not promotions:
        return

    # Step 3: Apply promotions to callee functions
    total_promoted = 0
    for func in mir_program.functions:
        if func.name in promotions:
            func_promos = promotions[func.name]
            func.scratch_param_addrs = func_promos
            total_promoted += len(func_promos)

            # Recompute stack_param_offsets excluding promoted params
            _recompute_stack_offsets(func)

            # Clear far pointer flag if all far ptrs promoted
            if func.far_ptr_param_indices:
                remaining_far = func.far_ptr_param_indices - set(func_promos.keys())
                if not remaining_far:
                    func.has_far_ptr_stack_params = False
                    func.far_ptr_param_indices.clear()

    # Step 4: Update call sites
    for func in mir_program.functions:
        _update_call_sites(func, promotions)

    if total_promoted > 0:
        print(f"Scratch parameter promotion: {total_promoted} parameter(s) promoted")


def _find_address_taken_functions(mir_program: MIRProgram) -> Set[str]:
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


def _analyze_function(func: MIRFunction, scratch_pool: ScratchRegisterPool) -> Dict[int, int]:
    """
    Determine which stack parameters of a function can use scratch registers.

    Args:
        func: MIR function to analyze
        scratch_pool: Pool of available scratch registers

    Returns:
        Dict mapping param_index -> scratch_address for promotable params
    """
    if not func.stack_param_offsets:
        return {}  # No stack parameters

    # Build liveness analyzer for this function
    from r65.compiler.mir.liveness import InstructionLivenessAnalyzer
    liveness = InstructionLivenessAnalyzer(func)

    # Collect eligible params: stack params whose vregs are not live across any call
    eligible: List[tuple] = []  # (param_idx, vreg, size)

    for param_idx in func.stack_param_offsets:
        vreg = func.param_to_vreg.get(param_idx)
        if vreg is None:
            continue

        # Check if vreg is live across any call
        if liveness.is_live_across_any_call(vreg):
            continue  # Can't use scratch - would be clobbered by callee

        param_size = get_type_size(func.parameters[param_idx].param_type)
        eligible.append((param_idx, vreg, param_size))

    if not eligible:
        return {}

    # Assign eligible params to available scratches (greedy, matching by size)
    # Use a copy of scratch availability so we don't mutate the pool
    available = [(s.address, s.size, s.name) for s in scratch_pool.scratches]
    used_scratches: Set[int] = set()  # Track used scratch addresses

    result: Dict[int, int] = {}

    for param_idx, vreg, param_size in eligible:
        placed = False
        # Find a compatible scratch
        for addr, size, name in available:
            if addr in used_scratches:
                continue
            if size >= param_size:
                result[param_idx] = addr
                used_scratches.add(addr)
                placed = True
                break

        # Try composite: find adjacent free scratches that together cover param_size
        if not placed:
            composite = _find_composite_scratch(available, used_scratches, param_size)
            if composite:
                base_addr = composite[0][0]
                result[param_idx] = base_addr
                for addr, _, _ in composite:
                    used_scratches.add(addr)

    return result


def _find_composite_scratch(
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


def _recompute_stack_offsets(func: MIRFunction):
    """
    Recompute stack_param_offsets after promoting some params to scratch.

    Removes promoted params from stack_param_offsets and recomputes
    offsets for remaining stack params (they shift down because fewer
    bytes are pushed by the caller).
    """
    promoted = func.scratch_param_addrs

    # Remove promoted params from stack_param_offsets only
    # (keep param_to_vreg intact - function_gen.py uses it for scratch pre-allocation)
    for param_idx in promoted:
        if param_idx in func.stack_param_offsets:
            del func.stack_param_offsets[param_idx]

    # Recompute offsets for remaining stack params
    if not func.stack_param_offsets:
        return  # No remaining stack params

    from r65.compiler.codegen.abi import ABIInfo
    abi = ABIInfo(is_far=func.is_far)
    current_offset = abi.return_addr_size + 1

    # Get remaining stack param indices in order
    remaining_indices = sorted(func.stack_param_offsets.keys())

    # Clear offsets and recompute (param_to_vreg stays unchanged)
    func.stack_param_offsets.clear()

    for idx in remaining_indices:
        func.stack_param_offsets[idx] = current_offset
        param_size = get_type_size(func.parameters[idx].param_type)
        current_offset += param_size


def _update_call_sites(func: MIRFunction, promotions: Dict[str, Dict[int, int]]):
    """
    Update Call instructions to use SCRATCH_PARAM mechanism for promoted params.

    For each Call to a function with promoted params, change the matching
    Argument entries from STACK to SCRATCH_PARAM with the assigned scratch address.

    Args:
        func: MIR function containing call sites to update
        promotions: Map of function_name -> {param_idx: scratch_addr}
    """
    for block in func.blocks.values():
        for instr in block.instructions:
            if not isinstance(instr, (Call, TraitDispatch)):
                continue

            # TraitDispatch doesn't have a static callee — skip
            if isinstance(instr, TraitDispatch):
                continue

            # Only direct calls (not indirect through function pointer)
            if not isinstance(instr.function, str):
                continue

            callee_promos = promotions.get(instr.function)
            if not callee_promos:
                continue

            # Build map from param position to promotion info
            # We need to map argument positions to parameter indices.
            # Arguments in instr.args correspond to parameters in order,
            # but we need to identify which args are stack args and their
            # parameter index.
            #
            # The args list contains all arguments in parameter order.
            # Stack args have mechanism==STACK. We need to find which
            # parameter index each arg corresponds to.
            #
            # Args are in parameter order (same index as parameters).
            for arg_idx, arg in enumerate(instr.args):
                if arg.mechanism == ArgumentMechanism.STACK and arg_idx in callee_promos:
                    scratch_addr = callee_promos[arg_idx]
                    arg.mechanism = ArgumentMechanism.SCRATCH_PARAM
                    arg.scratch_addr = scratch_addr
