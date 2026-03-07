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

from typing import Dict, Set, List
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, Call, ArgumentMechanism, TraitDispatch, Move, VirtualRegister,
)
from r65.compiler.codegen.register_alloc import ScratchRegisterPool
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.analysis.param_utils import find_address_taken_functions, find_composite_scratch
from r65.compiler.analysis.far_ptr_strategy import _is_set_dbr_safe


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
    address_taken = find_address_taken_functions(mir_program)

    # Step 2: For each function, determine which stack params can be promoted
    # Build map: function_name -> {param_idx: scratch_addr}
    promotions: Dict[str, Dict[int, int]] = {}

    for func in mir_program.functions:
        if func.name in address_taken:
            continue  # Skip functions whose address is taken

        # Skip functions with far ptr stack params that would need D=S.
        # D=S moves DP to the stack, making scratch regs inaccessible.
        if func.has_far_ptr_stack_params and not _is_set_dbr_safe(func):
            continue

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

        # Also check vregs that receive a Move from this param vreg.
        # If Move(dest=w, source=param_vreg) exists and w is live across
        # calls, w may be coalesced with param_vreg during register alloc,
        # inheriting the scratch address. The callee would then clobber w.
        if _any_move_target_live_across_call(func, vreg, liveness):
            continue

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
            composite = find_composite_scratch(available, used_scratches, param_size)
            if composite:
                base_addr = composite[0][0]
                result[param_idx] = base_addr
                for addr, _, _ in composite:
                    used_scratches.add(addr)

    return result


def _any_move_target_live_across_call(func: MIRFunction, param_vreg, liveness) -> bool:
    """Check if any vreg that receives a Move from param_vreg is live across calls.

    This catches the case where `let d = digits` creates Move(dest=d, source=digits),
    and d is live across calls. If d gets coalesced with digits during register alloc,
    it inherits the scratch address and gets clobbered by the callee.
    """
    for block in func.blocks.values():
        for instr in block.instructions:
            if (isinstance(instr, Move) and
                isinstance(instr.source, VirtualRegister) and
                instr.source == param_vreg and
                isinstance(instr.dest, VirtualRegister) and
                instr.dest != param_vreg):
                if liveness.is_live_across_any_call(instr.dest):
                    return True
    return False


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
