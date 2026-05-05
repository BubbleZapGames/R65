# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
    MIRProgram, MIRFunction, Call, ArgumentMechanism, Move, VirtualRegister,
)
from r65.compiler.codegen.register_alloc import ScratchRegisterPool
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.analysis.param_utils import find_address_taken_functions, find_composite_scratch
from r65.compiler.analysis.far_ptr_strategy import _is_set_dbr_safe


def _param_bytes(func: MIRFunction, param_idx: int, base_addr: int) -> range:
    """The byte addresses occupied by `param_idx` placed at `base_addr`."""
    size = get_type_size(func.parameters[param_idx].param_type)
    return range(base_addr, base_addr + size)


def _is_eligible_for_any_promotion(func: MIRFunction, address_taken: Set[str]) -> bool:
    """Whether `func` could have any of its params promoted to scratch."""
    if func.name in address_taken:
        return False  # Function pointer callers can't honor scratch ABI
    if func.is_recursive:
        return False  # Scratch DP slots are non-reentrant
    # D=S moves DP onto the stack, leaving scratch DP addresses inaccessible
    if func.has_far_ptr_stack_params and not _is_set_dbr_safe(func):
        return False
    return True


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
        if not _is_eligible_for_any_promotion(func, address_taken):
            continue
        func_promotions = _analyze_function(func, scratch_pool)
        if func_promotions:
            promotions[func.name] = func_promotions

    if not promotions:
        return

    # Step 2b: Demote any caller param whose scratch addr would be
    # overwritten by a callee's scratch param at a call where the caller
    # param is used as an argument. Iterate to fixed point — demoting a
    # promotion can free up addresses for other functions, but can't
    # introduce new conflicts.
    func_by_name = {f.name: f for f in mir_program.functions}
    while True:
        demoted_any = False
        for caller in mir_program.functions:
            caller_promos = promotions.get(caller.name)
            if not caller_promos:
                continue
            conflicts = _find_war_conflicts(caller, caller_promos, promotions, func_by_name)
            if conflicts:
                for param_idx in conflicts:
                    del caller_promos[param_idx]
                if not caller_promos:
                    del promotions[caller.name]
                demoted_any = True
        if not demoted_any:
            break

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

    # Step 3b: Collect global set of all scratch param addresses.
    # Any function's locals must avoid these addresses because a caller's
    # scratch params persist across calls while the callee runs.
    global_scratch_param_addrs: Set[int] = set()
    for func in mir_program.functions:
        for param_idx, base_addr in func.scratch_param_addrs.items():
            global_scratch_param_addrs.update(_param_bytes(func, param_idx, base_addr))
    # Store on each function so function_gen.py can mark them as occupied
    for func in mir_program.functions:
        func._global_scratch_param_addrs = global_scratch_param_addrs

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
    available = [(s.address, s.size, s.name) for s in scratch_pool.scratches]
    used_scratches: Set[int] = set()
    result: Dict[int, int] = {}

    for param_idx, _vreg, param_size in eligible:
        # First try a single scratch that fits
        for addr, size, _name in available:
            if addr not in used_scratches and size >= param_size:
                result[param_idx] = addr
                used_scratches.add(addr)
                break
        else:
            # Otherwise try a composite of adjacent free scratches
            composite = find_composite_scratch(available, used_scratches, param_size)
            if composite:
                result[param_idx] = composite[0][0]
                for addr, _, _ in composite:
                    used_scratches.add(addr)

    return result


def _find_war_conflicts(caller: MIRFunction,
                        caller_promos: Dict[int, int],
                        all_promos: Dict[str, Dict[int, int]],
                        func_by_name: Dict[str, MIRFunction]) -> Set[int]:
    """Find caller params whose scratch addr would be overwritten at a call.

    A WAR conflict exists when:
    - caller's param P is at scratch addr X (size N).
    - caller has a Call to G where P (its vreg) is used as a non-stack arg.
    - G has a scratch param Q whose [addr, addr+size) overlaps [X, X+N).

    Call-arg setup writes Q to that overlapping address before P is loaded
    from X, so the value at X is the new Q — not P.

    Returns set of caller param indices to demote.
    """
    conflicts: Set[int] = set()
    caller_param_bytes: Dict[int, Set[int]] = {
        param_idx: set(_param_bytes(caller, param_idx, addr))
        for param_idx, addr in caller_promos.items()
    }

    for block in caller.blocks.values():
        for instr in block.instructions:
            if not isinstance(instr, Call) or not isinstance(instr.function, str):
                continue
            callee_promos = all_promos.get(instr.function)
            callee = func_by_name.get(instr.function)
            if not callee_promos or callee is None:
                continue

            overwritten: Set[int] = set()
            for cp_idx, cp_addr in callee_promos.items():
                overwritten.update(_param_bytes(callee, cp_idx, cp_addr))

            arg_vregs = {
                arg.value for arg in instr.args
                if isinstance(arg.value, VirtualRegister)
            }
            for param_idx, byte_range in caller_param_bytes.items():
                if param_idx in conflicts:
                    continue
                param_vreg = caller.param_to_vreg.get(param_idx)
                if param_vreg in arg_vregs and not byte_range.isdisjoint(overwritten):
                    conflicts.add(param_idx)

    return conflicts


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

    Removes promoted params (they no longer occupy stack space) and renumbers
    the remainder. param_to_vreg is left intact — function_gen.py reads it
    for scratch pre-allocation regardless of mechanism.
    """
    from r65.compiler.codegen.abi import ABIInfo

    promoted = set(func.scratch_param_addrs)
    remaining = sorted(idx for idx in func.stack_param_offsets if idx not in promoted)
    offset = ABIInfo(is_far=func.is_far).return_addr_size + 1

    func.stack_param_offsets = {}
    for idx in remaining:
        func.stack_param_offsets[idx] = offset
        offset += get_type_size(func.parameters[idx].param_type)


def _update_call_sites(func: MIRFunction, promotions: Dict[str, Dict[int, int]]):
    """Rewrite STACK args to SCRATCH_PARAM at calls whose callee has promotions.

    Args in `instr.args` are in parameter order, so the arg index is the
    parameter index. Indirect calls (`instr.function` is a vreg) and trait
    dispatches have no static callee, so they're left alone.
    """
    for block in func.blocks.values():
        for instr in block.instructions:
            if not isinstance(instr, Call) or not isinstance(instr.function, str):
                continue
            callee_promos = promotions.get(instr.function)
            if not callee_promos:
                continue
            for arg_idx, arg in enumerate(instr.args):
                if arg.mechanism == ArgumentMechanism.STACK and arg_idx in callee_promos:
                    arg.mechanism = ArgumentMechanism.SCRATCH_PARAM
                    arg.scratch_addr = callee_promos[arg_idx]
