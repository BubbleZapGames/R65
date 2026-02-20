"""
FixedStack ABI mandatory parameter promotion.

Pre-pass that runs before codegen when --abi FixedStack is active. Promotes ALL
remaining stack parameters to hardware registers or scratch registers. Emits a
compile error if any parameter cannot be placed.

This replaces scratch_params.py for FixedStack mode.
"""

from typing import Dict, Set, List, Optional, Tuple
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, Call, Argument, ArgumentMechanism,
    FunctionPointer, Move, VirtualRegister, HardwareRegister, TraitDispatch,
)
from r65.compiler.codegen.register_alloc import ScratchRegisterPool
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.errors import CodegenError


def promote_all_stack_params(mir_program: MIRProgram, scratch_pool: ScratchRegisterPool):
    """
    Promote ALL stack parameters to hw registers or scratch registers.

    Compile error if any parameter cannot be placed. This is mandatory for
    FixedStack ABI where no stack-passed parameters are allowed.

    Also sets max_outgoing_arg_bytes = 0 for all functions (FixedStack has no
    outgoing arg area).

    Args:
        mir_program: MIR program to analyze (mutated in place)
        scratch_pool: Pool of available scratch registers
    """
    # Step 1: Find functions whose address is taken
    address_taken = _find_address_taken_functions(mir_program)

    # Step 2: For each function, promote all stack params
    # Build promotions: function_name -> {param_idx: ('hw', reg_name) | ('scratch', addr)}
    all_promotions: Dict[str, Dict[int, Tuple[str, object]]] = {}

    for func in mir_program.functions:
        if not func.stack_param_offsets:
            continue

        # Address-taken functions in FixedStack mode: error if they have unbound stack params
        if func.name in address_taken:
            raise CodegenError(
                f"FixedStack ABI: function '{func.name}' has its address taken but has "
                f"stack parameters. Add explicit register bindings (@ A, @ X, etc.) to all parameters.",
                source_loc=func.source_loc,
            )

        promotions = _promote_function_params(func, scratch_pool)
        if promotions:
            all_promotions[func.name] = promotions

    if not all_promotions:
        # No stack params in the whole program — nothing to do
        for func in mir_program.functions:
            func.max_outgoing_arg_bytes = 0
        return

    # Step 3: Apply promotions to callee functions
    total_hw = 0
    total_scratch = 0
    for func in mir_program.functions:
        if func.name not in all_promotions:
            continue
        func_promos = all_promotions[func.name]
        for param_idx, (kind, loc) in func_promos.items():
            if kind == 'hw':
                func.hw_param_regs[param_idx] = loc
                total_hw += 1
            elif kind == 'scratch':
                func.scratch_param_addrs[param_idx] = loc
                total_scratch += 1

        # Add Move instructions from hw reg to vreg for hw-promoted params
        _inject_hw_param_moves(func)

        # Remove promoted params from stack_param_offsets
        for param_idx in func_promos:
            if param_idx in func.stack_param_offsets:
                del func.stack_param_offsets[param_idx]

    # Step 4: Update call sites
    for func in mir_program.functions:
        _update_call_sites(func, all_promotions)

    # Step 5: Zero out outgoing arg bytes for all functions
    for func in mir_program.functions:
        func.max_outgoing_arg_bytes = 0

    total = total_hw + total_scratch
    if total > 0:
        parts = []
        if total_hw > 0:
            parts.append(f"{total_hw} to hw regs")
        if total_scratch > 0:
            parts.append(f"{total_scratch} to scratch")
        print(f"FixedStack parameter promotion: {total} parameter(s) promoted ({', '.join(parts)})")


def _find_address_taken_functions(mir_program: MIRProgram) -> Set[str]:
    """Find all functions whose address is taken (used as function pointer)."""
    address_taken: Set[str] = set()

    for func in mir_program.functions:
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Move) and isinstance(instr.source, FunctionPointer):
                    address_taken.add(instr.source.function_name)
                if isinstance(instr, (Call, TraitDispatch)):
                    for arg in instr.args:
                        if isinstance(arg.value, FunctionPointer):
                            address_taken.add(arg.value.function_name)

    if hasattr(mir_program, 'trait_dispatch_info') and mir_program.trait_dispatch_info:
        for trait_info in mir_program.trait_dispatch_info.values():
            for impl in trait_info.get('implementors', []):
                for mangled_name in impl.get('mangled', []):
                    address_taken.add(mangled_name)

    return address_taken


def _promote_function_params(
    func: MIRFunction, scratch_pool: ScratchRegisterPool
) -> Dict[int, Tuple[str, object]]:
    """
    Promote all stack parameters of a function to hw regs or scratch.

    Returns:
        Dict mapping param_idx -> ('hw', reg_name) or ('scratch', scratch_addr)

    Raises:
        CodegenError: If any parameter cannot be placed
    """
    if not func.stack_param_offsets:
        return {}

    from r65.compiler.mir.liveness import InstructionLivenessAnalyzer
    from r65.compiler.hir.types import PointerTypeInfo

    liveness = InstructionLivenessAnalyzer(func)

    # Determine which hw regs are already taken by explicit bindings
    taken_hw_regs: Set[str] = set()
    for param in func.parameters:
        from r65.compiler.hir.nodes import RegisterBinding
        if isinstance(param.binding, RegisterBinding):
            taken_hw_regs.add(param.binding.register_name)

    # Also account for trait method self in Y
    if func.is_trait_method:
        taken_hw_regs.add('Y')

    # Classify each stack param
    params_to_place: List[Tuple[int, VirtualRegister, int, bool]] = []  # (idx, vreg, size, cross_call)
    for param_idx in sorted(func.stack_param_offsets.keys()):
        vreg = func.param_to_vreg.get(param_idx)
        if vreg is None:
            continue
        param_size = get_type_size(func.parameters[param_idx].param_type)
        cross_call = liveness.is_live_across_any_call(vreg)

        # Far pointers (3 bytes) can only go to scratch, not hw regs
        is_far_ptr = isinstance(func.parameters[param_idx].param_type, PointerTypeInfo) and \
                     func.parameters[param_idx].param_type.is_far

        if is_far_ptr:
            # Force to scratch only
            params_to_place.append((param_idx, vreg, param_size, cross_call))
        else:
            params_to_place.append((param_idx, vreg, param_size, cross_call))

    # Available hw regs for promotion (order: A, B for u8; X, Y for u16)
    available_hw_u8 = [r for r in ['A', 'B'] if r not in taken_hw_regs]
    available_hw_u16 = [r for r in ['X', 'Y'] if r not in taken_hw_regs]

    # Available scratch registers
    scratch_available = [(s.address, s.size, s.name) for s in scratch_pool.scratches]
    used_scratches: Set[int] = set()

    result: Dict[int, Tuple[str, object]] = {}

    # First pass: assign cross-call params to hw regs (they have existing region-spill support)
    # Second pass: assign local params to hw regs, then scratch
    for is_cross_call_pass in [True, False]:
        for param_idx, vreg, param_size, cross_call in params_to_place:
            if param_idx in result:
                continue
            if cross_call != is_cross_call_pass:
                continue

            # Check if this is a far pointer (3 bytes) — scratch only
            is_far_ptr = param_idx in func.far_ptr_param_indices

            placed = False

            # Try hw regs first (not for far pointers)
            if not is_far_ptr:
                if param_size <= 1:
                    # u8/i8 → try A, then B
                    for reg in available_hw_u8:
                        result[param_idx] = ('hw', reg)
                        available_hw_u8.remove(reg)
                        placed = True
                        break
                elif param_size == 2:
                    # u16/i16 → try X, then Y
                    for reg in available_hw_u16:
                        result[param_idx] = ('hw', reg)
                        available_hw_u16.remove(reg)
                        placed = True
                        break

            # Try scratch registers if hw failed
            if not placed:
                for addr, size, name in scratch_available:
                    if addr in used_scratches:
                        continue
                    if size >= param_size:
                        result[param_idx] = ('scratch', addr)
                        used_scratches.add(addr)
                        placed = True
                        break

            if not placed:
                param_name = func.parameters[param_idx].name
                raise CodegenError(
                    f"FixedStack ABI: cannot place parameter '{param_name}' (index {param_idx}) "
                    f"of function '{func.name}'. All hardware registers and scratch registers "
                    f"are exhausted. Add explicit register bindings or reduce parameter count.",
                    source_loc=func.source_loc,
                )

    return result


def _inject_hw_param_moves(func: MIRFunction):
    """
    Inject Move(hw_reg -> vreg) instructions at the start of the entry block
    for hw-promoted parameters.

    These are analogous to the Move instructions that the MIR builder emits
    for explicitly register-bound parameters (e.g., `param @ A: u8`).
    """
    if not func.hw_param_regs:
        return

    entry_block = func.blocks[func.entry_block_id]
    moves_to_inject = []

    for param_idx, hw_reg_name in sorted(func.hw_param_regs.items()):
        vreg = func.param_to_vreg.get(param_idx)
        if vreg is None:
            continue
        param_type = func.parameters[param_idx].param_type
        move = Move(
            dest=vreg,
            source=HardwareRegister(hw_reg_name),
            type_info=param_type,
        )
        moves_to_inject.append(move)

    # Insert moves at the beginning of the entry block
    # (after any existing register-param moves, but before other instructions)
    # To keep it simple, just prepend them — register-param moves are already
    # at the start, and order among hw-promoted moves doesn't matter much
    # as long as they come before the first use.
    entry_block.instructions = moves_to_inject + entry_block.instructions


def _update_call_sites(func: MIRFunction, all_promotions: Dict[str, Dict[int, Tuple[str, object]]]):
    """
    Update Call instructions to use REGISTER or SCRATCH_PARAM mechanism
    for promoted parameters.
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

            callee_promos = all_promotions.get(instr.function)
            if not callee_promos:
                continue

            for arg_idx, arg in enumerate(instr.args):
                if arg.mechanism != ArgumentMechanism.STACK:
                    continue
                if arg_idx not in callee_promos:
                    continue

                kind, loc = callee_promos[arg_idx]
                if kind == 'hw':
                    arg.mechanism = ArgumentMechanism.REGISTER
                    arg.location = HardwareRegister(loc)
                elif kind == 'scratch':
                    arg.mechanism = ArgumentMechanism.SCRATCH_PARAM
                    arg.scratch_addr = loc
