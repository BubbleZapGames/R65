# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
FixedStack ABI mandatory parameter promotion.

Pre-pass that runs before codegen when --abi FixedStack is active. Promotes ALL
remaining stack parameters to hardware registers or scratch registers. Emits a
compile error if any parameter cannot be placed.

This replaces scratch_params.py for FixedStack mode.
"""

import sys
from typing import Dict, Set, List, Tuple
from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, Call, ArgumentMechanism,
    Move, VirtualRegister, HardwareRegister, TraitDispatch,
)
from r65.compiler.codegen.register_alloc import ScratchRegisterPool
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.errors import CodegenError
from r65.compiler.analysis.param_utils import (
    find_composite_scratch, find_trait_impl_groups,
    clear_far_ptr_flag_if_all_promoted, reserve_scratch_param_bytes,
)
from r65.compiler.analysis.call_graph import CallGraph, build_call_graph
from r65.compiler.analysis.far_ptr_strategy import far_self_needs_save


def promote_all_stack_params(mir_program: MIRProgram, scratch_pool: ScratchRegisterPool):
    """
    Promote ALL stack parameters to hw registers or scratch registers.

    Compile error if any parameter cannot be placed. This is mandatory for
    FixedStack ABI where no stack-passed parameters are allowed.

    Args:
        mir_program: MIR program to analyze (mutated in place)
        scratch_pool: Pool of available scratch registers
    """
    cg = build_call_graph(mir_program)

    # Step 0: Reject recursive functions — DP scratch params are non-reentrant
    _reject_recursive_functions(mir_program, cg)

    # Step 1: Find functions whose address is taken
    address_taken = cg.address_taken

    # Step 1b: Coordinate trait method scratch params.
    # All impls of the same trait method share the same scratch addresses.
    # Allocate once per trait method, then force-assign to all impls.
    trait_impl_groups = find_trait_impl_groups(mir_program)
    func_by_name = {f.name: f for f in mir_program.functions}
    # Maps (trait_name, method_idx) -> {param_idx: ('scratch', addr)}
    trait_method_promos: Dict[Tuple[str, int], Dict[int, Tuple[str, object]]] = {}
    trait_promoted_funcs: Set[str] = set()

    for (trait_name, method_idx), impl_names in trait_impl_groups.items():
        # Find the first impl function that has stack params
        first_func = None
        for name in impl_names:
            f = func_by_name.get(name)
            if f and f.stack_param_offsets:
                first_func = f
                break

        if first_func is None:
            continue  # No stack params on this trait method

        # Allocate scratch addresses using the first impl's signature.
        # scratch_only=True: dispatch wrapper clobbers A/X, so only DP scratch is safe.
        promos = _promote_function_params(first_func, scratch_pool, scratch_only=True)
        if not promos:
            continue

        trait_method_promos[(trait_name, method_idx)] = promos

        # Force-assign the same addresses to ALL impls
        for name in impl_names:
            f = func_by_name.get(name)
            if f:
                trait_promoted_funcs.add(name)

    # Step 2: For each function, promote all stack params
    # Build promotions: function_name -> {param_idx: ('hw', reg_name) | ('scratch', addr)}
    all_promotions: Dict[str, Dict[int, Tuple[str, object]]] = {}

    # First, register trait method promotions
    for (trait_name, method_idx), promos in trait_method_promos.items():
        impl_names = trait_impl_groups[(trait_name, method_idx)]
        for name in impl_names:
            all_promotions[name] = promos

    # The call graph constrains scratch param promotion: a
    # callee's scratch params must not collide with any caller's scratch params
    # on the same call chain (the callee's prologue would overwrite the
    # caller's param). Functions are processed in program order; when
    # promoting G we exclude scratch bytes already chosen by any transitive
    # caller or callee of G.

    # Per-function record of chosen scratch bytes.
    func_scratch_bytes: Dict[str, Set[int]] = {}

    # Seed with trait-method scratch addresses (dynamic dispatch coordination).
    for promos in trait_method_promos.values():
        for (tn, mi), impl_names in trait_impl_groups.items():
            if trait_method_promos.get((tn, mi)) is not promos:
                continue
            for name in impl_names:
                f = func_by_name.get(name)
                if f is None:
                    continue
                func_scratch_bytes.setdefault(name, set()).update(
                    _scratch_param_bytes(promos, f))

    for func in mir_program.functions:
        if not func.stack_param_offsets:
            continue

        # Skip trait impl functions already promoted by the coordination pre-pass
        if func.name in trait_promoted_funcs:
            continue

        # Address-taken functions in FixedStack mode: error if they have unbound stack params
        if func.name in address_taken:
            raise CodegenError(
                f"FixedStack ABI: function '{func.name}' has its address taken but has "
                f"stack parameters. Add explicit register bindings (@ A, @ X, etc.) to all parameters.",
                source_loc=func.source_loc,
            )

        # Exclude scratch bytes used by functions that may have this function
        # on the active call stack (transitive callers) or be reached by this
        # function (transitive callees) — either direction would let a callee
        # prologue overwrite a still-live caller param.
        reserved = set()
        for related in cg.transitive_callers(func.name) | cg.transitive_callees(func.name):
            reserved |= func_scratch_bytes.get(related, set())

        promotions = _promote_function_params(
            func, scratch_pool,
            reserved_scratch_bytes=reserved,
        )
        if promotions:
            all_promotions[func.name] = promotions
            chosen_bytes: Set[int] = set()
            for param_idx, (kind, loc) in promotions.items():
                if kind != 'scratch':
                    continue
                param_size = get_type_size(func.parameters[param_idx].param_type)
                for offset in range(param_size):
                    chosen_bytes.add(loc + offset)
            if chosen_bytes:
                func_scratch_bytes.setdefault(func.name, set()).update(chosen_bytes)

    if not all_promotions:
        # No stack params in the whole program — nothing to do
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

        # Clear far pointer stack param flag if all far ptrs promoted off stack
        clear_far_ptr_flag_if_all_promoted(func, func_promos.keys())

    # Step 4: Reserve every scratch param byte from all functions' locals.
    # Trait dispatch scratch addresses are NOT globally reserved — they use
    # caller-scoped reservation (only functions that contain TraitDispatch
    # calls mark those addresses as occupied).
    global_scratch_param_addrs = reserve_scratch_param_bytes(
        mir_program, skip=trait_promoted_funcs)

    # Step 4b: Caller-scoped reservation for trait dispatch scratches.
    # Scan each function for TraitDispatch instructions. For each caller that
    # dispatches a trait method, collect the scratch addresses used by that
    # dispatch so function_gen.py reserves them as occupied for that function.
    # Two different trait methods CAN share the same scratch addresses since
    # a given caller only dispatches one at a time (liveness doesn't cross).
    _collect_trait_dispatch_scratch_addrs(mir_program, trait_method_promos,
                                          trait_impl_groups)

    # Step 4c: Allocate scratch slots for far-self trait methods that need to
    # save self across calls/ROM. In FixedStack mode, self is never pushed to
    # the stack — instead it lives in a 3-byte zeropage scratch slot, accessed
    # via [scratch],Y. This keeps DP=0 so scratch params remain reachable.
    _allocate_far_self_scratches(mir_program, scratch_pool, trait_impl_groups,
                                  func_by_name, global_scratch_param_addrs)

    # Step 5: Update call sites
    for func in mir_program.functions:
        _update_call_sites(func, all_promotions, trait_method_promos)

    total = total_hw + total_scratch
    if total > 0:
        parts = []
        if total_hw > 0:
            parts.append(f"{total_hw} to hw regs")
        if total_scratch > 0:
            parts.append(f"{total_scratch} to scratch")
        print(f"FixedStack parameter promotion: {total} parameter(s) promoted ({', '.join(parts)})", file=sys.stderr)


def _reject_recursive_functions(mir_program: MIRProgram, cg: CallGraph):
    """Detect recursive functions and raise an error.

    FixedStack ABI promotes parameters to DP scratch registers which are
    global, not per-invocation. Recursive calls would overwrite the
    caller's scratch values, producing incorrect code.

    Detects both direct recursion (A calls A) and mutual recursion
    (A calls B, B calls A).
    """
    cycles = cg.find_cycles()

    if not cycles:
        return

    # Report the first cycle found
    cycle = cycles[0]
    func_by_name = {f.name: f for f in mir_program.functions}
    func = func_by_name.get(cycle[0])
    source_loc = func.source_loc if func else None

    if len(cycle) == 1:
        raise CodegenError(
            f"FixedStack ABI does not support recursive functions. "
            f"Function '{cycle[0]}' calls itself. "
            f"Use '--abi Default' instead.",
            source_loc=source_loc,
        )
    else:
        chain = " -> ".join(cycle + [cycle[0]])
        raise CodegenError(
            f"FixedStack ABI does not support recursive functions. "
            f"Mutual recursion detected: {chain}. "
            f"Use '--abi Default' instead.",
            source_loc=source_loc,
        )


def _promote_function_params(
    func: MIRFunction, scratch_pool: ScratchRegisterPool,
    scratch_only: bool = False,
    reserved_scratch_bytes: Set[int] = None,
) -> Dict[int, Tuple[str, object]]:
    """
    Promote all stack parameters of a function to hw regs or scratch.

    Args:
        reserved_scratch_bytes: Set of zeropage bytes that must be avoided.
            Each scratch whose [base, base+size) range overlaps this set is
            excluded from the candidate pool. Used to prevent collisions
            between a callee's params and an in-call caller's params (e.g.
            trait method scratch params persist across calls and would be
            overwritten if a callee uses the same scratch addresses).

    Returns:
        Dict mapping param_idx -> ('hw', reg_name) or ('scratch', scratch_addr)

    Raises:
        CodegenError: If any parameter cannot be placed
    """
    if not func.stack_param_offsets:
        return {}

    from r65.compiler.mir.liveness import InstructionLivenessAnalyzer

    if reserved_scratch_bytes is None:
        reserved_scratch_bytes = set()

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
        params_to_place.append((param_idx, vreg, param_size, cross_call))

    # Available hw regs for promotion (order: A, B for u8; X, Y for u16)
    # B is only usable in m8 mode — exclude it from m16 functions (@ A: u16)
    # because in m16, B is the high byte of the 16-bit accumulator and
    # every accumulator operation clobbers it.
    # scratch_only=True: trait dispatch wrapper clobbers A/X, so only scratch is safe.
    if scratch_only:
        available_hw_u8 = []
        available_hw_u16 = []
    else:
        func_is_m16 = any(
            isinstance(p.binding, RegisterBinding) and p.binding.register_name == 'A'
            and get_type_size(p.param_type) == 2
            for p in func.parameters
        )
        available_hw_u8 = [r for r in ['A', 'B'] if r not in taken_hw_regs]
        if func_is_m16:
            available_hw_u8 = [r for r in available_hw_u8 if r != 'B']
        available_hw_u16 = [r for r in ['X', 'Y'] if r not in taken_hw_regs]

    # Available scratch registers — exclude any that overlap reserved bytes.
    scratch_available = []
    for s in scratch_pool.scratches:
        if any(b in reserved_scratch_bytes for b in range(s.address, s.address + s.size)):
            continue
        scratch_available.append((s.address, s.size, s.name))
    used_scratches: Set[int] = set()

    result: Dict[int, Tuple[str, object]] = {}

    # First pass: assign cross-call params to hw regs (they have existing region-spill support)
    #   - Try A, X, Y first (B has no spill support, callees clobber it via m16 ops)
    #   - Then scratch: Step 4's global reservation keeps callee locals off scratch param bytes
    # Second pass: assign local params to hw regs (incl. B), then scratch
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
                    # u8/i8 → try A, then B (B only for non-cross-call params)
                    for reg in available_hw_u8:
                        if is_cross_call_pass and reg == 'B':
                            continue  # B has no region-spill support
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

            # Try composite: find adjacent free scratches that together cover param_size
            if not placed:
                composite = find_composite_scratch(scratch_available, used_scratches, param_size)
                if composite:
                    base_addr = composite[0][0]
                    result[param_idx] = ('scratch', base_addr)
                    for addr, _, _ in composite:
                        used_scratches.add(addr)
                    placed = True

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

    # Prepended: each reads a register no explicit binding uses
    # (`taken_hw_regs`), so order among the entry moves is free.
    entry_block.instructions = moves_to_inject + entry_block.instructions


def _update_call_sites(
    func: MIRFunction,
    all_promotions: Dict[str, Dict[int, Tuple[str, object]]],
    trait_method_promos: Dict[Tuple[str, int], Dict[int, Tuple[str, object]]] = None,
):
    """
    Update Call and TraitDispatch instructions to use REGISTER or SCRATCH_PARAM
    mechanism for promoted parameters.
    """
    for block in func.blocks.values():
        for instr in block.instructions:
            if isinstance(instr, TraitDispatch):
                if trait_method_promos is None:
                    continue
                # Look up shared scratch addresses for this trait method
                key = (instr.trait_name, instr.method_index)
                callee_promos = trait_method_promos.get(key)
                if not callee_promos:
                    continue
                _apply_promos_to_args(instr.args, callee_promos)

            elif isinstance(instr, Call):
                # Only direct calls (not indirect through function pointer)
                if not isinstance(instr.function, str):
                    continue
                callee_promos = all_promotions.get(instr.function)
                if not callee_promos:
                    continue
                _apply_promos_to_args(instr.args, callee_promos)


def _apply_promos_to_args(args, callee_promos: Dict[int, Tuple[str, object]]):
    """Apply promotions to argument list (shared by Call and TraitDispatch)."""
    for arg_idx, arg in enumerate(args):
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


def _allocate_far_self_scratches(
    mir_program: MIRProgram,
    scratch_pool: ScratchRegisterPool,
    trait_impl_groups: Dict[Tuple[str, int], List[str]],
    func_by_name: Dict[str, MIRFunction],
    global_scratch_param_addrs: Set[int],
):
    """Allocate 3-byte zeropage scratch slots for non-leaf far-self trait methods.

    Per-function allocation: each impl picks a scratch slot independently for
    static calls. For dynamically dispatched trait methods (those present in
    trait_dispatch_info), all impls in the group share one address so the
    dispatch wrapper sees a single, consistent layout.

    Reserved bytes are added to the global scratch reservation set so callee
    locals avoid them, and to the caller-side trait-dispatch reservation so
    callers that dispatch a shared-address method also avoid those bytes.

    Raises CodegenError if a far-self trait method needs saving but no 3-byte
    scratch block is available.
    """
    # Reserved addresses = param scratches + already-used composite slots.
    used_byte_addrs = set(global_scratch_param_addrs)

    # Build the candidate list: each scratch as (addr, size, name).
    # Mask out scratches whose any byte is already in used_byte_addrs.
    scratch_available: List[Tuple[int, int, str]] = []
    for s in scratch_pool.scratches:
        overlaps = any(b in used_byte_addrs for b in range(s.address, s.address + s.size))
        if not overlaps:
            scratch_available.append((s.address, s.size, s.name))

    # Build per-function reverse-lookup of trait_dispatch keys: name -> (trait, idx).
    # Only impls present in this map share an address across the group.
    impl_to_key: Dict[str, Tuple[str, int]] = {}
    for key, names in trait_impl_groups.items():
        for name in names:
            impl_to_key[name] = key

    # Track which scratch start addresses we've consumed in this pass.
    consumed_starts: Set[int] = set()
    # For dynamically-dispatched groups, remember the chosen address per key.
    group_chosen_addr: Dict[Tuple[str, int], int] = {}
    # Caller-side reservation: (trait, idx) -> {addr, addr+1, addr+2}.
    trait_self_addrs_per_method: Dict[Tuple[str, int], Set[int]] = {}

    def _pick_3_byte_slot(target_func: MIRFunction) -> int:
        """Find or fail. Returns the base address of a fresh 3-byte block."""
        for addr, size, name in scratch_available:
            if addr in consumed_starts:
                continue
            if size >= 3:
                consumed_starts.add(addr)
                return addr
        composite = find_composite_scratch(scratch_available, consumed_starts, 3)
        if composite:
            base = composite[0][0]
            for caddr, _, _ in composite:
                consumed_starts.add(caddr)
            return base
        raise CodegenError(
            f"FixedStack ABI: trait method '{target_func.name}' has a far *self that must "
            f"be saved across calls/ROM access, but no free 3-byte zeropage scratch "
            f"register is available. Declare an additional `#[zeropage(addr, register)] "
            f"static mut <name>: far *u8;` in your code.",
            source_loc=target_func.source_loc,
        )

    for func in mir_program.functions:
        if not far_self_needs_save(func):
            continue

        # Shared-address branch for dynamic-dispatch groups.
        key = impl_to_key.get(func.name)
        if key is not None:
            chosen_addr = group_chosen_addr.get(key)
            if chosen_addr is None:
                chosen_addr = _pick_3_byte_slot(func)
                group_chosen_addr[key] = chosen_addr
                global_scratch_param_addrs.add(chosen_addr)
                global_scratch_param_addrs.add(chosen_addr + 1)
                global_scratch_param_addrs.add(chosen_addr + 2)
                trait_self_addrs_per_method[key] = {
                    chosen_addr, chosen_addr + 1, chosen_addr + 2,
                }
        else:
            # Static-only impl: allocate independently.
            chosen_addr = _pick_3_byte_slot(func)
            global_scratch_param_addrs.add(chosen_addr)
            global_scratch_param_addrs.add(chosen_addr + 1)
            global_scratch_param_addrs.add(chosen_addr + 2)

        func.self_far_uses_scratch = True
        func.self_scratch_addr = chosen_addr

    # Add the trait-self scratch bytes to the caller-side reservation set.
    if trait_self_addrs_per_method:
        for func in mir_program.functions:
            per_func: Set[int] = set()
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, TraitDispatch):
                        addrs = trait_self_addrs_per_method.get(
                            (instr.trait_name, instr.method_index)
                        )
                        if addrs:
                            per_func |= addrs
            func.trait_dispatch_scratch_bytes |= per_func


def _scratch_param_bytes(promos: Dict[int, Tuple[str, object]],
                         func: MIRFunction) -> Set[int]:
    """The scratch bytes `promos` places `func`'s parameters in."""
    addrs: Set[int] = set()
    for param_idx, (kind, base_addr) in promos.items():
        if kind == 'scratch' and param_idx < len(func.parameters):
            param_size = get_type_size(func.parameters[param_idx].param_type)
            addrs.update(range(base_addr, base_addr + param_size))
    return addrs


def _collect_trait_dispatch_scratch_addrs(
    mir_program: MIRProgram,
    trait_method_promos: Dict[Tuple[str, int], Dict[int, Tuple[str, object]]],
    trait_impl_groups: Dict[Tuple[str, int], List[str]],
):
    """
    For each function that contains TraitDispatch calls, collect the scratch
    addresses used by those dispatches and store them on the function.

    These addresses are reserved in function_gen.py so the function's locals
    don't collide with trait dispatch scratch params.
    """
    if not trait_method_promos:
        return

    # Scratch bytes per trait method, sized by an impl of that method (all
    # its impls share one signature).
    func_by_name = {f.name: f for f in mir_program.functions}
    trait_scratch_bytes: Dict[Tuple[str, int], Set[int]] = {}
    for key, promos in trait_method_promos.items():
        impl = next((func_by_name[n] for n in trait_impl_groups.get(key, ())
                     if n in func_by_name), None)
        if impl is not None:
            addrs = _scratch_param_bytes(promos, impl)
            if addrs:
                trait_scratch_bytes[key] = addrs

    # Scan each function for TraitDispatch instructions
    for func in mir_program.functions:
        func_trait_addrs: Set[int] = set()
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, TraitDispatch):
                    key = (instr.trait_name, instr.method_index)
                    addrs = trait_scratch_bytes.get(key)
                    if addrs:
                        func_trait_addrs |= addrs
        func.trait_dispatch_scratch_bytes = func_trait_addrs
