"""
Far pointer strategy analysis.

For functions with far pointer stack parameters, choose between two strategies:
- D_EQUALS_S: PHD/TSC/TCD sets D=S, enables [dp],Y indirect long addressing.
  Disables zeropage/scratch registers (DP no longer points to zeropage).
- SET_DBR: PHB/LDA/PHA/PLB sets DBR to pointer's bank, enables bare (d,S),Y
  addressing. Keeps DP intact, preserving scratch registers and zeropage access.

Cost model:
  D=S cost     = 13 + 1*N_zp + 13*N_calls
  SET_DBR cost = 19 + 1*N_rom + 1*N_hw + 1*N_ram + 14*N_calls

SET_DBR wins when: N_zp > N_rom + N_hw + N_ram + N_calls + 6
"""

from r65.compiler.mir.nodes import (
    FarPtrStrategy, MIRFunction, BasicBlock,
    Load, Store, LoadIndirect, StoreIndirect,
    Call, TraitDispatch, MemoryLocation, VirtualRegister,
)


def analyze_far_ptr_strategy(mir_program):
    """Analyze all functions and set far_ptr_strategy on those with far pointer stack params."""
    for func in mir_program.functions:
        if not func.has_far_ptr_stack_params:
            continue
        func.far_ptr_strategy = _choose_strategy(func)


def _choose_strategy(func: MIRFunction) -> FarPtrStrategy:
    """Choose the optimal far pointer access strategy for a function."""
    # D=S is incompatible with scratch params: D=S moves DP to the stack,
    # so DP addresses no longer reach zeropage scratch registers.
    # Force SET_DBR when scratch params are present.
    if func.scratch_param_addrs and _is_set_dbr_safe(func):
        return FarPtrStrategy.SET_DBR

    # Safety checks — force D=S if any disqualifier
    if not _is_set_dbr_safe(func):
        return FarPtrStrategy.D_EQUALS_S

    # Count access types
    n_zp, n_rom, n_hw, n_ram, n_calls = _count_accesses(func)

    # Cost comparison:
    # D=S cost     = 13 + n_zp + 13*n_calls
    # SET_DBR cost = 19 + n_rom + n_hw + n_ram + 14*n_calls
    # RAM needs LONG under SET_DBR because DBR may not be $7E
    # SET_DBR wins when: n_zp > n_rom + n_hw + n_ram + n_calls + 6
    d_equals_s_cost = 13 + n_zp + 13 * n_calls
    set_dbr_cost = 19 + n_rom + n_hw + n_ram + 14 * n_calls

    if set_dbr_cost < d_equals_s_cost:
        return FarPtrStrategy.SET_DBR
    return FarPtrStrategy.D_EQUALS_S


def _is_set_dbr_safe(func: MIRFunction) -> bool:
    """Check if SET_DBR strategy is safe for this function."""
    # 1. Multiple far pointer stack params — can't set DBR to two banks
    if len(func.far_ptr_param_indices) > 1:
        return False

    # 2. Trait methods with self_far_uses_d_equals_s — complex interaction
    if func.self_far_uses_d_equals_s:
        return False

    # 3. Check for near pointer derefs and non-param far pointer derefs
    far_param_vregs = set()
    for idx in func.far_ptr_param_indices:
        if idx in func.param_to_vreg:
            far_param_vregs.add(func.param_to_vreg[idx])

    for block in func.blocks.values():
        for instr in block.instructions:
            if isinstance(instr, (LoadIndirect, StoreIndirect)):
                # Near pointer dereference — changing DBR changes bank semantics
                if not instr.is_far:
                    ptr = instr.pointer if isinstance(instr, LoadIndirect) else instr.pointer
                    if isinstance(ptr, VirtualRegister):
                        return False

                # Far pointer dereference through non-param pointer
                if instr.is_far:
                    ptr = instr.pointer if isinstance(instr, LoadIndirect) else instr.pointer
                    if isinstance(ptr, VirtualRegister) and ptr not in far_param_vregs:
                        return False

    return True


def _count_accesses(func: MIRFunction):
    """Count memory access types for cost comparison.

    Returns (n_zp, n_rom, n_hw, n_ram, n_calls).
    """
    n_zp = 0
    n_rom = 0
    n_hw = 0
    n_ram = 0
    n_calls = 0

    for block in func.blocks.values():
        for instr in block.instructions:
            if isinstance(instr, (Load, Store)):
                loc = None
                if isinstance(instr, Load) and isinstance(instr.source, MemoryLocation):
                    loc = instr.source
                elif isinstance(instr, Store) and isinstance(instr.dest, MemoryLocation):
                    loc = instr.dest

                if loc is not None:
                    st = loc.storage_type
                    if st == 'zeropage':
                        n_zp += 1
                    elif st == 'rom':
                        n_rom += 1
                    elif st == 'hw':
                        n_hw += 1
                    elif st == 'ram':
                        n_ram += 1

            elif isinstance(instr, (Call, TraitDispatch)):
                n_calls += 1

    return n_zp, n_rom, n_hw, n_ram, n_calls
