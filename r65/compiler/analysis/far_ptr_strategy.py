# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Far pointer strategy analysis.

For functions with far pointer stack parameters, choose between two strategies:
- D_EQUALS_S: PHD/TSC/TCD sets D=S, enables [dp],Y indirect long addressing.
  Disables zeropage/scratch registers (DP no longer points to zeropage).
- SET_DBR: PHB/LDA/PHA/PLB sets DBR to pointer's bank, enables bare (d,S),Y
  addressing. Keeps DP intact, preserving scratch registers and zeropage access.

Cost model (data pointer params):
  D=S cost     = 13 + 1*N_zp + 13*N_calls
  SET_DBR cost = 19 + 1*N_rom + 1*N_hw + 1*N_ram + 14*N_calls

SET_DBR wins when: N_zp > N_rom + N_hw + N_ram + N_calls + 6
"""

from r65.compiler.mir.nodes import (
    CallBase, InlineAsm, iter_memory_operands,
    FarPtrStrategy, MIRFunction,
    Load, Store, LoadIndirect, StoreIndirect,
    MemoryLocation, VirtualRegister,
)
from r65.compiler.hir.types import PointerTypeInfo


def analyze_far_ptr_strategy(mir_program):
    """Set ``func.far_ptr_strategy`` for every function with far ptr stack params."""
    for func in mir_program.functions:
        if func.has_far_ptr_stack_params:
            func.far_ptr_strategy = _choose_strategy(func)


def forces_d_equals_s(func: MIRFunction) -> bool:
    """Whether `func` will use D=S whatever its body costs.

    D=S moves DP onto the stack, so DP addresses no longer reach zero-page
    scratch registers: scratch-param promotion skips these functions, which
    is what lets `_choose_strategy` pick SET_DBR for every function that has
    scratch params.
    """
    return func.has_far_ptr_stack_params and not is_set_dbr_safe(func)


def _choose_strategy(func: MIRFunction) -> FarPtrStrategy:
    """Choose the optimal far pointer access strategy for a function."""
    if forces_d_equals_s(func):
        return FarPtrStrategy.D_EQUALS_S

    # D=S would hide the scratch params' DP addresses.
    if func.scratch_param_addrs:
        return FarPtrStrategy.SET_DBR

    # Cost comparison:
    # D=S cost     = 13 + n_zp + 13*n_calls
    # SET_DBR cost = 19 + n_rom + n_hw + n_ram + 14*n_calls
    # RAM needs LONG under SET_DBR because DBR may not be $7E
    n_zp, n_rom, n_hw, n_ram, n_calls = _count_accesses(func)
    d_equals_s_cost = 13 + n_zp + 13 * n_calls
    set_dbr_cost = 19 + n_rom + n_hw + n_ram + 14 * n_calls

    if set_dbr_cost < d_equals_s_cost:
        return FarPtrStrategy.SET_DBR
    return FarPtrStrategy.D_EQUALS_S


def is_set_dbr_safe(func: MIRFunction) -> bool:
    """Check if SET_DBR strategy is safe for this function."""
    # Defensive: a function with only far fn ptr params (no data ptr params)
    # should never use SET_DBR. The fn pointer's bank is encoded in the target
    # address, not DBR — setting DBR to a code bank is meaningless for indirect
    # call lowering and breaks RAM/$7E absolute access in the body.
    if not func.far_ptr_param_indices and func.fn_ptr_param_indices:
        return False

    # Multiple far pointer stack params — can't set DBR to two banks
    if len(func.far_ptr_param_indices) > 1:
        return False

    # Trait methods with self_far_uses_d_equals_s — complex interaction
    if func.self_far_uses_d_equals_s:
        return False

    # Check for near pointer derefs and non-param far pointer derefs
    far_param_vregs = {
        func.param_to_vreg[idx]
        for idx in func.far_ptr_param_indices
        if idx in func.param_to_vreg
    }

    for block in func.blocks.values():
        for instr in block.instructions:
            if not isinstance(instr, (LoadIndirect, StoreIndirect)):
                continue
            ptr = instr.pointer
            if not isinstance(ptr, VirtualRegister):
                continue
            if not instr.is_far:
                # Near pointer dereference — changing DBR changes bank semantics
                return False
            if ptr not in far_param_vregs:
                # Far pointer deref through a non-param pointer
                return False

    return True


def far_self_body_needs_dbr(func: MIRFunction) -> bool:
    """Whether a far-self trait method's body needs a DBR other than self's.

    Self arrives as DBR:Y, so the body may keep that DBR only if nothing in it
    expects the program's: no call (the callee assumes a normal DBR) and no
    ROM, `#[hw]` or `#[lowram]` operand (absolute addressing goes through DBR,
    and a self in ROM or $7F has no I/O or low-RAM mirror) and no inline asm.
    Such a
    method saves self first: D=S under the Default ABI
    (`self_far_uses_d_equals_s`), a zero-page slot under FixedStack
    (`self_far_uses_scratch`).
    """
    for block in func.blocks.values():
        for instr in block.instructions:
            # Inline asm is not analyzed: it may address I/O or low RAM.
            if isinstance(instr, (CallBase, InlineAsm)):
                return True
            # Any operand may name memory: a load, a store, a folded operand.
            if any(m.storage_type in ('rom', 'hw', 'lowram')
                   for m in iter_memory_operands(instr)):
                return True
    return False


def far_self_needs_save(func: MIRFunction) -> bool:
    """Whether `func` is a far-self trait method that must save self first.

    See `far_self_body_needs_dbr`.
    """
    if not func.is_trait_method or not func.self_y_vreg:
        return False
    self_type = func.self_y_vreg.type_info
    if not isinstance(self_type, PointerTypeInfo) or not self_type.is_far:
        return False
    return far_self_body_needs_dbr(func)


def runs_with_d_equals_s(func: MIRFunction) -> bool:
    """Whether codegen gives `func` D=S, so that its zero-page operands are
    addressed absolute, through DBR (DP no longer reaches page 0).

    Holds for a function with far-pointer stack params on any strategy but
    SET_DBR, and for a far-self trait method that saves self on the stack
    rather than in a FixedStack scratch slot.
    """
    if func.has_far_ptr_stack_params:
        return func.far_ptr_strategy != FarPtrStrategy.SET_DBR
    return not func.self_far_uses_scratch and far_self_needs_save(func)


def _count_accesses(func: MIRFunction):
    """Count memory access types for cost comparison.

    Returns (n_zp, n_rom, n_hw, n_ram, n_calls).
    """
    n_zp = n_rom = n_hw = n_ram = n_calls = 0

    for block in func.blocks.values():
        for instr in block.instructions:
            if isinstance(instr, CallBase):
                n_calls += 1
                continue
            if isinstance(instr, Load):
                loc = instr.source
            elif isinstance(instr, Store):
                loc = instr.dest
            else:
                continue
            if not isinstance(loc, MemoryLocation):
                continue
            st = loc.storage_type
            if st == 'zeropage':
                n_zp += 1
            elif st == 'rom':
                n_rom += 1
            elif st == 'hw':
                n_hw += 1
            elif st == 'ram':
                n_ram += 1

    return n_zp, n_rom, n_hw, n_ram, n_calls
