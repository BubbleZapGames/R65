# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Shared multiplication helpers for MIR lowering.

Array indexing scales a variable index by the element size:
    - a power of 2: one shift.
    - other sizes up to 16 bytes: shift-and-add, inline.
    - other sizes over 16 bytes: a call to stdlib `mul16(size, index)`, a
      16x16 -> 16 multiply, so `include!("lib/math.r65")` is required. Any
      u16 index whose byte offset fits in 16 bits is exact.
    A constant index is scaled at compile time.
"""

from typing import TYPE_CHECKING, Callable

from r65.compiler.mir.nodes import VirtualRegister, Immediate, BinaryOp
from r65.compiler.errors import MIRLoweringError, InternalCompilerError

if TYPE_CHECKING:
    from r65.compiler.mir.context import LoweringContext
    from r65.compiler.mir.lowerers.call import CallLowerer


def _check_runtime_mul_available(ctx: 'LoweringContext', struct_size: int):
    """
    Check if runtime multiplication is available.

    Raises a helpful error if mul16 is needed but math.r65
    hasn't been included.
    """
    # Check if mul16 is defined in function declarations
    if 'mul16' not in ctx.function_decls:
        raise MIRLoweringError(
            f"Indexing an array of {struct_size}-byte elements (over 16 bytes and "
            f"not a power of 2) requires runtime multiplication.\n"
            f"\n"
            f"Add this to your source file:\n"
            f"    include!(\"lib/math.r65\")",
            source_loc=ctx.current_source_loc
        )


# x*N for each non-power-of-2 N in 3..15, as terms of x: the first is
# x << shift, and each later one is added (+1) or subtracted (-1).
_DECOMPOSITIONS = {
    3: (1, (0, +1)),                  # (x<<1) + x
    5: (2, (0, +1)),                  # (x<<2) + x
    6: (2, (1, +1)),                  # (x<<2) + (x<<1)
    7: (3, (0, -1)),                  # (x<<3) - x
    9: (3, (0, +1)),                  # (x<<3) + x
    10: (3, (1, +1)),                 # (x<<3) + (x<<1)
    11: (3, (1, +1), (0, +1)),        # (x<<3) + (x<<1) + x
    12: (3, (2, +1)),                 # (x<<3) + (x<<2)
    13: (4, (1, -1), (0, -1)),        # (x<<4) - (x<<1) - x
    14: (4, (1, -1)),                 # (x<<4) - (x<<1)
    15: (4, (0, -1)),                 # (x<<4) - x
}


def emit_shift_and_add_multiply(
    operand,
    multiplier: int,
    type_info,
    ctx: 'LoweringContext',
    emit: Callable,
):
    """
    Emit shift-and-add instructions to multiply operand by a constant: a
    power of 2 is one shift, anything else (3-15) follows `_DECOMPOSITIONS`.

    Args:
        operand: The value to multiply (VirtualRegister or HardwareRegister)
        multiplier: Constant multiplier (a power of 2, or 3-15)
        type_info: Type information for the result
        ctx: Lowering context for allocating virtual registers
        emit: Function to emit MIR instructions

    Returns:
        The operand holding the result
    """
    def shifted(shift: int, name: str):
        if shift == 0:
            return operand
        dest = ctx.alloc_vreg(type_info, name)
        emit(BinaryOp(dest=dest, left=operand, right=Immediate(shift),
                      op='<<', type_info=type_info))
        return dest

    # Power of 2: single shift
    if multiplier & (multiplier - 1) == 0:
        return shifted((multiplier - 1).bit_length(), "scaled_index")

    if multiplier not in _DECOMPOSITIONS:
        raise InternalCompilerError(
            f"no shift-and-add decomposition for x*{multiplier}")

    first_shift, *terms = _DECOMPOSITIONS[multiplier]
    accumulated = shifted(first_shift, "shifted")
    for i, (shift, sign) in enumerate(terms, 1):
        term = shifted(shift, f"shifted{i}")
        new_accumulated = ctx.alloc_vreg(type_info, "accumulated")
        emit(BinaryOp(dest=new_accumulated, left=accumulated, right=term,
                      op='+' if sign > 0 else '-', type_info=type_info))
        accumulated = new_accumulated
    return accumulated


def compute_scaled_index(
    index_operand,
    element_size: int,
    index_type,
    ctx: 'LoweringContext',
    emit: Callable,
    call_lowerer: 'CallLowerer',
):
    """
    Compute the byte offset of element `index` of an array: index * element_size.

    The one index-scaling path for every array, struct-array, pointer and
    address-of access. Does NOT add a field offset: the caller folds that into
    the MemoryLocation address constant, saving a CLC+ADC per field access.

    The result is always u16: a byte offset can exceed 255 (16 elements * 18
    bytes = 288), so a 1-byte index is zero- (or sign-) extended first. That
    holds even for element_size 1, whose callers still Move it into X or add
    it to a pointer as u16.

    Args:
        index_operand: The index value (VirtualRegister, HardwareRegister, or Immediate)
        element_size: Size of each element in bytes
        index_type: The index expression's type; sizes a HardwareRegister index
        ctx: Lowering context for allocating virtual registers
        emit: Function to emit MIR instructions
        call_lowerer: Emits the mul16 call (over 16 bytes, not a power of 2)

    Returns:
        Operand containing index * element_size
    """
    from r65.compiler.hir.types import BasicTypeInfo
    from r65.compiler.hir.unified_type_utils import get_unified_type_size
    from r65.compiler.mir.nodes import TypeConvert
    u16_type = BasicTypeInfo('u16')

    source_type = (index_operand.type_info if isinstance(index_operand, VirtualRegister)
                   else index_type)
    if (not isinstance(index_operand, Immediate) and
            get_unified_type_size(source_type) < 2):
        extended = ctx.alloc_vreg(u16_type, "idx_ext")
        emit(TypeConvert(
            dest=extended,
            source=index_operand,
            source_type=source_type,
            target_type=u16_type,
        ))
        index_operand = extended

    if element_size == 1:
        return index_operand

    if element_size <= 16 or element_size & (element_size - 1) == 0:
        return emit_shift_and_add_multiply(
            index_operand, element_size, u16_type, ctx, emit
        )

    _check_runtime_mul_available(ctx, element_size)

    # An ordinary call to the declared mul16(multA, multB): the
    # arguments are passed per its parameters (and the ABI), and the
    # Call carries its entry/exit modes.
    scaled_index = ctx.alloc_vreg(u16_type, "scaled_index")
    call_lowerer.emit_runtime_call(
        ctx.function_decls['mul16'],
        [Immediate(element_size), index_operand],
        [scaled_index],
    )
    return scaled_index
