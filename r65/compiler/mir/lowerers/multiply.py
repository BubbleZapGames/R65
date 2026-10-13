# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Shared multiplication helpers for MIR lowering.

Provides shift-and-add multiplication for struct array indexing optimization.
Used by both expression.py and assignment.py lowerers.

Struct array indexing scales a variable index by the element size:
    - 1-16 bytes: shift-and-add, inline.
    - over 16 bytes: a call to stdlib `mul16(size, index)`, a 16x16 -> 16
      multiply, so `include!("lib/math.r65")` is required. Any u16 index
      whose byte offset fits in 16 bits is exact.
    A constant index is scaled at compile time.
"""

from typing import TYPE_CHECKING, Callable

from r65.compiler.mir.nodes import (
    VirtualRegister, Immediate, BinaryOp, Call, Argument,
    ArgumentMechanism, HardwareRegister,
)
from r65.compiler.errors import MIRLoweringError

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
            f"Array indexing on structs larger than 16 bytes (this struct is {struct_size} bytes) "
            f"requires runtime multiplication.\n"
            f"\n"
            f"Add this to your source file:\n"
            f"    include!(\"lib/math.r65\")",
            source_loc=ctx.current_source_loc
        )


def emit_shift_and_add_multiply(
    operand,
    multiplier: int,
    type_info,
    ctx: 'LoweringContext',
    emit: Callable,
):
    """
    Emit shift-and-add instructions to multiply operand by a constant.

    Uses optimal decomposition for multipliers 1-16:
    - Powers of 2: single shift
    - 2^n + 1 forms (3, 5, 9): shift + add
    - 2^n - 1 forms (7, 15): shift + subtract
    - Other forms: combination of shifts and adds/subs

    Args:
        operand: The value to multiply (VirtualRegister or HardwareRegister)
        multiplier: Constant multiplier (1-16)
        type_info: Type information for the result
        ctx: Lowering context for allocating virtual registers
        emit: Function to emit MIR instructions

    Returns:
        VirtualRegister containing the result
    """
    # Power of 2: single shift
    if multiplier & (multiplier - 1) == 0:
        shift_amount = (multiplier - 1).bit_length() if multiplier > 1 else 0
        if shift_amount == 0:
            return operand
        result = ctx.alloc_vreg(type_info, "scaled_index")
        emit(BinaryOp(
            dest=result,
            left=operand,
            right=Immediate(shift_amount),
            op='<<',
            type_info=type_info
        ))
        return result

    # Decomposition table for non-power-of-2 multipliers
    # Format: (shifts_for_terms, is_subtract)
    # For x*N: compute (x << a) + (x << b) + ... or (x << a) - x
    decompositions = {
        3: ([1, 0], False),           # (x<<1) + x
        5: ([2, 0], False),           # (x<<2) + x
        6: ([2, 1], False),           # (x<<2) + (x<<1)
        7: ([3], True),               # (x<<3) - x
        9: ([3, 0], False),           # (x<<3) + x
        10: ([3, 1], False),          # (x<<3) + (x<<1)
        11: ([3, 1, 0], False),       # (x<<3) + (x<<1) + x
        12: ([3, 2], False),          # (x<<3) + (x<<2)
        13: ([4], True, [1, 0]),      # (x<<4) - (x<<1) - x
        14: ([4], True, [1]),         # (x<<4) - (x<<1)
        15: ([4], True),              # (x<<4) - x
    }

    if multiplier not in decompositions:
        # Fallback (shouldn't happen for 1-16)
        result = ctx.alloc_vreg(type_info, "scaled_index")
        emit(Call(
            function='mul',
            args=[
                Argument(value=operand, mechanism=ArgumentMechanism.REGISTER,
                         location=HardwareRegister('A'), param_type=type_info),
                Argument(value=Immediate(multiplier), mechanism=ArgumentMechanism.REGISTER,
                         location=HardwareRegister('X'), param_type=type_info),
            ],
            returns=[result],
            builtin_name='mul'
        ))
        return result

    decomp = decompositions[multiplier]

    # Handle special cases for subtraction forms
    if multiplier == 7 or multiplier == 15:
        # (x << n) - x
        shifts, is_sub = decomp
        shift_amount = shifts[0]

        # Compute x << n
        shifted = ctx.alloc_vreg(type_info, "shifted")
        emit(BinaryOp(
            dest=shifted,
            left=operand,
            right=Immediate(shift_amount),
            op='<<',
            type_info=type_info
        ))

        # Subtract x
        result = ctx.alloc_vreg(type_info, "scaled_index")
        emit(BinaryOp(
            dest=result,
            left=shifted,
            right=operand,
            op='-',
            type_info=type_info
        ))
        return result

    elif multiplier == 13:
        # (x<<4) - (x<<1) - x
        # First: x << 4
        shifted4 = ctx.alloc_vreg(type_info, "shifted4")
        emit(BinaryOp(
            dest=shifted4,
            left=operand,
            right=Immediate(4),
            op='<<',
            type_info=type_info
        ))
        # Second: x << 1
        shifted1 = ctx.alloc_vreg(type_info, "shifted1")
        emit(BinaryOp(
            dest=shifted1,
            left=operand,
            right=Immediate(1),
            op='<<',
            type_info=type_info
        ))
        # (x<<4) - (x<<1)
        temp = ctx.alloc_vreg(type_info, "temp")
        emit(BinaryOp(
            dest=temp,
            left=shifted4,
            right=shifted1,
            op='-',
            type_info=type_info
        ))
        # - x
        result = ctx.alloc_vreg(type_info, "scaled_index")
        emit(BinaryOp(
            dest=result,
            left=temp,
            right=operand,
            op='-',
            type_info=type_info
        ))
        return result

    elif multiplier == 14:
        # (x<<4) - (x<<1)
        shifted4 = ctx.alloc_vreg(type_info, "shifted4")
        emit(BinaryOp(
            dest=shifted4,
            left=operand,
            right=Immediate(4),
            op='<<',
            type_info=type_info
        ))
        shifted1 = ctx.alloc_vreg(type_info, "shifted1")
        emit(BinaryOp(
            dest=shifted1,
            left=operand,
            right=Immediate(1),
            op='<<',
            type_info=type_info
        ))
        result = ctx.alloc_vreg(type_info, "scaled_index")
        emit(BinaryOp(
            dest=result,
            left=shifted4,
            right=shifted1,
            op='-',
            type_info=type_info
        ))
        return result

    else:
        # Addition forms: (x << a) + (x << b) + ...
        shifts, is_sub = decomp

        # Compute first term
        if shifts[0] == 0:
            accumulated = operand
        else:
            accumulated = ctx.alloc_vreg(type_info, "shifted")
            emit(BinaryOp(
                dest=accumulated,
                left=operand,
                right=Immediate(shifts[0]),
                op='<<',
                type_info=type_info
            ))

        # Add remaining terms
        for i, shift in enumerate(shifts[1:], 1):
            if shift == 0:
                term = operand
            else:
                term = ctx.alloc_vreg(type_info, f"shifted{i}")
                emit(BinaryOp(
                    dest=term,
                    left=operand,
                    right=Immediate(shift),
                    op='<<',
                    type_info=type_info
                ))

            new_accumulated = ctx.alloc_vreg(type_info, "accumulated")
            emit(BinaryOp(
                dest=new_accumulated,
                left=accumulated,
                right=term,
                op='+',
                type_info=type_info
            ))
            accumulated = new_accumulated

        return accumulated


def compute_scaled_index(
    index_operand,
    struct_size: int,
    type_info,
    ctx: 'LoweringContext',
    emit: Callable,
    call_lowerer: 'CallLowerer',
):
    """
    Compute scaled index for array element access: index * struct_size.

    Does NOT add a field offset: the caller folds that into the
    MemoryLocation address constant, saving a CLC+ADC per field access.

    Args:
        index_operand: The index value (VirtualRegister, HardwareRegister, or Immediate)
        struct_size: Size of each struct element in bytes
        type_info: Type information for intermediate values
        ctx: Lowering context for allocating virtual registers
        emit: Function to emit MIR instructions
        call_lowerer: Emits the mul16 call for sizes over 16 bytes

    Returns:
        Operand containing index * struct_size
    """
    # A scaled index is a byte offset (index * struct_size). It can exceed 255
    # (e.g. 16 elements * 18 bytes = 288), so it is intrinsically u16 and must
    # NOT be sized by the caller's type_info (which is the struct/element type).
    # Sizing it as the struct type produces an 8-bit store / 16-bit load
    # mismatch in codegen, corrupting the index. Widen a u8/i8 index to u16
    # first so the result is consistently u16 for ALL struct sizes — including
    # size 1, whose callers still Move it into X with a u16 type_info (an
    # un-widened u8 there would read 2 bytes from a 1-byte slot).
    from r65.compiler.hir.types import BasicTypeInfo
    from r65.compiler.mir.nodes import TypeConvert
    from r65.compiler.mir.nodes import VirtualRegister
    u16_type = BasicTypeInfo('u16')

    from r65.compiler.hir.types import strip_newtype
    idx_payload = strip_newtype(index_operand.type_info) if isinstance(
        index_operand, VirtualRegister) else None
    if (isinstance(index_operand, VirtualRegister) and
            isinstance(idx_payload, BasicTypeInfo) and
            idx_payload.name in ('u8', 'i8')):
        extended = ctx.alloc_vreg(u16_type, "idx_ext")
        emit(TypeConvert(
            dest=extended,
            source=index_operand,
            source_type=index_operand.type_info,
            target_type=u16_type,
        ))
        index_operand = extended

    if struct_size == 1:
        return index_operand

    if struct_size <= 16:
        return emit_shift_and_add_multiply(
            index_operand, struct_size, u16_type, ctx, emit
        )
    else:
        _check_runtime_mul_available(ctx, struct_size)

        # An ordinary call to the declared mul16(multA, multB): the
        # arguments are passed per its parameters (and the ABI), and the
        # Call carries its entry/exit modes.
        scaled_index = ctx.alloc_vreg(u16_type, "scaled_index")
        call_lowerer.emit_runtime_call(
            ctx.function_decls['mul16'],
            [Immediate(struct_size), index_operand],
            [scaled_index],
        )
        return scaled_index

