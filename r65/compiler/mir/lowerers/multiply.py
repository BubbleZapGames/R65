"""
Shared multiplication helpers for MIR lowering.

Provides shift-and-add multiplication for struct array indexing optimization.
Used by both expression.py and assignment.py lowerers.

LIMITATION - Large Struct Array Indexing:
    For structs larger than 16 bytes, array indexing uses the SNES hardware
    multiplier (8x8=16 bit) via mul16(). This limits VARIABLE indices to 0-255.

    | Struct Size  | Index Type | Max Index | Method              |
    |--------------|------------|-----------|---------------------|
    | 1-16 bytes   | any        | 65535     | Shift-and-add       |
    | 17-255 bytes | constant   | 65535     | Compile-time calc   |
    | 17-255 bytes | variable   | 255       | mul16() hardware    |

    Constant indices work correctly for any value because the multiplication
    is computed at compile time. Variable indices > 255 will silently use
    only the low 8 bits, producing incorrect results.

    To support larger arrays of large structs with variable indexing:
    - Restructure to use smaller structs (≤16 bytes)
    - Use manual pointer arithmetic with a software 16x8 multiply
"""

from typing import TYPE_CHECKING, Callable

from r65.compiler.mir.nodes import (
    VirtualRegister, Immediate, BinaryOp, Call, Argument,
    ArgumentMechanism, HardwareRegister,
)
from r65.compiler.errors import MIRLoweringError

if TYPE_CHECKING:
    from r65.compiler.mir.context import LoweringContext


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
):
    """
    Compute scaled index for array element access: index * struct_size.

    Unlike compute_array_field_offset(), does NOT add the field offset.
    The field offset should be folded into the MemoryLocation address
    constant instead, saving a CLC+ADC per non-zero field access.

    Args:
        index_operand: The index value (VirtualRegister, HardwareRegister, or Immediate)
        struct_size: Size of each struct element in bytes
        type_info: Type information for intermediate values
        ctx: Lowering context for allocating virtual registers
        emit: Function to emit MIR instructions

    Returns:
        Operand containing index * struct_size
    """
    if struct_size == 1:
        return index_operand
    elif struct_size <= 16:
        return emit_shift_and_add_multiply(
            index_operand, struct_size, type_info, ctx, emit
        )
    else:
        _check_runtime_mul_available(ctx, struct_size)
        from r65.compiler.hir.types import BasicTypeInfo
        u8_type = BasicTypeInfo('u8')
        u16_type = BasicTypeInfo('u16')

        scaled_index = ctx.alloc_vreg(type_info, "scaled_index")
        emit(Call(
            function='mul16',
            args=[
                Argument(value=Immediate(struct_size), mechanism=ArgumentMechanism.REGISTER,
                         location=HardwareRegister('A'), param_type=u8_type),
                Argument(value=index_operand, mechanism=ArgumentMechanism.STACK,
                         location=None, param_type=u16_type),
            ],
            returns=[scaled_index],
            is_far=True
        ))
        return scaled_index


def compute_array_field_offset(
    index_operand,
    struct_size: int,
    field_offset: int,
    type_info,
    ctx: 'LoweringContext',
    emit: Callable,
):
    """
    Compute byte offset for array[index].field access.

    Result = (index * struct_size) + field_offset

    Optimization strategies (see docs/struct-array-indexing.md):
    - struct_size == 1: no multiplication needed
    - struct_size <= 16: use shift or shift-and-add decomposition
    - struct_size > 16: call mul() runtime function

    Args:
        index_operand: The index value (VirtualRegister, HardwareRegister, or Immediate)
        struct_size: Size of each struct element in bytes
        field_offset: Offset of the field within the struct
        type_info: Type information for intermediate values
        ctx: Lowering context for allocating virtual registers
        emit: Function to emit MIR instructions

    Returns:
        Operand containing the computed byte offset
    """
    scaled_index = compute_scaled_index(
        index_operand, struct_size, type_info, ctx, emit
    )

    # Add field offset if non-zero
    if field_offset == 0:
        return scaled_index
    else:
        final_offset = ctx.alloc_vreg(type_info, "field_offset")
        emit(BinaryOp(
            dest=final_offset,
            left=scaled_index,
            right=Immediate(field_offset),
            op='+',
            type_info=type_info
        ))
        return final_offset
