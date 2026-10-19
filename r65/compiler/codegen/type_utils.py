# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type utilities for code generation.

Width and A-residency predicates shared by the allocator and the emitters.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.mir.nodes import VirtualRegister


def get_type_size(type_info) -> int:
    """Get size of a type in bytes."""
    if type_info is None:
        return 1
    return type_info.size_bytes


def scalar_width(type_info):
    """Bytes a scalar of *type_info* moves as: 1, 2 or 3, or None.

    The one width rule for codegen. A newtype answers for its payload, an enum
    by its discriminant width, a pointer or `fn` pointer by near/far. A struct,
    array or missing type is None: a 2-byte struct must not ride the 16-bit A
    path just because it is two bytes.
    """
    from r65.compiler.hir.types import (BasicTypeInfo, EnumTypeInfo,
                                        FunctionTypeInfo, PointerTypeInfo,
                                        strip_newtype)
    type_info = strip_newtype(type_info)
    if isinstance(type_info, (PointerTypeInfo, FunctionTypeInfo)):
        return 3 if type_info.is_far else 2
    if isinstance(type_info, (BasicTypeInfo, EnumTypeInfo)):
        return type_info.size_bytes
    return None


def is_16bit_scalar(type_info) -> bool:
    """Whether *type_info* moves through A as one 16-bit value."""
    return scalar_width(type_info) == 2


def is_signed_scalar(type_info) -> bool:
    """i8/i16, or a newtype over one: the types `>>` and `/` must shift
    arithmetically, keeping the sign bit."""
    from r65.compiler.hir.types import BasicTypeInfo, strip_newtype
    t = strip_newtype(type_info) if type_info is not None else None
    return isinstance(t, BasicTypeInfo) and t.name in ('i8', 'i16')


def is_far_ptr(type_info) -> bool:
    """True for the 3-byte pointer types: ``far *T`` and ``far fn()``.

    Neither the 16-bit nor the 8-bit path can carry three bytes, so every
    caller that copies a pointer has to ask this explicitly or it silently
    moves only the low byte.
    """
    from r65.compiler.hir.types import FunctionTypeInfo, PointerTypeInfo, strip_newtype
    type_info = strip_newtype(type_info)
    return (isinstance(type_info, (PointerTypeInfo, FunctionTypeInfo))
            and type_info.is_far)


def pointer_element_size(type_info) -> int:
    """The pointee's size for a pointer type, else 0 — what `ptr++` steps by."""
    pointee = getattr(type_info, 'pointee_type', None)
    if pointee is None:
        return 0
    from r65.compiler.hir.unified_type_utils import get_unified_type_size
    try:
        return get_unified_type_size(pointee)
    except Exception:
        return 0


def arg_param_size(arg) -> int:
    """Bytes an argument occupies at its parameter: the parameter's type, else
    the value's, else an untyped immediate's own width (1, 2 or 3)."""
    for type_info in (arg.param_type, getattr(arg.value, 'type_info', None)):
        if type_info:
            return get_type_size(type_info)
    from r65.compiler.mir.nodes import Immediate
    if isinstance(arg.value, Immediate):
        value = arg.value.value
        if isinstance(value, int):
            if value > 0xFFFF or value < -32768:
                return 3
            if value > 0xFF or value < -128:
                return 2
    return 1


def get_vreg_size(vreg: 'VirtualRegister') -> int:
    """Get size of virtual register in bytes."""
    if vreg.type_info is None:
        return 1
    return vreg.type_info.size_bytes


def convert_operand_widths(instr):
    """The (source, target) widths `select_type_convert` dispatches on.

    Returns ``None`` for a pointer conversion, which has its own emitter.

    It exists so the register allocator asks the same question as the
    emitter. A one-byte scalar is 1; anything else counts as two.
    """
    from r65.compiler.hir.types import PointerTypeInfo, strip_newtype

    source_type = strip_newtype(instr.source_type)
    target_type = strip_newtype(instr.target_type)

    if isinstance(source_type, PointerTypeInfo) or isinstance(target_type, PointerTypeInfo):
        return None

    def width(type_info):
        return 1 if scalar_width(type_info) == 1 else 2

    return (width(source_type), width(target_type))


def is_narrowing_convert(instr) -> bool:
    """True for a TypeConvert that truncates two bytes down to one.

    Narrowing with the source in A is a bare `SEP #$20`
    (`_emit_narrowing_conversion`); see `is_a_resident_convert` for the
    widening case.

    Shares `convert_operand_widths` with `select_type_convert`, so the
    allocator cannot come to a different conclusion than the emitter it is
    predicting.
    """
    from r65.compiler.mir.nodes import TypeConvert

    if type(instr) is not TypeConvert:
        return False

    widths = convert_operand_widths(instr)
    return widths == (2, 1)


def is_a_resident_convert(instr) -> bool:
    """True for a TypeConvert whose result may live in A.

    A narrowing convert is a bare `SEP #$20` there, and an unsigned widening
    is `REP #$20 / AND #$00FF`. A signed widening is not: its sign extension
    branches and writes the high byte through memory.
    """
    from r65.compiler.hir.types import strip_newtype

    if is_narrowing_convert(instr):
        return True
    if convert_operand_widths(instr) != (1, 2):
        return False
    return not str(strip_newtype(instr.source_type)).startswith('i')


def is_inc_dec_step(value, register: str) -> bool:
    """Whether `reg = reg ± value` is emitted as repeated INC/DEC: 1 or 2, or
    3 and 4 in X/Y (INX x4 is 8 cycles to TXA / CLC / ADC #4 / TAX's 9; A
    takes 3 and 4 as CLC / ADC). The slot allocator relies on X/Y steps
    leaving A alone."""
    return value in (1, 2) or (value in (3, 4) and register in ('X', 'Y'))


def defines_in_a(instr) -> bool:
    """True for a node whose codegen computes its result in A on the way to
    its dest: arithmetic, a load, a bank byte, or a convert
    `is_a_resident_convert` accepts.

    The slot allocator coalesces such a dest into A, and the inliner returns
    it as A; both must agree with the emitters on which nodes these are.
    """
    from r65.compiler.mir.nodes import (
        BinaryOp, UnaryOp, Load, LoadIndirect, BankByte, TypeConvert,
    )
    kind = type(instr)
    if kind is TypeConvert:
        return is_a_resident_convert(instr)
    return kind in (BinaryOp, UnaryOp, Load, LoadIndirect, BankByte)
