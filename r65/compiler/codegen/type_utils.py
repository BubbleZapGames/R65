# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type utilities for code generation.

Delegates to TypeInfo.size_bytes for HIR types.
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

    Narrowing is the one conversion direction that costs nothing when the
    value is already in A: `_emit_narrowing_conversion` recognises an A
    destination and emits a bare `SEP #$20`. Widening has no such path —
    it ends in `STA dest_loc` plus sign/zero-extension writes *to that
    location* — so the register allocator must be able to tell the two
    apart before it lets a conversion result live in A.

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
