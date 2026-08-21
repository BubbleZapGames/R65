# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type utilities for code generation.

Delegates to TypeInfo.size_bytes for HIR types.
"""

from typing import TYPE_CHECKING

from r65.compiler.hir.types import _TYPE_SIZES as TYPE_SIZES

if TYPE_CHECKING:
    from r65.compiler.mir.nodes import VirtualRegister


def get_type_size(type_info) -> int:
    """Get size of a type in bytes."""
    if type_info is None:
        return 1
    return type_info.size_bytes


def get_vreg_size(vreg: 'VirtualRegister') -> int:
    """Get size of virtual register in bytes."""
    if vreg.type_info is None:
        return 1
    return vreg.type_info.size_bytes


def convert_operand_widths(instr):
    """The (source, target) widths `select_type_convert` dispatches on.

    Returns ``None`` for a pointer conversion, which has its own emitter.

    The rule is deliberately a *name* test rather than ``size_bytes``: it is
    the one instruction selection has always used, and this exists so the
    register allocator asks the same question rather than a second, subtly
    different one. Newtypes are stripped first so they answer for their
    payload; anything else that is not a one-byte scalar counts as two.
    """
    from r65.compiler.hir.types import PointerTypeInfo, strip_newtype

    source_type = strip_newtype(instr.source_type)
    target_type = strip_newtype(instr.target_type)

    if isinstance(source_type, PointerTypeInfo) or isinstance(target_type, PointerTypeInfo):
        return None

    one_byte = ('u8', 'i8', 'bool')
    return (1 if str(source_type) in one_byte else 2,
            1 if str(target_type) in one_byte else 2)


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
