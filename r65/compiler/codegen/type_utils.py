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
