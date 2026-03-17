# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Type utilities for code generation.

Thin wrapper around the canonical implementation in hir/unified_type_utils.py.
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.mir.nodes import VirtualRegister


# =============================================================================
# Type Size Constants
# =============================================================================

# Size in bytes for basic types (kept for direct importers)
TYPE_SIZES: Dict[str, int] = {
    'u8': 1, 'i8': 1, 'bool': 1,
    'u16': 2, 'i16': 2,
    'u24': 3,  # For far pointers
}


# =============================================================================
# Type Size Utilities
# =============================================================================

def get_type_size(type_info) -> int:
    """
    Get size of a type in bytes.

    Delegates to the canonical implementation in hir/unified_type_utils.py.

    Args:
        type_info: Type information object

    Returns:
        Size in bytes (defaults to 1 if unknown)
    """
    from r65.compiler.hir.unified_type_utils import get_unified_type_size
    try:
        return get_unified_type_size(type_info)
    except Exception:
        return 1


def get_vreg_size(vreg: 'VirtualRegister') -> int:
    """Get size of virtual register in bytes."""
    return get_type_size(vreg.type_info)
