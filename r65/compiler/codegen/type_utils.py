"""
Type utilities for code generation.

Thin wrapper around the canonical implementation in hir/unified_type_utils.py.
"""

from typing import Dict


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
