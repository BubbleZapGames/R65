"""
Type utilities for code generation.

Provides centralized type size calculations used across the code generation pipeline.
"""

from typing import Dict


# =============================================================================
# Type Size Constants
# =============================================================================

# Size in bytes for basic types
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

    Handles:
    - Basic types (u8, i8, u16, i16, bool)
    - Array types
    - Struct types
    - Pointer types

    Args:
        type_info: Type information object

    Returns:
        Size in bytes (defaults to 1 if unknown)
    """
    if type_info is None:
        return 1

    # Handle BasicTypeInfo or similar with name attribute
    if hasattr(type_info, 'name'):
        type_name = type_info.name
        if type_name in TYPE_SIZES:
            return TYPE_SIZES[type_name]

    # Handle string type names directly
    if isinstance(type_info, str):
        if type_info in TYPE_SIZES:
            return TYPE_SIZES[type_info]

    # Handle ArrayTypeInfo
    if hasattr(type_info, 'element_type') and hasattr(type_info, 'size'):
        elem_size = get_type_size(type_info.element_type)
        return elem_size * type_info.size

    # Handle StructTypeInfo (fields on type directly or via definition)
    if hasattr(type_info, 'fields'):
        total = 0
        for field in type_info.fields:
            total += get_type_size(field.field_type)
        return total

    # Handle StructTypeInfo with definition attribute (HIR StructTypeInfo)
    if hasattr(type_info, 'definition') and hasattr(type_info.definition, 'fields'):
        total = 0
        for field in type_info.definition.fields:
            total += get_type_size(field.field_type)
        return total

    # Handle PointerTypeInfo
    if hasattr(type_info, 'pointee_type'):
        # Near pointer = 2 bytes, far pointer = 3 bytes
        if hasattr(type_info, 'is_far') and type_info.is_far:
            return 3
        return 2

    # Check for pointer type names (fallback for string representations)
    type_name_str = str(type_info)
    if type_name_str.startswith('far *'):
        return 3  # 24-bit far pointer
    elif type_name_str.startswith('*'):
        return 2  # 16-bit near pointer

    # Default to 1 byte
    return 1
