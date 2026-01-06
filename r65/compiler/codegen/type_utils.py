"""
Type utilities for code generation.

Provides centralized type size calculations and type checking utilities
used across the code generation pipeline.
"""

from typing import Any, Dict, Optional


# =============================================================================
# Type Size Constants
# =============================================================================

# Size in bytes for basic types
TYPE_SIZES: Dict[str, int] = {
    'u8': 1, 'i8': 1, 'bool': 1,
    'u16': 2, 'i16': 2,
    'u24': 3,  # For far pointers
}

# Types that are 16-bit
TYPES_16BIT = {'u16', 'i16'}

# Types that are 8-bit
TYPES_8BIT = {'u8', 'i8', 'bool'}


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

    # Handle StructTypeInfo
    if hasattr(type_info, 'fields'):
        total = 0
        for field in type_info.fields:
            total += get_type_size(field.field_type)
        return total

    # Handle PointerTypeInfo
    if hasattr(type_info, 'pointee_type'):
        # Near pointer = 2 bytes, far pointer = 3 bytes
        if hasattr(type_info, 'is_far') and type_info.is_far:
            return 3
        return 2

    # Check for pointer type names
    type_name_str = str(type_info)
    if type_name_str.startswith('near<'):
        return 2  # 16-bit pointer
    elif type_name_str.startswith('far<'):
        return 3  # 24-bit pointer

    # Default to 1 byte
    return 1


def is_16bit(type_info) -> bool:
    """
    Check if type is 16-bit.

    Args:
        type_info: Type information object

    Returns:
        True if type is 16-bit (u16, i16)
    """
    if type_info is None:
        return False

    # Handle BasicTypeInfo or similar with name attribute
    if hasattr(type_info, 'name'):
        return type_info.name in TYPES_16BIT

    # Handle string type names directly
    if isinstance(type_info, str):
        return type_info in TYPES_16BIT

    return False


def is_8bit(type_info) -> bool:
    """
    Check if type is 8-bit.

    Args:
        type_info: Type information object

    Returns:
        True if type is 8-bit (u8, i8, bool)
    """
    if type_info is None:
        return True  # Default to 8-bit

    # Handle BasicTypeInfo or similar with name attribute
    if hasattr(type_info, 'name'):
        return type_info.name in TYPES_8BIT

    # Handle string type names directly
    if isinstance(type_info, str):
        return type_info in TYPES_8BIT

    return True  # Default to 8-bit


def is_signed(type_info) -> bool:
    """
    Check if type is signed.

    Args:
        type_info: Type information object

    Returns:
        True if type is signed (i8, i16)
    """
    if type_info is None:
        return False

    # Handle BasicTypeInfo or similar with name attribute
    if hasattr(type_info, 'name'):
        return type_info.name in ('i8', 'i16')

    # Handle string type names directly
    if isinstance(type_info, str):
        return type_info in ('i8', 'i16')

    return False


# =============================================================================
# Bit Utilities
# =============================================================================

def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


