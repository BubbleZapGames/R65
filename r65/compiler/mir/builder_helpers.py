# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
MIR Builder helper classes.

Extracted from MIRBuilder to improve modularity and reduce file size.
"""

from typing import Dict, Any, TYPE_CHECKING
from r65.compiler.hir.unified_type_utils import get_unified_type_size

if TYPE_CHECKING:
    pass


class TypeSizeCalculator:
    """
    Calculates sizes of types for MIR generation.

    Used for memory allocation, array indexing, and struct field access.
    """

    # Cache for type sizes
    _size_cache: Dict[Any, int] = {}

    @classmethod
    def get_size(cls, type_info) -> int:
        """
        Get size of a type in bytes.

        Args:
            type_info: Type information (TypeInfo or similar)

        Returns:
            Size in bytes
        """
        # Check cache first
        type_key = id(type_info)
        if type_key in cls._size_cache:
            return cls._size_cache[type_key]

        size = cls._calculate_size(type_info)
        cls._size_cache[type_key] = size
        return size

    @classmethod
    def _calculate_size(cls, type_info) -> int:
        """Calculate size of a type."""
        try:
            return get_unified_type_size(type_info)
        except Exception:
            return 1

    @classmethod
    def clear_cache(cls):
        """Clear the type size cache."""
        cls._size_cache.clear()


