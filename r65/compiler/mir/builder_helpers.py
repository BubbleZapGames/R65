"""
MIR Builder helper classes.

Extracted from MIRBuilder to improve modularity and reduce file size.
"""

from typing import Union, Optional, Dict, Any, List, TYPE_CHECKING
from r65.compiler.mir.nodes import (
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation,
    MIRInstruction, Load, Store, BinaryOp, UnaryOp, TypeConvert,
    Compare, Move, Call
)
from r65.compiler.errors import MIRLoweringError
from r65.compiler.hir.unified_type_utils import get_unified_type_size

if TYPE_CHECKING:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.hir.nodes import HIRExpression


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


class MemoryLocationBuilder:
    """
    Builds MemoryLocation objects for MIR.

    Handles symbol resolution and address calculation for
    static variables, arrays, and struct fields.
    """

    def __init__(self, builder: 'MIRBuilder'):
        self.builder = builder

    def create_offset_location(self, base_loc: MemoryLocation, offset: int,
                               size: int) -> MemoryLocation:
        """
        Create a new MemoryLocation offset from a base location.

        Args:
            base_loc: Base memory location
            offset: Byte offset from base
            size: Size of the new location

        Returns:
            New MemoryLocation at the offset
        """
        return MemoryLocation(
            symbol=base_loc.symbol,
            address=base_loc.address + offset if base_loc.address is not None else None,
            size=size,
            is_zeropage=base_loc.is_zeropage,
            bank=base_loc.bank
        )

    def has_explicit_location(self, symbol) -> bool:
        """Check if a symbol has an explicit memory location."""
        if symbol is None:
            return False

        definition = getattr(symbol, 'definition', None)
        if definition is None:
            return False

        storage_attr = getattr(definition, 'storage_attr', None)
        if storage_attr is None:
            return False

        return storage_attr.address is not None

    def get_memory_location(self, symbol) -> MemoryLocation:
        """
        Get the memory location for a symbol.

        Args:
            symbol: Symbol to get location for

        Returns:
            MemoryLocation for the symbol

        Raises:
            MIRLoweringError: If symbol has no storage
        """
        if symbol is None:
            raise MIRLoweringError("Cannot get memory location for None symbol")

        definition = getattr(symbol, 'definition', None)
        if definition is None:
            raise MIRLoweringError(f"Symbol '{symbol.name}' has no definition")

        storage_attr = getattr(definition, 'storage_attr', None)
        var_type = getattr(definition, 'var_type', None)

        size = TypeSizeCalculator.get_size(var_type)

        if storage_attr is None:
            # No storage attribute - this might be a local variable
            return MemoryLocation(
                symbol=symbol,
                address=None,
                size=size,
                is_zeropage=False
            )

        # Determine if zeropage based on storage kind
        from r65.compiler.hir.attributes import StorageKind
        is_zp = storage_attr.storage_kind == StorageKind.ZEROPAGE

        return MemoryLocation(
            symbol=symbol,
            address=storage_attr.address,
            size=size,
            is_zeropage=is_zp
        )
