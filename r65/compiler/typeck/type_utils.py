# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Type comparison and compatibility utilities for R65."""

from typing import Optional, Tuple
from r65.compiler.hir import (
    TypeInfo, BasicTypeInfo, ArrayTypeInfo, PointerTypeInfo,
    FunctionTypeInfo, StructTypeInfo, EnumTypeInfo,
    NeverTypeInfo, RegisterTypeInfo
)
from r65.compiler.hir.types import TraitTypeInfo


# Valid ranges for integer types
TYPE_RANGES = {
    'u8': (0, 255),
    'i8': (-128, 127),
    'u16': (0, 65535),
    'i16': (-32768, 32767),
}


def get_type_range(type_name: str) -> Optional[Tuple[int, int]]:
    """Get the valid (min, max) range for an integer type, or None if unknown."""
    return TYPE_RANGES.get(type_name)


def value_fits_type(value: int, type_name: str) -> bool:
    """Check if an integer value fits within the range of a type."""
    range_info = TYPE_RANGES.get(type_name)
    if range_info is None:
        return True  # Unknown type, assume it fits
    min_val, max_val = range_info
    return min_val <= value <= max_val


class TypeUtils:
    """Utilities for type comparison and compatibility."""

    @staticmethod
    def _pointee_types_compatible(t1: TypeInfo, t2: TypeInfo) -> bool:
        """
        Check if pointee types are compatible for pointer assignment.

        This allows *[T; N] to be compatible with *T (array pointer coerces
        to element pointer).
        """
        # Exact match
        if TypeUtils.types_equal(t1, t2):
            return True

        # *[T; N] can match *T (array pointer coerces to element pointer)
        if isinstance(t1, ArrayTypeInfo) and not isinstance(t2, ArrayTypeInfo):
            return TypeUtils.types_equal(t1.element_type, t2)
        if isinstance(t2, ArrayTypeInfo) and not isinstance(t1, ArrayTypeInfo):
            return TypeUtils.types_equal(t2.element_type, t1)

        return False

    @staticmethod
    def types_equal(t1: TypeInfo, t2: TypeInfo) -> bool:
        """Check if two types are exactly equal."""
        # Handle pointer type special case: array is compatible with slice
        if isinstance(t1, PointerTypeInfo) and isinstance(t2, PointerTypeInfo):
            if t1.is_far != t2.is_far:
                return False
            # Allow *[T; N] to match *T via array pointer coercion
            return TypeUtils._pointee_types_compatible(t1.pointee_type, t2.pointee_type)

        if type(t1) != type(t2):
            return False

        if isinstance(t1, BasicTypeInfo):
            return t1.name == t2.name

        elif isinstance(t1, ArrayTypeInfo):
            return (t1.size == t2.size and
                    TypeUtils.types_equal(t1.element_type, t2.element_type))

        elif isinstance(t1, FunctionTypeInfo):
            if t1.is_far != t2.is_far:
                return False
            if len(t1.param_types) != len(t2.param_types):
                return False
            for p1, p2 in zip(t1.param_types, t2.param_types):
                if not TypeUtils.types_equal(p1, p2):
                    return False
            if t1.return_type is None and t2.return_type is None:
                return True
            if t1.return_type is None or t2.return_type is None:
                return False
            return TypeUtils.types_equal(t1.return_type, t2.return_type)

        elif isinstance(t1, (StructTypeInfo, EnumTypeInfo, TraitTypeInfo)):
            return t1.name == t2.name

        elif isinstance(t1, NeverTypeInfo):
            return True

        elif isinstance(t1, RegisterTypeInfo):
            return t1.register_name == t2.register_name

        return False

    @staticmethod
    def is_integer_type(t: TypeInfo) -> bool:
        """Check if type is an integer type."""
        return (isinstance(t, BasicTypeInfo) and
                t.name in ['u8', 'u16', 'i8', 'i16'])

    @staticmethod
    def types_compatible(t1: TypeInfo, t2: TypeInfo) -> bool:
        """
        Check if two types are compatible for assignment.

        This is more permissive than types_equal - it allows:
        - Exact type matches
        - Enum types with compatible integer types (u8)
        - Pointer coercion: far/near and *[T; N] to *T
        """
        # Exact match
        if TypeUtils.types_equal(t1, t2):
            return True

        # Pointer compatibility: allow *[T; N] to *T coercion and struct-to-trait coercion
        if isinstance(t1, PointerTypeInfo) and isinstance(t2, PointerTypeInfo):
            # Pointee types must be compatible (allows [T; N] -> T)
            if TypeUtils._pointee_types_compatible(t1.pointee_type, t2.pointee_type):
                # Allow far/near pointer widening (near *T compatible with far *T)
                return True
            # far/near must match for trait coercion
            if t1.is_far == t2.is_far:
                # *Struct -> *Trait coercion: if struct implements the trait
                # t1 = expected (*Trait), t2 = actual (*Struct)
                if isinstance(t1.pointee_type, TraitTypeInfo) and isinstance(t2.pointee_type, StructTypeInfo):
                    if TypeUtils._struct_implements_trait(t2.pointee_type.name, t1.pointee_type.name):
                        return True
                # *Trait -> *Trait is already handled by types_equal (name equality)

        # Integer type compatibility for comparisons
        # Allow comparing different-size integers (e.g., u16 vs u8)
        # This is common in 65816 code and safe for comparisons
        if isinstance(t1, BasicTypeInfo) and isinstance(t2, BasicTypeInfo):
            int_types = {'u8', 'i8', 'u16', 'i16'}
            if t1.name in int_types and t2.name in int_types:
                return True

        # Enum to/from integer compatibility
        # Enums are compatible with u8 (their underlying type)
        if isinstance(t1, EnumTypeInfo) and isinstance(t2, BasicTypeInfo) and t2.name == 'u8':
            return True
        if isinstance(t2, EnumTypeInfo) and isinstance(t1, BasicTypeInfo) and t1.name == 'u8':
            return True

        return False

    @staticmethod
    def _struct_implements_trait(struct_name: str, trait_name: str) -> bool:
        """Check if a struct implements a trait by looking for dispatch symbols.

        This uses a naming convention check: the HIR builder registers
        'TraitName.method_name.StructName' dispatch symbols for each trait impl.
        We only need to check if any such symbol exists.
        """
        # We can't easily access the symbol table from a static method,
        # so we use a class variable that gets set by the type checker.
        if TypeUtils._symbol_table is not None:
            # Check if the trait dispatch symbol exists for any method
            trait_symbol = TypeUtils._symbol_table.lookup(trait_name)
            if trait_symbol and hasattr(trait_symbol.definition, 'methods'):
                trait_def = trait_symbol.definition
                if trait_def.methods:
                    first_method = trait_def.methods[0].name
                    dispatch_key = f"{trait_name}.{first_method}.{struct_name}"
                    return TypeUtils._symbol_table.lookup(dispatch_key) is not None
        return False

    # Class variable to hold symbol table reference for trait impl checking
    _symbol_table = None

    @staticmethod
    def is_boolean_type(t: TypeInfo) -> bool:
        """Check if type is boolean."""
        return isinstance(t, BasicTypeInfo) and t.name == 'bool'

    @staticmethod
    def get_integer_size(t: TypeInfo) -> Optional[int]:
        """Get size in bytes of integer type."""
        if isinstance(t, BasicTypeInfo):
            if t.name in ['u8', 'i8', 'bool']:
                return 1
            elif t.name in ['u16', 'i16']:
                return 2
        return None

    @staticmethod
    def is_signed(t: TypeInfo) -> bool:
        """Check if integer type is signed."""
        return isinstance(t, BasicTypeInfo) and t.name in ['i8', 'i16']

    @staticmethod
    def is_aggregate_type(t: TypeInfo) -> bool:
        """Check if type is an aggregate (array or struct) that cannot be passed by value."""
        return isinstance(t, (ArrayTypeInfo, StructTypeInfo))

    @staticmethod
    def can_cast(from_type: TypeInfo, to_type: TypeInfo) -> bool:
        """
        Check if explicit cast is valid.

        Allowed casts:
        - Integer to integer (any size/signedness)
        - Integer/bool to bool
        - Pointer to pointer (any combination)
        - Enum to integer
        - Integer to enum
        """
        # Integer <-> Integer
        if TypeUtils.is_integer_type(from_type) and TypeUtils.is_integer_type(to_type):
            return True

        # Bool <-> Integer
        if TypeUtils.is_boolean_type(from_type) and TypeUtils.is_integer_type(to_type):
            return True
        if TypeUtils.is_integer_type(from_type) and TypeUtils.is_boolean_type(to_type):
            return True

        # Pointer <-> Pointer
        if isinstance(from_type, PointerTypeInfo) and isinstance(to_type, PointerTypeInfo):
            return True

        # Pointer <-> Integer (needed for DMA address setup and low-level programming)
        if isinstance(from_type, PointerTypeInfo) and TypeUtils.is_integer_type(to_type):
            return True
        if TypeUtils.is_integer_type(from_type) and isinstance(to_type, PointerTypeInfo):
            return True

        # Enum <-> Integer
        if isinstance(from_type, EnumTypeInfo) and TypeUtils.is_integer_type(to_type):
            return True
        if TypeUtils.is_integer_type(from_type) and isinstance(to_type, EnumTypeInfo):
            return True

        return False
