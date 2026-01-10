"""Type comparison and compatibility utilities for R65."""

from typing import Optional
from r65.compiler.hir import (
    TypeInfo, BasicTypeInfo, ArrayTypeInfo, SliceTypeInfo, PointerTypeInfo,
    FunctionTypeInfo, StructTypeInfo, EnumTypeInfo,
    NeverTypeInfo, RegisterTypeInfo
)


class TypeUtils:
    """Utilities for type comparison and compatibility."""

    @staticmethod
    def _pointee_types_compatible(t1: TypeInfo, t2: TypeInfo) -> bool:
        """
        Check if pointee types are compatible for pointer assignment.

        This allows a pointer to a sized array [T; N] to be compatible with
        a pointer to an unsized slice [T].
        """
        # Exact match
        if TypeUtils.types_equal(t1, t2):
            return True

        # Array [T; N] can match slice [T]
        if isinstance(t1, ArrayTypeInfo) and isinstance(t2, SliceTypeInfo):
            return TypeUtils.types_equal(t1.element_type, t2.element_type)
        if isinstance(t1, SliceTypeInfo) and isinstance(t2, ArrayTypeInfo):
            return TypeUtils.types_equal(t1.element_type, t2.element_type)

        return False

    @staticmethod
    def types_equal(t1: TypeInfo, t2: TypeInfo) -> bool:
        """Check if two types are exactly equal."""
        # Handle pointer type special case: array is compatible with slice
        if isinstance(t1, PointerTypeInfo) and isinstance(t2, PointerTypeInfo):
            if t1.is_far != t2.is_far:
                return False
            # Allow sized array pointer to match slice pointer
            return TypeUtils._pointee_types_compatible(t1.pointee_type, t2.pointee_type)

        if type(t1) != type(t2):
            return False

        if isinstance(t1, BasicTypeInfo):
            return t1.name == t2.name

        elif isinstance(t1, ArrayTypeInfo):
            return (t1.size == t2.size and
                    TypeUtils.types_equal(t1.element_type, t2.element_type))

        elif isinstance(t1, SliceTypeInfo):
            return TypeUtils.types_equal(t1.element_type, t2.element_type)

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

        elif isinstance(t1, (StructTypeInfo, EnumTypeInfo)):
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
        - Pointer coercion: far/near and sized array to slice
        """
        # Exact match
        if TypeUtils.types_equal(t1, t2):
            return True

        # Pointer compatibility: allow array-to-slice coercion (same far/near)
        if isinstance(t1, PointerTypeInfo) and isinstance(t2, PointerTypeInfo):
            # far/near must match - no implicit coercion between them
            if t1.is_far == t2.is_far:
                # Pointee types must be compatible (allows [T; N] -> [T])
                if TypeUtils._pointee_types_compatible(t1.pointee_type, t2.pointee_type):
                    return True

        # Enum to/from integer compatibility
        # Enums are compatible with u8 (their underlying type)
        if isinstance(t1, EnumTypeInfo) and isinstance(t2, BasicTypeInfo) and t2.name == 'u8':
            return True
        if isinstance(t2, EnumTypeInfo) and isinstance(t1, BasicTypeInfo) and t1.name == 'u8':
            return True

        return False

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

        # Enum <-> Integer
        if isinstance(from_type, EnumTypeInfo) and TypeUtils.is_integer_type(to_type):
            return True
        if TypeUtils.is_integer_type(from_type) and isinstance(to_type, EnumTypeInfo):
            return True

        return False
