# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Pointer operation validation for R65 type checker.

Handles validation of dereference (*ptr) and address-of (&var) operations.
"""

from typing import Callable
from r65.compiler.hir import (
    HIRDereference, HIRAddressOf, HIRExpression,
    HIRIdentifier, HIRArrayIndex, HIRFieldAccess, HIRStaticDecl
)
from r65.compiler.hir.types import TypeInfo, PointerTypeInfo
from r65.compiler.hir.attributes import StorageAttribute, StorageKind
from r65.compiler.typeck.errors import TypeCheckError


class PointerValidator:
    """Validates pointer operations."""

    def __init__(self, check_expression_fn: Callable):
        """
        Initialize with expression checker callback.

        Args:
            check_expression_fn: Function to type check expressions
        """
        self.check_expression = check_expression_fn

    def check_dereference(self, expr: HIRDereference) -> TypeInfo:
        """Type check pointer dereference (*ptr)."""
        pointer_type = self.check_expression(expr.pointer)

        # Pointer must be a pointer type
        if not isinstance(pointer_type, PointerTypeInfo):
            raise TypeCheckError(
                f"Cannot dereference non-pointer type {pointer_type}",
                source_loc=expr.pointer.source_loc
            )

        # Dereference yields the pointee type
        expr.expr_type = pointer_type.pointee_type
        return expr.expr_type

    def check_addressof(self, expr: HIRAddressOf) -> TypeInfo:
        """Type check address-of operator (&variable)."""
        # Checked before the operand, so `&ARR[i]` reports the address-of
        # problem rather than the generic whole-element one it would trip
        # inside check_array_index.
        self._reject_soa_addressof(expr)

        operand_type = self.check_expression(expr.operand)

        # Operand must be an lvalue (identifier, array index, or field access)
        if not isinstance(expr.operand, (HIRIdentifier, HIRArrayIndex, HIRFieldAccess)):
            raise TypeCheckError(
                "Cannot take address of non-lvalue expression",
                source_loc=expr.operand.source_loc
            )

        # Determine if far pointer is needed based on storage attribute
        is_far = self._needs_far_pointer(expr.operand)

        pointer_type = PointerTypeInfo(is_far=is_far, pointee_type=operand_type)
        expr.expr_type = pointer_type
        return expr.expr_type

    @staticmethod
    def _reject_soa_addressof(expr: HIRAddressOf) -> None:
        """Refuse `&ARR` and `&ARR[i]` for a `#[soa]` array.

        Neither has a single address: the array is stored as one plane per
        field, so the whole array is not contiguous and an element is spread
        across every plane. Callers pass the index instead.
        """
        from r65.compiler.typeck.type_utils import soa_static_of, soa_static_of_index

        decl = soa_static_of(expr.operand)
        if decl is not None:
            raise TypeCheckError(
                f"cannot take the address of #[soa] array '{decl.name}'",
                source_loc=expr.source_loc,
                hint="its fields live in separate arrays, so there is no single "
                     "base address; index a field directly instead",
            )

        decl = soa_static_of_index(expr.operand)
        if decl is not None:
            raise TypeCheckError(
                f"cannot take the address of an element of #[soa] array '{decl.name}'",
                source_loc=expr.source_loc,
                hint=f"an SoA element has no contiguous base; pass the index and "
                     f"read `{decl.name}[i].field` at the use site",
            )

        # `&POOL[i].field` (and `&self.field` in an #[soa] method, which is the
        # same thing by the time it gets here), or a payload binding's field,
        # already rewritten to its plane `PLANE[i]`. The address exists, but it
        # is not the element's: the next byte is the next element's same field.
        operand = expr.operand
        while isinstance(operand, HIRFieldAccess):
            operand = operand.base
        pool = None
        if operand is not expr.operand:
            decl = soa_static_of_index(operand)
            pool = decl.name if decl is not None else None
        elif isinstance(operand, HIRArrayIndex):
            plane = getattr(getattr(operand.array, 'symbol', None), 'definition', None)
            if getattr(plane, 'is_synthetic', False) and getattr(plane, 'soa_stride', None):
                pool = plane.name.split('__')[0]
        if pool is not None:
            raise TypeCheckError(
                f"cannot take the address of a field of #[soa] array '{pool}'",
                source_loc=expr.source_loc,
                hint="consecutive bytes of a field plane are the same field of "
                     "consecutive elements, so a pointer would walk the plane, not "
                     f"the element; read or write `{pool}[i].field` directly, or "
                     "copy it to a local",
            )

    @staticmethod
    def _needs_far_pointer(operand: HIRExpression) -> bool:
        """
        Determine if taking address of operand requires a far pointer.

        Far pointers (24-bit) are needed for:
        - #[ram] variables: stored in bank $7E ($7E2000-$7FFFFF)
        - ROM variables (immutable statics) in banks other than 0

        Near pointers (16-bit) are sufficient for:
        - #[zeropage] variables: bank 0 ($0000-$00FF)
        - #[lowram] variables: bank 0 ($0000-$1FFF)
        - #[hw] variables: typically bank 0
        - ROM variables in bank 0: same bank as code
        - Local variables: on stack in current bank

        Args:
            operand: The expression being addressed

        Returns:
            True if far pointer needed, False for near pointer
        """
        # Extract symbol from the operand (may be wrapped in ArrayIndex or FieldAccess)
        if isinstance(operand, HIRIdentifier):
            symbol = operand.symbol
        elif isinstance(operand, HIRArrayIndex):
            # &p[i] through a pointer lives wherever p points, so it is as
            # far as p is - a far pointer parameter's element is not in bank 0.
            base = operand.array
            base_type = getattr(base, 'expr_type', None)
            if isinstance(base_type, PointerTypeInfo):
                return base_type.is_far
            # &array[i] — check the array's storage
            if isinstance(base, HIRIdentifier):
                symbol = base.symbol
            else:
                return False
        elif isinstance(operand, HIRFieldAccess):
            # &var.field — check the struct's storage
            base = operand.base
            if isinstance(base, HIRIdentifier):
                symbol = base.symbol
            else:
                return False
        else:
            return False
        if not symbol or not symbol.definition:
            return False

        # Check if it's a static variable with storage attribute
        if not isinstance(symbol.definition, HIRStaticDecl):
            return False

        static_decl = symbol.definition
        if not static_decl.storage_attr:
            return False

        if not isinstance(static_decl.storage_attr, StorageAttribute):
            return False

        storage_kind = static_decl.storage_attr.storage_kind

        # RAM always requires far pointers (bank $7E)
        if storage_kind == StorageKind.RAM:
            return True

        # ZEROPAGE, LOWRAM, and HW are all in bank 0, so near pointers work
        # Note: Immutable statics (ROM) don't have storage_attr, they're handled
        # earlier by returning False when storage_attr is None.
        return False
