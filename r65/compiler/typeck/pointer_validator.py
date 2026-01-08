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

    def _needs_far_pointer(self, operand: HIRExpression) -> bool:
        """
        Determine if taking address of operand requires a far pointer.

        Far pointers (24-bit) are needed for:
        - #[ram] variables: stored in bank $7E ($7E2000-$7FFFFF)
        - #[rom] variables: stored in ROM banks

        Near pointers (16-bit) are sufficient for:
        - #[zeropage] variables: bank 0 ($0000-$00FF)
        - #[lowram] variables: bank 0 ($0000-$1FFF)
        - #[hw] variables: typically bank 0
        - Local variables: on stack in current bank

        Args:
            operand: The expression being addressed

        Returns:
            True if far pointer needed, False for near pointer
        """
        # Only identifiers can have storage attributes
        if not isinstance(operand, HIRIdentifier):
            return False

        symbol = operand.symbol
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

        # RAM and ROM require far pointers (different banks)
        storage_kind = static_decl.storage_attr.storage_kind
        return storage_kind in (StorageKind.RAM, StorageKind.ROM)
