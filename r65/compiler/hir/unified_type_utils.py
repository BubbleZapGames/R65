# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Unified type size utilities for both AST and HIR types.

For HIR types (TypeInfo subclasses), use the `size_bytes` property directly.
This module provides `get_unified_type_size()` which additionally handles
AST type nodes — needed during const evaluation before HIR lowering.
"""

from typing import Any
from r65.compiler.hir.errors import HIRError
from r65.compiler.hir.types import TypeInfo, _TYPE_SIZES


# Import AST types for isinstance checks
try:
    from r65.compiler.frontend import ast as ast_types
except ImportError:
    ast_types = None


def get_unified_type_size(type_obj: Any, symbol_table=None) -> int:
    """
    Get size of a type in bytes, supporting both AST and HIR type objects.

    For HIR TypeInfo objects, delegates to TypeInfo.size_bytes.
    For AST type nodes, resolves size using the symbol table.

    Args:
        type_obj: AST type node, HIR type object, or type name string
        symbol_table: Symbol table for resolving struct definitions (for AST types)

    Returns:
        Size in bytes

    Raises:
        HIRError: If type cannot be resolved or is unsupported
    """
    if type_obj is None:
        return 1

    # HIR TypeInfo — use the size_bytes property directly
    if isinstance(type_obj, TypeInfo):
        return type_obj.size_bytes

    # Handle basic type names (strings)
    if isinstance(type_obj, str):
        if type_obj in _TYPE_SIZES:
            return _TYPE_SIZES[type_obj]
        raise HIRError(f"Unknown basic type: {type_obj}", source_loc=None)

    # --- AST type nodes (used during const evaluation before HIR lowering) ---

    if ast_types and isinstance(type_obj, ast_types.BasicType):
        type_name = type_obj.name
        if type_name in _TYPE_SIZES:
            return _TYPE_SIZES[type_name]
        raise HIRError(f"Unknown basic type: {type_name}", source_loc=None)

    if ast_types and isinstance(type_obj, ast_types.Identifier):
        type_name = type_obj.name
        if type_name in _TYPE_SIZES:
            return _TYPE_SIZES[type_name]
        if symbol_table:
            symbol = symbol_table.lookup(type_name)
            if symbol and hasattr(symbol, 'kind'):
                if hasattr(symbol, 'definition') and symbol.definition:
                    if hasattr(symbol.definition, 'fields'):
                        return _get_ast_struct_size(symbol.definition, symbol_table)
                elif symbol.kind.value == "enum":
                    return 1
        raise HIRError(
            f"Cannot determine size of type: {type_name}",
            source_loc=getattr(type_obj, 'source_loc', None),
        )

    if ast_types and isinstance(type_obj, ast_types.ArrayType):
        elem_size = get_unified_type_size(type_obj.element_type, symbol_table)
        if hasattr(type_obj, 'length') and type_obj.length:
            if hasattr(type_obj.length, 'value'):
                return elem_size * type_obj.length.value
        raise HIRError(
            "Array type must have constant length for size calculation",
            source_loc=getattr(type_obj, 'source_loc', None),
        )

    if ast_types and isinstance(type_obj, ast_types.ArrayFillExpr):
        elem_size = get_unified_type_size(type_obj.value, symbol_table)
        if hasattr(type_obj, 'count') and hasattr(type_obj.count, 'value'):
            return elem_size * type_obj.count.value
        raise HIRError(
            "Array fill expression must have constant count",
            source_loc=getattr(type_obj, 'source_loc', None),
        )

    # AST struct-like object with fields
    if hasattr(type_obj, 'fields'):
        return _get_ast_struct_size(type_obj, symbol_table)

    # Default to 1 byte if unknown
    return 1


def _get_ast_struct_size(struct_obj: Any, symbol_table=None) -> int:
    """Calculate size of an AST struct by summing field sizes."""
    total = 0
    for field in struct_obj.fields:
        total += get_unified_type_size(field.field_type, symbol_table)
    return total


def get_basic_type_size(type_name: str) -> int:
    """Get size of a basic type by name."""
    return _TYPE_SIZES.get(type_name, 1)
