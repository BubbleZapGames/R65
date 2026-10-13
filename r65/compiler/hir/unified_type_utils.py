# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Unified type size utilities for both AST and HIR types.

For HIR types (TypeInfo subclasses), use the `size_bytes` property directly.
This module provides `get_unified_type_size()` which additionally handles
AST type nodes — needed during const evaluation before HIR lowering.
"""

from typing import Any
from r65.compiler.hir.symbol_table import SymbolKind
from r65.compiler.hir.errors import HIRError
from r65.compiler.hir.types import TypeInfo, _TYPE_SIZES, discriminant_type


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
        raise HIRError("Cannot determine size of an unresolved type", source_loc=None)

    # HIR TypeInfo — use the size_bytes property directly
    if isinstance(type_obj, TypeInfo):
        return type_obj.size_bytes

    # Handle basic type names (strings)
    if isinstance(type_obj, str):
        if type_obj in _TYPE_SIZES:
            return _TYPE_SIZES[type_obj]
        raise HIRError(f"Unknown basic type: {type_obj}", source_loc=None)

    # --- AST type nodes (used during const evaluation before HIR lowering) ---

    if ast_types and isinstance(type_obj, (ast_types.BasicType, ast_types.Identifier)):
        return _named_type_size(type_obj, symbol_table)

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

    raise HIRError(
        f"Cannot determine size of {type(type_obj).__name__}",
        source_loc=getattr(type_obj, 'source_loc', None),
    )


def _named_type_size(type_obj, symbol_table) -> int:
    """Size of a type named in the AST: a basic type, a struct or an enum.

    A field's named type parses as a `BasicType` and a `size_of` argument as an
    `Identifier`, so both come here. This runs from const evaluation in pass 1,
    where a struct's or enum's symbol still holds its AST declaration.
    """
    type_name = type_obj.name
    if type_name in _TYPE_SIZES:
        return _TYPE_SIZES[type_name]
    symbol = symbol_table.lookup(type_name) if symbol_table else None
    definition = getattr(symbol, 'definition', None)
    if symbol is not None and symbol.kind is SymbolKind.ENUM:
        return _enum_size(type_name, definition, symbol_table)
    if definition is not None and hasattr(definition, 'fields'):
        return _get_ast_struct_size(definition, symbol_table)
    raise HIRError(
        f"Cannot determine size of type: {type_name}",
        source_loc=getattr(type_obj, 'source_loc', None),
    )


def _enum_size(name, definition, symbol_table) -> int:
    """Discriminant width of enum *name*: 1 byte, or 2 once a variant exceeds 255.

    The HIR declaration carries it; before that, pass 1 has already declared
    every `Enum::Variant` symbol with its value.
    """
    underlying = getattr(definition, 'underlying_type', None)
    if underlying is not None:
        return underlying.size_bytes
    values = (getattr(symbol_table.lookup(f"{name}::{v.name}"), 'const_value', 0)
              for v in definition.variants)
    return discriminant_type(max(values, default=0)).size_bytes


def layout_fields(field_sizes, is_union: bool):
    """Compute packed field offsets and total size for a struct or union.

    Structs lay fields out end to end (packed, no padding); unions overlay every
    field at offset 0 and take the size of their largest field. This is the single
    definition of aggregate layout — every site that needs offsets or a total size
    must go through here, because a site that keeps summing sizes for a union
    produces silent memory corruption rather than a compile error.

    Args:
        field_sizes: iterable of field sizes in bytes, in declaration order
        is_union: True to overlay fields, False to pack them

    Returns:
        (offsets, total_size) - offsets is a list parallel to field_sizes
    """
    offsets = []
    total = 0
    for size in field_sizes:
        if is_union:
            offsets.append(0)
            total = max(total, size)
        else:
            offsets.append(total)
            total += size
    return offsets, total


def _get_ast_struct_size(struct_obj: Any, symbol_table=None) -> int:
    """Calculate size of an AST struct or union from its field sizes."""
    sizes = [get_unified_type_size(f.field_type, symbol_table) for f in struct_obj.fields]
    _, total = layout_fields(sizes, getattr(struct_obj, 'is_union', False))
    return total
