"""
Unified type size utilities for both AST and HIR types.

This module provides a single source of truth for type size calculations,
eliminating duplication across const evaluator, HIR builder, and type checker.
"""

from typing import Union, Any
from r65.compiler.hir.errors import HIRError


# =============================================================================
# Type Size Constants
# =============================================================================

# Size in bytes for basic types
TYPE_SIZES = {
    'u8': 1, 'i8': 1, 'bool': 1,
    'u16': 2, 'i16': 2,
    'u24': 3,  # For far pointers
}

# Import AST types for isinstance checks
try:
    from r65.compiler.frontend import ast as ast_types
except ImportError:
    ast_types = None

# Import HIR types for isinstance checks  
try:
    from r65.compiler.hir.types import BasicTypeInfo, ArrayTypeInfo, StructTypeInfo
    from r65.compiler.hir.nodes import HIRStructDecl
except ImportError:
    BasicTypeInfo = ArrayTypeInfo = StructTypeInfo = HIRStructDecl = None


def get_unified_type_size(type_obj: Any, symbol_table=None) -> int:
    """
    Get size of a type in bytes, supporting both AST and HIR type objects.
    
    This is the unified interface that should be used throughout the compiler
    to eliminate duplication of type size logic.
    
    Args:
        type_obj: AST type node, HIR type object, or type name string
        symbol_table: Symbol table for resolving struct definitions (for AST types)
        
    Returns:
        Size in bytes
        
    Raises:
        HIRError: If type cannot be resolved or is unsupported
    """
    
    # Handle basic type names (strings)
    if isinstance(type_obj, str):
        if type_obj in TYPE_SIZES:
            return TYPE_SIZES[type_obj]
        raise HIRError(f"Unknown basic type: {type_obj}")
    
    # Handle HIR BasicTypeInfo (or similar with name attribute)
    if hasattr(type_obj, 'name') and not hasattr(type_obj, 'element_type'):
        type_name = type_obj.name
        if type_name in TYPE_SIZES:
            return TYPE_SIZES[type_name]
        # Could be a struct name - delegate to HIR type handler
        if hasattr(type_obj, 'fields'):  # StructTypeInfo
            return _get_struct_size(type_obj)
        # Don't raise error - might be a struct name to look up
        # Let the struct lookup logic below handle it
    
    # Handle AST BasicType
    if ast_types and isinstance(type_obj, ast_types.BasicType):
        type_name = type_obj.name
        if type_name in TYPE_SIZES:
            return TYPE_SIZES[type_name]
        raise HIRError(f"Unknown basic type: {type_name}")
    
    # Handle AST Identifier (type names)
    if ast_types and isinstance(type_obj, ast_types.Identifier):
        type_name = type_obj.name
        if type_name in TYPE_SIZES:
            return TYPE_SIZES[type_name]
        
        # Look up struct/enum in symbol table
        if symbol_table:
            symbol = symbol_table.lookup(type_name)
            if symbol and hasattr(symbol, 'kind'):
                if hasattr(symbol, 'definition') and symbol.definition:
                    if hasattr(symbol.definition, 'fields'):  # Struct
                        return _get_struct_size(symbol.definition)
                    # Add other types as needed
                elif symbol.kind.value == "enum":
                    return 1
        
        raise HIRError(f"Cannot determine size of type: {type_name}")
    
    # Handle Array types (both HIR and AST)
    if hasattr(type_obj, 'element_type') and hasattr(type_obj, 'size'):
        # HIR ArrayTypeInfo
        elem_size = get_unified_type_size(type_obj.element_type, symbol_table)
        return elem_size * type_obj.size
    
    if ast_types and isinstance(type_obj, ast_types.ArrayType):
        # AST ArrayType
        elem_size = get_unified_type_size(type_obj.element_type, symbol_table)
        if hasattr(type_obj, 'length') and type_obj.length:
            if hasattr(type_obj.length, 'value'):  # IntegerLiteral
                return elem_size * type_obj.length.value
        raise HIRError("Array type must have constant length for size calculation")
    
    # Handle AST ArrayFillExpr ([u8; 10] syntax)
    if ast_types and isinstance(type_obj, ast_types.ArrayFillExpr):
        elem_size = get_unified_type_size(type_obj.value, symbol_table)
        if hasattr(type_obj, 'count') and hasattr(type_obj.count, 'value'):
            return elem_size * type_obj.count.value
        raise HIRError("Array fill expression must have constant count")
    
    # Handle Struct types (HIR StructTypeInfo or similar)
    if hasattr(type_obj, 'fields'):
        return _get_struct_size(type_obj)
    
    # Handle HIR struct definition nodes
    if HIRStructDecl and isinstance(type_obj, HIRStructDecl):
        return _get_struct_size(type_obj)
    
    # Handle pointer types
    if hasattr(type_obj, 'pointee_type'):
        # Near pointer = 2 bytes, far pointer = 3 bytes
        if hasattr(type_obj, 'is_far') and type_obj.is_far:
            return 3
        return 2
    
    # Check for pointer type names
    type_name_str = str(type_obj)
    if type_name_str.startswith('near<'):
        return 2  # 16-bit pointer
    elif type_name_str.startswith('far<'):
        return 3  # 24-bit pointer
    
    # Handle other types (delegate to codegen if available, otherwise use defaults)
    try:
        from r65.compiler.codegen.type_utils import get_type_size
        return get_type_size(type_obj)
    except ImportError:
        # Fallback implementation for basic cases
        if hasattr(type_obj, 'is_far'):
            return 3 if type_obj.is_far else 2
        return 1  # Default to 1 byte if unknown


def _get_struct_size(struct_obj: Any) -> int:
    """
    Calculate size of a struct type by summing field sizes.
    
    Supports both HIR struct definitions and AST struct declarations.
    """
    if not hasattr(struct_obj, 'fields'):
        raise HIRError("Struct type has no fields")
    
    total_size = 0
    for field in struct_obj.fields:
        field_size = get_unified_type_size(field.field_type)
        total_size += field_size
    
    return total_size


def get_basic_type_size(type_name: str) -> int:
    """
    Get size of a basic type by name.
    
    Quick lookup for common basic types.
    """
    return TYPE_SIZES.get(type_name, 1)  # Default to 1 byte if unknown