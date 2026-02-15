"""
Type information structures for R65 HIR.

These types are attached to HIR nodes but not fully checked until the type checking phase.
"""

from dataclasses import dataclass
from typing import Optional, List, Any
from r65.compiler.frontend import ast
from r65.compiler.hir.errors import *


# =============================================================================
# Type Information
# =============================================================================

@dataclass
class TypeInfo:
    """Base class for type information."""
    pass


@dataclass
class BasicTypeInfo(TypeInfo):
    """Basic type: u8, u16, i8, i16, bool."""
    name: str  # "u8", "u16", "i8", "i16", "bool"

    def __str__(self):
        return self.name


@dataclass
class ArrayTypeInfo(TypeInfo):
    """Array type: [T; N]."""
    element_type: TypeInfo
    size: int  # Must be compile-time constant

    def __str__(self):
        return f"[{self.element_type}; {self.size}]"


@dataclass
class SliceTypeInfo(TypeInfo):
    """Unsized array type for pointers: [T]."""
    element_type: TypeInfo

    def __str__(self):
        return f"[{self.element_type}]"


@dataclass
class PointerTypeInfo(TypeInfo):
    """Pointer type: *T (near) or far *T."""
    is_far: bool
    pointee_type: TypeInfo

    def __str__(self):
        dyn_str = "dyn " if isinstance(self.pointee_type, TraitTypeInfo) else ""
        if self.is_far:
            return f"far *{dyn_str}{self.pointee_type}"
        return f"*{dyn_str}{self.pointee_type}"


@dataclass
class FunctionTypeInfo(TypeInfo):
    """Function pointer type."""
    is_far: bool
    param_types: List[TypeInfo]
    return_type: Optional[TypeInfo]

    def __str__(self):
        far_str = "far " if self.is_far else ""
        params = ", ".join(str(p) for p in self.param_types)
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{far_str}fn({params}){ret}"


@dataclass
class StructTypeInfo(TypeInfo):
    """Struct type reference."""
    name: str
    definition: Optional[Any] = None  # Will be HIRStructDecl (resolved during HIR)
    symbol: Optional[Any] = None  # Symbol table entry (for accessing current definition)

    def __str__(self):
        return self.name


@dataclass
class EnumTypeInfo(TypeInfo):
    """Enum type reference."""
    name: str
    definition: Optional[Any] = None  # Will be HIREnumDecl (resolved during HIR)

    def __str__(self):
        return self.name


@dataclass
class TraitTypeInfo(TypeInfo):
    """Trait type reference (used as pointee type in trait pointers like *Drawable)."""
    name: str
    definition: Optional[Any] = None  # Will be HIRTraitDecl (resolved during HIR)

    def __str__(self):
        return self.name


@dataclass
class NeverTypeInfo(TypeInfo):
    """Never type: ! (function never returns)."""

    def __str__(self):
        return "!"


@dataclass
class TupleTypeInfo(TypeInfo):
    """Tuple type: (T1, T2, ...) for multiple return values."""
    element_types: List[TypeInfo]

    def __str__(self):
        types_str = ", ".join(str(t) for t in self.element_types)
        return f"({types_str})"


@dataclass
class RegisterTypeInfo(TypeInfo):
    """Type of a hardware register (mode-dependent)."""
    register_name: str
    # Actual type depends on processor mode (resolved in type checker)

    def __str__(self):
        return f"Register<{self.register_name}>"


# =============================================================================
# Type Resolver
# =============================================================================

class TypeResolver:
    """Resolves AST types to HIR TypeInfo."""

    def __init__(self, symbol_table: Any, const_evaluator: Optional[Any] = None):
        self.symbol_table = symbol_table
        self.const_evaluator = const_evaluator  # Will be ConstEvaluator

    def resolve_type(self, ast_type: ast.Type) -> TypeInfo:
        """Convert AST type to HIR TypeInfo."""
        if isinstance(ast_type, ast.BasicType):
            # Check if this is a built-in type or a user-defined type
            built_in_types = {'u8', 'i8', 'u16', 'i16', 'bool', 'void'}
            if ast_type.name in built_in_types:
                return BasicTypeInfo(name=ast_type.name)
            else:
                # User-defined type (struct, enum, or type alias)
                return self.resolve_named_type(ast_type.name)

        elif isinstance(ast_type, ast.ArrayType):
            elem_type = self.resolve_type(ast_type.element_type)

            # Evaluate size expression to constant
            if self.const_evaluator is None:
                raise HIRError("Const evaluator required for array size evaluation", source_loc=getattr(ast_type, 'source_loc', None))

            size = self.const_evaluator.eval(ast_type.size)
            if not isinstance(size, int) or size <= 0:
                raise HIRError(f"Array size must be a positive integer, got {size}", source_loc=getattr(ast_type, 'source_loc', None))

            return ArrayTypeInfo(element_type=elem_type, size=size)

        elif isinstance(ast_type, ast.SliceType):
            elem_type = self.resolve_type(ast_type.element_type)
            return SliceTypeInfo(element_type=elem_type)

        elif isinstance(ast_type, ast.PointerType):
            pointee = self.resolve_type(ast_type.pointee_type)

            # Pointers cannot point to sized arrays - must use unsized array [T]
            if isinstance(pointee, ArrayTypeInfo):
                raise HIRError(
                    f"pointer cannot point to sized array type [{pointee.element_type}; {pointee.size}]",
                    source_loc=getattr(ast_type, 'source_loc', None),
                    hint=f"use unsized array type [{pointee.element_type}] instead"
                )

            # Validate dyn keyword usage
            if isinstance(pointee, TraitTypeInfo) and not ast_type.is_dyn:
                raise HIRError(
                    f"trait pointer requires 'dyn' keyword: use '*dyn {pointee.name}' instead of '*{pointee.name}'",
                    source_loc=getattr(ast_type, 'source_loc', None)
                )
            if ast_type.is_dyn and not isinstance(pointee, TraitTypeInfo):
                raise HIRError(
                    f"'dyn' can only be used with trait types, but '{pointee.name}' is not a trait",
                    source_loc=getattr(ast_type, 'source_loc', None)
                )

            return PointerTypeInfo(is_far=ast_type.is_far, pointee_type=pointee)

        elif isinstance(ast_type, ast.FunctionType):
            params = [self.resolve_type(p) for p in ast_type.param_types]
            ret_type = None if ast_type.return_type is None else self.resolve_type(ast_type.return_type)
            return FunctionTypeInfo(
                is_far=ast_type.is_far,
                param_types=params,
                return_type=ret_type
            )

        elif isinstance(ast_type, ast.NeverType):
            return NeverTypeInfo()

        elif isinstance(ast_type, ast.TupleType):
            element_types = [self.resolve_type(t) for t in ast_type.element_types]
            return TupleTypeInfo(element_types=element_types)

        else:
            raise HIRError(f"Unknown type node: {type(ast_type).__name__}", source_loc=getattr(ast_type, 'source_loc', None))

    def resolve_named_type(self, name: str) -> TypeInfo:
        """Resolve a named type (struct, enum, trait, or type alias) by name."""
        symbol = self.symbol_table.lookup(name)

        if symbol is None:
            raise HIRError(f"Undefined type: {name}", source_loc=None)

        if symbol.kind.value == "struct":
            return StructTypeInfo(name=name, definition=symbol.definition, symbol=symbol)
        elif symbol.kind.value == "enum":
            return EnumTypeInfo(name=name, definition=symbol.definition)
        elif symbol.kind.value == "trait":
            return TraitTypeInfo(name=name, definition=symbol.definition)
        elif symbol.kind.value == "type_alias":
            # Type aliases are resolved to their underlying type
            return symbol.type_info
        else:
            raise HIRError(f"'{name}' is not a type", source_loc=None)
