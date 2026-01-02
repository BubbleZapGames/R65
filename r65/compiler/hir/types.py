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
class PointerTypeInfo(TypeInfo):
    """Pointer type: near<T> or far<T>."""
    is_far: bool
    pointee_type: TypeInfo

    def __str__(self):
        kind = "far" if self.is_far else "near"
        return f"{kind}<{self.pointee_type}>"


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
class NeverTypeInfo(TypeInfo):
    """Never type: ! (function never returns)."""

    def __str__(self):
        return "!"


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
            return BasicTypeInfo(name=ast_type.name)

        elif isinstance(ast_type, ast.ArrayType):
            elem_type = self.resolve_type(ast_type.element_type)

            # Evaluate size expression to constant
            if self.const_evaluator is None:
                raise HIRError("Const evaluator required for array size evaluation")

            size = self.const_evaluator.eval(ast_type.size)
            if not isinstance(size, int) or size <= 0:
                raise HIRError(f"Array size must be a positive integer, got {size}")

            return ArrayTypeInfo(element_type=elem_type, size=size)

        elif isinstance(ast_type, ast.PointerType):
            pointee = self.resolve_type(ast_type.pointee_type)
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

        else:
            raise HIRError(f"Unknown type node: {type(ast_type).__name__}")

    def resolve_named_type(self, name: str) -> TypeInfo:
        """Resolve a named type (struct, enum, or type alias) by name."""
        symbol = self.symbol_table.lookup(name)

        if symbol is None:
            raise HIRError(f"Undefined type: {name}")

        if symbol.kind.value == "struct":
            return StructTypeInfo(name=name, definition=symbol.definition)
        elif symbol.kind.value == "enum":
            return EnumTypeInfo(name=name, definition=symbol.definition)
        elif symbol.kind.value == "type_alias":
            # Type aliases are resolved to their underlying type
            return symbol.type_info
        else:
            raise HIRError(f"'{name}' is not a type")
