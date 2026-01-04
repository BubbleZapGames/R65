"""
Abstract Syntax Tree (AST) node definitions for R65.

Each node represents a syntactic construct in the language.
"""
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum


# ============================================================================
# Base Classes
# ============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    pass


@dataclass
class Expression(ASTNode):
    """Base class for expression nodes."""
    pass


@dataclass
class Statement(ASTNode):
    """Base class for statement nodes."""
    pass


@dataclass
class Declaration(ASTNode):
    """Base class for declaration nodes."""
    pass


@dataclass
class Type(ASTNode):
    """Base class for type nodes."""
    pass


# ============================================================================
# Program Structure
# ============================================================================

@dataclass
class Program(ASTNode):
    """Top-level program node containing all declarations."""
    items: List[Declaration]


# ============================================================================
# Types
# ============================================================================

@dataclass
class BasicType(Type):
    """Basic type like u8, u16, bool."""
    name: str  # "u8", "u16", "i8", "i16", "bool", "near"


@dataclass
class ArrayType(Type):
    """Array type: [T; N]"""
    element_type: Type
    size: Expression


@dataclass
class PointerType(Type):
    """Pointer type: near<T> or far<T>"""
    is_far: bool
    pointee_type: Type


@dataclass
class FunctionType(Type):
    """Function type: fn(params) -> return_type"""
    is_far: bool
    param_types: List[Type]
    return_type: Optional[Type]  # None for no return, Type for return type


@dataclass
class NeverType(Type):
    """Never type: !"""
    pass


# ============================================================================
# Attributes
# ============================================================================

@dataclass
class AttributeArg(ASTNode):
    """Single argument in an attribute."""
    name: Optional[str]  # None if positional
    value: Union[Expression, str]  # Expression or identifier


@dataclass
class Attribute(ASTNode):
    """Attribute like #[mode(m8, x8)]"""
    name: str
    args: List[AttributeArg]


# ============================================================================
# Declarations
# ============================================================================

@dataclass
class Parameter(ASTNode):
    """Function parameter."""
    name: str
    binding: Optional[Union[str, 'Register']]  # Register name or variable name for aliasing
    param_type: Type


@dataclass
class FunctionDecl(Declaration):
    """Function declaration."""
    attributes: List[Attribute]
    is_far: bool
    name: str
    params: List[Parameter]
    return_type: Optional[Union[Type, NeverType]]
    body: 'Block'


@dataclass
class StaticDecl(Declaration):
    """Static variable declaration."""
    attributes: List[Attribute]
    is_mut: bool
    name: str
    var_type: Type
    initializer: Optional[Expression]


@dataclass
class ConstDecl(Declaration):
    """Const declaration."""
    name: str
    const_type: Type
    value: Expression


@dataclass
class StructField(ASTNode):
    """Field in a struct."""
    name: str
    field_type: Type


@dataclass
class StructDecl(Declaration):
    """Struct declaration."""
    name: str
    fields: List[StructField]


@dataclass
class EnumVariant(ASTNode):
    """Variant in an enum."""
    name: str
    value: Optional[Expression]  # None for auto-increment


@dataclass
class EnumDecl(Declaration):
    """Enum declaration."""
    name: str
    variants: List[EnumVariant]


@dataclass
class TypeAlias(Declaration):
    """Type alias declaration."""
    name: str
    aliased_type: Type


@dataclass
class IncludeStmt(Declaration):
    """Include statement."""
    path: str


@dataclass
class StackDirective(Declaration):
    """Stack region directive: #[stack(lower, upper)]"""
    lower: int
    upper: int


# ============================================================================
# Statements
# ============================================================================

@dataclass
class Block(Statement):
    """Block of statements."""
    statements: List[Statement]


@dataclass
class LetStmt(Statement):
    """Let binding statement."""
    is_mut: bool
    name: str
    binding: Optional[Union[str, 'Register']]  # Register or variable for aliasing
    var_type: Optional[Type]
    initializer: Expression


@dataclass
class ExprStmt(Statement):
    """Expression statement."""
    expr: Expression


@dataclass
class ReturnStmt(Statement):
    """Return statement."""
    values: List[Expression]  # Multiple values for multiple returns


@dataclass
class BreakStmt(Statement):
    """Break statement."""
    pass


@dataclass
class ContinueStmt(Statement):
    """Continue statement."""
    pass


@dataclass
class IfStmt(Statement):
    """If statement."""
    condition: Expression
    then_block: Block
    else_block: Optional[Union[Block, 'IfStmt']]  # Can chain with else if


@dataclass
class LoopStmt(Statement):
    """Loop statement."""
    body: Block


@dataclass
class WhileStmt(Statement):
    """While statement."""
    condition: Expression
    body: Block


@dataclass
class AsmStmt(Statement):
    """Inline assembly statement."""
    instructions: List[str]


# ============================================================================
# Expressions
# ============================================================================

@dataclass
class IntegerLiteral(Expression):
    """Integer literal."""
    value: int


@dataclass
class BooleanLiteral(Expression):
    """Boolean literal."""
    value: bool


@dataclass
class Identifier(Expression):
    """Identifier reference."""
    name: str


@dataclass
class EnumVariantExpr(Expression):
    """Enum variant expression (e.g., Direction::North)."""
    enum_name: str
    variant_name: str


@dataclass
class Register(Expression):
    """Hardware register reference."""
    name: str  # "A", "X", "Y", "B", "Status", "D", "DBR", "PBR", "S"


@dataclass
class IncludeBytesExpr(Expression):
    """Include binary data from file (e.g., include_bytes!("data.bin"))."""
    path: str


@dataclass
class BinaryOp(Expression):
    """Binary operation."""
    op: str  # "+", "-", "*", "/", etc.
    left: Expression
    right: Expression


@dataclass
class UnaryOp(Expression):
    """Unary operation."""
    op: str  # "!", "~", "-"
    operand: Expression


@dataclass
class TypeCast(Expression):
    """Type cast expression."""
    expr: Expression
    target_type: Type


@dataclass
class FunctionCall(Expression):
    """Function call."""
    func: Expression
    args: List[Expression]


@dataclass
class ArrayIndex(Expression):
    """Array indexing."""
    array: Expression
    index: Expression


@dataclass
class FieldAccess(Expression):
    """Field access."""
    base: Expression
    field: str


@dataclass
class Dereference(Expression):
    """Pointer dereference."""
    pointer: Expression


@dataclass
class AddressOf(Expression):
    """Address-of operator (&variable)."""
    operand: Expression


@dataclass
class Assignment(Expression):
    """Assignment expression."""
    target: Union[Identifier, ArrayIndex, FieldAccess]
    value: Expression


@dataclass
class CompoundAssignment(Expression):
    """Compound assignment expression (e.g., +=, -=, &=)."""
    target: Union[Identifier, Register, ArrayIndex, FieldAccess]
    operator: str  # "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"
    value: Expression


# ============================================================================
# Pattern Matching
# ============================================================================

@dataclass
class Pattern(ASTNode):
    """Base class for pattern nodes."""
    pass


@dataclass
class LiteralPattern(Pattern):
    """Literal pattern (integer or boolean)."""
    value: Union[int, bool]


@dataclass
class EnumPattern(Pattern):
    """Enum variant pattern (e.g., State::Idle)."""
    enum_name: str
    variant_name: str


@dataclass
class WildcardPattern(Pattern):
    """Wildcard pattern (_)."""
    pass


@dataclass
class IdentifierPattern(Pattern):
    """Identifier pattern (binds value to variable)."""
    name: str


@dataclass
class OrPattern(Pattern):
    """Or pattern (pattern1 | pattern2 | ...)."""
    patterns: List[Pattern]


@dataclass
class MatchArm(ASTNode):
    """Single arm of a match expression."""
    pattern: Pattern
    body: Expression


@dataclass
class MatchExpression(Expression):
    """Match expression."""
    scrutinee: Expression  # Expression being matched
    arms: List[MatchArm]


# ============================================================================
# Utility Functions
# ============================================================================

def ast_to_string(node: ASTNode, indent: int = 0) -> str:
    """
    Convert an AST node to a readable string representation.

    Args:
        node: AST node to convert
        indent: Current indentation level

    Returns:
        String representation of the AST
    """
    prefix = "  " * indent

    if isinstance(node, Program):
        result = f"{prefix}Program:\n"
        for item in node.items:
            result += ast_to_string(item, indent + 1)
        return result

    elif isinstance(node, FunctionDecl):
        far = "far " if node.is_far else ""
        result = f"{prefix}Function {far}{node.name}:\n"
        if node.attributes:
            result += f"{prefix}  Attributes: {[a.name for a in node.attributes]}\n"
        if node.params:
            result += f"{prefix}  Params:\n"
            for param in node.params:
                result += f"{prefix}    {param.name}: {param.param_type}\n"
        if node.return_type:
            result += f"{prefix}  Returns: {node.return_type}\n"
        result += f"{prefix}  Body:\n"
        result += ast_to_string(node.body, indent + 2)
        return result

    elif isinstance(node, Block):
        result = f"{prefix}Block:\n"
        for stmt in node.statements:
            result += ast_to_string(stmt, indent + 1)
        return result

    elif isinstance(node, LetStmt):
        mut = "mut " if node.is_mut else ""
        result = f"{prefix}Let {mut}{node.name}"
        if node.binding:
            result += f" @ {node.binding}"
        if node.var_type:
            result += f": {node.var_type}"
        result += f" = {node.initializer}\n"
        return result

    elif isinstance(node, ExprStmt):
        return f"{prefix}ExprStmt: {node.expr}\n"

    elif isinstance(node, ReturnStmt):
        return f"{prefix}Return: {node.values}\n"

    else:
        return f"{prefix}{node.__class__.__name__}: {node}\n"
