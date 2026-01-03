"""
HIR (High-level IR) node definitions for the R65 compiler.

These nodes represent the program after name resolution, attribute processing,
and desugaring, but before full type checking.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Any
from r65.compiler.hir.errors import *


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class HIRNode:
    """Base class for all HIR nodes."""
    source_loc: Optional[SourceLocation] = None


@dataclass
class HIRExpression(HIRNode):
    """Base class for HIR expressions."""
    # Type will be filled in by type checker
    expr_type: Optional[Any] = None  # Will be TypeInfo after type checking


@dataclass
class HIRStatement(HIRNode):
    """Base class for HIR statements."""
    pass


@dataclass
class HIRDeclaration(HIRNode):
    """Base class for HIR declarations."""
    pass


# =============================================================================
# Program
# =============================================================================

@dataclass
class HIRProgram(HIRNode):
    """Top-level program."""
    declarations: List[HIRDeclaration] = field(default_factory=list)
    symbol_table: Any = None  # Will be SymbolTable


# =============================================================================
# Bindings (for parameters and let statements)
# =============================================================================

@dataclass
class ParameterBinding:
    """Base class for parameter binding information."""
    pass


@dataclass
class RegisterBinding(ParameterBinding):
    """Parameter bound to hardware register (e.g., param @ A)."""
    register_name: str  # "A", "X", "Y"


@dataclass
class VariableBinding(ParameterBinding):
    """Parameter bound to static variable (e.g., param @ TEMP)."""
    variable_name: str
    variable_symbol: Any  # Will be Symbol (resolved reference)


@dataclass
class LetBinding:
    """Base class for let statement binding information."""
    pass


@dataclass
class RegisterLetBinding(LetBinding):
    """Let bound to register (e.g., let x @ A = expr)."""
    register_name: str


@dataclass
class VariableLetBinding(LetBinding):
    """Let bound to variable (e.g., let x @ VAR = expr)."""
    variable_name: str
    variable_symbol: Any  # Will be Symbol


# =============================================================================
# Declarations
# =============================================================================

@dataclass
class HIRParameter(HIRNode):
    """Function parameter."""
    name: str = ""
    param_type: Any = None  # Will be TypeInfo
    binding: Optional[ParameterBinding] = None

    # Symbol reference (resolved during HIR construction)
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRFunctionDecl(HIRDeclaration):
    """Function declaration."""
    name: str = ""
    is_far: bool = False
    parameters: List[HIRParameter] = field(default_factory=list)
    return_type: Optional[Any] = None  # Will be TypeInfo or None
    body: Optional['HIRBlock'] = None

    # Processed attributes
    mode_attr: Optional[Any] = None  # Will be ModeAttribute
    preserves_attr: Optional[Any] = None  # Will be PreservesAttribute
    bank_attr: Optional[Any] = None  # Will be BankAttribute
    interrupt_attr: Optional[Any] = None  # Will be InterruptAttribute
    is_entry: bool = False

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRStaticDecl(HIRDeclaration):
    """Static variable declaration."""
    name: str = ""
    is_mutable: bool = False
    var_type: Any = None  # Will be TypeInfo
    initializer: Optional[HIRExpression] = None

    # Processed attributes
    storage_attr: Optional[Any] = None  # Will be StorageAttribute

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRConstDecl(HIRDeclaration):
    """Const declaration."""
    name: str = ""
    const_type: Any = None  # Will be TypeInfo
    value: Optional[HIRExpression] = None
    evaluated_value: Optional[Any] = None  # Const-evaluated value

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRStructField(HIRNode):
    """Struct field."""
    name: str = ""
    field_type: Any = None  # Will be TypeInfo
    offset: Optional[int] = None  # Computed during HIR construction


@dataclass
class HIRStructDecl(HIRDeclaration):
    """Struct declaration."""
    name: str = ""
    fields: List[HIRStructField] = field(default_factory=list)

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIREnumVariant(HIRNode):
    """Enum variant."""
    name: str = ""
    value: int = 0  # Explicit or auto-incremented (resolved during HIR)

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIREnumDecl(HIRDeclaration):
    """Enum declaration."""
    name: str = ""
    variants: List[HIREnumVariant] = field(default_factory=list)
    underlying_type: Optional[Any] = None  # Inferred u8/u16 (will be TypeInfo)

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRTypeAlias(HIRDeclaration):
    """Type alias declaration."""
    name: str = ""
    aliased_type: Any = None  # Will be TypeInfo

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


# =============================================================================
# Statements
# =============================================================================

@dataclass
class HIRBlock(HIRStatement):
    """Block of statements."""
    statements: List[HIRStatement] = field(default_factory=list)
    scope_id: Optional[int] = None  # Associated scope ID


@dataclass
class HIRLetStmt(HIRStatement):
    """Let binding (variable declaration)."""
    name: str = ""
    is_mutable: bool = False
    var_type: Optional[Any] = None  # Will be TypeInfo (may be inferred)
    initializer: Optional[HIRExpression] = None
    binding: Optional[LetBinding] = None

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRExprStmt(HIRStatement):
    """Expression statement."""
    expr: Optional[HIRExpression] = None


@dataclass
class HIRReturnStmt(HIRStatement):
    """Return statement."""
    values: List[HIRExpression] = field(default_factory=list)  # Empty for implicit A return


@dataclass
class HIRBreakStmt(HIRStatement):
    """Break statement."""
    pass


@dataclass
class HIRContinueStmt(HIRStatement):
    """Continue statement."""
    pass


@dataclass
class HIRIfStmt(HIRStatement):
    """If statement."""
    condition: Optional[HIRExpression] = None
    then_block: Optional[HIRBlock] = None
    else_block: Optional[Union[HIRBlock, 'HIRIfStmt']] = None  # Can be else-if


@dataclass
class HIRWhileStmt(HIRStatement):
    """While statement (loop desugared to this)."""
    condition: Optional[HIRExpression] = None
    body: Optional[HIRBlock] = None
    is_infinite: bool = False  # True for `loop` (while true)


@dataclass
class HIRAsmStmt(HIRStatement):
    """Inline assembly statement."""
    instructions: List[str] = field(default_factory=list)


# =============================================================================
# Expressions
# =============================================================================

@dataclass
class HIRIntegerLiteral(HIRExpression):
    """Integer literal."""
    value: int = 0


@dataclass
class HIRBooleanLiteral(HIRExpression):
    """Boolean literal."""
    value: bool = False


@dataclass
class HIRIdentifier(HIRExpression):
    """Identifier reference (resolved to symbol)."""
    name: str = ""
    symbol: Any = None  # Resolved Symbol


@dataclass
class HIRFunctionAddress(HIRExpression):
    """
    Function address expression.

    Represents taking the address of a function to store in a function pointer.
    Used when assigning a function to a function pointer variable.

    Example: let handler: fn(u8) -> u8 = some_function;
    The 'some_function' becomes HIRFunctionAddress.
    """
    function_name: str = ""
    symbol: Any = None  # Resolved function Symbol
    # expr_type will be FunctionTypeInfo


@dataclass
class HIRRegister(HIRExpression):
    """Hardware register reference."""
    name: str = ""  # "A", "X", "Y", "STATUS", "D", "DBR", "PBR", "S"
    symbol: Any = None  # Points to register symbol


@dataclass
class HIRIncludeBytesExpr(HIRExpression):
    """Include binary data from file."""
    path: str = ""  # Path to binary file


@dataclass
class HIRBinaryOp(HIRExpression):
    """Binary operation."""
    op: str = ""  # "+", "-", "*", "/", etc.
    left: Optional[HIRExpression] = None
    right: Optional[HIRExpression] = None


@dataclass
class HIRUnaryOp(HIRExpression):
    """Unary operation."""
    op: str = ""  # "!", "~", "-"
    operand: Optional[HIRExpression] = None


@dataclass
class HIRTypeCast(HIRExpression):
    """Type cast (explicit conversion)."""
    expr: Optional[HIRExpression] = None
    target_type: Any = None  # Will be TypeInfo


@dataclass
class HIRFunctionCall(HIRExpression):
    """Function call."""
    func: Optional[HIRExpression] = None  # Usually HIRIdentifier (resolved)
    args: List[HIRExpression] = field(default_factory=list)


@dataclass
class HIRArrayIndex(HIRExpression):
    """Array indexing."""
    array: Optional[HIRExpression] = None
    index: Optional[HIRExpression] = None
    # Store original AST for const evaluation
    original_ast: Optional[Any] = None  # ast.ArrayIndex


@dataclass
class HIRFieldAccess(HIRExpression):
    """Struct field access."""
    base: Optional[HIRExpression] = None
    field_name: str = ""
    field_index: Optional[int] = None  # Resolved during HIR construction
    field_offset: Optional[int] = None  # Computed during HIR construction


@dataclass
class HIRDereference(HIRExpression):
    """Pointer dereference."""
    pointer: Optional[HIRExpression] = None


@dataclass
class HIRAssignment(HIRExpression):
    """Assignment expression."""
    target: Optional[HIRExpression] = None  # Must be lvalue
    value: Optional[HIRExpression] = None
