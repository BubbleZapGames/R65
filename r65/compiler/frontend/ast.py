"""
Abstract Syntax Tree (AST) node definitions for R65.

Each node represents a syntactic construct in the language.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from r65.compiler.hir.errors import SourceLocation


# ============================================================================
# Base Classes
# ============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    # Source location (populated by parser from Lark meta)
    # Using kw_only=True to allow subclasses to have non-default fields
    source_loc: Optional['SourceLocation'] = field(default=None, repr=False, kw_only=True)


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
class SliceType(Type):
    """Unsized array type for pointers: [T]"""
    element_type: Type


@dataclass
class PointerType(Type):
    """Pointer type: *T (near) or far *T"""
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


@dataclass
class TupleType(Type):
    """Tuple type: (T1, T2, ...) for multiple return values."""
    element_types: List[Type]


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
    """Attribute like #[mode(databank=inline)] or #[bank(2)]"""
    name: str
    args: List[AttributeArg]


# ============================================================================
# Conditional Compilation
# ============================================================================

@dataclass
class CfgCondition(ASTNode):
    """Base class for cfg condition expressions."""
    pass


@dataclass
class CfgIdentifier(CfgCondition):
    """Simple cfg identifier (e.g., snes, debug)."""
    name: str


@dataclass
class CfgNot(CfgCondition):
    """Negated cfg condition (e.g., not(target = "nes"))."""
    condition: CfgCondition


@dataclass  
class CfgAny(CfgCondition):
    """Any of several conditions (e.g., any(snes, genesis))."""
    conditions: List[CfgCondition]


@dataclass
class CfgAll(CfgCondition):
    """All of several conditions (e.g., all(snes, debug))."""
    conditions: List[CfgCondition]


@dataclass
class CfgComparison(CfgCondition):
    """Key-value comparison (e.g., target = "snes")."""
    key: str
    operator: str  # "=", "!="
    value: str  # String literal value


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
    is_far: bool  # True if declared with 'far' keyword (required for ROM statics in auto-bank mode)
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
class ImplMethod(ASTNode):
    """Method declaration in an impl block.

    Example:
        fn take_damage(*self, amount @ A: u8) {
            self.health -= amount;
        }
    """
    attributes: List[Attribute]
    is_far: bool  # True if fn is declared far
    name: str
    self_is_far: bool  # True if self param is `far *self`
    params: List[Parameter]  # Parameters after self
    return_type: Optional[Union[Type, NeverType]]
    body: 'Block'


@dataclass
class ImplConst(ASTNode):
    """Associated constant in an impl block.

    Example:
        const MAX_HEALTH: u8 = 100;
    """
    name: str
    const_type: Type
    value: Expression


@dataclass
class ImplDecl(Declaration):
    """Impl block declaration.

    Example:
        impl Player {
            const MAX_HEALTH: u8 = 100;

            fn take_damage(*self, amount @ A: u8) { ... }
        }

        impl far Player {
            fn update(far *self) { ... }
        }
    """
    struct_name: str
    is_far: bool  # True for `impl far StructName`
    methods: List[ImplMethod]
    constants: List[ImplConst]


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


@dataclass
class BankDirective(Declaration):
    """
    Bank directive: #[bank(n)] or #[bank(auto)]

    Sets current ROM bank context for following declarations.
    - #[bank(n)]: Explicit bank number, near functions allowed
    - #[bank(auto)]: Automatic placement, requires far functions and far ROM statics

    bank_number is None for auto mode, otherwise the explicit bank number.
    """
    bank_number: Optional[int]  # None = auto mode

    @property
    def is_auto(self) -> bool:
        """Return True if this is an auto-bank directive."""
        return self.bank_number is None


@dataclass
class SnesRomDirective(Declaration):
    """
    SNES ROM header directive: #[snesrom(name="...", ...)]

    Configures the WLA-DX .SNESHEADER directive parameters.

    Required:
        name: ROM name (max 21 characters)

    Optional with defaults:
        id: Cartridge ID (default: "SNES")
        cartridge_type: Cartridge type (default: 0x00)
        sram_size: SRAM size (default: 0x00)
        country: Country code (default: 0x01 for USA)
        version: ROM version (default: 0x00)

    ROM type flags (mutually exclusive where applicable):
        lorom: Use LoROM mapping (default: true)
        hirom: Use HiROM mapping
        exhirom: Use ExHiROM mapping
        slowrom: Use SlowROM timing (default: true)
        fastrom: Use FastROM timing
    """
    name: str  # Required: ROM name
    id: str = "SNES"  # Cartridge ID
    cartridge_type: int = 0x00  # Cartridge type
    sram_size: int = 0x00  # SRAM size
    country: int = 0x01  # Country code (USA)
    version: int = 0x00  # ROM version
    # ROM type flags
    lorom: bool = True
    hirom: bool = False
    exhirom: bool = False
    slowrom: bool = True
    fastrom: bool = False


# ============================================================================
# Macros
# ============================================================================

@dataclass
class MacroParam(ASTNode):
    """Macro parameter definition (e.g., $name:expr)."""
    name: str           # Parameter name (without $)
    fragment_type: str  # Fragment type: 'expr', 'ident', 'literal', 'ty', 'reg', 'tt'
    is_repeated: bool = False  # True if inside $()*


@dataclass
class MacroDecl(Declaration):
    """Macro definition.

    Example:
        macro! inc_twice($reg:reg) {
            $reg++;
            $reg++;
        }
    """
    name: str
    params: List[MacroParam]
    body_tokens: List[str]  # Raw token strings for the body


@dataclass
class MacroInvocation(Expression):
    """Macro invocation (e.g., my_macro!(arg1, arg2)).

    This is parsed as an expression but expanded during preprocessing.
    The args are stored as raw token strings until expansion.
    """
    name: str
    args: List[str]  # Raw argument token strings


@dataclass
class MacroInvocationStmt(Declaration):
    """Top-level macro invocation statement.

    Macro invocations at the top level (not in expressions) are
    stored as declarations and expanded during preprocessing.
    """
    name: str
    args: List[str]  # Raw argument token strings


@dataclass
class MacroInvocationStmtInner(Statement):
    """Statement-level macro invocation (inside function bodies).

    Example: inc_twice!(X);
    """
    name: str
    args: List[str]  # Raw argument token strings


# ============================================================================
# Statements
# ============================================================================

@dataclass
class Block(Statement):
    """Block of statements."""
    statements: List[Statement]


@dataclass
class TuplePattern:
    """Tuple pattern for destructuring: (a, b, c)"""
    names: List[str]


@dataclass
class LetStmt(Statement):
    """Let binding statement.

    Supports both single binding and tuple destructuring:
      let x = expr;           # name="x", pattern=None
      let (a, b) = expr;      # name=None, pattern=TuplePattern(["a", "b"])
    """
    is_mut: bool
    name: Optional[str]  # Single binding name (None for tuple patterns)
    binding: Optional[Union[str, 'Register']]  # Register or variable for aliasing
    var_type: Optional[Type]
    initializer: Expression
    pattern: Optional[TuplePattern] = None  # Tuple pattern for destructuring


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
    """Break statement with optional label."""
    label: Optional[str] = None  # Target label for labeled break


@dataclass
class ContinueStmt(Statement):
    """Continue statement with optional label."""
    label: Optional[str] = None  # Target label for labeled continue


@dataclass
class IfStmt(Statement):
    """If statement."""
    condition: Expression
    then_block: Block
    else_block: Optional[Union[Block, 'IfStmt']]  # Can chain with else if


@dataclass
class LoopStmt(Statement):
    """Loop statement with optional label."""
    body: Block
    label: Optional[str] = None  # Loop label for break/continue


@dataclass
class WhileStmt(Statement):
    """While statement with optional label."""
    condition: Expression
    body: Block
    label: Optional[str] = None  # Loop label for break/continue


@dataclass
class ForStmt(Statement):
    """For loop statement: for i in start..end { body }"""
    variable: str           # Loop variable name
    start: Expression       # Start expression (inclusive)
    end: Expression         # End expression (exclusive)
    body: Block
    label: Optional[str] = None  # Loop label for break/continue


@dataclass
class AsmNamedArg:
    """Named argument for asm! format string."""
    name: str
    value: Union[str, int, 'Expression']  # String literal, integer, or const expression


@dataclass
class AsmStmt(Statement):
    """Inline assembly statement with optional format string support."""
    instructions: List[str]
    format_args: Optional[Dict[str, Union[str, int, 'Expression']]] = None  # Named args for format substitution


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
class StringLiteral(Expression):
    """String literal for byte array initialization."""
    value: str  # Raw string value (escape sequences not yet processed)


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
class ArrayFillExpr(Expression):
    """Array fill expression (e.g., [0; 256] - fill with value repeated count times)."""
    value: Expression  # The value to repeat
    count: Expression  # Number of repetitions (must be const)


@dataclass
class ArrayLiteralExpr(Expression):
    """Array literal expression (e.g., [1, 2, 3, 4] - explicit elements)."""
    elements: List[Expression]


@dataclass
class StructFieldInit(ASTNode):
    """Field initializer in a struct literal."""
    name: str
    value: Expression


@dataclass
class StructLiteralExpr(Expression):
    """Struct literal expression (e.g., Player { x: 10, y: 20, health: 100 })."""
    struct_name: str
    fields: List[StructFieldInit]


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
class StringifyCall(Expression):
    """Stringify built-in function call."""
    func: Expression  # Should be 'stringify' identifier
    args: List[Expression]


@dataclass
class CfgFunctionCall(Expression):
    """cfg!(condition) compile-time conditional check.

    Evaluates to true/false based on cfg flags passed to compiler.
    Used in if conditions: if cfg!(snes) { ... }
    """
    condition: Expression  # The condition identifier (e.g., 'snes')


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


@dataclass
class MultiAssignment(Expression):
    """Multiple assignment expression for multiple return values (e.g., lo, hi = func())."""
    targets: List[Union[Identifier, Register, ArrayIndex, FieldAccess]]
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
