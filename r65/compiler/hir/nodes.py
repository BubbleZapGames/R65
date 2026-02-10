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
# Program Configuration
# =============================================================================

@dataclass
class SnesRomConfig:
    """
    SNES ROM header configuration from #[snesrom(...)] directive.

    Configures the WLA-DX .SNESHEADER directive output.
    """
    name: str  # ROM name (max 21 characters)
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


# =============================================================================
# Program
# =============================================================================

@dataclass
class HIRProgram(HIRNode):
    """Top-level program."""
    declarations: List[HIRDeclaration] = field(default_factory=list)
    symbol_table: Any = None  # Will be SymbolTable
    stack_attr: Any = None  # StackAttribute from #[stack(...)]
    snesrom_config: Optional[SnesRomConfig] = None  # SNES ROM header config from #[snesrom(...)]

    def get_declarations_by_type(self, decl_type: type) -> List[HIRDeclaration]:
        """Filter declarations by type."""
        return [d for d in self.declarations if isinstance(d, decl_type)]

    @property
    def functions(self) -> List['HIRFunctionDecl']:
        """Get all function declarations."""
        return self.get_declarations_by_type(HIRFunctionDecl)

    @property
    def statics(self) -> List['HIRStaticDecl']:
        """Get all static variable declarations."""
        return self.get_declarations_by_type(HIRStaticDecl)

    @property
    def constants(self) -> List['HIRConstDecl']:
        """Get all constant declarations."""
        return self.get_declarations_by_type(HIRConstDecl)

    @property
    def structs(self) -> List['HIRStructDecl']:
        """Get all struct declarations."""
        return self.get_declarations_by_type(HIRStructDecl)

    @property
    def enums(self) -> List['HIREnumDecl']:
        """Get all enum declarations."""
        return self.get_declarations_by_type(HIREnumDecl)


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
    register_name: str  # "A", "X", "Y", "B"


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
    is_const: bool = False
    parameters: List[HIRParameter] = field(default_factory=list)
    return_type: Optional[Any] = None  # Will be TypeInfo or None
    body: Optional['HIRBlock'] = None

    # Processed attributes
    mode_attr: Optional[Any] = None  # Will be ModeAttribute (databank only)
    preserves_attr: Optional[Any] = None  # Will be PreservesAttribute
    bank_attr: Optional[Any] = None  # Will be BankAttribute
    interrupt_attr: Optional[Any] = None  # Will be InterruptAttribute
    inline_attr: Optional[Any] = None  # Will be InlineAttribute
    is_entry: bool = False

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol

    # STATUS flag return tracking (for optimized branch generation at call sites)
    returns_status_flag: Optional[str] = None  # Flag name if function directly returns STATUS.Flag

    # Inferred processor mode (populated by HIR builder)
    # - entry_m_mode: m8 if no u16 A parameter, m16 if u16 @ A parameter
    # - exit_m_mode: m16 if return type is u16/i16, else m8
    # - x_mode is always x16 (16-bit X/Y)
    entry_m_mode: Optional[Any] = None  # Will be ModeState (M8 or M16)
    exit_m_mode: Optional[Any] = None   # Will be ModeState (M8 or M16)


@dataclass
class HIRStaticDecl(HIRDeclaration):
    """Static variable declaration."""
    name: str = ""
    is_mutable: bool = False
    var_type: Any = None  # Will be TypeInfo
    initializer: Optional[HIRExpression] = None

    # Processed attributes
    storage_attr: Optional[Any] = None  # Will be StorageAttribute (None = ROM)
    bank_attr: Optional[Any] = None  # Will be BankAttribute (for ROM statics only)

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


@dataclass
class HIRImplDecl(HIRDeclaration):
    """Impl block declaration.

    Contains methods and associated constants for a struct.
    Methods are desugared to regular functions with mangled names (StructName__method).
    """
    struct_name: str = ""
    is_far: bool = False  # True for `impl far StructName`
    methods: List['HIRFunctionDecl'] = field(default_factory=list)
    constants: List['HIRConstDecl'] = field(default_factory=list)


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
class HIRTupleLetStmt(HIRStatement):
    """Tuple destructuring let binding.

    Used for: let (a, b) = func_returning_tuple();

    Supports partial capture - binding fewer names than the tuple size.
    Extra return values are discarded.
    """
    names: List[str] = field(default_factory=list)  # Variable names to bind
    is_mutable: bool = False
    var_types: List[Any] = field(default_factory=list)  # Type for each binding
    initializer: Optional[HIRExpression] = None

    # Symbol references for each binding
    symbols: List[Any] = field(default_factory=list)  # Will be List[Symbol]


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
    """Break statement with optional label target."""
    label: Optional[str] = None  # Target label for labeled break


@dataclass
class HIRContinueStmt(HIRStatement):
    """Continue statement with optional label target."""
    label: Optional[str] = None  # Target label for labeled continue


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
    label: Optional[str] = None  # Loop label for labeled break/continue


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
class HIREnumVariantExpr(HIRExpression):
    """Enum variant expression (e.g., Suit::Spades)."""
    enum_name: str = ""
    variant_name: str = ""
    value: int = 0  # Resolved integer value


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
    name: str = ""  # "A", "X", "Y", "B", "STATUS", "D", "DBR", "PBR", "S"
    symbol: Any = None  # Points to register symbol


@dataclass
class HIRStatusFlagAccess(HIRExpression):
    """
    Access to a STATUS register flag (e.g., STATUS.Carry).

    Represents property access on the STATUS register for individual CPU flags.
    Used for optimized branch generation (BCS, BCC, BEQ, etc.) and flag manipulation.

    Flags: Carry, Zero, Irq, Decimal, Index, Accumulator, Overflow, Negative
    """
    flag_name: str = ""       # "Carry", "Zero", "Irq", "Decimal", "Index", "Accumulator", "Overflow", "Negative"
    bit_position: int = 0     # 0-7
    bit_mask: int = 0x01      # 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80


@dataclass
class HIRIncludeBytesExpr(HIRExpression):
    """Include binary data from file."""
    path: str = ""  # Path to binary file (original, may be relative)
    resolved_path: str = ""  # Fully resolved path to binary file
    size: int = 0  # Size of the file in bytes


@dataclass
class HIRArrayFillExpr(HIRExpression):
    """
    Array fill expression (e.g., [0; 256]).

    Creates an array filled with a repeated value.
    Used for efficient memory initialization.
    """
    fill_value: Optional[HIRExpression] = None  # Value to repeat
    count: int = 0  # Number of repetitions (evaluated at compile time)


@dataclass
class HIRArrayLiteralExpr(HIRExpression):
    """
    Array literal expression (e.g., [1, 2, 3, 4]).

    Creates an array with explicit element values.
    Used for ROM data that gets block-copied to RAM.
    """
    elements: List[HIRExpression] = field(default_factory=list)


@dataclass
class HIRStringLiteral(HIRExpression):
    """
    String literal for byte array initialization.

    Only valid in static array initializers. The raw string value is preserved
    here and converted to bytes during type checking/code generation.

    Escape sequences supported: \\n, \\t, \\r, \\0, \\\\, \\", \\x##
    Extended ASCII characters (0x00-0xFF) are allowed.
    UTF-8 multi-byte characters (code points > 255) are an error.
    """
    value: str = ""  # Raw string value (escape sequences not yet processed)
    processed_bytes: List[int] = field(default_factory=list)  # Populated by type checker


@dataclass
class HIRStructFieldInit(HIRNode):
    """Field initializer in a struct literal."""
    name: str = ""
    value: Optional[HIRExpression] = None
    field_offset: Optional[int] = None  # Computed during HIR construction


@dataclass
class HIRStructLiteralExpr(HIRExpression):
    """
    Struct literal expression (e.g., Player { x: 10, y: 20, health: 100 }).

    Creates a struct with explicit field values.
    Used for ROM data that gets block-copied to RAM.
    """
    struct_name: str = ""
    struct_decl: Optional['HIRStructDecl'] = None  # Resolved during HIR construction
    fields: List[HIRStructFieldInit] = field(default_factory=list)


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
    builtin_name: Optional[str] = None  # Set if this is a built-in function call
    method_call_info: Optional[dict] = None  # Set by type checker for method calls


@dataclass
class HIRMethodCall(HIRExpression):
    """Method call (e.g., value.rotate_left(3))."""
    receiver: Optional[HIRExpression] = None  # The object/value the method is called on
    method_name: str = ""  # Name of the method
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
    auto_deref: bool = False  # True if base is a pointer that gets auto-dereferenced


@dataclass
class HIRDereference(HIRExpression):
    """Pointer dereference."""
    pointer: Optional[HIRExpression] = None


@dataclass
class HIRAddressOf(HIRExpression):
    """Address-of operator (&variable)."""
    operand: Optional[HIRExpression] = None


@dataclass
class HIRAssignment(HIRExpression):
    """Assignment expression."""
    target: Optional[HIRExpression] = None  # Must be lvalue
    value: Optional[HIRExpression] = None


@dataclass
class HIRMultiAssignment(HIRExpression):
    """Multiple assignment expression for multiple return values."""
    targets: List[HIRExpression] = field(default_factory=list)  # Must be lvalues
    value: Optional[HIRExpression] = None


# ============================================================================
# Pattern Matching
# ============================================================================

@dataclass
class HIRPattern(HIRNode):
    """Base class for HIR pattern nodes."""
    pass


@dataclass
class HIRLiteralPattern(HIRPattern):
    """Literal pattern (integer or boolean)."""
    value: Union[int, bool] = 0


@dataclass
class HIREnumPattern(HIRPattern):
    """Enum variant pattern."""
    enum_name: str = ""
    variant_name: str = ""
    variant_value: Optional[int] = None  # Resolved during HIR building


@dataclass
class HIRWildcardPattern(HIRPattern):
    """Wildcard pattern (_) - matches anything."""
    pass


@dataclass
class HIRIdentifierPattern(HIRPattern):
    """Identifier pattern - binds value to variable."""
    name: str = ""
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIROrPattern(HIRPattern):
    """Or pattern (p1 | p2 | ...)."""
    patterns: List[HIRPattern] = field(default_factory=list)


@dataclass
class HIRMatchArm(HIRNode):
    """Single arm of a match expression."""
    pattern: Optional[HIRPattern] = None
    body: Optional[HIRExpression] = None
    # Scope for pattern bindings
    scope_id: Optional[int] = None


@dataclass
class HIRMatchExpression(HIRExpression):
    """Match expression."""
    scrutinee: Optional[HIRExpression] = None  # Expression being matched
    arms: List[HIRMatchArm] = field(default_factory=list)
