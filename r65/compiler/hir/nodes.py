# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
HIR (High-level IR) node definitions for the R65 compiler.

These nodes represent the program after name resolution, attribute processing,
and desugaring, but before full type checking.
"""

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Optional, List, NamedTuple, Union, Any
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


@dataclass
class HIRIncludeAsm(HIRDeclaration):
    """include_asm! directive — lowered to WLA-DX `.INCLUDE` at codegen.

    Carries the bank context active at the point of the macro so codegen
    can place the `.INCLUDE` inside the surrounding bank/section.
    """
    path: str = ""
    bank_number: Optional[int] = None  # None = auto-bank mode


# =============================================================================
# Program
# =============================================================================

@dataclass
class HIRProgram(HIRNode):
    """Top-level program."""
    declarations: List[HIRDeclaration] = field(default_factory=list)
    symbol_table: Any = None  # Will be SymbolTable
    stack_attr: Any = None  # StackAttribute from #[stack(...)]
    snesrom_config: Optional[Any] = None  # the ast.SnesRomDirective from #[snesrom(...)]
    trait_dispatch_info: Optional[dict] = None  # Trait dispatch tables for codegen

    def get_declarations_by_type(self, decl_type: type) -> List[HIRDeclaration]:
        """Filter declarations by type."""
        return [d for d in self.declarations if isinstance(d, decl_type)]

    def all_functions(self):
        """Every function body: top-level functions and impl methods alike.

        Impl methods are HIRFunctionDecls named ``Struct__method`` that live in
        their HIRImplDecl rather than at top level.
        """
        for decl in self.declarations:
            if isinstance(decl, HIRFunctionDecl):
                yield decl
            elif isinstance(decl, HIRImplDecl):
                yield from decl.methods

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

    @property
    def traits(self) -> List['HIRTraitDecl']:
        """Get all trait declarations."""
        return self.get_declarations_by_type(HIRTraitDecl)


# =============================================================================
# Bindings (for parameters and let statements)
# =============================================================================

@dataclass
class RegisterBinding:
    """A parameter or let bound to a hardware register (`x @ A`)."""
    register_name: str  # any register with bindable=True in registers.REGISTERS


@dataclass
class VariableBinding:
    """A parameter bound to a static variable (`param @ TEMP`)."""
    variable_name: str
    variable_symbol: Any  # Will be Symbol (resolved reference)


# =============================================================================
# Declarations
# =============================================================================

@dataclass
class HIRParameter(HIRNode):
    """Function parameter."""
    name: str = ""
    param_type: Any = None  # Will be TypeInfo
    binding: Optional[Union[RegisterBinding, VariableBinding]] = None

    # Symbol reference (resolved during HIR construction)
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRFunctionDecl(HIRDeclaration):
    """Function declaration."""
    name: str = ""
    is_far: bool = False
    is_const: bool = False
    is_trait_method: bool = False  # True for trait impl methods (self passed in Y)
    is_extern: bool = False  # `extern fn` — implemented in an included .s file
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
    is_used: bool = False  # #[used]: kept though nothing calls it
    # Lint codes suppressed on this function via #[allow(...)]. The sentinel
    # "*" (attributes.ALLOW_ALL) means every code.
    allow_lints: frozenset = field(default_factory=frozenset)

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol

    # Inferred processor mode (populated by HIR builder)
    # - entry_m_mode: m8 if no u16 A parameter, m16 if u16 @ A parameter
    # - exit_m_mode: m16 if the value returned in A is 2 bytes (u16/i16, near
    #   pointer or fn pointer, 2-byte enum, newtype over any of those), else m8
    # - x_mode is always x16 (16-bit X/Y)
    entry_m_mode: Optional[Any] = None  # Will be ModeState (M8 or M16)
    exit_m_mode: Optional[Any] = None   # Will be ModeState (M8 or M16)


class SoaPlane(NamedTuple):
    """One plane of an `#[soa]` static: a field's array, and where it came from.

    `aos_offset` is the field's offset in the packed layout, the key MIR
    inverts to find the owning plane of a folded field chain.
    """
    name: str
    aos_offset: int
    child: 'HIRStaticDecl'

    @property
    def stride(self) -> int:
        return self.child.soa_stride


@dataclass
class HIRStaticDecl(HIRDeclaration):
    """Static variable declaration."""
    name: str = ""
    is_mutable: bool = False
    is_extern: bool = False  # `extern static` — data lives in an included .s file
    var_type: Any = None  # Will be TypeInfo
    initializer: Optional[HIRExpression] = None

    # Processed attributes
    storage_attr: Optional[Any] = None  # Will be StorageAttribute (None = ROM)
    bank_attr: Optional[Any] = None  # Will be BankAttribute (for ROM statics only)
    # #[soa]: this static is laid out as one array per field of its element
    # struct. The parent itself is allocated no bytes; see docs/soa.md.
    is_soa: bool = False
    # True for the per-field arrays SoA expansion synthesizes. They have no
    # source location the user would recognise, so lint rules skip them.
    is_synthetic: bool = False
    # Parent only: its planes, in declaration order.
    soa_planes: Optional[List[SoaPlane]] = None
    # Child only: bytes per element in this plane. Normally the field's size,
    # rounded up when that costs at most one byte (3 -> 4, 7 -> 8) so the index
    # scales by a shift instead of a shift-and-add.
    soa_stride: Optional[int] = None
    # Child only: `(region_decl, byte_offset)` when this plane lies over the
    # bytes of another plane instead of reserving its own — an `#[soa]`
    # payload's field plane inside its enum's data region (docs/enums.md).
    soa_overlay: Optional[tuple] = None
    # Lint codes suppressed on this static via #[allow(...)]; "*" means all.
    allow_lints: frozenset = field(default_factory=frozenset)

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
    """Struct or union declaration.

    `is_union` only affects field offsets and total size (see
    `unified_type_utils.layout_fields`); everything downstream reads
    `HIRStructField.offset` and treats both alike.
    """
    name: str = ""
    fields: List[HIRStructField] = field(default_factory=list)
    is_union: bool = False
    # #[soa]: arrays of this struct are laid out as one plane per field, and its
    # methods take `self` as an element index. Carried here as well as on the AST
    # node because pass 2 replaces `Symbol.definition` with this one, and the SoA
    # predicate has to answer the same on both. See docs/soa.md.
    is_soa: bool = False

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRNewtypeDecl(HIRDeclaration):
    """Newtype declaration: `struct TileId(u8);`.

    Carries no layout — a newtype *is* its payload at runtime. The node exists so
    the declaration survives into the HIR program for diagnostics.
    """
    name: str = ""
    inner_type: Any = None  # TypeInfo

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIREnumVariant(HIRNode):
    """Enum variant."""
    name: str = ""
    value: int = 0  # Explicit or auto-incremented (resolved during HIR)
    # StructTypeInfo of the payload, or None for a unit variant.
    payload_type: Optional[Any] = None

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIREnumDecl(HIRDeclaration):
    """Enum declaration.

    Two shapes share this node. A *fieldless* enum (every variant unit) is a
    plain integer constant of `underlying_type`. A *tagged* enum — one with at
    least one payload variant — is stored SoA-style as two planes, and
    `payload_union` is the synthetic union holding one field per payload variant
    that gives the data plane its element type. See docs/enums.md.
    """
    name: str = ""
    variants: List[HIREnumVariant] = field(default_factory=list)
    underlying_type: Optional[Any] = None  # Inferred u8/u16 (will be TypeInfo)
    # Synthetic `Enum__Payload` union, or None for a fieldless enum.
    payload_union: Optional['HIRStructDecl'] = None

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol

    @property
    def is_tagged(self) -> bool:
        """True if any variant carries a payload."""
        return any(v.payload_type is not None for v in self.variants)


@dataclass
class HIRTypeAlias(HIRDeclaration):
    """Type alias declaration."""
    name: str = ""
    aliased_type: Any = None  # Will be TypeInfo

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRTraitMethod(HIRNode):
    """Method signature in a trait declaration."""
    is_far: bool = False
    name: str = ""
    self_is_far: bool = False
    params: List['HIRParameter'] = field(default_factory=list)
    return_type: Optional[Any] = None  # Will be TypeInfo


@dataclass
class HIRTraitConst(HIRNode):
    """Associated constant declaration in a trait."""
    name: str = ""
    const_type: Any = None  # Will be TypeInfo


@dataclass
class HIRTraitDecl(HIRDeclaration):
    """Trait declaration.

    Defines method signatures and associated constants that implementing structs must provide.
    Used for TypeId-based dynamic dispatch via jump tables.
    """
    name: str = ""
    methods: List[HIRTraitMethod] = field(default_factory=list)
    constants: List[HIRTraitConst] = field(default_factory=list)
    supertraits: List[str] = field(default_factory=list)  # Inherited trait names

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRImplDecl(HIRDeclaration):
    """Impl block declaration.

    Contains methods and associated constants for a struct.
    Methods are desugared to regular functions with mangled names (StructName__method).
    """
    struct_name: str = ""
    methods: List['HIRFunctionDecl'] = field(default_factory=list)
    constants: List['HIRConstDecl'] = field(default_factory=list)
    trait_name: Optional[str] = None  # Set for trait impls


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
    binding: Optional[RegisterBinding] = None

    # Symbol reference
    symbol: Optional[Any] = None  # Will be Symbol


@dataclass
class HIRMultiLetStmt(HIRStatement):
    """Multi-binding let statement: let [mut] a, b = multi_return_func();

    Supports partial capture — binding fewer names than the return count.
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
    """Break statement with optional label target and optional value (for loop expressions)."""
    label: Optional[str] = None  # Target label for labeled break
    value: Optional[HIRExpression] = None  # Break value for loop expressions


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
    suffix: Optional[str] = None


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

    Flags: see hir/status_flags.STATUS_FLAGS (Carry, Zero, Irq, Decimal, XY16, A16, Overflow, Negative)
    """
    flag_name: str = ""       # a key of status_flags.STATUS_FLAGS


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
    """Type cast (explicit conversion).

    Also carries newtype payload access (`t.0`), which is a retype and nothing
    more: `newtype_field` is set and `target_type` stays None until the type checker
    fills it in from the operand's newtype. Both spellings lower identically —
    equal-size casts emit a bare `Move` (`mir/lowerers/expression.py`).
    """
    expr: Optional[HIRExpression] = None
    target_type: Any = None  # Will be TypeInfo
    newtype_field: Optional[int] = None  # Set for `.N` payload access
    # Set for `Newtype(x)`. Lowers exactly like a cast, but is *checked* like an
    # assignment into the payload — so it rejects what `let t: Newtype = x;`
    # rejects, and `as` stays the only spelling that truncates.
    newtype_construct: bool = False


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
    # Tagged enums only: the name a payload pattern binds (`Shape::C(c)`), and
    # its symbol. The symbol carries `enum_payload_alias`, so every use of the
    # name expands to the pool element rather than reading a local copy.
    binding: Optional[str] = None
    binding_symbol: Optional[Any] = None


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
class HIRRangePattern(HIRPattern):
    """Range pattern (e.g., 0..5 or 0..=5)."""
    start: int = 0
    end: int = 0
    inclusive: bool = True


@dataclass
class HIROrPattern(HIRPattern):
    """Or pattern (p1 | p2 | ...)."""
    patterns: List[HIRPattern] = field(default_factory=list)


@dataclass
class HIRMatchArm(HIRNode):
    """Single arm of a match expression.

    body is an HIRExpression for normal arms, or an HIRStatement (e.g. HIRReturnStmt)
    for arms with control flow statements like return/break/continue.
    """
    pattern: Optional[HIRPattern] = None
    body: Optional['HIRNode'] = None  # HIRExpression or HIRStatement
    # Scope for pattern bindings
    scope_id: Optional[int] = None


@dataclass
class HIRBlockExpression(HIRExpression):
    """Block expression: { stmt*; expr } - a block whose last item is its value."""
    statements: List[HIRStatement] = field(default_factory=list)
    final_expr: Optional[HIRExpression] = None
    scope_id: Optional[int] = None


@dataclass
class HIRIfExpression(HIRExpression):
    """If expression: if cond { expr } else { expr } - produces a value.

    Both branches must be present (else is required for expressions).
    Each branch is either a HIRBlockExpression or a nested HIRIfExpression.
    """
    condition: Optional[HIRExpression] = None
    then_block: Optional['HIRBlockExpression'] = None
    else_block: Optional[Union['HIRBlockExpression', 'HIRIfExpression']] = None


@dataclass
class HIRLoopExpression(HIRExpression):
    """Loop expression: loop { ... break val; ... } - produces a value via break."""
    body: Optional[HIRBlock] = None
    label: Optional[str] = None


@dataclass
class HIRMatchExpression(HIRExpression):
    """Match expression."""
    scrutinee: Optional[HIRExpression] = None  # Expression being matched
    arms: List[HIRMatchArm] = field(default_factory=list)
    # Name of the tagged enum being matched, if any. The scrutinee has already
    # been rewritten to a read of that enum's tag plane, so it types as a plain
    # integer — this is what tells the type checker to check variant coverage
    # rather than demand a wildcard the way it does for any other integer.
    tagged_enum: Optional[str] = None
    # 'if_let' / 'let_else' when the parser desugared this match from
    # `if let` / `while let` / `let ... else` (see ast.MatchExpression).
    origin: Optional[str] = None


def walk_hir(node):
    """Every HIR node under *node*, *node* included, each parent before its
    children.

    Follows dataclass fields holding HIR nodes or lists of them; a Symbol is
    not an HIR node, so a reference never leads back into a declaration.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
            continue
        if not isinstance(current, HIRNode):
            continue
        yield current
        for f in reversed(dataclass_fields(current)):
            value = getattr(current, f.name, None)
            if isinstance(value, (HIRNode, list, tuple)):
                stack.append(value)
