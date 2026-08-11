# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Attribute validation and processing for R65 HIR.

Validates and transforms AST attributes into structured HIR attributes.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from r65.compiler.frontend import ast
from r65.compiler.frontend.ast import CfgCondition
from r65.compiler.hir.errors import *


# =============================================================================
# Processed Attributes
# =============================================================================

@dataclass
class ProcessedAttribute:
    """Base class for validated attributes."""
    name: str
    source_loc: Optional[SourceLocation] = None


# Mode attribute - simplified to only contain databank parameter
# CPU mode (m8/m16, x8/x16) is now inferred from parameter types:
# - Default: m8 (8-bit A), x16 (16-bit X/Y)
# - Function entry: m16 if A parameter is u16, else m8
# - X/Y always u16 (compiler error if u8 X/Y parameter)
# - Auto REP/SEP inserted around 16-bit A operations


class DataBankMode(Enum):
    """Data bank register management mode."""
    NONE = "none"        # No DBR management (default)
    INLINE = "inline"    # Callee manages DBR (inlined in function)
    CALLER = "caller"    # Caller manages DBR


@dataclass
class ModeAttribute(ProcessedAttribute):
    """#[mode(databank=...)] - only databank parameter retained.

    CPU mode (m8/m16, x8/x16) is now automatically inferred from:
    - Parameter types: u16 @ A -> m16 entry, otherwise m8
    - X/Y registers are always u16 (x16 mode)
    """
    databank: DataBankMode = field(default=DataBankMode.NONE)


# Preserves attribute
@dataclass
class PreservesAttribute(ProcessedAttribute):
    """#[preserves(A, X, Y, STATUS, D, DBR, S)] - B and PBR not allowed"""
    registers: List[str] = field(default_factory=list)


# Storage attributes
class StorageKind(Enum):
    """Storage location kind."""
    ZEROPAGE = "zeropage"  # Direct page ($0000-$00FF) - uses DP addressing
    LOWRAM = "lowram"      # Low RAM ($0000-$1FFF) - auto starts at $0100
    RAM = "ram"            # Main RAM ($7E2000-$7FFFFF)
    # ROM removed - immutable statics are implicitly ROM
    HW = "hw"              # Hardware-mapped I/O (automatically volatile)


@dataclass
class StorageAttribute(ProcessedAttribute):
    """#[zeropage], #[lowram], #[ram], #[hw]"""
    storage_kind: StorageKind = StorageKind.RAM
    address: Optional[int] = None  # For explicit addresses
    is_register: bool = False  # True if marked as scratch register with 'register' parameter


# Bank attribute
@dataclass
class BankAttribute(ProcessedAttribute):
    """#[bank(n)] or #[bank(auto)] - specifies ROM bank placement.

    bank_number: Explicit bank number, or None for auto-placement mode.
    """
    bank_number: Optional[int] = 0

    @property
    def is_auto(self) -> bool:
        """Return True if this is an auto-bank attribute."""
        return self.bank_number is None


# Interrupt attribute
class InterruptVector(Enum):
    """Interrupt vector types."""
    NMI = "nmi"
    IRQ = "irq"
    BRK = "brk"
    COP = "cop"
    ABORT = "abort"


@dataclass
class InterruptAttribute(ProcessedAttribute):
    """#[interrupt(nmi/irq/brk/cop/abort, preserve=...)]"""
    vector: InterruptVector = InterruptVector.NMI
    preserve: bool = True  # Auto-preservation (default)


# Entry attribute
@dataclass
class EntryAttribute(ProcessedAttribute):
    """#[entry] - marks main entry point"""
    pass


# Inline attribute
class InlineMode(Enum):
    """Inline attribute mode."""
    HINT = "hint"      # #[inline] - hint, size-gated by INLINE_COST_WITH_ATTR
    ALWAYS = "always"  # #[inline(always)] - directive, bypass size budget
    NEVER = "never"    # #[inline(never)] - never inline this function


@dataclass
class InlineAttribute(ProcessedAttribute):
    """#[inline], #[inline(always)], or #[inline(never)] - controls function inlining.

    Modes:
    - HINT (bare #[inline]): size-gated. Inlined if body cost < INLINE_COST_WITH_ATTR.
    - ALWAYS (#[inline(always)]): directive. Inlined regardless of body size
      (modulo can_inline soundness requirements). Matches Rust/LLVM semantics.
    - NEVER (#[inline(never)]): never inlined.

    Functions without any inline attribute use the implicit-inline heuristics
    (called-once, small leaf, etc.) when -O2 is enabled.
    """
    mode: InlineMode = InlineMode.HINT


# CFG attribute
@dataclass
class CfgAttribute(ProcessedAttribute):
    """#[cfg(condition)] - conditional compilation attribute"""
    condition: CfgCondition = field(kw_only=True)  # CFG condition AST node


# Stack attribute
@dataclass
class StackAttribute(ProcessedAttribute):
    """#[stack(lower, upper)] - reserves stack region in low RAM"""
    lower: int = 0x1F00  # Lower bound of stack region
    upper: int = 0x1FFF  # Upper bound of stack region


# Allow attribute (lint suppression)
ALLOW_ALL = "*"  # sentinel in a codes set meaning "every lint code"


@dataclass
class AllowAttribute(ProcessedAttribute):
    """#[allow(CODE, ...)] - suppress the named lint codes for this item.

    Rust-style, lexically scoped: the codes are suppressed for diagnostics
    arising inside the annotated item (a function's body, a static's decl), not
    transitively through calls. ``#[allow(all)]`` stores the ``ALLOW_ALL``
    sentinel, which suppresses every code.
    """
    codes: frozenset = field(default_factory=frozenset)


# =============================================================================
# Unsupported Rust attribute hints
# =============================================================================

_PROC_MACRO_HINT = (
    "procedural macros are not supported in R65 — use `macro_rules!` "
    "for declarative macros (see docs/macros.md)"
)

_UNSUPPORTED_RUST_ATTRS = {
    'proc_macro': _PROC_MACRO_HINT,
    'proc_macro_derive': _PROC_MACRO_HINT,
    'proc_macro_attribute': _PROC_MACRO_HINT,
    'derive': (
        "derive macros are not supported in R65 — traits do not auto-derive; "
        "implement the behavior explicitly with free functions or `impl` blocks"
    ),
}


# =============================================================================
# Attribute Processor
# =============================================================================

class AttributeProcessor:
    """Validates and transforms AST attributes into HIR attributes."""

    def process_attributes(
        self,
        ast_attrs: List[ast.Attribute],
        context: str  # "function", "static", etc.
    ) -> List[ProcessedAttribute]:
        """Process list of AST attributes."""
        processed = []

        for attr in ast_attrs:
            if attr.name == 'mode':
                processed.append(self._process_mode(attr, context))
            elif attr.name == 'preserves':
                processed.append(self._process_preserves(attr, context))
            elif attr.name in ['zeropage', 'lowram', 'ram', 'hw']:
                processed.append(self._process_storage(attr, context))
            elif attr.name == 'stack':
                processed.append(self._process_stack(attr, context))
            elif attr.name == 'cfg':
                processed.append(self._process_cfg(attr, context))
            elif attr.name == 'bank':
                processed.append(self._process_bank(attr, context))
            elif attr.name == 'interrupt':
                processed.append(self._process_interrupt(attr, context))
            elif attr.name == 'entry':
                processed.append(self._process_entry(attr, context))
            elif attr.name == 'inline':
                processed.append(self._process_inline(attr, context))
            elif attr.name == 'allow':
                processed.append(self._process_allow(attr, context))
            else:
                hint = _UNSUPPORTED_RUST_ATTRS.get(attr.name)
                raise HIRError(
                    f"Unknown attribute '{attr.name}'",
                    source_loc=attr.source_loc,
                    hint=hint,
                )

        return processed

    def _process_mode(self, attr: ast.Attribute, context: str) -> ModeAttribute:
        """Process #[mode(...)] attribute.

        The #[mode] attribute now only supports the databank parameter.
        CPU mode (m8/m16, x8/x16) is automatically inferred from parameter types.
        """
        if context not in ['function']:
            raise HIRError(f"#[mode] attribute only valid on functions", source_loc=attr.source_loc)

        databank = DataBankMode.NONE

        for arg in attr.args:
            if arg.name is None:  # Positional argument
                value_str = self._get_arg_identifier(arg.value)

                # Reject old m8/m16/x8/x16 positional arguments
                if value_str in ('m8', 'm16', 'x8', 'x16'):
                    raise HIRError(
                        f"#[mode({value_str})] is no longer supported.\n"
                        f"  CPU mode is now automatically inferred from parameter types:\n"
                        f"  - u16 @ A parameter -> m16 entry mode\n"
                        f"  - otherwise -> m8 entry mode (default)\n"
                        f"  - X/Y registers are always u16 (x16 mode)\n"
                        f"  Use #[mode(databank=...)] for data bank management only.",
                        source_loc=attr.source_loc
                    )
                else:
                    raise HIRError(f"Invalid mode value: {value_str}", source_loc=attr.source_loc)
            elif arg.name == 'transition':
                raise HIRError(
                    f"#[mode(transition=...)] is no longer supported.\n"
                    f"  Mode transitions are now handled automatically:\n"
                    f"  - Compiler inserts REP/SEP around 16-bit A operations\n"
                    f"  - Call sites switch to callee's entry mode as needed",
                    source_loc=attr.source_loc
                )
            elif arg.name == 'databank':
                value_str = self._get_arg_identifier(arg.value)

                if value_str == 'none':
                    databank = DataBankMode.NONE
                elif value_str == 'inline':
                    databank = DataBankMode.INLINE
                elif value_str == 'caller':
                    databank = DataBankMode.CALLER
                else:
                    raise HIRError(f"Invalid databank value: {value_str}", source_loc=attr.source_loc)
            else:
                raise HIRError(f"Unknown argument to #[mode]: {arg.name}", source_loc=attr.source_loc)

        return ModeAttribute(
            name='mode',
            databank=databank
        )

    def _process_preserves(self, attr: ast.Attribute, context: str) -> PreservesAttribute:
        """Process #[preserves(...)] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[preserves] attribute only valid on functions", source_loc=attr.source_loc)

        registers = []
        valid_registers = {'A', 'X', 'Y', 'STATUS', 'D', 'DBR', 'S'}

        for arg in attr.args:
            if arg.name is not None:
                raise HIRError(f"#[preserves] does not accept named arguments", source_loc=attr.source_loc)

            # Get register name
            if isinstance(arg.value, ast.Register):
                reg_name = arg.value.name
            elif isinstance(arg.value, ast.Identifier):
                reg_name = arg.value.name
            else:
                raise HIRError(f"#[preserves] expects register names", source_loc=attr.source_loc)

            if reg_name == 'B':
                raise HIRError(
                    f"B register not allowed in preserves attribute\n"
                    f"  B cannot be preserved separately from A\n"
                    f"  B is the high byte of the A register",
                    source_loc=attr.source_loc
                )

            if reg_name not in valid_registers:
                raise HIRError(f"Invalid register in #[preserves]: {reg_name}", source_loc=attr.source_loc)

            if reg_name == 'PBR':
                raise HIRError(f"PBR cannot be in #[preserves] (it's read-only)", source_loc=attr.source_loc)

            registers.append(reg_name)

        return PreservesAttribute(
            name='preserves',
            registers=registers
        )

    def _process_storage(self, attr: ast.Attribute, context: str) -> StorageAttribute:
        """Process #[zeropage], #[lowram], #[ram], #[hw] attributes."""
        if context not in ['static']:
            raise HIRError(f"#{attr.name} attribute only valid on static variables", source_loc=attr.source_loc)

        # Map attribute name to storage kind
        storage_map = {
            'zeropage': StorageKind.ZEROPAGE,
            'lowram': StorageKind.LOWRAM,
            'ram': StorageKind.RAM,
            'hw': StorageKind.HW,
        }
        storage_kind = storage_map[attr.name]

        # Parse arguments: address (positional) and register (named or flag)
        address = None
        is_register = False

        for arg in attr.args:
            if arg.name is None:  # Positional argument
                # Check if this is a flag keyword (bare identifier)
                if isinstance(arg.value, ast.Identifier) and arg.value.name == 'register':
                    # Flag-style: #[zeropage(0x10, register)]
                    is_register = True
                elif address is not None:
                    raise HIRError(f"#{attr.name} can only have one positional argument (address)", source_loc=attr.source_loc)
                elif isinstance(arg.value, ast.IntegerLiteral):
                    # Address argument
                    address = arg.value.value
                else:
                    raise HIRError(f"#{attr.name} address must be an integer literal", source_loc=attr.source_loc)

            elif arg.name == 'register':
                # Handle register parameter: 'register' (keyword only)
                # No value means register=true
                if arg.value is None:
                    is_register = True
                elif isinstance(arg.value, ast.BooleanLiteral):
                    is_register = arg.value.value
                elif isinstance(arg.value, ast.Identifier) and arg.value.name == 'true':
                    is_register = True
                elif isinstance(arg.value, ast.Identifier) and arg.value.name == 'false':
                    is_register = False
                else:
                    raise HIRError(f"#{attr.name} register parameter must be boolean or omitted", source_loc=attr.source_loc)

            else:
                raise HIRError(f"Unknown argument to #{attr.name}: {arg.name}", source_loc=attr.source_loc)

        return StorageAttribute(
            name=attr.name,
            storage_kind=storage_kind,
            address=address,
            is_register=is_register
        )

    def _process_bank(self, attr: ast.Attribute, context: str) -> BankAttribute:
        """Process #[bank(n)] attribute - now a directive, not a function attribute."""
        # #[bank(n)] is now a global directive, not a function attribute
        raise HIRError(
            f"#[bank(n)] is a global directive, not a function attribute. "
            f"Place #[bank(n)] before function declarations to set the bank for "
            f"all following functions and ROM statics. Example:\n"
            f"  #[bank(1)]\n"
            f"  far fn my_function() {{ }}",
            source_loc=attr.source_loc
        )

    def _process_interrupt(self, attr: ast.Attribute, context: str) -> InterruptAttribute:
        """Process #[interrupt(vector, preserve=...)] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[interrupt] attribute only valid on functions", source_loc=attr.source_loc)

        vector = None
        preserve = True  # Default

        for arg in attr.args:
            if arg.name is None:  # Positional argument (vector)
                if vector is not None:
                    raise HIRError(f"#[interrupt] can only have one positional argument (vector)", source_loc=attr.source_loc)

                value_str = self._get_arg_identifier(arg.value)

                if value_str == 'nmi':
                    vector = InterruptVector.NMI
                elif value_str == 'irq':
                    vector = InterruptVector.IRQ
                elif value_str == 'brk':
                    vector = InterruptVector.BRK
                elif value_str == 'cop':
                    vector = InterruptVector.COP
                elif value_str == 'abort':
                    vector = InterruptVector.ABORT
                else:
                    raise HIRError(f"Invalid interrupt vector: {value_str}", source_loc=attr.source_loc)

            elif arg.name == 'preserve':
                if isinstance(arg.value, ast.BooleanLiteral):
                    preserve = arg.value.value
                elif isinstance(arg.value, ast.Identifier):
                    value_str = arg.value.name
                    if value_str == 'true':
                        preserve = True
                    elif value_str == 'false':
                        preserve = False
                    else:
                        raise HIRError(f"Invalid preserve value: {value_str}", source_loc=attr.source_loc)
                else:
                    raise HIRError(f"preserve must be a boolean", source_loc=attr.source_loc)
            else:
                raise HIRError(f"Unknown argument to #[interrupt]: {arg.name}", source_loc=attr.source_loc)

        if vector is None:
            raise HIRError(f"#[interrupt] requires an interrupt vector", source_loc=attr.source_loc)

        return InterruptAttribute(
            name='interrupt',
            vector=vector,
            preserve=preserve
        )

    def _process_entry(self, attr: ast.Attribute, context: str) -> EntryAttribute:
        """Process #[entry] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[entry] attribute only valid on functions", source_loc=attr.source_loc)

        if len(attr.args) > 0:
            raise HIRError(f"#[entry] does not accept arguments", source_loc=attr.source_loc)

        return EntryAttribute(name='entry')

    def _process_inline(self, attr: ast.Attribute, context: str) -> InlineAttribute:
        """Process #[inline], #[inline(always)], or #[inline(never)] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[inline] attribute only valid on functions", source_loc=attr.source_loc)

        # Bare #[inline] is HINT (size-gated); explicit #[inline(always)] /
        # #[inline(never)] override that.
        mode = InlineMode.HINT

        if len(attr.args) > 1:
            raise HIRError(f"#[inline] accepts at most one argument (always or never)", source_loc=attr.source_loc)

        if len(attr.args) == 1:
            arg = attr.args[0]
            if arg.name is not None:
                raise HIRError(f"#[inline] does not accept named arguments", source_loc=attr.source_loc)

            value_str = self._get_arg_identifier(arg.value)

            if value_str == 'always':
                mode = InlineMode.ALWAYS
            elif value_str == 'never':
                mode = InlineMode.NEVER
            else:
                raise HIRError(f"Invalid #[inline] argument: {value_str}. Expected 'always' or 'never'", source_loc=attr.source_loc)

        return InlineAttribute(name='inline', mode=mode)

    def _process_allow(self, attr: ast.Attribute, context: str) -> AllowAttribute:
        """Process #[allow(CODE, ...)] — lint suppression for this item.

        Each positional argument is a lint code identifier (e.g. ``C001``,
        ``L001``); the special identifier ``all`` suppresses every code. Named
        arguments are rejected. Codes are not validated against the enabled
        rule set here — an allow for an unknown code is simply inert, matching
        the leniency of Rust's ``#[allow]``.
        """
        if not attr.args:
            raise HIRError(
                "#[allow(...)] requires at least one lint code (or `all`)",
                source_loc=attr.source_loc,
                hint="e.g. #[allow(C001)] or #[allow(C001, L001)]",
            )
        codes = set()
        for arg in attr.args:
            if arg.name is not None:
                raise HIRError(
                    "#[allow] does not accept named arguments",
                    source_loc=attr.source_loc,
                )
            code = self._get_arg_identifier(arg.value)
            codes.add(ALLOW_ALL if code == 'all' else code)
        return AllowAttribute(name='allow', codes=frozenset(codes))

    def _process_stack(self, attr: ast.Attribute, context: str) -> StackAttribute:
        """Process #[stack(lower, upper)] attribute."""
        if context not in ['static']:
            raise HIRError(f"#[stack] attribute only valid on static declarations", source_loc=attr.source_loc)

        lower = None
        upper = None

        for i, arg in enumerate(attr.args):
            if arg.name is not None:
                raise HIRError(f"#[stack] only accepts positional arguments", source_loc=attr.source_loc)

            if not isinstance(arg.value, ast.IntegerLiteral):
                raise HIRError(f"#[stack] arguments must be integer literals", source_loc=attr.source_loc)

            if i == 0:
                lower = arg.value.value
            elif i == 1:
                upper = arg.value.value
            else:
                raise HIRError(f"#[stack] accepts exactly 2 arguments (lower, upper)", source_loc=attr.source_loc)

        if lower is None or upper is None:
            raise HIRError(f"#[stack] requires both lower and upper bounds: #[stack(lower, upper)]", source_loc=attr.source_loc)

        if lower > upper:
            raise HIRError(f"Stack lower bound ${lower:04X} must be <= upper bound ${upper:04X}", source_loc=attr.source_loc)

        if lower < 0x0000 or upper > 0x1FFF:
            raise HIRError(
                f"Stack region ${lower:04X}-${upper:04X} must be within "
                f"low RAM ($0000-$1FFF)",
                source_loc=attr.source_loc
            )

        return StackAttribute(
            name='stack',
            lower=lower,
            upper=upper
        )

    def _process_cfg(self, attr: ast.Attribute, context: str) -> CfgAttribute:
        """Process #[cfg(condition)] attribute."""
        if len(attr.args) != 1:
            raise HIRError(f"#[cfg] requires exactly one argument (the condition)", source_loc=attr.source_loc)
        
        condition_arg = attr.args[0]
        if not isinstance(condition_arg.value, CfgCondition):
            raise HIRError(f"#[cfg] argument must be a condition expression", source_loc=attr.source_loc)
        
        return CfgAttribute(
            name='cfg',
            condition=condition_arg.value
        )

    def _get_arg_identifier(self, value) -> str:
        """Extract identifier name from argument value."""
        if isinstance(value, ast.Identifier):
            return value.name
        else:
            raise HIRError(f"Expected identifier, got {type(value).__name__}", source_loc=getattr(value, 'source_loc', None))
