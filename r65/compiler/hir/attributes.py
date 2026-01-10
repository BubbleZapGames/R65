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


# Mode attribute
class MMode(Enum):
    """Accumulator mode."""
    M8 = "m8"    # 8-bit accumulator
    M16 = "m16"  # 16-bit accumulator


class XMode(Enum):
    """Index register mode."""
    X8 = "x8"    # 8-bit index
    X16 = "x16"  # 16-bit index


class ModeTransition(Enum):
    """Mode transition strategy."""
    NONE = "none"        # No automatic transitions (default)
    INLINE = "inline"    # Callee manages transition (PHP/REP/SEP/PLP inlined in function)
    CALLER = "caller"    # Caller manages transition (batching)


class DataBankMode(Enum):
    """Data bank register management mode."""
    NONE = "none"        # No DBR management (default)
    INLINE = "inline"    # Callee manages DBR (inlined in function)
    CALLER = "caller"    # Caller manages DBR


@dataclass
class ModeAttribute(ProcessedAttribute):
    """#[mode(m8/m16, x8/x16, transition=..., databank=...)]"""
    m_mode: Optional[MMode] = None  # None if not specified
    x_mode: Optional[XMode] = None  # None if not specified
    transition: ModeTransition = field(default=ModeTransition.NONE)
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
    ROM = "rom"
    HW = "hw"              # Hardware-mapped I/O (automatically volatile)


@dataclass
class StorageAttribute(ProcessedAttribute):
    """#[zeropage], #[ram], #[rom], #[hw]"""
    storage_kind: StorageKind = StorageKind.RAM
    address: Optional[int] = None  # For explicit addresses
    is_register: bool = False  # True if marked as scratch register with 'register' parameter


# Bank attribute
@dataclass
class BankAttribute(ProcessedAttribute):
    """#[bank(n)] - specifies which ROM bank the function is placed in"""
    bank_number: int = 0


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
            elif attr.name in ['zeropage', 'lowram', 'ram', 'rom', 'hw']:
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
            else:
                raise HIRError(f"Unknown attribute '{attr.name}'")

        return processed

    def _process_mode(self, attr: ast.Attribute, context: str) -> ModeAttribute:
        """Process #[mode(...)] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[mode] attribute only valid on functions")

        m_mode = None
        x_mode = None
        transition = ModeTransition.NONE
        databank = DataBankMode.NONE

        for arg in attr.args:
            if arg.name is None:  # Positional argument
                value_str = self._get_arg_identifier(arg.value)

                if value_str == 'm8':
                    m_mode = MMode.M8
                elif value_str == 'm16':
                    m_mode = MMode.M16
                elif value_str == 'x8':
                    x_mode = XMode.X8
                elif value_str == 'x16':
                    x_mode = XMode.X16
                else:
                    raise HIRError(f"Invalid mode value: {value_str}")
            elif arg.name == 'transition':
                value_str = self._get_arg_identifier(arg.value)

                if value_str == 'none':
                    transition = ModeTransition.NONE
                elif value_str == 'inline':
                    transition = ModeTransition.INLINE
                elif value_str == 'caller':
                    transition = ModeTransition.CALLER
                else:
                    raise HIRError(f"Invalid transition value: {value_str}")
            elif arg.name == 'databank':
                value_str = self._get_arg_identifier(arg.value)

                if value_str == 'none':
                    databank = DataBankMode.NONE
                elif value_str == 'inline':
                    databank = DataBankMode.INLINE
                elif value_str == 'caller':
                    databank = DataBankMode.CALLER
                else:
                    raise HIRError(f"Invalid databank value: {value_str}")
            else:
                raise HIRError(f"Unknown argument to #[mode]: {arg.name}")

        return ModeAttribute(
            name='mode',
            m_mode=m_mode,
            x_mode=x_mode,
            transition=transition,
            databank=databank
        )

    def _process_preserves(self, attr: ast.Attribute, context: str) -> PreservesAttribute:
        """Process #[preserves(...)] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[preserves] attribute only valid on functions")

        registers = []
        valid_registers = {'A', 'X', 'Y', 'STATUS', 'D', 'DBR', 'S'}

        for arg in attr.args:
            if arg.name is not None:
                raise HIRError(f"#[preserves] does not accept named arguments")

            # Get register name
            if isinstance(arg.value, ast.Register):
                reg_name = arg.value.name
            elif isinstance(arg.value, ast.Identifier):
                reg_name = arg.value.name
            else:
                raise HIRError(f"#[preserves] expects register names")

            if reg_name == 'B':
                raise HIRError(
                    f"B register not allowed in preserves attribute\n"
                    f"  B cannot be preserved separately from A\n"
                    f"  B is the high byte of the A register"
                )

            if reg_name not in valid_registers:
                raise HIRError(f"Invalid register in #[preserves]: {reg_name}")

            if reg_name == 'PBR':
                raise HIRError(f"PBR cannot be in #[preserves] (it's read-only)")

            registers.append(reg_name)

        return PreservesAttribute(
            name='preserves',
            registers=registers
        )

    def _process_storage(self, attr: ast.Attribute, context: str) -> StorageAttribute:
        """Process #[zeropage], #[ram], #[rom], #[hw] attributes."""
        if context not in ['static']:
            raise HIRError(f"#{attr.name} attribute only valid on static variables")

        # Map attribute name to storage kind
        storage_map = {
            'zeropage': StorageKind.ZEROPAGE,
            'lowram': StorageKind.LOWRAM,
            'ram': StorageKind.RAM,
            'rom': StorageKind.ROM,
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
                    raise HIRError(f"#{attr.name} can only have one positional argument (address)")
                elif isinstance(arg.value, ast.IntegerLiteral):
                    # Address argument
                    address = arg.value.value
                else:
                    raise HIRError(f"#{attr.name} address must be an integer literal")

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
                    raise HIRError(f"#{attr.name} register parameter must be boolean or omitted")

            else:
                raise HIRError(f"Unknown argument to #{attr.name}: {arg.name}")

        return StorageAttribute(
            name=attr.name,
            storage_kind=storage_kind,
            address=address,
            is_register=is_register
        )

    def _process_bank(self, attr: ast.Attribute, context: str) -> BankAttribute:
        """Process #[bank(n)] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[bank] attribute only valid on functions")

        bank_number = None

        for arg in attr.args:
            if arg.name is None:  # Positional argument (bank number)
                if bank_number is not None:
                    raise HIRError(f"#[bank] can only have one positional argument (bank number)")

                if isinstance(arg.value, ast.IntegerLiteral):
                    bank_number = arg.value.value
                else:
                    raise HIRError(f"#[bank] number must be an integer literal")
            else:
                raise HIRError(f"Unknown argument to #[bank]: {arg.name}")

        if bank_number is None:
            raise HIRError(f"#[bank] requires a bank number")

        return BankAttribute(
            name='bank',
            bank_number=bank_number
        )

    def _process_interrupt(self, attr: ast.Attribute, context: str) -> InterruptAttribute:
        """Process #[interrupt(vector, preserve=...)] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[interrupt] attribute only valid on functions")

        vector = None
        preserve = True  # Default

        for arg in attr.args:
            if arg.name is None:  # Positional argument (vector)
                if vector is not None:
                    raise HIRError(f"#[interrupt] can only have one positional argument (vector)")

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
                    raise HIRError(f"Invalid interrupt vector: {value_str}")

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
                        raise HIRError(f"Invalid preserve value: {value_str}")
                else:
                    raise HIRError(f"preserve must be a boolean")
            else:
                raise HIRError(f"Unknown argument to #[interrupt]: {arg.name}")

        if vector is None:
            raise HIRError(f"#[interrupt] requires an interrupt vector")

        return InterruptAttribute(
            name='interrupt',
            vector=vector,
            preserve=preserve
        )

    def _process_entry(self, attr: ast.Attribute, context: str) -> EntryAttribute:
        """Process #[entry] attribute."""
        if context not in ['function']:
            raise HIRError(f"#[entry] attribute only valid on functions")

        if len(attr.args) > 0:
            raise HIRError(f"#[entry] does not accept arguments")

        return EntryAttribute(name='entry')

    def _process_stack(self, attr: ast.Attribute, context: str) -> StackAttribute:
        """Process #[stack(lower, upper)] attribute."""
        if context not in ['static']:
            raise HIRError(f"#[stack] attribute only valid on static declarations")

        lower = None
        upper = None

        for i, arg in enumerate(attr.args):
            if arg.name is not None:
                raise HIRError(f"#[stack] only accepts positional arguments")

            if not isinstance(arg.value, ast.IntegerLiteral):
                raise HIRError(f"#[stack] arguments must be integer literals")

            if i == 0:
                lower = arg.value.value
            elif i == 1:
                upper = arg.value.value
            else:
                raise HIRError(f"#[stack] accepts exactly 2 arguments (lower, upper)")

        if lower is None or upper is None:
            raise HIRError(f"#[stack] requires both lower and upper bounds: #[stack(lower, upper)]")

        if lower > upper:
            raise HIRError(f"Stack lower bound ${lower:04X} must be <= upper bound ${upper:04X}")

        if lower < 0x0000 or upper > 0x1FFF:
            raise HIRError(
                f"Stack region ${lower:04X}-${upper:04X} must be within "
                f"low RAM ($0000-$1FFF)"
            )

        return StackAttribute(
            name='stack',
            lower=lower,
            upper=upper
        )

    def _process_cfg(self, attr: ast.Attribute, context: str) -> CfgAttribute:
        """Process #[cfg(condition)] attribute."""
        if len(attr.args) != 1:
            raise HIRError(f"#[cfg] requires exactly one argument (the condition)")
        
        condition_arg = attr.args[0]
        if not isinstance(condition_arg.value, CfgCondition):
            raise HIRError(f"#[cfg] argument must be a condition expression")
        
        return CfgAttribute(
            name='cfg',
            condition=condition_arg.value
        )

    def _get_arg_identifier(self, value) -> str:
        """Extract identifier name from argument value."""
        if isinstance(value, ast.Identifier):
            return value.name
        else:
            raise HIRError(f"Expected identifier, got {type(value).__name__}")
