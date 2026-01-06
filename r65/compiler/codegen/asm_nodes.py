"""
Assembly output node types for structured code generation.

Instead of emitting strings directly, the code generator builds a list of
AsmNode objects. This enables:
- Pattern matching in optimization passes (no string parsing)
- Easy instruction size calculation
- Clean separation between instruction selection and text emission

The final emission step converts nodes to WLA-DX assembly text.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

from r65.compiler.codegen.opcodes import (
    Opcode, mnemonic, addressing_mode, instruction_size,
    is_branch, BRANCH_OPCODES,
)


# ============================================================================
# Operand Types
# ============================================================================

@dataclass(frozen=True)
class Immediate:
    """
    Immediate value operand (#value).

    value can be:
    - int: literal value (#$0A, #10)
    - str: label or expression (#LABEL, #<LABEL, #>LABEL)
    """
    value: int | str

    def __repr__(self) -> str:
        if isinstance(self.value, int):
            return f"Immediate(${self.value:02X})"
        return f"Immediate({self.value})"


@dataclass(frozen=True)
class Address:
    """
    Memory address operand (absolute or direct page).

    value can be:
    - int: literal address ($1234, $10)
    - str: label reference
    """
    value: int | str

    def __repr__(self) -> str:
        if isinstance(self.value, int):
            return f"Address(${self.value:04X})"
        return f"Address({self.value})"


@dataclass(frozen=True)
class StackOffset:
    """Stack-relative addressing operand (offset,S)."""
    offset: int

    def __repr__(self) -> str:
        return f"StackOffset(${self.offset:02X},S)"


@dataclass(frozen=True)
class BlockMove:
    """Block move operand for MVN/MVP (src_bank, dst_bank)."""
    src_bank: int
    dst_bank: int

    def __repr__(self) -> str:
        return f"BlockMove(${self.src_bank:02X}, ${self.dst_bank:02X})"


# Union of all operand types
Operand = Immediate | Address | StackOffset | BlockMove


# ============================================================================
# Assembly Node Types
# ============================================================================

@dataclass
class AsmNode:
    """Base class for all assembly output nodes."""
    pass


@dataclass
class Instruction(AsmNode):
    """
    A machine instruction.

    The opcode encodes both the mnemonic and addressing mode.
    The operand provides the value/address for non-implied instructions.
    """
    opcode: Opcode
    operand: Operand | None = None
    comment: str | None = None

    def size(self, m16: bool = False, x16: bool = False) -> int:
        """Get instruction size in bytes."""
        return instruction_size(self.opcode, m16, x16)

    def mnemonic(self) -> str:
        """Get the instruction mnemonic (e.g., 'LDA')."""
        return mnemonic(self.opcode)

    def addressing_mode(self) -> str | None:
        """Get the addressing mode (e.g., 'IMMEDIATE') or None for implied."""
        return addressing_mode(self.opcode)

    def is_branch(self) -> bool:
        """Check if this is a branch instruction."""
        return is_branch(self.opcode)

    def __repr__(self) -> str:
        if self.operand:
            return f"Instruction({self.opcode.name}, {self.operand})"
        return f"Instruction({self.opcode.name})"


@dataclass
class Label(AsmNode):
    """A label definition."""
    name: str

    def __repr__(self) -> str:
        return f"Label({self.name})"


@dataclass
class Comment(AsmNode):
    """
    A comment line.

    section_header=True generates a decorated section header.
    """
    text: str
    section_header: bool = False

    def __repr__(self) -> str:
        if self.section_header:
            return f"Comment(HEADER: {self.text})"
        return f"Comment({self.text})"


@dataclass
class Directive(AsmNode):
    """
    An assembler directive.

    Examples:
        Directive(".ACCU", ["8"])           -> .ACCU 8
        Directive(".DEFINE", ["FOO", "$10"]) -> .DEFINE FOO $10
        Directive(".db", ["$01", "$02"])    -> .db $01, $02
    """
    name: str
    args: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.args:
            return f"Directive({self.name}, {self.args})"
        return f"Directive({self.name})"


@dataclass
class BlankLine(AsmNode):
    """An empty line for formatting."""

    def __repr__(self) -> str:
        return "BlankLine()"


@dataclass
class RawAsm(AsmNode):
    """
    Raw assembly text (for inline asm! or special cases).

    Used when structured representation isn't appropriate.
    """
    text: str

    def __repr__(self) -> str:
        return f"RawAsm({self.text})"


# ============================================================================
# Emission to WLA-DX Assembly
# ============================================================================

def emit_node(node: AsmNode, indent: str = "    ") -> str:
    """
    Convert a single AsmNode to WLA-DX assembly text.

    Args:
        node: The node to emit
        indent: Indentation for instructions (default 4 spaces)

    Returns:
        Assembly text for this node
    """
    match node:
        case Instruction(opcode, operand, comment):
            text = f"{indent}{_emit_instruction(opcode, operand)}"
            if comment:
                text += f"  ; {comment}"
            return text

        case Label(name):
            return f"{name}:"

        case Comment(text, section_header=True):
            line = "=" * 76
            return f"; {line}\n; {text}\n; {line}"

        case Comment(text, section_header=False):
            return f"; {text}"

        case Directive(name, args):
            if args:
                return f"{name} {', '.join(args)}"
            return name

        case BlankLine():
            return ""

        case RawAsm(text):
            # Raw assembly is emitted as-is without additional indent
            return text

        case _:
            raise ValueError(f"Unknown node type: {type(node)}")


def emit_nodes(nodes: list[AsmNode], indent: str = "    ") -> str:
    """
    Convert a list of AsmNodes to WLA-DX assembly text.

    Args:
        nodes: List of nodes to emit
        indent: Indentation for instructions

    Returns:
        Complete assembly text with newlines
    """
    return '\n'.join(emit_node(n, indent) for n in nodes)


def _emit_instruction(opcode: Opcode, operand: Operand | None) -> str:
    """Format an instruction with its operand."""
    mnem = mnemonic(opcode)
    mode = addressing_mode(opcode)

    # Branch instructions: no mode suffix but take an operand
    if opcode in BRANCH_OPCODES:
        if operand is None:
            return mnem  # Shouldn't happen
        match operand:
            case Address(value):
                return f"{mnem} {_format_value(value)}"
            case _:
                return f"{mnem} {operand}"

    # Call instructions: JSR/JSL have no mode suffix but take an operand
    if opcode in (Opcode.JSR, Opcode.JSL):
        if operand is None:
            return mnem
        match operand:
            case Address(value):
                return f"{mnem} {_format_value(value)}"
            case _:
                return f"{mnem} {operand}"

    # Accumulator mode instructions (need explicit 'A' operand in WLA-DX)
    ACCUMULATOR_OPCODES = {Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR, Opcode.INC, Opcode.DEC}
    if opcode in ACCUMULATOR_OPCODES:
        return f"{mnem} A"

    # Implied addressing (no operand)
    if mode is None:
        return mnem

    # Format operand based on addressing mode
    if operand is None:
        # Shouldn't happen for non-implied, but handle gracefully
        return mnem

    match (mode, operand):
        # Immediate
        case ("IMMEDIATE", Immediate(value)):
            return f"{mnem} #{_format_value(value)}"

        # Direct Page
        case ("DP", Address(value)):
            return f"{mnem} {_format_value(value)}"
        case ("DP_X", Address(value)):
            return f"{mnem} {_format_value(value)},X"
        case ("DP_Y", Address(value)):
            return f"{mnem} {_format_value(value)},Y"

        # Absolute
        case ("ABSOLUTE", Address(value)):
            return f"{mnem} {_format_value(value)}"
        case ("ABSOLUTE_X", Address(value)):
            return f"{mnem} {_format_value(value)},X"
        case ("ABSOLUTE_Y", Address(value)):
            return f"{mnem} {_format_value(value)},Y"

        # Long (24-bit)
        case ("LONG", Address(value)):
            return f"{mnem} {_format_long_value(value)}"
        case ("LONG_X", Address(value)):
            return f"{mnem} {_format_long_value(value)},X"

        # Indirect
        case ("INDIRECT", Address(value)):
            return f"{mnem} ({_format_value(value)})"
        case ("INDIRECT_X", Address(value)):
            return f"{mnem} ({_format_value(value)},X)"
        case ("INDIRECT_LONG", Address(value)):
            return f"{mnem} [{_format_value(value)}]"

        # Direct Page Indirect
        case ("DP_INDIRECT", Address(value)):
            return f"{mnem} ({_format_value(value)})"
        case ("DP_INDIRECT_X", Address(value)):
            return f"{mnem} ({_format_value(value)},X)"
        case ("DP_INDIRECT_Y", Address(value)):
            return f"{mnem} ({_format_value(value)}),Y"
        case ("DP_INDIRECT_LONG", Address(value)):
            return f"{mnem} [{_format_value(value)}]"
        case ("DP_INDIRECT_LONG_Y", Address(value)):
            return f"{mnem} [{_format_value(value)}],Y"

        # Stack Relative
        case ("STACK", StackOffset(offset)):
            return f"{mnem} {_format_value(offset)},S"
        case ("STACK_INDIRECT_Y", StackOffset(offset)):
            return f"{mnem} ({_format_value(offset)},S),Y"

        # Block Move
        case (_, BlockMove(src, dst)):
            return f"{mnem} ${src:02X}, ${dst:02X}"

        case _:
            # Fallback for unhandled cases
            return f"{mnem} {operand}"


def _format_value(value: int | str) -> str:
    """Format an immediate or address value."""
    if isinstance(value, str):
        return value
    if value < 0x100:
        return f"${value:02X}"
    return f"${value:04X}"


def _format_long_value(value: int | str) -> str:
    """Format a 24-bit long address value."""
    if isinstance(value, str):
        return value
    return f"${value:06X}"


# ============================================================================
# Branch Inversion (for long branch fixup)
# ============================================================================

BRANCH_INVERSIONS: dict[Opcode, Opcode] = {
    Opcode.BEQ: Opcode.BNE,
    Opcode.BNE: Opcode.BEQ,
    Opcode.BCC: Opcode.BCS,
    Opcode.BCS: Opcode.BCC,
    Opcode.BMI: Opcode.BPL,
    Opcode.BPL: Opcode.BMI,
    Opcode.BVC: Opcode.BVS,
    Opcode.BVS: Opcode.BVC,
}


def invert_branch(opcode: Opcode) -> Opcode | None:
    """
    Get the inverted branch opcode for long branch fixup.

    Returns None if the opcode cannot be inverted (e.g., BRA, BRL).
    """
    return BRANCH_INVERSIONS.get(opcode)
