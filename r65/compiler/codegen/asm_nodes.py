# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.errors import SourceLocation

from r65.compiler.codegen.opcodes import (
    Opcode, mnemonic, addressing_mode, instruction_size, is_branch,
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
    """Block move operand for MVN/MVP (src_bank, dst_bank).

    A bank may be an int (literal bank byte) or a str (a symbol name whose
    bank is resolved by WLA-DX via the `:label` operator). The str form is
    used for ROM data tables whose final bank is decided by the linker.
    """
    src_bank: object  # int | str
    dst_bank: object  # int | str

    @staticmethod
    def _fmt(bank) -> str:
        return f":{bank}" if isinstance(bank, str) else f"${bank:02X}"

    def __repr__(self) -> str:
        return f"BlockMove({self._fmt(self.src_bank)}, {self._fmt(self.dst_bank)})"


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
    source_loc: SourceLocation | None = None

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
    source_loc: SourceLocation | None = None

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
        Directive(".DEFINE", ["FOO", "$10"]) -> .DEFINE FOO $10
        Directive(".db", ["$01", "$02"])    -> .db $01, $02

    For mode-tracking directives (`.ACCU` / `.INDEX`), prefer the
    typed `ModeChange` node — it's the single canonical representation
    consumed by the asm-mode dataflow and downstream size/branch
    passes, and it serializes with the indentation those passes
    expect.
    """
    name: str
    args: list[str] = field(default_factory=list)
    source_loc: SourceLocation | None = None

    def __repr__(self) -> str:
        if self.args:
            return f"Directive({self.name}, {self.args})"
        return f"Directive({self.name})"


@dataclass
class ModeChange(AsmNode):
    """A WLA-DX `.ACCU N` / `.INDEX N` mode-tracking directive.

    These directives don't generate any bytes; they tell WLA-DX what
    width to use when sizing accumulator-immediate (m flag) or
    index-immediate (x flag) operands that follow. WLA-DX assembles
    linearly — it doesn't track SEP/REP semantically — so `ModeChange`
    nodes have to appear at every point where the linear assembler
    needs to be told the runtime mode just shifted.

    `flag` is `"ACCU"` (m flag, accumulator width) or `"INDEX"`
    (x flag, index-register width). `bits` is 8 or 16.

    Codegen does NOT scatter these around at SEP/REP sites anymore —
    `r65.compiler.optimize.mode_directive_rewrite.normalize_mode_directives`
    regenerates them from the asm-mode dataflow at the start and end
    of peephole optimization. The single source of truth for runtime
    mode is `compute_modes()` (see asm_mode_dataflow.py), not these
    nodes.
    """
    flag: str   # "ACCU" or "INDEX"
    bits: int   # 8 or 16
    source_loc: SourceLocation | None = None

    def __repr__(self) -> str:
        return f"ModeChange(.{self.flag} {self.bits})"


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
