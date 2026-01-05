"""
Node-based assembly emitter.

Builds a list of AsmNode objects instead of strings. This enables
optimization passes to work on structured data.

Has a similar interface to AssemblyEmitter for easy migration.
"""

from typing import Optional, List

from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Comment, Directive, BlankLine, RawAsm,
    Immediate, Address, StackOffset, BlockMove,
    emit_nodes,
)
from r65.compiler.codegen.opcodes import Opcode


class NodeEmitter:
    """
    Emitter that builds AsmNode objects instead of strings.

    Provides a similar interface to AssemblyEmitter but outputs
    structured nodes that can be manipulated by optimization passes.
    """

    def __init__(self):
        """Initialize node emitter."""
        self.nodes: List[AsmNode] = []

    def clear(self):
        """Clear all emitted nodes."""
        self.nodes = []

    def get_nodes(self) -> List[AsmNode]:
        """Get the list of emitted nodes."""
        return self.nodes

    def to_string(self) -> str:
        """Convert nodes to assembly string."""
        return emit_nodes(self.nodes)

    # ========================================================================
    # Basic Emission
    # ========================================================================

    def emit(self, node: AsmNode):
        """Emit a single node."""
        self.nodes.append(node)

    def emit_blank_line(self):
        """Emit a blank line."""
        self.nodes.append(BlankLine())

    def emit_comment(self, text: str):
        """Emit a comment."""
        self.nodes.append(Comment(text))

    def emit_section_header(self, title: str):
        """Emit a section header comment."""
        self.nodes.append(Comment(title, section_header=True))

    def emit_label(self, name: str):
        """Emit a label."""
        self.nodes.append(Label(name))

    # ========================================================================
    # Instructions - Structured API
    # ========================================================================

    def emit_instr(self, opcode: Opcode, operand=None, comment: str = None):
        """
        Emit an instruction using structured Opcode enum.

        Args:
            opcode: The Opcode enum value (e.g., Opcode.LDA_IMMEDIATE)
            operand: The operand (Immediate, Address, StackOffset, etc.)
            comment: Optional comment
        """
        self.nodes.append(Instruction(opcode, operand, comment))

    # ========================================================================
    # Instructions - String API (for migration compatibility)
    # ========================================================================

    def emit_instruction(self, mnemonic: str, operand: Optional[str] = None,
                         comment: Optional[str] = None):
        """
        Emit an instruction from string mnemonic and operand.

        This is a compatibility layer for migrating from AssemblyEmitter.
        It parses the strings and creates structured Instruction nodes.

        Args:
            mnemonic: Instruction mnemonic (e.g., "LDA", "STA", "BEQ")
            operand: Optional operand string (e.g., "#$0A", "$10", "label")
            comment: Optional comment
        """
        # Parse mnemonic and operand into Opcode + Operand
        opcode, op = self._parse_instruction(mnemonic, operand)

        if opcode is not None:
            self.nodes.append(Instruction(opcode, op, comment))
        else:
            # Fallback: emit as RawAsm if we can't parse it
            if operand:
                self.nodes.append(RawAsm(f"{mnemonic} {operand}"))
            else:
                self.nodes.append(RawAsm(mnemonic))

    def _parse_instruction(self, mnemonic: str, operand_str: Optional[str]):
        """
        Parse string instruction into Opcode and Operand.

        Returns (Opcode, Operand) or (None, None) if unparseable.
        """
        mnem_upper = mnemonic.upper().strip()

        # Handle implied addressing (no operand)
        if operand_str is None or operand_str.strip() == "":
            opcode = self._lookup_opcode(mnem_upper, None)
            return opcode, None

        operand_str = operand_str.strip()

        # Determine addressing mode from operand syntax
        mode, operand = self._parse_operand(operand_str)

        # Look up the opcode
        opcode = self._lookup_opcode(mnem_upper, mode)

        return opcode, operand

    def _parse_operand(self, operand_str: str):
        """
        Parse operand string to determine addressing mode and value.

        Returns (mode_string, Operand) where mode_string matches the
        Opcode enum suffix (e.g., "IMMEDIATE", "DP", "ABSOLUTE").
        """
        s = operand_str.strip()

        # Immediate: #$XX, #value, #label
        if s.startswith('#'):
            value = self._parse_value(s[1:])
            return "IMMEDIATE", Immediate(value)

        # Stack relative: $XX,S or (XX,S),Y
        if s.endswith(',S'):
            addr_part = s[:-2].strip()
            value = self._parse_value(addr_part)
            return "STACK", StackOffset(value if isinstance(value, int) else 0)

        if s.endswith(',S),Y'):
            # Stack indirect indexed: ($XX,S),Y
            inner = s[1:-5]  # Remove ( and ,S),Y
            value = self._parse_value(inner)
            return "STACK_INDIRECT_Y", StackOffset(value if isinstance(value, int) else 0)

        # Indirect long: [$XX] or [$XX],Y
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]
            value = self._parse_value(inner)
            return "DP_INDIRECT_LONG", Address(value)

        if s.startswith('[') and s.endswith('],Y'):
            inner = s[1:-3]
            value = self._parse_value(inner)
            return "DP_INDIRECT_LONG_Y", Address(value)

        # Indirect: ($XX), ($XX,X), ($XX),Y
        if s.startswith('('):
            if s.endswith(',X)'):
                inner = s[1:-3]
                value = self._parse_value(inner)
                return "DP_INDIRECT_X", Address(value)
            elif s.endswith('),Y'):
                inner = s[1:-3]
                value = self._parse_value(inner)
                return "DP_INDIRECT_Y", Address(value)
            elif s.endswith(')'):
                inner = s[1:-1]
                value = self._parse_value(inner)
                # Could be DP_INDIRECT or INDIRECT depending on address size
                if isinstance(value, int) and value < 0x100:
                    return "DP_INDIRECT", Address(value)
                else:
                    return "INDIRECT", Address(value)

        # Indexed: $XXXX,X or $XXXX,Y or $XX,X or $XX,Y
        if s.endswith(',X'):
            addr_part = s[:-2].strip()
            value = self._parse_value(addr_part)
            if isinstance(value, int):
                if value < 0x100:
                    return "DP_X", Address(value)
                elif value < 0x10000:
                    return "ABSOLUTE_X", Address(value)
                else:
                    return "LONG_X", Address(value)
            else:
                # Label - assume absolute
                return "ABSOLUTE_X", Address(value)

        if s.endswith(',Y'):
            addr_part = s[:-2].strip()
            value = self._parse_value(addr_part)
            if isinstance(value, int):
                if value < 0x100:
                    return "DP_Y", Address(value)
                else:
                    return "ABSOLUTE_Y", Address(value)
            else:
                return "ABSOLUTE_Y", Address(value)

        # Direct/Absolute: $XX, $XXXX, $XXXXXX, or label
        value = self._parse_value(s)
        if isinstance(value, int):
            if value < 0x100:
                return "DP", Address(value)
            elif value < 0x10000:
                return "ABSOLUTE", Address(value)
            else:
                return "LONG", Address(value)
        else:
            # Label reference - assume absolute
            return "ABSOLUTE", Address(value)

    def _parse_value(self, s: str):
        """
        Parse a value string to int or label string.

        Handles: $XX, $XXXX, decimal, label, <label, >label
        """
        s = s.strip()

        if not s:
            return 0

        # Hex: $XX or $XXXX
        if s.startswith('$'):
            try:
                return int(s[1:], 16)
            except ValueError:
                return s  # Return as label if not valid hex

        # Decimal
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)

        # Low/high byte operators
        if s.startswith('<') or s.startswith('>'):
            return s  # Keep as expression string

        # Label or expression
        return s

    def _lookup_opcode(self, mnemonic: str, mode: Optional[str]) -> Optional[Opcode]:
        """
        Look up Opcode enum from mnemonic and addressing mode.

        Args:
            mnemonic: Upper-case mnemonic (e.g., "LDA")
            mode: Addressing mode string (e.g., "IMMEDIATE", "DP") or None for implied

        Returns:
            Opcode enum value or None if not found
        """
        # Branch instructions have no mode suffix - they're just BEQ, BNE, etc.
        # but they take an operand (the target label/address)
        if mnemonic in ('BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS',
                        'BRA', 'BRL'):
            try:
                return Opcode[mnemonic]
            except KeyError:
                return None

        # JSR and JSL are call instructions - JSR is absolute, JSL is long
        if mnemonic == 'JSR':
            return Opcode.JSR
        if mnemonic == 'JSL':
            return Opcode.JSL

        if mode is None:
            # Implied addressing - just the mnemonic
            name = mnemonic
        else:
            # Mnemonic_MODE format
            name = f"{mnemonic}_{mode}"

        try:
            return Opcode[name]
        except KeyError:
            return None

    # ========================================================================
    # Directives
    # ========================================================================

    def emit_directive(self, name: str, *args: str):
        """Emit an assembler directive."""
        self.nodes.append(Directive(name, list(args)))

    def emit_define(self, name: str, value: int, comment: Optional[str] = None):
        """Emit a .DEFINE directive."""
        if comment:
            self.nodes.append(Comment(comment))
        self.nodes.append(Directive(".DEFINE", [name, f"${value:04X}"]))

    def emit_accu_mode(self, bits: int):
        """Emit .ACCU directive."""
        self.nodes.append(Directive(".ACCU", [str(bits)]))

    def emit_index_mode(self, bits: int):
        """Emit .INDEX directive."""
        self.nodes.append(Directive(".INDEX", [str(bits)]))

    # ========================================================================
    # Raw Assembly (for asm! blocks)
    # ========================================================================

    def emit_raw(self, text: str):
        """Emit raw assembly text."""
        self.nodes.append(RawAsm(text))
