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
)
from r65.compiler.codegen.emitter import emit_nodes
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
