"""
Base class for instruction selector components.

Provides common functionality shared across all selector classes:
- Parent reference and emitter property
- Common emit helper methods
"""

from typing import TYPE_CHECKING

from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector
    from r65.compiler.codegen.emitter import AssemblyEmitter


class BaseSelector:
    """
    Base class for instruction selector components.

    Provides common emit helpers and parent reference pattern used by
    all selector classes (CallSelector, CompareSelector, etc.).
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize selector.

        Args:
            parent: Parent instruction selector (for emitter and helper access)
        """
        self.parent = parent

    @property
    def emitter(self) -> 'AssemblyEmitter':
        """Get the assembly emitter from parent."""
        return self.parent.emitter

    # ========================================================================
    # Common Emit Helpers
    # ========================================================================

    def _emit_instr(self, opcode: Opcode, operand=None, comment: str | None = None):
        """Emit an instruction with optional operand and comment."""
        self.emitter.emit_instr(opcode, operand, comment)

    def _emit_implied(self, opcode: Opcode, comment: str | None = None):
        """Emit an implied addressing mode instruction (no operand)."""
        self.emitter.emit_instr(opcode, None, comment)

    def _emit_immediate(self, opcode: Opcode, value: int, comment: str | None = None):
        """Emit an immediate addressing mode instruction."""
        self.emitter.emit_instr(opcode, Immediate(value), comment)

    def _emit_address(self, opcode: Opcode, addr: int | str, comment: str | None = None):
        """Emit an instruction with an address operand (absolute, DP, or label)."""
        self.emitter.emit_instr(opcode, Address(addr), comment)

    def _emit_branch(self, opcode: Opcode, label: str, comment: str | None = None):
        """Emit a branch instruction to a label."""
        self.emitter.emit_instr(opcode, Address(label), comment)

    def _emit_jump(self, opcode: Opcode, label: str, comment: str | None = None):
        """Emit a jump instruction to a label."""
        self.emitter.emit_instr(opcode, Address(label), comment)
