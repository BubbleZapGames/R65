"""
Instruction selection helper classes.

Extracted from InstructionSelector to improve modularity and testability.
"""

from enum import Enum
from typing import Dict, Optional
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.opcodes import Opcode


# =============================================================================
# Register Instruction Mappings
# =============================================================================

class RegisterMappings:
    """
    Centralized mappings for register-related instructions.

    Provides consistent instruction selection for hardware registers.
    """

    # Load instructions by register
    LOAD = {'A': 'LDA', 'X': 'LDX', 'Y': 'LDY'}

    # Store instructions by register
    STORE = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}

    # Compare instructions by register
    COMPARE = {'A': 'CMP', 'X': 'CPX', 'Y': 'CPY'}

    # Push instructions by register (string mnemonics - legacy)
    PUSH = {'A': 'PHA', 'X': 'PHX', 'Y': 'PHY', 'STATUS': 'PHP', 'D': 'PHD', 'DBR': 'PHB'}

    # Pull instructions by register (string mnemonics - legacy)
    PULL = {'A': 'PLA', 'X': 'PLX', 'Y': 'PLY', 'STATUS': 'PLP', 'D': 'PLD', 'DBR': 'PLB'}

    # Push instructions by register (typed opcodes)
    PUSH_OPCODES = {
        'A': Opcode.PHA, 'X': Opcode.PHX, 'Y': Opcode.PHY,
        'STATUS': Opcode.PHP, 'D': Opcode.PHD, 'DBR': Opcode.PHB
    }

    # Pull instructions by register (typed opcodes)
    PULL_OPCODES = {
        'A': Opcode.PLA, 'X': Opcode.PLX, 'Y': Opcode.PLY,
        'STATUS': Opcode.PLP, 'D': Opcode.PLD, 'DBR': Opcode.PLB
    }

    # Transfer instructions (from, to) → mnemonic
    TRANSFER = {
        ('A', 'X'): 'TAX', ('A', 'Y'): 'TAY',
        ('X', 'A'): 'TXA', ('X', 'Y'): 'TXY',
        ('Y', 'A'): 'TYA', ('Y', 'X'): 'TYX',
        ('A', 'S'): 'TCS', ('S', 'A'): 'TSC',
        ('A', 'D'): 'TCD', ('D', 'A'): 'TDC',
    }

    @classmethod
    def get_load(cls, register: str) -> Optional[str]:
        """Get load instruction for register."""
        return cls.LOAD.get(register)

    @classmethod
    def get_store(cls, register: str) -> Optional[str]:
        """Get store instruction for register."""
        return cls.STORE.get(register)

    @classmethod
    def get_compare(cls, register: str) -> Optional[str]:
        """Get compare instruction for register."""
        return cls.COMPARE.get(register)

    @classmethod
    def get_push(cls, register: str) -> Optional[str]:
        """Get push instruction for register."""
        return cls.PUSH.get(register)

    @classmethod
    def get_pull(cls, register: str) -> Optional[str]:
        """Get pull instruction for register."""
        return cls.PULL.get(register)

    @classmethod
    def get_transfer(cls, from_reg: str, to_reg: str) -> Optional[str]:
        """Get transfer instruction between registers."""
        return cls.TRANSFER.get((from_reg, to_reg))


class XBAState(Enum):
    """
    Tracks the current state of A and B registers after XBA operations.

    NORMAL: A contains its original value, B contains its original value
    SWAPPED: A contains B's original value, B contains A's original value
    UNKNOWN: State is unknown (after branches, calls, or other invalidations)
    """
    NORMAL = "normal"
    SWAPPED = "swapped"
    UNKNOWN = "unknown"


class XBAStateManager:
    """
    Manages XBA (exchange B and A) optimization state.

    Tracks register swapping to avoid redundant XBA instructions
    and optimize B register access patterns.
    """

    def __init__(self, emitter: AssemblyEmitter):
        self.emitter = emitter
        self.state = XBAState.NORMAL

    def invalidate(self):
        """Invalidate XBA state (after branches, calls, etc.)."""
        self.state = XBAState.UNKNOWN

    def emit_xba(self, comment: str = None):
        """Emit XBA instruction and update state tracking."""
        self.emitter._node_emitter.emit_instr(Opcode.XBA, None, comment or "Exchange B and A")

        if self.state == XBAState.NORMAL:
            self.state = XBAState.SWAPPED
        elif self.state == XBAState.SWAPPED:
            self.state = XBAState.NORMAL
        # UNKNOWN stays UNKNOWN

    def ensure_normal(self, comment: str = None):
        """Ensure XBA state is normal (A has original value)."""
        if self.state == XBAState.SWAPPED:
            self.emit_xba(comment or "Restore A from B")
        elif self.state == XBAState.UNKNOWN:
            # Can't optimize - need to assume it might need restoration
            pass

    def ensure_swapped(self, comment: str = None):
        """Ensure XBA state is swapped (A has B's value)."""
        if self.state == XBAState.NORMAL:
            self.emit_xba(comment or "Swap A and B")
        elif self.state == XBAState.UNKNOWN:
            self.emit_xba(comment or "Swap A and B (state unknown)")

    def mark_a_modified(self):
        """Mark that A has been modified (B unaffected)."""
        if self.state == XBAState.SWAPPED:
            # A was holding B's value, now it's modified
            self.state = XBAState.UNKNOWN
        # NORMAL and UNKNOWN stay the same

    def mark_b_modified(self):
        """Mark that B has been modified through XBA pattern."""
        if self.state == XBAState.NORMAL:
            # B was holding its original value, now modified
            # This typically means we did XBA, modified A, XBA
            pass
        # State tracking for B modifications is complex

    def access_b_value_in_a(self):
        """
        Set up to access B's value in A register.

        Call this before operations that need to read B's value.
        """
        self.ensure_swapped("Access B value in A")

    def store_to_b_from_a(self):
        """
        Complete storing A's current value to B register.

        Call this after loading a value into A that should go to B.
        """
        self.emit_xba("Store to B")
        if self.state == XBAState.SWAPPED:
            # Now we have: A=original_A, B=new_value
            self.state = XBAState.NORMAL
