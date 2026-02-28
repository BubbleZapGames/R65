"""
Instruction selection helper classes.

Extracted from InstructionSelector to improve modularity and testability.
"""

from enum import Enum
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.opcodes import Opcode


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
        self.emitter.emit_instr(Opcode.XBA, None, comment or "Exchange B and A")

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
