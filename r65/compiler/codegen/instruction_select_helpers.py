"""
Instruction selection helper classes.

Extracted from InstructionSelector to improve modularity and testability.
"""

from enum import Enum
from typing import Optional
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.mir.nodes import Immediate
from r65.compiler.errors import InstructionSelectionError


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
        self.emitter.emit_instruction("XBA", comment=comment or "Exchange B and A")

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


class BinaryOpEmitter:
    """
    Emits binary operation instructions.

    Handles arithmetic, bitwise, shift, multiply, and divide operations
    with proper operand formatting.
    """

    # Power-of-2 to shift count mapping
    POWER_OF_2_SHIFTS = {1: 0, 2: 1, 4: 2, 8: 3}

    def __init__(self, emitter: AssemblyEmitter):
        self.emitter = emitter

    def require_immediate(self, operand, operation: str) -> int:
        """Validate operand is immediate and return its value."""
        if not isinstance(operand, Immediate):
            raise InstructionSelectionError(f"{operation} requires constant operand")
        return operand.value

    def emit_repeated(self, mnemonic: str, operand: str, count: int):
        """Emit an instruction repeated count times."""
        for _ in range(count):
            self.emitter.emit_instruction(mnemonic, operand)

    def emit_with_operand(self, mnemonic: str, right_operand, is_u16: bool,
                          format_operand_fn):
        """
        Emit binary operation with formatted operand.

        Args:
            mnemonic: Instruction mnemonic (ADC, SBC, AND, etc.)
            right_operand: Right operand (Immediate or location)
            is_u16: True if 16-bit operation
            format_operand_fn: Function to format operand location
        """
        if isinstance(right_operand, Immediate):
            value = right_operand.value
            if is_u16:
                self.emitter.emit_instruction(mnemonic, f"#${value:04X}")
            else:
                self.emitter.emit_instruction(mnemonic, f"#${value:02X}")
        else:
            # Assume it's a location that needs formatting
            operand_str = format_operand_fn(right_operand)
            self.emitter.emit_instruction(mnemonic, operand_str)

    def emit_add(self, right_operand, is_u16: bool, format_fn):
        """Emit add operation (CLC + ADC)."""
        self.emitter.emit_instruction("CLC")
        self.emit_with_operand("ADC", right_operand, is_u16, format_fn)

    def emit_sub(self, right_operand, is_u16: bool, format_fn):
        """Emit subtract operation (SEC + SBC)."""
        self.emitter.emit_instruction("SEC")
        self.emit_with_operand("SBC", right_operand, is_u16, format_fn)

    def emit_and(self, right_operand, is_u16: bool, format_fn):
        """Emit bitwise AND operation."""
        self.emit_with_operand("AND", right_operand, is_u16, format_fn)

    def emit_or(self, right_operand, is_u16: bool, format_fn):
        """Emit bitwise OR operation."""
        self.emit_with_operand("ORA", right_operand, is_u16, format_fn)

    def emit_xor(self, right_operand, is_u16: bool, format_fn):
        """Emit bitwise XOR operation."""
        self.emit_with_operand("EOR", right_operand, is_u16, format_fn)

    def emit_shift_left(self, right_operand, is_u16: bool):
        """Emit left shift operation (A << count)."""
        count = self.require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self.emitter.emit_instruction("LDA", "#$00",
                comment=f"Shift by {count} >= {bit_width} bits = 0")
            return

        self.emit_repeated("ASL", "A", count)

    def emit_shift_right(self, right_operand, is_u16: bool):
        """Emit right shift operation (A >> count)."""
        count = self.require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self.emitter.emit_instruction("LDA", "#$00",
                comment=f"Shift by {count} >= {bit_width} bits = 0")
            return

        self.emit_repeated("LSR", "A", count)

    def emit_multiply(self, right_operand, is_u16: bool):
        """Emit multiply by power of 2 (A * 1/2/4/8) using ASL."""
        value = self.require_immediate(right_operand, "Multiply")
        shift_count = self.POWER_OF_2_SHIFTS.get(value)

        if shift_count is None:
            raise InstructionSelectionError(
                f"Multiply operator only supports 1, 2, 4, 8 (got {value}). "
                f"Use mul() for general multiplication.")

        self.emit_repeated("ASL", "A", shift_count)

    def emit_divide(self, right_operand, is_u16: bool):
        """Emit divide by power of 2 (A / 1/2/4/8) using LSR."""
        value = self.require_immediate(right_operand, "Divide")
        shift_count = self.POWER_OF_2_SHIFTS.get(value)

        if shift_count is None:
            raise InstructionSelectionError(
                f"Divide operator only supports 1, 2, 4, 8 (got {value}). "
                f"Use div() for general division.")

        self.emit_repeated("LSR", "A", shift_count)
