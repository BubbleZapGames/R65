"""
Instruction selection helper classes.

Extracted from InstructionSelector to improve modularity and testability.
"""

from enum import Enum
from typing import Optional, Dict
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.mir.nodes import Immediate
from r65.compiler.errors import InstructionSelectionError


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

    # Push instructions by register
    PUSH = {'A': 'PHA', 'X': 'PHX', 'Y': 'PHY', 'STATUS': 'PHP', 'D': 'PHD', 'DBR': 'PHB'}

    # Pull instructions by register
    PULL = {'A': 'PLA', 'X': 'PLX', 'Y': 'PLY', 'STATUS': 'PLP', 'D': 'PLD', 'DBR': 'PLB'}

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


# =============================================================================
# Branch Instruction Helpers
# =============================================================================

class BranchEmitter:
    """
    Emits conditional branch instructions.

    Handles the common pattern of:
    - Simple branch + fallthrough JMP
    - Complex signed comparison patterns
    """

    # Simple comparison → branch instruction mapping (unsigned)
    UNSIGNED_BRANCHES = {
        '==': 'BEQ',
        '!=': 'BNE',
        '<': 'BCC',
        '>=': 'BCS',
        'bit7_set': 'BMI',
        'bit7_clear': 'BPL',
        'bit6_set': 'BVS',
        'bit6_clear': 'BVC',
    }

    # Comments for branch instructions
    BRANCH_COMMENTS = {
        '==': "Branch if equal",
        '!=': "Branch if not equal",
        '<': "Branch if less than (unsigned)",
        '>=': "Branch if >= (unsigned)",
        'bit7_set': "Branch if bit 7 set",
        'bit7_clear': "Branch if bit 7 clear",
        'bit6_set': "Branch if bit 6 set",
        'bit6_clear': "Branch if bit 6 clear",
    }

    def __init__(self, emitter: AssemblyEmitter, label_counter_fn):
        """
        Initialize branch emitter.

        Args:
            emitter: Assembly emitter
            label_counter_fn: Function to get unique labels
        """
        self.emitter = emitter
        self.get_unique_label = label_counter_fn

    def emit_simple_branch(self, branch_instr: str, true_target: int,
                           false_target: int, comment: Optional[str] = None):
        """
        Emit simple branch + fallthrough JMP pattern.

        Args:
            branch_instr: Branch instruction (BEQ, BNE, BCC, etc.)
            true_target: Block ID for true branch
            false_target: Block ID for false branch
            comment: Optional comment
        """
        self.emitter.emit_instruction(branch_instr, f"__L{true_target}", comment)
        self.emitter.emit_instruction("JMP", f"__L{false_target}")

    def emit_unsigned_comparison(self, comparison: str, true_target: int, false_target: int):
        """
        Emit unsigned comparison branch.

        Args:
            comparison: Comparison operator ('==', '!=', '<', '>=', '>', '<=')
            true_target: Block ID for true branch
            false_target: Block ID for false branch
        """
        if comparison in ('>', '<='):
            # Compound comparisons need special handling
            if comparison == '>':
                # Unsigned >: (C set) AND (Z clear)
                self.emitter.emit_instruction("BEQ", f"__L{false_target}", "Skip if equal")
                self.emitter.emit_instruction("BCS", f"__L{true_target}", "Branch if > (unsigned)")
                self.emitter.emit_instruction("JMP", f"__L{false_target}")
            else:  # <=
                # Unsigned <=: (C clear) OR (Z set)
                self.emitter.emit_instruction("BEQ", f"__L{true_target}", "Branch if equal")
                self.emitter.emit_instruction("BCC", f"__L{true_target}", "Branch if less than")
                self.emitter.emit_instruction("JMP", f"__L{false_target}")
        else:
            branch_instr = self.UNSIGNED_BRANCHES.get(comparison)
            comment = self.BRANCH_COMMENTS.get(comparison)
            self.emit_simple_branch(branch_instr, true_target, false_target, comment)

    def _emit_signed_xor_setup(self) -> str:
        """
        Emit the signed comparison N XOR V setup pattern.

        Uses BVC/EOR trick to compute N XOR V.

        Returns:
            Label used for skip
        """
        label = self.get_unique_label()
        self.emitter.emit_instruction("BVC", label, "Skip if no overflow")
        self.emitter.emit_instruction("EOR", "#$80", "Flip sign bit if overflow")
        self.emitter.emit_label(label)
        return label

    def emit_signed_comparison(self, comparison: str, true_target: int, false_target: int):
        """
        Emit signed comparison branch.

        Args:
            comparison: Comparison operator ('<', '>=', '>', '<=')
            true_target: Block ID for true branch
            false_target: Block ID for false branch
        """
        if comparison == '<':
            # Signed less than: N XOR V = 1
            self._emit_signed_xor_setup()
            self.emitter.emit_instruction("BMI", f"__L{true_target}", "Branch if less than (signed)")
            self.emitter.emit_instruction("JMP", f"__L{false_target}")
        elif comparison == '>=':
            # Signed >= : N XOR V = 0
            self._emit_signed_xor_setup()
            self.emitter.emit_instruction("BPL", f"__L{true_target}", "Branch if >= (signed)")
            self.emitter.emit_instruction("JMP", f"__L{false_target}")
        elif comparison == '>':
            # Signed >: (N XOR V = 0) AND Z = 0
            self.emitter.emit_instruction("BEQ", f"__L{false_target}", "Skip if equal")
            self._emit_signed_xor_setup()
            self.emitter.emit_instruction("BPL", f"__L{true_target}", "Branch if > (signed)")
            self.emitter.emit_instruction("JMP", f"__L{false_target}")
        elif comparison == '<=':
            # Signed <=: (N XOR V = 1) OR Z = 1
            self.emitter.emit_instruction("BEQ", f"__L{true_target}", "Branch if equal")
            self._emit_signed_xor_setup()
            self.emitter.emit_instruction("BMI", f"__L{true_target}", "Branch if <= (signed)")
            self.emitter.emit_instruction("JMP", f"__L{false_target}")

    def emit_comparison_branch(self, comparison: str, is_signed: bool,
                                true_target: int, false_target: int):
        """
        Emit comparison branch, choosing signed or unsigned path.

        Args:
            comparison: Comparison operator
            is_signed: True for signed comparison
            true_target: Block ID for true branch
            false_target: Block ID for false branch
        """
        # Bit tests and equality don't care about signed/unsigned
        if comparison in ('==', '!=', 'bit7_set', 'bit7_clear', 'bit6_set', 'bit6_clear'):
            self.emit_unsigned_comparison(comparison, true_target, false_target)
        elif is_signed:
            self.emit_signed_comparison(comparison, true_target, false_target)
        else:
            self.emit_unsigned_comparison(comparison, true_target, false_target)
