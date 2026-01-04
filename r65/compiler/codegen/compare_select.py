"""
Compare operation selector: Compare, BitTest, Rotate instructions.

Handles comparison and bit manipulation instruction generation
for the 65816 processor.
"""

from typing import TYPE_CHECKING
from r65.compiler.mir.nodes import Compare, BitTest, Rotate, Immediate
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector


class CompareSelector:
    """
    Handles compare and bit operation instruction selection.

    Manages generation of comparison instructions (CMP, CPX, CPY),
    bit test instructions (BIT), and rotate instructions (ROL, ROR).
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize compare selector.

        Args:
            parent: Parent instruction selector (for helper method access)
        """
        self.parent = parent

    @property
    def emitter(self):
        return self.parent.emitter

    # ========================================================================
    # Compare Instruction
    # ========================================================================

    def select_compare(self, instr: Compare):
        """
        Generate code for Compare instruction.

        Emits CMP/CPX/CPY instruction and sets processor flags for subsequent
        conditional branch.

        Args:
            instr: Compare instruction
        """
        # Store type info for subsequent CondBranch (for signed/unsigned detection)
        self.parent.last_comparison_type = instr.type_info

        left_loc = self.parent._get_operand_location(instr.left)
        right_operand = self._prepare_right_operand(instr.right)

        # Emit appropriate comparison instruction based on left operand
        self._emit_comparison(left_loc, right_operand)

    def _prepare_right_operand(self, right) -> str:
        """
        Prepare right operand for comparison.

        Hardware registers must be stored to temp location first.

        Args:
            right: Right operand

        Returns:
            Formatted operand string
        """
        if isinstance(right, Immediate):
            return f"#${right.value:02X}"

        right_loc = self.parent._get_operand_location(right)

        if right_loc.kind == LocationKind.HARDWARE:
            return self._store_hw_register_to_temp(right_loc)

        return self.parent._format_operand(right_loc)

    def _store_hw_register_to_temp(self, right_loc) -> str:
        """Store hardware register to temp location for comparison."""
        if right_loc.hw_register == 'B':
            self.parent._access_b_value_in_a()
            self.emitter.emit_instruction("STA", "$00", "Store B to temp")
            self.parent._ensure_xba_state_normal("Restore A")
            return "$00"
        elif right_loc.hw_register in ['A', 'X', 'Y']:
            store_instr = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}[right_loc.hw_register]
            self.emitter.emit_instruction(store_instr, "$00", f"Store {right_loc.hw_register} to temp")
            return "$00"
        else:
            raise InstructionSelectionError(f"Unsupported hardware register: {right_loc.hw_register}")

    def _emit_comparison(self, left_loc, right_operand: str):
        """
        Emit appropriate comparison instruction.

        Args:
            left_loc: Left operand location
            right_operand: Formatted right operand string
        """
        if left_loc.kind == LocationKind.HARDWARE:
            self._emit_hw_register_comparison(left_loc, right_operand)
        else:
            # Memory or virtual register - load to A and compare
            self.emitter.emit_instruction("LDA", self.parent._format_operand(left_loc))
            self.emitter.emit_instruction("CMP", right_operand)

    def _emit_hw_register_comparison(self, left_loc, right_operand: str):
        """Emit comparison for hardware register left operand."""
        reg = left_loc.hw_register

        if reg == 'X':
            self.emitter.emit_instruction("CPX", right_operand)
        elif reg == 'Y':
            self.emitter.emit_instruction("CPY", right_operand)
        elif reg == 'A':
            self.emitter.emit_instruction("CMP", right_operand)
        elif reg == 'B':
            # B register - transfer to A and compare
            self.parent._access_b_value_in_a()
            self.emitter.emit_instruction("CMP", right_operand)
            # Note: Don't restore A since this is just a comparison
            # State is now SWAPPED (A=B, B=A)
        else:
            raise InstructionSelectionError(
                f"Unsupported hardware register for comparison: {reg}")

    # ========================================================================
    # Bit Test Instruction
    # ========================================================================

    def select_bit_test(self, instr: BitTest):
        """
        Generate code for BitTest instruction using BIT instruction.

        BIT instruction sets flags based on memory value:
        - N flag = bit 7 of memory
        - V flag = bit 6 of memory
        - Z flag = (A & memory) == 0

        Args:
            instr: BitTest instruction
        """
        value_loc = self.parent._get_operand_location(instr.value)

        # BIT instruction requires a memory operand
        self.emitter.emit_instruction("BIT", self.parent._format_operand(value_loc))

        # Flags are now set:
        # - For bit 7 test: N flag indicates bit 7 value
        # - For bit 6 test: V flag indicates bit 6 value
        # - Z flag can also be used if needed

    # ========================================================================
    # Rotate Instruction
    # ========================================================================

    def select_rotate(self, instr: Rotate):
        """
        Generate code for Rotate instruction using ROL/ROR instructions.

        Emits ROL (rotate left) or ROR (rotate right) instructions.
        Each rotation is performed count times.

        Args:
            instr: Rotate instruction
        """
        # Load source into A
        source_loc = self.parent._get_operand_location(instr.source)
        self.emitter.emit_instruction("LDA", self.parent._format_operand(source_loc))

        # Determine instruction based on direction
        rotate_instr = "ROL" if instr.direction == 'left' else "ROR"

        # Emit rotate instruction 'count' times
        for _ in range(instr.count):
            self.emitter.emit_instruction(rotate_instr)

        # Store result to destination
        dest_loc = self.parent._get_operand_location(instr.dest)
        self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))
