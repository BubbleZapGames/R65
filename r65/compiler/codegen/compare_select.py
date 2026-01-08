"""
Compare operation selector: Compare, BitTest, Rotate instructions.

Handles comparison and bit manipulation instruction generation
for the 65816 processor.
"""

from r65.compiler.mir.nodes import Compare, BitTest, Rotate, Immediate as MIRImmediate
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address
from r65.compiler.codegen.base_selector import BaseSelector


class CompareSelector(BaseSelector):
    """
    Handles compare and bit operation instruction selection.

    Manages generation of comparison instructions (CMP, CPX, CPY),
    bit test instructions (BIT), and rotate instructions (ROL, ROR).
    """

    # ========================================================================
    # Emission Helpers
    # ========================================================================

    def _emit_load_store(self, mnemonic: str, location, comment: str = None):
        """Emit a load/store instruction using parent's opcode selection."""
        opcode, operand = self.parent._get_opcode_for_location(mnemonic, location)
        self._emit_instr(opcode, operand, comment)

    def _emit_cmp(self, mnemonic: str, operand, is_immediate: bool):
        """Emit a comparison instruction (CMP, CPX, CPY)."""
        if is_immediate:
            opcode = getattr(Opcode, f"{mnemonic}_IMMEDIATE")
            self._emit_instr(opcode, operand)
        elif isinstance(operand, Address):
            # Direct page address (like $00 for temp)
            opcode = getattr(Opcode, f"{mnemonic}_DP")
            self._emit_instr(opcode, operand)
        else:
            # It's a location - use parent's opcode selection
            opcode, op = self.parent._get_opcode_for_location(mnemonic, operand)
            self._emit_instr(opcode, op)

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
        right_operand, is_immediate, pushed_reg = self._prepare_right_operand(instr.right)

        # Emit appropriate comparison instruction based on left operand
        self._emit_comparison(left_loc, right_operand, is_immediate)

        # Pop register if we pushed it for temp storage
        if pushed_reg:
            if pushed_reg == 'X':
                self._emit_instr(Opcode.PLX, comment="Restore X")
            elif pushed_reg == 'Y':
                self._emit_instr(Opcode.PLY, comment="Restore Y")

    def _prepare_right_operand(self, right):
        """
        Prepare right operand for comparison.

        Hardware registers must be stored to temp location first.

        Args:
            right: Right operand

        Returns:
            Tuple of (operand, is_immediate, needs_pop) for comparison emission
            needs_pop is True if we pushed X/Y and need to pop after comparison
        """
        if isinstance(right, MIRImmediate):
            return Immediate(right.value & 0xFF), True, False

        right_loc = self.parent._get_operand_location(right)

        if right_loc.kind == LocationKind.HARDWARE:
            operand, needs_pop = self._store_hw_register_to_temp(right_loc)
            return operand, False, needs_pop

        return right_loc, False, False

    def _store_hw_register_to_temp(self, right_loc):
        """
        Store hardware register to temp location for comparison.

        Returns:
            Tuple of (operand, needs_pop) where operand is Address or PhysicalLocation
            and needs_pop indicates if PLX/PLY is needed after comparison
        """
        if right_loc.hw_register in ['A', 'B']:
            # A and B can use stack-relative via STA
            temp_loc = self.parent._get_temp_location()
            if right_loc.hw_register == 'B':
                self.parent._access_b_value_in_a()
                self.parent._emit_store('STA', temp_loc, "Store B to temp")
                self.parent._ensure_xba_state_normal("Restore A")
            else:
                self.parent._emit_store('STA', temp_loc, "Store A to temp")
            return temp_loc, False

        # X and Y need special handling - try scratch, fall back to push
        temp_addr = self.parent._get_temp_address()
        if temp_addr:
            # Use scratch register
            if right_loc.hw_register == 'X':
                self._emit_instr(Opcode.STX_DP, temp_addr, "Store X to temp")
            else:  # Y
                self._emit_instr(Opcode.STY_DP, temp_addr, "Store Y to temp")
            return temp_addr, False
        else:
            # No scratch available - use push/pop pattern
            from r65.compiler.codegen.register_alloc import PhysicalLocation
            if right_loc.hw_register == 'X':
                self._emit_instr(Opcode.PHX, comment="Push X for temp")
            else:  # Y
                self._emit_instr(Opcode.PHY, comment="Push Y for temp")
            # Stack-relative location at offset 1
            temp_loc = PhysicalLocation(kind=LocationKind.STACK, stack_offset=1, size=1)
            return temp_loc, right_loc.hw_register  # Return register name for pop

    def _emit_comparison(self, left_loc, right_operand, is_immediate: bool):
        """
        Emit appropriate comparison instruction.

        Args:
            left_loc: Left operand location
            right_operand: Right operand (Immediate, Address, or location)
            is_immediate: True if right operand is immediate
        """
        if left_loc.kind == LocationKind.HARDWARE:
            self._emit_hw_register_comparison(left_loc, right_operand, is_immediate)
        else:
            # Memory or virtual register - load to A and compare
            self._emit_load_store('LDA', left_loc)
            self._emit_cmp('CMP', right_operand, is_immediate)

    def _emit_hw_register_comparison(self, left_loc, right_operand, is_immediate: bool):
        """Emit comparison for hardware register left operand."""
        reg = left_loc.hw_register

        if reg == 'X':
            self._emit_cmp('CPX', right_operand, is_immediate)
        elif reg == 'Y':
            self._emit_cmp('CPY', right_operand, is_immediate)
        elif reg == 'A':
            self._emit_cmp('CMP', right_operand, is_immediate)
        elif reg == 'B':
            # B register - transfer to A and compare
            self.parent._access_b_value_in_a()
            self._emit_cmp('CMP', right_operand, is_immediate)
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
        opcode, operand = self.parent._get_opcode_for_location('BIT', value_loc)
        self._emit_instr(opcode, operand)

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
        self._emit_load_store('LDA', source_loc)

        # Determine instruction based on direction
        rotate_opcode = Opcode.ROL_A if instr.direction == 'left' else Opcode.ROR_A

        # Emit rotate instruction 'count' times
        for _ in range(instr.count):
            self._emit_instr(rotate_opcode)

        # Store result to destination
        dest_loc = self.parent._get_operand_location(instr.dest)
        self._emit_load_store('STA', dest_loc)
