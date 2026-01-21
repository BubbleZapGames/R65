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
            # It's a location - check for addressing mode limitations
            # CPX and CPY don't support stack-relative or indexed addressing
            if mnemonic in ('CPX', 'CPY') and (
                operand.kind == LocationKind.STACK or
                operand.index_register is not None
            ):
                # Load to temp via A, then compare with temp
                temp_addr = self.parent._get_temp_address()
                if temp_addr:
                    self._emit_load_store('LDA', operand)
                    self._emit_instr(Opcode.STA_DP, temp_addr, "Store to temp for CPX/CPY")
                    opcode = getattr(Opcode, f"{mnemonic}_DP")
                    self._emit_instr(opcode, temp_addr)
                else:
                    # No scratch available - use push/pop pattern
                    # LDA operand, PHA, then compare X/Y with stack value
                    # This is more complex: transfer X/Y to A, compare with memory
                    # Actually, simpler: transfer to A, push, compare using CMP
                    # But we want CPX/CPY because the comparison is with X/Y
                    # Solution: Load value to A, then use TXA/TYA + CMP approach
                    # But that changes the register... Let's use the opposite approach:
                    # Push X/Y, load A with operand, store to temp location on stack,
                    # then compare X/Y with stack[1]
                    # Actually, simplest: transfer X/Y to A and compare with memory
                    if mnemonic == 'CPX':
                        self._emit_instr(Opcode.PHX, comment="Save X")
                        self._emit_instr(Opcode.TXA, comment="Transfer X to A for comparison")
                        self._emit_load_store('CMP', operand)  # CMP supports all addressing modes
                        self._emit_instr(Opcode.PLX, comment="Restore X")
                    else:  # CPY
                        self._emit_instr(Opcode.PHY, comment="Save Y")
                        self._emit_instr(Opcode.TYA, comment="Transfer Y to A for comparison")
                        self._emit_load_store('CMP', operand)  # CMP supports all addressing modes
                        self._emit_instr(Opcode.PLY, comment="Restore Y")
            else:
                # Use parent's opcode selection
                opcode, op = self.parent._get_opcode_for_location(mnemonic, operand)
                self._emit_instr(opcode, op)

    # ========================================================================
    # Compare Instruction
    # ========================================================================

    def select_compare(self, instr: Compare):
        """
        Generate code for Compare instruction.

        Emits CMP/CPX/CPY instruction and sets processor flags for subsequent
        conditional branch. For 16-bit types (u16/i16), switches to 16-bit
        accumulator mode.

        Args:
            instr: Compare instruction
        """
        # Store type info for subsequent CondBranch (for signed/unsigned detection)
        self.parent.last_comparison_type = instr.type_info

        # Check if this is a 16-bit comparison
        is_16bit = self._is_16bit_type(instr.type_info)

        left_loc = self.parent._get_operand_location(instr.left)
        right_operand, is_immediate, pushed_reg = self._prepare_right_operand(instr.right, is_16bit)

        # Switch to 16-bit mode if needed (for A comparisons only)
        needs_mode_switch = is_16bit and left_loc.kind == LocationKind.HARDWARE and left_loc.hw_register == 'A'
        already_in_16bit = self.parent.emitter.get_accu_mode() == 16

        if needs_mode_switch and not already_in_16bit:
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20), "16-bit A for comparison")
            self.parent.emitter.emit_accu_mode(16)

        # For stack-relative comparisons with 16-bit values, we need to use A
        if is_16bit and left_loc.kind == LocationKind.STACK and not already_in_16bit:
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20), "16-bit A for comparison")
            self.parent.emitter.emit_accu_mode(16)
            needs_mode_switch = True

        # Emit appropriate comparison instruction based on left operand
        self._emit_comparison(left_loc, right_operand, is_immediate)

        # Switch back to 8-bit mode if we changed it
        if needs_mode_switch and not already_in_16bit:
            self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "8-bit A")
            self.parent.emitter.emit_accu_mode(8)

        # Pop register if we pushed it for temp storage
        if pushed_reg:
            if pushed_reg == 'X':
                self._emit_instr(Opcode.PLX, comment="Restore X")
            elif pushed_reg == 'Y':
                self._emit_instr(Opcode.PLY, comment="Restore Y")

    def _is_16bit_type(self, type_info) -> bool:
        """Check if the type is a 16-bit type (u16 or i16)."""
        if type_info is None:
            return False
        from r65.compiler.hir import BasicTypeInfo
        if isinstance(type_info, BasicTypeInfo):
            return type_info.name in ('u16', 'i16')
        return False

    def _prepare_right_operand(self, right, is_16bit: bool = False):
        """
        Prepare right operand for comparison.

        Hardware registers must be stored to temp location first.

        Args:
            right: Right operand
            is_16bit: Whether the comparison is 16-bit

        Returns:
            Tuple of (operand, is_immediate, needs_pop) for comparison emission
            needs_pop is True if we pushed X/Y and need to pop after comparison
        """
        if isinstance(right, MIRImmediate):
            if is_16bit:
                return Immediate(right.value & 0xFFFF), True, False
            else:
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
