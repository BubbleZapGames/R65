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
                # When D=S is active (far pointer functions), stack offsets
                # ARE direct page offsets. Use CPX/CPY with DP addressing.
                if (operand.kind == LocationKind.STACK and
                    self.parent.current_function and
                    self.parent.current_function.has_far_ptr_stack_params):
                    opcode = getattr(Opcode, f"{mnemonic}_DP")
                    self._emit_instr(opcode, Address(operand.stack_offset))
                else:
                    # Load to temp via A, then compare with temp
                    temp_addr = self.parent._get_temp_address()
                    if temp_addr:
                        self._emit_load_store('LDA', operand)
                        self._emit_instr(Opcode.STA_DP, temp_addr, "Store to temp for CPX/CPY")
                        opcode = getattr(Opcode, f"{mnemonic}_DP")
                        self._emit_instr(opcode, temp_addr)
                    else:
                        # Transfer X/Y to A and use CMP (which supports all modes).
                        # TXA/TYA don't modify X/Y so no save/restore needed.
                        # This avoids PHX/PHY which would shift stack offsets and
                        # PLX/PLY which clobber N/Z flags needed by CondBranch.
                        if mnemonic == 'CPX':
                            self._emit_instr(Opcode.TXA, comment="Transfer X to A for comparison")
                        else:  # CPY
                            self._emit_instr(Opcode.TYA, comment="Transfer Y to A for comparison")
                        self._emit_load_store('CMP', operand)
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
        self.parent._comparison_reversed = False
        # Track whether the right operand is immediate 0 for branch optimization
        self.parent._compare_rhs_is_zero = (
            isinstance(instr.right, MIRImmediate) and instr.right.value == 0
        )

        # Check if this is a 16-bit comparison
        is_16bit = self._is_16bit_type(instr.type_info)

        left_loc = self.parent._get_operand_location(instr.left)
        right_operand, is_immediate, pushed_reg = self._prepare_right_operand(instr.right, is_16bit)

        # Handle swapped comparison: right value is already in A, compare with left.
        # This avoids push/pop which clobbers flags needed by CondBranch.
        # CMP computes A - operand, so flags represent (right - left) = reversed.
        if pushed_reg == 'SWAPPED':
            if is_16bit:
                self.parent._ensure_m16_mode()
            else:
                self.parent._ensure_m8_mode()
            self._emit_cmp('CMP', left_loc, False)
            self.parent._comparison_reversed = True
            return

        # If right operand was pushed to stack (PHX/PHY/PHA), all stack-relative
        # offsets shift. Adjust the left operand if it's on the stack.
        if pushed_reg and left_loc.kind == LocationKind.STACK:
            from r65.compiler.codegen.register_alloc import PhysicalLocation
            # X/Y pushes are 2 bytes (always 16-bit), A is 1 byte (m8)
            push_size = 2 if pushed_reg in ('X', 'Y') else 1
            left_loc = PhysicalLocation(
                kind=LocationKind.STACK,
                stack_offset=left_loc.stack_offset + push_size,
                size=left_loc.size
            )

        # Switch to appropriate mode for the comparison
        # Need 16-bit mode for any 16-bit comparison that uses the accumulator:
        # - If left is in A register
        # - If left is in memory/stack (will be loaded to A)
        # X/Y comparisons (CPX/CPY) use the X flag, not M flag, so no mode switch needed
        needs_16bit = (is_16bit and
            left_loc.kind != LocationKind.HARDWARE or
            (is_16bit and left_loc.kind == LocationKind.HARDWARE and left_loc.hw_register == 'A'))

        if needs_16bit:
            self.parent._ensure_m16_mode()
        else:
            # For 8-bit comparisons, ensure we're in m8 mode
            self.parent._ensure_m8_mode()

        # Emit appropriate comparison instruction based on left operand
        self._emit_comparison(left_loc, right_operand, is_immediate)

        # NOTE: We do NOT restore m8 mode after 16-bit comparisons.
        # The branch instructions that follow don't depend on the M flag.
        # Loop back-edges need to preserve the mode expected by the loop header.
        # Code paths that need m8 (like function returns) explicitly switch modes.
        # Previously this did `self.parent._ensure_m8_mode()` which broke loops
        # where the loop body runs in m16 mode.

        # Pop register if we pushed it for temp storage
        if pushed_reg:
            if pushed_reg == 'X':
                self._emit_instr(Opcode.PLX, comment="Restore X")
            elif pushed_reg == 'Y':
                self._emit_instr(Opcode.PLY, comment="Restore Y")
            elif pushed_reg == 'A':
                self._emit_instr(Opcode.PLA, comment="Restore A")

    def _is_signed_type(self) -> bool:
        """Check if the current comparison type is signed (i8, i16)."""
        type_info = self.parent.last_comparison_type
        if type_info is not None:
            from r65.compiler.hir import BasicTypeInfo
            if isinstance(type_info, BasicTypeInfo):
                return type_info.name.startswith('i')
        return False

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
            # A and B - try scratch, fall back to swap
            temp_addr = self.parent._get_temp_address()
            if temp_addr:
                if right_loc.hw_register == 'B':
                    self.parent._access_b_value_in_a()
                    self._emit_instr(Opcode.STA_DP, temp_addr, "Store B to temp")
                    self.parent._ensure_xba_state_normal("Restore A")
                else:
                    self._emit_instr(Opcode.STA_DP, temp_addr, "Store A to temp")
                return temp_addr, False
            else:
                # No scratch available - use swap approach: A already has the
                # right value, reverse comparison direction. This avoids
                # push/pop which clobbers CPU flags needed by CondBranch.
                if right_loc.hw_register == 'B':
                    self.parent._access_b_value_in_a()
                    # A now has B's value, comparison will be reversed
                return None, 'SWAPPED'

        # X and Y need special handling - try scratch, fall back to swap
        temp_addr = self.parent._get_temp_address()
        if temp_addr:
            # Use scratch register
            if right_loc.hw_register == 'X':
                self._emit_instr(Opcode.STX_DP, temp_addr, "Store X to temp")
            else:  # Y
                self._emit_instr(Opcode.STY_DP, temp_addr, "Store Y to temp")
            return temp_addr, False
        else:
            # No scratch available - use swap approach: transfer to A and
            # reverse comparison direction. This avoids push/pop which
            # clobbers CPU flags needed by the subsequent CondBranch.
            if right_loc.hw_register == 'X':
                self._emit_instr(Opcode.TXA, comment="Transfer X to A for comparison")
            else:  # Y
                self._emit_instr(Opcode.TYA, comment="Transfer Y to A for comparison")
            return None, 'SWAPPED'

    def _can_elide_cmp_zero(self, reg: str, right_operand, is_immediate: bool) -> bool:
        """
        Check if CMP/CPX/CPY #0 can be skipped because N/Z flags already
        reflect the register's value from a prior instruction (LDA, DEX,
        TXA, XBA, etc.).

        Only safe for unsigned types — signed ordered comparisons need
        the V flag which only CMP/CPX/CPY sets.
        """
        return (
            is_immediate
            and isinstance(right_operand, Immediate)
            and right_operand.value == 0
            and not self._is_signed_type()
            and self.emitter.nz_valid_for == reg
        )

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
            # LDA sets N/Z (tracked by emitter). For unsigned comparisons
            # against 0, CMP #0 is redundant. For signed types, keep CMP
            # because the standard signed branch codegen needs V flag.
            if self._can_elide_cmp_zero('A', right_operand, is_immediate):
                return
            self._emit_cmp('CMP', right_operand, is_immediate)

    def _emit_hw_register_comparison(self, left_loc, right_operand, is_immediate: bool):
        """Emit comparison for hardware register left operand."""
        reg = left_loc.hw_register

        if reg == 'X':
            if self._can_elide_cmp_zero('X', right_operand, is_immediate):
                return
            self._emit_cmp('CPX', right_operand, is_immediate)
        elif reg == 'Y':
            if self._can_elide_cmp_zero('Y', right_operand, is_immediate):
                return
            self._emit_cmp('CPY', right_operand, is_immediate)
        elif reg == 'A':
            if self._can_elide_cmp_zero('A', right_operand, is_immediate):
                return
            self._emit_cmp('CMP', right_operand, is_immediate)
        elif reg == 'B':
            # B register - XBA transfers B's value to A (sets N/Z)
            self.parent._access_b_value_in_a()
            if self._can_elide_cmp_zero('A', right_operand, is_immediate):
                return
            self._emit_cmp('CMP', right_operand, is_immediate)
            # Note: Don't restore A since this is just a comparison
            # State is now SWAPPED (A=B, B=A)
        else:
            raise InstructionSelectionError(
                f"Unsupported hardware register for comparison: {reg}", source_loc=self.parent._current_source_loc)

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

        # BIT instruction doesn't support stack-relative addressing
        # If value is on stack, use alternative approach
        if value_loc.kind == LocationKind.STACK:
            # Load stack value to A
            self._emit_load_store('LDA', value_loc)

            # Try to get a scratch for true BIT instruction
            temp_addr = self.parent._get_temp_address()
            if temp_addr:
                self._emit_instr(Opcode.STA_DP, temp_addr, "Store to temp for BIT")
                self._emit_instr(Opcode.BIT_DP, temp_addr)
            else:
                # No scratch available - use AND-based approach
                # For bit test, we want to set Z flag based on whether bits are set
                # The value is already in A, so we just use BIT #immediate with A value
                # However, BIT #imm tests A against immediate, which is what we want
                # for Z flag tests (A AND imm == 0)
                if instr.test_bit == 7:
                    # Test bit 7: AND #$80, then check Z flag
                    self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x80), "Test bit 7")
                elif instr.test_bit == 6:
                    # Test bit 6: AND #$40, then check Z flag
                    self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x40), "Test bit 6")
                else:
                    # General Z flag test - value is already in A
                    # BIT #imm equivalent: CMP #0 sets Z if A==0
                    # But we want to preserve A... use ORA #0 to just set flags
                    self._emit_instr(Opcode.ORA_IMMEDIATE, Immediate(0x00), "Set Z flag from A")
            return

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
