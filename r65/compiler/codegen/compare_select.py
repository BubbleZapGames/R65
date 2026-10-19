# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
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
from r65.compiler.codegen.type_utils import is_16bit_scalar


class CompareSelector(BaseSelector):
    """
    Handles compare and bit operation instruction selection.

    Manages generation of comparison instructions (CMP, CPX, CPY),
    bit test instructions (BIT), and rotate instructions (ROL, ROR).
    """

    # ========================================================================
    # Emission Helpers
    # ========================================================================

    def _emit_cmp(self, mnemonic: str, operand, is_immediate: bool,
                  a_is_scratch: bool = False):
        """Emit a comparison instruction (CMP, CPX, CPY).

        *a_is_scratch*: A was loaded only to make this compare, so a signed
        compare may leave the difference in it rather than restoring.
        """
        if self._is_signed_type():
            # The signed branches read N XOR V, and CMP/CPX/CPY do not set
            # V - so a signed compare is a real subtraction instead.
            if mnemonic in ('CPX', 'CPY'):
                # No SBC on X/Y. Never emitted by the projects in tree
                # (X/Y hold u16); TXA/TYA as the CPX fallback below does.
                self.parent._ensure_m16_mode()
                self._emit_instr(Opcode.TXA if mnemonic == 'CPX' else Opcode.TYA,
                                 comment="Signed compare: X/Y to A for SBC")
                a_is_scratch = True  # A holds the copy, not a value of its own
            self._emit_signed_cmp(operand, is_immediate, restore_a=not a_is_scratch)
            return
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
                # Only applies to D_EQUALS_S strategy (SET_DBR keeps DP at zeropage).
                from r65.compiler.mir.nodes import FarPtrStrategy
                if (operand.kind == LocationKind.STACK and
                    self.parent.current_function and
                    self.parent.current_function.has_far_ptr_stack_params and
                    self.parent.current_function.far_ptr_strategy != FarPtrStrategy.SET_DBR):
                    opcode = getattr(Opcode, f"{mnemonic}_DP")
                    self._emit_instr(opcode, Address(operand.stack_offset))
                else:
                    # CPX/CPY compare 16 bits (a u8 compare took TXA/TYA +
                    # CMP in m8 before reaching here), so the A that stands
                    # in for them below must be 16-bit too.
                    self.parent._ensure_m16_mode()
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

    def _emit_signed_cmp(self, operand, is_immediate: bool, restore_a: bool = True):
        """
        Signed compare of A against *operand*, leaving A unchanged unless
        *restore_a* is False (A was loaded only for the compare).

        CMP sets N, Z and C but never V, so the signed branches (which test
        N XOR V) would read whatever V the previous arithmetic left - an
        overflow anywhere before the compare inverts it. SBC sets V for
        the subtraction itself:

            SEC / SBC rhs     ; N, V, Z, C of lhs - rhs
            PHP               ; keep them
            CLC / ADC rhs     ; A = lhs again
            PLP               ; flags back to the subtraction's

        so the flags are the compare's and A still holds the left operand,
        as callers that reuse it after a CMP expect. PHP moves S by one, so
        a stack-relative rhs is one further away for the ADC. The rhs is
        read twice; a hardware register is read once into scratch first.
        When the compare loaded A itself, nothing reads it afterwards and
        the restore (PHP..PLP, ~14 cycles) is skipped.

        Against a constant there is no V to read at all. Flipping the sign
        bit of both sides turns the signed order into the unsigned one, so

            EOR #sign / CMP #(rhs ^ sign)     ; C, Z of an unsigned compare

        and the branch uses BCC/BCS/BEQ (`_compare_form = 'unsigned'`). A is
        put back with a second EOR, under PHP/PLP because EOR changes Z.
        Against 0 the sign is the answer: N and Z of A itself, no V
        (`_compare_form = 'sign'`).
        """
        from r65.compiler.codegen.asm_nodes import StackOffset

        # Against 0 the order is A's sign: N and Z of A, which a load has
        # often set already, and the branch reads nothing else.
        if is_immediate and isinstance(operand, Immediate) and operand.value == 0:
            if self.emitter.nz_valid_for != 'A':
                self._emit_instr(Opcode.CMP_IMMEDIATE, operand)
            self.parent._compare_form = 'sign'
            return

        width = self.emitter.get_accu_mode()
        if (is_immediate and isinstance(operand, Immediate)
                and isinstance(operand.value, int) and width in (8, 16)):
            sign, mask = (0x8000, 0xFFFF) if width == 16 else (0x80, 0xFF)
            self._emit_instr(Opcode.EOR_IMMEDIATE, Immediate(sign),
                             "Signed compare: flip the sign bit")
            self._emit_instr(Opcode.CMP_IMMEDIATE, Immediate((operand.value & mask) ^ sign),
                             "... so an unsigned compare orders it")
            if restore_a:
                self._emit_instr(Opcode.PHP, comment="Keep the compare's flags")
                self._emit_instr(Opcode.EOR_IMMEDIATE, Immediate(sign), "A = lhs")
                self._emit_instr(Opcode.PLP)
            self.parent._compare_form = 'unsigned'
            return

        if (not is_immediate and not isinstance(operand, (Address, StackOffset))
                and operand.kind == LocationKind.MEMORY
                and getattr(operand, 'storage_type', None) == 'hw'):
            temp_addr = self.parent._get_temp_address()
            if temp_addr is None:
                raise InstructionSelectionError(
                    "Signed comparison against a hardware register needs a scratch "
                    "register (the register must be read exactly once)",
                    source_loc=self.parent._current_source_loc)
            self._emit_instr(Opcode.PHA, comment="Signed compare: save lhs")
            self._emit_load_store('LDA', operand)
            self._emit_instr(Opcode.STA_DP, temp_addr, "Read hw register once")
            self._emit_instr(Opcode.PLA, comment="Restore lhs")
            operand = temp_addr

        def arith(mnemonic: str, s_shift: int):
            if is_immediate:
                self._emit_instr(getattr(Opcode, f"{mnemonic}_IMMEDIATE"), operand)
            elif isinstance(operand, Address):
                self._emit_instr(getattr(Opcode, f"{mnemonic}_DP"), operand)
            elif isinstance(operand, StackOffset):
                self._emit_instr(getattr(Opcode, f"{mnemonic}_STACK"),
                                 StackOffset(operand.offset + s_shift))
            else:
                # Resolved now, so past the PHP already (the emitter counts it);
                # only an operand resolved before it needs the shift above.
                opcode, op = self.parent._get_opcode_for_location(mnemonic, operand)
                self._emit_instr(opcode, op)

        self._emit_instr(Opcode.SEC, comment="Signed compare: lhs - rhs sets V")
        arith('SBC', 0)
        if not restore_a:
            return
        self._emit_instr(Opcode.PHP, comment="Keep the compare's flags")
        self._emit_instr(Opcode.CLC)
        arith('ADC', 1)
        self._emit_instr(Opcode.PLP, comment="A = lhs, flags = lhs - rhs")

    def _emit_swapped_hw_compare(self, left_reg: str, is_16bit: bool):
        """
        Handle comparison when both operands are in hardware registers
        and no scratch is available.

        A = right value, left_reg (X or Y) = left value.
        Pushes right to stack, transfers left to A, compares against stack
        (normal direction: left - right), then preserves flags across
        stack cleanup using PHP/PLP.

        Sets comparison_reversed = False (normal direction).
        """
        from r65.compiler.codegen.asm_nodes import StackOffset
        # Push right (A) to stack
        self._emit_instr(Opcode.PHA, comment="Push right operand for hw-hw compare")
        # Transfer left to A
        if left_reg == 'Y':
            self._emit_instr(Opcode.TYA, comment="Transfer left (Y) to A")
        else:
            self._emit_instr(Opcode.TXA, comment="Transfer left (X) to A")
        # Compare: A(left) - stack(right) = normal direction. A signed
        # compare subtracts so V is set (CMP leaves it alone); A is
        # overwritten by the cleanup below either way.
        if self._is_signed_type():
            self._emit_instr(Opcode.SEC)
            self._emit_instr(Opcode.SBC_STACK, StackOffset(1), "Signed compare left vs right")
        else:
            self._emit_instr(Opcode.CMP_STACK, StackOffset(1), "Compare left vs right")
        # Save comparison flags
        self._emit_instr(Opcode.PHP, comment="Save comparison flags")
        if is_16bit:
            # PHA pushed 2 bytes + PHP pushed 1 byte = 3 bytes on stack.
            # Use m8 to pop individual bytes, then PLP restores m16 + flags.
            # The emitter sizes each PLA by its tracked width, so it has to
            # see this SEP, and the PLP that undoes it.
            self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "m8 for cleanup")
            self.parent.emitter.set_accu_mode_tracking(8)
            # Move saved P past the pushed right value
            self._emit_instr(Opcode.LDA_STACK, StackOffset(1), "Load saved P")
            self._emit_instr(Opcode.STA_STACK, StackOffset(3), "Move P past right")
            self._emit_instr(Opcode.PLA, comment="Pop old P position")
            self._emit_instr(Opcode.PLA, comment="Pop right byte")
            self._emit_instr(Opcode.PLP, comment="Restore flags and m16 mode")
            self.parent.emitter.set_accu_mode_tracking(16)
        else:
            # PHA pushed 1 byte + PHP pushed 1 byte = 2 bytes on stack.
            # Move saved P past the pushed right value, then PLP.
            self._emit_instr(Opcode.LDA_STACK, StackOffset(1), "Load saved P")
            self._emit_instr(Opcode.STA_STACK, StackOffset(2), "Move P past right")
            self._emit_instr(Opcode.PLA, comment="Pop old P position")
            self._emit_instr(Opcode.PLP, comment="Restore flags and m8 mode")
        # comparison_reversed = False: flags are left - right (normal order)
        self.parent._comparison_reversed = False

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
        self.parent._compare_form = None
        # A signed compare puts A back to the left operand afterwards; not
        # needed when that is a vreg nothing reads after the compare.
        self._left_dead = self._dead_after(instr.left)
        # Track whether the right operand is immediate 0 for branch optimization
        self.parent._compare_rhs_is_zero = (
            isinstance(instr.right, MIRImmediate) and instr.right.value == 0
        )

        # Check if this is a 16-bit comparison
        is_16bit = self._is_16bit_type(instr.type_info)

        left_loc = self.parent._get_operand_location(instr.left)
        right_operand, is_immediate, swapped = self._prepare_right_operand(instr.right, is_16bit)

        # Handle swapped comparison: right value is already in A, compare with left.
        # This avoids push/pop which clobbers flags needed by CondBranch.
        # CMP computes A - operand, so flags represent (right - left) = reversed.
        if swapped:
            if is_16bit:
                self.parent._ensure_m16_mode()
            else:
                self.parent._ensure_m8_mode()
            if left_loc.is_hw() and left_loc.hw_register in ('X', 'Y'):
                # Both operands in hardware registers, no scratch available.
                # A = right value. X/Y = left value.
                # Strategy: push right to stack, transfer left to A, compare
                # against stack (NORMAL direction: left - right), then use
                # PHP/PLP to preserve flags across stack cleanup.
                self._emit_swapped_hw_compare(left_loc.hw_register, is_16bit)
                # Comparison is left - right = normal order, NOT reversed
                return
            if left_loc.is_hw() and left_loc.hw_register == 'A':
                # Both left and right are in A — they must be the same value.
                # CMP A,A always yields zero/equal. Emit CMP #$00 as equivalent.
                self._emit_cmp('CMP', MIRImmediate(0), True)
            else:
                self._emit_cmp('CMP', left_loc, False)
            self.parent._comparison_reversed = True
            return

        # Switch to appropriate mode for the comparison
        # Need 16-bit mode for any 16-bit comparison that uses the accumulator:
        # - If left is in A register
        # - If left is in memory/stack (will be loaded to A)
        # X/Y comparisons (CPX/CPY) use the X flag, not M flag, so no mode switch needed
        needs_16bit = (is_16bit and
            left_loc.kind != LocationKind.HARDWARE or
            (is_16bit and left_loc.is_hw('A')))

        if needs_16bit:
            self.parent._ensure_m16_mode()
        elif not left_loc.is_hw('X') and not left_loc.is_hw('Y'):
            # Otherwise m8 for an 8-bit compare. CPX/CPY follow the X flag
            # and run in either width: a rotated loop's test stays in the
            # width its body ends in.
            self.parent._ensure_m8_mode()

        # Emit appropriate comparison instruction based on left operand
        self._emit_comparison(left_loc, right_operand, is_immediate)

        # Leave the width as is: branches do not read M, and a loop back-edge
        # must arrive in the header's mode. Code paths that need m8 (like
        # function returns) switch explicitly.

    def _dead_after(self, operand) -> bool:
        """Whether *operand* is a vreg no instruction reads after the one
        being selected. A bare register, or an unknown position, is not."""
        from r65.compiler.mir.nodes import VirtualRegister
        if type(operand) is not VirtualRegister:
            return False
        liveness = self.parent.reg_alloc.instr_liveness
        pos = self.parent._current_pos
        if liveness is None or pos is None:
            return False
        return operand not in liveness._get_live_after(*pos)

    def _is_signed_type(self) -> bool:
        """Check if the current comparison type is signed (i8, i16).

        A newtype answers for its payload: `struct Q10(i16)` must compare with
        BMI/BPL, not BCC/BCS.
        """
        type_info = self.parent.last_comparison_type
        if type_info is not None:
            from r65.compiler.hir import BasicTypeInfo
            from r65.compiler.hir.types import strip_newtype
            type_info = strip_newtype(type_info)
            if isinstance(type_info, BasicTypeInfo):
                return type_info.name.startswith('i')
        return False

    def _is_16bit_type(self, type_info) -> bool:
        """Check if the type is a 16-bit type (u16 or i16)."""
        return is_16bit_scalar(type_info)

    def _prepare_right_operand(self, right, is_16bit: bool = False):
        """
        Prepare right operand for comparison.

        Hardware registers must be stored to temp location first.

        Args:
            right: Right operand
            is_16bit: Whether the comparison is 16-bit

        Returns:
            Tuple of (operand, is_immediate, swapped): swapped is True when
            the right value was moved into A and the compare must run
            reversed
        """
        if isinstance(right, MIRImmediate):
            if is_16bit:
                return Immediate(right.value & 0xFFFF), True, False
            else:
                return Immediate(right.value & 0xFF), True, False

        right_loc = self.parent._get_operand_location(right)

        if right_loc.is_hw():
            operand, swapped = self._store_hw_register_to_temp(right_loc)
            return operand, False, swapped

        return right_loc, False, False

    def _store_hw_register_to_temp(self, right_loc):
        """
        Store hardware register to temp location for comparison.

        Returns:
            Tuple of (operand, swapped): (temp Address, False), or
            (None, True) when the value was moved into A and the compare
            must run reversed
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
                return None, True

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
            return None, True

    def _can_elide_cmp_zero(self, reg: str, right_operand, is_immediate: bool) -> bool:
        """
        Check if CMP/CPX/CPY #0 can be skipped because N/Z flags already
        reflect the register's value, at its current width, from a prior
        instruction (LDA, DEX, TXA, XBA, etc.). The emitter drops that
        validity at a REP/SEP that changes the register's width.

        Only safe for unsigned types — signed ordered comparisons need
        the V flag, which only the signed compare's SBC sets.
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
        if left_loc.is_hw():
            self._emit_hw_register_comparison(left_loc, right_operand, is_immediate)
        else:
            # Memory or virtual register - load to A and compare
            self._emit_load_store('LDA', left_loc)
            # LDA sets N/Z (tracked by emitter). For unsigned comparisons
            # against 0, CMP #0 is redundant. Signed types still compare,
            # because the signed branch codegen needs V from the SBC.
            if self._can_elide_cmp_zero('A', right_operand, is_immediate):
                return
            self._emit_cmp('CMP', right_operand, is_immediate, a_is_scratch=True)

    def _emit_hw_register_comparison(self, left_loc, right_operand, is_immediate: bool):
        """Emit comparison for hardware register left operand."""
        reg = left_loc.hw_register

        if reg in ('X', 'Y'):
            if self._can_elide_cmp_zero(reg, right_operand, is_immediate):
                return
            # CPX/CPY always do 16-bit compares in x16 mode.
            # For u8 memory operands, CPX/CPY dp reads 2 bytes from memory
            # but only 1 byte was written — the adjacent byte may be garbage.
            # So a u8 operand goes through TXA/TYA + CMP in m8.
            is_u8_compare = not self._is_16bit_type(self.parent.last_comparison_type)
            if is_u8_compare and not is_immediate:
                if reg == 'X':
                    self._emit_instr(Opcode.TXA, comment="Transfer X to A for u8 comparison")
                else:
                    self._emit_instr(Opcode.TYA, comment="Transfer Y to A for u8 comparison")
                self.parent._ensure_m8_mode()
                self._emit_cmp('CMP', right_operand, is_immediate)
                return
            mnem = 'CPX' if reg == 'X' else 'CPY'
            self._emit_cmp(mnem, right_operand, is_immediate)
        elif reg == 'A':
            if self._can_elide_cmp_zero('A', right_operand, is_immediate):
                return
            self._emit_cmp('CMP', right_operand, is_immediate,
                           a_is_scratch=getattr(self, '_left_dead', False))
        elif reg == 'B':
            # B register - XBA transfers B's value to A (sets N/Z)
            self.parent._access_b_value_in_a()
            if self._can_elide_cmp_zero('A', right_operand, is_immediate):
                return
            self._emit_cmp('CMP', right_operand, is_immediate)
            # Note: Don't restore A since this is just a comparison
            # State is now SWAPPED (A=B, B=A)
        elif reg in ('D', 'S', 'DBR', 'PBR'):
            # No compare instruction reads these: bring the value into A,
            # which a comparison may use as scratch, at its own width.
            self.parent._emit_register_transfer(reg, 'A')
            self._emit_cmp('CMP', right_operand, is_immediate, a_is_scratch=True)
        else:
            raise InstructionSelectionError(
                f"Unsupported hardware register for comparison: {reg}", source_loc=self.parent._current_source_loc)

    # ========================================================================
    # Bit Test Instruction
    # ========================================================================

    def select_bit_test(self, instr: BitTest):
        """
        Set N (and, for memory, V) from the bits a BitTest names.

        Bits 7/6 are tested in m8, which reads the low byte of a wider value;
        bits 15/14 in m16. N is the top bit of any load, so a value on the
        stack is just loaded, one in A is compared with 0, which leaves A
        alone, and one in X/Y is compared with 0 (x16, so bit 15) or moved
        to A (m8, so bit 7). One in B is swapped into A. V comes only from
        BIT on memory, and MIR builds a V test for nothing else.

        Args:
            instr: BitTest instruction
        """
        value_loc = self.parent._get_operand_location(instr.value)
        if instr.test_bit == 15 and value_loc.hw_register in ('X', 'Y'):
            # CPX/CPY follow the index width (always x16), not M.
            opcode = Opcode.CPX_IMMEDIATE if value_loc.is_hw('X') else Opcode.CPY_IMMEDIATE
            self._emit_instr(opcode, Immediate(0), "Set N from bit 15")
            return
        if instr.test_bit >= 8:
            self.parent._ensure_m16_mode()
        else:
            self.parent._ensure_m8_mode()

        if value_loc.kind == LocationKind.STACK or value_loc.is_hw():
            if instr.test_bit not in (7, 15):
                raise InstructionSelectionError(
                    f"BitTest of bit {instr.test_bit} needs a memory operand",
                    source_loc=self.parent._current_source_loc)
            if value_loc.kind == LocationKind.STACK:
                self._emit_load_store('LDA', value_loc)
            elif value_loc.is_hw('A'):
                self._emit_instr(Opcode.CMP_IMMEDIATE, Immediate(0), "Set N from A")
            elif value_loc.hw_register in ('X', 'Y'):
                self._emit_instr(Opcode.TXA if value_loc.is_hw('X') else Opcode.TYA,
                                 comment="Set N from bit 7")
            elif value_loc.is_hw('B'):
                self.parent._access_b_value_in_a()
            else:
                raise InstructionSelectionError(
                    f"BitTest of hardware register {value_loc.hw_register}",
                    source_loc=self.parent._current_source_loc)
            return

        opcode, operand = self.parent._get_opcode_for_location('BIT', value_loc)
        self._emit_instr(opcode, operand)

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
        # Load source into A (skip if already in A)
        source_loc = self.parent._get_operand_location(instr.source)
        if not source_loc.is_hw('A'):
            self._emit_load_store('LDA', source_loc)

        # Determine instruction based on direction
        rotate_opcode = Opcode.ROL if instr.direction == 'left' else Opcode.ROR

        # Emit rotate instruction 'count' times
        for _ in range(instr.count):
            self._emit_instr(rotate_opcode)

        # Store result to destination (skip if dest is A)
        dest_loc = self.parent._get_operand_location(instr.dest)
        if not dest_loc.is_hw('A'):
            self._emit_load_store('STA', dest_loc)
