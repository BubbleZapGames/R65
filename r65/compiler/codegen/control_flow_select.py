# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Control flow instruction selector: Jump, Branch, Return.

Handles control flow instruction generation including conditional branches
with proper signed/unsigned comparison handling.
"""

from r65.compiler.mir.nodes import Jump, JumpTable, LookupTable, CondBranch, Return
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode, PULL_OPCODES
from r65.compiler.codegen.asm_nodes import Address
from r65.compiler.codegen.base_selector import BaseSelector
from r65.compiler.codegen.type_utils import is_16bit_scalar


class ControlFlowInstructionSelector(BaseSelector):
    """
    Handles control flow instruction selection.

    Manages generation of jump, branch, and return instructions
    with proper signed/unsigned comparison handling.
    """

    @property
    def current_function(self):
        return self.parent.current_function

    @property
    def last_comparison_type(self):
        return self.parent.last_comparison_type

    def _block_label(self, block_id: int) -> str:
        """Format a block label with function-scoped naming."""
        return self.parent._block_label(block_id)

    # ========================================================================
    # Jump Instructions
    # ========================================================================

    def select_jump(self, instr: Jump):
        """
        Generate code for Jump instruction.

        Args:
            instr: Jump instruction
        """
        self._emit_jump(Opcode.BRA, self._block_label(instr.target))

    def select_jump_table(self, instr: JumpTable):
        """
        Generate code for JumpTable instruction using 65816 JMP (addr,X).

        Emits a true O(1) indexed jump table:
        1. Load scrutinee into A
        2. Subtract base_value (SEC; SBC #base; BCC default)
        3. Bounds check (CMP #size; BCS default)
        4. ASL A (index *= 2 for word-sized entries)
        5. TAX (move to X for indexed addressing)
        6. JMP (table_label,X) (indirect indexed jump)
        7. Emit table data inline (.DW target_label for each entry)

        The table is emitted inline after the JMP instruction, in the same
        program bank, satisfying the JMP (addr,X) constraint that reads
        from PBR:addr+X.

        Args:
            instr: JumpTable instruction
        """
        table_size = len(instr.targets)
        # The dispatch sequence below uses 8-bit immediates throughout
        # (CMP #size, SBC #base, ASL byte index). table_size of 256+
        # would emit `CMP #$00`, silently disabling the bounds check.
        # The match analyser caps tables at MAX_JUMP_TABLE_SIZE = 128, so
        # this should always hold; assert to fail loud if the cap shifts.
        assert table_size <= 255, (
            f"JumpTable size {table_size} exceeds 8-bit bounds-check capacity"
        )

        scrutinee_loc = self.parent._get_operand_location(instr.scrutinee)

        # Load scrutinee into A in m8 so the bounds-check immediates and
        # the AND/ASL below all operate on a single byte. Without this, an
        # m16-on-entry caller would have LDA read 2 bytes from a u8 location
        # and the CMP/SBC immediates would extend to 16-bit, silently
        # corrupting the dispatch.
        # Jump table dispatch expects m8 scrutinee
        self.parent._ensure_m8_mode()
        if scrutinee_loc.is_hw('A'):
            pass  # Already in A
        else:
            self.parent._emit_load('LDA', scrutinee_loc)

        default_label = self._block_label(instr.default_target)

        # Subtract base_value to compute index
        if instr.base_value != 0:
            self._emit_implied(Opcode.SEC)
            self._emit_immediate(Opcode.SBC_IMMEDIATE, instr.base_value, "Compute index = scrutinee - base")
            self._emit_branch(Opcode.BCC, default_label, "Out of bounds (< base)")

        # Check if index >= table_size - out of bounds
        self._emit_immediate(Opcode.CMP_IMMEDIATE, table_size, "Check upper bound")
        self._emit_branch(Opcode.BCS, default_label, "Out of bounds (>= size)")

        # Switch to m16 to compute the index. In m8 mode, TAX still transfers
        # the FULL 16-bit accumulator (B:A_low) into X regardless of the m
        # flag; if the caller left B nonzero, X gets `(B << 8) | (index*2)`
        # and the indirect JMP reads from the wrong table entry. Going
        # through m16 with AND #$00FF zeroes B, then ASL doubles the index,
        # then SEP returns to m8 so the arm bodies start with the expected
        # accumulator width.
        from r65.compiler.codegen.constants import M_FLAG
        self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "m16 to zero B before TAX")
        self.parent.emitter.emit_accu_mode(16)
        self._emit_immediate(Opcode.AND_IMMEDIATE, 0x00FF, "Zero B (clear high byte)")
        self._emit_implied(Opcode.ASL, "Index *= 2 for word table")
        self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Back to m8 for arm bodies")
        self.parent.emitter.emit_accu_mode(8)

        # TAX - move to X for indexed addressing (B is now 0, so X = index*2)
        self._emit_implied(Opcode.TAX)

        # JMP (table_label,X) - indirect indexed jump through table
        table_label = self.parent._get_unique_label()
        self.emitter.emit_instr(Opcode.JMP_INDIRECT_X, Address(table_label), "Jump table dispatch")

        # Emit jump table data inline (same bank as code)
        self.emitter.emit_label(table_label)
        for target_block_id in instr.targets:
            target_label = self._block_label(target_block_id)
            self.emitter.emit_directive(f"    .DW {target_label}")

    def select_lookup_table(self, instr: LookupTable):
        """
        Generate code for LookupTable instruction.

        Emits an inline ROM table lookup:
        - u8 result: LDA table,X with .DB entries (no ASL, no mode switch)
        - u16 result: LDA table,X with .DW entries (ASL, REP/SEP mode switch)
        """
        from r65.compiler.codegen.type_utils import get_type_size
        from r65.compiler.codegen.constants import M_FLAG

        result_size = get_type_size(instr.type_info)
        is_u16 = (result_size == 2)

        table_size = len(instr.values)
        # Same 8-bit-immediate constraint as JumpTable's bounds check.
        assert table_size <= 255, (
            f"LookupTable size {table_size} exceeds 8-bit bounds-check capacity"
        )
        # For u8 LUTs, the default LDA #imm is emitted in m8 below and a
        # default_value > $FF would be silently truncated. The analyzer
        # builds u8 LUTs only when every arm body fits in u8, so this
        # should always hold.
        if not is_u16:
            assert 0 <= instr.default_value <= 0xFF, (
                f"u8 LookupTable default_value {instr.default_value:#x} "
                f"exceeds 8-bit range"
            )

        scrutinee_loc = self.parent._get_operand_location(instr.scrutinee)

        # Load scrutinee into A in m8. Same rationale as select_jump_table:
        # if m16 is live on entry the bounds check and SBC immediates run
        # 16-bit and silently corrupt the dispatch.
        # LUT dispatch expects m8 scrutinee
        self.parent._ensure_m8_mode()
        if not scrutinee_loc.is_hw('A'):
            self.parent._emit_load('LDA', scrutinee_loc)

        default_label = self.parent._get_unique_label()
        merge_label = self.parent._get_unique_label()
        table_label = self.parent._get_unique_label()

        # Base adjustment (8-bit, before any mode switch)
        if instr.base_value != 0:
            self._emit_implied(Opcode.SEC)
            self._emit_immediate(Opcode.SBC_IMMEDIATE, instr.base_value, "Compute index = scrutinee - base")
            self._emit_branch(Opcode.BCC, default_label, "Out of bounds (< base)")

        # Upper bounds check (8-bit)
        self._emit_immediate(Opcode.CMP_IMMEDIATE, table_size, "Check upper bound")
        self._emit_branch(Opcode.BCS, default_label, "Out of bounds (>= size)")

        # Switch to m16 to compute the index. TAX in m8 still transfers the
        # full 16-bit accumulator (B:A_low) into X; if B is nonzero on entry
        # the index is corrupted. AND #$00FF in m16 zeroes B; ASL doubles
        # the index in the same width for the u16 path. For the u8 path we
        # immediately drop back to m8 so the absolute,X load width matches.
        self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "m16 to zero B before TAX")
        self.parent.emitter.emit_accu_mode(16)
        self._emit_immediate(Opcode.AND_IMMEDIATE, 0x00FF, "Zero B (clear high byte)")
        if is_u16:
            self._emit_implied(Opcode.ASL, "Index *= 2 for word table")
        if not is_u16:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Back to m8 for byte LUT")
            self.parent.emitter.emit_accu_mode(8)

        self._emit_implied(Opcode.TAX)

        self.emitter.emit_instr(Opcode.LDA_ABSOLUTE_X, Address(table_label), "LUT lookup")
        self._emit_jump(Opcode.BRA, merge_label)

        # Default path
        self.emitter.emit_label(default_label)
        if is_u16:
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "m16 for default")
            self.parent.emitter.emit_accu_mode(16)
        self._emit_immediate(Opcode.LDA_IMMEDIATE, instr.default_value, "Default value")

        # Merge — store result
        self.emitter.emit_label(merge_label)
        dest_loc = self.parent._get_operand_location(instr.dest)
        if not dest_loc.is_hw('A'):
            self.parent._emit_store('STA', dest_loc)

        # u16: restore m8
        if is_u16:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore m8")
            self.parent.emitter.emit_accu_mode(8)

        # Block terminator: branch to merge block
        self._emit_jump(Opcode.BRA, self._block_label(instr.merge_target))

        # Inline table data (unreachable — after BRA)
        self.emitter.emit_label(table_label)
        if is_u16:
            for val in instr.values:
                self.emitter.emit_directive(f"    .DW ${val:04X}")
        else:
            for val in instr.values:
                self.emitter.emit_directive(f"    .DB ${val:02X}")

    # ========================================================================
    # Conditional Branch
    # ========================================================================

    # Map for reversing comparison when operands are swapped
    _REVERSED_COMPARISON = {
        '==': '==', '!=': '!=',
        '<': '>', '>': '<',
        '<=': '>=', '>=': '<=',
    }

    def select_cond_branch(self, instr: CondBranch):
        """
        Generate code for CondBranch instruction.

        Two modes:
        1. If condition is None: Branch based on flags from preceding Compare
        2. If condition is a vreg: Load condition and branch on zero/non-zero

        Args:
            instr: CondBranch instruction
        """
        is_signed = self._is_signed_comparison()

        # Consume compare-against-zero flag before it goes stale
        compare_rhs_zero = getattr(self.parent, '_compare_rhs_is_zero', False)
        self.parent._compare_rhs_is_zero = False

        # If the preceding Compare swapped operands (right in A, CMP left),
        # the flags represent (right - left) instead of (left - right).
        # Reverse the comparison to compensate.
        comparison = instr.comparison
        if getattr(self.parent, '_comparison_reversed', False):
            comparison = self._REVERSED_COMPARISON.get(comparison, comparison)
            self.parent._comparison_reversed = False

        if instr.condition is None:
            self._emit_flag_based_branch(instr, is_signed, comparison, compare_rhs_zero)
        else:
            self._emit_value_based_branch(instr)

    def _is_signed_comparison(self) -> bool:
        """Check if the last comparison was signed.

        A newtype answers for its payload — otherwise `struct Q10(i16)` would
        branch with BCC/BCS and compare negatives as large unsigned values.
        """
        if self.last_comparison_type is not None:
            from r65.compiler.hir.types import BasicTypeInfo, strip_newtype
            payload = strip_newtype(self.last_comparison_type)
            if isinstance(payload, BasicTypeInfo):
                return payload.name.startswith('i')
        return False

    # comparison kind -> (branch opcode, comment); each also emits BRA to false.
    _FLAG_BRANCH_OPS = {
        'bit7_set':              (Opcode.BMI, "Branch if bit 7 set"),
        'bit7_clear':            (Opcode.BPL, "Branch if bit 7 clear"),
        'bit6_set':              (Opcode.BVS, "Branch if bit 6 set"),
        'bit6_clear':            (Opcode.BVC, "Branch if bit 6 clear"),
        'status_carry_set':      (Opcode.BCS, "Branch if Carry set"),
        'status_carry_clear':    (Opcode.BCC, "Branch if Carry clear"),
        'status_zero_set':       (Opcode.BEQ, "Branch if Zero set"),
        'status_zero_clear':     (Opcode.BNE, "Branch if Zero clear"),
        'status_overflow_set':   (Opcode.BVS, "Branch if Overflow set"),
        'status_overflow_clear': (Opcode.BVC, "Branch if Overflow clear"),
        'status_negative_set':   (Opcode.BMI, "Branch if Negative set"),
        'status_negative_clear': (Opcode.BPL, "Branch if Negative clear"),
        'status_nonbranch_set':  (Opcode.BNE, "Branch if flag set (AND result != 0)"),
        'status_nonbranch_clear': (Opcode.BEQ, "Branch if flag clear (AND result == 0)"),
        '==':                    (Opcode.BEQ, "Branch if equal"),
        '!=':                    (Opcode.BNE, "Branch if not equal"),
    }

    def _emit_flag_based_branch(self, instr: CondBranch, is_signed: bool,
                                comparison: str = None,
                                compare_rhs_zero: bool = False):
        """Emit branch based on CPU flags from preceding Compare."""
        if comparison is None:
            comparison = instr.comparison
        true_target = self._block_label(instr.true_target)
        false_target = self._block_label(instr.false_target)

        # Optimize comparisons against zero.
        # When RHS is immediate 0, several simplifications apply:
        # - Unsigned: x>0 is x!=0, x<=0 is x==0, x>=0 is always true, x<0 always false
        # - Signed: no overflow possible (V=0) so N flag directly indicates sign,
        #   avoiding the complex BVC/EOR N-XOR-V trick
        if compare_rhs_zero and not is_signed and comparison in ('<', '<=', '>', '>='):
            result = self._emit_zero_comparison_branch(
                comparison, true_target, false_target
            )
            if result:
                return

        # BIT-based, STATUS-flag, and ==/!= comparisons all emit a single
        # conditional branch to true plus a BRA to false — table-driven.
        flag_branch = self._FLAG_BRANCH_OPS.get(comparison)
        if flag_branch is not None:
            opcode, comment = flag_branch
            self._emit_branch(opcode, true_target, comment)
            self._emit_jump(Opcode.BRA, false_target)
        # Relational comparisons need signed/unsigned-aware multi-branch helpers.
        elif comparison == '<':
            self._emit_less_than_branch(true_target, false_target, is_signed)
        elif comparison == '>=':
            self._emit_greater_equal_branch(true_target, false_target, is_signed)
        elif comparison == '>':
            self._emit_greater_than_branch(true_target, false_target, is_signed)
        elif comparison == '<=':
            self._emit_less_equal_branch(true_target, false_target, is_signed)
        else:
            raise InstructionSelectionError(
                f"Unsupported comparison type for flag-based branch: {comparison}", source_loc=self.parent._current_source_loc)

    def _emit_zero_comparison_branch(self, comparison: str,
                                     true_target: str, false_target: str) -> bool:
        """
        Emit optimized branches for unsigned comparisons against zero.

        When the right operand is immediate 0 and the type is unsigned:
        - x > 0 ≡ x != 0 (single BNE instead of BEQ+BCS)
        - x <= 0 ≡ x == 0 (single BEQ instead of BEQ+BCC)
        - x >= 0 is always true
        - x < 0 is always false

        Only Z flag is needed, which LDA sets correctly (allowing
        CMP #0 elision in compare_select).

        Returns:
            True if optimization was applied, False otherwise.
        """
        if comparison == '>':
            # x > 0 ≡ x != 0
            self._emit_branch(Opcode.BNE, true_target, "Branch if != 0 (unsigned > 0)")
            self._emit_jump(Opcode.BRA, false_target)
            return True
        elif comparison == '<=':
            # x <= 0 ≡ x == 0
            self._emit_branch(Opcode.BEQ, true_target, "Branch if == 0 (unsigned <= 0)")
            self._emit_jump(Opcode.BRA, false_target)
            return True
        elif comparison == '>=':
            # x >= 0 is always true for unsigned
            self._emit_jump(Opcode.BRA, true_target)
            return True
        elif comparison == '<':
            # x < 0 is always false for unsigned
            self._emit_jump(Opcode.BRA, false_target)
            return True

        return False

    # Signed branches:
    #   The compare leaves A holding the LHS, not (LHS - RHS), so the "EOR
    #   #$80 on overflow then test N" idiom (which needs the difference in A)
    #   is out. Instead a control-flow-only pattern picks BMI or BPL on V:
    #   the V flag says whether N was inverted by signed overflow - `N XOR V`
    #   with no A touch. That needs V from the subtraction itself, and
    #   CMP/CPX/CPY never set V, so a signed compare is emitted as SEC/SBC
    #   with A restored under PHP/PLP (CompareSelector._emit_signed_cmp).

    def _emit_less_than_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for < comparison."""
        if is_signed:
            # Signed less than: N XOR V = 1.
            #   V=0 → N is correct → BMI taken means less
            #   V=1 → N is inverted → BPL taken means less
            v_set_label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVS, v_set_label, "V=1 path: N is inverted")
            self._emit_branch(Opcode.BMI, true_target, "V=0: less if N=1 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
            self.emitter.emit_label(v_set_label)
            self._emit_branch(Opcode.BPL, true_target, "V=1: less if N=0 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
        else:
            # Unsigned less than: C flag clear
            self._emit_branch(Opcode.BCC, true_target, "Branch if less than (unsigned)")
            self._emit_jump(Opcode.BRA, false_target)

    def _emit_greater_equal_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for >= comparison."""
        if is_signed:
            # Signed >=: N XOR V = 0.
            #   V=0 → BPL taken means >=
            #   V=1 → BMI taken means >= (N inverted by overflow)
            v_set_label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVS, v_set_label, "V=1 path: N is inverted")
            self._emit_branch(Opcode.BPL, true_target, "V=0: >= if N=0 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
            self.emitter.emit_label(v_set_label)
            self._emit_branch(Opcode.BMI, true_target, "V=1: >= if N=1 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
        else:
            # Unsigned >=: C flag set
            self._emit_branch(Opcode.BCS, true_target, "Branch if >= (unsigned)")
            self._emit_jump(Opcode.BRA, false_target)

    def _emit_greater_than_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for > comparison."""
        if is_signed:
            # Signed >: (N XOR V = 0) AND Z = 0.
            self._emit_branch(Opcode.BEQ, false_target, "Skip if equal")
            v_set_label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVS, v_set_label, "V=1 path: N is inverted")
            self._emit_branch(Opcode.BPL, true_target, "V=0: > if N=0 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
            self.emitter.emit_label(v_set_label)
            self._emit_branch(Opcode.BMI, true_target, "V=1: > if N=1 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
        else:
            # Unsigned >: (C set) AND (Z clear)
            self._emit_branch(Opcode.BEQ, false_target, "Skip if equal")
            self._emit_branch(Opcode.BCS, true_target, "Branch if > (unsigned)")
            self._emit_jump(Opcode.BRA, false_target)

    def _emit_less_equal_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for <= comparison."""
        if is_signed:
            # Signed <=: (N XOR V = 1) OR Z = 1.
            self._emit_branch(Opcode.BEQ, true_target, "Branch if equal")
            v_set_label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVS, v_set_label, "V=1 path: N is inverted")
            self._emit_branch(Opcode.BMI, true_target, "V=0: <= if N=1 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
            self.emitter.emit_label(v_set_label)
            self._emit_branch(Opcode.BPL, true_target, "V=1: <= if N=0 (signed)")
            self._emit_jump(Opcode.BRA, false_target)
        else:
            # Unsigned <=: (C clear) OR (Z set)
            self._emit_branch(Opcode.BEQ, true_target, "Branch if equal")
            self._emit_branch(Opcode.BCC, true_target, "Branch if less than")
            self._emit_jump(Opcode.BRA, false_target)

    def _emit_value_based_branch(self, instr: CondBranch):
        """Emit branch based on condition value (zero/non-zero)."""
        cond_loc = self.parent._get_operand_location(instr.condition)
        # Skip load if condition is already in A (from bitwise optimization)
        # The Z flag is already set from the previous BinaryOp
        if cond_loc.is_hw('A'):
            pass  # Z flag already set from previous operation
        else:
            # Ensure accumulator mode matches the condition value's size before
            # loading. Without this, loading a u8/bool variable in m16 mode reads
            # 2 bytes — including garbage from the byte above the local frame —
            # causing BEQ/BNE to branch on a corrupted 16-bit value.
            from r65.compiler.codegen.type_utils import get_vreg_size
            from r65.compiler.mir.nodes import VirtualRegister
            if isinstance(instr.condition, VirtualRegister):
                cond_size = get_vreg_size(instr.condition)
                if cond_size == 1:
                    self.parent._ensure_m8_mode()
                elif cond_size == 2:
                    self.parent._ensure_m16_mode()
            self.parent._emit_load('LDA', cond_loc)

        true_target = self._block_label(instr.true_target)
        false_target = self._block_label(instr.false_target)

        if instr.comparison == '!=':
            self._emit_branch(Opcode.BEQ, false_target, "Branch if zero")
            self._emit_jump(Opcode.BRA, true_target)
        elif instr.comparison == '==':
            self._emit_branch(Opcode.BNE, false_target, "Branch if non-zero")
            self._emit_jump(Opcode.BRA, true_target)
        else:
            # For other comparisons on boolean values, treat as != 0
            self._emit_branch(Opcode.BEQ, false_target)
            self._emit_jump(Opcode.BRA, true_target)

    # ========================================================================
    # Return Instruction
    # ========================================================================

    def select_return(self, instr: Return):
        """
        Generate code for Return instruction.

        Handles loading return values into appropriate registers before returning.
        Mode switching for exit mode happens BEFORE loading return values so that
        16-bit return values are loaded in the correct mode.

        When a shared epilogue label is set (multiple returns in function),
        branches to the shared label instead of emitting the epilogue inline.

        Args:
            instr: Return instruction
        """
        # Switch to exit mode BEFORE loading return values
        # This ensures u16 return values are loaded in m16 mode
        self._switch_to_exit_mode()

        self._emit_return_values(instr)

        # If shared epilogue is active, branch there instead of inline epilogue
        shared_label = getattr(self, 'shared_epilogue_label', None)
        if shared_label:
            self._emit_instr(Opcode.BRA, Address(shared_label))
            return

        # Use consolidated emit_epilogue from FunctionCodeGenerator
        # Note: emit_epilogue no longer does mode switching (we did it above)
        if self.parent.func_gen:
            self.parent.func_gen.emit_epilogue(self.current_function, self.parent.reg_alloc)
        else:
            # Fallback to inline methods if func_gen not available
            self._emit_preserved_register_restores()
            self._emit_dbr_restore()
            # Mode restore is now handled by _switch_to_exit_mode above

        self._emit_return_instruction()

    def _switch_to_exit_mode(self):
        """
        Switch to function's exit mode before loading return values.

        Checks the CURRENT tracked mode (not entry mode) and switches if needed.
        This must happen BEFORE loading return values so 16-bit values are
        loaded correctly.

        For entry functions with a u16 register alias for A, we preserve the
        m16 mode since the user explicitly requested 16-bit storage via
        `let x @ A : u16 = ...`.
        """
        from r65.compiler.typeck.processor_mode import ModeState
        from r65.compiler.codegen.constants import M_FLAG

        if not self.current_function:
            return

        exit_mode = self.current_function.exit_m_mode or ModeState.M8

        # For entry functions, check if there's a u16 alias for A register
        # If so, preserve the 16-bit mode at exit
        if self.current_function.is_entry:
            alias_tracker = getattr(self.current_function, 'alias_tracker', None)
            if alias_tracker:
                a_binding_type = alias_tracker.get_register_binding_type('A')
                if is_16bit_scalar(a_binding_type):
                    # Entry function with u16 @ A binding - stay in m16 mode
                    return

        # Check current tracked mode vs required exit mode
        current_mode_bits = self.parent.emitter.get_accu_mode()
        exit_is_m16 = (exit_mode == ModeState.M16)

        if current_mode_bits is None:
            # Mode unknown (e.g., after inline asm) - unconditionally restore
            if exit_is_m16:
                self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "Restore m16 for return (mode unknown after asm!)")
                self.parent.emitter.emit_accu_mode(16)
            else:
                self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore m8 for return (mode unknown after asm!)")
                self.parent.emitter.emit_accu_mode(8)
            return

        current_is_m16 = (current_mode_bits == 16)

        if current_is_m16 == exit_is_m16:
            return  # Already in correct mode

        if exit_is_m16:
            # Need to switch to m16 for u16 return value
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "Switch to m16 for return")
            self.parent.emitter.emit_accu_mode(16)
        else:
            # Need to switch to m8 for u8 return value
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Switch to m8 for return")
            self.parent.emitter.emit_accu_mode(8)

    def _get_return_register_order(self):
        """
        Get the return register order for the current function.

        Uses B as second return register for (u8, u8) tuples in m8 mode.

        Returns:
            List of register names, e.g. ['A', 'B', 'X', 'Y'] or ['A', 'X', 'Y']
        """
        from r65.compiler.codegen.constants import get_return_registers

        if not self.current_function:
            return ['A', 'X', 'Y']
        return get_return_registers(
            self.current_function.return_type,
            self.current_function.entry_m_mode
        )

    def _emit_return_values(self, instr: Return):
        """Emit return values. Delegates to ABIModel.emit_return_values."""
        self.parent.abi_model.emit_return_values(self, instr)

    def _emit_preserved_register_restores(self):
        """Restore preserved registers in reverse order."""
        if not (self.current_function and self.current_function.preserves_attr):
            return

        preserved_regs = self.current_function.preserves_attr.registers
        pop_order = ['DBR', 'D', 'Y', 'X', 'A', 'STATUS']

        for reg in pop_order:
            if reg in preserved_regs:
                pull_opcode = PULL_OPCODES.get(reg)
                if pull_opcode:
                    self._emit_implied(pull_opcode, f"Restore {reg}")

    def _emit_dbr_restore(self):
        """Restore DBR for databank=inline functions."""
        if not (self.current_function and self.current_function.is_far and self.current_function.mode_attr):
            return

        from r65.compiler.hir.attributes import DataBankMode
        if self.current_function.mode_attr.databank == DataBankMode.INLINE:
            self._emit_implied(Opcode.PLB, "Restore data bank")

    def _emit_mode_restore(self):
        """Restore processor mode for m16 functions.

        In the simplified mode system, if a function runs in m16 mode,
        we need to restore m8 mode before returning so the caller
        (which is in m8 mode) can continue correctly.
        """
        if not self.current_function:
            return

        from r65.compiler.typeck.processor_mode import ModeState
        from r65.compiler.codegen.constants import M_FLAG

        if self.current_function.entry_m_mode == ModeState.M16:
            self.parent._force_accu_mode(8, "Restore m8 mode")

    def _emit_return_instruction(self):
        """Emit appropriate return instruction (RTL, RTS, or infinite loop).

        For functions returning ! (never type) or entry functions, we emit a
        branch-to-self infinite loop instead of a return instruction since
        there's no valid return address.

        For functions with stack parameters, emits callee cleanup code before
        the return instruction.
        """
        from r65.compiler.hir.types import NeverTypeInfo

        # Never type or entry functions have no valid return address
        if self.current_function and (
            isinstance(self.current_function.return_type, NeverTypeInfo)
            or self.current_function.is_entry
        ):
            halt_label = self.parent._get_unique_label()
            self.emitter.emit_label(halt_label)
            self._emit_jump(Opcode.BRA, halt_label, "No return - halt")
            return

        # Emit callee cleanup for stack parameters before return
        self._emit_stack_param_cleanup()

        if self.current_function and self.current_function.is_far:
            self._emit_implied(Opcode.RTL)
        else:
            self._emit_implied(Opcode.RTS)

    def _get_stack_param_bytes(self) -> int:
        """
        Calculate total bytes of stack parameters for callee cleanup.

        With caller-owned outgoing args convention (Default ABI), the callee does
        NOT clean up stack parameters — they live in the caller's outgoing area.

        With Pascal ABI, the callee DOES clean up stack parameters (but NOT
        result space — that stays on stack for the caller to pull).

        Returns:
            0 for Default ABI, total_param_bytes for Pascal ABI
        """
        from r65.compiler.codegen.abi_model import ABIKind
        if (self.parent.abi_model and
            self.parent.abi_model.kind == ABIKind.PASCAL and
            self.current_function):
            return self.current_function.pascal_total_param_bytes
        return 0

    def _get_return_register_count(self) -> int:
        """
        Get the number of registers used for return values.

        Based on return type: 0 for void/unit, 1 for single value,
        2+ for tuple returns.

        Returns:
            Number of registers (0-3) used for return values
        """
        if not self.current_function or not self.current_function.return_type:
            return 0

        from r65.compiler.hir.types import NeverTypeInfo, MultiReturnTypeInfo, BasicTypeInfo

        ret_type = self.current_function.return_type

        # Check for void/never types (no return value)
        if isinstance(ret_type, NeverTypeInfo):
            return 0
        if isinstance(ret_type, BasicTypeInfo) and ret_type.name == 'void':
            return 0

        # Tuple types return multiple values
        if isinstance(ret_type, MultiReturnTypeInfo):
            return len(ret_type.element_types)

        # Single value return
        return 1

    def _function_returns_b(self) -> bool:
        """
        Check if the current function returns a value in the B register.

        B register returns happen in two cases:
        1. Explicit: return B; / return A, B;  (HardwareRegister('B') in Return values)
        2. Implicit: function return type uses B ordering (u8, u8 tuple in m8 mode)

        Returns:
            True if B is used as a return register, False otherwise
        """
        if not self.current_function:
            return False

        # Check if the return register ordering includes B
        return_registers = self._get_return_register_order()
        if 'B' in return_registers:
            # Check if the function actually has enough return values to use B
            from r65.compiler.mir.nodes import Return, HardwareRegister
            for block in self.current_function.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Return) and len(instr.values) >= 2:
                        # B is at index 1 in the register order, and we have >= 2 values
                        return True

        # Also check for explicit B register returns
        from r65.compiler.mir.nodes import Return, HardwareRegister
        for block in self.current_function.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Return):
                    for value in instr.values:
                        if isinstance(value, HardwareRegister) and value.name == 'B':
                            return True
        return False

    def _emit_stack_param_cleanup(self):
        """
        Emit callee cleanup code for stack parameters and frame deallocation.

        This combines:
        1. Frame deallocation (if function allocated a stack frame)
        2. Stack parameter cleanup (if function has stack parameters)

        These are combined into a single SP adjustment to avoid issues with
        return address offsets when they're done separately.

        Strategy varies based on parameter count and return registers:
        - 0 bytes: no cleanup needed
        - Near functions: PLX, adjust SP, PHX, (RTS will follow)
        - Far functions: more complex due to 3-byte return address

        The cleanup moves the return address from its current position
        (S+frame_size+1) to its new position (S+total_cleanup+1), then
        adjusts SP past frame + parameters.
        """
        stack_param_bytes = self._get_stack_param_bytes()

        # Get frame size from register allocator
        frame_size = 0
        if self.parent.reg_alloc and self.parent.reg_alloc.has_frame_allocation:
            frame_size = self.parent.reg_alloc.frame_size

        # Include far self bytes pushed before frame alloc.
        # D=S path: PHB + PHY = 3 bytes (caller DBR + self addr).
        # Scratch path: PHB = 1 byte (caller DBR; self addr lives in zeropage).
        # These sit between the frame and the return address on the stack.
        far_self_bytes = 0
        if self.current_function and getattr(self.current_function, 'self_far_uses_d_equals_s', False):
            far_self_bytes = 3
        elif self.current_function and getattr(self.current_function, 'self_far_uses_scratch', False):
            far_self_bytes = 1

        # Total bytes to clean up: frame + far_self + stack parameters
        total_cleanup_bytes = frame_size + far_self_bytes + stack_param_bytes

        if total_cleanup_bytes == 0:
            return

        return_count = self._get_return_register_count()
        is_far = self.current_function and self.current_function.is_far

        # When B is a return register, it doesn't consume an A/X/Y slot.
        # Adjust return_count for stack cleanup purposes: B is part of
        # the accumulator high byte, so A/X/Y are still free for address manipulation.
        # For cleanup, what matters is how many of A/X/Y are occupied.
        returns_b = self._function_returns_b()
        if returns_b:
            effective_return_count = return_count - 1
        else:
            effective_return_count = return_count

        # Include far_self_bytes in frame_size for cleanup — these bytes sit
        # between the frame and the return address, like extra frame allocation
        cleanup_frame_size = frame_size + far_self_bytes

        if is_far:
            self._emit_far_stack_cleanup(cleanup_frame_size, stack_param_bytes, effective_return_count, returns_b)
        else:
            self._emit_near_stack_cleanup(cleanup_frame_size, stack_param_bytes, effective_return_count, returns_b)

    # ------------------------------------------------------------------
    # Shared pieces of the SP adjustments below. Each moves SP with
    # TSC/CLC/ADC/TCS, which needs A, so a returned A is parked first.
    # ------------------------------------------------------------------

    def _choose_a_save(self, return_count: int, busy=()) -> 'str | None':
        """Where to park a returned A across an SP adjustment.

        'X' or 'Y' when one is free - not carrying a return value, not *busy*,
        and not just restored by #[preserves] - else 'stack'; None when A
        carries no return value.
        """
        if return_count < 1:
            return None
        avoid = set(busy)
        if self.current_function and self.current_function.preserves_attr:
            avoid |= set(self.current_function.preserves_attr.registers)
        free = ('X', 'Y')[return_count - 1:] if return_count <= 2 else ()
        return next((reg for reg in free if reg not in avoid), 'stack')

    def _save_a(self, method) -> int:
        """Park A by *method*; returns the bytes pushed (0 unless 'stack')."""
        if method == 'X':
            self._emit_implied(Opcode.TAX, "Save return value A in X")
        elif method == 'Y':
            self._emit_implied(Opcode.TAY, "Save return value A in Y")
        elif method == 'stack':
            self._emit_implied(Opcode.PHA, "Save return value A")
            return 2 if self.parent.emitter.get_accu_mode() == 16 else 1
        return 0

    def _restore_a(self, method):
        if method == 'X':
            self._emit_implied(Opcode.TXA, "Restore return value A from X")
        elif method == 'Y':
            self._emit_implied(Opcode.TYA, "Restore return value A from Y")
        elif method == 'stack':
            self._emit_implied(Opcode.PLA, "Restore return value A")

    def _emit_sp_add(self, amount: int, cleaned: int):
        """SP += amount, in m16. The emitter's tracked mode may not match the
        runtime one at an epilogue merge point, so the REP is unconditional."""
        from r65.compiler.codegen.constants import M_FLAG
        self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for SP adjust")
        self.parent.emitter.emit_accu_mode(16)
        self._emit_implied(Opcode.TSC, "SP to A")
        self._emit_implied(Opcode.CLC)
        self._emit_immediate(Opcode.ADC_IMMEDIATE, amount, f"Adjust past {cleaned} bytes")
        self._emit_implied(Opcode.TCS, "A to SP")

    def _restore_exit_accu_mode(self):
        """Return to the function's exit width before A is restored: TXA/TYA
        in m8 would move only the low byte of a 16-bit return value."""
        from r65.compiler.codegen.constants import M_FLAG
        from r65.compiler.typeck.processor_mode import ModeState
        exit_mode = getattr(self.current_function, 'exit_m_mode', None) or ModeState.M8
        if exit_mode == ModeState.M16:
            self.parent.emitter.emit_accu_mode(16)
        else:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)

    def _emit_near_stack_cleanup(self, frame_size: int, stack_param_bytes: int, return_count: int, returns_b: bool = False):
        """
        Emit stack cleanup for near functions (2-byte return address).

        Args:
            frame_size: Number of bytes allocated for stack frame (0 if no frame)
            stack_param_bytes: Number of parameter bytes to clean
            return_count: Number of registers used for return (0-3)
            returns_b: True if function returns a value in B register
        """
        # Frame-only cleanup (no stack params): the return address need not
        # move. PLA-based deallocation preserves B, X, Y and DBR; FixedStack
        # always uses it.
        if stack_param_bytes == 0 and frame_size > 0:
            self.parent.abi_model.emit_frame_dealloc(self, frame_size, return_count)
            return
        # No frame and a register to spare for both the return address and A:
        # pull the address, adjust SP, push it back. Otherwise (a frame over
        # the address, or A would have to go on the stack the adjustment
        # frees) move the address in place with stack-relative loads/stores.
        addr_reg = 'X' if return_count <= 1 else 'Y'
        method = self._choose_a_save(return_count, busy={addr_reg})
        if frame_size > 0 or return_count > 2 or method == 'stack':
            self._emit_ret_addr_move_cleanup(frame_size, stack_param_bytes, return_count)
            return

        self._emit_implied(Opcode.PLX if addr_reg == 'X' else Opcode.PLY,
                           f"Pop return address into {addr_reg}")
        self._save_a(method)
        total = frame_size + stack_param_bytes
        self._emit_sp_add(total, total)
        self._restore_exit_accu_mode()
        self._restore_a(method)
        self._emit_implied(Opcode.PHX if addr_reg == 'X' else Opcode.PHY,
                           "Push return address back")

    def _emit_ret_addr_move_cleanup(self, frame_size: int, stack_param_bytes: int, return_count: int):
        """Near cleanup that moves the 2-byte return address past the area
        being freed, then adjusts SP. Return address is at S+frame_size+1."""
        total = frame_size + stack_param_bytes
        method = self._choose_a_save(return_count)
        pushed = self._save_a(method)
        self._emit_immediate(Opcode.REP_IMMEDIATE, 0x20, "16-bit A for cleanup")
        self.parent.emitter.emit_accu_mode(16)
        self._emit_stack_relative(Opcode.LDA_STACK, frame_size + 1 + pushed, "Load return address")
        self._emit_stack_relative(Opcode.STA_STACK, total + 1 + pushed, "Store past cleanup area")
        self._emit_sp_add(total + pushed, total)
        self._restore_exit_accu_mode()
        self._restore_a(method)

    def emit_sp_adjust_preserving_a(self, adjust_bytes: int, return_count: int, current_mode: int):
        """
        Emit SP adjustment that preserves return value in A.

        Args:
            adjust_bytes: Number of bytes to add to SP
            return_count: Number of registers used for return (0-3)
            current_mode: Current accumulator mode (8 or 16)
        """
        method = self._choose_a_save(return_count)
        pushed = self._save_a(method)
        self._emit_sp_add(adjust_bytes + pushed, adjust_bytes)
        self._restore_exit_accu_mode()
        self._restore_a(method)

    def emit_pla_frame_dealloc(self, frame_size: int, return_count: int):
        """
        Deallocate a small stack frame using PLA instructions.

        PLA pops 1 byte per call (in m8 mode), clobbering A's low byte but
        preserving B (accumulator high byte), X, Y, and DBR. This makes it
        safe for B returns (TSC/ADC/TCS would destroy B) and for functions
        with preserved registers (TAX/TAY would clobber them).

        When A is a return value (return_count >= 1), we save A into the
        top frame byte via STA frame_size,S. After all PLAs, the last PLA
        naturally restores A from that byte.

        Args:
            frame_size: Number of bytes to deallocate (should be <= 4)
            return_count: Number of A/X/Y registers used for return (0-3),
                         NOT counting B
        """
        from r65.compiler.codegen.constants import M_FLAG

        current_mode = self.parent.emitter.get_accu_mode()

        # Need m8 mode for 1-byte-per-PLA deallocation
        if current_mode != 8:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "m8 for frame dealloc")
            self.parent.emitter.emit_accu_mode(8)

        # If A is a return value, save it into the top frame byte.
        # After all PLAs pop through the frame, the last PLA restores A.
        if return_count >= 1:
            self._emit_stack_relative(Opcode.STA_STACK, frame_size,
                                      "Save A return to top of frame")

        # PLA pops 1 byte each, deallocating the frame
        for _ in range(frame_size):
            self._emit_implied(Opcode.PLA, f"Deallocate frame ({frame_size} bytes)")

        # Restore mode if we changed it
        if current_mode != 8:
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "Restore m16")
            self.parent.emitter.emit_accu_mode(16)

    def _emit_far_stack_cleanup(self, frame_size: int, stack_param_bytes: int, return_count: int, returns_b: bool = False):
        """
        Emit stack cleanup for far functions (3-byte return address).

        Args:
            frame_size: Number of bytes allocated for stack frame (0 if no frame)
            stack_param_bytes: Number of parameter bytes to clean
            return_count: Number of registers used for return (0-3)
            returns_b: True if function returns a value in B register
        """
        from r65.compiler.codegen.constants import M_FLAG

        if stack_param_bytes == 0 and frame_size > 0:
            self.parent.abi_model.emit_frame_dealloc(self, frame_size, return_count)
            return

        total = frame_size + stack_param_bytes
        method = self._choose_a_save(return_count)
        pushed = self._save_a(method)
        # Return address: addr_lo, addr_hi at S+frame_size+1, bank after.
        ret_addr_offset = frame_size + 1 + pushed
        bank_offset = frame_size + 3 + pushed
        store_offset = total + 1 + pushed

        # Move the bank byte first: the 16-bit address store below covers
        # store_offset+1, which may be where the bank still is.
        if self.parent.emitter.get_accu_mode() != 8:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "8-bit A for bank")
            self.parent.emitter.emit_accu_mode(8)
        self._emit_stack_relative(Opcode.LDA_STACK, bank_offset, "Load bank byte")
        self._emit_stack_relative(Opcode.STA_STACK, store_offset + 2, "Store bank past cleanup area")
        self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for far cleanup")
        self.parent.emitter.emit_accu_mode(16)
        self._emit_stack_relative(Opcode.LDA_STACK, ret_addr_offset, "Load return address")
        self._emit_stack_relative(Opcode.STA_STACK, store_offset, "Store past cleanup area")
        self._emit_sp_add(total + pushed, total)
        self._restore_exit_accu_mode()
        self._restore_a(method)

    def _emit_stack_relative(self, opcode: Opcode, offset: int, comment: str = None):
        """Emit a stack-relative instruction."""
        from r65.compiler.codegen.asm_nodes import StackOffset
        self.emitter.emit_instr(opcode, StackOffset(offset), comment)
