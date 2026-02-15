"""
Control flow instruction selector: Jump, Branch, Return.

Handles control flow instruction generation including conditional branches
with proper signed/unsigned comparison handling.
"""

from r65.compiler.mir.nodes import Jump, JumpTable, LookupTable, CondBranch, Return
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Address
from r65.compiler.codegen.base_selector import BaseSelector


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
        scrutinee_loc = self.parent._get_operand_location(instr.scrutinee)

        # Load scrutinee into A (if not already there)
        if scrutinee_loc.kind == LocationKind.HARDWARE and scrutinee_loc.hw_register == 'A':
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

        # ASL A - multiply index by 2 for word-sized table entries
        self._emit_implied(Opcode.ASL, "Index *= 2 for word table")

        # TAX - move to X for indexed addressing
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
        scrutinee_loc = self.parent._get_operand_location(instr.scrutinee)

        # Load scrutinee into A (always 8-bit at this point)
        if not (scrutinee_loc.kind == LocationKind.HARDWARE and scrutinee_loc.hw_register == 'A'):
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

        # u16: ASL to double index for word entries
        if is_u16:
            self._emit_implied(Opcode.ASL, "Index *= 2 for word table")

        self._emit_implied(Opcode.TAX)

        # u16: switch to m16 for 16-bit load
        if is_u16:
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "m16 for LUT word load")
            self.parent.emitter.emit_accu_mode(16)

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
        if not (dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A'):
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

        # If the preceding Compare swapped operands (right in A, CMP left),
        # the flags represent (right - left) instead of (left - right).
        # Reverse the comparison to compensate.
        comparison = instr.comparison
        if getattr(self.parent, '_comparison_reversed', False):
            comparison = self._REVERSED_COMPARISON.get(comparison, comparison)
            self.parent._comparison_reversed = False

        if instr.condition is None:
            self._emit_flag_based_branch(instr, is_signed, comparison)
        else:
            self._emit_value_based_branch(instr)

    def _is_signed_comparison(self) -> bool:
        """Check if the last comparison was signed."""
        if self.last_comparison_type is not None:
            from r65.compiler.hir.types import BasicTypeInfo
            if isinstance(self.last_comparison_type, BasicTypeInfo):
                return self.last_comparison_type.name.startswith('i')
        return False

    def _emit_flag_based_branch(self, instr: CondBranch, is_signed: bool,
                                comparison: str = None):
        """Emit branch based on CPU flags from preceding Compare."""
        if comparison is None:
            comparison = instr.comparison
        true_target = self._block_label(instr.true_target)
        false_target = self._block_label(instr.false_target)

        # Handle BIT-based comparisons
        if comparison == 'bit7_set':
            self._emit_branch(Opcode.BMI, true_target, "Branch if bit 7 set")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'bit7_clear':
            self._emit_branch(Opcode.BPL, true_target, "Branch if bit 7 clear")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'bit6_set':
            self._emit_branch(Opcode.BVS, true_target, "Branch if bit 6 set")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'bit6_clear':
            self._emit_branch(Opcode.BVC, true_target, "Branch if bit 6 clear")
            self._emit_jump(Opcode.BRA, false_target)
        # Handle STATUS flag comparisons (branchable flags)
        elif comparison == 'status_carry_set':
            self._emit_branch(Opcode.BCS, true_target, "Branch if Carry set")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_carry_clear':
            self._emit_branch(Opcode.BCC, true_target, "Branch if Carry clear")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_zero_set':
            self._emit_branch(Opcode.BEQ, true_target, "Branch if Zero set")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_zero_clear':
            self._emit_branch(Opcode.BNE, true_target, "Branch if Zero clear")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_overflow_set':
            self._emit_branch(Opcode.BVS, true_target, "Branch if Overflow set")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_overflow_clear':
            self._emit_branch(Opcode.BVC, true_target, "Branch if Overflow clear")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_negative_set':
            self._emit_branch(Opcode.BMI, true_target, "Branch if Negative set")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_negative_clear':
            self._emit_branch(Opcode.BPL, true_target, "Branch if Negative clear")
            self._emit_jump(Opcode.BRA, false_target)
        # Handle STATUS flag comparisons (non-branchable flags after PHP; PLA; AND #mask)
        elif comparison == 'status_nonbranch_set':
            self._emit_branch(Opcode.BNE, true_target, "Branch if flag set (AND result != 0)")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == 'status_nonbranch_clear':
            self._emit_branch(Opcode.BEQ, true_target, "Branch if flag clear (AND result == 0)")
            self._emit_jump(Opcode.BRA, false_target)
        # Handle comparison operators
        elif comparison == '==':
            self._emit_branch(Opcode.BEQ, true_target, "Branch if equal")
            self._emit_jump(Opcode.BRA, false_target)
        elif comparison == '!=':
            self._emit_branch(Opcode.BNE, true_target, "Branch if not equal")
            self._emit_jump(Opcode.BRA, false_target)
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
                f"Unsupported comparison type for flag-based branch: {comparison}")

    def _emit_less_than_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for < comparison."""
        if is_signed:
            # Signed less than: N XOR V = 1
            label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVC, label, "Skip if no overflow")
            self._emit_immediate(Opcode.EOR_IMMEDIATE, 0x80, "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self._emit_branch(Opcode.BMI, true_target, "Branch if less than (signed)")
            self._emit_jump(Opcode.BRA, false_target)
        else:
            # Unsigned less than: C flag clear
            self._emit_branch(Opcode.BCC, true_target, "Branch if less than (unsigned)")
            self._emit_jump(Opcode.BRA, false_target)

    def _emit_greater_equal_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for >= comparison."""
        if is_signed:
            # Signed >= : N XOR V = 0
            label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVC, label, "Skip if no overflow")
            self._emit_immediate(Opcode.EOR_IMMEDIATE, 0x80, "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self._emit_branch(Opcode.BPL, true_target, "Branch if >= (signed)")
            self._emit_jump(Opcode.BRA, false_target)
        else:
            # Unsigned >=: C flag set
            self._emit_branch(Opcode.BCS, true_target, "Branch if >= (unsigned)")
            self._emit_jump(Opcode.BRA, false_target)

    def _emit_greater_than_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for > comparison."""
        if is_signed:
            # Signed >: (N XOR V = 0) AND Z = 0
            self._emit_branch(Opcode.BEQ, false_target, "Skip if equal")
            label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVC, label, "Skip if no overflow")
            self._emit_immediate(Opcode.EOR_IMMEDIATE, 0x80, "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self._emit_branch(Opcode.BPL, true_target, "Branch if > (signed)")
            self._emit_jump(Opcode.BRA, false_target)
        else:
            # Unsigned >: (C set) AND (Z clear)
            self._emit_branch(Opcode.BEQ, false_target, "Skip if equal")
            self._emit_branch(Opcode.BCS, true_target, "Branch if > (unsigned)")
            self._emit_jump(Opcode.BRA, false_target)

    def _emit_less_equal_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for <= comparison."""
        if is_signed:
            # Signed <=: (N XOR V = 1) OR Z = 1
            self._emit_branch(Opcode.BEQ, true_target, "Branch if equal")
            label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVC, label, "Skip if no overflow")
            self._emit_immediate(Opcode.EOR_IMMEDIATE, 0x80, "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self._emit_branch(Opcode.BMI, true_target, "Branch if <= (signed)")
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
        if cond_loc.kind == LocationKind.HARDWARE and cond_loc.hw_register == 'A':
            pass  # Z flag already set from previous operation
        else:
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

        Args:
            instr: Return instruction
        """
        # Switch to exit mode BEFORE loading return values
        # This ensures u16 return values are loaded in m16 mode
        self._switch_to_exit_mode()

        self._emit_return_values(instr)

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
                if a_binding_type and hasattr(a_binding_type, 'name'):
                    if a_binding_type.name in ('u16', 'i16'):
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

    # Mappings from register name to pull opcodes
    _PULL_OPCODES = {
        'A': Opcode.PLA, 'X': Opcode.PLX, 'Y': Opcode.PLY,
        'STATUS': Opcode.PLP, 'P': Opcode.PLP,
        'D': Opcode.PLD, 'DBR': Opcode.PLB, 'B': Opcode.PLB,
    }

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
        """Load return values into appropriate registers.

        Values are loaded in reverse order so that transfers
        through A (needed for stack-relative X/Y loads) don't clobber
        the final A value. B is handled via XBA.
        """
        if not instr.values:
            return

        return_registers = self._get_return_register_order()
        if len(instr.values) > len(return_registers):
            raise InstructionSelectionError(
                f"Too many return values (max {len(return_registers)})")

        # Process in reverse order to avoid clobbering A
        # Reverse order: Y first, then X, then B (XBA to store), then A last
        for i in range(len(instr.values) - 1, -1, -1):
            value = instr.values[i]
            target_reg = return_registers[i]
            value_loc = self.parent._get_operand_location(value)

            if value_loc.kind == LocationKind.RETURN_SINKABLE:
                # Deferred load: emit the load directly into the target register
                src_loc = self.parent._get_operand_location(value_loc.source_location)
                if target_reg == 'B':
                    self.parent._emit_load('LDA', src_loc)
                    self.parent._store_to_b_from_a()
                elif target_reg in ('X', 'Y'):
                    self.parent._emit_load('LDA', src_loc)
                    if target_reg == 'X':
                        self._emit_implied(Opcode.TAX)
                    else:
                        self._emit_implied(Opcode.TAY)
                else:  # 'A'
                    self.parent._emit_load('LDA', src_loc)
                continue
            elif value_loc.kind == LocationKind.HARDWARE and value_loc.hw_register == target_reg:
                pass  # Already in correct register
            elif target_reg == 'B':
                # B return: load value into A, then XBA to store in B
                if value_loc.kind == LocationKind.HARDWARE and value_loc.hw_register == 'A':
                    # Value already in A, just XBA
                    self.parent._store_to_b_from_a()
                elif value_loc.kind == LocationKind.HARDWARE:
                    self.parent._emit_register_transfer(value_loc.hw_register, 'A')
                    self.parent._store_to_b_from_a()
                else:
                    self.parent._emit_load('LDA', value_loc)
                    self.parent._store_to_b_from_a()
            elif value_loc.kind == LocationKind.HARDWARE:
                self.parent._emit_register_transfer(value_loc.hw_register, target_reg)
            elif target_reg in ('X', 'Y') and value_loc.kind == LocationKind.STACK:
                # Handle stack-relative addressing: LDX/LDY don't support sr,S mode
                self.parent._emit_load('LDA', value_loc)
                if target_reg == 'X':
                    self._emit_implied(Opcode.TAX, "Transfer to X (no LDX sr,S)")
                else:
                    self._emit_implied(Opcode.TAY, "Transfer to Y (no LDY sr,S)")
            else:
                # Use parent's _emit_load method with appropriate mnemonic
                load_mnem = {'A': 'LDA', 'X': 'LDX', 'Y': 'LDY'}.get(target_reg, 'LDA')
                self.parent._emit_load(load_mnem, value_loc)

    def _emit_preserved_register_restores(self):
        """Restore preserved registers in reverse order."""
        if not (self.current_function and self.current_function.preserves_attr):
            return

        preserved_regs = self.current_function.preserves_attr.registers
        pop_order = ['DBR', 'D', 'Y', 'X', 'A', 'STATUS']

        for reg in pop_order:
            if reg in preserved_regs:
                pull_opcode = self._PULL_OPCODES.get(reg)
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
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore m8 mode")

    def _emit_return_instruction(self):
        """Emit appropriate return instruction (RTL, RTS, or WAI).

        For functions returning ! (never type) or entry functions, we emit WAI
        instead of a return instruction since there's no valid return address.

        For functions with stack parameters, emits callee cleanup code before
        the return instruction.
        """
        from r65.compiler.hir.types import NeverTypeInfo

        # Never type or entry functions have no valid return address
        if self.current_function and (
            isinstance(self.current_function.return_type, NeverTypeInfo)
            or self.current_function.is_entry
        ):
            self._emit_implied(Opcode.WAI, "No return - wait for interrupt")
            return

        # Emit callee cleanup for stack parameters before return
        self._emit_stack_param_cleanup()

        if self.current_function and self.current_function.is_far:
            self._emit_implied(Opcode.RTL)
        else:
            self._emit_implied(Opcode.RTS)

    def _get_stack_param_bytes(self) -> int:
        """
        Calculate total bytes of stack parameters for current function.

        Stack parameters are those with no binding (binding is None)
        and not promoted to scratch parameters.
        Register-bound parameters have RegisterBinding, variable-bound
        have VariableBinding.

        Returns:
            Number of bytes passed via stack parameters
        """
        from r65.compiler.codegen.type_utils import get_type_size

        if not self.current_function:
            return 0

        scratch_addrs = self.current_function.scratch_param_addrs
        is_trait_method = getattr(self.current_function, 'is_trait_method', False)
        total_bytes = 0
        for i, param in enumerate(self.current_function.parameters):
            # Skip self parameter for trait methods (passed in Y, not on stack)
            if is_trait_method and i == 0 and param.name == 'self':
                continue
            # Stack parameters have no binding (binding is None)
            # and are not promoted to scratch
            if param.binding is None and i not in scratch_addrs:
                total_bytes += get_type_size(param.param_type)

        return total_bytes

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

        from r65.compiler.hir.types import NeverTypeInfo, TupleTypeInfo, BasicTypeInfo

        ret_type = self.current_function.return_type

        # Check for void/never types (no return value)
        if isinstance(ret_type, NeverTypeInfo):
            return 0
        if isinstance(ret_type, BasicTypeInfo) and ret_type.name == 'void':
            return 0

        # Tuple types return multiple values
        if isinstance(ret_type, TupleTypeInfo):
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

        # Total bytes to clean up: frame + stack parameters
        total_cleanup_bytes = frame_size + stack_param_bytes

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

        if is_far:
            self._emit_far_stack_cleanup(frame_size, stack_param_bytes, effective_return_count, returns_b)
        else:
            self._emit_near_stack_cleanup(frame_size, stack_param_bytes, effective_return_count, returns_b)

    def _emit_near_stack_cleanup(self, frame_size: int, stack_param_bytes: int, return_count: int, returns_b: bool = False):
        """
        Emit stack cleanup for near functions (2-byte return address).

        Uses PLX/PLY to grab return address, adjusts SP, pushes back.

        Args:
            frame_size: Number of bytes allocated for stack frame (0 if no frame)
            stack_param_bytes: Number of parameter bytes to clean
            return_count: Number of registers used for return (0-3)
            returns_b: True if function returns a value in B register
        """
        from r65.compiler.codegen.constants import M_FLAG

        total_cleanup = frame_size + stack_param_bytes

        # Special case: frame-only cleanup (no stack params).
        # No return address relocation needed - just deallocate the frame.
        # For small frames, use PLA-based deallocation which preserves B, X, Y,
        # and DBR (PLB would corrupt DBR, TSC/ADC/TCS would clobber B and
        # require saving A to a register which could clobber preserved regs).
        if stack_param_bytes == 0 and frame_size > 0:
            if frame_size <= 4:
                self._emit_pla_frame_dealloc(frame_size, return_count)
                return
            current_mode = self.parent.emitter.get_accu_mode()
            self._emit_sp_adjust_preserving_a(frame_size, return_count, current_mode)
            return

        # Determine which register to use for return address
        # A is most commonly used for return, so prefer X, then Y
        if return_count <= 1:
            # Only A (or nothing) returned - X is free
            addr_reg = 'X'
            pull_op = Opcode.PLX
            push_op = Opcode.PHX
        elif return_count == 2:
            # A and X returned - use Y
            addr_reg = 'Y'
            pull_op = Opcode.PLY
            push_op = Opcode.PHY
        else:
            # All three registers used for return - need scratch or special handling
            # For now, use inline adjustment without a temp register
            self._emit_near_stack_cleanup_no_free_reg(frame_size, stack_param_bytes)
            return

        # With frame allocation, return address is at S+frame_size+1, not S+1
        # We need to load it from the correct offset before PLX can work
        if frame_size > 0:
            # Use inline stack manipulation instead of PL/PH for frame case
            self._emit_near_stack_cleanup_with_frame(frame_size, stack_param_bytes, return_count)
            return

        # Emit cleanup sequence (no frame case):
        # PLX/PLY - pop return address (2 bytes)
        # (save A return value if needed)
        # REP #$20 - switch to m16 for 16-bit SP arithmetic
        # TSC - transfer SP to A  (clobbers A!)
        # CLC
        # ADC #N - add cleanup bytes
        # TCS - transfer back to SP
        # SEP #$20 - restore m8 mode
        # (restore A return value if needed)
        # PHX/PHY - push return address back

        self._emit_implied(pull_op, f"Pop return address into {addr_reg}")

        # Save A return value before TSC clobbers it.
        # addr_reg already holds the return address (X or Y).
        # Find a free register for A's return value.
        a_save_method = None  # 'X', 'Y', or 'stack'
        if return_count >= 1:
            if addr_reg == 'X' and return_count <= 1:
                a_save_method = 'Y'  # X has ret addr, Y is free
            elif addr_reg == 'Y' and return_count <= 1:
                a_save_method = 'X'  # Y has ret addr, X is free (A-only return)
            else:
                a_save_method = 'stack'

        if a_save_method == 'Y':
            self._emit_implied(Opcode.TAY, "Save return value A in Y")
        elif a_save_method == 'X':
            self._emit_implied(Opcode.TAX, "Save return value A in X")
        elif a_save_method == 'stack':
            self._emit_implied(Opcode.PHA, "Save return value A")

        # Check current mode - if already m16, skip mode switches
        current_mode = self.parent.emitter.get_accu_mode()
        need_mode_switch = (current_mode != 16)

        if need_mode_switch:
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for SP adjust")
            self.parent.emitter.emit_accu_mode(16)

        self._emit_implied(Opcode.TSC, "SP to A")
        self._emit_implied(Opcode.CLC)
        self._emit_immediate(Opcode.ADC_IMMEDIATE, total_cleanup, f"Adjust past {total_cleanup} bytes")
        self._emit_implied(Opcode.TCS, "A to SP")

        if need_mode_switch:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)

        # Restore A return value
        if a_save_method == 'Y':
            self._emit_implied(Opcode.TYA, "Restore return value A from Y")
        elif a_save_method == 'X':
            self._emit_implied(Opcode.TXA, "Restore return value A from X")
        elif a_save_method == 'stack':
            self._emit_implied(Opcode.PLA, "Restore return value A")

        self._emit_implied(push_op, "Push return address back")

    def _emit_near_stack_cleanup_with_frame(self, frame_size: int, stack_param_bytes: int, return_count: int):
        """
        Emit stack cleanup for near functions when a stack frame is allocated.

        With frame allocation, return address is at S+frame_size+1, not S+1.
        Uses inline stack-relative loads/stores instead of PL/PH instructions.

        Args:
            frame_size: Number of bytes allocated for stack frame
            stack_param_bytes: Number of parameter bytes to clean
            return_count: Number of registers used for return (0-3)
        """
        from r65.compiler.codegen.constants import M_FLAG

        total_cleanup = frame_size + stack_param_bytes

        # Return address is at S+frame_size+1 (2 bytes for near)
        ret_addr_offset = frame_size + 1

        # Determine how to preserve A during cleanup
        # Must not use a register that was just restored by #[preserves(...)],
        # since the restore (PLX/PLY) happened in the epilogue before this cleanup.
        preserved = set()
        if self.current_function and self.current_function.preserves_attr:
            preserved = set(self.current_function.preserves_attr.registers)

        save_method = None  # 'X', 'Y', or 'stack'
        if return_count >= 1:
            if return_count == 1:
                # A is the return value; pick X or Y for temp, avoiding preserved regs
                if 'X' not in preserved:
                    save_method = 'X'
                elif 'Y' not in preserved:
                    save_method = 'Y'
                else:
                    save_method = 'stack'
            elif return_count == 2:
                if 'Y' not in preserved:
                    save_method = 'Y'
                else:
                    save_method = 'stack'
            else:
                save_method = 'stack'

        # Save return value if needed
        if save_method == 'X':
            self._emit_implied(Opcode.TAX, "Save return value A in X")
        elif save_method == 'Y':
            self._emit_implied(Opcode.TAY, "Save return value A in Y")
        elif save_method == 'stack':
            self._emit_implied(Opcode.PHA, "Save return value A")
            ret_addr_offset += 1

        current_mode = self.parent.emitter.get_accu_mode()
        need_mode_switch = (current_mode != 16)

        if need_mode_switch:
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for cleanup")
            self.parent.emitter.emit_accu_mode(16)

        # Load return address from current position, store at new position
        store_offset = total_cleanup + 1 + (1 if save_method == 'stack' else 0)
        self._emit_stack_relative(Opcode.LDA_STACK, ret_addr_offset, "Load return address")
        self._emit_stack_relative(Opcode.STA_STACK, store_offset, "Store past cleanup area")

        # Adjust SP
        self._emit_implied(Opcode.TSC, "SP to A")
        self._emit_implied(Opcode.CLC)
        adj_amount = total_cleanup + (1 if save_method == 'stack' else 0)
        self._emit_immediate(Opcode.ADC_IMMEDIATE, adj_amount, f"Adjust past {total_cleanup} bytes")
        self._emit_implied(Opcode.TCS, "A to SP")

        if need_mode_switch:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)

        # Restore return value
        if save_method == 'X':
            self._emit_implied(Opcode.TXA, "Restore return value A from X")
        elif save_method == 'Y':
            self._emit_implied(Opcode.TYA, "Restore return value A from Y")
        elif save_method == 'stack':
            self._emit_implied(Opcode.PLA, "Restore return value A")

    def _emit_near_stack_cleanup_no_free_reg(self, frame_size: int, stack_param_bytes: int):
        """
        Emit stack cleanup when all A/X/Y have return values.

        Uses inline stack manipulation since no registers are free.
        Since A has a return value, must save it to stack first.

        Args:
            frame_size: Number of bytes allocated for stack frame (0 if no frame)
            stack_param_bytes: Number of parameter bytes to clean
        """
        from r65.compiler.codegen.constants import M_FLAG

        total_cleanup = frame_size + stack_param_bytes

        # Return address offset depends on whether we have a frame
        # Without frame: SP+1, With frame: SP+frame_size+1
        ret_addr_offset = frame_size + 1

        current_mode = self.parent.emitter.get_accu_mode()

        # Save return value A to stack first (A has a return value since all regs are used)
        self._emit_implied(Opcode.PHA, "Save return value A")
        # After PHA, return address is one byte deeper
        ret_addr_offset += 1

        need_mode_switch = (current_mode != 16)

        if need_mode_switch:
            self._emit_immediate(Opcode.REP_IMMEDIATE, 0x20, "16-bit A for cleanup")
            self.parent.emitter.emit_accu_mode(16)

        # Load return address from stack, store it higher up
        # Account for the extra byte we pushed
        store_offset = total_cleanup + 2  # +1 for original, +1 for saved A
        self._emit_stack_relative(Opcode.LDA_STACK, ret_addr_offset, "Load return address")
        self._emit_stack_relative(Opcode.STA_STACK, store_offset, "Store past cleanup area")

        # Adjust SP (include the saved A byte)
        self._emit_implied(Opcode.TSC, "SP to A")
        self._emit_implied(Opcode.CLC)
        self._emit_immediate(Opcode.ADC_IMMEDIATE, total_cleanup + 1, f"Adjust past {total_cleanup} bytes + saved A")
        self._emit_implied(Opcode.TCS, "A to SP")

        if need_mode_switch:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, 0x20, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)

        # Restore return value A
        self._emit_implied(Opcode.PLA, "Restore return value A")

    def _emit_sp_adjust_preserving_a(self, adjust_bytes: int, return_count: int, current_mode: int):
        """
        Emit SP adjustment that preserves return value in A.

        Handles saving/restoring A to X, Y, or stack based on return_count.
        Uses TSC/CLC/ADC/TCS sequence for adjustment.

        Args:
            adjust_bytes: Number of bytes to add to SP
            return_count: Number of registers used for return (0-3)
            current_mode: Current accumulator mode (8 or 16)
        """
        from r65.compiler.codegen.constants import M_FLAG

        # Determine save method based on return count
        if return_count == 0:
            save_reg = None
            restore_op = None
        elif return_count == 1:
            save_reg = Opcode.TAX
            restore_op = Opcode.TXA
        elif return_count == 2:
            save_reg = Opcode.TAY
            restore_op = Opcode.TYA
        else:
            save_reg = 'stack'
            restore_op = 'stack'

        # Calculate adjustment (account for push if using stack save)
        actual_adjust = adjust_bytes
        if save_reg == 'stack':
            push_size = 2 if current_mode == 16 else 1
            actual_adjust += push_size

        # Save A if needed
        if save_reg == 'stack':
            self._emit_implied(Opcode.PHA, "Save return value A")
        elif save_reg is not None:
            self._emit_implied(save_reg, f"Save return value A in {'X' if save_reg == Opcode.TAX else 'Y'}")

        # Switch to m16 if needed
        if current_mode != 16:
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for SP adjust")
            self.parent.emitter.emit_accu_mode(16)

        # TSC/CLC/ADC/TCS sequence
        self._emit_implied(Opcode.TSC, "SP to A")
        self._emit_implied(Opcode.CLC)
        self._emit_immediate(Opcode.ADC_IMMEDIATE, actual_adjust, f"Adjust past {adjust_bytes} bytes")
        self._emit_implied(Opcode.TCS, "A to SP")

        # Restore original mode before restoring A (preserves full register width)
        if current_mode != 16:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)

        # Restore A if needed
        if restore_op == 'stack':
            self._emit_implied(Opcode.PLA, "Restore return value A")
        elif restore_op is not None:
            self._emit_implied(restore_op, f"Restore return value A from {'X' if restore_op == Opcode.TXA else 'Y'}")

    def _emit_pla_frame_dealloc(self, frame_size: int, return_count: int):
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

        Far functions have bank byte + address on stack. More complex
        because we need to handle 3 bytes instead of 2.

        Args:
            frame_size: Number of bytes allocated for stack frame (0 if no frame)
            stack_param_bytes: Number of parameter bytes to clean
            return_count: Number of registers used for return (0-3)
            returns_b: True if function returns a value in B register
        """
        from r65.compiler.codegen.constants import M_FLAG

        total_cleanup = frame_size + stack_param_bytes

        # Special case: frame-only cleanup (no stack params)
        # When there are no stack params, we don't need to move the return address.
        # For small frames, use PLA-based deallocation which preserves B, X, Y,
        # and DBR (PLB would corrupt DBR, TSC/ADC/TCS would clobber B and
        # require saving A to a register which could clobber preserved regs).
        if stack_param_bytes == 0 and frame_size > 0:
            if frame_size <= 4:
                self._emit_pla_frame_dealloc(frame_size, return_count)
                return
            current_mode = self.parent.emitter.get_accu_mode()
            self._emit_sp_adjust_preserving_a(frame_size, return_count, current_mode)
            return

        # General case: need to move return address past stack params
        # For far functions, return address is 3 bytes: addr_lo, addr_hi, bank
        ret_addr_offset = frame_size + 1
        bank_offset = frame_size + 3

        # Determine how to preserve A during cleanup
        # Must not use a register that was just restored by #[preserves(...)].
        preserved = set()
        if self.current_function and self.current_function.preserves_attr:
            preserved = set(self.current_function.preserves_attr.registers)

        save_method = None  # 'X', 'Y', or 'stack'
        if return_count >= 1:
            if return_count == 1:
                if 'X' not in preserved:
                    save_method = 'X'
                elif 'Y' not in preserved:
                    save_method = 'Y'
                else:
                    save_method = 'stack'
            elif return_count == 2:
                if 'Y' not in preserved:
                    save_method = 'Y'
                else:
                    save_method = 'stack'
            else:
                save_method = 'stack'

        # Save return value if needed
        if save_method == 'X':
            self._emit_implied(Opcode.TAX, "Save return value A in X")
        elif save_method == 'Y':
            self._emit_implied(Opcode.TAY, "Save return value A in Y")
        elif save_method == 'stack':
            self._emit_implied(Opcode.PHA, "Save return value A")
            ret_addr_offset += 1
            bank_offset += 1

        current_mode = self.parent.emitter.get_accu_mode()
        store_offset = total_cleanup + 1 + (1 if save_method == 'stack' else 0)

        # CRITICAL: Read bank byte FIRST before the 16-bit store can overwrite it.
        # The 16-bit store at store_offset writes to bytes store_offset and store_offset+1.
        # If bank_offset == store_offset+1, the bank would be overwritten before we read it.
        # By reading bank first and storing it at store_offset+2, we avoid this race.
        need_mode_switch = (current_mode != 8)
        if need_mode_switch:
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "8-bit A for bank")
            self.parent.emitter.emit_accu_mode(8)

        # Load and store bank byte first (to avoid overwrite by 16-bit store)
        self._emit_stack_relative(Opcode.LDA_STACK, bank_offset, "Load bank byte")
        self._emit_stack_relative(Opcode.STA_STACK, store_offset + 2, "Store bank past cleanup area")

        # Now switch to 16-bit for address portion
        self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for far cleanup")
        self.parent.emitter.emit_accu_mode(16)

        # Load and store 16-bit address portion (safe now that bank is already moved)
        self._emit_stack_relative(Opcode.LDA_STACK, ret_addr_offset, "Load return address")
        self._emit_stack_relative(Opcode.STA_STACK, store_offset, "Store past cleanup area")

        # Switch to 16-bit for SP adjustment
        self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for SP adjust")
        self.parent.emitter.emit_accu_mode(16)

        # Adjust SP
        self._emit_implied(Opcode.TSC, "SP to A")
        self._emit_implied(Opcode.CLC)
        adj_amount = total_cleanup + (1 if save_method == 'stack' else 0)
        self._emit_immediate(Opcode.ADC_IMMEDIATE, adj_amount, f"Adjust past {total_cleanup} bytes")
        self._emit_implied(Opcode.TCS, "A to SP")

        # Restore 8-bit mode
        self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
        self.parent.emitter.emit_accu_mode(8)

        # Restore return value
        if save_method == 'X':
            self._emit_implied(Opcode.TXA, "Restore return value A from X")
        elif save_method == 'Y':
            self._emit_implied(Opcode.TYA, "Restore return value A from Y")
        elif save_method == 'stack':
            self._emit_implied(Opcode.PLA, "Restore return value A")

    def _emit_stack_relative(self, opcode: Opcode, offset: int, comment: str = None):
        """Emit a stack-relative instruction."""
        from r65.compiler.codegen.asm_nodes import StackOffset
        self.emitter.emit_instr(opcode, StackOffset(offset), comment)
