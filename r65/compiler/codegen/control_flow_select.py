"""
Control flow instruction selector: Jump, Branch, Return.

Handles control flow instruction generation including conditional branches
with proper signed/unsigned comparison handling.
"""

from r65.compiler.mir.nodes import Jump, JumpTable, CondBranch, Return
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode
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
        self._emit_jump(Opcode.JMP_ABSOLUTE, self._block_label(instr.target))

    def select_jump_table(self, instr: JumpTable):
        """
        Generate code for JumpTable instruction (optimized pattern matching).

        Generates efficient jump table for dense integer pattern matching:
        1. Compute index = scrutinee - base_value
        2. Bounds check (0 <= index < table_size)
        3. Indirect jump through jump table

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

        # Subtract base_value to compute index
        if instr.base_value != 0:
            self._emit_implied(Opcode.SEC)
            self._emit_immediate(Opcode.SBC_IMMEDIATE, instr.base_value, "Compute index = scrutinee - base")
            self._emit_branch(Opcode.BMI, self._block_label(instr.default_target), "Out of bounds (< min)")

        # Check if index >= table_size - out of bounds
        self._emit_immediate(Opcode.CMP_IMMEDIATE, table_size, "Check upper bound")
        self._emit_branch(Opcode.BCS, self._block_label(instr.default_target), "Out of bounds (>= size)")

        # Generate comparison chain with optimized jump targets
        for i, target_block in enumerate(instr.targets):
            if i == table_size - 1:
                self._emit_jump(Opcode.JMP_ABSOLUTE, self._block_label(target_block))
            else:
                self._emit_immediate(Opcode.CMP_IMMEDIATE, i)
                self._emit_branch(Opcode.BEQ, self._block_label(target_block))

        # Fallback
        self._emit_jump(Opcode.JMP_ABSOLUTE, self._block_label(instr.default_target))

    # ========================================================================
    # Conditional Branch
    # ========================================================================

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

        if instr.condition is None:
            self._emit_flag_based_branch(instr, is_signed)
        else:
            self._emit_value_based_branch(instr)

    def _is_signed_comparison(self) -> bool:
        """Check if the last comparison was signed."""
        if self.last_comparison_type is not None:
            from r65.compiler.hir.types import BasicTypeInfo
            if isinstance(self.last_comparison_type, BasicTypeInfo):
                return self.last_comparison_type.name.startswith('i')
        return False

    def _emit_flag_based_branch(self, instr: CondBranch, is_signed: bool):
        """Emit branch based on CPU flags from preceding Compare."""
        comparison = instr.comparison
        true_target = self._block_label(instr.true_target)
        false_target = self._block_label(instr.false_target)

        # Handle BIT-based comparisons
        if comparison == 'bit7_set':
            self._emit_branch(Opcode.BMI, true_target, "Branch if bit 7 set")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'bit7_clear':
            self._emit_branch(Opcode.BPL, true_target, "Branch if bit 7 clear")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'bit6_set':
            self._emit_branch(Opcode.BVS, true_target, "Branch if bit 6 set")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'bit6_clear':
            self._emit_branch(Opcode.BVC, true_target, "Branch if bit 6 clear")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        # Handle STATUS flag comparisons (branchable flags)
        elif comparison == 'status_carry_set':
            self._emit_branch(Opcode.BCS, true_target, "Branch if Carry set")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_carry_clear':
            self._emit_branch(Opcode.BCC, true_target, "Branch if Carry clear")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_zero_set':
            self._emit_branch(Opcode.BEQ, true_target, "Branch if Zero set")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_zero_clear':
            self._emit_branch(Opcode.BNE, true_target, "Branch if Zero clear")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_overflow_set':
            self._emit_branch(Opcode.BVS, true_target, "Branch if Overflow set")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_overflow_clear':
            self._emit_branch(Opcode.BVC, true_target, "Branch if Overflow clear")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_negative_set':
            self._emit_branch(Opcode.BMI, true_target, "Branch if Negative set")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_negative_clear':
            self._emit_branch(Opcode.BPL, true_target, "Branch if Negative clear")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        # Handle STATUS flag comparisons (non-branchable flags after PHP; PLA; AND #mask)
        elif comparison == 'status_nonbranch_set':
            self._emit_branch(Opcode.BNE, true_target, "Branch if flag set (AND result != 0)")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == 'status_nonbranch_clear':
            self._emit_branch(Opcode.BEQ, true_target, "Branch if flag clear (AND result == 0)")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        # Handle comparison operators
        elif comparison == '==':
            self._emit_branch(Opcode.BEQ, true_target, "Branch if equal")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        elif comparison == '!=':
            self._emit_branch(Opcode.BNE, true_target, "Branch if not equal")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
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
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        else:
            # Unsigned less than: C flag clear
            self._emit_branch(Opcode.BCC, true_target, "Branch if less than (unsigned)")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)

    def _emit_greater_equal_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for >= comparison."""
        if is_signed:
            # Signed >= : N XOR V = 0
            label = self.parent._get_unique_label()
            self._emit_branch(Opcode.BVC, label, "Skip if no overflow")
            self._emit_immediate(Opcode.EOR_IMMEDIATE, 0x80, "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self._emit_branch(Opcode.BPL, true_target, "Branch if >= (signed)")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        else:
            # Unsigned >=: C flag set
            self._emit_branch(Opcode.BCS, true_target, "Branch if >= (unsigned)")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)

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
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        else:
            # Unsigned >: (C set) AND (Z clear)
            self._emit_branch(Opcode.BEQ, false_target, "Skip if equal")
            self._emit_branch(Opcode.BCS, true_target, "Branch if > (unsigned)")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)

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
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)
        else:
            # Unsigned <=: (C clear) OR (Z set)
            self._emit_branch(Opcode.BEQ, true_target, "Branch if equal")
            self._emit_branch(Opcode.BCC, true_target, "Branch if less than")
            self._emit_jump(Opcode.JMP_ABSOLUTE, false_target)

    def _emit_value_based_branch(self, instr: CondBranch):
        """Emit branch based on condition value (zero/non-zero)."""
        cond_loc = self.parent._get_operand_location(instr.condition)
        self.parent._emit_load('LDA', cond_loc)

        true_target = self._block_label(instr.true_target)
        false_target = self._block_label(instr.false_target)

        if instr.comparison == '!=':
            self._emit_branch(Opcode.BEQ, false_target, "Branch if zero")
            self._emit_jump(Opcode.JMP_ABSOLUTE, true_target)
        elif instr.comparison == '==':
            self._emit_branch(Opcode.BNE, false_target, "Branch if non-zero")
            self._emit_jump(Opcode.JMP_ABSOLUTE, true_target)
        else:
            # For other comparisons on boolean values, treat as != 0
            self._emit_branch(Opcode.BEQ, false_target)
            self._emit_jump(Opcode.JMP_ABSOLUTE, true_target)

    # ========================================================================
    # Return Instruction
    # ========================================================================

    def select_return(self, instr: Return):
        """
        Generate code for Return instruction.

        Handles loading return values into appropriate registers before returning.

        Args:
            instr: Return instruction
        """
        self._emit_return_values(instr)

        # Use consolidated emit_epilogue from FunctionCodeGenerator
        if self.parent.func_gen:
            self.parent.func_gen.emit_epilogue(self.current_function, self.parent.reg_alloc)
        else:
            # Fallback to inline methods if func_gen not available
            self._emit_preserved_register_restores()
            self._emit_dbr_restore()
            self._emit_mode_restore()

        self._emit_return_instruction()

    # Mappings from register name to pull opcodes
    _PULL_OPCODES = {
        'A': Opcode.PLA, 'X': Opcode.PLX, 'Y': Opcode.PLY,
        'STATUS': Opcode.PLP, 'P': Opcode.PLP,
        'D': Opcode.PLD, 'DBR': Opcode.PLB, 'B': Opcode.PLB,
    }

    def _emit_return_values(self, instr: Return):
        """Load return values into appropriate registers.

        Values are loaded in reverse order (Y, X, A) so that transfers
        through A (needed for stack-relative X/Y loads) don't clobber
        the final A value.
        """
        if not instr.values:
            return

        return_registers = ['A', 'X', 'Y']
        if len(instr.values) > len(return_registers):
            raise InstructionSelectionError(
                f"Too many return values (max {len(return_registers)})")

        # Process in reverse order to avoid clobbering A
        for i in range(len(instr.values) - 1, -1, -1):
            value = instr.values[i]
            target_reg = return_registers[i]
            value_loc = self.parent._get_operand_location(value)

            if value_loc.kind == LocationKind.HARDWARE and value_loc.hw_register == target_reg:
                pass  # Already in correct register
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
                load_mnem = {'A': 'LDA', 'X': 'LDX', 'Y': 'LDY'}[target_reg]
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
        """Restore processor mode for transition=inline functions."""
        if not (self.current_function and self.current_function.mode_attr):
            return

        from r65.compiler.hir.attributes import ModeTransition
        if self.current_function.mode_attr.transition == ModeTransition.INLINE:
            self._emit_implied(Opcode.PLP, "Restore processor status")

    def _emit_return_instruction(self):
        """Emit appropriate return instruction (RTL, RTS, or WAI).

        For functions returning ! (never type) or entry functions, we emit WAI
        instead of a return instruction since there's no valid return address.
        """
        from r65.compiler.hir.types import NeverTypeInfo

        # Never type or entry functions have no valid return address
        if self.current_function and (
            isinstance(self.current_function.return_type, NeverTypeInfo)
            or self.current_function.is_entry
        ):
            self._emit_implied(Opcode.WAI, "No return - wait for interrupt")
            return

        if self.current_function and self.current_function.is_far:
            self._emit_implied(Opcode.RTL)
        else:
            self._emit_implied(Opcode.RTS)
