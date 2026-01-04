"""
Control flow instruction selector: Jump, Branch, Return.

Handles control flow instruction generation including conditional branches
with proper signed/unsigned comparison handling.
"""

from typing import TYPE_CHECKING
from r65.compiler.mir.nodes import Jump, JumpTable, CondBranch, Return
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.codegen.instruction_select_helpers import RegisterMappings
from r65.compiler.errors import InstructionSelectionError

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector


class ControlFlowInstructionSelector:
    """
    Handles control flow instruction selection.

    Manages generation of jump, branch, and return instructions
    with proper signed/unsigned comparison handling.
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize control flow selector.

        Args:
            parent: Parent instruction selector (for helper method access)
        """
        self.parent = parent

    @property
    def emitter(self):
        return self.parent.emitter

    @property
    def current_function(self):
        return self.parent.current_function

    @property
    def last_comparison_type(self):
        return self.parent.last_comparison_type

    # ========================================================================
    # Jump Instructions
    # ========================================================================

    def select_jump(self, instr: Jump):
        """
        Generate code for Jump instruction.

        Args:
            instr: Jump instruction
        """
        self.emitter.emit_instruction("JMP", f"__L{instr.target}")

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
            self.emitter.emit_instruction("LDA", self.parent._format_operand(scrutinee_loc))

        # Subtract base_value to compute index
        if instr.base_value != 0:
            self.emitter.emit_instruction("SEC")
            self.emitter.emit_instruction("SBC", f"#{instr.base_value}", "Compute index = scrutinee - base")
            self.emitter.emit_instruction("BMI", f"__L{instr.default_target}", "Out of bounds (< min)")

        # Check if index >= table_size - out of bounds
        self.emitter.emit_instruction("CMP", f"#{table_size}", "Check upper bound")
        self.emitter.emit_instruction("BCS", f"__L{instr.default_target}", "Out of bounds (>= size)")

        # Generate comparison chain with optimized jump targets
        for i, target_block in enumerate(instr.targets):
            if i == table_size - 1:
                self.emitter.emit_instruction("JMP", f"__L{target_block}")
            else:
                self.emitter.emit_instruction("CMP", f"#{i}")
                self.emitter.emit_instruction("BEQ", f"__L{target_block}")

        # Fallback
        self.emitter.emit_instruction("JMP", f"__L{instr.default_target}")

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
        true_target = f"__L{instr.true_target}"
        false_target = f"__L{instr.false_target}"

        # Handle BIT-based comparisons
        if comparison == 'bit7_set':
            self.emitter.emit_instruction("BMI", true_target, "Branch if bit 7 set")
            self.emitter.emit_instruction("JMP", false_target)
        elif comparison == 'bit7_clear':
            self.emitter.emit_instruction("BPL", true_target, "Branch if bit 7 clear")
            self.emitter.emit_instruction("JMP", false_target)
        elif comparison == 'bit6_set':
            self.emitter.emit_instruction("BVS", true_target, "Branch if bit 6 set")
            self.emitter.emit_instruction("JMP", false_target)
        elif comparison == 'bit6_clear':
            self.emitter.emit_instruction("BVC", true_target, "Branch if bit 6 clear")
            self.emitter.emit_instruction("JMP", false_target)
        # Handle comparison operators
        elif comparison == '==':
            self.emitter.emit_instruction("BEQ", true_target, "Branch if equal")
            self.emitter.emit_instruction("JMP", false_target)
        elif comparison == '!=':
            self.emitter.emit_instruction("BNE", true_target, "Branch if not equal")
            self.emitter.emit_instruction("JMP", false_target)
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
            self.emitter.emit_instruction("BVC", label, "Skip if no overflow")
            self.emitter.emit_instruction("EOR", "#$80", "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self.emitter.emit_instruction("BMI", true_target, "Branch if less than (signed)")
            self.emitter.emit_instruction("JMP", false_target)
        else:
            # Unsigned less than: C flag clear
            self.emitter.emit_instruction("BCC", true_target, "Branch if less than (unsigned)")
            self.emitter.emit_instruction("JMP", false_target)

    def _emit_greater_equal_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for >= comparison."""
        if is_signed:
            # Signed >= : N XOR V = 0
            label = self.parent._get_unique_label()
            self.emitter.emit_instruction("BVC", label, "Skip if no overflow")
            self.emitter.emit_instruction("EOR", "#$80", "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self.emitter.emit_instruction("BPL", true_target, "Branch if >= (signed)")
            self.emitter.emit_instruction("JMP", false_target)
        else:
            # Unsigned >=: C flag set
            self.emitter.emit_instruction("BCS", true_target, "Branch if >= (unsigned)")
            self.emitter.emit_instruction("JMP", false_target)

    def _emit_greater_than_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for > comparison."""
        if is_signed:
            # Signed >: (N XOR V = 0) AND Z = 0
            self.emitter.emit_instruction("BEQ", false_target, "Skip if equal")
            label = self.parent._get_unique_label()
            self.emitter.emit_instruction("BVC", label, "Skip if no overflow")
            self.emitter.emit_instruction("EOR", "#$80", "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self.emitter.emit_instruction("BPL", true_target, "Branch if > (signed)")
            self.emitter.emit_instruction("JMP", false_target)
        else:
            # Unsigned >: (C set) AND (Z clear)
            self.emitter.emit_instruction("BEQ", false_target, "Skip if equal")
            self.emitter.emit_instruction("BCS", true_target, "Branch if > (unsigned)")
            self.emitter.emit_instruction("JMP", false_target)

    def _emit_less_equal_branch(self, true_target: str, false_target: str, is_signed: bool):
        """Emit branch for <= comparison."""
        if is_signed:
            # Signed <=: (N XOR V = 1) OR Z = 1
            self.emitter.emit_instruction("BEQ", true_target, "Branch if equal")
            label = self.parent._get_unique_label()
            self.emitter.emit_instruction("BVC", label, "Skip if no overflow")
            self.emitter.emit_instruction("EOR", "#$80", "Flip sign bit if overflow")
            self.emitter.emit_label(label)
            self.emitter.emit_instruction("BMI", true_target, "Branch if <= (signed)")
            self.emitter.emit_instruction("JMP", false_target)
        else:
            # Unsigned <=: (C clear) OR (Z set)
            self.emitter.emit_instruction("BEQ", true_target, "Branch if equal")
            self.emitter.emit_instruction("BCC", true_target, "Branch if less than")
            self.emitter.emit_instruction("JMP", false_target)

    def _emit_value_based_branch(self, instr: CondBranch):
        """Emit branch based on condition value (zero/non-zero)."""
        cond_loc = self.parent._get_operand_location(instr.condition)
        self.emitter.emit_instruction("LDA", self.parent._format_operand(cond_loc))

        true_target = f"__L{instr.true_target}"
        false_target = f"__L{instr.false_target}"

        if instr.comparison == '!=':
            self.emitter.emit_instruction("BEQ", false_target, "Branch if zero")
            self.emitter.emit_instruction("JMP", true_target)
        elif instr.comparison == '==':
            self.emitter.emit_instruction("BNE", false_target, "Branch if non-zero")
            self.emitter.emit_instruction("JMP", true_target)
        else:
            # For other comparisons on boolean values, treat as != 0
            self.emitter.emit_instruction("BEQ", false_target)
            self.emitter.emit_instruction("JMP", true_target)

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

    def _emit_return_values(self, instr: Return):
        """Load return values into appropriate registers."""
        if not instr.values:
            return

        return_registers = ['A', 'X', 'Y']
        for i, value in enumerate(instr.values):
            if i >= len(return_registers):
                raise InstructionSelectionError(
                    f"Too many return values (max {len(return_registers)})")

            target_reg = return_registers[i]
            value_loc = self.parent._get_operand_location(value)

            if value_loc.kind == LocationKind.HARDWARE and value_loc.hw_register == target_reg:
                pass  # Already in correct register
            elif value_loc.kind == LocationKind.HARDWARE:
                self.parent._emit_register_transfer(value_loc.hw_register, target_reg)
            else:
                load_instr = RegisterMappings.LOAD.get(target_reg)
                if load_instr:
                    self.emitter.emit_instruction(load_instr, self.parent._format_operand(value_loc))

    def _emit_preserved_register_restores(self):
        """Restore preserved registers in reverse order."""
        if not (self.current_function and self.current_function.preserves_attr):
            return

        preserved_regs = self.current_function.preserves_attr.registers
        pop_order = ['DBR', 'D', 'Y', 'X', 'A', 'STATUS']

        for reg in pop_order:
            if reg in preserved_regs:
                pull_instr = RegisterMappings.PULL.get(reg)
                if pull_instr:
                    self.emitter.emit_instruction(pull_instr, comment=f"Restore {reg}")

    def _emit_dbr_restore(self):
        """Restore DBR for data_bank=inline functions."""
        if not (self.current_function and self.current_function.is_far and self.current_function.bank_attr):
            return

        from r65.compiler.hir.attributes import DataBankMode
        if self.current_function.bank_attr.data_bank == DataBankMode.INLINE:
            self.emitter.emit_instruction("PLB", comment="Restore data bank")

    def _emit_mode_restore(self):
        """Restore processor mode for transition=inline functions."""
        if not (self.current_function and self.current_function.mode_attr):
            return

        from r65.compiler.hir.attributes import ModeTransition
        if self.current_function.mode_attr.transition == ModeTransition.INLINE:
            self.emitter.emit_instruction("PLP", comment="Restore processor status")

    def _emit_return_instruction(self):
        """Emit appropriate return instruction (RTL or RTS)."""
        if self.current_function and self.current_function.is_far:
            self.emitter.emit_instruction("RTL")
        else:
            self.emitter.emit_instruction("RTS")
