"""
Call instruction selector: Function calls and built-ins.

Handles function call generation including argument setup, call emission,
return value collection, and built-in function expansion.
"""

from typing import TYPE_CHECKING
from r65.compiler.mir.nodes import Call, VirtualRegister, ArgumentMechanism, Immediate
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector


class CallInstructionSelector:
    """
    Handles call instruction selection.

    Manages generation of function calls, built-in expansions,
    and indirect call trampolines.
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize call selector.

        Args:
            parent: Parent instruction selector (for helper method access)
        """
        self.parent = parent

    @property
    def emitter(self):
        return self.parent.emitter

    # ========================================================================
    # Main Call Selection
    # ========================================================================

    def select_call(self, instr: Call):
        """
        Generate code for Call instruction.

        Handles argument setup, call, and return value collection.
        Also handles built-in function calls.

        Args:
            instr: Call instruction
        """
        # Handle built-in functions
        if instr.builtin_name:
            self._emit_builtin_call(instr)
            return

        # Step 1: Set up arguments
        stack_arg_count = self._emit_argument_setup(instr)

        # Step 2: Handle caller-managed DBR (data_bank=caller)
        needs_dbr_restore = self._emit_caller_dbr_setup(instr)

        # Step 3: Make the call
        self._emit_call_instruction(instr)

        # Invalidate XBA state after call (function may have modified A/B)
        self.parent._invalidate_xba_state()

        # Step 4: Restore DBR if caller-managed
        if needs_dbr_restore:
            self.emitter.emit_instruction("PLB", comment="Restore data bank (caller)")

        # Step 5: Clean up stack arguments (callee cleanup for R65)
        self._emit_stack_cleanup(stack_arg_count)

        # Step 6: Collect return values
        self._emit_return_value_collection(instr)

    # ========================================================================
    # Argument Setup
    # ========================================================================

    def _emit_argument_setup(self, instr: Call) -> int:
        """
        Set up call arguments in correct order.

        Process in specific order to avoid clobbering:
        1. Stack arguments (pushed in order)
        2. Variable-bound arguments
        3. B register arguments (these clobber A via XBA)
        4. X and Y register arguments
        5. A register arguments (set up last to avoid being clobbered)

        Args:
            instr: Call instruction

        Returns:
            Number of stack arguments pushed
        """
        stack_arg_count = 0

        sorted_args = sorted(instr.args, key=self._arg_sort_key)

        for arg in sorted_args:
            arg_loc = self.parent._get_operand_location(arg.value)

            if arg.mechanism == ArgumentMechanism.STACK:
                self._emit_stack_argument(arg, arg_loc)
                stack_arg_count += 1

            elif arg.mechanism == ArgumentMechanism.REGISTER:
                self._emit_register_argument(arg, arg_loc)

            elif arg.mechanism == ArgumentMechanism.VARIABLE:
                self._emit_variable_argument(arg, arg_loc)

        return stack_arg_count

    def _arg_sort_key(self, arg):
        """Sort key for argument processing order."""
        if arg.mechanism == ArgumentMechanism.STACK:
            return 0  # Stack first
        elif arg.mechanism == ArgumentMechanism.VARIABLE:
            return 1  # Variable-bound second
        elif arg.mechanism == ArgumentMechanism.REGISTER:
            target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)
            if target_reg == 'B':
                return 2  # B third (clobbers A)
            elif target_reg in ['X', 'Y']:
                return 3  # X, Y fourth
            elif target_reg == 'A':
                return 4  # A last (to avoid being clobbered)
        return 5

    def _emit_stack_argument(self, arg, arg_loc):
        """Emit stack argument (push onto stack)."""
        # Load value into A and push
        if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
            pass  # Already in A
        elif arg_loc.kind == LocationKind.HARDWARE:
            if arg_loc.hw_register == 'X':
                self.emitter.emit_instruction("TXA")
            elif arg_loc.hw_register == 'Y':
                self.emitter.emit_instruction("TYA")
        elif isinstance(arg.value, Immediate):
            self.emitter.emit_instruction("LDA", f"#${arg.value.value:02X}")
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(arg_loc))

        self.emitter.emit_instruction("PHA", comment="Push stack arg")

    def _emit_register_argument(self, arg, arg_loc):
        """Emit register argument (move to specified register)."""
        target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)

        if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == target_reg:
            return  # Already in correct register

        if target_reg == 'B':
            self._emit_b_register_argument(arg, arg_loc)
        elif isinstance(arg.value, Immediate):
            self._emit_immediate_to_register(arg.value.value, target_reg)
        else:
            self._emit_memory_to_register(arg_loc, target_reg)

    def _emit_b_register_argument(self, arg, arg_loc):
        """Emit B register argument (special handling)."""
        if isinstance(arg.value, Immediate):
            self.emitter.emit_instruction("LDA", f"#${arg.value.value:02X}")
            self.parent._store_to_b_from_a()
        elif arg_loc.kind == LocationKind.HARDWARE:
            if arg_loc.hw_register != 'A':
                self.parent._emit_register_transfer(arg_loc.hw_register, 'A')
            self.parent._store_to_b_from_a()
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(arg_loc))
            self.parent._store_to_b_from_a()

    def _emit_immediate_to_register(self, value: int, target_reg: str):
        """Load immediate value into target register."""
        if target_reg == 'A':
            self.emitter.emit_instruction("LDA", f"#${value:02X}")
        elif target_reg == 'X':
            self.emitter.emit_instruction("LDX", f"#${value:02X}")
        elif target_reg == 'Y':
            self.emitter.emit_instruction("LDY", f"#${value:02X}")

    def _emit_memory_to_register(self, arg_loc, target_reg: str):
        """Load from memory into target register."""
        if target_reg == 'A':
            self.emitter.emit_instruction("LDA", self.parent._format_operand(arg_loc))
        elif target_reg == 'X':
            self.emitter.emit_instruction("LDX", self.parent._format_operand(arg_loc))
        elif target_reg == 'Y':
            self.emitter.emit_instruction("LDY", self.parent._format_operand(arg_loc))

    def _emit_variable_argument(self, arg, arg_loc):
        """Emit variable-bound argument (store to memory location)."""
        # Load into A
        if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
            pass  # Already in A
        elif arg_loc.kind == LocationKind.HARDWARE:
            if arg_loc.hw_register == 'X':
                self.emitter.emit_instruction("TXA")
            elif arg_loc.hw_register == 'Y':
                self.emitter.emit_instruction("TYA")
        elif isinstance(arg.value, Immediate):
            self.emitter.emit_instruction("LDA", f"#${arg.value.value:02X}")
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(arg_loc))

        # Store to variable location
        var_loc = self.parent._get_operand_location(arg.location)
        self.emitter.emit_instruction("STA", self.parent._format_operand(var_loc))

    # ========================================================================
    # DBR Management
    # ========================================================================

    def _emit_caller_dbr_setup(self, instr: Call) -> bool:
        """
        Handle caller-managed DBR setup if needed.

        Args:
            instr: Call instruction

        Returns:
            True if DBR restore needed after call
        """
        if not (instr.is_far and instr.bank_attr):
            return False

        from r65.compiler.hir.attributes import DataBankMode

        if instr.bank_attr.data_bank != DataBankMode.CALLER:
            return False

        # Caller manages DBR: save, set, call, restore
        self.emitter.emit_instruction("PHB", comment="Save current data bank (caller)")
        self.emitter.emit_instruction("LDA", f"#${instr.bank_attr.bank_number:02X}",
                                      "Load callee's bank number")
        self.emitter.emit_instruction("PHA", comment="Push bank number")
        self.emitter.emit_instruction("PLB", comment="Set data bank for callee")
        return True

    # ========================================================================
    # Call Emission
    # ========================================================================

    def _emit_call_instruction(self, instr: Call):
        """
        Emit the actual call instruction.

        Handles both direct and indirect calls.

        Args:
            instr: Call instruction
        """
        if isinstance(instr.function, VirtualRegister):
            # Indirect call through function pointer
            self._emit_indirect_call_trampoline(instr.function, instr.is_far)
        elif isinstance(instr.function, str):
            # Direct call
            if instr.is_far:
                self.emitter.emit_instruction("JSL", instr.function)
            else:
                self.emitter.emit_instruction("JSR", instr.function)
        else:
            raise InstructionSelectionError(f"Unknown function type in Call: {type(instr.function)}")

    def _emit_indirect_call_trampoline(self, func_ptr_vreg: VirtualRegister, is_far: bool):
        """
        Generate trampoline for indirect function call through function pointer.

        Near trampoline (16-bit address):
            LDA func_ptr+1  ; High byte
            PHA
            LDA func_ptr    ; Low byte
            PHA
            RTS             ; Jumps to address on stack

        Far trampoline (24-bit address):
            LDA func_ptr+2  ; Bank byte
            PHA
            LDA func_ptr+1  ; High byte
            PHA
            LDA func_ptr    ; Low byte
            PHA
            RTL             ; Long return

        Args:
            func_ptr_vreg: VirtualRegister holding the function pointer
            is_far: True for far call (24-bit), False for near call (16-bit)
        """
        ptr_loc = self.parent._get_operand_location(func_ptr_vreg)

        if is_far:
            # Far call trampoline (24-bit address)
            bank_loc = self.parent._offset_location(ptr_loc, 2)
            self.emitter.emit_instruction("LDA", self.parent._format_operand(bank_loc), "Load bank byte")
            self.emitter.emit_instruction("PHA", comment="Push bank")

            high_loc = self.parent._offset_location(ptr_loc, 1)
            self.emitter.emit_instruction("LDA", self.parent._format_operand(high_loc), "Load high byte")
            self.emitter.emit_instruction("PHA", comment="Push high")

            self.emitter.emit_instruction("LDA", self.parent._format_operand(ptr_loc), "Load low byte")
            self.emitter.emit_instruction("PHA", comment="Push low")

            self.emitter.emit_instruction("RTL", comment="Indirect far call via trampoline")
        else:
            # Near call trampoline (16-bit address)
            high_loc = self.parent._offset_location(ptr_loc, 1)
            self.emitter.emit_instruction("LDA", self.parent._format_operand(high_loc), "Load high byte")
            self.emitter.emit_instruction("PHA", comment="Push high")

            self.emitter.emit_instruction("LDA", self.parent._format_operand(ptr_loc), "Load low byte")
            self.emitter.emit_instruction("PHA", comment="Push low")

            self.emitter.emit_instruction("RTS", comment="Indirect near call via trampoline")

    # ========================================================================
    # Stack Cleanup and Return Values
    # ========================================================================

    def _emit_stack_cleanup(self, stack_arg_count: int):
        """Clean up stack arguments after call."""
        for _ in range(stack_arg_count):
            self.emitter.emit_instruction("PLA", comment="Clean up stack arg")

    def _emit_return_value_collection(self, instr: Call):
        """
        Collect return values from registers.

        Return values come back in A, X, Y (in order).
        """
        if not instr.returns:
            return

        return_registers = ['A', 'X', 'Y']

        for i, return_vreg in enumerate(instr.returns):
            if i >= len(return_registers):
                raise InstructionSelectionError(f"Too many return values (max {len(return_registers)})")

            source_reg = return_registers[i]
            dest_loc = self.parent._get_operand_location(return_vreg)

            if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == source_reg:
                pass  # Already in correct location
            elif dest_loc.kind == LocationKind.HARDWARE:
                self._emit_return_register_transfer(source_reg, dest_loc.hw_register)
            else:
                self._emit_return_store(source_reg, dest_loc)

    def _emit_return_register_transfer(self, source_reg: str, dest_reg: str):
        """Transfer return value between hardware registers."""
        if source_reg == 'A' and dest_reg == 'X':
            self.emitter.emit_instruction("TAX")
        elif source_reg == 'A' and dest_reg == 'Y':
            self.emitter.emit_instruction("TAY")
        elif source_reg == 'X' and dest_reg == 'A':
            self.emitter.emit_instruction("TXA")
        elif source_reg == 'Y' and dest_reg == 'A':
            self.emitter.emit_instruction("TYA")

    def _emit_return_store(self, source_reg: str, dest_loc):
        """Store return value from register to memory."""
        if source_reg == 'A':
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))
        elif source_reg == 'X':
            self.emitter.emit_instruction("STX", self.parent._format_operand(dest_loc))
        elif source_reg == 'Y':
            self.emitter.emit_instruction("STY", self.parent._format_operand(dest_loc))

    # ========================================================================
    # Built-in Function Calls
    # ========================================================================

    def _emit_builtin_call(self, instr: Call):
        """
        Emit code for built-in function call.

        Built-in categories:
        - Processor control: wai(), stp(), xba(), NOP([count])
        - Mode control: SEP(flags), REP(flags)
        - Block moves: mvn(src_bank, dst_bank), mvp(src_bank, dst_bank)
        - Arithmetic: mul(a, b), div(a, b), mod(a, b) - call runtime library
        - Shifts: shl(a, n), shr(a, n) - call runtime library

        Args:
            instr: Call instruction with builtin_name set
        """
        from r65.compiler.builtins import BuiltinRegistry, BuiltinKind

        builtin = BuiltinRegistry.get_builtin(instr.builtin_name)
        if not builtin:
            raise InstructionSelectionError(f"Unknown built-in function: {instr.builtin_name}")

        if builtin.kind == BuiltinKind.PROCESSOR_CONTROL:
            self._emit_processor_control_builtin(instr, builtin)
        elif builtin.kind == BuiltinKind.MODE_CONTROL:
            self._emit_mode_control_builtin(instr, builtin)
        elif builtin.kind == BuiltinKind.BLOCK_MOVE:
            self._emit_block_move_builtin(instr, builtin)
        elif builtin.kind in (BuiltinKind.ARITHMETIC, BuiltinKind.SHIFT):
            self._emit_runtime_builtin(instr, builtin)

    def _emit_processor_control_builtin(self, instr: Call, builtin):
        """Emit processor control built-in (wai, stp, xba, NOP)."""
        if instr.builtin_name == 'NOP':
            count = 1  # Default
            if len(instr.args) == 1:
                arg = instr.args[0]
                if isinstance(arg.value, Immediate):
                    count = arg.value.value
                else:
                    raise InstructionSelectionError("NOP() count must be a constant immediate value")

            for _ in range(count):
                self.emitter.emit_instruction(builtin.instruction)
        else:
            self.emitter.emit_instruction(builtin.instruction)

    def _emit_mode_control_builtin(self, instr: Call, builtin):
        """Emit mode control built-in (SEP, REP)."""
        if len(instr.args) != 1:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 1 argument, got {len(instr.args)}")

        arg = instr.args[0]
        arg_loc = self.parent._get_operand_location(arg.value)

        if isinstance(arg.value, Immediate):
            self.emitter.emit_instruction(builtin.instruction, f"#${arg.value.value:02X}")
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(arg_loc))
            self.emitter.emit_instruction(builtin.instruction, "A")

    def _emit_block_move_builtin(self, instr: Call, builtin):
        """Emit block move built-in (mvn, mvp)."""
        if len(instr.args) != 2:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 2 arguments, got {len(instr.args)}")

        src_bank_arg = instr.args[0]
        dst_bank_arg = instr.args[1]

        if not isinstance(src_bank_arg.value, Immediate) or not isinstance(dst_bank_arg.value, Immediate):
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects immediate bank numbers")

        src_bank = src_bank_arg.value.value
        dst_bank = dst_bank_arg.value.value

        self.emitter.emit_instruction(builtin.instruction, f"${dst_bank:02X}", f"${src_bank:02X}")

    def _emit_runtime_builtin(self, instr: Call, builtin):
        """Emit runtime library built-in (mul, div, mod, shl, shr)."""
        if len(instr.args) != 2:
            raise InstructionSelectionError(
                f"{instr.builtin_name}() expects 2 arguments, got {len(instr.args)}")

        # Load first argument into A
        arg0 = instr.args[0]
        arg0_loc = self.parent._get_operand_location(arg0.value)
        if isinstance(arg0.value, Immediate):
            self.emitter.emit_instruction("LDA", f"#${arg0.value.value:02X}")
        elif arg0_loc.kind == LocationKind.HARDWARE and arg0_loc.hw_register == 'A':
            pass  # Already in A
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(arg0_loc))

        # Load second argument into X
        arg1 = instr.args[1]
        arg1_loc = self.parent._get_operand_location(arg1.value)
        if isinstance(arg1.value, Immediate):
            self.emitter.emit_instruction("LDX", f"#${arg1.value.value:02X}")
        elif arg1_loc.kind == LocationKind.HARDWARE and arg1_loc.hw_register == 'X':
            pass  # Already in X
        else:
            self.emitter.emit_instruction("LDX", self.parent._format_operand(arg1_loc))

        # Call runtime library function
        runtime_func_name = f"__builtin_{instr.builtin_name}"
        self.emitter.emit_instruction("JSR", runtime_func_name)

        # Store result if needed
        if instr.returns:
            return_vreg = instr.returns[0]
            dest_loc = self.parent._get_operand_location(return_vreg)

            if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
                pass  # Already in A
            else:
                self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))
