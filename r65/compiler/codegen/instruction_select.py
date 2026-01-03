"""
Instruction selection: MIR → 65816 assembly.

Converts MIR instructions to WLA-DX assembly mnemonics with proper
addressing modes and register usage.
"""

from typing import Union, Optional
from r65.compiler.mir.nodes import *
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.register_alloc import *
from r65.compiler.codegen.memory_alloc import MemoryAllocator


class InstructionSelector:
    """
    Selects and emits 65816 instructions for MIR.

    Converts high-level MIR operations to actual 65816 assembly,
    handling addressing modes, register allocation, and instruction
    selection.
    """

    def __init__(self,
                 emitter: AssemblyEmitter,
                 register_allocator: RegisterAllocator,
                 memory_allocator: MemoryAllocator,
                 current_function: 'MIRFunction' = None):
        """
        Initialize instruction selector.

        Args:
            emitter: Assembly emitter
            register_allocator: Register allocator for virtual registers
            memory_allocator: Memory allocator for static variables
            current_function: Current MIR function being generated (for far/near context)
        """
        self.emitter = emitter
        self.reg_alloc = register_allocator
        self.mem_alloc = memory_allocator
        self.current_function = current_function

    # ========================================================================
    # Main Dispatch
    # ========================================================================

    def select_instruction(self, instr: MIRInstruction):
        """
        Select and emit assembly for MIR instruction.

        Args:
            instr: MIR instruction to convert
        """
        if isinstance(instr, Load):
            self.select_load(instr)
        elif isinstance(instr, Store):
            self.select_store(instr)
        elif isinstance(instr, Move):
            self.select_move(instr)
        elif isinstance(instr, BinaryOp):
            self.select_binary_op(instr)
        elif isinstance(instr, UnaryOp):
            self.select_unary_op(instr)
        elif isinstance(instr, Compare):
            self.select_compare(instr)
        elif isinstance(instr, Jump):
            self.select_jump(instr)
        elif isinstance(instr, CondBranch):
            self.select_cond_branch(instr)
        elif isinstance(instr, Return):
            self.select_return(instr)
        elif isinstance(instr, Call):
            self.select_call(instr)
        elif isinstance(instr, SetMode):
            self.select_set_mode(instr)
        elif isinstance(instr, SaveRegister):
            self.select_save_register(instr)
        elif isinstance(instr, RestoreRegister):
            self.select_restore_register(instr)
        elif isinstance(instr, Push):
            self.select_push(instr)
        elif isinstance(instr, Pull):
            self.select_pull(instr)
        elif isinstance(instr, ReturnFromInterrupt):
            self.select_return_from_interrupt(instr)
        else:
            raise Exception(f"Unsupported MIR instruction: {type(instr).__name__}")

    # ========================================================================
    # Memory Operations
    # ========================================================================

    def select_load(self, instr: Load):
        """
        Generate code for Load instruction.

        Load dest = *source

        Args:
            instr: Load instruction
        """
        dest_loc = self._get_operand_location(instr.dest)
        src_loc = self._get_operand_location(instr.source)

        # Determine size
        is_u16 = self._is_16bit(instr.type_info)

        if is_u16:
            # 16-bit load
            self._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            # 8-bit load
            self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

    def select_store(self, instr: Store):
        """
        Generate code for Store instruction.

        *dest = source

        Args:
            instr: Store instruction
        """
        # SPECIAL CASE: Handle immediate values
        # 65816 cannot store immediates directly - must go through accumulator
        if isinstance(instr.source, Immediate):
            dest_loc = self._get_operand_location(instr.dest)
            is_u16 = self._is_16bit(instr.type_info)

            if is_u16:
                # 16-bit immediate store
                self._emit_16bit_immediate_store(instr.source.value, dest_loc)
            else:
                # 8-bit immediate store
                value = instr.source.value & 0xFF
                self.emitter.emit_instruction("LDA", f"#${value:02X}")
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

            return

        # Normal case: memory-to-memory or register-to-memory store
        src_loc = self._get_operand_location(instr.source)
        dest_loc = self._get_operand_location(instr.dest)

        # Determine size
        is_u16 = self._is_16bit(instr.type_info)

        # Check if source is a hardware register
        if src_loc.kind == LocationKind.HARDWARE:
            # Map hardware registers to their store instructions
            store_instructions = {
                'A': 'STA',
                'X': 'STX',
                'Y': 'STY'
            }

            reg = src_loc.hw_register
            if reg not in store_instructions:
                raise Exception(f"Cannot store from hardware register: {reg}")

            # Emit appropriate store instruction
            # Note: Hardware register width (8-bit vs 16-bit) is determined by processor mode:
            # - A: 8-bit in m8 mode, 16-bit in m16 mode
            # - X/Y: 8-bit in x8 mode, 16-bit in x16 mode
            # The STA/STX/STY instruction stores the full register width automatically
            # Note: STX and STY have addressing mode restrictions on 65816:
            # - STX: zero-page, zero-page,Y, absolute only
            # - STY: zero-page, zero-page,X, absolute only
            self.emitter.emit_instruction(store_instructions[reg], self._format_operand(dest_loc))
        elif is_u16:
            # 16-bit store (memory-to-memory)
            self._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            # 8-bit store (memory-to-memory)
            self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

    def select_move(self, instr: Move):
        """
        Generate code for Move instruction.

        dest = source

        Args:
            instr: Move instruction
        """
        dest_loc = self._get_operand_location(instr.dest)
        src_operand = instr.source

        # Determine size
        is_u16 = self._is_16bit(instr.type_info)

        # SPECIAL CASE: Destination is hardware register
        if dest_loc.kind == LocationKind.HARDWARE:
            # Moving TO a hardware register
            if isinstance(src_operand, Immediate):
                # Load immediate into hardware register
                value = src_operand.value
                if dest_loc.hw_register in ['A', 'X', 'Y']:
                    self._emit_load_immediate_to_register(dest_loc.hw_register, value, is_u16)
                else:
                    raise Exception(f"Cannot load immediate into register {dest_loc.hw_register}")
            else:
                # Load from memory/register into hardware register
                src_loc = self._get_operand_location(src_operand)

                if src_loc.kind == LocationKind.HARDWARE:
                    # Register-to-register transfer
                    self._emit_register_transfer(src_loc.hw_register, dest_loc.hw_register)
                else:
                    # Load from memory into hardware register
                    operand = self._format_operand(src_loc)
                    if dest_loc.hw_register == 'A':
                        self.emitter.emit_instruction("LDA", operand)
                    elif dest_loc.hw_register == 'X':
                        self.emitter.emit_instruction("LDX", operand)
                    elif dest_loc.hw_register == 'Y':
                        self.emitter.emit_instruction("LDY", operand)
                    else:
                        raise Exception(f"Cannot load into register {dest_loc.hw_register}")
            return

        # Normal case: destination is memory/scratch
        # Handle function pointers
        from r65.compiler.mir.nodes import FunctionPointer
        if isinstance(src_operand, FunctionPointer):
            # Load address of function into destination
            # The address is a label that will be resolved by the assembler
            func_name = src_operand.function_name

            # Determine if this is near (2 bytes) or far (3 bytes) based on type
            from r65.compiler.hir.types import FunctionTypeInfo
            is_far_ptr = False
            if instr.type_info and isinstance(instr.type_info, FunctionTypeInfo):
                is_far_ptr = instr.type_info.is_far

            if is_far_ptr:
                # Far function pointer (3 bytes: bank, high, low)
                # Load low byte
                self.emitter.emit_instruction("LDA", f"#<{func_name}", "Load function address low byte")
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

                # Load high byte
                dest_high = self._offset_location(dest_loc, 1)
                self.emitter.emit_instruction("LDA", f"#>{func_name}", "Load function address high byte")
                self.emitter.emit_instruction("STA", self._format_operand(dest_high))

                # Load bank byte
                dest_bank = self._offset_location(dest_loc, 2)
                self.emitter.emit_instruction("LDA", f"#^{func_name}", "Load function bank byte")
                self.emitter.emit_instruction("STA", self._format_operand(dest_bank))
            else:
                # Near function pointer (2 bytes: high, low)
                # Load low byte
                self.emitter.emit_instruction("LDA", f"#<{func_name}", "Load function address low byte")
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

                # Load high byte
                dest_high = self._offset_location(dest_loc, 1)
                self.emitter.emit_instruction("LDA", f"#>{func_name}", "Load function address high byte")
                self.emitter.emit_instruction("STA", self._format_operand(dest_high))
            return

        # Handle immediate values
        if isinstance(src_operand, Immediate):
            value = src_operand.value

            if is_u16:
                # 16-bit immediate
                self._emit_16bit_immediate_store(value, dest_loc)
            else:
                # 8-bit immediate
                self.emitter.emit_instruction("LDA", f"#${value:02X}")
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
        else:
            # Move from register/memory
            src_loc = self._get_operand_location(src_operand)

            # Check if source is a hardware register
            if src_loc.kind == LocationKind.HARDWARE:
                # Moving FROM a hardware register TO memory
                src_reg = src_loc.hw_register

                if is_u16:
                    # 16-bit move from hardware register to memory
                    # In 16-bit mode, a single store instruction handles the full width
                    if src_reg == 'A':
                        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
                    elif src_reg == 'X':
                        self.emitter.emit_instruction("STX", self._format_operand(dest_loc))
                    elif src_reg == 'Y':
                        self.emitter.emit_instruction("STY", self._format_operand(dest_loc))
                    else:
                        raise Exception(f"Cannot move 16-bit value from register {src_reg} to memory")
                else:
                    # 8-bit move from hardware register to memory
                    if src_reg == 'A':
                        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
                    elif src_reg == 'X':
                        self.emitter.emit_instruction("STX", self._format_operand(dest_loc))
                    elif src_reg == 'Y':
                        self.emitter.emit_instruction("STY", self._format_operand(dest_loc))
                    else:
                        raise Exception(f"Cannot move from register {src_reg} to memory")
            else:
                # Moving from memory to memory
                if is_u16:
                    # 16-bit move
                    self._emit_16bit_mem_to_mem(src_loc, dest_loc)
                else:
                    # 8-bit move
                    self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
                    self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

    # ========================================================================
    # Arithmetic Operations
    # ========================================================================

    def select_binary_op(self, instr: BinaryOp):
        """
        Generate code for BinaryOp instruction.

        dest = left op right

        Args:
            instr: BinaryOp instruction
        """
        op = instr.op
        is_u16 = self._is_16bit(instr.type_info)

        # OPTIMIZATION: Detect register increment/decrement patterns
        # reg = reg + 1  →  INX/INY/INC A
        # reg = reg - 1  →  DEX/DEY/DEC A
        # Check this BEFORE getting operand locations
        if (op in ('+', '-') and
            isinstance(instr.right, Immediate) and
            instr.right.value == 1 and
            isinstance(instr.left, HardwareRegister) and
            isinstance(instr.dest, HardwareRegister) and
            instr.left.name == instr.dest.name):

            register = instr.dest.name
            if op == '+':
                # Increment
                if register == 'X':
                    self.emitter.emit_instruction("INX", comment=f"{register}++")
                    return
                elif register == 'Y':
                    self.emitter.emit_instruction("INY", comment=f"{register}++")
                    return
                elif register == 'A':
                    self.emitter.emit_instruction("INC", "A", comment="A++")
                    return
            else:  # op == '-'
                # Decrement
                if register == 'X':
                    self.emitter.emit_instruction("DEX", comment=f"{register}--")
                    return
                elif register == 'Y':
                    self.emitter.emit_instruction("DEY", comment=f"{register}--")
                    return
                elif register == 'A':
                    self.emitter.emit_instruction("DEC", "A", comment="A--")
                    return

        # Get operand locations
        left_loc = self._get_operand_location(instr.left)
        dest_loc = self._get_operand_location(instr.dest)

        # Load left operand into A (if not already there)
        if left_loc.kind == LocationKind.HARDWARE and left_loc.hw_register == 'A':
            # Left operand is already in A, no need to load
            pass
        elif left_loc.kind == LocationKind.HARDWARE:
            # Transfer from other hardware register to A
            self._emit_register_transfer(left_loc.hw_register, 'A')
        else:
            # Load left operand from memory/stack into A
            self.emitter.emit_instruction("LDA", self._format_operand(left_loc))

        # Perform operation
        if op == '+':
            self._emit_add(instr.right, is_u16)
        elif op == '-':
            self._emit_sub(instr.right, is_u16)
        elif op == '&':
            self._emit_and(instr.right, is_u16)
        elif op == '|':
            self._emit_or(instr.right, is_u16)
        elif op == '^':
            self._emit_xor(instr.right, is_u16)
        elif op == '<<':
            self._emit_shift_left(instr.right, is_u16)
        elif op == '>>':
            self._emit_shift_right(instr.right, is_u16)
        else:
            raise Exception(f"Unsupported binary operation: {op}")

        # Store result from A (if destination is not A)
        if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
            # Result is already in A, no need to store
            pass
        elif dest_loc.kind == LocationKind.HARDWARE:
            # Transfer from A to other hardware register
            self._emit_register_transfer('A', dest_loc.hw_register)
        else:
            # Store result from A to memory/stack
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

        # Handle high byte for 16-bit operations
        # NOTE: Only needed for memory-to-memory operations
        # Hardware registers in 16-bit mode are handled as single 16-bit values
        if is_u16 and op in ('+', '-'):
            # Skip high byte handling if any operand is a hardware register
            # In 16-bit mode (m16/x16), hardware registers are accessed as complete 16-bit values
            if (left_loc.kind == LocationKind.HARDWARE or
                dest_loc.kind == LocationKind.HARDWARE):
                # Hardware registers don't need separate high byte handling
                # The single operation above already handled the full 16-bit value
                pass
            else:
                # Memory-to-memory 16-bit operation: handle high byte separately
                left_high = self._offset_location(left_loc, 1)
                dest_high = self._offset_location(dest_loc, 1)

                self.emitter.emit_instruction("LDA", self._format_operand(left_high))

                if isinstance(instr.right, Immediate):
                    high_value = (instr.right.value >> 8) & 0xFF
                    if op == '+':
                        self.emitter.emit_instruction("ADC", f"#${high_value:02X}")
                    else:  # '-'
                        self.emitter.emit_instruction("SBC", f"#${high_value:02X}")
                else:
                    right_loc = self._get_operand_location(instr.right)
                    if right_loc.kind != LocationKind.HARDWARE:
                        right_high = self._offset_location(right_loc, 1)
                        if op == '+':
                            self.emitter.emit_instruction("ADC", self._format_operand(right_high))
                        else:  # '-'
                            self.emitter.emit_instruction("SBC", self._format_operand(right_high))

                self.emitter.emit_instruction("STA", self._format_operand(dest_high))

    def select_unary_op(self, instr: UnaryOp):
        """
        Generate code for UnaryOp instruction.

        dest = op operand

        Args:
            instr: UnaryOp instruction
        """
        op = instr.op
        operand_loc = self._get_operand_location(instr.operand)
        dest_loc = self._get_operand_location(instr.dest)

        # Load operand
        self.emitter.emit_instruction("LDA", self._format_operand(operand_loc))

        # Perform operation
        if op == '!':
            # Logical NOT: convert to 0 or 1, then XOR with 1
            self.emitter.emit_instruction("CMP", "#0", "Check if zero")
            self.emitter.emit_instruction("BEQ", "+", "Branch if zero")
            self.emitter.emit_instruction("LDA", "#0", "Was non-zero, result = 0")
            self.emitter.emit_instruction("BRA", "++")
            self.emitter.emit_label("+")
            self.emitter.emit_instruction("LDA", "#1", "Was zero, result = 1")
            self.emitter.emit_label("++")
        elif op == '~':
            # Bitwise NOT
            self.emitter.emit_instruction("EOR", "#$FF", "Bitwise complement")
        elif op == '-':
            # Negation
            self.emitter.emit_instruction("EOR", "#$FF", "Complement")
            self.emitter.emit_instruction("INC", "A", "Add 1 for two's complement")
        else:
            raise Exception(f"Unsupported unary operation: {op}")

        # Store result
        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

    def select_compare(self, instr: 'Compare'):
        """
        Generate code for Compare instruction.

        Emits CMP/CPX/CPY instruction and sets processor flags for subsequent
        conditional branch.

        Args:
            instr: Compare instruction
        """
        left_loc = self._get_operand_location(instr.left)
        right_loc = self._get_operand_location(instr.right)

        # Determine which comparison instruction to use based on left operand
        if left_loc.kind == LocationKind.HARDWARE:
            # Hardware register comparison
            if left_loc.hw_register == 'X':
                # CPX instruction
                self.emitter.emit_instruction("CPX", self._format_operand(right_loc))
            elif left_loc.hw_register == 'Y':
                # CPY instruction
                self.emitter.emit_instruction("CPY", self._format_operand(right_loc))
            elif left_loc.hw_register == 'A':
                # CMP instruction (A is implicit)
                self.emitter.emit_instruction("CMP", self._format_operand(right_loc))
            else:
                # Other hardware registers - load to A first
                self.emitter.emit_instruction("LDA", self._format_operand(left_loc))
                self.emitter.emit_instruction("CMP", self._format_operand(right_loc))
        else:
            # Memory or virtual register - load to A and compare
            self.emitter.emit_instruction("LDA", self._format_operand(left_loc))
            self.emitter.emit_instruction("CMP", self._format_operand(right_loc))

        # Flags are now set for conditional branch
        # Z flag: set if left == right
        # C flag: set if left >= right (unsigned)
        # N flag: set if result is negative (signed)

    # ========================================================================
    # Arithmetic Helpers
    # ========================================================================

    def _emit_add(self, right_operand, is_u16: bool):
        """Emit addition operation."""
        self.emitter.emit_instruction("CLC")
        self._emit_binary_operation_with_operand("ADC", right_operand, is_u16)

    def _emit_sub(self, right_operand, is_u16: bool):
        """Emit subtraction operation."""
        self.emitter.emit_instruction("SEC")
        self._emit_binary_operation_with_operand("SBC", right_operand, is_u16)

    def _emit_and(self, right_operand, is_u16: bool):
        """Emit bitwise AND operation."""
        self._emit_binary_operation_with_operand("AND", right_operand, is_u16)

    def _emit_or(self, right_operand, is_u16: bool):
        """Emit bitwise OR operation."""
        self._emit_binary_operation_with_operand("ORA", right_operand, is_u16)

    def _emit_xor(self, right_operand, is_u16: bool):
        """Emit bitwise XOR operation."""
        self._emit_binary_operation_with_operand("EOR", right_operand, is_u16)

    def _emit_shift_left(self, right_operand, is_u16: bool):
        """Emit left shift operation (A << count)."""
        # Right operand must be immediate constant for shifts
        if not isinstance(right_operand, Immediate):
            raise Exception("Shift count must be constant")

        count = right_operand.value
        for _ in range(count):
            self.emitter.emit_instruction("ASL", "A")

    def _emit_shift_right(self, right_operand, is_u16: bool):
        """Emit right shift operation (A >> count)."""
        # Right operand must be immediate constant for shifts
        if not isinstance(right_operand, Immediate):
            raise Exception("Shift count must be constant")

        count = right_operand.value
        for _ in range(count):
            self.emitter.emit_instruction("LSR", "A")

    # ========================================================================
    # Control Flow
    # ========================================================================

    def select_jump(self, instr: Jump):
        """
        Generate code for Jump instruction.

        Args:
            instr: Jump instruction
        """
        self.emitter.emit_instruction("JMP", f"__L{instr.target}")

    def select_cond_branch(self, instr: CondBranch):
        """
        Generate code for CondBranch instruction.

        Two modes:
        1. If condition is None: Branch based on flags from preceding Compare instruction
        2. If condition is a vreg: Load condition and branch on zero/non-zero

        Args:
            instr: CondBranch instruction
        """
        if instr.condition is None:
            # Branch based on comparison flags from preceding Compare instruction
            # Flags are already set by CMP/CPX/CPY instruction
            # Z flag: set if left == right
            # C flag: set if left >= right (unsigned)
            # N flag: set if result is negative (signed)

            if instr.comparison == '==':
                # Branch if equal (Z flag set)
                self.emitter.emit_instruction("BEQ", f"__L{instr.true_target}", "Branch if equal")
                self.emitter.emit_instruction("JMP", f"__L{instr.false_target}")
            elif instr.comparison == '!=':
                # Branch if not equal (Z flag clear)
                self.emitter.emit_instruction("BNE", f"__L{instr.true_target}", "Branch if not equal")
                self.emitter.emit_instruction("JMP", f"__L{instr.false_target}")
            elif instr.comparison == '<':
                # Branch if less than (unsigned: C flag clear)
                # For signed, we'd need to check N and V flags
                self.emitter.emit_instruction("BCC", f"__L{instr.true_target}", "Branch if less than (unsigned)")
                self.emitter.emit_instruction("JMP", f"__L{instr.false_target}")
            elif instr.comparison == '>=':
                # Branch if greater or equal (unsigned: C flag set)
                self.emitter.emit_instruction("BCS", f"__L{instr.true_target}", "Branch if >= (unsigned)")
                self.emitter.emit_instruction("JMP", f"__L{instr.false_target}")
            elif instr.comparison == '>':
                # Branch if greater than: !(left <= right) = !(C clear OR Z set)
                # Equivalent to: (C set) AND (Z clear)
                self.emitter.emit_instruction("BEQ", f"__L{instr.false_target}", "Skip if equal")
                self.emitter.emit_instruction("BCS", f"__L{instr.true_target}", "Branch if > (unsigned)")
                self.emitter.emit_instruction("JMP", f"__L{instr.false_target}")
            elif instr.comparison == '<=':
                # Branch if less or equal: (C clear) OR (Z set)
                self.emitter.emit_instruction("BEQ", f"__L{instr.true_target}", "Branch if equal")
                self.emitter.emit_instruction("BCC", f"__L{instr.true_target}", "Branch if less than")
                self.emitter.emit_instruction("JMP", f"__L{instr.false_target}")
            else:
                raise Exception(f"Unsupported comparison type for flag-based branch: {instr.comparison}")
        else:
            # Branch based on condition value (zero/non-zero)
            cond_loc = self._get_operand_location(instr.condition)
            self.emitter.emit_instruction("LDA", self._format_operand(cond_loc))

            if instr.comparison == '!=':
                # Branch if not equal to zero
                self.emitter.emit_instruction("BEQ", f"__L{instr.false_target}", "Branch if zero")
                self.emitter.emit_instruction("JMP", f"__L{instr.true_target}")
            elif instr.comparison == '==':
                # Branch if equal to zero
                self.emitter.emit_instruction("BNE", f"__L{instr.false_target}", "Branch if non-zero")
                self.emitter.emit_instruction("JMP", f"__L{instr.true_target}")
            else:
                # For other comparisons on boolean values, treat as != 0
                self.emitter.emit_instruction("BEQ", f"__L{instr.false_target}")
                self.emitter.emit_instruction("JMP", f"__L{instr.true_target}")

    def select_return(self, instr: Return):
        """
        Generate code for Return instruction.

        Handles loading return values into appropriate registers before returning.

        Args:
            instr: Return instruction
        """
        # Handle return values
        # Convention: First return value in A, second in X, third in Y
        if instr.values:
            return_registers = ['A', 'X', 'Y']

            for i, value in enumerate(instr.values):
                if i >= len(return_registers):
                    raise Exception(f"Too many return values (max {len(return_registers)})")

                target_reg = return_registers[i]
                value_loc = self._get_operand_location(value)

                # Load value into target return register
                if value_loc.kind == LocationKind.HARDWARE and value_loc.hw_register == target_reg:
                    # Already in correct register
                    pass
                elif value_loc.kind == LocationKind.HARDWARE:
                    # Transfer from one hardware register to another
                    if value_loc.hw_register == 'A' and target_reg == 'X':
                        self.emitter.emit_instruction("TAX", comment="Transfer return value A to X")
                    elif value_loc.hw_register == 'A' and target_reg == 'Y':
                        self.emitter.emit_instruction("TAY", comment="Transfer return value A to Y")
                    elif value_loc.hw_register == 'X' and target_reg == 'A':
                        self.emitter.emit_instruction("TXA", comment="Transfer return value X to A")
                    elif value_loc.hw_register == 'Y' and target_reg == 'A':
                        self.emitter.emit_instruction("TYA", comment="Transfer return value Y to A")
                    else:
                        # Complex transfer - go through A or memory
                        raise Exception(f"Cannot transfer {value_loc.hw_register} to {target_reg}")
                else:
                    # Load from memory/stack into return register
                    if target_reg == 'A':
                        self.emitter.emit_instruction("LDA", self._format_operand(value_loc))
                    elif target_reg == 'X':
                        self.emitter.emit_instruction("LDX", self._format_operand(value_loc))
                    elif target_reg == 'Y':
                        self.emitter.emit_instruction("LDY", self._format_operand(value_loc))

        # Emit epilogue (DBR restoration for data_bank=auto)
        if self.current_function and self.current_function.is_far and self.current_function.bank_attr:
            from r65.compiler.hir.attributes import DataBankMode

            if self.current_function.bank_attr.data_bank == DataBankMode.AUTO:
                # Restore original DBR before returning
                self.emitter.emit_instruction("PLB", comment="Restore data bank")

        # Emit return instruction (RTL for far functions, RTS for near)
        if self.current_function and self.current_function.is_far:
            self.emitter.emit_instruction("RTL")
        else:
            self.emitter.emit_instruction("RTS")

    # ========================================================================
    # Function Calls
    # ========================================================================

    def select_call(self, instr: Call):
        """
        Generate code for Call instruction.

        Handles argument setup, call, and return value collection.

        Args:
            instr: Call instruction
        """
        from r65.compiler.mir.nodes import ArgumentMechanism, Immediate

        # Step 1: Set up arguments
        # Process in order: stack arguments first, then register/variable arguments
        stack_arg_count = 0

        for arg in instr.args:
            arg_loc = self._get_operand_location(arg.value)

            if arg.mechanism == ArgumentMechanism.STACK:
                # Push argument onto stack
                # Load value into A and push
                if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
                    # Already in A
                    pass
                elif arg_loc.kind == LocationKind.HARDWARE:
                    # Transfer to A
                    if arg_loc.hw_register == 'X':
                        self.emitter.emit_instruction("TXA")
                    elif arg_loc.hw_register == 'Y':
                        self.emitter.emit_instruction("TYA")
                elif isinstance(arg.value, Immediate):
                    # Load immediate
                    self.emitter.emit_instruction("LDA", f"#${arg.value.value:02X}")
                else:
                    # Load from memory/stack
                    self.emitter.emit_instruction("LDA", self._format_operand(arg_loc))

                # Push A onto stack
                self.emitter.emit_instruction("PHA", comment=f"Push stack arg {stack_arg_count}")
                stack_arg_count += 1

            elif arg.mechanism == ArgumentMechanism.REGISTER:
                # Move value into specified hardware register
                target_reg = arg.location.name if hasattr(arg.location, 'name') else str(arg.location)

                if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == target_reg:
                    # Already in correct register
                    pass
                elif isinstance(arg.value, Immediate):
                    # Load immediate into target register
                    if target_reg == 'A':
                        self.emitter.emit_instruction("LDA", f"#${arg.value.value:02X}")
                    elif target_reg == 'X':
                        self.emitter.emit_instruction("LDX", f"#${arg.value.value:02X}")
                    elif target_reg == 'Y':
                        self.emitter.emit_instruction("LDY", f"#${arg.value.value:02X}")
                else:
                    # Load from memory/vreg into target register
                    if target_reg == 'A':
                        self.emitter.emit_instruction("LDA", self._format_operand(arg_loc))
                    elif target_reg == 'X':
                        self.emitter.emit_instruction("LDX", self._format_operand(arg_loc))
                    elif target_reg == 'Y':
                        self.emitter.emit_instruction("LDY", self._format_operand(arg_loc))

            elif arg.mechanism == ArgumentMechanism.VARIABLE:
                # Store value into specified memory location
                # First load into A, then store
                if arg_loc.kind == LocationKind.HARDWARE and arg_loc.hw_register == 'A':
                    # Already in A
                    pass
                elif arg_loc.kind == LocationKind.HARDWARE:
                    # Transfer to A
                    if arg_loc.hw_register == 'X':
                        self.emitter.emit_instruction("TXA")
                    elif arg_loc.hw_register == 'Y':
                        self.emitter.emit_instruction("TYA")
                elif isinstance(arg.value, Immediate):
                    self.emitter.emit_instruction("LDA", f"#${arg.value.value:02X}")
                else:
                    self.emitter.emit_instruction("LDA", self._format_operand(arg_loc))

                # Store to variable location
                # arg.location should be a symbol or memory location
                # For now, assume it's a simple memory address
                # TODO: Handle symbol resolution properly
                self.emitter.emit_instruction("STA", str(arg.location))

        # Step 2: Handle caller-managed DBR (data_bank=caller)
        needs_dbr_restore = False
        if instr.is_far and instr.bank_attr:
            from r65.compiler.hir.attributes import DataBankMode

            if instr.bank_attr.data_bank == DataBankMode.CALLER:
                # Caller manages DBR: save, set, call, restore
                needs_dbr_restore = True
                self.emitter.emit_instruction("PHB", comment="Save current data bank (caller)")
                self.emitter.emit_instruction("LDA", f"#${instr.bank_attr.bank_number:02X}",
                                            "Load callee's bank number")
                self.emitter.emit_instruction("PHA", comment="Push bank number")
                self.emitter.emit_instruction("PLB", comment="Set data bank for callee")

        # Step 3: Make the call
        # Check if this is an indirect call (function pointer)
        from r65.compiler.mir.nodes import VirtualRegister

        if isinstance(instr.function, VirtualRegister):
            # Indirect call through function pointer - generate trampoline
            self._emit_indirect_call_trampoline(instr.function, instr.is_far)
        elif isinstance(instr.function, str):
            # Direct call
            if instr.is_far:
                self.emitter.emit_instruction("JSL", instr.function)
            else:
                self.emitter.emit_instruction("JSR", instr.function)
        else:
            raise Exception(f"Unknown function type in Call: {type(instr.function)}")

        # Step 4: Restore DBR if caller-managed
        if needs_dbr_restore:
            self.emitter.emit_instruction("PLB", comment="Restore data bank (caller)")

        # Step 5: Clean up stack arguments (callee cleanup for R65)
        # Stack grows downward on 65816, so we adjust SP
        if stack_arg_count > 0:
            # Each argument is 1-2 bytes (depends on type)
            # For simplicity, assume each arg is 1 byte (will need type info for proper handling)
            # We can pull and discard, or adjust stack pointer directly
            for _ in range(stack_arg_count):
                self.emitter.emit_instruction("PLA", comment="Clean up stack arg")

        # Step 4: Collect return values
        # Return values come back in A, X, Y (in order)
        if instr.returns:
            return_registers = ['A', 'X', 'Y']

            for i, return_vreg in enumerate(instr.returns):
                if i >= len(return_registers):
                    raise Exception(f"Too many return values (max {len(return_registers)})")

                source_reg = return_registers[i]
                dest_loc = self._get_operand_location(return_vreg)

                # Move from return register to destination
                if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == source_reg:
                    # Already in correct location
                    pass
                elif dest_loc.kind == LocationKind.HARDWARE:
                    # Transfer between hardware registers
                    if source_reg == 'A' and dest_loc.hw_register == 'X':
                        self.emitter.emit_instruction("TAX")
                    elif source_reg == 'A' and dest_loc.hw_register == 'Y':
                        self.emitter.emit_instruction("TAY")
                    elif source_reg == 'X' and dest_loc.hw_register == 'A':
                        self.emitter.emit_instruction("TXA")
                    elif source_reg == 'Y' and dest_loc.hw_register == 'A':
                        self.emitter.emit_instruction("TYA")
                else:
                    # Store from return register to memory/stack
                    if source_reg == 'A':
                        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
                    elif source_reg == 'X':
                        self.emitter.emit_instruction("STX", self._format_operand(dest_loc))
                    elif source_reg == 'Y':
                        self.emitter.emit_instruction("STY", self._format_operand(dest_loc))

    def _emit_indirect_call_trampoline(self, func_ptr_vreg, is_far: bool):
        """
        Generate trampoline for indirect function call through function pointer.

        Near trampoline (16-bit address):
            LDA func_ptr+1  ; High byte
            PHA
            LDA func_ptr    ; Low byte
            PHA
            RTS             ; Jumps to address on stack (PC = popped address)

        Far trampoline (24-bit address):
            LDA func_ptr+2  ; Bank byte
            PHA
            LDA func_ptr+1  ; High byte
            PHA
            LDA func_ptr    ; Low byte
            PHA
            RTL             ; Long return (PC:PBR = popped 24-bit address)

        Args:
            func_ptr_vreg: VirtualRegister holding the function pointer
            is_far: True for far call (24-bit), False for near call (16-bit)
        """
        # Get the location of the function pointer
        ptr_loc = self._get_operand_location(func_ptr_vreg)

        if is_far:
            # Far call trampoline (24-bit address)
            # Push bank byte (highest byte)
            bank_loc = self._offset_location(ptr_loc, 2)
            self.emitter.emit_instruction("LDA", self._format_operand(bank_loc), "Load bank byte")
            self.emitter.emit_instruction("PHA", comment="Push bank")

            # Push high byte
            high_loc = self._offset_location(ptr_loc, 1)
            self.emitter.emit_instruction("LDA", self._format_operand(high_loc), "Load high byte")
            self.emitter.emit_instruction("PHA", comment="Push high")

            # Push low byte
            self.emitter.emit_instruction("LDA", self._format_operand(ptr_loc), "Load low byte")
            self.emitter.emit_instruction("PHA", comment="Push low")

            # RTL pops 24-bit address from stack and jumps
            self.emitter.emit_instruction("RTL", comment="Indirect far call via trampoline")
        else:
            # Near call trampoline (16-bit address)
            # Push high byte first (stack is little-endian on 65816)
            high_loc = self._offset_location(ptr_loc, 1)
            self.emitter.emit_instruction("LDA", self._format_operand(high_loc), "Load high byte")
            self.emitter.emit_instruction("PHA", comment="Push high")

            # Push low byte
            self.emitter.emit_instruction("LDA", self._format_operand(ptr_loc), "Load low byte")
            self.emitter.emit_instruction("PHA", comment="Push low")

            # RTS pops 16-bit address from stack and jumps
            self.emitter.emit_instruction("RTS", comment="Indirect near call via trampoline")

    # ========================================================================
    # Mode Control
    # ========================================================================

    def select_set_mode(self, instr: SetMode):
        """
        Generate code for SetMode instruction.

        Args:
            instr: SetMode instruction
        """
        if instr.is_set:
            self.emitter.emit_instruction("SEP", f"#${instr.mask:02X}")
        else:
            self.emitter.emit_instruction("REP", f"#${instr.mask:02X}")

    # ========================================================================
    # Register Save/Restore
    # ========================================================================

    def select_save_register(self, instr: SaveRegister):
        """
        Generate code for SaveRegister instruction.

        Args:
            instr: SaveRegister instruction
        """
        reg_name = instr.register.name

        if reg_name == 'A':
            self.emitter.emit_instruction("PHA")
        elif reg_name == 'X':
            self.emitter.emit_instruction("PHX")
        elif reg_name == 'Y':
            self.emitter.emit_instruction("PHY")
        elif reg_name == 'STATUS':
            self.emitter.emit_instruction("PHP")
        elif reg_name == 'D':
            self.emitter.emit_instruction("PHD")
        elif reg_name == 'DBR':
            self.emitter.emit_instruction("PHB")
        else:
            raise Exception(f"Cannot push register: {reg_name}")

    def select_restore_register(self, instr: RestoreRegister):
        """
        Generate code for RestoreRegister instruction.

        Args:
            instr: RestoreRegister instruction
        """
        reg_name = instr.register.name

        if reg_name == 'A':
            self.emitter.emit_instruction("PLA")
        elif reg_name == 'X':
            self.emitter.emit_instruction("PLX")
        elif reg_name == 'Y':
            self.emitter.emit_instruction("PLY")
        elif reg_name == 'STATUS':
            self.emitter.emit_instruction("PLP")
        elif reg_name == 'D':
            self.emitter.emit_instruction("PLD")
        elif reg_name == 'DBR':
            self.emitter.emit_instruction("PLB")
        else:
            raise Exception(f"Cannot pull register: {reg_name}")

    # ========================================================================
    # Interrupt Handler Instructions
    # ========================================================================

    def select_push(self, instr: Push):
        """
        Generate code for Push instruction (save register to stack).

        Args:
            instr: Push instruction
        """
        reg = instr.register.name
        if reg == 'STATUS':
            self.emitter.emit_instruction("PHP")  # Push processor status
        elif reg == 'A':
            self.emitter.emit_instruction("PHA")  # Push accumulator
        elif reg == 'X':
            self.emitter.emit_instruction("PHX")  # Push X
        elif reg == 'Y':
            self.emitter.emit_instruction("PHY")  # Push Y
        elif reg == 'D':
            self.emitter.emit_instruction("PHD")  # Push direct page
        elif reg == 'DBR':
            self.emitter.emit_instruction("PHB")  # Push data bank
        else:
            raise Exception(f"Cannot push register: {reg}")

    def select_pull(self, instr: Pull):
        """
        Generate code for Pull instruction (restore register from stack).

        Args:
            instr: Pull instruction
        """
        reg = instr.register.name
        if reg == 'STATUS':
            self.emitter.emit_instruction("PLP")  # Pull processor status
        elif reg == 'A':
            self.emitter.emit_instruction("PLA")  # Pull accumulator
        elif reg == 'X':
            self.emitter.emit_instruction("PLX")  # Pull X
        elif reg == 'Y':
            self.emitter.emit_instruction("PLY")  # Pull Y
        elif reg == 'D':
            self.emitter.emit_instruction("PLD")  # Pull direct page
        elif reg == 'DBR':
            self.emitter.emit_instruction("PLB")  # Pull data bank
        else:
            raise Exception(f"Cannot pull register: {reg}")

    def select_return_from_interrupt(self, instr: ReturnFromInterrupt):
        """
        Generate code for ReturnFromInterrupt instruction.

        Args:
            instr: ReturnFromInterrupt instruction
        """
        self.emitter.emit_instruction("RTI")  # Return from interrupt

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _emit_binary_operation_with_operand(self, operation: str, right_operand, is_u16: bool):
        """
        Emit a binary operation with right operand.

        Handles immediate, memory, and hardware register operands.
        For hardware registers, stores to temp location $00 first.

        Args:
            operation: Instruction mnemonic (ADC, SBC, AND, ORA, EOR)
            right_operand: Right operand (Immediate, VirtualRegister, HardwareRegister)
            is_u16: Whether this is a 16-bit operation
        """
        if isinstance(right_operand, Immediate):
            value = right_operand.value & 0xFF if not is_u16 else right_operand.value
            self.emitter.emit_instruction(operation, f"#${value:02X}")
        else:
            right_loc = self._get_operand_location(right_operand)
            if right_loc.kind == LocationKind.HARDWARE:
                # Hardware register - must store to temp location first
                # (65816 can't use hardware registers as operands for these ops)
                if right_loc.hw_register in ['A', 'X', 'Y']:
                    store_instr = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}[right_loc.hw_register]
                    self.emitter.emit_instruction(store_instr, "$00", f"Store {right_loc.hw_register} to temp")
                    self.emitter.emit_instruction(operation, "$00")
                else:
                    raise Exception(f"Cannot use hardware register in operation: {right_loc.hw_register}")
            else:
                # Memory location
                self.emitter.emit_instruction(operation, self._format_operand(right_loc))

    def _emit_16bit_mem_to_mem(self, src_loc: PhysicalLocation, dest_loc: PhysicalLocation, comment: str = None):
        """
        Emit 16-bit memory-to-memory move (low byte + high byte).

        Args:
            src_loc: Source memory location
            dest_loc: Destination memory location
            comment: Optional comment for first instruction
        """
        # Low byte
        self.emitter.emit_instruction("LDA", self._format_operand(src_loc), comment)
        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

        # High byte
        src_high = self._offset_location(src_loc, 1)
        dest_high = self._offset_location(dest_loc, 1)
        self.emitter.emit_instruction("LDA", self._format_operand(src_high))
        self.emitter.emit_instruction("STA", self._format_operand(dest_high))

    def _emit_16bit_immediate_store(self, value: int, dest_loc: PhysicalLocation):
        """
        Emit 16-bit immediate store (split into low/high bytes).

        Args:
            value: 16-bit immediate value
            dest_loc: Destination memory location
        """
        low = value & 0xFF
        high = (value >> 8) & 0xFF

        # Low byte
        self.emitter.emit_instruction("LDA", f"#${low:02X}")
        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

        # High byte
        dest_high = self._offset_location(dest_loc, 1)
        self.emitter.emit_instruction("LDA", f"#${high:02X}")
        self.emitter.emit_instruction("STA", self._format_operand(dest_high))

    def _emit_register_transfer(self, src_reg: str, dest_reg: str):
        """
        Emit register-to-register transfer.

        Handles all valid 65816 register transfer combinations.
        For indirect transfers (e.g., X to Y), routes through A.

        Args:
            src_reg: Source register name ('A', 'X', 'Y')
            dest_reg: Destination register name ('A', 'X', 'Y')
        """
        if src_reg == dest_reg:
            # No-op
            return

        # Direct transfers
        transfer_map = {
            ('A', 'X'): 'TAX',
            ('A', 'Y'): 'TAY',
            ('X', 'A'): 'TXA',
            ('Y', 'A'): 'TYA',
        }

        if (src_reg, dest_reg) in transfer_map:
            self.emitter.emit_instruction(transfer_map[(src_reg, dest_reg)])
        else:
            # Indirect transfer through A (e.g., X to Y)
            if src_reg != 'A':
                self._emit_register_transfer(src_reg, 'A')
            if dest_reg != 'A':
                self._emit_register_transfer('A', dest_reg)

    def _emit_load_immediate_to_register(self, reg: str, value: int, is_u16: bool):
        """
        Emit load immediate into hardware register.

        Args:
            reg: Register name ('A', 'X', 'Y')
            value: Immediate value
            is_u16: Whether to use 16-bit format
        """
        load_instr = {'A': 'LDA', 'X': 'LDX', 'Y': 'LDY'}[reg]

        if is_u16:
            self.emitter.emit_instruction(load_instr, f"#${value:04X}")
        else:
            self.emitter.emit_instruction(load_instr, f"#${value:02X}")

    def _get_operand_location(self, operand) -> PhysicalLocation:
        """
        Get physical location for an operand.

        Args:
            operand: VirtualRegister, HardwareRegister, MemoryLocation, or Immediate

        Returns:
            PhysicalLocation for the operand
        """
        if isinstance(operand, VirtualRegister):
            return self.reg_alloc.get_location(operand)
        elif isinstance(operand, HardwareRegister):
            return self.reg_alloc.get_hw_location(operand)
        elif isinstance(operand, MemoryLocation):
            # Get address from memory allocator
            alloc = self.mem_alloc.get_allocation(operand.symbol)
            if alloc:
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_addr=alloc.address,
                    size=alloc.size
                )
            else:
                raise Exception(f"No allocation for symbol: {operand.symbol.name}")
        elif isinstance(operand, Immediate):
            # Immediate value - return as immediate location
            return PhysicalLocation(
                kind=LocationKind.IMMEDIATE,
                immediate_value=operand.value,
                size=1  # Will be determined by context
            )
        else:
            raise Exception(f"Unknown operand type: {type(operand)}")

    def _format_operand(self, location: PhysicalLocation) -> str:
        """
        Format physical location as assembly operand.

        Args:
            location: Physical location

        Returns:
            Formatted operand string
        """
        if location.kind == LocationKind.HARDWARE:
            # Hardware register - can't be used as memory operand
            # This shouldn't happen in normal code generation
            raise Exception(f"Cannot use hardware register as memory operand: {location.hw_register}")
        elif location.kind == LocationKind.SCRATCH:
            return f"${location.scratch_addr:02X}"
        elif location.kind == LocationKind.MEMORY:
            if location.memory_addr < 0x100:
                # Zero-page
                return f"${location.memory_addr:02X}"
            else:
                # Absolute
                return f"${location.memory_addr:04X}"
        elif location.kind == LocationKind.STACK:
            # Stack-relative addressing using 65816 stack-relative mode
            # Format: $XX,S where XX is the offset from stack pointer
            return f"${location.stack_offset:02X},S"
        elif location.kind == LocationKind.IMMEDIATE:
            # Immediate value
            return f"#{location.immediate_value}"
        else:
            raise Exception(f"Unknown location kind: {location.kind}")

    def _offset_location(self, location: PhysicalLocation, offset: int) -> PhysicalLocation:
        """
        Create new location offset from given location.

        Args:
            location: Base location
            offset: Byte offset

        Returns:
            New PhysicalLocation at offset
        """
        if location.kind == LocationKind.SCRATCH:
            return PhysicalLocation(
                kind=LocationKind.SCRATCH,
                scratch_addr=location.scratch_addr + offset,
                size=1
            )
        elif location.kind == LocationKind.MEMORY:
            return PhysicalLocation(
                kind=LocationKind.MEMORY,
                memory_addr=location.memory_addr + offset,
                size=1
            )
        elif location.kind == LocationKind.STACK:
            return PhysicalLocation(
                kind=LocationKind.STACK,
                stack_offset=location.stack_offset + offset,
                size=1
            )
        else:
            raise Exception(f"Cannot offset location kind: {location.kind}")

    def _is_16bit(self, type_info) -> bool:
        """
        Check if type is 16-bit.

        Args:
            type_info: Type information

        Returns:
            True if 16-bit type
        """
        if hasattr(type_info, 'name'):
            return type_info.name in ('u16', 'i16')
        return False
