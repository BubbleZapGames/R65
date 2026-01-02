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
                 memory_allocator: MemoryAllocator):
        """
        Initialize instruction selector.

        Args:
            emitter: Assembly emitter
            register_allocator: Register allocator for virtual registers
            memory_allocator: Memory allocator for static variables
        """
        self.emitter = emitter
        self.reg_alloc = register_allocator
        self.mem_alloc = memory_allocator

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
            self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

            # High byte
            src_high = self._offset_location(src_loc, 1)
            dest_high = self._offset_location(dest_loc, 1)
            self.emitter.emit_instruction("LDA", self._format_operand(src_high))
            self.emitter.emit_instruction("STA", self._format_operand(dest_high))
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
                value = instr.source.value
                low = value & 0xFF
                high = (value >> 8) & 0xFF

                # Low byte
                self.emitter.emit_instruction("LDA", f"#${low:02X}")
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

                # High byte
                dest_high = self._offset_location(dest_loc, 1)
                self.emitter.emit_instruction("LDA", f"#${high:02X}")
                self.emitter.emit_instruction("STA", self._format_operand(dest_high))
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

        # Check if source is already in A
        if src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            # Source is already in A, just store it
            if is_u16:
                # 16-bit store - this shouldn't happen with hardware registers
                # (A is 8-bit, need special handling for 16-bit)
                raise Exception("Cannot store 16-bit value from A register")
            else:
                # 8-bit store from A
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
        elif is_u16:
            # 16-bit store (memory-to-memory)
            self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

            # High byte
            src_high = self._offset_location(src_loc, 1)
            dest_high = self._offset_location(dest_loc, 1)
            self.emitter.emit_instruction("LDA", self._format_operand(src_high))
            self.emitter.emit_instruction("STA", self._format_operand(dest_high))
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
                if dest_loc.hw_register == 'A':
                    if is_u16:
                        self.emitter.emit_instruction("LDA", f"#${value:04X}")
                    else:
                        self.emitter.emit_instruction("LDA", f"#${value:02X}")
                elif dest_loc.hw_register == 'X':
                    if is_u16:
                        self.emitter.emit_instruction("LDX", f"#${value:04X}")
                    else:
                        self.emitter.emit_instruction("LDX", f"#${value:02X}")
                elif dest_loc.hw_register == 'Y':
                    if is_u16:
                        self.emitter.emit_instruction("LDY", f"#${value:04X}")
                    else:
                        self.emitter.emit_instruction("LDY", f"#${value:02X}")
                else:
                    raise Exception(f"Cannot load immediate into register {dest_loc.hw_register}")
            else:
                # Load from memory/register into hardware register
                src_loc = self._get_operand_location(src_operand)

                if src_loc.kind == LocationKind.HARDWARE:
                    # Register-to-register transfer
                    src_reg = src_loc.hw_register
                    dest_reg = dest_loc.hw_register

                    if src_reg == dest_reg:
                        # No-op: same register
                        return
                    elif src_reg == 'A' and dest_reg == 'X':
                        self.emitter.emit_instruction("TAX")
                    elif src_reg == 'A' and dest_reg == 'Y':
                        self.emitter.emit_instruction("TAY")
                    elif src_reg == 'X' and dest_reg == 'A':
                        self.emitter.emit_instruction("TXA")
                    elif src_reg == 'Y' and dest_reg == 'A':
                        self.emitter.emit_instruction("TYA")
                    else:
                        # For other combinations, go through A
                        if src_reg == 'X':
                            self.emitter.emit_instruction("TXA")
                        elif src_reg == 'Y':
                            self.emitter.emit_instruction("TYA")

                        if dest_reg == 'X':
                            self.emitter.emit_instruction("TAX")
                        elif dest_reg == 'Y':
                            self.emitter.emit_instruction("TAY")
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
        # Handle immediate values
        if isinstance(src_operand, Immediate):
            value = src_operand.value

            if is_u16:
                # 16-bit immediate
                low_byte = value & 0xFF
                high_byte = (value >> 8) & 0xFF

                self.emitter.emit_instruction("LDA", f"#${low_byte:02X}")
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

                dest_high = self._offset_location(dest_loc, 1)
                self.emitter.emit_instruction("LDA", f"#${high_byte:02X}")
                self.emitter.emit_instruction("STA", self._format_operand(dest_high))
            else:
                # 8-bit immediate
                self.emitter.emit_instruction("LDA", f"#${value:02X}")
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
        else:
            # Move from register/memory
            src_loc = self._get_operand_location(src_operand)

            if is_u16:
                # 16-bit move
                self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

                src_high = self._offset_location(src_loc, 1)
                dest_high = self._offset_location(dest_loc, 1)
                self.emitter.emit_instruction("LDA", self._format_operand(src_high))
                self.emitter.emit_instruction("STA", self._format_operand(dest_high))
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

        # Get operand locations
        left_loc = self._get_operand_location(instr.left)
        dest_loc = self._get_operand_location(instr.dest)

        # Load left operand into A (if not already there)
        if left_loc.kind == LocationKind.HARDWARE and left_loc.hw_register == 'A':
            # Left operand is already in A, no need to load
            pass
        else:
            # Load left operand into A
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
        else:
            # Store result from A
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

        # Handle high byte for 16-bit operations
        if is_u16 and op in ('+', '-'):
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
                right_high = self._offset_location(self._get_operand_location(instr.right), 1)
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

    # ========================================================================
    # Arithmetic Helpers
    # ========================================================================

    def _emit_add(self, right_operand, is_u16: bool):
        """Emit addition operation."""
        self.emitter.emit_instruction("CLC")

        if isinstance(right_operand, Immediate):
            value = right_operand.value & 0xFF if not is_u16 else right_operand.value
            self.emitter.emit_instruction("ADC", f"#${value:02X}")
        else:
            right_loc = self._get_operand_location(right_operand)
            self.emitter.emit_instruction("ADC", self._format_operand(right_loc))

    def _emit_sub(self, right_operand, is_u16: bool):
        """Emit subtraction operation."""
        self.emitter.emit_instruction("SEC")

        if isinstance(right_operand, Immediate):
            value = right_operand.value & 0xFF if not is_u16 else right_operand.value
            self.emitter.emit_instruction("SBC", f"#${value:02X}")
        else:
            right_loc = self._get_operand_location(right_operand)
            self.emitter.emit_instruction("SBC", self._format_operand(right_loc))

    def _emit_and(self, right_operand, is_u16: bool):
        """Emit bitwise AND operation."""
        if isinstance(right_operand, Immediate):
            value = right_operand.value & 0xFF if not is_u16 else right_operand.value
            self.emitter.emit_instruction("AND", f"#${value:02X}")
        else:
            right_loc = self._get_operand_location(right_operand)
            self.emitter.emit_instruction("AND", self._format_operand(right_loc))

    def _emit_or(self, right_operand, is_u16: bool):
        """Emit bitwise OR operation."""
        if isinstance(right_operand, Immediate):
            value = right_operand.value & 0xFF if not is_u16 else right_operand.value
            self.emitter.emit_instruction("ORA", f"#${value:02X}")
        else:
            right_loc = self._get_operand_location(right_operand)
            self.emitter.emit_instruction("ORA", self._format_operand(right_loc))

    def _emit_xor(self, right_operand, is_u16: bool):
        """Emit bitwise XOR operation."""
        if isinstance(right_operand, Immediate):
            value = right_operand.value & 0xFF if not is_u16 else right_operand.value
            self.emitter.emit_instruction("EOR", f"#${value:02X}")
        else:
            right_loc = self._get_operand_location(right_operand)
            self.emitter.emit_instruction("EOR", self._format_operand(right_loc))

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

        Args:
            instr: CondBranch instruction
        """
        # Load condition into A
        cond_loc = self._get_operand_location(instr.condition)
        self.emitter.emit_instruction("LDA", self._format_operand(cond_loc))

        # Compare based on comparison type
        if instr.comparison == '!=':
            # Branch if not equal to zero
            self.emitter.emit_instruction("BEQ", f"__L{instr.false_target}", "Branch if zero")
            self.emitter.emit_instruction("JMP", f"__L{instr.true_target}")
        elif instr.comparison == '==':
            # Branch if equal to zero
            self.emitter.emit_instruction("BNE", f"__L{instr.false_target}", "Branch if non-zero")
            self.emitter.emit_instruction("JMP", f"__L{instr.true_target}")
        else:
            # For other comparisons, we'd need to do actual comparison
            # For now, treat as != 0
            self.emitter.emit_instruction("BEQ", f"__L{instr.false_target}")
            self.emitter.emit_instruction("JMP", f"__L{instr.true_target}")

    def select_return(self, instr: Return):
        """
        Generate code for Return instruction.

        Args:
            instr: Return instruction
        """
        # TODO: Handle return values
        # For now, just emit RTS
        self.emitter.emit_instruction("RTS")

    # ========================================================================
    # Function Calls
    # ========================================================================

    def select_call(self, instr: Call):
        """
        Generate code for Call instruction.

        Args:
            instr: Call instruction
        """
        # TODO: Handle arguments properly
        # For now, simple call

        if instr.is_far:
            self.emitter.emit_instruction("JSL", instr.function)
        else:
            self.emitter.emit_instruction("JSR", instr.function)

        # TODO: Handle return values

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
            # Stack access - would need special handling
            raise Exception("Stack operands not yet implemented")
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
