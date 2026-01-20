"""
Move operation selector: Data movement between registers and memory.

Handles Move instruction generation including register transfers,
immediate loads, function pointers, and memory-to-memory moves.
"""

from r65.compiler.mir.nodes import Move, Immediate as MIRImmediate, FunctionPointer
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode, STORE_MNEMONICS, LOAD_MNEMONICS
from r65.compiler.codegen.asm_nodes import Immediate
from r65.compiler.codegen.base_selector import BaseSelector
from r65.compiler.hir.types import PointerTypeInfo


class MoveOperationSelector(BaseSelector):
    """
    Handles move operation instruction selection.

    Manages generation of data movement instructions including
    register-to-register transfers, immediate loads, and memory moves.
    """

    # ========================================================================
    # Emission Helpers
    # ========================================================================

    def _emit_load_store(self, mnemonic: str, location, comment: str = None):
        """Emit a load/store instruction using parent's opcode selection."""
        opcode, operand = self.parent._get_opcode_for_location(mnemonic, location)
        self._emit_instr(opcode, operand, comment)

    # ========================================================================
    # Main Move Selection
    # ========================================================================

    def select_move(self, instr: Move):
        """
        Generate code for Move instruction.

        dest = source

        Args:
            instr: Move instruction
        """
        dest_loc = self.parent._get_operand_location(instr.dest)
        src_operand = instr.source
        is_u16 = self.parent._is_16bit(instr.type_info)

        # Check if this is a far pointer (3 bytes)
        is_far_ptr = isinstance(instr.type_info, PointerTypeInfo) and instr.type_info.is_far

        # SPECIAL CASE: Destination is hardware register
        if dest_loc.kind == LocationKind.HARDWARE:
            self._move_to_hardware_register(instr, dest_loc, src_operand, is_u16)
            return

        # Handle function pointers
        if isinstance(src_operand, FunctionPointer):
            self._move_function_pointer(src_operand, dest_loc, instr.type_info)
            return

        # Handle immediate values (including symbolic addresses)
        if isinstance(src_operand, MIRImmediate):
            self._move_immediate(instr, dest_loc, src_operand, is_u16)
            return

        # Move from register/memory to memory
        self._move_from_location(instr, dest_loc, src_operand, is_u16, is_far_ptr)

    # ========================================================================
    # Move to Hardware Register
    # ========================================================================

    def _move_to_hardware_register(self, instr: Move, dest_loc, src_operand, is_u16: bool):
        """Handle moving data to a hardware register."""
        persist_mode = getattr(instr, 'persist_16bit_mode', False)
        if isinstance(src_operand, MIRImmediate):
            self._load_immediate_to_hw_register(dest_loc.hw_register, src_operand.value, is_u16, persist_mode)
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                self.parent._emit_register_transfer(src_loc.hw_register, dest_loc.hw_register)
            else:
                self._load_memory_to_hw_register(dest_loc.hw_register, src_loc, is_u16, persist_mode)

    def _load_immediate_to_hw_register(self, hw_register: str, value: int, is_u16: bool,
                                        persist_16bit_mode: bool = False):
        """Load an immediate value into a hardware register."""
        if hw_register in ['A', 'X', 'Y']:
            self.parent._emit_load_immediate_to_register(hw_register, value, is_u16, persist_16bit_mode)
        elif hw_register == 'B':
            value_masked = value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value_masked))
            self.parent._mark_a_modified()
            self.parent._store_to_b_from_a()
        elif hw_register == 'S':
            # Set stack pointer: TCS always transfers full 16-bit A
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20), "16-bit A for stack")
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            self._emit_instr(Opcode.TCS, comment="Set stack pointer")
            self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "Restore 8-bit A")
            self.parent._mark_a_modified()
        elif hw_register == 'D':
            # Set direct page register
            self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20), "16-bit A for direct page")
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            self._emit_instr(Opcode.TCD, comment="Set direct page")
            self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "Restore 8-bit A")
            self.parent._mark_a_modified()
        else:
            raise InstructionSelectionError(f"Cannot load immediate into register {hw_register}")

    def _load_memory_to_hw_register(self, hw_register: str, src_loc, is_u16: bool = False,
                                      persist_16bit_mode: bool = False):
        """Load from memory into a hardware register.

        Args:
            hw_register: Target register ('A', 'X', 'Y', 'B')
            src_loc: Source memory location
            is_u16: Whether this is a 16-bit load
            persist_16bit_mode: If True and loading to A, stay in m16 mode after load
        """
        # Handle 16-bit A register loads
        if hw_register == 'A' and is_u16:
            # Check if we're already in 16-bit mode
            already_in_16bit = self.parent.emitter.get_accu_mode() == 16
            if not already_in_16bit:
                self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20), "16-bit A")
                self.parent.emitter.emit_accu_mode(16)
            self._emit_load_store('LDA', src_loc)
            # Switch back to 8-bit unless persist mode
            if not persist_16bit_mode and not already_in_16bit:
                self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "8-bit A")
                self.parent.emitter.emit_accu_mode(8)
            return

        # Handle 8-bit A register loads - must switch to m8 if currently in m16
        if hw_register == 'A' and not is_u16:
            currently_in_16bit = self.parent.emitter.get_accu_mode() == 16
            if currently_in_16bit:
                self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "8-bit A")
                self.parent.emitter.emit_accu_mode(8)
            self._emit_load_store('LDA', src_loc)
            return

        # Handle stack-relative addressing: LDX/LDY don't support sr,S mode
        # Must go through A with transfer
        if hw_register in ('X', 'Y') and src_loc.kind == LocationKind.STACK:
            self._emit_load_store('LDA', src_loc)
            if hw_register == 'X':
                self._emit_instr(Opcode.TAX, comment="Transfer to X (no LDX sr,S)")
            else:
                self._emit_instr(Opcode.TAY, comment="Transfer to Y (no LDY sr,S)")
        elif hw_register in LOAD_MNEMONICS:
            self._emit_load_store(LOAD_MNEMONICS[hw_register], src_loc)
        elif hw_register == 'B':
            self._emit_load_store('LDA', src_loc)
            self._emit_instr(Opcode.XBA, comment="Load into B register")
        else:
            raise InstructionSelectionError(f"Cannot load into register {hw_register}")

    # ========================================================================
    # Function Pointer Handling
    # ========================================================================

    def _move_function_pointer(self, func_ptr: FunctionPointer, dest_loc, type_info):
        """Move a function pointer address to memory."""
        from r65.compiler.hir.types import FunctionTypeInfo

        func_name = func_ptr.function_name
        is_far_ptr = False
        if type_info and isinstance(type_info, FunctionTypeInfo):
            is_far_ptr = type_info.is_far

        if is_far_ptr:
            self._emit_far_function_pointer(func_name, dest_loc)
        else:
            self._emit_near_function_pointer(func_name, dest_loc)

    def _emit_far_function_pointer(self, func_name: str, dest_loc):
        """Emit code to store a far function pointer (3 bytes)."""
        # Low byte
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{func_name}"), "Load function address low byte")
        self._emit_load_store('STA', dest_loc)

        # High byte
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{func_name}"), "Load function address high byte")
        self._emit_load_store('STA', dest_high)

        # Bank byte
        dest_bank = self.parent._offset_location(dest_loc, 2)
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f":{func_name}"), "Load function bank byte")
        self._emit_load_store('STA', dest_bank)

    def _emit_near_function_pointer(self, func_name: str, dest_loc):
        """Emit code to store a near function pointer (2 bytes)."""
        # Low byte
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{func_name}"), "Load function address low byte")
        self._emit_load_store('STA', dest_loc)

        # High byte
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{func_name}"), "Load function address high byte")
        self._emit_load_store('STA', dest_high)

    # ========================================================================
    # Immediate Value Handling
    # ========================================================================

    def _move_immediate(self, instr: Move, dest_loc, src_operand: MIRImmediate, is_u16: bool):
        """Handle moving immediate values including symbolic addresses."""
        # Check for symbolic address (from address-of operator or function identifier)
        if hasattr(src_operand, 'symbol') and src_operand.symbol is not None:
            self._move_symbolic_address(instr, dest_loc, src_operand, is_u16)
            return

        value = src_operand.value
        if is_u16:
            self.parent._emit_16bit_immediate_store(value, dest_loc)
        else:
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value & 0xFF))
            self._emit_load_store('STA', dest_loc)

    def _move_symbolic_address(self, instr: Move, dest_loc, src_operand: MIRImmediate, is_u16: bool):
        """Handle moving a symbolic address (variable or function address)."""
        from r65.compiler.hir.symbol_table import SymbolKind
        from r65.compiler.hir.types import FunctionTypeInfo, PointerTypeInfo

        symbol = src_operand.symbol

        if symbol.kind == SymbolKind.FUNCTION:
            # Function pointer
            func_name = symbol.name
            is_far_ptr = False
            if instr.type_info and isinstance(instr.type_info, FunctionTypeInfo):
                is_far_ptr = instr.type_info.is_far

            if is_far_ptr:
                self._emit_far_function_pointer(func_name, dest_loc)
            else:
                self._emit_near_function_pointer(func_name, dest_loc)
        else:
            # Variable address - check if it's a far pointer
            is_far_ptr = False
            if instr.type_info and isinstance(instr.type_info, PointerTypeInfo):
                is_far_ptr = instr.type_info.is_far
            self._emit_variable_address(symbol, dest_loc, is_u16, is_far_ptr)

    def _emit_variable_address(self, symbol, dest_loc, is_u16: bool, is_far_ptr: bool = False):
        """Emit code to store a variable's address."""
        alloc = self.parent.mem_alloc.get_allocation(symbol)
        if not alloc:
            raise InstructionSelectionError(f"No allocation for symbol: {symbol.name}")

        # For ROM data, use the label name instead of numeric address
        # ROM data has a rom_label attribute set during MIR building
        if hasattr(symbol, 'rom_label') and symbol.rom_label:
            addr_ref = symbol.rom_label
        else:
            addr_ref = f"${alloc.address:04X}"

        if is_far_ptr:
            # 24-bit far pointer: low, high, bank
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{addr_ref}"),
                            f"Load address of {symbol.name}")
            self._emit_load_store('STA', dest_loc)
            # High byte
            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{addr_ref}"))
            self._emit_load_store('STA', dest_high)
            # Bank byte
            dest_bank = self.parent._offset_location(dest_loc, 2)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f":{addr_ref}"))
            self._emit_load_store('STA', dest_bank)
        elif is_u16:
            # 16-bit near pointer: low, high
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{addr_ref}"),
                            f"Load address of {symbol.name}")
            self._emit_load_store('STA', dest_loc)
            # High byte
            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{addr_ref}"))
            self._emit_load_store('STA', dest_high)
        else:
            # 8-bit address (low byte only)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{addr_ref}"),
                            f"Load address of {symbol.name}")
            self._emit_load_store('STA', dest_loc)

    # ========================================================================
    # Memory-to-Memory Move
    # ========================================================================

    def _move_from_location(self, instr: Move, dest_loc, src_operand, is_u16: bool, is_far_ptr: bool = False):
        """Handle moving from a source location to memory destination."""
        src_loc = self.parent._get_operand_location(src_operand)

        if src_loc.kind == LocationKind.HARDWARE:
            self._store_hw_register_to_memory(src_loc.hw_register, dest_loc, is_u16)
        elif is_far_ptr:
            # 3-byte far pointer copy
            self._emit_far_pointer_mem_to_mem(src_loc, dest_loc)
        elif is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            self._emit_load_store('LDA', src_loc)
            self._emit_load_store('STA', dest_loc)

    def _emit_far_pointer_mem_to_mem(self, src_loc, dest_loc):
        """Copy a 3-byte far pointer from source to destination."""
        # Low byte
        self._emit_load_store('LDA', src_loc)
        self._emit_load_store('STA', dest_loc)
        # High byte
        src_high = self.parent._offset_location(src_loc, 1)
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_load_store('LDA', src_high)
        self._emit_load_store('STA', dest_high)
        # Bank byte
        src_bank = self.parent._offset_location(src_loc, 2)
        dest_bank = self.parent._offset_location(dest_loc, 2)
        self._emit_load_store('LDA', src_bank)
        self._emit_load_store('STA', dest_bank)

    def _store_hw_register_to_memory(self, src_reg: str, dest_loc, is_u16: bool):
        """Store a hardware register to memory."""
        if src_reg in STORE_MNEMONICS:
            # Check for unsupported addressing modes for STX and STY
            need_transfer_to_a = False

            # STX and STY don't support stack-relative addressing
            if src_reg in ('X', 'Y') and dest_loc.kind == LocationKind.STACK:
                need_transfer_to_a = True

            # STX doesn't support X-indexed addressing (can't use STX addr,X)
            # STY doesn't support Y-indexed addressing (can't use STY addr,Y)
            if src_reg == 'X' and dest_loc.index_register == 'X':
                need_transfer_to_a = True
            if src_reg == 'Y' and dest_loc.index_register == 'Y':
                need_transfer_to_a = True

            if need_transfer_to_a:
                transfer_op = Opcode.TXA if src_reg == 'X' else Opcode.TYA
                self._emit_instr(transfer_op, comment=f"Transfer to A (no {STORE_MNEMONICS[src_reg]} with this addressing)")
                self._emit_load_store('STA', dest_loc)
                self.parent._mark_a_modified()
            else:
                self._emit_load_store(STORE_MNEMONICS[src_reg], dest_loc)
        else:
            raise InstructionSelectionError(
                f"Cannot move {'16-bit ' if is_u16 else ''}value from register {src_reg} to memory")
