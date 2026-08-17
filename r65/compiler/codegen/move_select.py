# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Move operation selector: Data movement between registers and memory.

Handles Move instruction generation including register transfers,
immediate loads, function pointers, and memory-to-memory moves.
"""

from r65.compiler.mir.nodes import Move, Immediate as MIRImmediate, FunctionPointer, LabelRef, HardwareRegister, VirtualRegister
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
    # Main Move Selection
    # ========================================================================

    def select_move(self, instr: Move):
        """
        Generate code for Move instruction.

        dest = source

        Args:
            instr: Move instruction
        """
        # Dead volatile load: emit BIT instead of LDA for side-effect reads
        # in interrupt handlers where A's value is never used.
        func = self.parent.current_function
        dead_volatile = getattr(func, 'dead_volatile_loads', None) if func else None
        if dead_volatile and id(instr) in dead_volatile:
            src_loc = self.parent._get_operand_location(instr.source)
            self.parent._ensure_m8_mode()
            self._emit_load_store('BIT', src_loc)
            return

        dest_loc = self.parent._get_operand_location(instr.dest)
        src_operand = instr.source
        is_u16 = self.parent._is_16bit(instr.type_info)

        # Check if this is a far pointer (3 bytes)
        is_far_ptr = isinstance(instr.type_info, PointerTypeInfo) and instr.type_info.is_far

        # SPECIAL CASE: Far self D=S path — prologue PHB+PHY already captured
        # the self pointer (Y addr + DBR bank) to the stack. The MIR
        # Move(dest=self_y_vreg, source=HW_Y) is a no-op.
        # Same for the FixedStack scratch path: prologue copies Y + caller DBR
        # into the zeropage scratch slot.
        func = self.parent.current_function
        if (func and (getattr(func, 'self_far_uses_d_equals_s', False)
                      or getattr(func, 'self_far_uses_scratch', False))
                and isinstance(src_operand, HardwareRegister) and src_operand.name == 'Y'
                and isinstance(instr.dest, VirtualRegister)
                and func.self_y_vreg and instr.dest.id == func.self_y_vreg.id):
            return  # No-op: prologue already placed self in its final home

        # SPECIAL CASE: Destination is hardware register
        if dest_loc.is_hw():
            self._move_to_hardware_register(instr, dest_loc, src_operand, is_u16)
            return

        # Handle function pointers
        if isinstance(src_operand, FunctionPointer):
            self._move_function_pointer(src_operand, dest_loc, instr.type_info)
            return

        # Handle label references (string literals, etc.)
        if isinstance(src_operand, LabelRef):
            if is_far_ptr:
                self._emit_far_function_pointer(src_operand.label_name, dest_loc)
            else:
                self._emit_near_function_pointer(src_operand.label_name, dest_loc)
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
            from r65.compiler.codegen.type_utils import get_vreg_size
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.is_hw():
                # A → X/Y: TAX/TAY transfers the FULL 16-bit accumulator
                # (B:A_low). If the source value is u8 (a freshly computed
                # BinaryOp result, a u8 scratch param coalesced to A, …),
                # B may hold stale bits and contaminate X/Y. Detect this
                # via the source operand's nominal size and emit an
                # explicit zero-extend before the transfer.
                if (src_loc.hw_register == 'A'
                        and dest_loc.hw_register in ('X', 'Y')
                        and isinstance(src_operand, VirtualRegister)
                        and get_vreg_size(src_operand) < 2):
                    self.parent._ensure_m16_mode()
                    self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF),
                                     "Zero-extend u8 in A to 16-bit X/Y")
                    if dest_loc.hw_register == 'X':
                        self._emit_instr(Opcode.TAX, comment="Transfer (zero-extended)")
                    else:
                        self._emit_instr(Opcode.TAY, comment="Transfer (zero-extended)")
                    return
                # X/Y ↔ A transfers are mode-sensitive: TXA/TYA/TAX/TAY in m8
                # mode only copy the low 8 bits. Ensure m16 for 16-bit values.
                if is_u16 and (
                    (src_loc.hw_register in ('X', 'Y') and dest_loc.hw_register == 'A') or
                    (src_loc.hw_register == 'A' and dest_loc.hw_register in ('X', 'Y'))
                ):
                    self.parent._ensure_m16_mode()
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
            self.parent._ensure_m16_mode()
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            self._emit_instr(Opcode.TCS, comment="Set stack pointer")
            self.parent._mark_a_modified()
            # Note: Do NOT switch back - mode will be restored when needed
        elif hw_register == 'D':
            # Set direct page register
            self.parent._ensure_m16_mode()
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            self._emit_instr(Opcode.TCD, comment="Set direct page")
            self.parent._mark_a_modified()
            # Note: Do NOT switch back - mode will be restored when needed
        else:
            raise InstructionSelectionError(f"Cannot load immediate into register {hw_register}", source_loc=self.parent._current_source_loc)

    def _load_memory_to_hw_register(self, hw_register: str, src_loc, is_u16: bool = False,
                                      persist_16bit_mode: bool = False):
        """Load from memory into a hardware register.

        Args:
            hw_register: Target register ('A', 'X', 'Y', 'B')
            src_loc: Source memory location
            is_u16: Whether this is a 16-bit load (DEST type)
            persist_16bit_mode: If True and loading to A, stay in m16 mode after load
        """
        # X and Y are always 16-bit registers in R65, so the Move's nominal
        # `is_u16` (derived from the dest type) is always True for X/Y dest —
        # which would route every load to a plain LDX/LDY in x16 mode and
        # read 2 bytes from the source. For a u8 source (e.g. an `@ A: u8`
        # scratch param) that picks up the adjacent byte as the high byte of
        # X/Y. Re-derive `is_u16` from the SOURCE size so the zero-extend
        # path runs whenever the source is 8 bits.
        if hw_register in ('X', 'Y'):
            is_u16 = (src_loc.size >= 2)

        # Handle 16-bit A register loads
        if hw_register == 'A' and is_u16:
            self.parent._ensure_m16_mode()
            self._emit_load_store('LDA', src_loc)
            # Note: Do NOT switch back - mode will be restored when needed
            return

        # Handle 8-bit A register loads - must ensure m8 mode
        if hw_register == 'A' and not is_u16:
            self.parent._ensure_m8_mode()
            self._emit_load_store('LDA', src_loc)
            return

        # Handle stack-relative addressing: LDX/LDY don't support sr,S mode
        # Must go through A with transfer
        if hw_register in ('X', 'Y') and src_loc.kind == LocationKind.STACK:
            # For 16-bit values, we need to be in 16-bit A mode for the load/transfer
            if is_u16:
                self.parent._ensure_m16_mode()
                self._emit_load_store('LDA', src_loc)
                if hw_register == 'X':
                    self._emit_instr(Opcode.TAX, comment="Transfer to X (no LDX sr,S)")
                else:
                    self._emit_instr(Opcode.TAY, comment="Transfer to Y (no LDY sr,S)")
                # Note: Do NOT switch back - mode will be restored when needed
            else:
                # 8-bit source to 16-bit X/Y: must zero-extend to avoid B register pollution
                # The B register (high byte of C) may contain garbage that would
                # corrupt X/Y if we just did TAX/TAY in 8-bit mode
                self.parent._ensure_m8_mode()
                self._emit_load_store('LDA', src_loc)
                # Switch to 16-bit A to access full C register and zero-extend
                self.parent._ensure_m16_mode()
                # Clear high byte (B register) to zero-extend
                self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Zero-extend to 16-bit")
                if hw_register == 'X':
                    self._emit_instr(Opcode.TAX, comment="Transfer to X (no LDX sr,S)")
                else:
                    self._emit_instr(Opcode.TAY, comment="Transfer to Y (no LDY sr,S)")
                # Note: Do NOT switch back - mode will be restored when needed
        elif hw_register in ('X', 'Y') and not is_u16:
            # u8 value from DP/memory to 16-bit X/Y: must zero-extend through A
            # LDX/LDY always read 16 bits in x16 mode, but the stored value is
            # only 1 byte — the adjacent byte may be garbage
            self.parent._ensure_m8_mode()
            self._emit_load_store('LDA', src_loc)
            self.parent._ensure_m16_mode()
            self._emit_instr(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Zero-extend to 16-bit")
            if hw_register == 'X':
                self._emit_instr(Opcode.TAX, comment="Transfer to X (zero-extended)")
            else:
                self._emit_instr(Opcode.TAY, comment="Transfer to Y (zero-extended)")
        elif hw_register in LOAD_MNEMONICS:
            self._emit_load_store(LOAD_MNEMONICS[hw_register], src_loc)
        elif hw_register == 'B':
            self._emit_load_store('LDA', src_loc)
            self._emit_instr(Opcode.XBA, comment="Load into B register")
        else:
            raise InstructionSelectionError(f"Cannot load into register {hw_register}", source_loc=self.parent._current_source_loc)

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
        self._emit_label_pointer_store(func_name, dest_loc, 3)

    def _emit_near_function_pointer(self, func_name: str, dest_loc):
        """Emit code to store a near function pointer (2 bytes)."""
        self._emit_label_pointer_store(func_name, dest_loc, 2)

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
            offset = getattr(src_operand, 'symbol_offset', 0) or 0
            self._emit_variable_address(symbol, dest_loc, is_u16, is_far_ptr, offset)

    def _emit_variable_address(self, symbol, dest_loc, is_u16: bool, is_far_ptr: bool = False, offset: int = 0):
        """Emit code to store a variable's address, with optional byte offset for array indexing."""
        alloc = self.parent.mem_alloc.get_allocation(symbol)
        has_rom_label = hasattr(symbol, 'rom_label') and symbol.rom_label
        # Extern statics have no allocation — their address lives in the .s
        # file and is reached via rom_label. Only require alloc for the
        # numeric-address fallback.
        if not alloc and not has_rom_label:
            raise InstructionSelectionError(f"No allocation for symbol: {symbol.name}", source_loc=self.parent._current_source_loc)

        # For ROM data, use the label name instead of numeric address
        # ROM data has a rom_label attribute set during MIR building
        if has_rom_label:
            addr_ref = f"{symbol.rom_label} + {offset}" if offset else symbol.rom_label
        else:
            addr_ref = f"${alloc.address + offset:04X}"

        if is_far_ptr:
            # 24-bit far pointer: 3 byte-by-byte stores in m8 mode
            # (m16 store + m8 bank store pattern is fragile with peephole)
            self.parent._ensure_m8_mode()
            # Low byte
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{addr_ref}"),
                            f"Load address of {symbol.name}")
            self._emit_load_store('STA', dest_loc)
            # High byte
            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{addr_ref}"))
            self._emit_load_store('STA', dest_high)
            # Bank byte
            dest_bank = self.parent._offset_location(dest_loc, 2)
            # Compute bank byte directly for numeric addresses (WLA-DX : operator
            # doesn't always work correctly for numeric constants)
            if hasattr(symbol, 'rom_label') and symbol.rom_label:
                bank_ref = f":{addr_ref}"
            else:
                bank_byte = (alloc.address + offset) >> 16
                bank_ref = f"${bank_byte:02X}"
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(bank_ref))
            self._emit_load_store('STA', dest_bank)
        elif is_u16:
            # 16-bit near pointer: single 16-bit load/store
            self.parent._ensure_m16_mode()
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(addr_ref),
                            f"Load address of {symbol.name}")
            self._emit_load_store('STA', dest_loc)
        else:
            # 8-bit address (low byte only)
            self.parent._ensure_m8_mode()
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{addr_ref}"),
                            f"Load address of {symbol.name}")
            self._emit_load_store('STA', dest_loc)

    # ========================================================================
    # Memory-to-Memory Move
    # ========================================================================

    def _move_from_location(self, instr: Move, dest_loc, src_operand, is_u16: bool, is_far_ptr: bool = False):
        """Handle moving from a source location to memory destination."""
        src_loc = self.parent._get_operand_location(src_operand)

        if src_loc.is_hw():
            # Computed here rather than in the caller: get_type_size raises on a
            # struct/trait type_info, and only the hardware-register path needs
            # the width (a struct never lives in a register).
            from r65.compiler.codegen.type_utils import get_type_size
            value_bytes = get_type_size(instr.type_info) if instr.type_info else 1
            self._emit_store_from_reg(src_loc.hw_register, dest_loc, is_u16,
                                      value_bytes)
        elif is_far_ptr:
            # 3-byte far pointer copy
            self._emit_far_pointer_mem_to_mem(src_loc, dest_loc)
        elif is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            # u8 LDA/STA needs m8 — a preceding u16 Move may have left
            # the emitter in m16, which would silently widen this copy
            # to 2 bytes and clobber the byte after dest.
            self.parent._ensure_m8_mode()
            self._emit_load_store('LDA', src_loc)
            self._emit_load_store('STA', dest_loc)

    def _emit_far_pointer_mem_to_mem(self, src_loc, dest_loc):
        """Copy a 3-byte far pointer from source to destination."""
        self._emit_pointer_mem_copy(src_loc, dest_loc, 3)
