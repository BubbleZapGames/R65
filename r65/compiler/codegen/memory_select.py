"""
Memory operation selector: Load, Store, LoadIndirect, StoreIndirect.

Handles memory access instruction generation including direct and indirect
addressing modes for the 65816 processor.
"""

from r65.compiler.mir.nodes import Load, Store, LoadIndirect, StoreIndirect, Immediate as MIRImmediate, FunctionPointer, LabelRef
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode, STORE_MNEMONICS
from r65.compiler.codegen.asm_nodes import Immediate, Address, StackOffset
from r65.compiler.codegen.base_selector import BaseSelector
from r65.compiler.hir.types import PointerTypeInfo

# Mode flag constants
M_FLAG = 0x20  # Accumulator size (0=16-bit, 1=8-bit)


class MemoryOperationSelector(BaseSelector):
    """
    Handles memory operation instruction selection.

    Manages generation of load/store instructions with proper
    addressing modes for direct and indirect memory access.
    """

    # ========================================================================
    # Emission Helpers
    # ========================================================================

    def _emit_load_store(self, mnemonic: str, location, comment: str = None):
        """Emit a load/store instruction using parent's emit_load/emit_store."""
        opcode, operand = self.parent._get_opcode_for_location(mnemonic, location)
        self._emit_instr(opcode, operand, comment)

    def _ensure_m16_mode(self):
        """Ensure 16-bit accumulator mode. Delegates to parent for mode tracking."""
        self.parent._ensure_m16_mode()

    def _ensure_m8_mode(self):
        """Ensure 8-bit accumulator mode. Delegates to parent for mode tracking."""
        self.parent._ensure_m8_mode()

    # ========================================================================
    # Y-Self Save/Restore for Trait Methods
    # ========================================================================

    def _has_y_self(self) -> bool:
        """Check if current function has self bound to Y register (trait method)."""
        func = self.parent.current_function
        return (func is not None and
                getattr(func, 'is_trait_method', False) and
                getattr(func, 'self_y_vreg', None) is not None)

    def _save_y_self(self) -> 'Address | None':
        """Save Y (self) to scratch DP before clobbering Y for indirect access.
        Returns the temp address used, or None if PHY was used instead."""
        temp_addr = self.parent._get_temp_address()
        if temp_addr:
            self._emit_instr(Opcode.STY_DP, temp_addr, "Save Y (self) for indirect access")
            return temp_addr
        else:
            # Fallback: push Y to stack (adjusts stack offsets by 2)
            self._emit_instr(Opcode.PHY, comment="Save Y (self) for indirect access")
            return None

    def _restore_y_self(self, temp_addr):
        """Restore Y (self) after indirect access."""
        if temp_addr is not None:
            self._emit_instr(Opcode.LDY_DP, temp_addr, "Restore Y (self)")
        else:
            self._emit_instr(Opcode.PLY, comment="Restore Y (self)")

    # ========================================================================
    # Direct Memory Operations
    # ========================================================================

    def select_load(self, instr: Load):
        """
        Generate code for Load instruction.

        Load dest = *source

        Args:
            instr: Load instruction
        """
        # Check if dest is return-sinkable: load deferred to return site
        from r65.compiler.mir.nodes import VirtualRegister
        if isinstance(instr.dest, VirtualRegister):
            dest_loc = self.parent._get_operand_location(instr.dest)
            if dest_loc.kind == LocationKind.RETURN_SINKABLE:
                return  # No-op: load deferred to return site
        else:
            dest_loc = self.parent._get_operand_location(instr.dest)

        src_loc = self.parent._get_operand_location(instr.source)

        is_u16 = self.parent._is_16bit(instr.type_info)

        if is_u16:
            # Check for hardware register destination - handle differently
            if dest_loc.kind == LocationKind.HARDWARE:
                if dest_loc.hw_register == 'A':
                    # Load 16-bit into A - need 16-bit mode
                    self._ensure_m16_mode()
                    self._emit_load_store('LDA', src_loc)
                    # Note: Do NOT switch back - mode will be restored when needed
                elif dest_loc.hw_register in ('X', 'Y'):
                    # Load into A then transfer to X/Y - need 16-bit A for the transfer
                    self._ensure_m16_mode()
                    self._emit_load_store('LDA', src_loc)
                    if dest_loc.hw_register == 'X':
                        self._emit_implied(Opcode.TAX, "Transfer to X")
                    else:
                        self._emit_implied(Opcode.TAY, "Transfer to Y")
                    # Note: Do NOT switch back - mode will be restored when needed
                else:
                    raise InstructionSelectionError(f"Cannot load 16-bit into hardware register {dest_loc.hw_register}", source_loc=self.parent._current_source_loc)
            else:
                self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            # 8-bit load - ensure we're in m8 mode for A register
            self._ensure_m8_mode()
            self._emit_load_store('LDA', src_loc)
            # Skip store if destination is already the accumulator
            if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
                pass  # Value already in A from LDA
            else:
                self._emit_load_store('STA', dest_loc)

    def select_store(self, instr: Store):
        """
        Generate code for Store instruction.

        *dest = source

        Args:
            instr: Store instruction
        """
        dest_loc = self.parent._get_operand_location(instr.dest)
        is_u16 = self.parent._is_16bit(instr.type_info)

        # Check if this is a far pointer (3 bytes)
        is_far_ptr = isinstance(instr.type_info, PointerTypeInfo) and instr.type_info.is_far

        # SPECIAL CASE: Storing to B register
        if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'B':
            self._store_to_b_register(instr, dest_loc)
            return

        # SPECIAL CASE: Handle immediate values
        if isinstance(instr.source, MIRImmediate):
            self._store_immediate(instr.source.value, dest_loc, is_u16)
            return

        # SPECIAL CASE: Handle function pointers
        if isinstance(instr.source, FunctionPointer):
            self._store_function_pointer(instr.source, dest_loc, instr.type_info)
            return

        # SPECIAL CASE: Handle label references (string literals, etc.)
        if isinstance(instr.source, LabelRef):
            self._store_label_ref(instr.source, dest_loc)
            return

        # Normal case: memory-to-memory or register-to-memory store
        self._store_from_location(instr, dest_loc, is_u16, is_far_ptr)

    def _store_to_b_register(self, instr: Store, dest_loc):
        """Handle storing to the B register (hidden accumulator high byte)."""
        if isinstance(instr.source, MIRImmediate):
            value = instr.source.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            self.parent._mark_a_modified()
            self.parent._store_to_b_from_a()
        else:
            src_loc = self.parent._get_operand_location(instr.source)
            if src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
                self.parent._store_to_b_from_a()
            else:
                self._emit_load_store('LDA', src_loc)
                self.parent._mark_a_modified()
                self.parent._store_to_b_from_a()

    def _store_immediate(self, value: int, dest_loc, is_u16: bool):
        """Store an immediate value to memory or hardware register."""
        # Special handling for hardware register A with 16-bit values
        if is_u16 and dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
            # Load 16-bit immediate into A - need 16-bit mode
            self._ensure_m16_mode()
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value & 0xFFFF))
            # Note: Do NOT switch back - mode will be restored when needed
            return

        if is_u16:
            self.parent._emit_16bit_immediate_store(value, dest_loc)
        else:
            # 8-bit store - ensure correct mode
            self._ensure_m8_mode()
            value_masked = value & 0xFF
            # Use STZ for storing zero (more efficient than LDA #0; STA)
            # But STZ doesn't support stack-relative or 24-bit long addressing
            can_use_stz = (
                value_masked == 0 and
                dest_loc.kind != LocationKind.STACK and
                # STZ only supports 16-bit absolute addresses
                not (dest_loc.kind == LocationKind.MEMORY and
                     dest_loc.memory_addr is not None and
                     dest_loc.memory_addr > 0xFFFF)
            )
            if can_use_stz:
                self._emit_load_store('STZ', dest_loc)
            else:
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value_masked))
                self._emit_load_store('STA', dest_loc)

    def _store_from_location(self, instr: Store, dest_loc, is_u16: bool, is_far_ptr: bool = False):
        """Store from a source location to destination."""
        src_loc = self.parent._get_operand_location(instr.source)

        if src_loc.kind == LocationKind.HARDWARE:
            self._store_from_hardware_register(src_loc, dest_loc, is_u16)
        elif is_far_ptr:
            # 3-byte far pointer copy
            self._store_far_pointer(src_loc, dest_loc)
        elif is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            # 8-bit store requires m8 mode
            self._ensure_m8_mode()
            self._emit_load_store('LDA', src_loc)
            self._emit_load_store('STA', dest_loc)

    def _store_far_pointer(self, src_loc, dest_loc):
        """Store a 3-byte far pointer from source to destination."""
        # Far pointer copies are byte-by-byte, need 8-bit mode
        self._ensure_m8_mode()
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

    def _store_from_hardware_register(self, src_loc, dest_loc, is_u16: bool):
        """Store from a hardware register to memory."""
        reg = src_loc.hw_register

        if reg == 'B':
            # B register: swap to A, store, swap back
            self.parent._access_b_value_in_a()
            self._emit_load_store('STA', dest_loc)
            self.parent._ensure_xba_state_normal("Restore A register")
        elif reg not in STORE_MNEMONICS:
            raise InstructionSelectionError(f"Cannot store from hardware register: {reg}", source_loc=self.parent._current_source_loc)
        else:
            # Check for unsupported addressing modes for STX and STY
            need_transfer_to_a = False

            # STX and STY don't support stack-relative addressing
            if reg in ('X', 'Y') and dest_loc.kind == LocationKind.STACK:
                need_transfer_to_a = True

            # STX doesn't support X-indexed addressing (can't use STX addr,X)
            # STY doesn't support Y-indexed addressing (can't use STY addr,Y)
            if reg == 'X' and dest_loc.index_register == 'X':
                need_transfer_to_a = True
            if reg == 'Y' and dest_loc.index_register == 'Y':
                need_transfer_to_a = True

            if need_transfer_to_a:
                transfer_op = Opcode.TXA if reg == 'X' else Opcode.TYA
                self._emit_instr(transfer_op, comment=f"Transfer to A (no {STORE_MNEMONICS[reg]} with this addressing)")
                self._emit_load_store('STA', dest_loc)
                self.parent._mark_a_modified()
            elif reg == 'A' and is_u16:
                # 16-bit store from A register needs 16-bit mode
                self._ensure_m16_mode()
                self._emit_load_store('STA', dest_loc)
                # Note: Do NOT switch back - mode will be restored when needed
            else:
                self._emit_load_store(STORE_MNEMONICS[reg], dest_loc)

    def _store_function_pointer(self, func_ptr: FunctionPointer, dest_loc, type_info):
        """Store a function pointer address directly to memory."""
        from r65.compiler.hir.types import FunctionTypeInfo

        func_name = func_ptr.function_name
        is_far = False
        if type_info and isinstance(type_info, FunctionTypeInfo):
            is_far = type_info.is_far

        if is_far:
            # Far pointer: 3 bytes (low, high, bank)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{func_name}"), "Load function address low byte")
            self._emit_load_store('STA', dest_loc)

            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{func_name}"), "Load function address high byte")
            self._emit_load_store('STA', dest_high)

            dest_bank = self.parent._offset_location(dest_loc, 2)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f":{func_name}"), "Load function bank byte")
            self._emit_load_store('STA', dest_bank)
        else:
            # Near pointer: 2 bytes (low, high)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{func_name}"), "Load function address low byte")
            self._emit_load_store('STA', dest_loc)

            dest_high = self.parent._offset_location(dest_loc, 1)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{func_name}"), "Load function address high byte")
            self._emit_load_store('STA', dest_high)

    def _store_label_ref(self, label_ref: LabelRef, dest_loc):
        """Store a label reference address (near pointer) directly to memory."""
        label = label_ref.label_name
        # Near pointer: 2 bytes (low, high) — same pattern as near function pointers
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f"<{label}"), "Load label address low byte")
        self._emit_load_store('STA', dest_loc)

        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(f">{label}"), "Load label address high byte")
        self._emit_load_store('STA', dest_high)

    # ========================================================================
    # Indirect Memory Operations
    # ========================================================================

    def select_load_indirect(self, instr: LoadIndirect):
        """
        Generate code for LoadIndirect instruction.

        dest = *ptr  or  dest = *(ptr + offset)  (indirect addressing)

        For 65816:
        - Y-located pointers use LDA abs,Y (absolute indexed Y, uses DBR for bank)
        - near pointers use (zp) or (zp),Y addressing modes
        - far pointers use [zp] or [zp],Y addressing modes
        - Stack-located near pointers use (d,S),Y addressing mode
        - Stack-located far pointers are copied to temp DP location first
        - Field offset is loaded into Y for indexed indirect addressing

        Args:
            instr: LoadIndirect instruction
        """
        ptr_loc = self.parent._get_operand_location(instr.pointer)
        dest_loc = self.parent._get_operand_location(instr.dest)

        # Y-pointer optimization: pointer is in Y register (trait method self)
        # Use LDA $offset,Y — effective address is DBR:(offset + Y)
        if ptr_loc.kind == LocationKind.HARDWARE and ptr_loc.hw_register == 'Y':
            offset = getattr(instr, 'offset', 0)
            self._emit_y_pointer_load(offset, instr.type_info, dest_loc)
            return

        # Spill hardware register pointers to scratch for indirect addressing
        if ptr_loc.kind == LocationKind.HARDWARE:
            ptr_loc = self._spill_pointer_to_scratch(ptr_loc)

        self._validate_pointer_location(ptr_loc)

        # Handle field offset - load into Y for indexed indirect
        # For stack indirect addressing, Y is always required (d,S),Y mode
        offset = getattr(instr, 'offset', 0)
        needs_y_for_stack = ptr_loc.kind == LocationKind.STACK and instr.index_register is None

        # In trait methods, Y holds self. Save it before clobbering for indirect access.
        y_self_save_addr = None
        will_clobber_y = (not instr.index_register and
                          (offset > 0 or needs_y_for_stack))
        if will_clobber_y and self._has_y_self():
            y_self_save_addr = self._save_y_self()
            # If PHY was used (no scratch), adjust stack offsets by 2
            if y_self_save_addr is None:
                from r65.compiler.codegen.register_alloc import PhysicalLocation
                if ptr_loc.kind == LocationKind.STACK:
                    ptr_loc = PhysicalLocation(
                        kind=LocationKind.STACK,
                        stack_offset=ptr_loc.stack_offset + 2,
                        size=ptr_loc.size
                    )
                if dest_loc.kind == LocationKind.STACK:
                    dest_loc = PhysicalLocation(
                        kind=LocationKind.STACK,
                        stack_offset=dest_loc.stack_offset + 2,
                        size=dest_loc.size
                    )

        if instr.index_register:
            # Index register already set by MIR lowerer (e.g., for ptr[i] access)
            # The index value is already in the register, don't overwrite it
            instr_index = instr.index_register
        elif offset > 0:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(offset), f"Load field offset {offset}")
            instr_index = 'Y'
        elif needs_y_for_stack:
            # Stack indirect requires Y with offset 0
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Load field offset 0")
            instr_index = 'Y'
        else:
            instr_index = None

        is_u16 = self.parent._is_16bit(instr.type_info)

        # Handle 16-bit indirect loads with byte-by-byte operations
        if is_u16:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    current_func = self.parent.current_function
                    if current_func and current_func.has_far_ptr_stack_params:
                        self._emit_16bit_far_ptr_load_via_d_equals_s(ptr_loc, dest_loc, instr_index)
                    else:
                        self._emit_16bit_far_ptr_load_via_dbr(ptr_loc, dest_loc, instr_index)
                else:
                    self._emit_16bit_stack_indirect_load(ptr_loc, dest_loc, instr_index)
            else:
                self._emit_16bit_indirect_load(ptr_loc, dest_loc, instr.is_far, instr_index)
            if will_clobber_y and self._has_y_self():
                self._restore_y_self(y_self_save_addr)
            return

        # Handle 8-bit indirect loads
        self._ensure_m8_mode()

        # Handle stack-located pointers
        if ptr_loc.kind == LocationKind.STACK:
            if instr.is_far:
                current_func = self.parent.current_function
                if current_func and current_func.has_far_ptr_stack_params:
                    self._emit_far_ptr_access_via_d_equals_s('LDA', ptr_loc, instr_index)
                else:
                    self._emit_far_ptr_access_via_dbr('LDA', ptr_loc, instr_index)
                self._emit_load_store('STA', dest_loc)
                if will_clobber_y and self._has_y_self():
                    self._restore_y_self(y_self_save_addr)
                return
            else:
                opcode, operand = self._get_stack_indirect_opcode(
                    'LDA', ptr_loc, instr.is_far, instr_index
                )
        else:
            opcode, operand = self._get_indirect_opcode(
                'LDA', ptr_loc, instr.is_far, instr_index
            )

        self._emit_instr(opcode, operand, "Load through pointer")
        self._emit_load_store('STA', dest_loc)
        if will_clobber_y and self._has_y_self():
            self._restore_y_self(y_self_save_addr)

    def select_store_indirect(self, instr: StoreIndirect):
        """
        Generate code for StoreIndirect instruction.

        *ptr = source  or  *(ptr + offset) = source  (indirect addressing)

        For 65816:
        - Y-located pointers use STA abs,Y (absolute indexed Y, uses DBR for bank)
        - near pointers use (zp) or (zp),Y addressing modes
        - far pointers use [zp] or [zp],Y addressing modes
        - Field offset is loaded into Y for indexed indirect addressing

        Args:
            instr: StoreIndirect instruction
        """
        ptr_loc = self.parent._get_operand_location(instr.pointer)
        src_loc = self.parent._get_operand_location(instr.source)

        # Y-pointer optimization: pointer is in Y register (trait method self)
        # Use STA $offset,Y — effective address is DBR:(offset + Y)
        if ptr_loc.kind == LocationKind.HARDWARE and ptr_loc.hw_register == 'Y':
            offset = getattr(instr, 'offset', 0)
            self._emit_y_pointer_store(offset, instr.type_info, src_loc, instr.source)
            return

        # Spill hardware register pointers to scratch for indirect addressing
        if ptr_loc.kind == LocationKind.HARDWARE:
            ptr_loc = self._spill_pointer_to_scratch(ptr_loc)

        self._validate_pointer_location(ptr_loc)

        # Handle field offset - load into Y for indexed indirect
        # For stack indirect addressing, Y is always required (d,S),Y mode
        offset = getattr(instr, 'offset', 0)
        needs_y_for_stack = ptr_loc.kind == LocationKind.STACK and instr.index_register is None

        # In trait methods, Y holds self. Save it before clobbering for indirect access.
        y_self_save_addr_s = None
        will_clobber_y_s = (not instr.index_register and
                            (offset > 0 or needs_y_for_stack))
        if will_clobber_y_s and self._has_y_self():
            y_self_save_addr_s = self._save_y_self()
            # If PHY was used (no scratch), adjust stack offsets by 2
            if y_self_save_addr_s is None:
                from r65.compiler.codegen.register_alloc import PhysicalLocation
                if ptr_loc.kind == LocationKind.STACK:
                    ptr_loc = PhysicalLocation(
                        kind=LocationKind.STACK,
                        stack_offset=ptr_loc.stack_offset + 2,
                        size=ptr_loc.size
                    )
                if src_loc.kind == LocationKind.STACK:
                    src_loc = PhysicalLocation(
                        kind=LocationKind.STACK,
                        stack_offset=src_loc.stack_offset + 2,
                        size=src_loc.size
                    )

        if instr.index_register:
            # Index register already set by MIR lowerer (e.g., for ptr[i] access)
            # The index value is already in the register, don't overwrite it
            instr_index = instr.index_register
        elif offset > 0:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(offset), f"Load field offset {offset}")
            instr_index = 'Y'
        elif needs_y_for_stack:
            # Stack indirect requires Y with offset 0
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Load field offset 0")
            instr_index = 'Y'
        else:
            instr_index = None

        is_u16 = self.parent._is_16bit(instr.type_info)

        # Handle 16-bit indirect stores with byte-by-byte operations
        if is_u16:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    current_func = self.parent.current_function
                    if current_func and current_func.has_far_ptr_stack_params:
                        self._emit_16bit_far_ptr_store_via_d_equals_s(ptr_loc, src_loc, instr.source, instr_index)
                    else:
                        self._emit_16bit_far_ptr_store_via_dbr(ptr_loc, src_loc, instr.source, instr_index)
                else:
                    self._emit_16bit_stack_indirect_store(ptr_loc, src_loc, instr.source, instr_index)
            else:
                self._emit_16bit_indirect_store(ptr_loc, src_loc, instr.source, instr.is_far, instr_index)
            if will_clobber_y_s and self._has_y_self():
                self._restore_y_self(y_self_save_addr_s)
            return

        # Handle 8-bit indirect stores
        self._ensure_m8_mode()

        # Handle stack-located pointers
        if ptr_loc.kind == LocationKind.STACK:
            # Load source value into A first
            if isinstance(instr.source, MIRImmediate):
                value = instr.source.value & 0xFF
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
                pass  # Already in A
            else:
                self._emit_load_store('LDA', src_loc)

            if instr.is_far:
                # Far pointer on stack: choose approach based on prologue setup
                current_func = self.parent.current_function
                if current_func and current_func.has_far_ptr_stack_params:
                    # D = S is set up: use efficient [dp],Y addressing
                    self._emit_far_ptr_access_via_d_equals_s('STA', ptr_loc, instr_index)
                else:
                    # Fallback: use DBR manipulation
                    self._emit_far_ptr_access_via_dbr('STA', ptr_loc, instr_index)
            else:
                # Near pointer on stack: use (d,S),Y addressing
                opcode, operand = self._get_stack_indirect_opcode(
                    'STA', ptr_loc, instr.is_far, instr_index
                )
                self._emit_instr(opcode, operand, "Store through stack pointer")
            if will_clobber_y_s and self._has_y_self():
                self._restore_y_self(y_self_save_addr_s)
            return

        opcode, operand = self._get_indirect_opcode('STA', ptr_loc, instr.is_far, instr_index)

        # Load source value into A
        if isinstance(instr.source, MIRImmediate):
            value = instr.source.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
        elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            pass  # Already in A
        else:
            self._emit_load_store('LDA', src_loc)

        self._emit_instr(opcode, operand, "Store through pointer")
        if will_clobber_y_s and self._has_y_self():
            self._restore_y_self(y_self_save_addr_s)

    # ========================================================================
    # Y-Pointer (Trait Self) Addressing
    # ========================================================================

    def _emit_y_pointer_load(self, offset, type_info, dest_loc):
        """Load through Y-pointer using LDA abs,Y addressing.

        Effective address = DBR:(offset + Y). Used for trait method self field access
        where Y holds the struct base address and DBR is set to the object's bank.

        Args:
            offset: Field byte offset from struct base
            type_info: Type of the value being loaded
            dest_loc: Destination physical location
        """
        is_u16 = self.parent._is_16bit(type_info)

        if is_u16:
            # 16-bit load: switch to m16, LDA abs,Y, store, switch back
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for u16 field load")
            self.parent.emitter.emit_accu_mode(16)
            self.parent.emitter.emit_raw(f"    LDA ${offset:04X},Y")
            # Store 16-bit value
            if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
                pass  # Already in A
            else:
                self._emit_load_store('STA', dest_loc)
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
        else:
            # 8-bit load: LDA abs,Y directly
            self.parent.emitter.emit_raw(f"    LDA ${offset:04X},Y")
            if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'A':
                pass  # Already in A
            else:
                self._emit_load_store('STA', dest_loc)

    def _emit_y_pointer_store(self, offset, type_info, src_loc, source):
        """Store through Y-pointer using STA abs,Y addressing.

        Effective address = DBR:(offset + Y). Used for trait method self field writes
        where Y holds the struct base address and DBR is set to the object's bank.

        Args:
            offset: Field byte offset from struct base
            type_info: Type of the value being stored
            src_loc: Source physical location
            source: MIR source operand (for immediate detection)
        """

        is_u16 = self.parent._is_16bit(type_info)

        if is_u16:
            # 16-bit store
            self._emit_immediate(Opcode.REP_IMMEDIATE, M_FLAG, "16-bit A for u16 field store")
            self.parent.emitter.emit_accu_mode(16)
            # Load source into A
            if isinstance(source, MIRImmediate):
                value = source.value & 0xFFFF
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
                pass  # Already in A
            else:
                self._emit_load_store('LDA', src_loc)
            self.parent.emitter.emit_raw(f"    STA ${offset:04X},Y")
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
        else:
            # 8-bit store: load into A, STA abs,Y
            if isinstance(source, MIRImmediate):
                value = source.value & 0xFF
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
            elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
                pass  # Already in A
            else:
                self._emit_load_store('LDA', src_loc)
            self.parent.emitter.emit_raw(f"    STA ${offset:04X},Y")

    # ========================================================================

    def _validate_pointer_location(self, ptr_loc):
        """Validate that pointer is in memory (not immediate or hardware register).

        Note: Hardware register pointers should be spilled to scratch before indirect access.
        This method just checks the location is valid for indirect addressing.
        """
        if ptr_loc.kind == LocationKind.IMMEDIATE:
            raise InstructionSelectionError(
                f"Pointer for indirect addressing must be in memory, got immediate value",
                source_loc=self.parent._current_source_loc)
        if ptr_loc.kind == LocationKind.HARDWARE:
            raise InstructionSelectionError(
                f"Pointer for indirect addressing must be in memory, got: {ptr_loc.hw_register}",
                source_loc=self.parent._current_source_loc)

    def _spill_pointer_to_scratch(self, ptr_loc) -> 'PhysicalLocation':
        """Spill a hardware register pointer to a scratch location for indirect addressing.

        Args:
            ptr_loc: Physical location of pointer (must be hardware register)

        Returns:
            New PhysicalLocation with pointer in scratch memory
        """
        from r65.compiler.codegen.register_alloc import PhysicalLocation

        if ptr_loc.kind != LocationKind.HARDWARE:
            return ptr_loc

        # Find a scratch register with at least 2 bytes
        # Even if it's "in use", we can temporarily reuse it since this is
        # just a temporary spill within a single instruction sequence
        scratch_addr = None
        if hasattr(self.parent.reg_alloc, 'scratch_pool') and self.parent.reg_alloc.scratch_pool:
            for scratch in self.parent.reg_alloc.scratch_pool.scratches:
                if scratch.size >= 2:
                    scratch_addr = scratch.address
                    break

        if scratch_addr is None:
            raise InstructionSelectionError(
                f"No scratch register available for pointer spill. "
                f"Define a 2-byte scratch register using: #[zeropage(addr, register)] static mut SCRATCH: u16;",
                source_loc=self.parent._current_source_loc
            )

        scratch_loc = PhysicalLocation(
            kind=LocationKind.SCRATCH,
            scratch_addr=scratch_addr,
            size=2
        )

        # Spill the pointer from hardware register to scratch
        if ptr_loc.hw_register == 'X':
            self._emit_instr(Opcode.STX_DP, Address(scratch_addr), "Spill X pointer to scratch")
        elif ptr_loc.hw_register == 'Y':
            self._emit_instr(Opcode.STY_DP, Address(scratch_addr), "Spill Y pointer to scratch")
        elif ptr_loc.hw_register == 'A':
            self._emit_instr(Opcode.STA_DP, Address(scratch_addr), "Spill A pointer to scratch")
        else:
            raise InstructionSelectionError(f"Cannot spill pointer from hardware register {ptr_loc.hw_register}", source_loc=self.parent._current_source_loc)

        return scratch_loc

    def _get_indirect_opcode(self, mnemonic: str, ptr_loc, is_far: bool, index_register: str = None):
        """
        Get opcode and operand for indirect addressing mode.

        Args:
            mnemonic: Base mnemonic ('LDA' or 'STA')
            ptr_loc: Physical location of pointer
            is_far: True for far pointer (24-bit)
            index_register: Optional index register ('Y' for (zp),Y)

        Returns:
            Tuple of (Opcode, operand)
        """
        # Get the address value from the pointer location
        if ptr_loc.kind == LocationKind.SCRATCH:
            addr_value = ptr_loc.scratch_addr
        elif ptr_loc.kind == LocationKind.MEMORY:
            addr_value = ptr_loc.memory_addr
        else:
            raise InstructionSelectionError(f"Invalid pointer location for indirect addressing: {ptr_loc}", source_loc=self.parent._current_source_loc)

        operand = Address(addr_value)

        # Select the appropriate opcode based on indirect mode
        if mnemonic == 'LDA':
            if is_far:
                if index_register == 'Y':
                    return Opcode.LDA_DP_INDIRECT_LONG_Y, operand
                return Opcode.LDA_DP_INDIRECT_LONG, operand
            else:
                if index_register == 'Y':
                    return Opcode.LDA_DP_INDIRECT_Y, operand
                return Opcode.LDA_DP_INDIRECT, operand
        elif mnemonic == 'STA':
            if is_far:
                if index_register == 'Y':
                    return Opcode.STA_DP_INDIRECT_LONG_Y, operand
                return Opcode.STA_DP_INDIRECT_LONG, operand
            else:
                if index_register == 'Y':
                    return Opcode.STA_DP_INDIRECT_Y, operand
                return Opcode.STA_DP_INDIRECT, operand
        else:
            raise InstructionSelectionError(f"Indirect addressing not supported for: {mnemonic}", source_loc=self.parent._current_source_loc)

    def _emit_far_ptr_access_via_dbr(self, mnemonic: str, ptr_loc, index_register: str = None):
        """
        Access memory through a far pointer using DBR manipulation.

        Primary approach for far pointers on the stack:
        1. PHB - save current DBR
        2. Load bank byte from pointer, PHA, PLB - set DBR to pointer's bank
        3. Use (d,S),Y stack-relative indirect (DBR provides the bank)
        4. PLB - restore original DBR

        This is aggressive about saving/restoring DBR to avoid affecting
        other code that depends on the current data bank setting.

        Args:
            mnemonic: Base mnemonic ('LDA' or 'STA')
            ptr_loc: Physical location of pointer (on stack)
            index_register: Optional index register ('Y' for (d,S),Y)

        Returns:
            Tuple of (Opcode, operand)
        """
        if index_register and index_register != 'Y':
            raise InstructionSelectionError(
                f"Far pointer indirect only supports Y index register, got: {index_register}",
                source_loc=self.parent._current_source_loc
            )

        # Stack offset for the pointer (low byte location)
        stack_offset = ptr_loc.stack_offset

        # Save current DBR
        self._emit_instr(Opcode.PHB, None, "Save DBR")

        # Load bank byte from far pointer (3rd byte, at base offset+2)
        # Account for PHB pushing 1 byte onto stack (+1 to all stack offsets)
        self._emit_instr(Opcode.LDA_STACK, StackOffset(stack_offset + 2 + 1), "Load ptr bank")
        self._emit_instr(Opcode.PHA, None, "Push bank")
        self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

        # Now (d,S),Y will use DBR for the bank byte
        # Stack offset is +1 from original due to PHB (PHA/PLB cancel out)
        adjusted_offset = stack_offset + 1
        operand = StackOffset(adjusted_offset)

        # The (d,S),Y addressing mode requires Y index register
        # If no index provided, set Y to 0
        if not index_register:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect access")
            index_register = 'Y'

        # Select the appropriate opcode
        if mnemonic == 'LDA':
            opcode = Opcode.LDA_STACK_INDIRECT_Y
        elif mnemonic == 'STA':
            opcode = Opcode.STA_STACK_INDIRECT_Y
        else:
            raise InstructionSelectionError(f"Indirect addressing not supported for: {mnemonic}", source_loc=self.parent._current_source_loc)

        # Emit the actual memory access
        self._emit_instr(opcode, operand, f"{mnemonic} through far pointer")

        # Restore original DBR immediately after the access
        self._emit_instr(Opcode.PLB, None, "Restore DBR")

        # Return None to indicate we've already emitted the instruction
        return None, None

    def _emit_far_ptr_access_via_d_equals_s(self, mnemonic: str, ptr_loc, index_register: str = None):
        """
        Access memory through a far pointer using [dp],Y when D = S.

        When the function prologue has set D = S (for far pointer stack params),
        stack offsets become direct page offsets. This allows using the efficient
        [dp],Y indirect long addressing mode directly.

        This is much more efficient than the DBR manipulation approach because:
        - No PHB/PLB overhead
        - No bank byte loading
        - Just a single [dp],Y instruction

        Args:
            mnemonic: Base mnemonic ('LDA' or 'STA')
            ptr_loc: Physical location of pointer (on stack)
            index_register: Optional index register ('Y' for [dp],Y)
        """
        if index_register and index_register != 'Y':
            raise InstructionSelectionError(
                f"Far pointer indirect only supports Y index register, got: {index_register}",
                source_loc=self.parent._current_source_loc
            )

        # When D = S, the stack offset is the DP offset
        stack_offset = ptr_loc.stack_offset

        # Select the appropriate opcode
        if mnemonic == 'LDA':
            if index_register == 'Y':
                opcode = Opcode.LDA_DP_INDIRECT_LONG_Y
            else:
                opcode = Opcode.LDA_DP_INDIRECT_LONG
        elif mnemonic == 'STA':
            if index_register == 'Y':
                opcode = Opcode.STA_DP_INDIRECT_LONG_Y
            else:
                opcode = Opcode.STA_DP_INDIRECT_LONG
        else:
            raise InstructionSelectionError(f"Indirect addressing not supported for: {mnemonic}", source_loc=self.parent._current_source_loc)

        # Emit the memory access using DP indirect long
        # The operand is the stack offset, which is now a DP offset since D = S
        operand = Address(stack_offset)
        self._emit_instr(opcode, operand, f"{mnemonic} through far pointer [dp],Y")

        return None, None

    # ========================================================================
    # 16-bit Indirect Memory Operations
    # ========================================================================

    def _emit_16bit_indirect_load(self, ptr_loc, dest_loc, is_far: bool, index_register: str = None):
        """
        Emit 16-bit indirect load (two 8-bit operations).

        For 16-bit values, we perform two 8-bit loads at consecutive addresses.
        Must be in m8 mode since indirect addressing is affected by the M flag.

        Strategy: Use Y register to access high byte at offset +1
        - LDY #0 (if not already indexed)
        - LDA (dp),Y  ; load low byte
        - STA dest
        - INY
        - LDA (dp),Y  ; load high byte
        - STA dest+1

        Args:
            ptr_loc: Physical location of pointer
            dest_loc: Destination location for loaded value
            is_far: True for far pointer (24-bit)
            index_register: Optional index register (only 'Y' supported)
        """
        # Byte-by-byte operations require 8-bit accumulator mode
        self._ensure_m8_mode()

        # Get opcode for indirect access - always use Y-indexed for 16-bit
        opcode_load, operand = self._get_indirect_opcode('LDA', ptr_loc, is_far, 'Y')

        if index_register == 'Y':
            # Caller set up Y - we use it as-is for low byte
            pass
        else:
            # Need to set Y=0 for base access
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load and store low byte
        self._emit_instr(opcode_load, operand, "Load low byte through pointer")
        self._emit_load_store('STA', dest_loc)

        # Increment Y for high byte
        self._emit_instr(Opcode.INY, None, "Increment for high byte")

        # Load and store high byte
        self._emit_instr(opcode_load, operand, "Load high byte through pointer")
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_load_store('STA', dest_high)

    def _emit_16bit_indirect_store(self, ptr_loc, src_loc, source, is_far: bool, index_register: str = None):
        """
        Emit 16-bit indirect store (two 8-bit operations).

        For 16-bit values, we perform two 8-bit stores at consecutive addresses.
        Must be in m8 mode since indirect addressing is affected by the M flag.

        Args:
            ptr_loc: Physical location of pointer
            src_loc: Source location of value to store (may be None for immediate)
            source: MIR source operand (for immediate value extraction)
            is_far: True for far pointer (24-bit)
            index_register: Optional index register (only 'Y' supported)
        """
        # Byte-by-byte operations require 8-bit accumulator mode
        self._ensure_m8_mode()

        # Get opcode for indirect access - always use Y-indexed for 16-bit
        opcode_store, operand = self._get_indirect_opcode('STA', ptr_loc, is_far, 'Y')

        if index_register == 'Y':
            # Caller set up Y
            pass
        else:
            # Need to set Y=0 for base access
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load and store low byte
        if isinstance(source, MIRImmediate):
            value = source.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value), "Load low byte immediate")
        elif src_loc and src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            # Source is already in A register — use XBA to access both bytes
            # In m8 mode, A holds low byte, B (hidden high byte) holds high byte
            self._ensure_m8_mode()
            self._emit_instr(opcode_store, operand, "Store low byte (already in A)")
            self._emit_instr(Opcode.INY, None, "Increment for high byte")
            self._emit_instr(Opcode.XBA, None, "Swap to get high byte")
            self._emit_instr(opcode_store, operand, "Store high byte through pointer")
            self._emit_instr(Opcode.XBA, None, "Restore A low byte")
            return
        else:
            self._emit_load_store('LDA', src_loc)
        self._emit_instr(opcode_store, operand, "Store low byte through pointer")

        # Increment Y for high byte
        self._emit_instr(Opcode.INY, None, "Increment for high byte")

        # Load and store high byte
        if isinstance(source, MIRImmediate):
            value = (source.value >> 8) & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value), "Load high byte immediate")
        else:
            src_high = self.parent._offset_location(src_loc, 1)
            self._emit_load_store('LDA', src_high)
        self._emit_instr(opcode_store, operand, "Store high byte through pointer")

    def _emit_16bit_stack_indirect_load(self, ptr_loc, dest_loc, index_register: str = None):
        """
        Emit 16-bit indirect load through stack-relative pointer.

        Uses (d,S),Y addressing mode for near pointers on the stack.
        Must be in m8 mode since indirect addressing is affected by the M flag.

        Args:
            ptr_loc: Physical location of pointer (on stack)
            dest_loc: Destination location for loaded value
            index_register: Optional index register (only 'Y' supported)
        """
        # Byte-by-byte operations require 8-bit accumulator mode
        self._ensure_m8_mode()

        opcode_load, operand = self._get_stack_indirect_opcode('LDA', ptr_loc, False, 'Y')

        if index_register == 'Y':
            pass  # Caller set up Y
        else:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load and store low byte
        self._emit_instr(opcode_load, operand, "Load low byte through stack pointer")
        self._emit_load_store('STA', dest_loc)

        # Increment Y for high byte
        self._emit_instr(Opcode.INY, None, "Increment for high byte")

        # Load and store high byte
        self._emit_instr(opcode_load, operand, "Load high byte through stack pointer")
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_load_store('STA', dest_high)

    def _emit_16bit_stack_indirect_store(self, ptr_loc, src_loc, source, index_register: str = None):
        """
        Emit 16-bit indirect store through stack-relative pointer.

        Uses (d,S),Y addressing mode for near pointers on the stack.
        Must be in m8 mode since indirect addressing is affected by the M flag.

        Args:
            ptr_loc: Physical location of pointer (on stack)
            src_loc: Source location of value to store (may be None for immediate)
            source: MIR source operand (for immediate value extraction)
            index_register: Optional index register (only 'Y' supported)
        """
        # Byte-by-byte operations require 8-bit accumulator mode
        self._ensure_m8_mode()

        opcode_store, operand = self._get_stack_indirect_opcode('STA', ptr_loc, False, 'Y')

        if index_register == 'Y':
            pass  # Caller set up Y
        else:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load and store low byte
        if isinstance(source, MIRImmediate):
            value = source.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value), "Load low byte immediate")
        elif src_loc and src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            # Source is already in A register — use XBA to access both bytes
            # In m8 mode, A holds low byte, B (hidden high byte) holds high byte
            self._ensure_m8_mode()
            self._emit_instr(opcode_store, operand, "Store low byte (already in A)")
            self._emit_instr(Opcode.INY, None, "Increment for high byte")
            self._emit_instr(Opcode.XBA, None, "Swap to get high byte")
            self._emit_instr(opcode_store, operand, "Store high byte through stack pointer")
            self._emit_instr(Opcode.XBA, None, "Restore A low byte")
            return
        else:
            self._emit_load_store('LDA', src_loc)
        self._emit_instr(opcode_store, operand, "Store low byte through stack pointer")

        # Increment Y for high byte
        self._emit_instr(Opcode.INY, None, "Increment for high byte")

        # Load and store high byte
        if isinstance(source, MIRImmediate):
            value = (source.value >> 8) & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value), "Load high byte immediate")
        else:
            src_high = self.parent._offset_location(src_loc, 1)
            self._emit_load_store('LDA', src_high)
        self._emit_instr(opcode_store, operand, "Store high byte through stack pointer")

    def _emit_16bit_far_ptr_load_via_dbr(self, ptr_loc, dest_loc, index_register: str = None):
        """
        Emit 16-bit indirect load through far pointer on stack using DBR manipulation.
        Must be in m8 mode since indirect addressing is affected by the M flag.

        Stores each loaded byte directly to dest rather than using PHA/PLA,
        which avoids stack offset corruption and DBR mis-restore.
        Stack-relative STA and zeropage STA don't use DBR, so storing while
        DBR is set to the far pointer's bank is safe.

        Args:
            ptr_loc: Physical location of pointer (on stack)
            dest_loc: Destination location for loaded value
            index_register: Optional index register (only 'Y' supported)
        """
        from dataclasses import replace
        from r65.compiler.codegen.register_alloc import LocationKind

        # Byte-by-byte operations require 8-bit accumulator mode
        self._ensure_m8_mode()

        stack_offset = ptr_loc.stack_offset

        # Save current DBR
        self._emit_instr(Opcode.PHB, None, "Save DBR")

        # Load bank byte from far pointer (3rd byte, at base offset+2)
        # Account for PHB pushing 1 byte onto stack
        self._emit_instr(Opcode.LDA_STACK, StackOffset(stack_offset + 2 + 1), "Load ptr bank")
        self._emit_instr(Opcode.PHA, None, "Push bank")
        self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

        # Adjusted pointer offset due to PHB (+1 byte on stack)
        adjusted_offset = stack_offset + 1
        operand = StackOffset(adjusted_offset)

        # Adjust dest_loc for PHB stack shift if it's stack-relative
        if dest_loc.kind == LocationKind.STACK:
            adj_dest = replace(dest_loc, stack_offset=dest_loc.stack_offset + 1)
            adj_dest_high = self.parent._offset_location(adj_dest, 1)
        else:
            adj_dest = dest_loc
            adj_dest_high = self.parent._offset_location(dest_loc, 1)

        if index_register == 'Y':
            pass  # Caller set up Y
        else:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load low byte through far pointer and store directly to dest
        self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, operand, "Load low byte through far pointer")
        self._emit_load_store('STA', adj_dest)

        # Increment Y for high byte
        self._emit_instr(Opcode.INY, None, "Increment for high byte")

        # Load high byte through far pointer and store directly to dest+1
        self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, operand, "Load high byte through far pointer")
        self._emit_load_store('STA', adj_dest_high)

        # Restore DBR - PLB correctly pops the PHB-saved value
        self._emit_instr(Opcode.PLB, None, "Restore DBR")

    def _emit_16bit_far_ptr_store_via_dbr(self, ptr_loc, src_loc, source, index_register: str = None):
        """
        Emit 16-bit indirect store through far pointer on stack using DBR manipulation.
        Must be in m8 mode since indirect addressing is affected by the M flag.

        Args:
            ptr_loc: Physical location of pointer (on stack)
            src_loc: Source location of value to store (may be None for immediate)
            source: MIR source operand (for immediate value extraction)
            index_register: Optional index register (only 'Y' supported)
        """
        stack_offset = ptr_loc.stack_offset

        # Handle source in A register: save before DBR manipulation clobbers A
        if src_loc and src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            self._ensure_m8_mode()

            # Save current DBR
            self._emit_instr(Opcode.PHB, None, "Save DBR")

            # Load bank byte from far pointer
            # But first save A since we need it. Use XBA + PHA + XBA to save high byte,
            # then PHA to save low byte.
            # Actually simpler: push A (8-bit low byte) then handle high byte after
            self._emit_instr(Opcode.PHA, None, "Save A low byte")

            # Load bank byte (account for PHB+PHA = +2 on stack)
            self._emit_instr(Opcode.LDA_STACK, StackOffset(stack_offset + 2 + 2), "Load ptr bank")
            self._emit_instr(Opcode.PHA, None, "Push bank")
            self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

            # Adjusted offset: PHB (+1) + PHA (+1) = +2
            adjusted_offset = stack_offset + 2
            operand = StackOffset(adjusted_offset)

            if index_register == 'Y':
                pass
            else:
                self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

            # Restore and store low byte
            self._emit_instr(Opcode.PLA, None, "Restore A low byte")
            # Now account for PHB only (+1)
            adjusted_offset = stack_offset + 1
            operand = StackOffset(adjusted_offset)
            self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, operand, "Store low byte through far pointer")

            self._emit_instr(Opcode.INY, None, "Increment for high byte")
            self._emit_instr(Opcode.XBA, None, "Swap to get high byte")
            self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, operand, "Store high byte through far pointer")
            self._emit_instr(Opcode.XBA, None, "Restore A low byte")

            self._emit_instr(Opcode.PLB, None, "Restore DBR")
            return

        # Byte-by-byte operations require 8-bit accumulator mode
        self._ensure_m8_mode()

        # Save current DBR
        self._emit_instr(Opcode.PHB, None, "Save DBR")

        # Load bank byte from far pointer (3rd byte, at base offset+2)
        # Account for PHB pushing 1 byte onto stack
        self._emit_instr(Opcode.LDA_STACK, StackOffset(stack_offset + 2 + 1), "Load ptr bank")
        self._emit_instr(Opcode.PHA, None, "Push bank")
        self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

        # Adjusted offset due to PHB
        adjusted_offset = stack_offset + 1
        operand = StackOffset(adjusted_offset)

        if index_register == 'Y':
            pass  # Caller set up Y
        else:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load and store low byte
        if isinstance(source, MIRImmediate):
            value = source.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value), "Load low byte immediate")
        else:
            self._emit_load_store('LDA', src_loc)
        self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, operand, "Store low byte through far pointer")

        # Increment Y for high byte
        self._emit_instr(Opcode.INY, None, "Increment for high byte")

        # Load and store high byte
        if isinstance(source, MIRImmediate):
            value = (source.value >> 8) & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value), "Load high byte immediate")
        else:
            src_high = self.parent._offset_location(src_loc, 1)
            self._emit_load_store('LDA', src_high)
        self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, operand, "Store high byte through far pointer")

        # Restore DBR
        self._emit_instr(Opcode.PLB, None, "Restore DBR")

    def _emit_16bit_far_ptr_load_via_d_equals_s(self, ptr_loc, dest_loc, index_register: str = None):
        """
        Emit 16-bit indirect load through far pointer using [dp],Y when D = S.

        Uses m16 mode for a single 16-bit load instruction, which is both more
        efficient and avoids clobbering A with intermediate byte-by-byte operations.

        Args:
            ptr_loc: Physical location of pointer (on stack)
            dest_loc: Destination location for loaded value
            index_register: Optional index register (only 'Y' supported)
        """
        # Use 16-bit accumulator for direct 16-bit load
        self._ensure_m16_mode()

        # When D = S, stack offset is DP offset
        stack_offset = ptr_loc.stack_offset
        operand = Address(stack_offset)

        if index_register == 'Y':
            pass  # Caller set up Y
        else:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load 16-bit value through far pointer (single m16 instruction)
        self._emit_instr(Opcode.LDA_DP_INDIRECT_LONG_Y, operand, "Load u16 through far pointer [dp],Y")
        self._emit_load_store('STA', dest_loc)

    def _emit_16bit_far_ptr_store_via_d_equals_s(self, ptr_loc, src_loc, source, index_register: str = None):
        """
        Emit 16-bit indirect store through far pointer using [dp],Y when D = S.

        Uses m16 mode for a single 16-bit store instruction, which is both more
        efficient and avoids byte-by-byte complexity.

        Args:
            ptr_loc: Physical location of pointer (on stack)
            src_loc: Source location of value to store (may be None for immediate)
            source: MIR source operand (for immediate value extraction)
            index_register: Optional index register (only 'Y' supported)
        """
        # Use 16-bit accumulator for direct 16-bit store
        self._ensure_m16_mode()

        # When D = S, stack offset is DP offset
        stack_offset = ptr_loc.stack_offset
        operand = Address(stack_offset)

        if index_register == 'Y':
            pass
        else:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        # Load source value into A (16-bit) and store through far pointer
        if isinstance(source, MIRImmediate):
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(source.value & 0xFFFF), "Load 16-bit immediate")
        elif src_loc and src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            pass  # Already in A (16-bit in m16 mode)
        elif src_loc and src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'X':
            self._emit_instr(Opcode.TXA, None, "Transfer X to A for far ptr store")
        elif src_loc and src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'Y':
            self._emit_instr(Opcode.TYA, None, "Transfer Y to A for far ptr store")
        else:
            self._emit_load_store('LDA', src_loc)
        self._emit_instr(Opcode.STA_DP_INDIRECT_LONG_Y, operand, "Store u16 through far pointer [dp],Y")

    # ========================================================================
    # Stack-Relative Indirect Addressing
    # ========================================================================

    def _get_stack_indirect_opcode(self, mnemonic: str, ptr_loc, is_far: bool, index_register: str = None):
        """
        Get opcode and operand for stack-relative indirect addressing.

        The 65816 has (d,S),Y addressing mode for near pointers on the stack.
        For far pointers, use _emit_far_ptr_access_via_dbr instead.

        Args:
            mnemonic: Base mnemonic ('LDA' or 'STA')
            ptr_loc: Physical location of pointer (on stack)
            is_far: True for far pointer (24-bit)
            index_register: Optional index register ('Y' for (d,S),Y)

        Returns:
            Tuple of (Opcode, operand)
        """
        if is_far:
            # This should be handled by _emit_far_ptr_access_via_dbr
            raise InstructionSelectionError(
                "Far pointer stack indirect should use _emit_far_ptr_access_via_dbr",
                source_loc=self.parent._current_source_loc
            )

        if index_register and index_register != 'Y':
            raise InstructionSelectionError(
                f"Stack indirect addressing only supports Y index register, got: {index_register}",
                source_loc=self.parent._current_source_loc
            )

        if ptr_loc.kind != LocationKind.STACK:
            raise InstructionSelectionError(f"Expected STACK location, got: {ptr_loc.kind}", source_loc=self.parent._current_source_loc)

        # Stack offset for (d,S),Y addressing
        # ptr_loc.stack_offset is already the correct offset from S
        stack_offset = ptr_loc.stack_offset
        operand = StackOffset(stack_offset)

        if mnemonic == 'LDA':
            if index_register == 'Y':
                return Opcode.LDA_STACK_INDIRECT_Y, operand
            raise InstructionSelectionError("Stack indirect without Y index not supported", source_loc=self.parent._current_source_loc)
        elif mnemonic == 'STA':
            if index_register == 'Y':
                return Opcode.STA_STACK_INDIRECT_Y, operand
            raise InstructionSelectionError("Stack indirect without Y index not supported", source_loc=self.parent._current_source_loc)
        else:
            raise InstructionSelectionError(f"Stack indirect addressing not supported for: {mnemonic}", source_loc=self.parent._current_source_loc)
