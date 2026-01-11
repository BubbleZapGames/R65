"""
Memory operation selector: Load, Store, LoadIndirect, StoreIndirect.

Handles memory access instruction generation including direct and indirect
addressing modes for the 65816 processor.
"""

from r65.compiler.mir.nodes import Load, Store, LoadIndirect, StoreIndirect, Immediate as MIRImmediate, FunctionPointer
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address, StackOffset
from r65.compiler.codegen.base_selector import BaseSelector


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
        dest_loc = self.parent._get_operand_location(instr.dest)
        src_loc = self.parent._get_operand_location(instr.source)

        is_u16 = self.parent._is_16bit(instr.type_info)

        if is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            self._emit_load_store('LDA', src_loc)
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

        # Normal case: memory-to-memory or register-to-memory store
        self._store_from_location(instr, dest_loc, is_u16)

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
        """Store an immediate value to memory."""
        if is_u16:
            self.parent._emit_16bit_immediate_store(value, dest_loc)
        else:
            value_masked = value & 0xFF
            # Use STZ for storing zero (more efficient than LDA #0; STA)
            # But STZ doesn't support stack-relative addressing
            if value_masked == 0 and dest_loc.kind != LocationKind.STACK:
                self._emit_load_store('STZ', dest_loc)
            else:
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value_masked))
                self._emit_load_store('STA', dest_loc)

    def _store_from_location(self, instr: Store, dest_loc, is_u16: bool):
        """Store from a source location to destination."""
        src_loc = self.parent._get_operand_location(instr.source)

        if src_loc.kind == LocationKind.HARDWARE:
            self._store_from_hardware_register(src_loc, dest_loc, is_u16)
        elif is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            self._emit_load_store('LDA', src_loc)
            self._emit_load_store('STA', dest_loc)

    def _store_from_hardware_register(self, src_loc, dest_loc, is_u16: bool):
        """Store from a hardware register to memory."""
        store_mnemonics = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}
        reg = src_loc.hw_register

        if reg == 'B':
            # B register: swap to A, store, swap back
            self.parent._access_b_value_in_a()
            self._emit_load_store('STA', dest_loc)
            self.parent._ensure_xba_state_normal("Restore A register")
        elif reg not in store_mnemonics:
            raise InstructionSelectionError(f"Cannot store from hardware register: {reg}")
        else:
            self._emit_load_store(store_mnemonics[reg], dest_loc)

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

    # ========================================================================
    # Indirect Memory Operations
    # ========================================================================

    def select_load_indirect(self, instr: LoadIndirect):
        """
        Generate code for LoadIndirect instruction.

        dest = *ptr  (indirect addressing)

        For 65816:
        - near pointers use (zp) or (zp),Y addressing modes
        - far pointers use [zp] or [zp],Y addressing modes
        - Stack-located near pointers use (d,S),Y addressing mode
        - Stack-located far pointers are copied to temp DP location first

        Args:
            instr: LoadIndirect instruction
        """
        ptr_loc = self.parent._get_operand_location(instr.pointer)
        dest_loc = self.parent._get_operand_location(instr.dest)

        self._validate_pointer_location(ptr_loc)

        is_u16 = self.parent._is_16bit(instr.type_info)
        if is_u16:
            raise InstructionSelectionError("16-bit indirect loads not yet supported")

        # Handle stack-located pointers
        if ptr_loc.kind == LocationKind.STACK:
            if instr.is_far:
                # Far pointer on stack: use DBR manipulation (primary approach)
                # This emits the instruction directly and restores DBR immediately
                self._emit_far_ptr_access_via_dbr('LDA', ptr_loc, instr.index_register)
                self._emit_load_store('STA', dest_loc)
                return
            else:
                opcode, operand = self._get_stack_indirect_opcode(
                    'LDA', ptr_loc, instr.is_far, instr.index_register
                )
        else:
            opcode, operand = self._get_indirect_opcode(
                'LDA', ptr_loc, instr.is_far, instr.index_register
            )

        self._emit_instr(opcode, operand, "Load through pointer")
        self._emit_load_store('STA', dest_loc)

    def select_store_indirect(self, instr: StoreIndirect):
        """
        Generate code for StoreIndirect instruction.

        *ptr = source  (indirect addressing)

        For 65816:
        - near pointers use (zp) or (zp),Y addressing modes
        - far pointers use [zp] or [zp],Y addressing modes

        Args:
            instr: StoreIndirect instruction
        """
        ptr_loc = self.parent._get_operand_location(instr.pointer)
        src_loc = self.parent._get_operand_location(instr.source)

        self._validate_pointer_location(ptr_loc)

        opcode, operand = self._get_indirect_opcode('STA', ptr_loc, instr.is_far, instr.index_register)
        is_u16 = self.parent._is_16bit(instr.type_info)

        # Load source value into A
        if isinstance(instr.source, MIRImmediate):
            value = instr.source.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
        elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            pass  # Already in A
        else:
            self._emit_load_store('LDA', src_loc)

        if is_u16:
            raise InstructionSelectionError("16-bit indirect stores not yet supported")

        self._emit_instr(opcode, operand, "Store through pointer")

    def _validate_pointer_location(self, ptr_loc):
        """Validate that pointer is in memory (not immediate or hardware register)."""
        if ptr_loc.kind == LocationKind.HARDWARE or ptr_loc.kind == LocationKind.IMMEDIATE:
            raise InstructionSelectionError(
                f"Pointer for indirect addressing must be in memory, got: {ptr_loc}")

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
            addr_value = ptr_loc.scratch_address
        elif ptr_loc.kind == LocationKind.MEMORY:
            addr_value = ptr_loc.memory_address
        else:
            raise InstructionSelectionError(f"Invalid pointer location for indirect addressing: {ptr_loc}")

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
            raise InstructionSelectionError(f"Indirect addressing not supported for: {mnemonic}")

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
                f"Far pointer indirect only supports Y index register, got: {index_register}"
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

        # Select the appropriate opcode
        if mnemonic == 'LDA':
            if index_register == 'Y':
                opcode = Opcode.LDA_STACK_INDIRECT_Y
            else:
                raise InstructionSelectionError("Far pointer indirect without index not yet supported")
        elif mnemonic == 'STA':
            if index_register == 'Y':
                opcode = Opcode.STA_STACK_INDIRECT_Y
            else:
                raise InstructionSelectionError("Far pointer indirect without index not yet supported")
        else:
            raise InstructionSelectionError(f"Indirect addressing not supported for: {mnemonic}")

        # Emit the actual memory access
        self._emit_instr(opcode, operand, f"{mnemonic} through far pointer")

        # Restore original DBR immediately after the access
        self._emit_instr(Opcode.PLB, None, "Restore DBR")

        # Return None to indicate we've already emitted the instruction
        return None, None

    def _get_scratch_for_far_ptr(self):
        """
        Find a scratch register large enough for a far pointer (3 bytes).

        Only returns a scratch address if we're certain one exists and is free.

        Returns:
            Address of scratch register, or None if no suitable scratch available
        """
        if self.parent.reg_alloc.scratch_pool:
            # Only use scratch if we have a definite 3-byte scratch register
            for scratch in self.parent.reg_alloc.scratch_pool.scratches:
                if scratch.is_free and scratch.size >= 3:
                    return scratch.address

        return None

    def _emit_far_ptr_access_via_dp(self, mnemonic: str, ptr_loc, index_register: str = None):
        """
        Access memory through a far pointer using Direct Page Indirect Long.

        Fallback approach when we have a suitable scratch register:
        1. Copy the 3-byte pointer from stack to scratch DP location
        2. Use [dp],Y indirect long addressing

        Only use this when _get_scratch_for_far_ptr returns a valid address.

        Args:
            mnemonic: Base mnemonic ('LDA' or 'STA')
            ptr_loc: Physical location of pointer (on stack)
            index_register: Optional index register ('Y' for [dp],Y)

        Returns:
            Tuple of (Opcode, operand) or (None, None) if already emitted
        """
        scratch_addr = self._get_scratch_for_far_ptr()
        if scratch_addr is None:
            raise InstructionSelectionError(
                "No 3-byte scratch register available for [dp],Y fallback"
            )

        # Need to preserve Y if we're using it for indexing
        if index_register == 'Y':
            self._emit_instr(Opcode.PHY, None, "Save Y for indexed access")

        # Stack offset for the pointer (+1 for stack convention)
        stack_offset = ptr_loc.stack_offset + 1

        # Copy 3 bytes from stack to scratch DP location
        # Use 16-bit mode for efficiency
        self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20), "16-bit A for pointer copy")

        # Load low 16 bits of pointer from stack
        self._emit_instr(Opcode.LDA_STACK, StackOffset(stack_offset), "Load ptr low word")
        self._emit_instr(Opcode.STA_DP, Address(scratch_addr), "Store to scratch DP")

        self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "8-bit A")

        # Load bank byte (3rd byte of far pointer)
        self._emit_instr(Opcode.LDA_STACK, StackOffset(stack_offset + 2), "Load ptr bank")
        self._emit_instr(Opcode.STA_DP, Address(scratch_addr + 2), "Store bank to scratch DP")

        # Restore Y if we saved it
        if index_register == 'Y':
            self._emit_instr(Opcode.PLY, None, "Restore Y for indexed access")

        # Return opcode and operand for indirect long addressing
        operand = Address(scratch_addr)

        if mnemonic == 'LDA':
            if index_register == 'Y':
                return Opcode.LDA_DP_INDIRECT_LONG_Y, operand
            return Opcode.LDA_DP_INDIRECT_LONG, operand
        elif mnemonic == 'STA':
            if index_register == 'Y':
                return Opcode.STA_DP_INDIRECT_LONG_Y, operand
            return Opcode.STA_DP_INDIRECT_LONG, operand
        else:
            raise InstructionSelectionError(f"Indirect addressing not supported for: {mnemonic}")

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
                "Far pointer stack indirect should use _emit_far_ptr_access_via_dbr"
            )

        if index_register and index_register != 'Y':
            raise InstructionSelectionError(
                f"Stack indirect addressing only supports Y index register, got: {index_register}"
            )

        if ptr_loc.kind != LocationKind.STACK:
            raise InstructionSelectionError(f"Expected STACK location, got: {ptr_loc.kind}")

        # Stack offset for (d,S),Y addressing
        # Add 1 because the stack points to the next free byte, not the data
        stack_offset = ptr_loc.stack_offset + 1
        operand = StackOffset(stack_offset)

        if mnemonic == 'LDA':
            if index_register == 'Y':
                return Opcode.LDA_STACK_INDIRECT_Y, operand
            raise InstructionSelectionError("Stack indirect without Y index not supported")
        elif mnemonic == 'STA':
            if index_register == 'Y':
                return Opcode.STA_STACK_INDIRECT_Y, operand
            raise InstructionSelectionError("Stack indirect without Y index not supported")
        else:
            raise InstructionSelectionError(f"Stack indirect addressing not supported for: {mnemonic}")
