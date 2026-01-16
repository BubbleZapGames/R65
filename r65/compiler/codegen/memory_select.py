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
                self._emit_instr(transfer_op, comment=f"Transfer to A (no {store_mnemonics[reg]} with this addressing)")
                self._emit_load_store('STA', dest_loc)
                self.parent._mark_a_modified()
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

        # Handle 16-bit indirect loads with byte-by-byte operations
        if is_u16:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    self._emit_16bit_far_ptr_load_via_dbr(ptr_loc, dest_loc, instr.index_register)
                else:
                    self._emit_16bit_stack_indirect_load(ptr_loc, dest_loc, instr.index_register)
            else:
                self._emit_16bit_indirect_load(ptr_loc, dest_loc, instr.is_far, instr.index_register)
            return

        # Handle 8-bit indirect loads
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

        is_u16 = self.parent._is_16bit(instr.type_info)

        # Handle 16-bit indirect stores with byte-by-byte operations
        if is_u16:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    self._emit_16bit_far_ptr_store_via_dbr(ptr_loc, src_loc, instr.source, instr.index_register)
                else:
                    self._emit_16bit_stack_indirect_store(ptr_loc, src_loc, instr.source, instr.index_register)
            else:
                self._emit_16bit_indirect_store(ptr_loc, src_loc, instr.source, instr.is_far, instr.index_register)
            return

        # Handle 8-bit indirect stores
        opcode, operand = self._get_indirect_opcode('STA', ptr_loc, instr.is_far, instr.index_register)

        # Load source value into A
        if isinstance(instr.source, MIRImmediate):
            value = instr.source.value & 0xFF
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(value))
        elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            pass  # Already in A
        else:
            self._emit_load_store('LDA', src_loc)

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

    # ========================================================================
    # 16-bit Indirect Memory Operations
    # ========================================================================

    def _emit_16bit_indirect_load(self, ptr_loc, dest_loc, is_far: bool, index_register: str = None):
        """
        Emit 16-bit indirect load (two 8-bit operations).

        The 65816's indirect addressing instructions always operate on 8 bits,
        even in 16-bit accumulator mode. For 16-bit values, we perform two
        8-bit loads at consecutive addresses.

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

        The 65816's indirect addressing instructions always operate on 8 bits.
        For 16-bit values, we perform two 8-bit stores at consecutive addresses.

        Args:
            ptr_loc: Physical location of pointer
            src_loc: Source location of value to store (may be None for immediate)
            source: MIR source operand (for immediate value extraction)
            is_far: True for far pointer (24-bit)
            index_register: Optional index register (only 'Y' supported)
        """
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

        Args:
            ptr_loc: Physical location of pointer (on stack)
            dest_loc: Destination location for loaded value
            index_register: Optional index register (only 'Y' supported)
        """
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

        Args:
            ptr_loc: Physical location of pointer (on stack)
            src_loc: Source location of value to store (may be None for immediate)
            source: MIR source operand (for immediate value extraction)
            index_register: Optional index register (only 'Y' supported)
        """
        opcode_store, operand = self._get_stack_indirect_opcode('STA', ptr_loc, False, 'Y')

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

        Args:
            ptr_loc: Physical location of pointer (on stack)
            dest_loc: Destination location for loaded value
            index_register: Optional index register (only 'Y' supported)
        """
        stack_offset = ptr_loc.stack_offset

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
        self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, operand, "Load low byte through far pointer")
        # Need to save A temporarily since we need to restore DBR before storing
        self._emit_instr(Opcode.PHA, None, "Save low byte")

        # Increment Y for high byte
        self._emit_instr(Opcode.INY, None, "Increment for high byte")

        # Load high byte
        self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, operand, "Load high byte through far pointer")
        # Save high byte
        self._emit_instr(Opcode.PHA, None, "Save high byte")

        # Restore DBR (skip 2 pushed bytes)
        # Need to pull past our saved bytes first
        self._emit_instr(Opcode.PLB, None, "Restore DBR")

        # Now pull our saved bytes and store them
        # High byte is on top after PLB
        dest_high = self.parent._offset_location(dest_loc, 1)
        self._emit_instr(Opcode.PLA, None, "Restore high byte")
        self._emit_load_store('STA', dest_high)
        self._emit_instr(Opcode.PLA, None, "Restore low byte")
        self._emit_load_store('STA', dest_loc)

    def _emit_16bit_far_ptr_store_via_dbr(self, ptr_loc, src_loc, source, index_register: str = None):
        """
        Emit 16-bit indirect store through far pointer on stack using DBR manipulation.

        Args:
            ptr_loc: Physical location of pointer (on stack)
            src_loc: Source location of value to store (may be None for immediate)
            source: MIR source operand (for immediate value extraction)
            index_register: Optional index register (only 'Y' supported)
        """
        stack_offset = ptr_loc.stack_offset

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
                "Far pointer stack indirect should use _emit_far_ptr_access_via_dbr"
            )

        if index_register and index_register != 'Y':
            raise InstructionSelectionError(
                f"Stack indirect addressing only supports Y index register, got: {index_register}"
            )

        if ptr_loc.kind != LocationKind.STACK:
            raise InstructionSelectionError(f"Expected STACK location, got: {ptr_loc.kind}")

        # Stack offset for (d,S),Y addressing
        # ptr_loc.stack_offset is already the correct offset from S
        stack_offset = ptr_loc.stack_offset
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
