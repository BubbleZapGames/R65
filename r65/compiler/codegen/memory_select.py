# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Memory operation selector: Load, Store, LoadIndirect, StoreIndirect.

Handles memory access instruction generation including direct and indirect
addressing modes for the 65816 processor.
"""

from typing import Optional

from r65.compiler.mir.nodes import Load, Store, ModifyMemory, LoadIndirect, StoreIndirect, Immediate as MIRImmediate, FunctionPointer, LabelRef, FarPtrStrategy
from r65.compiler.codegen.register_alloc import LocationKind, PhysicalLocation
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.opcodes import Opcode, STORE_MNEMONICS
from r65.compiler.codegen.constants import M_FLAG
from r65.compiler.codegen.asm_nodes import Immediate, Address, StackOffset
from r65.compiler.codegen.base_selector import BaseSelector
from r65.compiler.codegen.type_utils import is_far_ptr
from r65.compiler.hir.types import PointerTypeInfo


is_far_pointer_type = is_far_ptr

class MemoryOperationSelector(BaseSelector):
    """
    Handles memory operation instruction selection.

    Manages generation of load/store instructions with proper
    addressing modes for direct and indirect memory access.
    """

    # ========================================================================
    # Y-Self Save/Restore for Trait Methods
    # ========================================================================

    def _has_y_self(self) -> bool:
        """Check if current function has self bound to Y register (trait method).

        Returns False for far self D=S path where self is on the stack, and for
        the FixedStack scratch path where self lives in a zeropage scratch slot.
        """
        func = self.parent.current_function
        if func is None or not getattr(func, 'is_trait_method', False):
            return False
        if getattr(func, 'self_far_uses_d_equals_s', False):
            return False  # Self is on stack, not in Y
        if getattr(func, 'self_far_uses_scratch', False):
            return False  # Self is in zeropage scratch, not in Y
        return getattr(func, 'self_y_vreg', None) is not None

    def _save_y_self(self) -> 'Address | None':
        """Save Y (self) to scratch DP before clobbering Y for indirect access.
        Returns the temp address used, or None if PHY was used instead."""
        temp_addr = self.parent._get_temp_address()
        if temp_addr:
            self._emit_instr(Opcode.STY_DP, temp_addr, "Save Y (self) for indirect access")
            return temp_addr
        else:
            # Fallback: push Y (the emitter counts it for stack offsets)
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
        # Check for dead volatile load (side-effect read in interrupt handler).
        # Use BIT instead of LDA to read the address without modifying A.
        mir_func = self.parent.current_function
        dead_volatile = getattr(mir_func, 'dead_volatile_loads', None) if mir_func else None
        if dead_volatile and id(instr) in dead_volatile:
            src_loc = self.parent._get_operand_location(instr.source)
            self._ensure_m8_mode()
            self._emit_load_store('BIT', src_loc)
            return

        dest_loc = self.parent._get_operand_location(instr.dest)
        src_loc = self.parent._get_operand_location(instr.source)

        if dest_loc.is_hw() and not is_far_pointer_type(instr.type_info):
            from r65.compiler.codegen.type_utils import get_type_size
            self.parent.move_selector._load_memory_to_hw_register(
                dest_loc.hw_register, src_loc, self.parent._is_16bit(instr.type_info),
                src_bytes=get_type_size(instr.type_info) if instr.type_info else None)
            return

        # A far pointer is 3 bytes. It fits neither path below, and
        # _is_16bit() reports False for it, so without this branch it would
        # fall into the 8-bit case and load only the low byte, leaving the
        # high two bytes of the destination untouched.
        if is_far_pointer_type(instr.type_info):
            if dest_loc.is_hw():
                raise InstructionSelectionError(
                    f"cannot load a 3-byte far pointer into hardware register "
                    f"{dest_loc.hw_register}",
                    source_loc=self.parent._current_source_loc)
            self._ensure_m8_mode()
            self._emit_pointer_mem_copy(src_loc, dest_loc, 3)
            return

        is_u16 = self.parent._is_16bit(instr.type_info)

        # The destination is memory: every register destination returned above.
        if is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            self._ensure_m8_mode()
            self._emit_load_store('LDA', src_loc)
            self._emit_load_store('STA', dest_loc)

    def select_modify_memory(self, instr: ModifyMemory):
        """
        Generate code for ModifyMemory: INC, DEC, ASL or LSR on memory.

        One read-modify-write at the location's width; A is not touched. The
        resolver picks the form (dp / abs / dp,X / abs,X) exactly as for a
        load, including the D=S switch from dp to absolute. The MIR fold only
        produces locations these instructions can address.

        With `sets_branch_flags`, the block's `== 0` / `!= 0` test was folded
        away: the CondBranch that follows reads the Z flag the instruction set.
        """
        loc = self.parent._get_operand_location(instr.location)
        is_u16 = self.parent._is_16bit(instr.type_info)
        if is_u16:
            self._ensure_m16_mode()
        else:
            self._ensure_m8_mode()
        mnemonic = instr.op
        try:
            opcode, operand = self.parent._get_opcode_for_location(mnemonic, loc)
        except InstructionSelectionError as exc:
            # The MIR fold only builds ModifyMemory for dp/abs, X-indexed or
            # not. Going through A instead would be correct here but invisible
            # to the interrupt-handler register analysis, which counts a
            # read-modify-write on memory as leaving A alone — so refuse
            # rather than fall back.
            raise InstructionSelectionError(
                f"ModifyMemory on a location {mnemonic} cannot address: {exc}",
                source_loc=self._current_source_loc if hasattr(self, '_current_source_loc') else None)
        self._emit_instr(opcode, operand)

        if instr.sets_branch_flags:
            # The CondBranch is `== 0` / `!= 0`: BEQ/BNE on the Z flag just
            # set. Leave no stale state from an earlier Compare behind.
            self.parent._compare_rhs_is_zero = True
            self.parent._comparison_reversed = False
            self.parent._compare_form = None
            self.parent.last_comparison_type = instr.type_info

    def select_store(self, instr: Store):
        """
        Generate code for Store instruction.

        *dest = source

        Args:
            instr: Store instruction
        """
        dest_loc = self.parent._get_operand_location(instr.dest)
        is_u16 = self.parent._is_16bit(instr.type_info)

        # Check if this is a far pointer (3 bytes) — `far fn()` as well as
        # `far *T`, both of which are 3 bytes and neither 8- nor 16-bit.
        is_far_ptr = is_far_pointer_type(instr.type_info)

        # SPECIAL CASE: Storing to B register
        if dest_loc.is_hw('B'):
            self._store_to_b_register(instr, dest_loc)
            return

        # SPECIAL CASE: Handle immediate values
        if isinstance(instr.source, MIRImmediate):
            self._store_immediate(instr.source.value, dest_loc, is_u16, is_far_ptr)
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
            if src_loc.is_hw('A'):
                self.parent._store_to_b_from_a()
            else:
                self._emit_load_store('LDA', src_loc)
                self.parent._mark_a_modified()
                self.parent._store_to_b_from_a()

    def _store_immediate(self, value: int, dest_loc, is_u16: bool,
                         is_far_ptr: bool = False):
        """Store an immediate value to memory or hardware register."""
        # A 3-byte constant pointer (`0x7E1234 as far *u8`) is neither 8- nor
        # 16-bit, so without this it would fall through to the 8-bit branch
        # and write only the low byte. Low word first, then the bank byte.
        if is_far_ptr and not dest_loc.is_hw():
            self.parent._emit_16bit_immediate_store(value & 0xFFFF, dest_loc)
            self._ensure_m8_mode()
            bank_loc = self.parent._offset_location(dest_loc, 2)
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate((value >> 16) & 0xFF))
            self._emit_load_store('STA', bank_loc)
            return

        # Special handling for hardware register A with 16-bit values
        if is_u16 and dest_loc.is_hw('A'):
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
            # STZ has no stack-relative or long form: under SET_DBR, ROM, hw
            # and RAM targets are forced long because DBR may not be their bank.
            is_long_under_set_dbr = False
            if (self.parent.current_function and
                self.parent.current_function.far_ptr_strategy == FarPtrStrategy.SET_DBR and
                dest_loc.kind == LocationKind.MEMORY):
                if dest_loc.memory_label and dest_loc.storage_type == 'rom':
                    is_long_under_set_dbr = True
                elif dest_loc.storage_type == 'hw':
                    is_long_under_set_dbr = True
                elif dest_loc.storage_type == 'ram':
                    is_long_under_set_dbr = True
            can_use_stz = (
                value_masked == 0 and
                dest_loc.kind != LocationKind.STACK and
                not is_long_under_set_dbr and
                # STZ indexes by X only: there is no STZ abs,Y
                dest_loc.index_register != 'Y' and
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
        self._emit_pointer_mem_copy(src_loc, dest_loc, 3)

    def _store_function_pointer(self, func_ptr: FunctionPointer, dest_loc, type_info):
        """Store a function pointer address directly to memory."""
        from r65.compiler.hir.types import FunctionTypeInfo

        func_name = func_ptr.function_name
        is_far = False
        if type_info and isinstance(type_info, FunctionTypeInfo):
            is_far = type_info.is_far

        # Far pointer is 3 bytes (low, high, bank); near is 2 (low, high).
        self._emit_label_pointer_store(func_name, dest_loc, 3 if is_far else 2)

    def _store_label_ref(self, label_ref: LabelRef, dest_loc):
        """Store a label reference address (near pointer) directly to memory."""
        # Near pointer: 2 bytes (low, high) — same pattern as near function pointers
        self._emit_label_pointer_store(label_ref.label_name, dest_loc, 2, noun="label")

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
        - Stack-located far pointers use the function's far-pointer strategy:
          [dp],Y under D=S, (d,S),Y under SET_DBR, else a PHB/PLB bracket
          around (d,S),Y
        - Field offset is loaded into Y for indexed indirect addressing

        Args:
            instr: LoadIndirect instruction
        """
        # Constant pointer optimization: if the pointer vreg has a known compile-time
        # value (e.g. from an inlined literal parameter), emit direct absolute/long
        # addressing instead of indirect through a DP scratch location.
        const_addr = self._const_pointer(instr)
        if const_addr is not None:
            self._emit_const_ptr_load(const_addr, instr)
            return

        ptr_loc = self.parent._get_operand_location(instr.pointer)
        dest_loc = self.parent._get_operand_location(instr.dest)

        # Y-pointer optimization: pointer is in Y register (trait method self)
        # Use LDA $offset,Y — effective address is DBR:(offset + Y)
        # Only valid when no index_register is set (otherwise Y holds the index, not the pointer)
        if (ptr_loc.is_hw('Y')
                and not instr.index_register):
            offset = getattr(instr, 'offset', 0)
            self._emit_y_pointer_load(offset, instr.type_info, dest_loc)
            return

        # Spill hardware register pointers to scratch for indirect addressing
        if ptr_loc.is_hw():
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
            # A PHY (no scratch free) is in the emitter's displacement, which
            # every stack-relative operand after it is placed past.
            y_self_save_addr = self._save_y_self()

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
        is_far_ptr = isinstance(instr.type_info, PointerTypeInfo) and instr.type_info.is_far

        # Handle 24-bit far pointer indirect loads (3 bytes)
        if is_far_ptr:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    strategy = self._get_far_ptr_strategy()
                    if strategy == 'set_dbr':
                        self._emit_24bit_far_ptr_load_via_set_dbr(ptr_loc, dest_loc, instr_index)
                    elif strategy == 'd_equals_s':
                        self._emit_24bit_far_ptr_load_via_d_equals_s(ptr_loc, dest_loc, instr_index)
                    else:
                        self._emit_24bit_far_ptr_load_via_dbr(ptr_loc, dest_loc, instr_index)
                else:
                    self._emit_24bit_stack_indirect_load(ptr_loc, dest_loc, instr_index)
            else:
                self._emit_24bit_indirect_load(ptr_loc, dest_loc, instr.is_far, instr_index)
            if will_clobber_y and self._has_y_self():
                self._restore_y_self(y_self_save_addr)
            return

        # Handle 16-bit indirect loads with byte-by-byte operations
        if is_u16:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    strategy = self._get_far_ptr_strategy()
                    if strategy == 'set_dbr':
                        self._emit_16bit_far_ptr_load_via_set_dbr(ptr_loc, dest_loc, instr_index)
                    elif strategy == 'd_equals_s':
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
                strategy = self._get_far_ptr_strategy()
                if strategy == 'set_dbr':
                    self._emit_far_ptr_access_via_set_dbr('LDA', ptr_loc, instr_index)
                elif strategy == 'd_equals_s':
                    self._emit_far_ptr_access_via_d_equals_s('LDA', ptr_loc, instr_index)
                else:
                    self._emit_far_ptr_access_via_dbr('LDA', ptr_loc, instr_index)
                # Skip store if dest is A (hw-coalesceable) — value already in A
                if not dest_loc.is_hw('A'):
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
        # Skip store if dest is A (hw-coalesceable) — value already in A
        if not dest_loc.is_hw('A'):
            self._emit_load_store('STA', dest_loc)
        if will_clobber_y and self._has_y_self():
            self._restore_y_self(y_self_save_addr)

    def indirect_operand(self, mnemonic: str, instr) -> tuple:
        """`(opcode, operand)` of the single *mnemonic* instruction that reads
        the byte at `instr.pointer[Y]` - a CompareIndirect's `CMP [dp],Y`.

        The addressing is select_load_indirect's 8-bit indexed paths, minus
        the ones that take more than one instruction: a far pointer on the
        stack in a function with no far-pointer strategy needs a PHB/PLB
        bracket, whose PLB would destroy the flags. Nor may the pointer be in
        zero page under D=S, where a DP operand addresses the stack. The
        folding pass only folds through stack parameters, which reach
        neither: one promoted to scratch makes the function SET_DBR. A
        pointer in a register is spilled first, so call this before loading
        A.
        """
        ptr_loc = self.parent._get_operand_location(instr.pointer)
        d_equals_s = self._get_far_ptr_strategy() == 'd_equals_s'
        if d_equals_s and ptr_loc.kind != LocationKind.STACK:
            raise InstructionSelectionError(
                f"{mnemonic} through a pointer outside the frame needs zero page, "
                f"which D=S does not address", source_loc=self.parent._current_source_loc)
        if ptr_loc.is_hw():
            ptr_loc = self._spill_pointer_to_scratch(ptr_loc)
        self._validate_pointer_location(ptr_loc)

        if ptr_loc.kind == LocationKind.STACK:
            if not instr.is_far:
                return (getattr(Opcode, f"{mnemonic}_STACK_INDIRECT_Y"),
                        self.parent.stack_operand(ptr_loc.stack_offset))
            strategy = self._get_far_ptr_strategy()
            if strategy == 'd_equals_s':
                return (getattr(Opcode, f"{mnemonic}_DP_INDIRECT_LONG_Y"),
                        Address(ptr_loc.stack_offset))
            if strategy == 'set_dbr':
                return (getattr(Opcode, f"{mnemonic}_STACK_INDIRECT_Y"),
                        self.parent.stack_operand(ptr_loc.stack_offset))
            raise InstructionSelectionError(
                f"{mnemonic} through a far pointer on the stack needs a "
                f"far-pointer strategy", source_loc=self.parent._current_source_loc)

        if ptr_loc.kind == LocationKind.SCRATCH:
            addr = ptr_loc.scratch_addr
        elif ptr_loc.kind == LocationKind.MEMORY:
            addr = ptr_loc.memory_addr
        else:
            raise InstructionSelectionError(
                f"Invalid pointer location for indirect addressing: {ptr_loc}",
                source_loc=self.parent._current_source_loc)
        mode = "DP_INDIRECT_LONG_Y" if instr.is_far else "DP_INDIRECT_Y"
        return getattr(Opcode, f"{mnemonic}_{mode}"), Address(addr)

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
        # Constant pointer optimization: emit direct absolute/long addressing
        # when the pointer value is statically known.
        const_addr = self._const_pointer(instr)
        if const_addr is not None:
            self._emit_const_ptr_store(const_addr, instr)
            return

        ptr_loc = self.parent._get_operand_location(instr.pointer)
        src_loc = self.parent._get_operand_location(instr.source)

        # Y-pointer optimization: pointer is in Y register (trait method self)
        # Use STA $offset,Y — effective address is DBR:(offset + Y)
        # Only valid when no index_register is set (otherwise Y holds the index, not the pointer)
        if (ptr_loc.is_hw('Y')
                and not instr.index_register):
            offset = getattr(instr, 'offset', 0)
            self._emit_y_pointer_store(offset, instr.type_info, src_loc, instr.source)
            return

        # Spill hardware register pointers to scratch for indirect addressing
        if ptr_loc.is_hw():
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
            # A PHY (no scratch free) is in the emitter's displacement, which
            # every stack-relative operand after it is placed past.
            y_self_save_addr_s = self._save_y_self()

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
        is_far_ptr = isinstance(instr.type_info, PointerTypeInfo) and instr.type_info.is_far

        # Handle 24-bit far pointer indirect stores (3 bytes)
        if is_far_ptr:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    strategy = self._get_far_ptr_strategy()
                    if strategy == 'set_dbr':
                        self._emit_24bit_far_ptr_store_via_set_dbr(ptr_loc, src_loc, instr.source, instr_index)
                    elif strategy == 'd_equals_s':
                        self._emit_24bit_far_ptr_store_via_d_equals_s(ptr_loc, src_loc, instr.source, instr_index)
                    else:
                        self._emit_24bit_far_ptr_store_via_dbr(ptr_loc, src_loc, instr.source, instr_index)
                else:
                    self._emit_24bit_stack_indirect_store(ptr_loc, src_loc, instr.source, instr_index)
            else:
                self._emit_24bit_indirect_store(ptr_loc, src_loc, instr.source, instr.is_far, instr_index)
            if will_clobber_y_s and self._has_y_self():
                self._restore_y_self(y_self_save_addr_s)
            return

        # Handle 16-bit indirect stores with byte-by-byte operations
        if is_u16:
            if ptr_loc.kind == LocationKind.STACK:
                if instr.is_far:
                    strategy = self._get_far_ptr_strategy()
                    if strategy == 'set_dbr':
                        self._emit_16bit_far_ptr_store_via_set_dbr(ptr_loc, src_loc, instr.source, instr_index)
                    elif strategy == 'd_equals_s':
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
            self._load_byte_into_a(src_loc, instr.source)

            if instr.is_far:
                # Far pointer on stack: choose approach based on prologue setup
                strategy = self._get_far_ptr_strategy()
                if strategy == 'set_dbr':
                    self._emit_far_ptr_access_via_set_dbr('STA', ptr_loc, instr_index)
                elif strategy == 'd_equals_s':
                    self._emit_far_ptr_access_via_d_equals_s('STA', ptr_loc, instr_index)
                else:
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

        self._load_byte_into_a(src_loc, instr.source)

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
            if dest_loc.is_hw('A'):
                pass  # Already in A
            else:
                self._emit_load_store('STA', dest_loc)
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
        else:
            # 8-bit load: LDA abs,Y directly, in m8 so it reads one byte
            self._ensure_m8_mode()
            self.parent.emitter.emit_raw(f"    LDA ${offset:04X},Y")
            if dest_loc.is_hw('A'):
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
            elif src_loc.is_hw('A'):
                pass  # Already in A
            else:
                self._emit_load_store('LDA', src_loc)
            self.parent.emitter.emit_raw(f"    STA ${offset:04X},Y")
            self._emit_immediate(Opcode.SEP_IMMEDIATE, M_FLAG, "Restore 8-bit A")
            self.parent.emitter.emit_accu_mode(8)
        else:
            # 8-bit store: load into A, STA abs,Y
            self._ensure_m8_mode()
            self._load_byte_into_a(src_loc, source)
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
        if ptr_loc.is_hw():
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
        if ptr_loc.kind != LocationKind.HARDWARE:
            return ptr_loc

        # Borrowed for this one access, by the selector's transient-scratch rule.
        scratch_loc = self.parent._get_transient_temp_location(size=2)
        if scratch_loc is None:
            raise InstructionSelectionError(
                f"No scratch register available for pointer spill. "
                f"Define a 2-byte scratch register using: #[zeropage(addr, register)] static mut SCRATCH: u16;",
                source_loc=self.parent._current_source_loc
            )
        scratch_addr = scratch_loc.scratch_addr

        # Spill the pointer from hardware register to scratch.
        #
        # X and Y are 16 bits whatever the accumulator is doing (x16 always), so
        # STX/STY write both bytes on their own. A does not: `STA` in m8 writes
        # the low byte and leaves the high byte of the scratch stale, which
        # silently turns the pointer into a wild address. A near pointer
        # returned from a call arrives in A with the caller already switched
        # back to m8, so this is the common path, not a corner.
        if ptr_loc.hw_register == 'X':
            self._emit_instr(Opcode.STX_DP, Address(scratch_addr), "Spill X pointer to scratch")
        elif ptr_loc.hw_register == 'Y':
            self._emit_instr(Opcode.STY_DP, Address(scratch_addr), "Spill Y pointer to scratch")
        elif ptr_loc.hw_register == 'A':
            was_m8 = self.parent.emitter.get_accu_mode() == 8
            self._ensure_m16_mode()
            self._emit_instr(Opcode.STA_DP, Address(scratch_addr), "Spill A pointer to scratch")
            if was_m8:
                self._ensure_m8_mode()
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

        Fallback when the function has no far-pointer strategy (neither D=S
        nor SET_DBR):
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
            (None, None): the access is emitted here.
        """
        if index_register and index_register != 'Y':
            raise InstructionSelectionError(
                f"Far pointer indirect only supports Y index register, got: {index_register}",
                source_loc=self.parent._current_source_loc
            )

        # Stack offset for the pointer (low byte location)
        stack_offset = ptr_loc.stack_offset

        is_store = (mnemonic == 'STA')

        # Save current DBR
        self._emit_instr(Opcode.PHB, None, "Save DBR")

        if is_store:
            # For STA: A holds the value to store. Save it before loading bank byte.
            # PHA saves the value (1 byte in m8 mode).
            self._emit_instr(Opcode.PHA, None, "Save store value")

            # Load bank byte from far pointer (3rd byte, at base offset+2)
            self._emit_instr(Opcode.LDA_STACK, self.parent.stack_operand(stack_offset + 2), "Load ptr bank")
            self._emit_instr(Opcode.PHA, None, "Push bank")
            self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

            # Set up Y index before restoring A
            if not index_register:
                self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect access")
                index_register = 'Y'

            # Restore value to store (PLA pops 1 byte)
            self._emit_instr(Opcode.PLA, None, "Restore store value")

            operand = self.parent.stack_operand(stack_offset)

            self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, operand, "STA through far pointer")
        else:
            # For LDA: bank byte load into A is fine (result goes into A anyway)
            # Load bank byte from far pointer (3rd byte, at base offset+2)
            self._emit_instr(Opcode.LDA_STACK, self.parent.stack_operand(stack_offset + 2), "Load ptr bank")
            self._emit_instr(Opcode.PHA, None, "Push bank")
            self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

            # Now (d,S),Y will use DBR for the bank byte
            operand = self.parent.stack_operand(stack_offset)

            # The (d,S),Y addressing mode requires Y index register
            # If no index provided, set Y to 0
            if not index_register:
                self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect access")
                index_register = 'Y'

            self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, operand, "LDA through far pointer")

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

    def _get_far_ptr_strategy(self):
        """Get the far pointer strategy for the current function.

        Returns 'set_dbr', 'd_equals_s', or None (no strategy / per-access DBR).
        """
        from r65.compiler.mir.nodes import FarPtrStrategy
        current_func = self.parent.current_function
        if current_func and current_func.has_far_ptr_stack_params:
            if current_func.far_ptr_strategy == FarPtrStrategy.SET_DBR:
                return 'set_dbr'
            return 'd_equals_s'
        return None

    def _emit_far_ptr_access_via_set_dbr(self, mnemonic: str, ptr_loc, index_register: str = None):
        """Access memory through a far pointer using bare (d,S),Y when DBR is already set.

        Under SET_DBR strategy, the prologue has already set DBR to the pointer's bank.
        We can use (d,S),Y stack-relative indirect addressing directly without any
        PHB/PLB wrapping — much cheaper than the per-access DBR approach.
        """
        if index_register and index_register != 'Y':
            raise InstructionSelectionError(
                f"Far pointer indirect only supports Y index register, got: {index_register}",
                source_loc=self.parent._current_source_loc
            )

        stack_offset = ptr_loc.stack_offset

        # (d,S),Y requires Y index register — set Y=0 if not already using Y
        if not index_register:
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect access")

        if mnemonic == 'LDA':
            self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, self.parent.stack_operand(stack_offset),
                           "LDA through far pointer (d,S),Y")
        elif mnemonic == 'STA':
            self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, self.parent.stack_operand(stack_offset),
                           "STA through far pointer (d,S),Y")
        else:
            raise InstructionSelectionError(
                f"Indirect addressing not supported for: {mnemonic}",
                source_loc=self.parent._current_source_loc
            )

        return None, None

    # ========================================================================
    # 16-bit Indirect Memory Operations
    # ========================================================================

    def _emit_16bit_indirect_load(self, ptr_loc, dest_loc, is_far: bool, index_register: str = None):
        """16-bit indirect load through (dp),Y / [dp],Y."""
        opcode_load, operand = self._get_indirect_opcode('LDA', ptr_loc, is_far, 'Y')
        self._emit_indirect_word_load(dest_loc, index_register, opcode_load, operand,
                                      "Load word through pointer")

    def _emit_indirect_word_load(self, dest_loc, index_register, opcode, operand, comment):
        """One m16 LDA through an indirect,Y mode, which reads both bytes and
        leaves Y as it was: byte by byte, the INY between them would move an
        index Y that something else still reads (a loop counter in Y)."""
        if index_register != 'Y':
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")
        self._ensure_m16_mode()
        self._emit_instr(opcode, operand, comment)
        if not dest_loc.is_hw('A'):
            self._emit_load_store('STA', dest_loc)

    def _emit_indirect_word_store(self, src_loc, source, index_register, store):
        """Store a 16-bit value byte by byte through *store(i)* (an indirect STA
        indexed by Y). A value in A, or in X/Y (moved to A in m16), stores A
        and then B via XBA; anything else goes byte by byte from memory or an
        immediate. Caller must have ensured m8.
        """
        reg = (src_loc.hw_register
               if src_loc is not None and src_loc.is_hw() and not isinstance(source, MIRImmediate)
               else None)
        if reg not in ('A', 'X', 'Y'):
            self._emit_indirect_store_bytes(2, src_loc, source, index_register, store)
            return
        if index_register != 'Y':
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")
        if reg != 'A':
            self._ensure_m16_mode()
            self._emit_instr(Opcode.TXA if reg == 'X' else Opcode.TYA, None,
                             f"Transfer {reg} to A for indirect store")
            self._ensure_m8_mode()  # A = low byte, B = high byte
        store(0)
        self._emit_instr(Opcode.INY, None, "Increment for high byte")
        self._emit_instr(Opcode.XBA, None, "Swap to get high byte")
        store(1)
        self._emit_instr(Opcode.XBA, None, "Restore A low byte")

    def _load_byte_into_a(self, src_loc, source):
        """A = the 8-bit store value, in m8 (caller's mode). A value in B is
        swapped in, one in X/Y transferred (its low byte)."""
        if isinstance(source, MIRImmediate):
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(source.value & 0xFF))
        elif src_loc.is_hw('A'):
            pass
        elif src_loc.is_hw('B'):
            self.parent._access_b_value_in_a()
        elif src_loc.is_hw('X') or src_loc.is_hw('Y'):
            transfer = Opcode.TXA if src_loc.is_hw('X') else Opcode.TYA
            self._emit_instr(transfer, None, f"Transfer {src_loc.hw_register} to A for store")
        else:
            self._emit_load_store('LDA', src_loc)

    def _load_word_into_a(self, src_loc, source):
        """A = the 16-bit store value, in m16 (caller's mode)."""
        if isinstance(source, MIRImmediate):
            self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(source.value & 0xFFFF), "Load 16-bit immediate")
        elif src_loc and src_loc.is_hw('A'):
            pass
        elif src_loc and src_loc.is_hw('X'):
            self._emit_instr(Opcode.TXA, None, "Transfer X to A for far ptr store")
        elif src_loc and src_loc.is_hw('Y'):
            self._emit_instr(Opcode.TYA, None, "Transfer Y to A for far ptr store")
        else:
            self._emit_load_store('LDA', src_loc)

    def _emit_16bit_indirect_store(self, ptr_loc, src_loc, source, is_far: bool, index_register: str = None):
        """16-bit indirect store as two 8-bit stores through (dp),Y / [dp],Y."""
        self._ensure_m8_mode()
        opcode_store, operand = self._get_indirect_opcode('STA', ptr_loc, is_far, 'Y')
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_word_store(
            src_loc, source, index_register,
            lambda i: self._emit_instr(opcode_store, operand, f"Store {words[i]} byte through pointer"))

    def _emit_indirect_store_bytes(self, n_bytes, src_loc, source, index_register, emit_store):
        """Byte-by-byte indirect store loop shared by the 24-bit store variants.

        For each byte i: load the value byte into A (immediate or src+i), then
        call emit_store(i) to emit the actual indirect store. INY is inserted
        between bytes; Y is zeroed first unless the caller already set it.
        Caller must have ensured 8-bit accumulator mode.
        """
        if index_register != 'Y':
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")
        for i in range(n_bytes):
            if i:
                self._emit_instr(Opcode.INY, None)
            if isinstance(source, MIRImmediate):
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate((source.value >> (8 * i)) & 0xFF))
            else:
                self._emit_load_store('LDA', src_loc if i == 0 else self.parent._offset_location(src_loc, i))
            emit_store(i)

    def _emit_indirect_load_bytes(self, n_bytes, dest_loc, index_register, emit_load):
        """Byte-by-byte indirect load loop shared by the 24-bit load variants.

        For each byte i: call emit_load(i) to load the byte through the pointer
        into A, then store it to dest+i. INY is inserted between bytes; Y is
        zeroed first unless the caller already set it. Caller must have ensured
        8-bit accumulator mode.
        """
        if index_register != 'Y':
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")
        for i in range(n_bytes):
            if i:
                self._emit_instr(Opcode.INY, None)
            emit_load(i)
            self._emit_load_store('STA', dest_loc if i == 0 else self.parent._offset_location(dest_loc, i))

    # Byte index -> comment word, for the indirect load/store helpers below.
    _PTR_BYTE_WORDS = ("low", "high", "bank")

    def _emit_24bit_indirect_store(self, ptr_loc, src_loc, source, is_far: bool, index_register: str = None):
        """Emit 24-bit far pointer indirect store (three 8-bit operations)."""
        self._ensure_m8_mode()
        opcode_store, operand = self._get_indirect_opcode('STA', ptr_loc, is_far, 'Y')
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_store_bytes(
            3, src_loc, source, index_register,
            lambda i: self._emit_instr(opcode_store, operand, f"Store {words[i]} byte through pointer"))

    def _emit_24bit_far_ptr_store_via_d_equals_s(self, ptr_loc, src_loc, source, index_register: str = None):
        """Emit 24-bit far pointer store using [dp],Y addressing (D=S mode)."""
        self._ensure_m8_mode()
        operand = Address(ptr_loc.stack_offset)
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_store_bytes(
            3, src_loc, source, index_register,
            lambda i: self._emit_instr(Opcode.STA_DP_INDIRECT_LONG_Y, operand,
                                       f"Store {words[i]} byte through far pointer [dp],Y"))

    def _emit_24bit_far_ptr_store_via_set_dbr(self, ptr_loc, src_loc, source, index_register: str = None):
        """Emit 24-bit far pointer store using bare (d,S),Y when DBR is set."""
        self._ensure_m8_mode()
        operand = self.parent.stack_operand(ptr_loc.stack_offset)
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_store_bytes(
            3, src_loc, source, index_register,
            lambda i: self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, operand,
                                       f"Store {words[i]} byte through far pointer (d,S),Y"))

    def _emit_24bit_far_ptr_store_via_dbr(self, ptr_loc, src_loc, source, index_register: str = None):
        """Emit 24-bit far pointer store using DBR manipulation."""
        self._ensure_m8_mode()
        # Low byte uses the caller's index register; high/bank bytes use Y.
        self._emit_indirect_store_bytes(
            3, src_loc, source, index_register,
            lambda i: self._emit_far_ptr_access_via_dbr('STA', ptr_loc, index_register if i == 0 else 'Y'))

    def _emit_24bit_stack_indirect_store(self, ptr_loc, src_loc, source, index_register: str = None):
        """Emit 24-bit far pointer store through near stack pointer using (d,S),Y addressing."""
        self._ensure_m8_mode()
        opcode_store, operand = self._get_stack_indirect_opcode('STA', ptr_loc, False, 'Y')
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_store_bytes(
            3, src_loc, source, index_register,
            lambda i: self._emit_instr(opcode_store, operand, f"Store {words[i]} byte through pointer"))

    def _emit_24bit_indirect_load(self, ptr_loc, dest_loc, is_far: bool, index_register: str = None):
        """Emit 24-bit far pointer indirect load (three 8-bit operations)."""
        self._ensure_m8_mode()
        opcode_load, operand = self._get_indirect_opcode('LDA', ptr_loc, is_far, 'Y')
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_load_bytes(
            3, dest_loc, index_register,
            lambda i: self._emit_instr(opcode_load, operand, f"Load {words[i]} byte through far pointer"))

    def _emit_24bit_far_ptr_load_via_d_equals_s(self, ptr_loc, dest_loc, index_register: str = None):
        """Emit 24-bit far pointer load using [dp],Y addressing (D=S mode)."""
        self._ensure_m8_mode()
        operand = Address(ptr_loc.stack_offset)
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_load_bytes(
            3, dest_loc, index_register,
            lambda i: self._emit_instr(Opcode.LDA_DP_INDIRECT_LONG_Y, operand,
                                       f"Load {words[i]} byte through far pointer [dp],Y"))

    def _emit_24bit_far_ptr_load_via_set_dbr(self, ptr_loc, dest_loc, index_register: str = None):
        """Emit 24-bit far pointer load using bare (d,S),Y when DBR is set."""
        self._ensure_m8_mode()
        operand = self.parent.stack_operand(ptr_loc.stack_offset)
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_load_bytes(
            3, dest_loc, index_register,
            lambda i: self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, operand,
                                       f"Load {words[i]} byte through far pointer (d,S),Y"))

    def _emit_24bit_far_ptr_load_via_dbr(self, ptr_loc, dest_loc, index_register: str = None):
        """Emit 24-bit far pointer load using DBR manipulation."""
        self._ensure_m8_mode()
        # Low byte uses the caller's index register; high/bank bytes use Y.
        self._emit_indirect_load_bytes(
            3, dest_loc, index_register,
            lambda i: self._emit_far_ptr_access_via_dbr('LDA', ptr_loc, index_register if i == 0 else 'Y'))

    def _emit_24bit_stack_indirect_load(self, ptr_loc, dest_loc, index_register: str = None):
        """Emit 24-bit far pointer load through near stack pointer using (d,S),Y addressing."""
        self._ensure_m8_mode()
        opcode_load, operand = self._get_stack_indirect_opcode('LDA', ptr_loc, False, 'Y')
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_load_bytes(
            3, dest_loc, index_register,
            lambda i: self._emit_instr(opcode_load, operand, f"Load {words[i]} byte through pointer"))

    def _emit_16bit_stack_indirect_load(self, ptr_loc, dest_loc, index_register: str = None):
        """16-bit load through a near pointer on the stack, (d,S),Y."""
        opcode_load, operand = self._get_stack_indirect_opcode('LDA', ptr_loc, False, 'Y')
        self._emit_indirect_word_load(dest_loc, index_register, opcode_load, operand,
                                      "Load word through stack pointer")

    def _emit_16bit_stack_indirect_store(self, ptr_loc, src_loc, source, index_register: str = None):
        """16-bit store through a near pointer on the stack, (d,S),Y."""
        self._ensure_m8_mode()
        opcode_store, operand = self._get_stack_indirect_opcode('STA', ptr_loc, False, 'Y')
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_word_store(
            src_loc, source, index_register,
            lambda i: self._emit_instr(opcode_store, operand, f"Store {words[i]} byte through stack pointer"))

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
        self._emit_instr(Opcode.LDA_STACK, self.parent.stack_operand(stack_offset + 2), "Load ptr bank")
        self._emit_instr(Opcode.PHA, None, "Push bank")
        self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

        operand = self.parent.stack_operand(stack_offset)

        # dest_loc goes through the resolver, which counts the PHB
        adj_dest = dest_loc
        words = self._PTR_BYTE_WORDS
        self._emit_indirect_load_bytes(
            2, adj_dest, index_register,
            lambda i: self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, operand,
                                       f"Load {words[i]} byte through far pointer"))

        # Restore DBR - PLB correctly pops the PHB-saved value
        self._emit_instr(Opcode.PLB, None, "Restore DBR")

        # Reassemble 16-bit result into A so that hw-coalescence works correctly.
        # The byte-by-byte load above only leaves the high byte in A (m8 mode).
        # Load the full 16-bit value from dest so A has the correct result.
        self._ensure_m16_mode()
        self._emit_load_store('LDA', dest_loc)

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
        if src_loc and src_loc.is_hw('A'):
            self._ensure_m8_mode()

            # Save current DBR
            self._emit_instr(Opcode.PHB, None, "Save DBR")

            # Save A's low byte; its high byte stays in B, which the m8
            # bank-byte load does not touch.
            self._emit_instr(Opcode.PHA, None, "Save A low byte")

            self._emit_instr(Opcode.LDA_STACK, self.parent.stack_operand(stack_offset + 2), "Load ptr bank")
            self._emit_instr(Opcode.PHA, None, "Push bank")
            self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

            operand = self.parent.stack_operand(stack_offset)

            if index_register == 'Y':
                pass
            else:
                self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

            # Restore and store low byte
            self._emit_instr(Opcode.PLA, None, "Restore A low byte")
            operand = self.parent.stack_operand(stack_offset)
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
        self._emit_instr(Opcode.LDA_STACK, self.parent.stack_operand(stack_offset + 2), "Load ptr bank")
        self._emit_instr(Opcode.PHA, None, "Push bank")
        self._emit_instr(Opcode.PLB, None, "Set DBR to ptr bank")

        operand = self.parent.stack_operand(stack_offset)

        # src_loc goes through the resolver, which counts the PHB
        adj_src = src_loc

        words = self._PTR_BYTE_WORDS
        self._emit_indirect_word_store(
            adj_src, source, index_register,
            lambda i: self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, operand,
                                       f"Store {words[i]} byte through far pointer"))

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

        self._load_word_into_a(src_loc, source)
        self._emit_instr(Opcode.STA_DP_INDIRECT_LONG_Y, operand, "Store u16 through far pointer [dp],Y")

    def _emit_16bit_far_ptr_load_via_set_dbr(self, ptr_loc, dest_loc, index_register: str = None):
        """Emit 16-bit indirect load through far pointer using (d,S),Y when DBR is set."""
        self._ensure_m16_mode()
        stack_offset = ptr_loc.stack_offset

        if not index_register or index_register != 'Y':
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        self._emit_instr(Opcode.LDA_STACK_INDIRECT_Y, self.parent.stack_operand(stack_offset),
                        "Load u16 through far pointer (d,S),Y")
        self._emit_load_store('STA', dest_loc)

    def _emit_16bit_far_ptr_store_via_set_dbr(self, ptr_loc, src_loc, source, index_register: str = None):
        """Emit 16-bit indirect store through far pointer using (d,S),Y when DBR is set."""
        self._ensure_m16_mode()
        stack_offset = ptr_loc.stack_offset

        if not index_register or index_register != 'Y':
            self._emit_instr(Opcode.LDY_IMMEDIATE, Immediate(0), "Set Y=0 for indirect")

        self._load_word_into_a(src_loc, source)
        self._emit_instr(Opcode.STA_STACK_INDIRECT_Y, self.parent.stack_operand(stack_offset),
                        "Store u16 through far pointer (d,S),Y")

    # ========================================================================
    # Stack-Relative Indirect Addressing
    # ========================================================================

    def _get_stack_indirect_opcode(self, mnemonic: str, ptr_loc, is_far: bool, index_register: str = None):
        """
        Get opcode and operand for stack-relative indirect addressing.

        The 65816 has (d,S),Y addressing mode for near pointers on the stack.
        Far pointers go through _emit_far_ptr_access_via_set_dbr,
        _via_d_equals_s or _via_dbr instead.

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
        # ptr_loc is frame-relative; the operand is placed past what is pushed
        stack_offset = ptr_loc.stack_offset
        operand = self.parent.stack_operand(stack_offset)

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

    # ========================================================================
    # Constant Pointer Direct Addressing
    # ========================================================================

    def _const_pointer(self, instr) -> Optional[int]:
        """The address a near LoadIndirect/StoreIndirect's pointer is known
        to hold, if it can be addressed directly. A near pointer reads the
        DBR bank, as `abs` and `abs,X/Y` do; an address below $100 would be
        emitted as direct page, which reads D's page instead."""
        if instr.is_far:
            return None
        addr = self.parent._vreg_const_map.get(instr.pointer.id)
        if addr is None or not 0x100 <= addr + instr.offset <= 0xFFFF:
            return None
        return addr

    def _emit_const_ptr_load(self, const_addr: int, instr: LoadIndirect):
        """Emit a load through a compile-time-constant pointer using direct addressing.

        Replaces indirect addressing (LDA (dp),Y) with absolute addressing
        (LDA $ADDR,Y) when the pointer value is known at compile time. A u16
        is one m16 load, which leaves the index register alone.
        """
        dest_loc = self.parent._get_operand_location(instr.dest)
        is_u16 = self.parent._is_16bit(instr.type_info)
        if is_u16:
            self._ensure_m16_mode()
        else:
            self._ensure_m8_mode()
        direct_loc = PhysicalLocation(kind=LocationKind.MEMORY,
                                      memory_addr=const_addr + instr.offset,
                                      size=2 if is_u16 else 1,
                                      index_register=instr.index_register)
        self._emit_load_store('LDA', direct_loc)
        if not dest_loc.is_hw('A'):
            self._emit_load_store('STA', dest_loc)

    def _emit_const_ptr_store(self, const_addr: int, instr: StoreIndirect):
        """Emit a store through a compile-time-constant pointer using direct addressing.

        Replaces indirect addressing (STA (dp),Y) with absolute addressing
        (STA $ADDR,Y) when the pointer value is known at compile time. A u16
        is one m16 store, which leaves the index register alone.
        """
        src_loc = self.parent._get_operand_location(instr.source)
        is_u16 = self.parent._is_16bit(instr.type_info)
        direct_loc = PhysicalLocation(kind=LocationKind.MEMORY,
                                      memory_addr=const_addr + instr.offset,
                                      size=2 if is_u16 else 1,
                                      index_register=instr.index_register)
        if is_u16:
            self._ensure_m16_mode()
            self._load_word_into_a(src_loc, instr.source)
        else:
            self._ensure_m8_mode()
            self._load_byte_into_a(src_loc, instr.source)
        self._emit_load_store('STA', direct_loc)
