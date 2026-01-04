"""
Memory operation selector: Load, Store, LoadIndirect, StoreIndirect.

Handles memory access instruction generation including direct and indirect
addressing modes for the 65816 processor.
"""

from typing import TYPE_CHECKING
from r65.compiler.mir.nodes import Load, Store, LoadIndirect, StoreIndirect, Immediate
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector


class MemoryOperationSelector:
    """
    Handles memory operation instruction selection.

    Manages generation of load/store instructions with proper
    addressing modes for direct and indirect memory access.
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize memory operation selector.

        Args:
            parent: Parent instruction selector (for helper method access)
        """
        self.parent = parent

    @property
    def emitter(self):
        return self.parent.emitter

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
            self.emitter.emit_instruction("LDA", self.parent._format_operand(src_loc))
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

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
        if isinstance(instr.source, Immediate):
            self._store_immediate(instr.source.value, dest_loc, is_u16)
            return

        # Normal case: memory-to-memory or register-to-memory store
        self._store_from_location(instr, dest_loc, is_u16)

    def _store_to_b_register(self, instr: Store, dest_loc):
        """Handle storing to the B register (hidden accumulator high byte)."""
        if isinstance(instr.source, Immediate):
            value = instr.source.value & 0xFF
            self.emitter.emit_instruction("LDA", f"#${value:02X}")
            self.parent._mark_a_modified()
            self.parent._store_to_b_from_a()
        else:
            src_loc = self.parent._get_operand_location(instr.source)
            if src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
                self.parent._store_to_b_from_a()
            else:
                self.emitter.emit_instruction("LDA", self.parent._format_operand(src_loc))
                self.parent._mark_a_modified()
                self.parent._store_to_b_from_a()

    def _store_immediate(self, value: int, dest_loc, is_u16: bool):
        """Store an immediate value to memory."""
        if is_u16:
            self.parent._emit_16bit_immediate_store(value, dest_loc)
        else:
            value_masked = value & 0xFF
            self.emitter.emit_instruction("LDA", f"#${value_masked:02X}")
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

    def _store_from_location(self, instr: Store, dest_loc, is_u16: bool):
        """Store from a source location to destination."""
        src_loc = self.parent._get_operand_location(instr.source)

        if src_loc.kind == LocationKind.HARDWARE:
            self._store_from_hardware_register(src_loc, dest_loc, is_u16)
        elif is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(src_loc))
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

    def _store_from_hardware_register(self, src_loc, dest_loc, is_u16: bool):
        """Store from a hardware register to memory."""
        store_instructions = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}
        reg = src_loc.hw_register

        if reg == 'B':
            # B register: swap to A, store, swap back
            self.parent._access_b_value_in_a()
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))
            self.parent._ensure_xba_state_normal("Restore A register")
        elif reg not in store_instructions:
            raise InstructionSelectionError(f"Cannot store from hardware register: {reg}")
        else:
            self.emitter.emit_instruction(store_instructions[reg], self.parent._format_operand(dest_loc))

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

        Args:
            instr: LoadIndirect instruction
        """
        ptr_loc = self.parent._get_operand_location(instr.pointer)
        dest_loc = self.parent._get_operand_location(instr.dest)

        self._validate_pointer_location(ptr_loc)

        indirect_mode = self._format_indirect_mode(ptr_loc, instr.is_far, instr.index_register)
        is_u16 = self.parent._is_16bit(instr.type_info)

        if is_u16:
            raise InstructionSelectionError("16-bit indirect loads not yet supported")

        self.emitter.emit_instruction("LDA", indirect_mode, "Load through pointer")
        self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

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

        indirect_mode = self._format_indirect_mode(ptr_loc, instr.is_far, instr.index_register)
        is_u16 = self.parent._is_16bit(instr.type_info)

        # Load source value into A
        if isinstance(instr.source, Immediate):
            value = instr.source.value & 0xFF
            self.emitter.emit_instruction("LDA", f"#${value:02X}")
        elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            pass  # Already in A
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(src_loc))

        if is_u16:
            raise InstructionSelectionError("16-bit indirect stores not yet supported")

        self.emitter.emit_instruction("STA", indirect_mode, "Store through pointer")

    def _validate_pointer_location(self, ptr_loc):
        """Validate that pointer is in memory (not immediate or hardware register)."""
        if ptr_loc.kind == LocationKind.HARDWARE or ptr_loc.kind == LocationKind.IMMEDIATE:
            raise InstructionSelectionError(
                f"Pointer for indirect addressing must be in memory, got: {ptr_loc}")

    def _format_indirect_mode(self, ptr_loc, is_far: bool, index_register: str = None) -> str:
        """
        Format indirect addressing mode string.

        Args:
            ptr_loc: Physical location of pointer
            is_far: True for far pointer (24-bit)
            index_register: Optional index register ('Y' for (zp),Y)

        Returns:
            Formatted addressing mode string
        """
        ptr_addr = self.parent._format_operand(ptr_loc)

        if is_far:
            # Far pointer - long indirect [zp] or [zp],Y
            if index_register:
                return f"[{ptr_addr}],{index_register}"
            return f"[{ptr_addr}]"
        else:
            # Near pointer - indirect (zp) or (zp),Y
            if index_register:
                return f"({ptr_addr}),{index_register}"
            return f"({ptr_addr})"
