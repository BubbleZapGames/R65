"""
Move operation selector: Data movement between registers and memory.

Handles Move instruction generation including register transfers,
immediate loads, function pointers, and memory-to-memory moves.
"""

from typing import TYPE_CHECKING
from r65.compiler.mir.nodes import Move, Immediate, FunctionPointer
from r65.compiler.codegen.register_alloc import LocationKind
from r65.compiler.errors import InstructionSelectionError

if TYPE_CHECKING:
    from r65.compiler.codegen.instruction_select import InstructionSelector


class MoveOperationSelector:
    """
    Handles move operation instruction selection.

    Manages generation of data movement instructions including
    register-to-register transfers, immediate loads, and memory moves.
    """

    def __init__(self, parent: 'InstructionSelector'):
        """
        Initialize move operation selector.

        Args:
            parent: Parent instruction selector (for helper method access)
        """
        self.parent = parent

    @property
    def emitter(self):
        return self.parent.emitter

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

        # SPECIAL CASE: Destination is hardware register
        if dest_loc.kind == LocationKind.HARDWARE:
            self._move_to_hardware_register(instr, dest_loc, src_operand, is_u16)
            return

        # Handle function pointers
        if isinstance(src_operand, FunctionPointer):
            self._move_function_pointer(src_operand, dest_loc, instr.type_info)
            return

        # Handle immediate values (including symbolic addresses)
        if isinstance(src_operand, Immediate):
            self._move_immediate(instr, dest_loc, src_operand, is_u16)
            return

        # Move from register/memory to memory
        self._move_from_location(instr, dest_loc, src_operand, is_u16)

    # ========================================================================
    # Move to Hardware Register
    # ========================================================================

    def _move_to_hardware_register(self, instr: Move, dest_loc, src_operand, is_u16: bool):
        """Handle moving data to a hardware register."""
        if isinstance(src_operand, Immediate):
            self._load_immediate_to_hw_register(dest_loc.hw_register, src_operand.value, is_u16)
        else:
            src_loc = self.parent._get_operand_location(src_operand)
            if src_loc.kind == LocationKind.HARDWARE:
                self.parent._emit_register_transfer(src_loc.hw_register, dest_loc.hw_register)
            else:
                self._load_memory_to_hw_register(dest_loc.hw_register, src_loc)

    def _load_immediate_to_hw_register(self, hw_register: str, value: int, is_u16: bool):
        """Load an immediate value into a hardware register."""
        if hw_register in ['A', 'X', 'Y']:
            self.parent._emit_load_immediate_to_register(hw_register, value, is_u16)
        elif hw_register == 'B':
            value_masked = value & 0xFF
            self.emitter.emit_instruction("LDA", f"#${value_masked:02X}")
            self.parent._mark_a_modified()
            self.parent._store_to_b_from_a()
        elif hw_register == 'S':
            # Set stack pointer: TCS always transfers full 16-bit A
            self.emitter.emit_instruction("REP", "#$20", "16-bit A for stack")
            self.emitter.emit_instruction("LDA", f"#${value:04X}")
            self.emitter.emit_instruction("TCS", comment="Set stack pointer")
            self.emitter.emit_instruction("SEP", "#$20", "Restore 8-bit A")
            self.parent._mark_a_modified()
        elif hw_register == 'D':
            # Set direct page register
            self.emitter.emit_instruction("REP", "#$20", "16-bit A for direct page")
            self.emitter.emit_instruction("LDA", f"#${value:04X}")
            self.emitter.emit_instruction("TCD", comment="Set direct page")
            self.emitter.emit_instruction("SEP", "#$20", "Restore 8-bit A")
            self.parent._mark_a_modified()
        else:
            raise InstructionSelectionError(f"Cannot load immediate into register {hw_register}")

    def _load_memory_to_hw_register(self, hw_register: str, src_loc):
        """Load from memory into a hardware register."""
        operand = self.parent._format_operand(src_loc)

        if hw_register == 'A':
            self.emitter.emit_instruction("LDA", operand)
        elif hw_register == 'X':
            self.emitter.emit_instruction("LDX", operand)
        elif hw_register == 'Y':
            self.emitter.emit_instruction("LDY", operand)
        elif hw_register == 'B':
            self.emitter.emit_instruction("LDA", operand)
            self.emitter.emit_instruction("XBA", comment="Load into B register")
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
        self.emitter.emit_instruction("LDA", f"#<{func_name}", "Load function address low byte")
        self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

        # High byte
        dest_high = self.parent._offset_location(dest_loc, 1)
        self.emitter.emit_instruction("LDA", f"#>{func_name}", "Load function address high byte")
        self.emitter.emit_instruction("STA", self.parent._format_operand(dest_high))

        # Bank byte
        dest_bank = self.parent._offset_location(dest_loc, 2)
        self.emitter.emit_instruction("LDA", f"#^{func_name}", "Load function bank byte")
        self.emitter.emit_instruction("STA", self.parent._format_operand(dest_bank))

    def _emit_near_function_pointer(self, func_name: str, dest_loc):
        """Emit code to store a near function pointer (2 bytes)."""
        # Low byte
        self.emitter.emit_instruction("LDA", f"#<{func_name}", "Load function address low byte")
        self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

        # High byte
        dest_high = self.parent._offset_location(dest_loc, 1)
        self.emitter.emit_instruction("LDA", f"#>{func_name}", "Load function address high byte")
        self.emitter.emit_instruction("STA", self.parent._format_operand(dest_high))

    # ========================================================================
    # Immediate Value Handling
    # ========================================================================

    def _move_immediate(self, instr: Move, dest_loc, src_operand: Immediate, is_u16: bool):
        """Handle moving immediate values including symbolic addresses."""
        # Check for symbolic address (from address-of operator or function identifier)
        if hasattr(src_operand, 'symbol') and src_operand.symbol is not None:
            self._move_symbolic_address(instr, dest_loc, src_operand, is_u16)
            return

        value = src_operand.value
        if is_u16:
            self.parent._emit_16bit_immediate_store(value, dest_loc)
        else:
            self.emitter.emit_instruction("LDA", f"#${value:02X}")
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

    def _move_symbolic_address(self, instr: Move, dest_loc, src_operand: Immediate, is_u16: bool):
        """Handle moving a symbolic address (variable or function address)."""
        from r65.compiler.hir.symbol_table import SymbolKind
        from r65.compiler.hir.types import FunctionTypeInfo

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
            # Variable address
            self._emit_variable_address(symbol, dest_loc, is_u16)

    def _emit_variable_address(self, symbol, dest_loc, is_u16: bool):
        """Emit code to store a variable's address."""
        alloc = self.parent.mem_alloc.get_allocation(symbol)
        if not alloc:
            raise InstructionSelectionError(f"No allocation for symbol: {symbol.name}")

        if is_u16:
            # Low byte
            self.emitter.emit_instruction("LDA", f"#<${alloc.address:04X}",
                                          f"Load address of {symbol.name}")
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))
            # High byte
            dest_high = self.parent._offset_location(dest_loc, 1)
            self.emitter.emit_instruction("LDA", f"#>${alloc.address:04X}")
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_high))
        else:
            # 8-bit address (low byte only)
            self.emitter.emit_instruction("LDA", f"#<${alloc.address:04X}",
                                          f"Load address of {symbol.name}")
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

    # ========================================================================
    # Memory-to-Memory Move
    # ========================================================================

    def _move_from_location(self, instr: Move, dest_loc, src_operand, is_u16: bool):
        """Handle moving from a source location to memory destination."""
        src_loc = self.parent._get_operand_location(src_operand)

        if src_loc.kind == LocationKind.HARDWARE:
            self._store_hw_register_to_memory(src_loc.hw_register, dest_loc, is_u16)
        elif is_u16:
            self.parent._emit_16bit_mem_to_mem(src_loc, dest_loc)
        else:
            self.emitter.emit_instruction("LDA", self.parent._format_operand(src_loc))
            self.emitter.emit_instruction("STA", self.parent._format_operand(dest_loc))

    def _store_hw_register_to_memory(self, src_reg: str, dest_loc, is_u16: bool):
        """Store a hardware register to memory."""
        store_map = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}

        if src_reg in store_map:
            self.emitter.emit_instruction(store_map[src_reg], self.parent._format_operand(dest_loc))
        else:
            raise InstructionSelectionError(
                f"Cannot move {'16-bit ' if is_u16 else ''}value from register {src_reg} to memory")
