"""
Instruction selection: MIR → 65816 assembly.

Converts MIR instructions to WLA-DX assembly mnemonics with proper
addressing modes and register usage.
"""

from typing import Union, Optional
from enum import Enum
from r65.compiler.mir.nodes import (
    MIRFunction, MIRInstruction,
    Load, Store, LoadIndirect, StoreIndirect,
    Move, Return, Jump, JumpTable, CondBranch, Call,
    BinaryOp, UnaryOp, Compare, BitTest, Rotate, SetMode, TypeConvert,
    Push, Pull, SaveRegister, RestoreRegister, ReturnFromInterrupt,
    VirtualRegister, HardwareRegister, Immediate, MemoryLocation
)
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.register_alloc import (
    RegisterAllocator, PhysicalLocation, LocationKind
)
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.errors import InstructionSelectionError, compiler_assert
from r65.compiler.codegen.instruction_select_helpers import (
    XBAState, XBAStateManager, BinaryOpEmitter, RegisterMappings
)
from r65.compiler.codegen.control_flow_select import ControlFlowInstructionSelector
from r65.compiler.codegen.call_select import CallInstructionSelector


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

        # Helper classes for modular instruction selection
        self.xba_manager = XBAStateManager(emitter)
        self.binary_op_emitter = BinaryOpEmitter(emitter)
        self.control_flow_selector = ControlFlowInstructionSelector(self)
        self.call_selector = CallInstructionSelector(self)

        # Track type info from last Compare instruction for signed/unsigned branching
        self.last_comparison_type = None

        # Counter for generating unique labels
        self._label_counter = 0

    @property
    def xba_state(self) -> XBAState:
        """XBA state (delegated to manager for backward compatibility)."""
        return self.xba_manager.state

    @xba_state.setter
    def xba_state(self, value: XBAState):
        """Set XBA state (delegated to manager)."""
        self.xba_manager.state = value

    def _get_unique_label(self) -> str:
        """Generate a unique label for inline branching."""
        self._label_counter += 1
        return f"__SCMP{self._label_counter}"

    # ========================================================================
    # XBA State Management (delegates to XBAStateManager)
    # ========================================================================

    def _invalidate_xba_state(self):
        """Invalidate XBA state at control flow boundaries."""
        self.xba_manager.invalidate()

    def _emit_xba(self, comment: str = None):
        """Emit XBA instruction with state tracking."""
        self.xba_manager.emit_xba(comment)

    def _ensure_xba_state_normal(self, comment: str = None):
        """Ensure A and B are in normal positions (A=A, B=B)."""
        self.xba_manager.ensure_normal(comment)

    def _ensure_xba_state_swapped(self, comment: str = None):
        """Ensure A and B are swapped (A=B, B=A)."""
        self.xba_manager.ensure_swapped(comment)

    def _mark_a_modified(self):
        """Mark A register as modified."""
        self.xba_manager.mark_a_modified()

    def _mark_b_modified(self):
        """Mark B register as modified."""
        self.xba_manager.mark_b_modified()

    def _access_b_value_in_a(self):
        """Make B register value available in A for reading."""
        self.xba_manager.access_b_value_in_a()

    def _store_to_b_from_a(self):
        """Store current A value into B register."""
        self.xba_manager.store_to_b_from_a()

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
        elif isinstance(instr, LoadIndirect):
            self.select_load_indirect(instr)
        elif isinstance(instr, StoreIndirect):
            self.select_store_indirect(instr)
        elif isinstance(instr, Move):
            self.select_move(instr)
        elif isinstance(instr, TypeConvert):
            self.select_type_convert(instr)
        elif isinstance(instr, BinaryOp):
            self.select_binary_op(instr)
        elif isinstance(instr, UnaryOp):
            self.select_unary_op(instr)
        elif isinstance(instr, Compare):
            self.select_compare(instr)
        elif isinstance(instr, BitTest):
            self.select_bit_test(instr)
        elif isinstance(instr, Rotate):
            self.select_rotate(instr)
        elif isinstance(instr, Jump):
            self.control_flow_selector.select_jump(instr)
        elif isinstance(instr, JumpTable):
            self.control_flow_selector.select_jump_table(instr)
        elif isinstance(instr, CondBranch):
            self.control_flow_selector.select_cond_branch(instr)
        elif isinstance(instr, Return):
            self.control_flow_selector.select_return(instr)
        elif isinstance(instr, Call):
            self.call_selector.select_call(instr)
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
            raise InstructionSelectionError(f"Unsupported MIR instruction: {type(instr).__name__}")

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
        # Get destination location first (needed for B register check)
        dest_loc = self._get_operand_location(instr.dest)
        is_u16 = self._is_16bit(instr.type_info)

        # SPECIAL CASE: Storing to B register
        if dest_loc.kind == LocationKind.HARDWARE and dest_loc.hw_register == 'B':
            # B register store: load value into A, then XBA to move to B
            if isinstance(instr.source, Immediate):
                value = instr.source.value & 0xFF
                self.emitter.emit_instruction("LDA", f"#${value:02X}")
                self._mark_a_modified()
                self._store_to_b_from_a()
            else:
                src_loc = self._get_operand_location(instr.source)
                if src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
                    # A to B: just XBA
                    self._store_to_b_from_a()
                else:
                    # Load from memory/other register to A, then XBA
                    self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
                    self._mark_a_modified()
                    self._store_to_b_from_a()
            return

        # SPECIAL CASE: Handle immediate values
        # 65816 cannot store immediates directly - must go through accumulator
        if isinstance(instr.source, Immediate):
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

        # Check if source is a hardware register
        if src_loc.kind == LocationKind.HARDWARE:
            # Map hardware registers to their store instructions
            store_instructions = {
                'A': 'STA',
                'X': 'STX',
                'Y': 'STY'
            }

            reg = src_loc.hw_register

            # Special handling for B register (requires XBA to access)
            if reg == 'B':
                # B register: swap to A, store, swap back
                self._access_b_value_in_a()
                self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
                self._ensure_xba_state_normal("Restore A register")
            elif reg not in store_instructions:
                raise InstructionSelectionError(f"Cannot store from hardware register: {reg}")
            else:
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

    def select_load_indirect(self, instr):
        """
        Generate code for LoadIndirect instruction.

        dest = *ptr  (indirect addressing)

        For 65816:
        - near pointers use (zp) or (zp),Y addressing modes
        - far pointers use [zp] or [zp],Y addressing modes

        Args:
            instr: LoadIndirect instruction
        """
        from r65.compiler.mir.nodes import LoadIndirect

        # Get the location where the pointer value is stored
        ptr_loc = self._get_operand_location(instr.pointer)
        dest_loc = self._get_operand_location(instr.dest)

        # Pointer should be in memory (zero-page, stack, or static memory)
        # SCRATCH is zero-page, which is perfect for indirect addressing
        if ptr_loc.kind == LocationKind.HARDWARE or ptr_loc.kind == LocationKind.IMMEDIATE:
            raise InstructionSelectionError(f"Pointer for indirect addressing must be in memory, got: {ptr_loc}")

        # Format indirect addressing mode
        ptr_addr = self._format_operand(ptr_loc)

        # Determine addressing mode based on pointer type and indexing
        if instr.is_far:
            # Far pointer - long indirect [zp] or [zp],Y
            if instr.index_register:
                indirect_mode = f"[{ptr_addr}],{instr.index_register}"
            else:
                indirect_mode = f"[{ptr_addr}]"
        else:
            # Near pointer - indirect (zp) or (zp),Y
            if instr.index_register:
                indirect_mode = f"({ptr_addr}),{instr.index_register}"
            else:
                indirect_mode = f"({ptr_addr})"

        # Load through pointer
        is_u16 = self._is_16bit(instr.type_info)

        if is_u16:
            # 16-bit load through pointer - load low then high byte
            self.emitter.emit_instruction("LDA", indirect_mode, "Load through pointer")
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
            # For 16-bit, we'd need to increment pointer and load again
            # This is complex - for now, only support 8-bit indirect loads
            raise InstructionSelectionError("16-bit indirect loads not yet supported")
        else:
            # 8-bit load through pointer
            self.emitter.emit_instruction("LDA", indirect_mode, "Load through pointer")
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

    def select_store_indirect(self, instr):
        """
        Generate code for StoreIndirect instruction.

        *ptr = source  (indirect addressing)

        For 65816:
        - near pointers use (zp) or (zp),Y addressing modes
        - far pointers use [zp] or [zp],Y addressing modes

        Args:
            instr: StoreIndirect instruction
        """
        from r65.compiler.mir.nodes import StoreIndirect, Immediate

        # Get the location where the pointer value is stored
        ptr_loc = self._get_operand_location(instr.pointer)
        src_loc = self._get_operand_location(instr.source)

        # Pointer should be in memory (zero-page, stack, or static memory)
        # SCRATCH is zero-page, which is perfect for indirect addressing
        if ptr_loc.kind == LocationKind.HARDWARE or ptr_loc.kind == LocationKind.IMMEDIATE:
            raise InstructionSelectionError(f"Pointer for indirect addressing must be in memory, got: {ptr_loc}")

        # Format indirect addressing mode
        ptr_addr = self._format_operand(ptr_loc)

        # Determine addressing mode based on pointer type and indexing
        if instr.is_far:
            # Far pointer - long indirect [zp] or [zp],Y
            if instr.index_register:
                indirect_mode = f"[{ptr_addr}],{instr.index_register}"
            else:
                indirect_mode = f"[{ptr_addr}]"
        else:
            # Near pointer - indirect (zp) or (zp),Y
            if instr.index_register:
                indirect_mode = f"({ptr_addr}),{instr.index_register}"
            else:
                indirect_mode = f"({ptr_addr})"

        # Load source value into A, then store through pointer
        is_u16 = self._is_16bit(instr.type_info)

        if isinstance(instr.source, Immediate):
            # Immediate value - load into A first
            value = instr.source.value & 0xFF
            self.emitter.emit_instruction("LDA", f"#${value:02X}")
        elif src_loc.kind == LocationKind.HARDWARE and src_loc.hw_register == 'A':
            # Already in A
            pass
        else:
            # Load from source location
            self.emitter.emit_instruction("LDA", self._format_operand(src_loc))

        # Store through pointer
        if is_u16:
            raise InstructionSelectionError("16-bit indirect stores not yet supported")
        else:
            # 8-bit store through pointer
            self.emitter.emit_instruction("STA", indirect_mode, "Store through pointer")

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
                elif dest_loc.hw_register == 'B':
                    # Load immediate into B: load into A, then XBA
                    value_masked = value & 0xFF
                    self.emitter.emit_instruction("LDA", f"#${value_masked:02X}")
                    self._mark_a_modified()
                    self._store_to_b_from_a()
                elif dest_loc.hw_register == 'S':
                    # Set stack pointer: load 16-bit value into A, then TCS
                    # TCS always transfers full 16-bit A to S regardless of M flag
                    self.emitter.emit_instruction("REP", "#$20", "16-bit A for stack")
                    self.emitter.emit_instruction("LDA", f"#${value:04X}")
                    self.emitter.emit_instruction("TCS", comment="Set stack pointer")
                    self.emitter.emit_instruction("SEP", "#$20", "Restore 8-bit A")
                    self._mark_a_modified()
                elif dest_loc.hw_register == 'D':
                    # Set direct page register: load 16-bit value into A, then TCD
                    self.emitter.emit_instruction("REP", "#$20", "16-bit A for direct page")
                    self.emitter.emit_instruction("LDA", f"#${value:04X}")
                    self.emitter.emit_instruction("TCD", comment="Set direct page")
                    self.emitter.emit_instruction("SEP", "#$20", "Restore 8-bit A")
                    self._mark_a_modified()
                else:
                    raise InstructionSelectionError(f"Cannot load immediate into register {dest_loc.hw_register}")
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
                    elif dest_loc.hw_register == 'B':
                        # Load from memory into B: load into A, then XBA
                        self.emitter.emit_instruction("LDA", operand)
                        self.emitter.emit_instruction("XBA", comment="Load into B register")
                    else:
                        raise InstructionSelectionError(f"Cannot load into register {dest_loc.hw_register}")
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
            # Check if this is a symbolic address (from address-of operator or function identifier)
            if hasattr(src_operand, 'symbol') and src_operand.symbol is not None:
                from r65.compiler.hir.symbol_table import SymbolKind
                symbol = src_operand.symbol

                # Check if this is a function symbol
                if symbol.kind == SymbolKind.FUNCTION:
                    # Function pointer - emit address of function label
                    # Same as FunctionPointer handling
                    func_name = symbol.name

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
                else:
                    # Variable address - get allocation
                    alloc = self.mem_alloc.get_allocation(symbol)
                    if alloc:
                        # Emit address of the symbol
                        if is_u16:
                            # 16-bit address
                            # Low byte
                            self.emitter.emit_instruction("LDA", f"#<${alloc.address:04X}", f"Load address of {symbol.name}")
                            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
                            # High byte
                            dest_high = self._offset_location(dest_loc, 1)
                            self.emitter.emit_instruction("LDA", f"#>${alloc.address:04X}")
                            self.emitter.emit_instruction("STA", self._format_operand(dest_high))
                        else:
                            # 8-bit address (low byte only)
                            self.emitter.emit_instruction("LDA", f"#<${alloc.address:04X}", f"Load address of {symbol.name}")
                            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
                    else:
                        raise InstructionSelectionError(f"No allocation for symbol: {symbol.name}")
                    return

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
                        raise InstructionSelectionError(f"Cannot move 16-bit value from register {src_reg} to memory")
                else:
                    # 8-bit move from hardware register to memory
                    if src_reg == 'A':
                        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))
                    elif src_reg == 'X':
                        self.emitter.emit_instruction("STX", self._format_operand(dest_loc))
                    elif src_reg == 'Y':
                        self.emitter.emit_instruction("STY", self._format_operand(dest_loc))
                    else:
                        raise InstructionSelectionError(f"Cannot move from register {src_reg} to memory")
            else:
                # Moving from memory to memory
                if is_u16:
                    # 16-bit move
                    self._emit_16bit_mem_to_mem(src_loc, dest_loc)
                else:
                    # 8-bit move
                    self.emitter.emit_instruction("LDA", self._format_operand(src_loc))
                    self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

    def select_type_convert(self, instr: 'TypeConvert'):
        """
        Generate code for TypeConvert instruction.

        Handles:
        - Widening: u8→u16 (zero-extend), i8→i16 (sign-extend)
        - Narrowing: u16→u8 (truncate to low byte)
        - Reinterpret: u8↔i8 (no operation, same bits)

        Args:
            instr: TypeConvert instruction
        """
        src_operand = instr.source
        dest_loc = self._get_operand_location(instr.dest)

        # Get type information
        source_type = str(instr.source_type)
        target_type = str(instr.target_type)
        source_size = 1 if source_type in ['u8', 'i8', 'bool'] else 2
        target_size = 1 if target_type in ['u8', 'i8', 'bool'] else 2
        source_signed = source_type.startswith('i')

        # Case 1: Widening (8-bit → 16-bit)
        if source_size == 1 and target_size == 2:
            # Load source into A
            if isinstance(src_operand, Immediate):
                value = src_operand.value & 0xFF
                self.emitter.emit_instruction("LDA", f"#${value:02X}")
            else:
                src_loc = self._get_operand_location(src_operand)
                self.emitter.emit_instruction("LDA", self._format_operand(src_loc))

            # Store low byte
            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

            if source_signed:
                # Sign extension for i8 → i16
                # If high bit is set (negative), extend with 0xFF, else 0x00
                self.emitter.emit_instruction("AND", "#$80", "Check sign bit")
                self.emitter.emit_instruction("BEQ", "+", "Branch if positive")
                self.emitter.emit_instruction("LDA", "#$FF", "Negative: extend with $FF")
                self.emitter.emit_instruction("BRA", "++")
                self.emitter.emit_label("+")
                self.emitter.emit_instruction("LDA", "#$00", "Positive: extend with $00")
                self.emitter.emit_label("++")
            else:
                # Zero extension for u8 → u16
                self.emitter.emit_instruction("LDA", "#$00", "Zero-extend high byte")

            # Store high byte
            dest_high = self._offset_location(dest_loc, 1)
            self.emitter.emit_instruction("STA", self._format_operand(dest_high))

        # Case 2: Narrowing (16-bit → 8-bit)
        elif source_size == 2 and target_size == 1:
            # Truncate to low byte - just copy low byte
            if isinstance(src_operand, Immediate):
                value = src_operand.value & 0xFF
                self.emitter.emit_instruction("LDA", f"#${value:02X}")
            else:
                src_loc = self._get_operand_location(src_operand)
                self.emitter.emit_instruction("LDA", self._format_operand(src_loc), "Load low byte")

            self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

        # Case 3: Same size - should not happen (handled as Move in MIR builder)
        else:
            raise InstructionSelectionError(f"Unexpected type conversion: {source_type} to {target_type}")

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
        elif op == '*':
            self._emit_multiply(instr.right, is_u16)
        elif op == '/':
            self._emit_divide(instr.right, is_u16)
        else:
            raise InstructionSelectionError(f"Unsupported binary operation: {op}")

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
            raise InstructionSelectionError(f"Unsupported unary operation: {op}")

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
        from r65.compiler.mir.nodes import Immediate

        # Store type info for subsequent CondBranch (for signed/unsigned detection)
        self.last_comparison_type = instr.type_info

        left_loc = self._get_operand_location(instr.left)

        # Handle right operand - if it's a hardware register (including B), store to temp
        if isinstance(instr.right, Immediate):
            right_operand = f"#${instr.right.value:02X}"
        else:
            right_loc = self._get_operand_location(instr.right)
            if right_loc.kind == LocationKind.HARDWARE:
                # Store hardware register to temp location for comparison
                if right_loc.hw_register == 'B':
                    self._access_b_value_in_a()
                    self.emitter.emit_instruction("STA", "$00", "Store B to temp")
                    self._ensure_xba_state_normal("Restore A")
                    right_operand = "$00"
                elif right_loc.hw_register in ['A', 'X', 'Y']:
                    store_instr = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}[right_loc.hw_register]
                    self.emitter.emit_instruction(store_instr, "$00", f"Store {right_loc.hw_register} to temp")
                    right_operand = "$00"
                else:
                    raise InstructionSelectionError(f"Unsupported hardware register: {right_loc.hw_register}")
            else:
                right_operand = self._format_operand(right_loc)

        # Determine which comparison instruction to use based on left operand
        if left_loc.kind == LocationKind.HARDWARE:
            # Hardware register comparison
            if left_loc.hw_register == 'X':
                # CPX instruction
                self.emitter.emit_instruction("CPX", right_operand)
            elif left_loc.hw_register == 'Y':
                # CPY instruction
                self.emitter.emit_instruction("CPY", right_operand)
            elif left_loc.hw_register == 'A':
                # CMP instruction (A is implicit)
                self.emitter.emit_instruction("CMP", right_operand)
            elif left_loc.hw_register == 'B':
                # B register - transfer to A and compare
                self._access_b_value_in_a()
                self.emitter.emit_instruction("CMP", right_operand)
                # Note: Don't restore A since this is just a comparison
                # State is now SWAPPED (A=B, B=A)
            else:
                # Other hardware registers (shouldn't reach here)
                raise InstructionSelectionError(f"Unsupported hardware register for comparison: {left_loc.hw_register}")
        else:
            # Memory or virtual register - load to A and compare
            self.emitter.emit_instruction("LDA", self._format_operand(left_loc))
            self.emitter.emit_instruction("CMP", right_operand)

        # Flags are now set for conditional branch
        # Z flag: set if left == right
        # C flag: set if left >= right (unsigned)
        # N flag: set if result is negative (signed)

    def select_bit_test(self, instr: 'BitTest'):
        """
        Generate code for BitTest instruction using BIT instruction.

        BIT instruction sets flags based on memory value:
        - N flag = bit 7 of memory
        - V flag = bit 6 of memory
        - Z flag = (A & memory) == 0

        Args:
            instr: BitTest instruction
        """
        value_loc = self._get_operand_location(instr.value)

        # BIT instruction requires a memory operand
        # If value is in a hardware register, we can't use BIT directly
        # Just emit BIT with the memory location
        self.emitter.emit_instruction("BIT", self._format_operand(value_loc))

        # Flags are now set:
        # - For bit 7 test: N flag indicates bit 7 value
        # - For bit 6 test: V flag indicates bit 6 value
        # - Z flag can also be used if needed

    def select_rotate(self, instr: 'Rotate'):
        """
        Generate code for Rotate instruction using ROL/ROR instructions.

        Emits ROL (rotate left) or ROR (rotate right) instructions.
        Each rotation is performed count times.

        Args:
            instr: Rotate instruction
        """
        # Load source into A
        source_loc = self._get_operand_location(instr.source)
        self.emitter.emit_instruction("LDA", self._format_operand(source_loc))

        # Determine instruction based on direction
        rotate_instr = "ROL" if instr.direction == 'left' else "ROR"

        # Emit rotate instruction 'count' times
        for _ in range(instr.count):
            self.emitter.emit_instruction(rotate_instr)

        # Store result to destination
        dest_loc = self._get_operand_location(instr.dest)
        self.emitter.emit_instruction("STA", self._format_operand(dest_loc))

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

    # ========================================================================
    # Shift/Multiply/Divide Helpers
    # ========================================================================

    def _require_immediate(self, operand, operation: str) -> int:
        """Validate operand is immediate and return its value."""
        if not isinstance(operand, Immediate):
            raise InstructionSelectionError(f"{operation} requires constant operand")
        return operand.value

    def _emit_repeated(self, mnemonic: str, operand: str, count: int):
        """Emit an instruction repeated count times."""
        for _ in range(count):
            self.emitter.emit_instruction(mnemonic, operand)

    def _emit_shift_left(self, right_operand, is_u16: bool):
        """Emit left shift operation (A << count)."""
        count = self._require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self.emitter.emit_instruction("LDA", "#$00",
                comment=f"Shift by {count} >= {bit_width} bits = 0")
            return

        self._emit_repeated("ASL", "A", count)

    def _emit_shift_right(self, right_operand, is_u16: bool):
        """Emit right shift operation (A >> count)."""
        count = self._require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self.emitter.emit_instruction("LDA", "#$00",
                comment=f"Shift by {count} >= {bit_width} bits = 0")
            return

        self._emit_repeated("LSR", "A", count)

    # Power-of-2 to shift count mapping
    _POWER_OF_2_SHIFTS = {1: 0, 2: 1, 4: 2, 8: 3}

    def _emit_multiply(self, right_operand, is_u16: bool):
        """Emit multiply by power of 2 (A * 1/2/4/8) using ASL instructions."""
        value = self._require_immediate(right_operand, "Multiply")
        shift_count = self._POWER_OF_2_SHIFTS.get(value)

        if shift_count is None:
            raise InstructionSelectionError(
                f"Multiply operator only supports 1, 2, 4, 8 (got {value}). "
                f"Use mul() for general multiplication.")

        self._emit_repeated("ASL", "A", shift_count)

    def _emit_divide(self, right_operand, is_u16: bool):
        """Emit divide by power of 2 (A / 1/2/4/8) using LSR instructions."""
        value = self._require_immediate(right_operand, "Divide")
        shift_count = self._POWER_OF_2_SHIFTS.get(value)

        if shift_count is None:
            raise InstructionSelectionError(
                f"Divide operator only supports 1, 2, 4, 8 (got {value}). "
                f"Use div() for general division.")

        self._emit_repeated("LSR", "A", shift_count)

    # ========================================================================
    # Control Flow (delegated to ControlFlowInstructionSelector)
    # ========================================================================
    # See control_flow_select.py for: select_jump, select_jump_table,
    # select_cond_branch, select_return

    # ========================================================================
    # Function Calls (delegated to CallInstructionSelector)
    # ========================================================================
    # See call_select.py for: select_call, built-in handling, indirect calls

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
        push_instr = RegisterMappings.PUSH.get(reg_name)
        if push_instr:
            self.emitter.emit_instruction(push_instr)
        else:
            raise InstructionSelectionError(f"Cannot push register: {reg_name}")

    def select_restore_register(self, instr: RestoreRegister):
        """
        Generate code for RestoreRegister instruction.

        Args:
            instr: RestoreRegister instruction
        """
        reg_name = instr.register.name
        pull_instr = RegisterMappings.PULL.get(reg_name)
        if pull_instr:
            self.emitter.emit_instruction(pull_instr)
        else:
            raise InstructionSelectionError(f"Cannot pull register: {reg_name}")

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
        push_instr = RegisterMappings.PUSH.get(reg)
        if push_instr:
            self.emitter.emit_instruction(push_instr)
        else:
            raise InstructionSelectionError(f"Cannot push register: {reg}")

    def select_pull(self, instr: Pull):
        """
        Generate code for Pull instruction (restore register from stack).

        Args:
            instr: Pull instruction
        """
        reg = instr.register.name
        pull_instr = RegisterMappings.PULL.get(reg)
        if pull_instr:
            self.emitter.emit_instruction(pull_instr)
        else:
            raise InstructionSelectionError(f"Cannot pull register: {reg}")

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
                if right_loc.hw_register == 'B':
                    # B register requires XBA to access
                    self._access_b_value_in_a()
                    self.emitter.emit_instruction("STA", "$00", "Store B to temp")
                    self._ensure_xba_state_normal("Restore A")
                    self.emitter.emit_instruction(operation, "$00")
                elif right_loc.hw_register in ['A', 'X', 'Y']:
                    store_instr = {'A': 'STA', 'X': 'STX', 'Y': 'STY'}[right_loc.hw_register]
                    self.emitter.emit_instruction(store_instr, "$00", f"Store {right_loc.hw_register} to temp")
                    self.emitter.emit_instruction(operation, "$00")
                else:
                    raise InstructionSelectionError(f"Cannot use hardware register in operation: {right_loc.hw_register}")
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
        B register transfers use XBA instruction.

        Args:
            src_reg: Source register name ('A', 'X', 'Y', 'B')
            dest_reg: Destination register name ('A', 'X', 'Y', 'B')
        """
        if src_reg == dest_reg:
            # No-op
            return

        # Special handling for B register
        if src_reg == 'B' or dest_reg == 'B':
            if src_reg == 'B' and dest_reg == 'A':
                # B to A: If we're swapped, A already has B's value
                self._ensure_xba_state_swapped("Transfer B to A")
                return
            elif src_reg == 'A' and dest_reg == 'B':
                # A to B: XBA swaps them
                self._emit_xba("Transfer A to B")
                return
            elif src_reg == 'B':
                # B to X/Y: B -> A -> X/Y
                self._access_b_value_in_a()
                self._emit_register_transfer('A', dest_reg)
                self._ensure_xba_state_normal("Restore A")
                return
            elif dest_reg == 'B':
                # X/Y to B: X/Y -> A -> B
                # Save current A
                self.emitter.emit_instruction("PHA", comment="Save A")
                self._emit_register_transfer(src_reg, 'A')
                self._emit_xba("Move to B")
                self.emitter.emit_instruction("PLA", comment="Restore A")
                self._invalidate_xba_state()  # State unknown after PLA
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
            # Check if MemoryLocation has explicit address (for offsets)
            if operand.address is not None:
                # Use explicit address (already includes offset for arrays/structs)
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_addr=operand.address,
                    size=1,  # Size determined by context
                    index_register=operand.index_register  # Pass indexed addressing info
                )
            else:
                # Get address from memory allocator
                alloc = self.mem_alloc.get_allocation(operand.symbol)
                if alloc:
                    return PhysicalLocation(
                        kind=LocationKind.MEMORY,
                        memory_addr=alloc.address,
                        size=alloc.size,
                        index_register=operand.index_register  # Pass indexed addressing info
                    )
                else:
                    raise InstructionSelectionError(f"No allocation for symbol: {operand.symbol.name}")
        elif isinstance(operand, Immediate):
            # Immediate value - return as immediate location
            return PhysicalLocation(
                kind=LocationKind.IMMEDIATE,
                immediate_value=operand.value,
                size=1  # Will be determined by context
            )
        else:
            raise InstructionSelectionError(f"Unknown operand type: {type(operand)}")

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
            raise InstructionSelectionError(f"Cannot use hardware register as memory operand: {location.hw_register}")
        elif location.kind == LocationKind.SCRATCH:
            base = f"${location.scratch_addr:02X}"
            if location.index_register:
                return f"{base},{location.index_register}"
            return base
        elif location.kind == LocationKind.MEMORY:
            if location.memory_addr < 0x100:
                # Zero-page
                base = f"${location.memory_addr:02X}"
            else:
                # Absolute
                base = f"${location.memory_addr:04X}"
            # Add index register if present (e.g., "$20,X")
            if location.index_register:
                return f"{base},{location.index_register}"
            return base
        elif location.kind == LocationKind.STACK:
            # Stack-relative addressing using 65816 stack-relative mode
            # Format: $XX,S where XX is the offset from stack pointer
            return f"${location.stack_offset:02X},S"
        elif location.kind == LocationKind.IMMEDIATE:
            # Immediate value
            return f"#{location.immediate_value}"
        else:
            raise InstructionSelectionError(f"Unknown location kind: {location.kind}")

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
            raise InstructionSelectionError(f"Cannot offset location kind: {location.kind}")

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
