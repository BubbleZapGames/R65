"""
Instruction selection: MIR → 65816 assembly.

Converts MIR instructions to WLA-DX assembly mnemonics with proper
addressing modes and register usage.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.codegen.function_gen import FunctionCodeGenerator
from r65.compiler.mir.nodes import (
    MIRFunction, MIRInstruction,
    Load, Store, LoadIndirect, StoreIndirect,
    Move, Return, Jump, JumpTable, CondBranch, Call,
    BinaryOp, UnaryOp, Compare, BitTest, Rotate, SetMode, TypeConvert,
    Push, Pull, SaveRegister, RestoreRegister, ReturnFromInterrupt,
    MemoryFill, BlockCopy, InlineAsm,
    VirtualRegister, HardwareRegister, Immediate as MIRImmediate, MemoryLocation
)
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.register_alloc import (
    RegisterAllocator, PhysicalLocation, LocationKind
)
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.errors import InstructionSelectionError
from r65.compiler.codegen.instruction_select_helpers import (
    XBAState, XBAStateManager
)
from r65.compiler.codegen.hw_register_tracker import (
    HardwareRegisterTracker, compute_vreg_last_uses
)
from r65.compiler.codegen.control_flow_select import ControlFlowInstructionSelector
from r65.compiler.codegen.call_select import CallInstructionSelector
from r65.compiler.codegen.memory_select import MemoryOperationSelector
from r65.compiler.codegen.move_select import MoveOperationSelector
from r65.compiler.codegen.type_conversion_select import TypeConversionSelector
from r65.compiler.codegen.compare_select import CompareSelector
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address, StackOffset


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
                 current_function: 'MIRFunction' = None,
                 func_gen: 'FunctionCodeGenerator' = None):
        """
        Initialize instruction selector.

        Args:
            emitter: Assembly emitter
            register_allocator: Register allocator for virtual registers
            memory_allocator: Memory allocator for static variables
            current_function: Current MIR function being generated (for far/near context)
            func_gen: Function code generator (for epilogue emission)
        """
        self.emitter = emitter
        self.reg_alloc = register_allocator
        self.mem_alloc = memory_allocator
        self.current_function = current_function
        self.func_gen = func_gen

        # Helper classes for modular instruction selection
        self.xba_manager = XBAStateManager(emitter)
        self.control_flow_selector = ControlFlowInstructionSelector(self)
        self.call_selector = CallInstructionSelector(self)
        self.memory_selector = MemoryOperationSelector(self)
        self.move_selector = MoveOperationSelector(self)
        self.type_conversion_selector = TypeConversionSelector(self)
        self.compare_selector = CompareSelector(self)

        # Track type info from last Compare instruction for signed/unsigned branching
        self.last_comparison_type = None

        # Counter for generating unique labels
        self._label_counter = 0

        # Hardware register state tracker for optimization
        self.hw_tracker = HardwareRegisterTracker()
        self._instruction_index = 0

        # Initialize tracker from function parameters if available
        if current_function:
            self._init_hw_tracker(current_function)

    # ========================================================================
    # Opcode Selection Helpers
    # ========================================================================

    # Mapping from base mnemonic to opcode variants by addressing mode
    _OPCODE_VARIANTS = {
        'LDA': {
            'DP': Opcode.LDA_DP, 'DP_X': Opcode.LDA_DP_X,
            'ABSOLUTE': Opcode.LDA_ABSOLUTE, 'ABSOLUTE_X': Opcode.LDA_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.LDA_ABSOLUTE_Y,
            'STACK': Opcode.LDA_STACK, 'IMMEDIATE': Opcode.LDA_IMMEDIATE,
        },
        'STA': {
            'DP': Opcode.STA_DP, 'DP_X': Opcode.STA_DP_X,
            'ABSOLUTE': Opcode.STA_ABSOLUTE, 'ABSOLUTE_X': Opcode.STA_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.STA_ABSOLUTE_Y,
            'STACK': Opcode.STA_STACK,
        },
        'LDX': {
            'DP': Opcode.LDX_DP, 'DP_Y': Opcode.LDX_DP_Y,
            'ABSOLUTE': Opcode.LDX_ABSOLUTE, 'ABSOLUTE_Y': Opcode.LDX_ABSOLUTE_Y,
            'IMMEDIATE': Opcode.LDX_IMMEDIATE,
        },
        'STX': {
            'DP': Opcode.STX_DP, 'DP_Y': Opcode.STX_DP_Y,
            'ABSOLUTE': Opcode.STX_ABSOLUTE,
        },
        'LDY': {
            'DP': Opcode.LDY_DP, 'DP_X': Opcode.LDY_DP_X,
            'ABSOLUTE': Opcode.LDY_ABSOLUTE, 'ABSOLUTE_X': Opcode.LDY_ABSOLUTE_X,
            'IMMEDIATE': Opcode.LDY_IMMEDIATE,
        },
        'STY': {
            'DP': Opcode.STY_DP, 'DP_X': Opcode.STY_DP_X,
            'ABSOLUTE': Opcode.STY_ABSOLUTE,
        },
        'STZ': {
            'DP': Opcode.STZ_DP, 'DP_X': Opcode.STZ_DP_X,
            'ABSOLUTE': Opcode.STZ_ABSOLUTE, 'ABSOLUTE_X': Opcode.STZ_ABSOLUTE_X,
        },
        'ADC': {
            'DP': Opcode.ADC_DP, 'DP_X': Opcode.ADC_DP_X,
            'ABSOLUTE': Opcode.ADC_ABSOLUTE, 'ABSOLUTE_X': Opcode.ADC_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.ADC_ABSOLUTE_Y,
            'STACK': Opcode.ADC_STACK, 'IMMEDIATE': Opcode.ADC_IMMEDIATE,
        },
        'SBC': {
            'DP': Opcode.SBC_DP, 'DP_X': Opcode.SBC_DP_X,
            'ABSOLUTE': Opcode.SBC_ABSOLUTE, 'ABSOLUTE_X': Opcode.SBC_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.SBC_ABSOLUTE_Y,
            'STACK': Opcode.SBC_STACK, 'IMMEDIATE': Opcode.SBC_IMMEDIATE,
        },
        'AND': {
            'DP': Opcode.AND_DP, 'DP_X': Opcode.AND_DP_X,
            'ABSOLUTE': Opcode.AND_ABSOLUTE, 'ABSOLUTE_X': Opcode.AND_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.AND_ABSOLUTE_Y,
            'STACK': Opcode.AND_STACK, 'IMMEDIATE': Opcode.AND_IMMEDIATE,
        },
        'ORA': {
            'DP': Opcode.ORA_DP, 'DP_X': Opcode.ORA_DP_X,
            'ABSOLUTE': Opcode.ORA_ABSOLUTE, 'ABSOLUTE_X': Opcode.ORA_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.ORA_ABSOLUTE_Y,
            'STACK': Opcode.ORA_STACK, 'IMMEDIATE': Opcode.ORA_IMMEDIATE,
        },
        'EOR': {
            'DP': Opcode.EOR_DP, 'DP_X': Opcode.EOR_DP_X,
            'ABSOLUTE': Opcode.EOR_ABSOLUTE, 'ABSOLUTE_X': Opcode.EOR_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.EOR_ABSOLUTE_Y,
            'STACK': Opcode.EOR_STACK, 'IMMEDIATE': Opcode.EOR_IMMEDIATE,
        },
        'CMP': {
            'DP': Opcode.CMP_DP, 'DP_X': Opcode.CMP_DP_X,
            'ABSOLUTE': Opcode.CMP_ABSOLUTE, 'ABSOLUTE_X': Opcode.CMP_ABSOLUTE_X,
            'ABSOLUTE_Y': Opcode.CMP_ABSOLUTE_Y,
            'STACK': Opcode.CMP_STACK, 'IMMEDIATE': Opcode.CMP_IMMEDIATE,
        },
        'CPX': {
            'DP': Opcode.CPX_DP, 'ABSOLUTE': Opcode.CPX_ABSOLUTE,
            'IMMEDIATE': Opcode.CPX_IMMEDIATE,
        },
        'CPY': {
            'DP': Opcode.CPY_DP, 'ABSOLUTE': Opcode.CPY_ABSOLUTE,
            'IMMEDIATE': Opcode.CPY_IMMEDIATE,
        },
    }

    # Mappings from register name to push/pull opcodes
    _PUSH_OPCODES = {
        'A': Opcode.PHA, 'X': Opcode.PHX, 'Y': Opcode.PHY,
        'STATUS': Opcode.PHP, 'P': Opcode.PHP,
        'D': Opcode.PHD, 'DBR': Opcode.PHB, 'B': Opcode.PHB,
    }

    _PULL_OPCODES = {
        'A': Opcode.PLA, 'X': Opcode.PLX, 'Y': Opcode.PLY,
        'STATUS': Opcode.PLP, 'P': Opcode.PLP,
        'D': Opcode.PLD, 'DBR': Opcode.PLB, 'B': Opcode.PLB,
    }

    # Mappings for register transfers
    _TRANSFER_OPCODES = {
        ('A', 'X'): Opcode.TAX,
        ('A', 'Y'): Opcode.TAY,
        ('X', 'A'): Opcode.TXA,
        ('Y', 'A'): Opcode.TYA,
        ('X', 'Y'): Opcode.TXY,
        ('Y', 'X'): Opcode.TYX,
    }

    # Mappings for load immediate by register
    _LOAD_IMMEDIATE_OPCODES = {
        'A': Opcode.LDA_IMMEDIATE,
        'X': Opcode.LDX_IMMEDIATE,
        'Y': Opcode.LDY_IMMEDIATE,
    }

    # Mappings for store to DP by register
    _STORE_DP_OPCODES = {
        'A': Opcode.STA_DP,
        'X': Opcode.STX_DP,
        'Y': Opcode.STY_DP,
    }

    def _get_opcode_for_location(self, mnemonic: str, location: PhysicalLocation) -> tuple[Opcode, Address | StackOffset]:
        """
        Get the appropriate Opcode variant and operand for a memory location.

        Args:
            mnemonic: Base instruction mnemonic (e.g., 'LDA', 'STA')
            location: Physical memory location

        Returns:
            Tuple of (Opcode variant, Operand)
        """
        variants = self._OPCODE_VARIANTS.get(mnemonic)
        if not variants:
            raise InstructionSelectionError(f"No opcode variants for mnemonic: {mnemonic}")

        if location.kind == LocationKind.STACK:
            opcode = variants.get('STACK')
            if not opcode:
                raise InstructionSelectionError(f"{mnemonic} does not support stack-relative addressing")
            return opcode, StackOffset(location.stack_offset)

        elif location.kind == LocationKind.SCRATCH:
            addr = location.scratch_addr
            if location.index_register == 'X':
                opcode = variants.get('DP_X')
                if not opcode:
                    raise InstructionSelectionError(f"{mnemonic} does not support DP,X addressing")
            elif location.index_register == 'Y':
                opcode = variants.get('DP_Y')
                if not opcode:
                    raise InstructionSelectionError(f"{mnemonic} does not support DP,Y addressing")
            else:
                opcode = variants.get('DP')
                if not opcode:
                    raise InstructionSelectionError(f"{mnemonic} does not support DP addressing")
            return opcode, Address(addr)

        elif location.kind == LocationKind.MEMORY:
            # Check for ROM label (for #[rom] data accessed directly)
            if location.memory_label:
                # ROM labels always use absolute addressing
                if location.index_register == 'X':
                    opcode = variants.get('ABSOLUTE_X')
                    if not opcode:
                        raise InstructionSelectionError(f"{mnemonic} does not support absolute X addressing")
                elif location.index_register == 'Y':
                    opcode = variants.get('ABSOLUTE_Y')
                    if not opcode:
                        raise InstructionSelectionError(f"{mnemonic} does not support absolute Y addressing")
                else:
                    opcode = variants.get('ABSOLUTE')
                    if not opcode:
                        raise InstructionSelectionError(f"{mnemonic} does not support absolute addressing")
                return opcode, Address(location.memory_label)

            addr = location.memory_addr
            is_dp = addr < 0x100

            if location.index_register == 'X':
                if is_dp:
                    opcode = variants.get('DP_X')
                else:
                    opcode = variants.get('ABSOLUTE_X')
                if not opcode:
                    raise InstructionSelectionError(f"{mnemonic} does not support indexed X addressing")
            elif location.index_register == 'Y':
                if is_dp:
                    opcode = variants.get('DP_Y')
                else:
                    opcode = variants.get('ABSOLUTE_Y')
                if not opcode:
                    raise InstructionSelectionError(f"{mnemonic} does not support indexed Y addressing")
            else:
                if is_dp:
                    opcode = variants.get('DP')
                else:
                    opcode = variants.get('ABSOLUTE')
                if not opcode:
                    raise InstructionSelectionError(f"{mnemonic} does not support {'DP' if is_dp else 'absolute'} addressing")
            return opcode, Address(addr)

        elif location.kind == LocationKind.IMMEDIATE:
            opcode = variants.get('IMMEDIATE')
            if not opcode:
                raise InstructionSelectionError(f"{mnemonic} does not support immediate addressing")
            return opcode, Immediate(location.immediate_value)

        else:
            raise InstructionSelectionError(f"Cannot use location kind {location.kind} as memory operand")

    def _emit_load(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """Emit a load instruction with the appropriate addressing mode."""
        opcode, operand = self._get_opcode_for_location(mnemonic, location)
        self.emitter.emit_instr(opcode, operand, comment)

    def _emit_store(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """Emit a store instruction with the appropriate addressing mode."""
        opcode, operand = self._get_opcode_for_location(mnemonic, location)
        self.emitter.emit_instr(opcode, operand, comment)

    def _emit_op(self, mnemonic: str, location: PhysicalLocation, comment: str = None):
        """Emit an ALU operation with the appropriate addressing mode."""
        opcode, operand = self._get_opcode_for_location(mnemonic, location)
        self.emitter.emit_instr(opcode, operand, comment)

    def _emit_implied(self, opcode: Opcode, comment: str = None):
        """Emit an implied addressing mode instruction."""
        self.emitter.emit_instr(opcode, None, comment)

    def _emit_immediate(self, opcode: Opcode, value: int, comment: str = None):
        """Emit an immediate addressing mode instruction."""
        self.emitter.emit_instr(opcode, Immediate(value), comment)

    def _emit_branch(self, opcode: Opcode, label: str, comment: str = None):
        """Emit a branch instruction to a label."""
        self.emitter.emit_instr(opcode, Address(label), comment)

    def _block_label(self, block_id: int) -> str:
        """Format a block label with function-scoped naming."""
        func_name = self.current_function.name if self.current_function else ""
        return f"{func_name}__L{block_id}"

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
    # Hardware Register Tracking
    # ========================================================================

    def _init_hw_tracker(self, mir_func: 'MIRFunction'):
        """
        Initialize hardware register tracker from function parameters.

        For parameters with register aliases (e.g., `idx @ X`), marks
        the register as containing that parameter value and computes
        when that value is last used.

        Args:
            mir_func: MIR function being generated
        """
        # Compute last use for each virtual register
        vreg_last_uses = compute_vreg_last_uses(mir_func)

        # Also track parameter symbols' last use
        # This requires scanning the alias tracker
        if hasattr(mir_func, 'alias_tracker') and mir_func.alias_tracker:
            # For now, we'll compute liveness for aliased parameters
            # by treating them similarly to vregs
            pass

        # Initialize tracker
        self.hw_tracker.initialize_from_parameters(mir_func, vreg_last_uses)

    def _advance_instruction(self):
        """
        Advance to the next instruction.

        Updates the hardware register tracker's liveness information.
        """
        self._instruction_index += 1
        self.hw_tracker.advance_to(self._instruction_index)

    def _hw_reg_contains_vreg(self, reg_name: str, vreg) -> bool:
        """
        Check if a hardware register contains a specific virtual register value.

        Args:
            reg_name: Hardware register name ('A', 'X', 'Y')
            vreg: VirtualRegister to check for

        Returns:
            True if the register contains that vreg's value
        """
        return self.hw_tracker.contains_vreg(reg_name, vreg)

    def _hw_reg_is_free(self, reg_name: str) -> bool:
        """
        Check if a hardware register is free for scratch use.

        A register is free if its value is no longer needed (dead).

        Args:
            reg_name: Hardware register name

        Returns:
            True if register can be used as scratch
        """
        return self.hw_tracker.is_free(reg_name)

    def _find_vreg_in_hw_reg(self, vreg) -> str:
        """
        Find which hardware register (if any) contains a virtual register.

        Args:
            vreg: VirtualRegister to look for

        Returns:
            Register name ('A', 'X', 'Y') or None if not in any register
        """
        return self.hw_tracker.find_register_containing(vreg)

    def _mark_hw_reg_clobbered(self, reg_name: str):
        """
        Mark a hardware register as clobbered (contents unknown).

        Called when an instruction modifies a register.

        Args:
            reg_name: Register name ('A', 'X', 'Y')
        """
        self.hw_tracker.mark_clobbered(reg_name)

    def _mark_hw_reg_contains_vreg(self, reg_name: str, vreg, last_use: int = -1):
        """
        Mark that a hardware register now contains a virtual register value.

        Args:
            reg_name: Register name
            vreg: VirtualRegister that was loaded
            last_use: Last instruction index where vreg is used
        """
        self.hw_tracker.mark_contains_vreg(reg_name, vreg, last_use)

    # ========================================================================
    # Temporary Storage Management
    # ========================================================================

    def _get_temp_location(self) -> PhysicalLocation:
        """
        Get a safe temporary location for instruction selection.

        Returns a scratch register location if available, otherwise a stack slot.
        This is used when we need to temporarily store a hardware register value
        for operations that can't use hardware registers directly.

        Returns:
            PhysicalLocation for temporary storage
        """
        # First, try to find a free scratch register
        if hasattr(self.reg_alloc, 'scratch_pool'):
            for scratch in self.reg_alloc.scratch_pool.scratches:
                if scratch.is_free:
                    # Use this scratch register (mark temporarily busy)
                    return PhysicalLocation(
                        kind=LocationKind.SCRATCH,
                        scratch_addr=scratch.address,
                        size=scratch.size
                    )

        # No scratch available - use a dedicated stack slot
        # We use a high stack offset that won't conflict with vreg allocation
        # The register allocator uses stack_base_offset (0x16), so we use an offset above that
        temp_stack_offset = 0x15  # Just below the vreg stack area
        return PhysicalLocation(
            kind=LocationKind.STACK,
            stack_offset=temp_stack_offset,
            size=1
        )

    def _get_temp_address(self) -> Address:
        """
        Get an Address object for the temp location.

        This is a convenience method for cases where we need an Address
        rather than a PhysicalLocation.

        Returns:
            Address for the temp location
        """
        temp_loc = self._get_temp_location()
        if temp_loc.kind == LocationKind.SCRATCH:
            return Address(temp_loc.scratch_addr)
        else:
            # Stack-relative - return as stack offset address
            # Note: This requires stack-relative addressing mode
            return StackOffset(temp_loc.stack_offset)

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
            self.memory_selector.select_load(instr)
        elif isinstance(instr, Store):
            self.memory_selector.select_store(instr)
        elif isinstance(instr, LoadIndirect):
            self.memory_selector.select_load_indirect(instr)
        elif isinstance(instr, StoreIndirect):
            self.memory_selector.select_store_indirect(instr)
        elif isinstance(instr, Move):
            self.move_selector.select_move(instr)
        elif isinstance(instr, TypeConvert):
            self.type_conversion_selector.select_type_convert(instr)
        elif isinstance(instr, BinaryOp):
            self.select_binary_op(instr)
        elif isinstance(instr, UnaryOp):
            self.select_unary_op(instr)
        elif isinstance(instr, Compare):
            self.compare_selector.select_compare(instr)
        elif isinstance(instr, BitTest):
            self.compare_selector.select_bit_test(instr)
        elif isinstance(instr, Rotate):
            self.compare_selector.select_rotate(instr)
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
        elif isinstance(instr, MemoryFill):
            self.select_memory_fill(instr)
        elif isinstance(instr, BlockCopy):
            self.select_block_copy(instr)
        elif isinstance(instr, InlineAsm):
            self.select_inline_asm(instr)
        else:
            raise InstructionSelectionError(f"Unsupported MIR instruction: {type(instr).__name__}")

        # Advance instruction index for liveness tracking
        self._advance_instruction()

    # ========================================================================
    # Memory/Move/TypeConvert Operations (delegated)
    # ========================================================================
    # See memory_select.py for: select_load, select_store, select_load_indirect,
    # select_store_indirect
    # See move_select.py for: select_move
    # See type_conversion_select.py for: select_type_convert

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
            isinstance(instr.right, MIRImmediate) and
            instr.right.value == 1 and
            isinstance(instr.left, HardwareRegister) and
            isinstance(instr.dest, HardwareRegister) and
            instr.left.name == instr.dest.name):

            register = instr.dest.name
            if op == '+':
                # Increment
                if register == 'X':
                    self._emit_implied(Opcode.INX, f"{register}++")
                    return
                elif register == 'Y':
                    self._emit_implied(Opcode.INY, f"{register}++")
                    return
                elif register == 'A':
                    self._emit_implied(Opcode.INC, "A++")
                    return
            else:  # op == '-'
                # Decrement
                if register == 'X':
                    self._emit_implied(Opcode.DEX, f"{register}--")
                    return
                elif register == 'Y':
                    self._emit_implied(Opcode.DEY, f"{register}--")
                    return
                elif register == 'A':
                    self._emit_implied(Opcode.DEC, "A--")
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
            # OPTIMIZATION: Check if vreg value is already in X or Y
            # If so, use TXA/TYA instead of loading from memory
            hw_reg = None
            if isinstance(instr.left, VirtualRegister):
                hw_reg = self._find_vreg_in_hw_reg(instr.left)

            if hw_reg and hw_reg in ('X', 'Y'):
                self._emit_register_transfer(hw_reg, 'A')
            else:
                # Load left operand from memory/stack into A
                self._emit_load('LDA', left_loc)

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
            self._emit_store('STA', dest_loc)

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

                self._emit_load('LDA', left_high)

                if isinstance(instr.right, MIRImmediate):
                    high_value = (instr.right.value >> 8) & 0xFF
                    if op == '+':
                        self._emit_immediate(Opcode.ADC_IMMEDIATE, high_value)
                    else:  # '-'
                        self._emit_immediate(Opcode.SBC_IMMEDIATE, high_value)
                else:
                    right_loc = self._get_operand_location(instr.right)
                    if right_loc.kind != LocationKind.HARDWARE:
                        right_high = self._offset_location(right_loc, 1)
                        if op == '+':
                            self._emit_op('ADC', right_high)
                        else:  # '-'
                            self._emit_op('SBC', right_high)

                self._emit_store('STA', dest_high)

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
        self._emit_load('LDA', operand_loc)

        # Perform operation
        if op == '!':
            # Logical NOT: convert to 0 or 1, then XOR with 1
            self._emit_immediate(Opcode.CMP_IMMEDIATE, 0, "Check if zero")
            self._emit_branch(Opcode.BEQ, "+", "Branch if zero")
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 0, "Was non-zero, result = 0")
            self._emit_branch(Opcode.BRA, "++")
            self.emitter.emit_label("+")
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 1, "Was zero, result = 1")
            self.emitter.emit_label("++")
        elif op == '~':
            # Bitwise NOT
            self._emit_immediate(Opcode.EOR_IMMEDIATE, 0xFF, "Bitwise complement")
        elif op == '-':
            # Negation
            self._emit_immediate(Opcode.EOR_IMMEDIATE, 0xFF, "Complement")
            self._emit_implied(Opcode.INC, "Add 1 for two's complement")
        else:
            raise InstructionSelectionError(f"Unsupported unary operation: {op}")

        # Store result
        self._emit_store('STA', dest_loc)

    # ========================================================================
    # Compare/BitTest/Rotate Operations (delegated)
    # ========================================================================
    # See compare_select.py for: select_compare, select_bit_test, select_rotate

    # ========================================================================
    # Arithmetic Helpers
    # ========================================================================

    def _emit_add(self, right_operand, is_u16: bool):
        """Emit addition operation."""
        self._emit_implied(Opcode.CLC)
        self._emit_binary_operation_with_operand("ADC", right_operand, is_u16)

    def _emit_sub(self, right_operand, is_u16: bool):
        """Emit subtraction operation."""
        self._emit_implied(Opcode.SEC)
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
        if not isinstance(operand, MIRImmediate):
            raise InstructionSelectionError(f"{operation} requires constant operand")
        return operand.value

    def _emit_repeated_opcode(self, opcode: Opcode, count: int):
        """Emit an implied opcode repeated count times."""
        for _ in range(count):
            self._emit_implied(opcode)

    def _emit_shift_left(self, right_operand, is_u16: bool):
        """Emit left shift operation (A << count)."""
        count = self._require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 0x00,
                f"Shift by {count} >= {bit_width} bits = 0")
            return

        self._emit_repeated_opcode(Opcode.ASL, count)

    def _emit_shift_right(self, right_operand, is_u16: bool):
        """Emit right shift operation (A >> count)."""
        count = self._require_immediate(right_operand, "Shift")
        bit_width = 16 if is_u16 else 8

        if count >= bit_width:
            self._emit_immediate(Opcode.LDA_IMMEDIATE, 0x00,
                f"Shift by {count} >= {bit_width} bits = 0")
            return

        self._emit_repeated_opcode(Opcode.LSR, count)

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

        self._emit_repeated_opcode(Opcode.ASL, shift_count)

    def _emit_divide(self, right_operand, is_u16: bool):
        """Emit divide by power of 2 (A / 1/2/4/8) using LSR instructions."""
        value = self._require_immediate(right_operand, "Divide")
        shift_count = self._POWER_OF_2_SHIFTS.get(value)

        if shift_count is None:
            raise InstructionSelectionError(
                f"Divide operator only supports 1, 2, 4, 8 (got {value}). "
                f"Use div() for general division.")

        self._emit_repeated_opcode(Opcode.LSR, shift_count)

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
            self._emit_immediate(Opcode.SEP_IMMEDIATE, instr.mask)
        else:
            self._emit_immediate(Opcode.REP_IMMEDIATE, instr.mask)

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
        push_opcode = self._PUSH_OPCODES.get(reg_name)
        if push_opcode:
            self._emit_implied(push_opcode)
        else:
            raise InstructionSelectionError(f"Cannot push register: {reg_name}")

    def select_restore_register(self, instr: RestoreRegister):
        """
        Generate code for RestoreRegister instruction.

        Args:
            instr: RestoreRegister instruction
        """
        reg_name = instr.register.name
        pull_opcode = self._PULL_OPCODES.get(reg_name)
        if pull_opcode:
            self._emit_implied(pull_opcode)
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
        push_opcode = self._PUSH_OPCODES.get(reg)
        if push_opcode:
            self._emit_implied(push_opcode)
        else:
            raise InstructionSelectionError(f"Cannot push register: {reg}")

    def select_pull(self, instr: Pull):
        """
        Generate code for Pull instruction (restore register from stack).

        Args:
            instr: Pull instruction
        """
        reg = instr.register.name
        pull_opcode = self._PULL_OPCODES.get(reg)
        if pull_opcode:
            self._emit_implied(pull_opcode)
        else:
            raise InstructionSelectionError(f"Cannot pull register: {reg}")

    def select_return_from_interrupt(self, instr: ReturnFromInterrupt):
        """
        Generate code for ReturnFromInterrupt instruction.

        Args:
            instr: ReturnFromInterrupt instruction
        """
        self._emit_implied(Opcode.RTI)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_temp_location(self, size: int = 1) -> PhysicalLocation:
        """
        Get a safe temporary location for instruction selection.

        Tries to use a scratch register from the pool first, falls back to
        a dedicated stack slot if no scratch is available.

        Args:
            size: Size in bytes (1 or 2)

        Returns:
            PhysicalLocation for temporary storage
        """
        # Try to find a free scratch register from the pool
        if self.reg_alloc.scratch_pool:
            for scratch in self.reg_alloc.scratch_pool.scratches:
                if scratch.is_free and scratch.size >= size:
                    return PhysicalLocation(
                        kind=LocationKind.SCRATCH,
                        scratch_addr=scratch.address,
                        size=size
                    )

        # No scratch available - use a dedicated stack slot
        # Use a high offset that won't conflict with vreg allocations
        # Stack grows down, so use offset past normal vreg slots
        temp_stack_offset = 0x15  # Reserved temp slot
        return PhysicalLocation(
            kind=LocationKind.STACK,
            stack_offset=temp_stack_offset,
            size=size
        )

    def _get_temp_address(self) -> Address | None:
        """
        Get a safe temporary direct page address for storing hardware registers.

        This is used when we need to store X/Y registers which don't support
        stack-relative addressing.

        Returns:
            Address for temporary storage, or None if no scratch available
        """
        if self.reg_alloc.scratch_pool:
            for scratch in self.reg_alloc.scratch_pool.scratches:
                if scratch.is_free:
                    return Address(scratch.address)
        return None

    def _emit_binary_operation_with_operand(self, operation: str, right_operand, is_u16: bool):
        """
        Emit a binary operation with right operand.

        Handles immediate, memory, and hardware register operands.
        For hardware registers, stores to temp location first.

        Args:
            operation: Instruction mnemonic (ADC, SBC, AND, ORA, EOR)
            right_operand: Right operand (MIRImmediate, VirtualRegister, HardwareRegister)
            is_u16: Whether this is a 16-bit operation
        """
        # Get the immediate opcode for this operation
        immediate_opcode = self._OPCODE_VARIANTS[operation]['IMMEDIATE']

        if isinstance(right_operand, MIRImmediate):
            value = right_operand.value & 0xFF if not is_u16 else right_operand.value
            self._emit_immediate(immediate_opcode, value)
        else:
            right_loc = self._get_operand_location(right_operand)
            if right_loc.kind == LocationKind.HARDWARE:
                # Hardware register - must store to temp location first
                # (65816 can't use hardware registers as operands for these ops)
                if right_loc.hw_register == 'B':
                    # B register requires XBA to access - can use stack or scratch
                    temp_loc = self._get_temp_location()
                    self._access_b_value_in_a()
                    self._emit_store('STA', temp_loc, "Store B to temp")
                    self._ensure_xba_state_normal("Restore A")
                    self._emit_op(operation, temp_loc)
                elif right_loc.hw_register == 'A':
                    # A can use stack-relative addressing
                    temp_loc = self._get_temp_location()
                    self._emit_store('STA', temp_loc, "Store A to temp")
                    self._emit_op(operation, temp_loc)
                elif right_loc.hw_register in ['X', 'Y']:
                    # X/Y don't support stack-relative for STX/STY
                    # Try scratch first, fall back to push/pop
                    temp_addr = self._get_temp_address()
                    if temp_addr:
                        # Use scratch register
                        temp_loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=temp_addr.value, size=1)
                        store_opcode = self._STORE_DP_OPCODES[right_loc.hw_register]
                        self.emitter.emit_instr(store_opcode, temp_addr, f"Store {right_loc.hw_register} to temp")
                        self._emit_op(operation, temp_loc)
                    else:
                        # No scratch available - use push/pop pattern
                        # PHX/PHY pushes value to stack, then we can use stack-relative
                        push_opcode = Opcode.PHX if right_loc.hw_register == 'X' else Opcode.PHY
                        pull_opcode = Opcode.PLX if right_loc.hw_register == 'X' else Opcode.PLY
                        self._emit_implied(push_opcode, f"Push {right_loc.hw_register} for temp")
                        # Stack-relative location at offset 1 (just pushed)
                        temp_loc = PhysicalLocation(kind=LocationKind.STACK, stack_offset=1, size=1)
                        self._emit_op(operation, temp_loc)
                        self._emit_implied(pull_opcode, f"Restore {right_loc.hw_register}")
                else:
                    raise InstructionSelectionError(f"Cannot use hardware register in operation: {right_loc.hw_register}")
            else:
                # Memory location
                self._emit_op(operation, right_loc)

    def _emit_16bit_mem_to_mem(self, src_loc: PhysicalLocation, dest_loc: PhysicalLocation, comment: str = None):
        """
        Emit 16-bit memory-to-memory move (low byte + high byte).

        Args:
            src_loc: Source memory location
            dest_loc: Destination memory location
            comment: Optional comment for first instruction
        """
        # Low byte
        self._emit_load('LDA', src_loc, comment)
        self._emit_store('STA', dest_loc)

        # High byte
        src_high = self._offset_location(src_loc, 1)
        dest_high = self._offset_location(dest_loc, 1)
        self._emit_load('LDA', src_high)
        self._emit_store('STA', dest_high)

    def _emit_16bit_immediate_store(self, value: int, dest_loc: PhysicalLocation):
        """
        Emit 16-bit immediate store (split into low/high bytes).

        Args:
            value: 16-bit immediate value
            dest_loc: Destination memory location
        """
        low = value & 0xFF
        high = (value >> 8) & 0xFF
        dest_high = self._offset_location(dest_loc, 1)

        # Use STZ for zero bytes (more efficient)
        # But STZ doesn't support stack-relative addressing
        can_use_stz = dest_loc.kind != LocationKind.STACK

        if low == 0 and can_use_stz:
            self._emit_store('STZ', dest_loc)
        else:
            self._emit_immediate(Opcode.LDA_IMMEDIATE, low)
            self._emit_store('STA', dest_loc)

        if high == 0 and can_use_stz:
            self._emit_store('STZ', dest_high)
        else:
            self._emit_immediate(Opcode.LDA_IMMEDIATE, high)
            self._emit_store('STA', dest_high)

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
                self._emit_implied(Opcode.PHA, "Save A")
                self._emit_register_transfer(src_reg, 'A')
                self._emit_xba("Move to B")
                self._emit_implied(Opcode.PLA, "Restore A")
                self._invalidate_xba_state()  # State unknown after PLA
                return

        # Direct transfers
        transfer_opcode = self._TRANSFER_OPCODES.get((src_reg, dest_reg))

        if transfer_opcode:
            self._emit_implied(transfer_opcode)
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
        load_opcode = self._LOAD_IMMEDIATE_OPCODES[reg]
        self._emit_immediate(load_opcode, value)

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
            # Check if this is a #[rom] static with a ROM label - access directly from ROM
            if (operand.symbol and
                hasattr(operand.symbol, 'rom_label') and
                operand.symbol.rom_label and
                operand.storage_type == 'rom'):
                return PhysicalLocation(
                    kind=LocationKind.MEMORY,
                    memory_label=operand.symbol.rom_label,
                    size=1,  # Size determined by context
                    index_register=operand.index_register  # Pass indexed addressing info
                )
            # Check if MemoryLocation has explicit address (for offsets)
            elif operand.address is not None:
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
        elif isinstance(operand, MIRImmediate):
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
            # Check if using a ROM label (for #[rom] data accessed directly)
            if location.memory_label:
                base = location.memory_label
            elif location.memory_addr is not None:
                if location.memory_addr < 0x100:
                    # Zero-page
                    base = f"${location.memory_addr:02X}"
                else:
                    # Absolute
                    base = f"${location.memory_addr:04X}"
            else:
                raise InstructionSelectionError("Memory location has neither address nor label")
            # Add index register if present (e.g., "$20,X" or "LABEL,X")
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
            True if 16-bit type (u16, i16, or near pointer)
        """
        if hasattr(type_info, 'name'):
            return type_info.name in ('u16', 'i16')
        # Near pointers are 16-bit (far pointers are 24-bit)
        if hasattr(type_info, 'pointee_type'):
            return not getattr(type_info, 'is_far', False)
        return False

    # ========================================================================
    # Array Initialization Operations
    # ========================================================================

    def _emit_indexed_store(self, base_addr: int, index_reg: str, comment: str = None):
        """Emit a store with indexed addressing: STA base,X or STA base,Y."""
        is_dp = base_addr < 0x100
        if index_reg == 'X':
            opcode = Opcode.STA_DP_X if is_dp else Opcode.STA_ABSOLUTE_X
        else:  # 'Y'
            opcode = Opcode.STA_ABSOLUTE_Y  # No STA dp,Y on 65816
        self.emitter.emit_instr(opcode, Address(base_addr), comment)

    def select_memory_fill(self, instr: MemoryFill):
        """
        Generate code for MemoryFill instruction.

        Fills a memory region with a constant value using a loop.
        Used for array fill expressions like [0; 256].

        For 8-bit elements:
            LDA #fill_value
            LDX #count-1
        .loop:
            STA dest,X
            DEX
            BPL .loop

        Args:
            instr: MemoryFill instruction
        """
        dest_loc = self._get_operand_location(instr.dest)
        fill_value = instr.fill_value
        count = instr.count
        element_size = instr.element_size

        # Total bytes to fill
        total_bytes = count * element_size

        # Generate unique label for this loop
        loop_label = self._get_unique_label()

        self.emitter.emit_comment(f"Fill {count} x {element_size}B elements with #{fill_value}")

        base_addr = dest_loc.memory_addr if dest_loc.kind == LocationKind.MEMORY else dest_loc.scratch_addr

        if element_size == 1:
            # 8-bit element fill
            if total_bytes <= 256:
                # Can use X as counter (0-255)
                self._emit_immediate(Opcode.LDA_IMMEDIATE, fill_value & 0xFF)
                self._emit_immediate(Opcode.LDX_IMMEDIATE, total_bytes - 1)
                self.emitter.emit_label(loop_label)
                self._emit_indexed_store(base_addr, 'X')
                self._emit_implied(Opcode.DEX)
                self._emit_branch(Opcode.BPL, loop_label)
            else:
                # Large array - use 16-bit counter
                # Use Y for counting, X for offset
                self._emit_immediate(Opcode.LDA_IMMEDIATE, fill_value & 0xFF)
                self._emit_immediate(Opcode.LDX_IMMEDIATE, 0x00)
                self._emit_immediate(Opcode.LDY_IMMEDIATE, total_bytes & 0xFFFF)
                self.emitter.emit_label(loop_label)
                self._emit_indexed_store(base_addr, 'X')
                self._emit_implied(Opcode.INX)
                self._emit_implied(Opcode.DEY)
                self._emit_branch(Opcode.BNE, loop_label)

        elif element_size == 2:
            # 16-bit element fill - need to fill low and high bytes
            low_byte = fill_value & 0xFF
            high_byte = (fill_value >> 8) & 0xFF

            # Use X as index (forward), Y as counter (decrement)
            self._emit_immediate(Opcode.LDX_IMMEDIATE, 0x00)
            self._emit_immediate(Opcode.LDY_IMMEDIATE, count)
            self.emitter.emit_label(loop_label)
            # Store low byte at base+X
            self._emit_immediate(Opcode.LDA_IMMEDIATE, low_byte)
            self._emit_indexed_store(base_addr, 'X')
            self._emit_implied(Opcode.INX)
            # Store high byte at base+X+1
            self._emit_immediate(Opcode.LDA_IMMEDIATE, high_byte)
            self._emit_indexed_store(base_addr, 'X')
            self._emit_implied(Opcode.INX)
            # Decrement counter and loop
            self._emit_implied(Opcode.DEY)
            self._emit_branch(Opcode.BNE, loop_label)

    def select_block_copy(self, instr: BlockCopy):
        """
        Generate code for BlockCopy instruction.

        Copies a block of data from ROM to RAM using MVN instruction.
        Used for array literal expressions like [1, 2, 3, 4].

        Generated code:
            LDA #count-1          ; A = byte count - 1
            LDX #<source_addr     ; X = source low word
            LDY #<dest_addr       ; Y = dest low word
            MVN src_bank, dst_bank

        Note: MVN copies A+1 bytes from src_bank:X to dst_bank:Y.

        Args:
            instr: BlockCopy instruction
        """
        from r65.compiler.codegen.asm_nodes import BlockMove

        dest_loc = self._get_operand_location(instr.dest)
        rom_label = instr.rom_data.label
        count = instr.count

        self.emitter.emit_comment(f"Block copy {count} bytes from ROM to RAM")

        # Set up for MVN: A = count - 1, X = source, Y = dest
        # For 16-bit index mode, we need REP #$10 first
        self._emit_immediate(Opcode.REP_IMMEDIATE, 0x30, "16-bit A and index")

        # Load count - 1 into A
        self._emit_immediate(Opcode.LDA_IMMEDIATE, count - 1)

        # Load source address (ROM data label) - use raw emission for label operand
        self.emitter.emit_instr(Opcode.LDX_IMMEDIATE, Address(rom_label))

        # Load destination address
        self._emit_immediate(Opcode.LDY_IMMEDIATE, dest_loc.memory_addr & 0xFFFF)

        # Perform block move
        # MVN src_bank, dst_bank
        # Assuming ROM is in bank 0 and RAM destination bank is $7E
        # For now, use bank 0 for ROM and calculate destination bank from address
        dest_addr = dest_loc.memory_addr
        if dest_addr >= 0x7E0000:
            dest_bank = 0x7E
        elif dest_addr >= 0x7F0000:
            dest_bank = 0x7F
        else:
            dest_bank = 0x00  # Low RAM or zeropage

        self.emitter.emit_instr(Opcode.MVN, BlockMove(0x00, dest_bank))

        # Restore 8-bit mode if needed (depends on context)
        self._emit_immediate(Opcode.SEP_IMMEDIATE, 0x30, "Restore 8-bit mode")

    # ========================================================================
    # Inline Assembly
    # ========================================================================

    def select_inline_asm(self, instr: InlineAsm):
        """
        Generate code for InlineAsm instruction.

        Emits raw assembly instructions verbatim. The compiler assumes all
        registers may be clobbered after inline assembly.

        Args:
            instr: InlineAsm instruction containing list of assembly strings
        """
        for asm_instr in instr.instructions:
            # Emit each assembly instruction as a raw line
            # The instruction string may contain operands, e.g., "LDA #$42"
            # Use raw emission since this is user-provided assembly
            self.emitter.emit_raw(asm_instr)
