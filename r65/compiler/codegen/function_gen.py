"""
Function code generation: MIR functions → assembly.

Generates complete function bodies with headers, labels, and instructions.
"""

from typing import List, Set
from r65.compiler.mir.nodes import MIRFunction
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.register_alloc import ScratchRegisterPool, RegisterAllocator
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.instruction_select_helpers import RegisterMappings
from r65.compiler.codegen.type_utils import get_type_size
from r65.compiler.codegen.constants import DEFAULT_STACK_UPPER
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address, StackOffset


class FunctionCodeGenerator:
    """
    Generates complete assembly functions from MIR.

    Orchestrates function-level code generation including:
    - Function headers with metadata
    - Basic block ordering and labels
    - Instruction selection
    - Register allocation
    """

    def __init__(self,
                 emitter: AssemblyEmitter,
                 memory_allocator: MemoryAllocator):
        """
        Initialize function code generator.

        Args:
            emitter: Assembly emitter
            memory_allocator: Memory allocator for static variables
        """
        self.emitter = emitter
        self.mem_alloc = memory_allocator

    # ========================================================================
    # Emission Helpers
    # ========================================================================

    def _emit_instr(self, opcode: Opcode, operand=None, comment: str = None):
        """Emit an instruction using the node emitter."""
        self.emitter.emit_instr(opcode, operand, comment)

    # ========================================================================
    # Main Generation
    # ========================================================================

    def generate_function(self, mir_func: MIRFunction, scratch_pool: ScratchRegisterPool = None):
        """
        Generate complete assembly for MIR function.

        Args:
            mir_func: MIR function to generate
            scratch_pool: Optional scratch register pool (if None, no scratch registers available)
        """
        # Setup register allocator for this function
        if scratch_pool is None:
            scratch_pool = ScratchRegisterPool()  # Empty pool if not provided

        reg_alloc = RegisterAllocator(scratch_pool=scratch_pool, mir_func=mir_func)

        # Allocate all virtual registers in function
        self._allocate_function_registers(mir_func, reg_alloc)

        # Create instruction selector with current function context
        instr_selector = InstructionSelector(self.emitter, reg_alloc, self.mem_alloc, mir_func, func_gen=self)

        # Emit function header comment
        self.emit_function_header(mir_func)

        # Emit function label
        self.emitter.emit_label(mir_func.name)

        # Emit mode directives for WLA-DX assembler
        self._emit_mode_directives(mir_func)

        # Emit prologue (if needed)
        self.emit_prologue(mir_func, reg_alloc)

        # Generate basic blocks
        block_order = self._compute_block_order(mir_func)

        for block_id in block_order:
            block = mir_func.blocks[block_id]

            # Emit block label (except entry block which uses function label)
            if block_id != mir_func.entry_block_id:
                self.emitter.emit_label(f"{mir_func.name}__L{block_id}")

            # Emit instructions in block
            for instr in block.instructions:
                instr_selector.select_instruction(instr)

        # Emit epilogue (if needed)
        # Note: Epilogue is emitted BEFORE the Return instruction in each block
        # So we don't emit it here. Instead, we handle it in the Return instruction
        # self.emit_epilogue(mir_func, reg_alloc)

        # Blank line after function
        self.emitter.emit_blank_line()

    # ========================================================================
    # Function Header
    # ========================================================================

    def emit_function_header(self, mir_func: MIRFunction):
        """
        Emit function header comment with metadata.

        Args:
            mir_func: MIR function
        """
        # Main divider
        divider = "-" * 76
        self.emitter.emit_comment(divider)

        # Function name
        self.emitter.emit_comment(f"{mir_func.name}")

        # Source location (if available)
        if mir_func.source_loc:
            loc = mir_func.source_loc
            self.emitter.emit_comment(f"Source: {loc.file_path}:{loc.line}")
            # Show include chain if this is from an included file
            if loc.included_from:
                parent = loc.included_from
                while parent:
                    self.emitter.emit_comment(f"  included from {parent.file_path}:{parent.line}")
                    parent = parent.included_from

        self.emitter.emit_comment("")

        # Parameters
        if mir_func.parameters:
            self.emitter.emit_comment("Parameters:")
            for param in mir_func.parameters:
                param_desc = f"  {param.name}: {param.param_type}"
                self.emitter.emit_comment(param_desc)
            self.emitter.emit_comment("")

        # Return type
        if mir_func.return_type:
            self.emitter.emit_comment(f"Returns: {mir_func.return_type}")
            self.emitter.emit_comment("")

        # Attributes
        if mir_func.mode_attr:
            mode_str = f"Mode: {mir_func.mode_attr}"
            self.emitter.emit_comment(mode_str)

        if mir_func.preserves_attr:
            preserves = ", ".join(mir_func.preserves_attr.registers)
            self.emitter.emit_comment(f"Preserves: {preserves}")

        if mir_func.is_entry:
            self.emitter.emit_comment("Entry: true")

        if mir_func.is_far:
            self.emitter.emit_comment("Far: true (JSL/RTL)")

        # Closing divider
        self.emitter.emit_comment(divider)

    # ========================================================================
    # Basic Block Ordering
    # ========================================================================

    def _compute_block_order(self, mir_func: MIRFunction) -> List[int]:
        """
        Compute optimal ordering of basic blocks.

        For now, simple DFS traversal from entry block.
        Future: optimize for fall-through and minimize jumps.

        Args:
            mir_func: MIR function

        Returns:
            List of block IDs in emission order
        """
        visited: Set[int] = set()
        order: List[int] = []

        def visit(block_id: int):
            if block_id in visited:
                return

            visited.add(block_id)
            order.append(block_id)

            # Visit successors
            block = mir_func.blocks.get(block_id)
            if block:
                for successor_id in block.successors:
                    visit(successor_id)

        # Start from entry block
        visit(mir_func.entry_block_id)

        # Visit any unreachable blocks (shouldn't happen, but be safe)
        for block_id in mir_func.blocks.keys():
            if block_id not in visited:
                visit(block_id)

        return order

    # ========================================================================
    # Register Allocation
    # ========================================================================

    def _allocate_function_registers(self,
                                    mir_func: MIRFunction,
                                    reg_alloc: RegisterAllocator):
        """
        Allocate all virtual registers in function.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
        """
        # Collect all virtual registers used in function
        vregs = set()

        for block in mir_func.blocks.values():
            for instr in block.instructions:
                # Extract virtual registers from instruction
                # This is simplified - actual implementation would use visitor pattern
                vregs.update(self._extract_vregs_from_instruction(instr))

        # Allocate all at once
        reg_alloc.allocate_all(list(vregs))

    def _extract_vregs_from_instruction(self, instr) -> Set:
        """
        Extract virtual registers from instruction.

        This is a simplified implementation. A real implementation
        would use a visitor pattern or instruction introspection.

        Args:
            instr: MIR instruction

        Returns:
            Set of VirtualRegister objects
        """
        from r65.compiler.mir.nodes import VirtualRegister
        vregs = set()

        # Check all attributes of instruction for VirtualRegisters
        for attr_name in dir(instr):
            if attr_name.startswith('_'):
                continue

            attr = getattr(instr, attr_name)

            if isinstance(attr, VirtualRegister):
                vregs.add(attr)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, VirtualRegister):
                        vregs.add(item)

        return vregs

    # ========================================================================
    # Scratch Pool Creation
    # ========================================================================

    def _create_scratch_pool(self, mir_program) -> ScratchRegisterPool:
        """
        Create scratch register pool from user-defined register variables.

        Scans static variables for those marked with register=true attribute.
        Memory management is the programmer's responsibility - the compiler
        only uses scratch registers explicitly defined by the programmer.

        Args:
            mir_program: MIR program containing static variable declarations

        Returns:
            ScratchRegisterPool populated with user-defined registers
        """
        pool = ScratchRegisterPool()

        # Scan static variables for register-marked variables
        for static_var in mir_program.statics:
            if hasattr(static_var, 'storage_attr') and static_var.storage_attr:
                storage_attr = static_var.storage_attr

                # Only use variables explicitly marked as registers
                if storage_attr.is_register:
                    # Get the variable's address from memory allocator
                    alloc = self.mem_alloc.get_allocation(static_var.symbol)
                    if alloc:
                        # Determine size from type
                        size = self._get_variable_size(static_var.var_type)

                        # Add to scratch pool
                        pool.add_scratch(
                            address=alloc.address,
                            size=size,
                            name=static_var.name
                        )

        return pool

    def _get_variable_size(self, type_info) -> int:
        """Get size of variable in bytes from type info."""
        return get_type_size(type_info)

    # ========================================================================
    # Prologue/Epilogue
    # ========================================================================

    def _emit_mode_directives(self, mir_func: MIRFunction):
        """
        Emit WLA-DX mode directives to inform the assembler of the expected processor mode.

        These directives (.ACCU and .INDEX) tell WLA-DX what size the accumulator and
        index registers are, so it can assemble instructions correctly. They don't
        emit any code - they're just for the assembler.

        Args:
            mir_func: MIR function
        """
        if mir_func.mode_attr:
            from r65.compiler.hir.attributes import MMode, XMode

            # Emit accumulator mode directive
            if mir_func.mode_attr.m_mode == MMode.M16:
                self.emitter.emit_directive("    .ACCU 16")
            elif mir_func.mode_attr.m_mode == MMode.M8:
                self.emitter.emit_directive("    .ACCU 8")

            # Emit index mode directive
            if mir_func.mode_attr.x_mode == XMode.X16:
                self.emitter.emit_directive("    .INDEX 16")
            elif mir_func.mode_attr.x_mode == XMode.X8:
                self.emitter.emit_directive("    .INDEX 8")

    def emit_prologue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
        """
        Emit function prologue.

        Prologue may include:
        - Stack pointer initialization (entry functions with custom stack)
        - Stack frame setup
        - Register preservation
        - Mode transitions
        - DBR management (data_bank=inline)

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
        """
        # Initialize stack pointer for entry functions with custom stack region
        if mir_func.is_entry and self.mem_alloc.stack_upper is not None:
            if self.mem_alloc.stack_upper != DEFAULT_STACK_UPPER:
                stack_addr = self.mem_alloc.stack_upper
                self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20), "16-bit A for stack setup")
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(stack_addr), "Stack top")
                self._emit_instr(Opcode.TCS, comment="Set stack pointer")
                self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(0x20), "Restore 8-bit A")

        # Handle DBR management for far functions with data_bank=inline
        if mir_func.is_far and mir_func.bank_attr:
            from r65.compiler.hir.attributes import DataBankMode

            if mir_func.bank_attr.data_bank == DataBankMode.INLINE:
                # Save current DBR and set to function's bank
                # Sequence: PHB, LDA #bank, PHA, PLB
                self._emit_instr(Opcode.PHB, comment="Save current data bank")
                self._emit_instr(Opcode.LDA_IMMEDIATE, Immediate(mir_func.bank_attr.bank_number),
                                "Load function's bank number")
                self._emit_instr(Opcode.PHA, comment="Push bank number")
                self._emit_instr(Opcode.PLB, comment="Set data bank register")

        # Handle processor mode transitions with transition=inline
        if mir_func.mode_attr:
            from r65.compiler.hir.attributes import ModeTransition, MMode, XMode

            if mir_func.mode_attr.transition == ModeTransition.INLINE:
                # Save current processor status and set required mode
                # Sequence: PHP, REP/SEP #bits, body, PLP, RTS
                self._emit_instr(Opcode.PHP, comment="Save processor status")

                # Determine which bits to set/clear based on mode
                # STATUS register bits: NV-BDIZC (- is unused, M is bit 5, X is bit 4)
                # Bit 5 (0x20): M flag (0=16-bit accumulator, 1=8-bit accumulator)
                # Bit 4 (0x10): X flag (0=16-bit index, 1=8-bit index)
                bits_to_clear = 0  # REP (Reset bits)
                bits_to_set = 0    # SEP (Set bits)

                # Determine M mode
                if mir_func.mode_attr.m_mode == MMode.M16:
                    bits_to_clear |= 0x20  # Clear M bit for 16-bit accumulator
                elif mir_func.mode_attr.m_mode == MMode.M8:
                    bits_to_set |= 0x20    # Set M bit for 8-bit accumulator

                # Determine X mode
                if mir_func.mode_attr.x_mode == XMode.X16:
                    bits_to_clear |= 0x10  # Clear X bit for 16-bit index
                elif mir_func.mode_attr.x_mode == XMode.X8:
                    bits_to_set |= 0x10    # Set X bit for 8-bit index

                # Emit REP and/or SEP instructions
                if bits_to_clear:
                    mode_comment = f"Set mode: {'m16 ' if bits_to_clear & 0x20 else ''}{'x16' if bits_to_clear & 0x10 else ''}".strip()
                    self._emit_instr(Opcode.REP_IMMEDIATE, Immediate(bits_to_clear), mode_comment)
                if bits_to_set:
                    mode_comment = f"Set mode: {'m8 ' if bits_to_set & 0x20 else ''}{'x8' if bits_to_set & 0x10 else ''}".strip()
                    self._emit_instr(Opcode.SEP_IMMEDIATE, Immediate(bits_to_set), mode_comment)

        # Emit register saves for #[preserves(...)]
        # Registers are pushed in order: STATUS, A, X, Y, D, DBR
        # (Popped in reverse order in epilogue/select_return)
        if mir_func.preserves_attr:
            preserved_regs = mir_func.preserves_attr.registers

            # Push in defined order (reverse order for pop)
            push_order = ['STATUS', 'A', 'X', 'Y', 'D', 'DBR']
            for reg in push_order:
                if reg in preserved_regs:
                    push_opcode = RegisterMappings.PUSH_OPCODES.get(reg)
                    if push_opcode:
                        self._emit_instr(push_opcode, comment=f"Preserve {reg}")

        # Save register-bound parameters that would be clobbered by stack loads
        # (Stack parameter loads use LDA, which clobbers A register)
        saved_a = self._emit_register_parameter_saves(mir_func, reg_alloc)

        # Emit stack parameter loads (must be after all prologue pushes)
        # Pass saved_a so stack offsets can be adjusted for the extra PHA
        self._emit_stack_parameter_loads(mir_func, reg_alloc, extra_stack_bytes=1 if saved_a else 0)

        # Restore A register parameter after stack loads
        self._emit_register_parameter_restores(saved_a)

    def _has_a_register_parameter(self, mir_func: MIRFunction) -> bool:
        """Check if function has a parameter bound to the A register."""
        from r65.compiler.hir.nodes import RegisterBinding

        for param in mir_func.parameters:
            if param.binding and isinstance(param.binding, RegisterBinding):
                if param.binding.register_name == 'A':
                    return True
        return False

    def _emit_register_parameter_saves(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
        """
        Save A register if it's a parameter and will be clobbered by stack loads.

        This pushes A before stack parameter loads. The corresponding PLA
        is emitted by _emit_register_parameter_restores after stack loads.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator

        Returns:
            True if A was pushed and needs to be restored
        """
        # Only needed if there are stack parameters (which use LDA to load)
        # AND there's a register parameter using A
        if not mir_func.stack_param_offsets:
            return False

        if not self._has_a_register_parameter(mir_func):
            return False

        # A register parameter will be clobbered by stack loads - save it
        self.emitter.emit_blank_line()
        self.emitter.emit_comment("Save A register parameter before stack loads")
        self._emit_instr(Opcode.PHA, comment="Save A (register parameter)")
        return True

    def _emit_register_parameter_restores(self, saved_a: bool):
        """
        Restore A register after stack parameter loads.

        Args:
            saved_a: True if A was saved and needs to be restored
        """
        if saved_a:
            self._emit_instr(Opcode.PLA, comment="Restore A (register parameter)")

    def _get_prologue_stack_bytes(self, mir_func: MIRFunction) -> int:
        """
        Calculate bytes pushed by prologue that affect stack parameter offsets.

        The prologue may push registers for DBR management, mode transitions,
        and register preservation. These pushes change the stack pointer,
        so stack parameter offsets must be adjusted accordingly.

        Args:
            mir_func: MIR function

        Returns:
            Number of bytes pushed by prologue
        """
        bytes_pushed = 0

        # DBR management: PHB pushes 1 byte
        if mir_func.is_far and mir_func.bank_attr:
            from r65.compiler.hir.attributes import DataBankMode
            if mir_func.bank_attr.data_bank == DataBankMode.INLINE:
                bytes_pushed += 1

        # Mode transition: PHP pushes 1 byte
        if mir_func.mode_attr:
            from r65.compiler.hir.attributes import ModeTransition
            if mir_func.mode_attr.transition == ModeTransition.INLINE:
                bytes_pushed += 1

        # Register preservation pushes
        if mir_func.preserves_attr:
            from r65.compiler.hir.attributes import MMode, XMode
            for reg in mir_func.preserves_attr.registers:
                if reg == 'STATUS':
                    bytes_pushed += 1  # PHP pushes 1 byte
                elif reg == 'A':
                    # PHA pushes 1 or 2 bytes depending on M mode
                    if mir_func.mode_attr and mir_func.mode_attr.m_mode == MMode.M16:
                        bytes_pushed += 2
                    else:
                        bytes_pushed += 1
                elif reg in ('X', 'Y'):
                    # PHX/PHY pushes 1 or 2 bytes depending on X mode
                    if mir_func.mode_attr and mir_func.mode_attr.x_mode == XMode.X16:
                        bytes_pushed += 2
                    else:
                        bytes_pushed += 1
                elif reg == 'D':
                    bytes_pushed += 2  # Direct page is always 16-bit
                elif reg == 'DBR':
                    bytes_pushed += 1  # Data bank is always 8-bit

        return bytes_pushed

    def _emit_stack_parameter_loads(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator,
                                     extra_stack_bytes: int = 0):
        """
        Emit code to load stack parameters into their allocated locations.

        Uses stack-relative addressing (LDA offset,S) to load parameters
        from the caller's stack frame into virtual register locations.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
            extra_stack_bytes: Additional bytes on stack (e.g., from PHA to save A register param)
        """
        if not mir_func.stack_param_offsets:
            return

        # Calculate adjustment for bytes pushed by prologue
        prologue_bytes = self._get_prologue_stack_bytes(mir_func) + extra_stack_bytes

        self.emitter.emit_blank_line()
        self.emitter.emit_comment("Load stack parameters")

        for param_idx in sorted(mir_func.stack_param_offsets.keys()):
            base_offset = mir_func.stack_param_offsets[param_idx]
            vreg = mir_func.param_to_vreg.get(param_idx)
            if not vreg:
                continue

            # Adjust offset for prologue pushes
            adjusted_offset = base_offset + prologue_bytes

            # Get physical location for this vreg
            phys_loc = reg_alloc.get_location(vreg)

            # Get parameter size
            param_size = get_type_size(vreg.type_info)

            # Emit load from stack to physical location
            if param_size == 1:
                # 8-bit load: LDA offset,S ; STA dest
                self._emit_instr(Opcode.LDA_STACK, StackOffset(adjusted_offset),
                                f"Load param {param_idx}")
                self._emit_store_to_location(phys_loc)
            elif param_size == 3:
                # 24-bit load (far pointer): copy all 3 bytes
                self._emit_instr(Opcode.LDA_STACK, StackOffset(adjusted_offset),
                                f"Load param {param_idx} low")
                self._emit_store_to_location(phys_loc)
                self._emit_instr(Opcode.LDA_STACK, StackOffset(adjusted_offset + 1),
                                f"Load param {param_idx} high")
                self._emit_store_to_location(self._offset_location(phys_loc, 1))
                self._emit_instr(Opcode.LDA_STACK, StackOffset(adjusted_offset + 2),
                                f"Load param {param_idx} bank")
                self._emit_store_to_location(self._offset_location(phys_loc, 2))
            else:
                # 16-bit load: LDA offset,S ; STA dest ; LDA offset+1,S ; STA dest+1
                # For simplicity, use byte-by-byte approach
                self._emit_instr(Opcode.LDA_STACK, StackOffset(adjusted_offset),
                                f"Load param {param_idx} low")
                self._emit_store_to_location(phys_loc)
                self._emit_instr(Opcode.LDA_STACK, StackOffset(adjusted_offset + 1),
                                f"Load param {param_idx} high")
                self._emit_store_to_location(self._offset_location(phys_loc, 1))

    def _emit_store_to_location(self, location):
        """Emit STA instruction to physical location."""
        from r65.compiler.codegen.register_alloc import LocationKind

        if location.kind == LocationKind.SCRATCH:
            self._emit_instr(Opcode.STA_DP, Address(location.scratch_addr))
        elif location.kind == LocationKind.MEMORY:
            if location.memory_addr < 0x100:
                self._emit_instr(Opcode.STA_DP, Address(location.memory_addr))
            else:
                self._emit_instr(Opcode.STA_ABSOLUTE, Address(location.memory_addr))
        elif location.kind == LocationKind.STACK:
            self._emit_instr(Opcode.STA_STACK, StackOffset(location.stack_offset))
        else:
            raise ValueError(f"Cannot store to location kind: {location.kind}")

    def _offset_location(self, location, offset: int):
        """Create new location offset from given location."""
        from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind

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
            raise ValueError(f"Cannot offset location kind: {location.kind}")

    def emit_epilogue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
        """
        Emit function epilogue.

        Epilogue includes (in order):
        1. Preserved register restoration (reverse of prologue push order)
        2. DBR restoration (for data_bank=inline)
        3. Mode restoration (for transition=inline)

        Note: Return value loading and RTS/RTL are handled separately
        by the return instruction.

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
        """
        self._emit_preserved_register_restores(mir_func)
        self._emit_dbr_restore(mir_func)
        self._emit_mode_restore(mir_func)

    def _emit_preserved_register_restores(self, mir_func: MIRFunction):
        """
        Restore preserved registers in reverse order of prologue pushes.

        Prologue pushes: STATUS, A, X, Y, D, DBR
        Epilogue pops:   DBR, D, Y, X, A, STATUS
        """
        if not mir_func.preserves_attr:
            return

        preserved_regs = mir_func.preserves_attr.registers
        pop_order = ['DBR', 'D', 'Y', 'X', 'A', 'STATUS']

        for reg in pop_order:
            if reg in preserved_regs:
                pull_opcode = RegisterMappings.PULL_OPCODES.get(reg)
                if pull_opcode:
                    self._emit_instr(pull_opcode, comment=f"Restore {reg}")

    def _emit_dbr_restore(self, mir_func: MIRFunction):
        """Restore DBR for data_bank=inline functions."""
        if not (mir_func.is_far and mir_func.bank_attr):
            return

        from r65.compiler.hir.attributes import DataBankMode
        if mir_func.bank_attr.data_bank == DataBankMode.INLINE:
            self._emit_instr(Opcode.PLB, comment="Restore data bank")

    def _emit_mode_restore(self, mir_func: MIRFunction):
        """Restore processor mode for transition=inline functions."""
        if not mir_func.mode_attr:
            return

        from r65.compiler.hir.attributes import ModeTransition
        if mir_func.mode_attr.transition == ModeTransition.INLINE:
            self._emit_instr(Opcode.PLP, comment="Restore processor status")


class ProgramFunctionGenerator:
    """
    Generates all functions in a program.

    Orchestrates function generation for entire MIR program.
    """

    def __init__(self,
                 emitter: AssemblyEmitter,
                 memory_allocator: MemoryAllocator):
        """
        Initialize program function generator.

        Args:
            emitter: Assembly emitter
            memory_allocator: Memory allocator
        """
        self.emitter = emitter
        self.mem_alloc = memory_allocator
        self.func_gen = FunctionCodeGenerator(emitter, memory_allocator)

    def generate_all_functions(self, mir_program):
        """
        Generate all functions in MIR program.

        Args:
            mir_program: MIRProgram to generate
        """
        # Create scratch pool from user-defined register variables
        scratch_pool = self.func_gen._create_scratch_pool(mir_program)

        # Emit section header
        self.emitter.emit_section_header("Functions")

        # Generate each function with the same scratch pool
        for mir_func in mir_program.functions:
            self.func_gen.generate_function(mir_func, scratch_pool=scratch_pool)

        # Blank line after all functions
        self.emitter.emit_blank_line()

    def generate_initialization_function(self, mir_program):
        """
        Generate __init_start function if needed.

        Args:
            mir_program: MIR program
        """
        # Check if __init_start exists in functions
        init_func = None
        for func in mir_program.functions:
            if func.name == "__init_start":
                init_func = func
                break

        if init_func:
            self.func_gen.generate_function(init_func)
