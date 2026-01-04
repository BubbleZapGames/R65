"""
Function code generation: MIR functions → assembly.

Generates complete function bodies with headers, labels, and instructions.
"""

from typing import List, Set, Dict, Optional
from r65.compiler.mir.nodes import MIRFunction, BasicBlock
from r65.compiler.codegen.emitter import *
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.register_alloc import *
from r65.compiler.codegen.memory_alloc import *


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
        instr_selector = InstructionSelector(self.emitter, reg_alloc, self.mem_alloc, mir_func)

        # Emit function header comment
        self.emit_function_header(mir_func)

        # Emit function label
        self.emitter.emit_label(mir_func.name)

        # Emit prologue (if needed)
        self.emit_prologue(mir_func, reg_alloc)

        # Generate basic blocks
        block_order = self._compute_block_order(mir_func)

        for block_id in block_order:
            block = mir_func.blocks[block_id]

            # Emit block label (except entry block which uses function label)
            if block_id != mir_func.entry_block_id:
                self.emitter.emit_label(f"__L{block_id}")

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
        # TODO: Add source location tracking
        # self.emitter.emit_comment(f"Source: {mir_func.source_file}:{start}-{end}")

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
        if hasattr(type_info, 'name'):
            type_name = type_info.name
            if type_name in ('u8', 'i8', 'bool'):
                return 1
            elif type_name in ('u16', 'i16'):
                return 2
        return 1  # Default to 1 byte

    # ========================================================================
    # Prologue/Epilogue (TODO)
    # ========================================================================

    def emit_prologue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
        """
        Emit function prologue.

        Prologue may include:
        - Stack frame setup
        - Register preservation
        - Mode transitions
        - DBR management (data_bank=auto)

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
        """
        # Handle DBR management for far functions with data_bank=auto
        if mir_func.is_far and mir_func.bank_attr:
            from r65.compiler.hir.attributes import DataBankMode

            if mir_func.bank_attr.data_bank == DataBankMode.AUTO:
                # Save current DBR and set to function's bank
                # Sequence: PHB, LDA #bank, PHA, PLB
                self.emitter.emit_instruction("PHB", comment="Save current data bank")
                self.emitter.emit_instruction("LDA", f"#${mir_func.bank_attr.bank_number:02X}",
                                            "Load function's bank number")
                self.emitter.emit_instruction("PHA", comment="Push bank number")
                self.emitter.emit_instruction("PLB", comment="Set data bank register")

        # TODO: Implement other prologue features
        # - Check if stack frame needed (spilled registers)
        # - Emit register saves (if preserves_attr)
        # - Emit mode transition (if mode_attr with transition=auto)

    def emit_epilogue(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
        """
        Emit function epilogue.

        Epilogue may include:
        - Stack frame teardown
        - Register restoration
        - Mode restoration

        Args:
            mir_func: MIR function
            reg_alloc: Register allocator
        """
        # TODO: Implement epilogue generation
        # - Emit mode restoration
        # - Emit register restores
        # - Stack cleanup
        # Note: RTS/RTL already emitted by Return instruction
        pass


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
