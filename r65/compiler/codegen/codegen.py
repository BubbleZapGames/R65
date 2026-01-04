"""
Program code generator: MIR → WLA-DX assembly.

Main code generation orchestrator that transforms a complete MIR program
into WLA-DX assembly output.
"""

from typing import Optional, Dict, List
from r65.compiler.mir import MIRProgram, MIRFunction
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.symbol_gen import SymbolDefinitionGenerator
from r65.compiler.codegen.function_gen import ProgramFunctionGenerator
from r65.compiler.codegen.peephole import optimize_assembly
from r65.compiler.codegen.constants import DEFAULT_STACK_LOWER, DEFAULT_STACK_UPPER


class ProgramCodeGenerator:
    """
    Generates complete assembly program from MIR.

    Orchestrates the entire code generation process:
    1. Memory allocation
    2. Symbol definitions
    3. Register allocation
    4. Instruction selection
    5. Assembly emission
    """

    def __init__(self):
        """Initialize code generator."""
        self.allocator: Optional[MemoryAllocator] = None
        self.emitter: Optional[AssemblyEmitter] = None
        self.func_gen: Optional[ProgramFunctionGenerator] = None

    def generate(self, mir_program: MIRProgram, output_file: Optional[str] = None) -> str:
        """
        Generate WLA-DX assembly from MIR program.

        Args:
            mir_program: MIR program to compile
            output_file: Optional output file path

        Returns:
            Generated assembly as string
        """
        # Create emitter
        self.emitter = AssemblyEmitter(source_file="<unknown>")

        # Emit file header and processor directives
        self.emitter.emit_file_header()
        self.emitter.emit_processor_directive()

        # Organize functions by bank first to determine bank count
        functions_by_bank = self._organize_functions_by_bank(mir_program.functions)
        max_bank = max(functions_by_bank.keys()) if functions_by_bank else 0
        bank_count = max_bank + 1  # Banks are 0-indexed

        # Emit memory map with correct bank count
        self.emitter.emit_memory_map(banks=bank_count)

        # Phase 1: Memory allocation
        self.allocator = MemoryAllocator()

        # Set stack region from global #[stack(...)] attribute or use default
        if mir_program.stack_attr:
            self.allocator.set_stack_region(
                mir_program.stack_attr.lower,
                mir_program.stack_attr.upper
            )
        else:
            # Default stack region: $0100-$01FF (256 bytes)
            self.allocator.set_stack_region(DEFAULT_STACK_LOWER, DEFAULT_STACK_UPPER)

        self.allocator.allocate_all(mir_program.statics)

        # Phase 2: Symbol definitions
        symbol_gen = SymbolDefinitionGenerator(self.emitter, self.allocator)
        symbol_gen.emit_all_definitions(mir_program.constants)

        # Phase 3-6: Function code generation
        # (Phases 3-6 are integrated within function generation)
        self.func_gen = ProgramFunctionGenerator(self.emitter, self.allocator)

        # Generate code for each bank
        for bank_num in sorted(functions_by_bank.keys()):
            bank_functions = functions_by_bank[bank_num]

            # Emit bank directive
            if bank_num > 0 or len(functions_by_bank) > 1:
                self.emitter.emit_section_header(f"Bank {bank_num}")
                self.emitter.emit_bank_directive(bank_num)

            # Generate functions in this bank
            for mir_func in bank_functions:
                self.func_gen.func_gen.generate_function(mir_func)

        # Phase 7: ROM data sections (for array literal initialization)
        self._emit_rom_data_sections(mir_program)

        # Phase 8: Interrupt vectors
        self._emit_interrupt_vectors(mir_program)

        # Symbol exports (for debugging and linking)
        self._emit_symbol_exports(mir_program)

        # Get assembly
        assembly = self.emitter.to_string()

        # Apply peephole optimizations
        assembly_lines = assembly.split('\n')
        optimized_lines, num_optimizations = optimize_assembly(assembly_lines)
        assembly = '\n'.join(optimized_lines)

        if num_optimizations > 0:
            print(f"Peephole optimizer: {num_optimizations} optimization(s) applied")

        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(assembly)

        return assembly

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _organize_functions_by_bank(self, functions: List[MIRFunction]) -> Dict[int, List[MIRFunction]]:
        """
        Organize functions by bank number.

        Functions with #[bank(n)] attribute go to bank n.
        Functions without bank attribute go to bank 0.

        Args:
            functions: List of MIR functions

        Returns:
            Dictionary mapping bank number to list of functions
        """
        by_bank: Dict[int, List[MIRFunction]] = {}

        for func in functions:
            # Determine bank number
            if func.bank_attr:
                bank_num = func.bank_attr.bank_number
            else:
                bank_num = 0  # Default to bank 0

            # Add to bank
            if bank_num not in by_bank:
                by_bank[bank_num] = []
            by_bank[bank_num].append(func)

        return by_bank

    def _emit_interrupt_vectors(self, mir_program: MIRProgram):
        """
        Emit SNES ROM header and interrupt vector table.

        Scans functions for #[interrupt(vector)] attributes and
        generates the SNES header and interrupt vector table.

        Args:
            mir_program: MIR program
        """
        # Find interrupt handlers
        nmi_handler = None
        irq_handler = None
        reset_handler = None

        for func in mir_program.functions:
            if func.interrupt_attr:
                # vector is an InterruptVector enum
                vector = func.interrupt_attr.vector.value.lower()

                if vector == 'nmi':
                    nmi_handler = func.name
                elif vector == 'irq':
                    irq_handler = func.name
                # Note: BRK, COP, ABORT not yet supported

            # Check for entry point (becomes reset vector)
            if func.is_entry:
                reset_handler = func.name

        # Emit SNES header and vectors if any handlers found
        if nmi_handler or irq_handler or reset_handler:
            # Emit empty interrupt handler for unused vectors
            self.emitter.emit_empty_interrupt_handler()

            # Emit SNES ROM header
            self.emitter.emit_snes_header(rom_name="R65 Compiled ROM", version=0)

            # Emit interrupt vectors
            self.emitter.emit_interrupt_vectors(
                nmi=nmi_handler,
                irq=irq_handler,
                reset=reset_handler
            )

    def _emit_rom_data_sections(self, mir_program: MIRProgram):
        """
        Emit ROM data sections for array literal initialization.

        Creates labeled data sections in ROM that will be copied to RAM
        during initialization using BlockCopy (MVN instruction).

        Args:
            mir_program: MIR program
        """
        if not mir_program.rom_data_sections:
            return

        self.emitter.emit_section_header("ROM Data Sections (array literal init data)")

        for rom_data in mir_program.rom_data_sections:
            # Emit label
            self.emitter.emit_label(rom_data.label)

            # Emit data bytes
            # Format as .db directives in groups of 16 bytes for readability
            data = rom_data.data
            for i in range(0, len(data), 16):
                chunk = data[i:i+16]
                bytes_str = ', '.join(f'${b:02X}' for b in chunk)
                self.emitter.emit_instruction(".db", bytes_str)

    def _emit_symbol_exports(self, mir_program: MIRProgram):
        """
        Emit symbol exports for debugging and linking.

        Exports all public functions and the entry point.

        Args:
            mir_program: MIR program
        """
        exports = []

        # Export all functions (for debugging)
        for func in mir_program.functions:
            exports.append(func.name)

        # Emit exports
        if exports:
            self.emitter.emit_exports(exports)
