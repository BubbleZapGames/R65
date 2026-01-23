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
from r65.compiler.optimize.peephole import optimize_nodes
from r65.compiler.codegen.branch_fixup import fixup_nodes
from r65.compiler.codegen.emitter import emit_nodes
from r65.compiler.codegen.constants import DEFAULT_STACK_LOWER, DEFAULT_STACK_UPPER, calculate_rom_size
from r65.compiler.codegen.bank_size import validate_bank_sizes
# Note: Optimize imports are done lazily in generate() to avoid circular imports


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
        self.warnings: List[str] = []

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

        # Import optimize modules lazily to avoid circular imports
        from r65.compiler.optimize.dead_function_elim import DeadFunctionEliminator
        from r65.compiler.optimize.dead_code_elim import DeadCodeEliminator
        from r65.compiler.optimize.inline import FunctionInliner

        # Dead function elimination - remove functions that are never called
        dead_func_elim = DeadFunctionEliminator(verbose=False)
        func_eliminated = dead_func_elim.eliminate(mir_program)
        if func_eliminated > 0:
            print(f"Dead function elimination: {func_eliminated} function(s) removed")

        # Dead code elimination - remove unreachable blocks and dead stores
        dead_code_elim = DeadCodeEliminator(verbose=False)
        code_eliminated = dead_code_elim.eliminate(mir_program)
        if code_eliminated > 0:
            print(f"Dead code elimination: {code_eliminated} block(s)/instruction(s) removed")

        # Function inlining - replace call sites with inlined function bodies
        func_inliner = FunctionInliner(verbose=False)
        inlined_count = func_inliner.run(mir_program)
        if inlined_count > 0:
            print(f"Function inlining: {inlined_count} call site(s) inlined")

            # Re-run dead function elimination after inlining
            # (inlined functions may now be unused)
            dead_func_elim2 = DeadFunctionEliminator(verbose=False)
            func_eliminated2 = dead_func_elim2.eliminate(mir_program)
            if func_eliminated2 > 0:
                print(f"Post-inlining dead function elimination: {func_eliminated2} function(s) removed")

        # Emit file header and processor directives
        self.emitter.emit_file_header()
        self.emitter.emit_processor_directive()

        # Organize functions by bank first to determine bank count
        functions_by_bank = self._organize_functions_by_bank(mir_program.functions)
        max_bank = max(functions_by_bank.keys()) if functions_by_bank else 0
        bank_count = max_bank + 1  # Banks are 0-indexed

        # Determine ROM type from snesrom_config
        is_hirom = False
        if mir_program.snesrom_config:
            is_hirom = mir_program.snesrom_config.hirom or mir_program.snesrom_config.exhirom

        # Calculate minimum ROM size (power-of-2 banks and ROMSIZE value)
        # calculate_rom_size rounds up to power of 2 and enforces 256KB minimum
        rom_banks, self.romsize_value = calculate_rom_size(bank_count, is_hirom)

        # Emit memory map with calculated bank count
        self.emitter.emit_memory_map(banks=rom_banks)

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

        # Create scratch pool once for all functions
        scratch_pool = self.func_gen.func_gen._create_scratch_pool(mir_program)

        # Generate code for each bank
        for bank_num in sorted(functions_by_bank.keys()):
            bank_functions = functions_by_bank[bank_num]

            # Always emit bank directive to ensure correct bank after ROM data sections
            # ROM data is emitted before code by symbol_gen and can leave assembler in different bank
            self.emitter.emit_section_header(f"Bank {bank_num}")
            self.emitter.emit_bank_directive(bank_num)

            # Generate functions in this bank
            for mir_func in bank_functions:
                self.func_gen.func_gen.generate_function(mir_func, scratch_pool=scratch_pool)

        # Phase 7: ROM data sections (for array literal initialization)
        self._emit_rom_data_sections(mir_program)

        # Phase 8: Interrupt vectors
        self._emit_interrupt_vectors(mir_program)

        # Symbol exports (for debugging and linking)
        self._emit_symbol_exports(mir_program)

        # Get structured nodes from emitter
        nodes = self.emitter.get_nodes()

        # Apply node-based peephole optimizations
        optimized_nodes, num_optimizations = optimize_nodes(nodes)

        if num_optimizations > 0:
            print(f"Peephole optimizer: {num_optimizations} optimization(s) applied")

        # Apply node-based long branch fixup
        fixed_nodes, num_branch_fixups = fixup_nodes(optimized_nodes)

        if num_branch_fixups > 0:
            print(f"Branch fixup: {num_branch_fixups} long branch(es) fixed")

        # Validate bank sizes
        self._validate_bank_sizes(fixed_nodes, mir_program)

        # Convert nodes to assembly string
        assembly = emit_nodes(fixed_nodes)

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
        Functions in auto-bank mode (bank_number=None) are placed in bank 0 initially.
        Future enhancement: auto-bank functions will be placed in remaining space.

        Args:
            functions: List of MIR functions

        Returns:
            Dictionary mapping bank number to list of functions
        """
        by_bank: Dict[int, List[MIRFunction]] = {}

        for func in functions:
            # Determine bank number
            if func.bank_attr and func.bank_attr.bank_number is not None:
                bank_num = func.bank_attr.bank_number
            else:
                # Auto-bank mode or no bank attribute: default to bank 0
                # These are always far functions, so cross-bank calls work
                bank_num = 0

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

        # Check for missing NMI handler - this is a common oversight
        # Only warn if there's an entry point (i.e., this is a standalone program)
        if reset_handler and not nmi_handler:
            self.warnings.append(
                "No NMI interrupt handler defined. "
                "The NMI fires every VBlank (~60Hz) and is typically used for game logic updates. "
                "Add an NMI handler with: #[interrupt(nmi)] fn nmi_handler() { ... }"
            )

        # Emit SNES header and vectors if any handlers found
        if nmi_handler or irq_handler or reset_handler:
            # Emit empty interrupt handler for unused vectors
            self.emitter.emit_empty_interrupt_handler()

            # Emit SNES ROM header (use config from #[snesrom(...)] if present)
            self.emitter.emit_snes_header(
                snesrom_config=mir_program.snesrom_config,
                romsize_value=self.romsize_value
            )

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
                self.emitter.emit_directive(f".db {bytes_str}")

    def _emit_symbol_exports(self, mir_program: MIRProgram):
        """
        Emit symbol exports for linking with external modules.

        Note: In WLA-DX, .EXPORT is only for .DEFINE symbols, not labels.
        Function labels are automatically visible within single-file assembly.
        This method is currently a no-op but preserved for future multi-file
        linking support where we might export .DEFINE symbols.

        Args:
            mir_program: MIR program
        """
        # In WLA-DX, labels (function entry points) don't need .EXPORT
        # They are automatically visible within the assembly file.
        # .EXPORT is only for .DEFINE symbols used in multi-file linking.
        pass

    def _validate_bank_sizes(self, nodes: list, mir_program: MIRProgram):
        """
        Validate that code/data fits within bank size limits.

        Checks each bank against the appropriate size limit:
        - LoROM: 32KB per bank
        - HiROM: 64KB per bank

        The header bank (bank 0) has reduced capacity due to the SNES header.

        Args:
            nodes: Optimized assembly nodes
            mir_program: MIR program (for snesrom config)

        Raises:
            BankSizeError: If any bank exceeds its limit
        """
        # Determine ROM type from snesrom config
        is_hirom = False
        if mir_program.snesrom_config:
            is_hirom = mir_program.snesrom_config.hirom or mir_program.snesrom_config.exhirom

        # Check if we have a header (any entry point or interrupt handler)
        has_header = any(
            func.is_entry or func.interrupt_attr
            for func in mir_program.functions
        )

        # Validate bank sizes
        validate_bank_sizes(nodes, is_hirom=is_hirom, has_header=has_header)
