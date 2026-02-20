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
from r65.compiler.codegen.debug_info import DebugInfoCollector
from r65.compiler.codegen.debug_writer import Cc65DebugWriter
from r65.compiler.codegen.address_calc import AddressCalculator
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
        self.debug_info: Optional[DebugInfoCollector] = None

    def generate(self, mir_program: MIRProgram, output_file: Optional[str] = None,
                 opt_level: int = 1, debug: bool = False,
                 disable_scratch_params: bool = False,
                 disable_loop_promotion: bool = False,
                 abi_model=None) -> str:
        """
        Generate WLA-DX assembly from MIR program.

        Args:
            mir_program: MIR program to compile
            output_file: Optional output file path
            opt_level: Optimization level (0=none, 1=basic, 2=with implicit inlining)
            debug: Generate Mesen-compatible .dbg file (default False)
            abi_model: ABIModel instance (default: ABI_DEFAULT)

        Returns:
            Generated assembly as string
        """
        from r65.compiler.codegen.abi_model import ABI_DEFAULT
        if abi_model is None:
            abi_model = ABI_DEFAULT
        self.abi_model = abi_model
        # Initialize debug info collector if debug mode enabled
        if debug:
            self.debug_info = DebugInfoCollector()

        # Create emitter
        self.emitter = AssemblyEmitter(source_file="<unknown>")

        # Run optimizations only if enabled (-O1 or higher)
        if opt_level >= 1:
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

            # Far-to-near call optimization - convert far calls to near when
            # call graph shows all callers and callee are in the same bank
            # This saves 1 byte and 2 cycles per call (JSR vs JSL)
            # Run before inlining so converted functions can become inlinable
            from r65.compiler.optimize.far_to_near import FarToNearOptimizer
            far_to_near = FarToNearOptimizer(verbose=False)
            far_converted = far_to_near.optimize(mir_program)
            if far_converted > 0:
                print(f"Far-to-near optimization: {far_converted} function(s) converted")

            # Function inlining - replace call sites with inlined function bodies
            # At -O1: only explicit inlining (#[inline] or #[inline(always)])
            # At -O2: also implicit inlining (called-once and small functions)
            implicit_inline = (opt_level >= 2)
            func_inliner = FunctionInliner(verbose=False, implicit_inline=implicit_inline)
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

        # Emit memory map with calculated bank count and ROM type
        rom_type = "hirom" if is_hirom else "lorom"
        self.emitter.emit_memory_map(rom_type=rom_type, banks=rom_banks)

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

        # Collect memory allocation warnings
        for w in self.allocator.warnings:
            print(w)
        self.warnings.extend(self.allocator.warnings)

        # Phase 2: Symbol definitions
        symbol_gen = SymbolDefinitionGenerator(self.emitter, self.allocator)
        symbol_gen.emit_all_definitions(mir_program.constants)

        # Phase 3-6: Function code generation
        # (Phases 3-6 are integrated within function generation)
        self.func_gen = ProgramFunctionGenerator(self.emitter, self.allocator)

        # Create scratch pool once for all functions
        scratch_pool = self.func_gen.func_gen._create_scratch_pool(mir_program)

        # Run parameter promotion analysis (before loop promotion)
        # FixedStack mode: mandatory promotion of ALL stack params to hw regs / scratch
        # Default mode: optional scratch promotion for eligible params
        if abi_model.requires_mandatory_param_promotion():
            from r65.compiler.analysis.fixedstack_params import promote_all_stack_params
            promote_all_stack_params(mir_program, scratch_pool)
        elif not disable_scratch_params:
            from r65.compiler.analysis.scratch_params import analyze_scratch_params
            analyze_scratch_params(mir_program, scratch_pool)

        # Run loop register promotion analysis (after scratch params)
        if not disable_loop_promotion:
            from r65.compiler.analysis.loop_register_promotion import analyze_loop_promotion
            analyze_loop_promotion(mir_program)

        # Compute max outgoing arg bytes for caller-owned outgoing args convention
        # FixedStack mode: no outgoing arg area (all params in regs/scratch)
        if not abi_model.requires_mandatory_param_promotion():
            self._compute_outgoing_arg_bytes(mir_program)

        # Generate code for each bank
        for bank_num in sorted(functions_by_bank.keys()):
            bank_functions = functions_by_bank[bank_num]

            # Always emit bank directive to ensure correct bank after ROM data sections
            # ROM data is emitted before code by symbol_gen and can leave assembler in different bank
            self.emitter.emit_section_header(f"Bank {bank_num}")
            self.emitter.emit_bank_directive(bank_num)

            # Generate functions in this bank
            for mir_func in bank_functions:
                self.func_gen.func_gen.generate_function(
                    mir_func,
                    scratch_pool=scratch_pool,
                    abi_model=abi_model
                )

        # Phase 6.5: Trait dispatch tables (jump tables and wrapper functions)
        self._emit_trait_dispatch_tables(mir_program)

        # Stack depth analysis (post-codegen, uses codegen_frame_size/codegen_prologue_bytes)
        self._analyze_stack_depth(mir_program)

        # Phase 7: ROM data sections (for array literal initialization)
        self._emit_rom_data_sections(mir_program)

        # Phase 8: Interrupt vectors
        self._emit_interrupt_vectors(mir_program)

        # Symbol exports (for debugging and linking)
        self._emit_symbol_exports(mir_program)

        # Get structured nodes from emitter
        nodes = self.emitter.get_nodes()

        # Collect volatile register names and addresses (from #[hw] attributes) for peephole optimizer
        volatile_names, volatile_addresses = self._collect_volatile_registers(mir_program)

        # Apply node-based peephole optimizations (only if opt_level >= 1)
        if opt_level >= 1:
            optimized_nodes, num_optimizations = optimize_nodes(nodes, volatile_names, volatile_addresses)
            if num_optimizations > 0:
                print(f"Peephole optimizer: {num_optimizations} optimization(s) applied")
        else:
            optimized_nodes = nodes

        # Apply node-based long branch fixup (always required for correct assembly)
        # Code generator emits BRA by default; fixup converts to JMP when target > 127 bytes
        final_nodes, num_branch_fixups = fixup_nodes(optimized_nodes)

        if num_branch_fixups > 0:
            print(f"Branch fixup: {num_branch_fixups} long branch(es) fixed")

        # Validate bank sizes
        self._validate_bank_sizes(final_nodes, mir_program)

        # Convert nodes to assembly string
        assembly = emit_nodes(final_nodes)

        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(assembly)

        # Generate debug file if requested
        if debug and output_file:
            self._generate_debug_file(mir_program, final_nodes, output_file)

        return assembly

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _emit_trait_dispatch_tables(self, mir_program: MIRProgram):
        """
        Emit trait dispatch wrapper functions and jump tables.

        For each trait method, generates:
        1. A dispatch wrapper function that loads TypeId from self_ptr and
           indexes into a jump table
        2. A jump table with word-size entries pointing to concrete implementations

        Near dispatch wrapper:
            TraitName__method__dispatch:
              LDY #$0000
              LDA ($03,S),Y        ; Load TypeId from *self_ptr (offset 0)
              REP #$20
              AND #$00FF           ; Zero-extend to 16-bit
              ASL A                ; x2 for word table
              TAX
              SEP #$20
              JMP (TraitName__method__table,X)

        Far dispatch uses JML trampolines instead of a jump table.
        """
        if not hasattr(mir_program, 'trait_dispatch_info') or not mir_program.trait_dispatch_info:
            return

        self.emitter.emit_section_header("Trait Dispatch Tables")

        for trait_name, trait_info in mir_program.trait_dispatch_info.items():
            is_far = trait_info.get('is_far', False)
            methods = trait_info.get('methods', [])
            implementors = trait_info.get('implementors', [])

            if not implementors:
                continue

            # Get max TypeId to size the table
            max_type_id = max(impl['type_id'] for impl in implementors)

            for method_idx, method_name in enumerate(methods):
                dispatch_label = f"{trait_name}__{method_name}__dispatch"
                table_label = f"{trait_name}__{method_name}__table"

                if is_far:
                    self._emit_far_dispatch(
                        dispatch_label, trait_name, method_name,
                        method_idx, implementors, max_type_id
                    )
                else:
                    self._emit_near_dispatch(
                        dispatch_label, table_label, trait_name, method_name,
                        method_idx, implementors, max_type_id
                    )

        # Emit _trait_error handler (STP to halt processor on invalid TypeId)
        self.emitter.emit_label("_trait_error")
        self.emitter.emit_raw("    STP")
        self.emitter.emit_blank_line()

    def _emit_near_dispatch(self, dispatch_label, table_label, trait_name, method_name,
                            method_idx, implementors, max_type_id):
        """Emit near dispatch wrapper and jump table for a trait method."""
        # Dispatch wrapper function
        self.emitter.emit_label(dispatch_label)

        # Self pointer is in Y register (DBR:Y addressing)
        # Load TypeId from offset 0 of the trait object: LDA abs,Y with abs=0
        self.emitter.emit_raw("    LDA $0000,Y")            # Load TypeId byte from DBR:Y+0

        # Zero-extend to 16-bit and compute table index
        self.emitter.emit_raw("    REP #$20")               # Switch to m16
        self.emitter.emit_raw("    AND #$00FF")              # Zero-extend
        self.emitter.emit_raw("    ASL A")                   # x2 for word table entries
        self.emitter.emit_raw("    TAX")
        self.emitter.emit_raw("    SEP #$20")                # Back to m8
        self.emitter.emit_raw(f"    JMP ({table_label},X)")
        self.emitter.emit_blank_line()

        # Jump table
        self.emitter.emit_label(table_label)

        # Build table: entry for each TypeId from 0 to max_type_id
        # TypeId 0 is invalid (unused), points to error handler
        type_id_to_impl = {}
        for impl in implementors:
            type_id_to_impl[impl['type_id']] = impl['mangled'][method_idx]

        for tid in range(max_type_id + 1):
            if tid in type_id_to_impl:
                self.emitter.emit_raw(f"    .dw {type_id_to_impl[tid]}")
            else:
                self.emitter.emit_raw("    .dw _trait_error")

        self.emitter.emit_blank_line()

    def _emit_far_dispatch(self, dispatch_label, trait_name, method_name,
                           method_idx, implementors, max_type_id):
        """Emit far dispatch using JMP (table,X) to JML trampolines.

        Strategy: use indirect indexed JMP to a word table, where each entry
        points to a JML stub that long-jumps to the concrete implementation.
        The caller's JSL return address stays on the stack, so the RTL in the
        concrete method returns directly to the original caller.

        Layout:
            dispatch:
                LDA TypeId; ASL (x2 for word table); JMP (table,X)
            table:
                .dw jml_stub_0, jml_stub_1, ...
            jml_stub_N:
                JML ConcreteImpl_N
        """
        table_label = f"{trait_name}__{method_name}__table"
        trampoline_label = f"{trait_name}__{method_name}__trampoline"

        self.emitter.emit_label(dispatch_label)

        # Self pointer is in Y register, DBR set to object's bank by caller
        # Load TypeId from offset 0: LDA abs,Y with abs=0
        self.emitter.emit_raw("    LDA $0000,Y")            # Load TypeId byte from DBR:Y+0

        # Compute table offset: TypeId * 2 (word table entries)
        self.emitter.emit_raw("    REP #$20")               # Switch to m16
        self.emitter.emit_raw("    AND #$00FF")              # Zero-extend
        self.emitter.emit_raw("    ASL A")                   # x2 for word table
        self.emitter.emit_raw("    TAX")
        self.emitter.emit_raw("    SEP #$20")                # Back to m8
        self.emitter.emit_raw(f"    JMP ({table_label},X)")
        self.emitter.emit_blank_line()

        # Word table: addresses of JML stubs
        self.emitter.emit_label(table_label)

        type_id_to_impl = {}
        for impl in implementors:
            type_id_to_impl[impl['type_id']] = impl['mangled'][method_idx]

        for tid in range(max_type_id + 1):
            jml_label = f"{trampoline_label}_{tid}"
            self.emitter.emit_raw(f"    .dw {jml_label}")

        self.emitter.emit_blank_line()

        # JML stubs: each is a long jump to the concrete implementation
        for tid in range(max_type_id + 1):
            jml_label = f"{trampoline_label}_{tid}"
            self.emitter.emit_label(jml_label)
            if tid in type_id_to_impl:
                self.emitter.emit_raw(f"    JML {type_id_to_impl[tid]}")
            else:
                self.emitter.emit_raw("    JML _trait_error")

        self.emitter.emit_blank_line()

    def _analyze_stack_depth(self, mir_program: MIRProgram):
        """
        Run stack depth analysis and collect warnings.

        Uses codegen-populated frame_size and prologue_bytes on each
        MIRFunction to compute worst-case stack depth across all call
        paths from entry points and interrupt handlers.
        """
        from r65.compiler.analysis.stack_depth import StackDepthAnalyzer

        analyzer = StackDepthAnalyzer(
            mir_program,
            self.allocator.stack_lower,
            self.allocator.stack_upper,
        )
        warnings = analyzer.analyze()
        for w in warnings:
            print(w)
        self.warnings.extend(warnings)

    def _collect_volatile_registers(self, mir_program: MIRProgram) -> tuple:
        """
        Collect volatile register names and addresses from #[hw] statics.

        Variables with #[hw] attribute are volatile - stores to them have
        side effects (I/O operations) and must never be eliminated by the
        peephole optimizer, even if the value appears to be overwritten.

        Args:
            mir_program: MIR program containing static declarations

        Returns:
            Tuple of (set of names, set of addresses) for volatile registers
        """
        from r65.compiler.hir.attributes import StorageKind

        volatile_names = set()
        volatile_addresses = set()

        for static in mir_program.statics:
            if hasattr(static, 'storage_attr') and static.storage_attr:
                if static.storage_attr.storage_kind == StorageKind.HW:
                    volatile_names.add(static.name)
                    if static.storage_attr.address is not None:
                        volatile_addresses.add(static.storage_attr.address)

        return volatile_names, volatile_addresses

    def _compute_outgoing_arg_bytes(self, mir_program: MIRProgram):
        """
        Compute max outgoing stack argument bytes for each function.

        For the caller-owned outgoing args convention, each function reserves
        space at the bottom of its frame for the largest set of stack arguments
        across all call sites. The caller writes args via STA d,S instead of PHA.
        """
        from r65.compiler.mir.nodes import Call, TraitDispatch, ArgumentMechanism
        from r65.compiler.codegen.type_utils import get_type_size

        for func in mir_program.functions:
            max_bytes = 0
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, (Call, TraitDispatch)):
                        call_bytes = 0
                        for arg in instr.args:
                            if arg.mechanism == ArgumentMechanism.STACK:
                                if arg.param_type is not None:
                                    call_bytes += get_type_size(arg.param_type)
                                elif hasattr(arg.value, 'type_info') and arg.value.type_info:
                                    call_bytes += get_type_size(arg.value.type_info)
                                else:
                                    call_bytes += 1
                        max_bytes = max(max_bytes, call_bytes)
            func.max_outgoing_arg_bytes = max_bytes

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

    # ========================================================================
    # Debug File Generation
    # ========================================================================

    def _generate_debug_file(self, mir_program: MIRProgram, nodes: list, output_file: str):
        """
        Generate Mesen-compatible .dbg file alongside assembly output.

        Collects debug information including:
        - Source files referenced
        - Memory segments (banks)
        - Symbols (functions, constants, statics)
        - Scopes (function boundaries)
        - Line mappings (source line to address)

        Args:
            mir_program: MIR program with source info
            nodes: Final assembly nodes
            output_file: Assembly output path (used to derive .dbg path)
        """
        from r65.compiler.codegen.asm_nodes import Label, Instruction, Directive

        # Determine ROM type for address calculation
        is_hirom = False
        if mir_program.snesrom_config:
            is_hirom = mir_program.snesrom_config.hirom or mir_program.snesrom_config.exhirom

        # Calculate base address for LoROM/HiROM
        base_address = 0xC000 if is_hirom else 0x8000

        # Collect segments (one per bank)
        self._collect_segments(mir_program, nodes, base_address, is_hirom)

        # Collect symbols (functions, constants, statics)
        self._collect_symbols(mir_program, nodes, base_address)

        # Collect scopes (function boundaries)
        self._collect_scopes(mir_program, nodes, base_address)

        # Collect line mappings
        self._collect_line_info(nodes, base_address)

        # Write debug file
        dbg_path = output_file.replace('.asm', '.dbg')
        if dbg_path == output_file:
            dbg_path = output_file + '.dbg'

        writer = Cc65DebugWriter(self.debug_info)
        writer.write_to_file(dbg_path)
        print(f"Debug info written to: {dbg_path}")

    def _collect_segments(self, mir_program: MIRProgram, nodes: list, base_address: int, is_hirom: bool):
        """
        Collect segment information for debug output.

        Creates a segment entry for each bank used in the program.

        Args:
            mir_program: MIR program
            nodes: Assembly nodes
            base_address: Base address (0x8000 for LoROM, 0xC000 for HiROM)
            is_hirom: Whether using HiROM mapping
        """
        # Organize functions by bank to determine which banks are used
        functions_by_bank = self._organize_functions_by_bank(mir_program.functions)

        # Bank size in bytes
        bank_size = 0x10000 if is_hirom else 0x8000

        # Create segment for each bank
        for bank_num in sorted(functions_by_bank.keys()):
            # Calculate start address for this bank
            start = base_address

            # Output file offset (bank * bank_size for LoROM)
            ooffs = bank_num * bank_size

            self.debug_info.add_segment(
                name=f"BANK{bank_num}",
                start=start,
                size=bank_size,
                bank=bank_num,
                ooffs=ooffs,
                seg_type="ro"
            )

    def _collect_symbols(self, mir_program: MIRProgram, nodes: list, base_address: int):
        """
        Collect symbol information for debug output.

        Collects:
        - Function labels
        - Static variables
        - Constants

        Args:
            mir_program: MIR program
            nodes: Assembly nodes
            base_address: Base address for the segment
        """
        from r65.compiler.codegen.asm_nodes import Label

        # Calculate label addresses
        calc = AddressCalculator(base_address)
        _, label_addresses = calc.calculate_with_labels(nodes)

        # Add function symbols
        for func in mir_program.functions:
            # Get address from label
            addr = label_addresses.get(func.name, base_address)

            # Determine segment (bank)
            bank_num = 0
            if func.bank_attr and func.bank_attr.bank_number is not None:
                bank_num = func.bank_attr.bank_number

            # Find segment ID for this bank
            seg_id = None
            for seg in self.debug_info.segments:
                if seg.bank == bank_num:
                    seg_id = seg.id
                    break

            # Register source file
            file_id = None
            if func.source_loc and func.source_loc.file_path:
                dbg_file = self.debug_info.get_or_create_file(func.source_loc.file_path)
                file_id = dbg_file.id

            self.debug_info.add_symbol(
                name=func.name,
                value=addr,
                seg_id=seg_id,
                sym_type="lab"
            )

        # Add constant symbols
        for const in mir_program.constants:
            # Use evaluated_value if available (it's the const-evaluated integer)
            value = const.evaluated_value if const.evaluated_value is not None else 0
            if not isinstance(value, int):
                value = 0
            self.debug_info.add_symbol(
                name=const.name,
                value=value,
                seg_id=None,  # Constants are absolute
                sym_type="equ"
            )

        # Add static variable symbols
        if self.allocator:
            for name, alloc in self.allocator.allocations.items():
                # Skip if already added (some overlap with functions)
                if any(s.name == name for s in self.debug_info.symbols):
                    continue

                self.debug_info.add_symbol(
                    name=name,
                    value=alloc.address,
                    seg_id=None,  # RAM addresses are absolute
                    sym_type="lab",
                    size=alloc.size
                )

    def _collect_scopes(self, mir_program: MIRProgram, nodes: list, base_address: int):
        """
        Collect scope information for debug output.

        Creates scope entries for each function.

        Args:
            mir_program: MIR program
            nodes: Assembly nodes
            base_address: Base address
        """
        # First add a global scope
        global_scope = self.debug_info.add_scope(
            name="",
            scope_type="global",
            size=0,
            span_ids=[]
        )

        # Calculate label addresses to get function sizes
        calc = AddressCalculator(base_address)
        _, label_addresses = calc.calculate_with_labels(nodes)

        # Sort functions by address to calculate sizes
        func_addrs = []
        for func in mir_program.functions:
            addr = label_addresses.get(func.name, base_address)
            func_addrs.append((func, addr))
        func_addrs.sort(key=lambda x: x[1])

        # Add scope for each function
        for i, (func, addr) in enumerate(func_addrs):
            # Estimate size (distance to next function or end of bank)
            if i + 1 < len(func_addrs):
                size = func_addrs[i + 1][1] - addr
            else:
                size = 0x100  # Default size for last function

            # Find symbol ID for this function
            sym_id = None
            for sym in self.debug_info.symbols:
                if sym.name == func.name and sym.sym_type == "lab":
                    sym_id = sym.id
                    break

            self.debug_info.add_scope(
                name=func.name,
                scope_type="scope",
                size=size,
                span_ids=[],  # Will be filled by line info collection
                parent_id=global_scope.id,
                sym_id=sym_id
            )

    def _collect_line_info(self, nodes: list, base_address: int):
        """
        Collect source line to address mappings for debug output.

        Walks through assembly nodes and creates line entries for
        instructions that have source location information.

        Args:
            nodes: Assembly nodes
            base_address: Base address
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label

        # Calculate addresses for all nodes
        calc = AddressCalculator(base_address)
        node_addresses = calc.calculate(nodes)

        # Track current segment (assume bank 0 for now)
        current_seg_id = 0 if self.debug_info.segments else None

        # Group instructions by source line
        line_spans: Dict[tuple, List[tuple]] = {}  # (file_id, line) -> [(start, size), ...]

        for i, node in enumerate(nodes):
            if not isinstance(node, Instruction):
                continue

            if node.source_loc is None:
                continue

            if i not in node_addresses:
                continue

            addr, size = node_addresses[i]

            # Get or create file entry
            file_path = node.source_loc.file_path
            dbg_file = self.debug_info.get_or_create_file(file_path)

            # Group by (file_id, line)
            key = (dbg_file.id, node.source_loc.line)
            if key not in line_spans:
                line_spans[key] = []
            line_spans[key].append((addr - base_address, size))

        # Create spans and line entries
        for (file_id, line), spans in sorted(line_spans.items()):
            span_ids = []
            for start, size in spans:
                if current_seg_id is not None:
                    span = self.debug_info.find_or_add_span(current_seg_id, start, size)
                    span_ids.append(span.id)

            if span_ids:
                self.debug_info.add_line(
                    file_id=file_id,
                    line=line,
                    span_ids=span_ids,
                    line_type=0  # Assembly source
                )
