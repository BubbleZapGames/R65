# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Program code generator: MIR → WLA-DX assembly.

Main code generation orchestrator that transforms a complete MIR program
into WLA-DX assembly output.
"""

import sys
from typing import Optional, Dict, List, Any
from r65.compiler.mir import MIRProgram, MIRFunction
from r65.compiler.mir.nodes import SymbolByte
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.symbol_gen import (
    SymbolDefinitionGenerator, auto_bank_number, rom_static_bank,
)
from r65.compiler.codegen.function_gen import ProgramFunctionGenerator
from r65.compiler.optimize.peephole import optimize_nodes
from r65.compiler.codegen.branch_fixup import fixup_nodes
from r65.compiler.codegen.bank_size import rom_bank_size
from r65.compiler.codegen.emitter import emit_nodes
from r65.compiler.codegen.constants import DEFAULT_STACK_LOWER, DEFAULT_STACK_UPPER, calculate_rom_size
from r65.compiler.codegen.debug_info import DebugInfoCollector
from r65.compiler.codegen.debug_writer import Cc65DebugWriter
from r65.compiler.codegen.address_calc import AddressCalculator
from r65.compiler.errors import CodegenError
# Note: Optimize imports are done lazily in generate() to avoid circular imports


def _acc_abi(mir_func):
    """How `mir_func`'s ABI uses the accumulator, for the peephole's
    A/B liveness: whether a value rides in B (a `@ B` parameter, written
    or promoted, or a B return slot) and whether A survives a call."""
    from r65.compiler.codegen.constants import get_return_registers
    from r65.compiler.hir.nodes import RegisterBinding
    from r65.compiler.optimize.peephole import AccAbi

    passes_b = (
        'B' in get_return_registers(mir_func.return_type, mir_func.entry_m_mode)
        or 'B' in mir_func.hw_param_regs.values()
        or any(isinstance(p.binding, RegisterBinding) and p.binding.register_name == 'B'
               for p in mir_func.parameters))
    preserves = mir_func.preserves_attr
    preserves_a = preserves is not None and 'A' in preserves.registers
    return AccAbi(passes_b=passes_b, preserves_a=preserves_a)


def _function_modes(mir_func):
    """The accumulator width (8/16) `mir_func` is entered and returns in.

    The entry mode is the one its `.ACCU` anchor declares and its callers
    switch to. An interrupt handler interrupts code in either mode and the
    reset entry starts in emulation mode, so theirs is unknown (None).
    """
    from r65.compiler.typeck.processor_mode import ModeState

    if mir_func.interrupt_attr is not None or mir_func.is_entry:
        entry = None
    else:
        entry = 16 if mir_func.entry_m_mode == ModeState.M16 else 8
    exit_mode = 16 if mir_func.exit_m_mode == ModeState.M16 else 8
    return entry, exit_mode


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
        # Last-run state, retained so callers can defer file writes (e.g. to
        # run the stack-usage analyzer between assembly emission and disk I/O)
        # and replay them via write_outputs().
        self._last_assembly: Optional[str] = None
        self._last_mir_program: Optional[MIRProgram] = None
        self._last_final_nodes: Optional[list] = None

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
                print(f"Dead function elimination: {func_eliminated} function(s) removed", file=sys.stderr)

            # Dead code elimination - remove unreachable blocks and dead stores
            dead_code_elim = DeadCodeEliminator(verbose=False)
            code_eliminated = dead_code_elim.eliminate(mir_program)
            if code_eliminated > 0:
                print(f"Dead code elimination: {code_eliminated} block(s)/instruction(s) removed", file=sys.stderr)

            # Byte-extract lowering - fold `(mem16 >> 8) as u8` into a load
            # of the high byte. Runs before inlining so callees shrink first:
            # the folded form is what gets cloned into call sites, and the
            # smaller body is a better inlining candidate.
            from r65.compiler.optimize.byte_extract import ByteExtractOptimizer
            byte_extract = ByteExtractOptimizer(verbose=False)
            bytes_folded = byte_extract.optimize(mir_program)
            if bytes_folded > 0:
                print(f"Byte extraction: {bytes_folded} high-byte load(s) folded", file=sys.stderr)

            # Far-to-near call optimization - convert far calls to near when
            # call graph shows all callers and callee are in the same bank
            # This saves 1 byte and 2 cycles per call (JSR vs JSL)
            # Run before inlining so converted functions can become inlinable
            from r65.compiler.optimize.far_to_near import FarToNearOptimizer
            far_to_near = FarToNearOptimizer(verbose=False)
            far_converted = far_to_near.optimize(mir_program)
            if far_converted > 0:
                print(f"Far-to-near optimization: {far_converted} function(s) converted", file=sys.stderr)

            # Function inlining - replace call sites with inlined function bodies
            # At -O1: only explicit inlining (#[inline] or #[inline(always)])
            # At -O2: also implicit inlining (called-once and small functions)
            implicit_inline = (opt_level >= 2)
            func_inliner = FunctionInliner(verbose=False, implicit_inline=implicit_inline)
            inlined_count = func_inliner.run(mir_program)
            if inlined_count > 0:
                print(f"Function inlining: {inlined_count} call site(s) inlined", file=sys.stderr)

                # Re-run mode tracking on every function the inliner mutated:
                # cloned blocks arrive with entry_mode/exit_mode = None, and
                # later passes (the peephole join-point fix among them) would
                # otherwise default those blocks to m8.
                from r65.compiler.mir.mode_tracker import reanalyze_function
                func_by_name = {f.name: f for f in mir_program.functions}
                for caller_name in func_inliner.mutated_funcs:
                    caller_func = func_by_name.get(caller_name)
                    if caller_func is None:
                        continue
                    if not reanalyze_function(caller_func):
                        from r65.compiler.errors import MIRLoweringError
                        raise MIRLoweringError(
                            f"Mode tracking failed for function '{caller_name}' "
                            "after inlining: mode conflicts detected. The inliner "
                            "produced an invalid CFG."
                        )

                # Re-run dead function elimination after inlining
                # (inlined functions may now be unused)
                dead_func_elim2 = DeadFunctionEliminator(verbose=False)
                func_eliminated2 = dead_func_elim2.eliminate(mir_program)
                if func_eliminated2 > 0:
                    print(f"Post-inlining dead function elimination: {func_eliminated2} function(s) removed", file=sys.stderr)

            # Constant propagation - a local assigned one constant feeds its
            # reads as an immediate (a macro's `let` of a constant argument,
            # a constant inlined into a body).
            from r65.compiler.optimize.const_prop import ConstantPropagator
            consts_propagated = ConstantPropagator(verbose=False).optimize(mir_program)
            if consts_propagated > 0:
                print(f"Constant propagation: {consts_propagated} operand(s) folded", file=sys.stderr)

            # Byte stores of a u16's low and high byte to adjacent addresses
            # as one word store (an OAM entry's tile and attr).
            from r65.compiler.optimize.store_merge import StoreMerger
            stores_merged = StoreMerger(verbose=False).optimize(mir_program)
            if stores_merged > 0:
                print(f"Store merging: {stores_merged} byte pair(s) stored as words", file=sys.stderr)

            # Constant-index folding - an X-indexed access with a constant X
            # becomes the absolute access, the X load dropped. What an inlined
            # #[soa] method called on a constant element compiles to:
            # `POOL[PLAYER].fly()` reads POOL__x_px, not POOL__x_px,X.
            from r65.compiler.optimize.const_index import ConstIndexFolder
            const_index = ConstIndexFolder(verbose=False)
            indices_folded = const_index.optimize(mir_program)
            if indices_folded > 0:
                print(f"Constant-index folding: {indices_folded} access(es) made absolute", file=sys.stderr)

            # Constant-bound compares - `x > K` as `x >= K+1` and `x <= K`
            # as `x < K+1`: one branch instead of BEQ plus BCS/BCC.
            from r65.compiler.optimize.compare_bound import CompareBoundFolder
            bounds_folded = CompareBoundFolder(verbose=False).optimize(mir_program)
            if bounds_folded > 0:
                print(f"Constant-bound compares: {bounds_folded} compare(s) made one-branch", file=sys.stderr)

            # Byte widening - a u8 load or small mask widened to u16 becomes
            # one m16 load or read plus AND, not an m8 load and a zero-extend.
            from r65.compiler.optimize.byte_widen import ByteWidener
            bytes_widened = ByteWidener(verbose=False).optimize(mir_program)
            if bytes_widened > 0:
                print(f"Byte widening: {bytes_widened} widening(s) done in m16", file=sys.stderr)

            # Known bits - a value known below 256 narrowed and widened again,
            # or zero-extended only to be shifted left by 8, needs no mask.
            from r65.compiler.optimize.known_bits import KnownBitsSimplifier
            bits_simplified = KnownBitsSimplifier(verbose=False).optimize(mir_program)
            if bits_simplified > 0:
                print(f"Known bits: {bits_simplified} conversion(s) simplified", file=sys.stderr)

            # Constant index offsets - `walkmap[t + 32]` as `walkmap+32,X`
            # with X = t, so neighbouring cells share one X load.
            from r65.compiler.optimize.index_offsets import IndexOffsetFolder
            offsets_folded = IndexOffsetFolder(verbose=False).optimize(mir_program)
            if offsets_folded > 0:
                print(f"Index offsets: {offsets_folded} constant(s) folded into addresses", file=sys.stderr)

            # Loop unrolling - fully unroll small static loops at -O2
            if opt_level >= 2:
                from r65.compiler.optimize.loop_unroll import LoopUnroller
                loop_unroller = LoopUnroller(verbose=False)
                unrolled_count = loop_unroller.unroll(mir_program)
                if unrolled_count > 0:
                    print(f"Loop unrolling: {unrolled_count} loop(s) unrolled", file=sys.stderr)

            # Loop rotation - a loop whose first test always passes is entered
            # at its body and tests at the bottom: one branch per iteration.
            # After unrolling, which matches the top-tested shape.
            from r65.compiler.optimize.loop_rotate import LoopRotator
            loop_rotator = LoopRotator(verbose=False)
            rotated_count = loop_rotator.optimize(mir_program)
            if rotated_count > 0:
                print(f"Loop rotation: {rotated_count} loop(s) rotated, "
                      f"{loop_rotator.guarded} guarded, "
                      f"{loop_rotator.counted_down} counting down", file=sys.stderr)

            # Memory-operand folding - read a single-use static directly in
            # the ADC/SBC/AND/ORA/EOR/CMP that consumes it. Runs last so it
            # sees inlined and unrolled bodies, and before register
            # allocation so the deleted load no longer blocks the other
            # operand from staying in A.
            from r65.compiler.optimize.memory_operand import MemoryOperandFolder
            operand_folder = MemoryOperandFolder(verbose=False)
            operands_folded = operand_folder.optimize(mir_program)
            if operands_folded > 0:
                print(f"Memory-operand folding: {operands_folded} load(s) folded", file=sys.stderr)

            # Read-modify-write on memory - `m = m ± 1`, `m << 1` and an
            # unsigned `m >> 1` on a direct-page / absolute static or plane
            # as one INC/DEC/ASL/LSR, and a following `m == 0` test as the Z
            # flag it leaves.
            from r65.compiler.optimize.memory_incdec import MemoryIncDecFolder
            incdec_folder = MemoryIncDecFolder(verbose=False)
            incdec_folded = incdec_folder.optimize(mir_program)
            if incdec_folded > 0:
                print(f"Memory read-modify-write folding: {incdec_folded} step(s) folded",
                      file=sys.stderr)

            # Web splitting - a local reassigned with a value its old one
            # never meets (`i` reused by a second loop) gets a vreg per
            # value, so loop promotion and slot allocation see each apart.
            from r65.compiler.optimize.web_split import WebSplitter
            webs_split = WebSplitter(verbose=False).optimize(mir_program)
            if webs_split > 0:
                print(f"Web splitting: {webs_split} vreg(s) split", file=sys.stderr)

            # Index stride - a counter read only as an index scaled for 2- or
            # 4-byte elements is carried scaled (INX x2/x4), not rebuilt with
            # ASL every pass.
            # After web splitting, so a reused counter has one def per loop.
            from r65.compiler.optimize.index_stride import IndexStrideReducer
            strides_reduced = IndexStrideReducer(verbose=False).optimize(mir_program)
            if strides_reduced > 0:
                print(f"Index stride: {strides_reduced} counter(s) carried scaled", file=sys.stderr)


        # Emit file header and processor directives
        self.emitter.emit_file_header()
        self.emitter.emit_processor_directive()

        # Organize functions by bank first to determine bank count
        functions_by_bank = self._organize_functions_by_bank(mir_program)
        # ROM data goes in its static's bank; open a window for each such bank.
        rom_data_by_bank = self._rom_data_by_bank(mir_program)
        for data_bank in rom_data_by_bank:
            functions_by_bank.setdefault(data_bank, [])
        max_bank = max(functions_by_bank.keys()) if functions_by_bank else 0
        bank_count = max_bank + 1  # Banks are 0-indexed

        # Determine ROM type from snesrom_config
        is_hirom = False
        is_fastrom = False
        if mir_program.snesrom_config:
            is_hirom = mir_program.snesrom_config.hirom or mir_program.snesrom_config.exhirom
            is_fastrom = mir_program.snesrom_config.fastrom

        # Calculate minimum ROM size (power-of-2 banks and ROMSIZE value)
        # calculate_rom_size rounds up to power of 2 and enforces 256KB minimum
        rom_banks, self.romsize_value = calculate_rom_size(bank_count, is_hirom)

        # LoROM FastROM relocates ROM labels into the $80-$BF mirror, which
        # only covers the first 4MB. Refuse rather than silently wrapping.
        if is_fastrom and not is_hirom and 0x80 + max_bank > 0xFF:
            raise CodegenError(
                f"fastrom needs bank ${0x80 + max_bank:02X} for bank {max_bank}, "
                f"past the $80-$BF LoROM mirror; ROMs over 4MB cannot use fastrom"
            )

        # Emit memory map with calculated bank count and ROM type
        rom_type = "hirom" if is_hirom else "lorom"
        self.emitter.emit_memory_map(rom_type=rom_type, banks=rom_banks,
                                     fastrom=is_fastrom)

        # Memory allocation
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
            print(w, file=sys.stderr)
        self.warnings.extend(self.allocator.warnings)

        # Symbol definitions
        symbol_gen = SymbolDefinitionGenerator(self.emitter, self.allocator)
        symbol_gen.emit_all_definitions(mir_program.constants, auto_bank_number(mir_program),
                                        rom_bank_size(is_hirom))

        # Function code generation
        self.func_gen = ProgramFunctionGenerator(self.emitter, self.allocator)

        # Create scratch pool once for all functions
        scratch_pool = self.func_gen.func_gen._create_scratch_pool(mir_program)

        # Run parameter promotion analysis (before loop promotion)
        abi_model.run_param_analysis(mir_program, scratch_pool, disable_scratch_params)

        # Analyze far pointer strategy (D=S vs SET_DBR) after param promotion
        # has set has_far_ptr_stack_params, before code generation
        from r65.compiler.analysis.far_ptr_strategy import analyze_far_ptr_strategy
        from r65.compiler.analysis.trait_chain_coalesce import analyze_trait_dispatch_chains
        analyze_far_ptr_strategy(mir_program)

        # Trait-dispatch chain coalescing: detect same-self runs of
        # TraitDispatch and assign ChainRole values so codegen can emit
        # the PHB/PLB DBR bracket once per run instead of per-call.
        # Builds the trait-resolved call graph for soundness verification.
        from r65.compiler.analysis.call_graph import build_call_graph
        analyze_trait_dispatch_chains(mir_program, build_call_graph(mir_program))

        # Run loop register promotion analysis (after scratch params)
        if not disable_loop_promotion:
            from r65.compiler.analysis.loop_register_promotion import analyze_loop_promotion
            analyze_loop_promotion(mir_program)

        # Index streams - of two index values alternating in X, give one Y
        # for the run, so `oam[spr].x = pool[i].x` stops reloading X per
        # field. After promotion: it sees who holds Y, and leaves a loop's
        # induction variable there.
        if opt_level >= 1:
            from r65.compiler.optimize.index_streams import IndexStreamSplitter
            streams_split = IndexStreamSplitter(verbose=False).optimize(mir_program)
            if streams_split > 0:
                print(f"Index streams: {streams_split} moved to Y", file=sys.stderr)

        # Whether the program needs a shared interrupt-vector placeholder.
        # The vector table is 16-bit and an interrupt clears PB, so
        # `__empty_handler` (the target of every unused vector) must be
        # at a bank-$00 address.
        needs_empty_handler = self._program_needs_interrupt_vectors(mir_program)

        # Guarantee a bank-0 window exists to hold it. A program whose
        # functions all carry #[bank(n>0)] otherwise emits no bank 0 at all,
        # and the vector table then references a __empty_handler that was
        # never defined - wlalink rejects it with "Reference to an unknown
        # label". Anything else that must be reachable at a bank-$00 address
        # lands in this window too.
        if needs_empty_handler:
            functions_by_bank.setdefault(0, [])

        # Group include_asm! directives by bank for placement inside .BANK windows
        asm_includes_by_bank: Dict[int, List[Any]] = {}
        for inc in getattr(mir_program, 'asm_includes', []):
            bank_for_inc = 0 if inc.bank_number is None else inc.bank_number
            asm_includes_by_bank.setdefault(bank_for_inc, []).append(inc)
            functions_by_bank.setdefault(bank_for_inc, [])  # ensure bank window opens

        # Generate callees before callers, so a call sees which index
        # registers its callee leaves alone (clobber_summary); each body is
        # laid out in its bank below. Every function starts from the same
        # emitter state, in its own bank.
        from r65.compiler.codegen.clobber_summary import (
            INDEX_REGS, callee_first, has_inline_asm, index_writes)
        from r65.compiler.codegen.emitter import _snapshot, _restore
        gen = self.func_gen.func_gen
        gen.index_writes = {}
        laid_out = [f for b in sorted(functions_by_bank) for f in functions_by_bank[b]]
        bank_of = {f.name: b for b, fs in functions_by_bank.items() for f in fs}
        function_names = frozenset(bank_of)
        fresh = _snapshot(self.emitter)
        bodies = {}
        for mir_func in callee_first(laid_out):
            _restore(self.emitter, fresh)
            self.emitter.code_bank = self.emitter.rom_base + bank_of[mir_func.name]
            outer, self.emitter.nodes = self.emitter.nodes, []
            try:
                gen.generate_function(mir_func, scratch_pool=scratch_pool,
                                      abi_model=abi_model)
            finally:
                body, self.emitter.nodes = self.emitter.nodes, outer
            bodies[mir_func.name] = body
            if opt_level >= 1:
                gen.index_writes[mir_func.name] = (
                    INDEX_REGS if has_inline_asm(mir_func)
                    else index_writes(body, gen.index_writes, function_names))
        _restore(self.emitter, fresh)

        # Generate code for each bank
        function_starts = []
        entry_modes = {}
        exit_modes = {}
        acc_abi = {}
        for bank_num in sorted(functions_by_bank.keys()):
            bank_functions = functions_by_bank[bank_num]

            # Always emit bank directive to ensure correct bank after ROM data sections
            # ROM data is emitted before code by symbol_gen and can leave assembler in different bank
            self.emitter.emit_section_header(f"Bank {bank_num}")
            self.emitter.emit_bank_directive(bank_num)

            # WLA-DX .INCLUDE pulls the asm file contents into the current bank/section
            for inc in asm_includes_by_bank.get(bank_num, []):
                self.emitter.emit_directive(f'.INCLUDE "{inc.path}"')

            # Generate functions in this bank
            for mir_func in bank_functions:
                start = len(self.emitter.nodes)
                self.emitter.nodes.extend(bodies[mir_func.name])
                # The peephole optimizer iterates each function separately.
                if len(self.emitter.nodes) > start:
                    function_starts.append(self.emitter.nodes[start])
                entry_modes[mir_func.name], exit_modes[mir_func.name] = (
                    _function_modes(mir_func))
                acc_abi[mir_func.name] = _acc_abi(mir_func)
                # Emit this function's local array/string/struct literal
                # init data immediately after its body so the data lands in
                # the same ROM bank as the function's inlined MVN block-copy.
                self._emit_function_local_rom_data(mir_func)

            # Anchor the empty interrupt handler at the END of
            # bank 0's emission, after all bank-0 functions but
            # before the next bank's directive. We must emit it
            # inside the same bank-0 emission window because
            # re-opening `.BANK 0 SLOT 0` later (e.g. after
            # auto-bank functions in bank 8+) is unsafe in WLA-DX
            # without explicit sections — it can place the handler
            # at an address that overlaps existing bank-0 content.
            # Emitting after bank-0 functions keeps the program's
            # entry point as the first node in bank 0 so downstream
            # asm-level dataflow analysis sees a connected CFG
            # rooted at the entry; emitting before functions would
            # leave the entry isolated past an `RTI` that has no
            # successor.
            if bank_num == 0 and needs_empty_handler:
                self.emitter.emit_empty_interrupt_handler()

            # LoROM FastROM: the vector table is 16-bit and the CPU clears PB
            # when it fires, so entry code runs from the slow $00-$3F mirror
            # until a JML reloads PBR. Emit the stubs that do it here, inside
            # the bank-0 window, for the same reason __empty_handler lives
            # here: they must be reachable at a bank-$00 address.
            if bank_num == 0 and is_fastrom and not is_hirom and needs_empty_handler:
                handlers, reset_h = self._find_vector_handlers(mir_program)
                self.emitter.emit_lorom_fast_vector_trampolines(
                    reset=reset_h, **handlers
                )

            # ROM static data and initializer tables placed in this bank
            self._emit_rom_data_sections(rom_data_by_bank.get(bank_num, []))

        # Trait dispatch tables (jump tables and wrapper functions): a
        # region of their own, each wrapper entered in its caller's mode.
        start = len(self.emitter.nodes)
        for dispatch_label in self._emit_trait_dispatch_tables(mir_program):
            entry_modes[dispatch_label] = None
        if len(self.emitter.nodes) > start:
            function_starts.append(self.emitter.nodes[start])

        # Interrupt vectors
        self._emit_interrupt_vectors(mir_program)

        # Get structured nodes from emitter
        nodes = self.emitter.get_nodes()

        # Collect volatile register names and addresses (from #[hw] attributes) for peephole optimizer
        volatile_names, volatile_addresses = self._collect_volatile_registers(mir_program)

        # Apply node-based peephole optimizations (only if opt_level >= 1)
        if opt_level >= 1:
            optimized_nodes, num_optimizations = optimize_nodes(
                nodes, volatile_names, volatile_addresses,
                region_starts=function_starts,
                entry_modes=entry_modes, call_exit_modes=exit_modes,
                acc_abi=acc_abi)
            if num_optimizations > 0:
                print(f"Peephole optimizer: {num_optimizations} optimization(s) applied", file=sys.stderr)
            from r65.compiler.codegen.clobber_summary import check_after_peephole
            check_after_peephole(optimized_nodes,
                                 {n: b[-1] for n, b in bodies.items() if b},
                                 gen.index_writes, function_names)
        else:
            optimized_nodes = nodes

        # Apply node-based long branch fixup (always required for correct assembly)
        # Code generator emits BRA by default; fixup converts to JMP when target > 127 bytes
        final_nodes, num_branch_fixups = fixup_nodes(optimized_nodes)

        if num_branch_fixups > 0:
            print(f"Branch fixup: {num_branch_fixups} long branch(es) fixed", file=sys.stderr)

        # Convert nodes to assembly string
        assembly = emit_nodes(final_nodes)

        # Retain so callers can defer disk I/O (run the stack-usage analyzer
        # between emit and write) and later call write_outputs().
        self._last_assembly = assembly
        self._last_mir_program = mir_program
        self._last_final_nodes = final_nodes

        # Write to file if specified
        if output_file:
            self.write_outputs(output_file, debug=debug)

        return assembly

    def write_outputs(self, output_file: str, debug: bool = False) -> None:
        """Write the most recent generate() result to disk.

        Splits the file I/O from generate() so callers can interpose checks
        (e.g. stack-usage analysis) between assembly emission and writes.
        Must be called after a successful generate().
        """
        if self._last_assembly is None:
            raise RuntimeError("write_outputs() called before generate()")
        with open(output_file, 'w') as f:
            f.write(self._last_assembly)
        self._generate_linkfile(output_file)
        if debug and self._last_mir_program is not None and self._last_final_nodes is not None:
            self._generate_debug_file(
                self._last_mir_program, self._last_final_nodes, output_file
            )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _emit_trait_dispatch_tables(self, mir_program: MIRProgram) -> List[str]:
        """
        Emit trait dispatch wrapper functions and jump tables; return the
        wrappers' entry labels.

        For each trait method, generates:
        1. A dispatch wrapper function that loads TypeId from self_ptr and
           indexes into a jump table
        2. A jump table with word-size entries pointing to concrete implementations

        Dispatch wrapper, near and far alike (self in Y, DBR = object's bank):
            TraitName__method__dispatch:
              LDA.w $0000,Y        ; Load TypeId from *self (offset 0)
              REP #$20
              AND #$00FF           ; Zero-extend to 16-bit (B is garbage)
              ASL A                ; x2 for word table
              TAX
              SEP #$20
              JMP (TraitName__method__table,X)

        Far dispatch indexes a word table of JML stubs (one per TypeId), so
        the caller's JSL return address stays for the method's RTL.
        """
        if not hasattr(mir_program, 'trait_dispatch_info') or not mir_program.trait_dispatch_info:
            return []

        self.emitter.emit_section_header("Trait Dispatch Tables")

        dispatch_labels = []
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
                dispatch_labels.append(dispatch_label)

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
        from r65.compiler.codegen.opcodes import Opcode
        self.emitter.emit_label("_trait_error")
        self.emitter.emit_instr(Opcode.STP)
        self.emitter.emit_blank_line()
        return dispatch_labels

    def _emit_dispatch_wrapper(self, dispatch_label, table_label):
        """Emit a dispatch entry: jump through the word table *table_label*
        by the TypeId at offset 0 of the trait object. The self pointer is
        in Y (DBR:Y addressing; a far caller has set DBR to the object's
        bank).

        Entered in the caller's accumulator width, which the TypeId load does
        not depend on: the index is computed in m16 (AND clears B, which TAX
        would otherwise carry into X's high byte) and the method is entered
        in m8.
        """
        from r65.compiler.codegen.asm_nodes import Address, Immediate
        from r65.compiler.codegen.constants import M_FLAG
        from r65.compiler.codegen.opcodes import Opcode

        emit = self.emitter.emit_instr
        self.emitter.emit_label(dispatch_label)
        self.emitter.set_accu_mode_tracking(None)
        emit(Opcode.LDA_ABSOLUTE_Y, Address(0), "TypeId at DBR:Y+0")
        emit(Opcode.REP_IMMEDIATE, Immediate(M_FLAG))
        self.emitter.emit_accu_mode(16)
        emit(Opcode.AND_IMMEDIATE, Immediate(0x00FF), "Zero-extend the TypeId")
        emit(Opcode.ASL, None, "x2 for word table entries")
        emit(Opcode.TAX)
        emit(Opcode.SEP_IMMEDIATE, Immediate(M_FLAG), "Methods are entered in m8")
        self.emitter.emit_accu_mode(8)
        emit(Opcode.JMP_INDIRECT_X, Address(table_label))
        self.emitter.emit_blank_line()

    def _emit_near_dispatch(self, dispatch_label, table_label, trait_name, method_name,
                            method_idx, implementors, max_type_id):
        """Emit near dispatch wrapper and jump table for a trait method."""
        self._emit_dispatch_wrapper(dispatch_label, table_label)

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

        self._emit_dispatch_wrapper(dispatch_label, table_label)

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
        from r65.compiler.codegen.asm_nodes import Address
        from r65.compiler.codegen.opcodes import Opcode
        for tid in range(max_type_id + 1):
            jml_label = f"{trampoline_label}_{tid}"
            self.emitter.emit_label(jml_label)
            self.emitter.emit_instr(
                Opcode.JMP_LONG, Address(type_id_to_impl.get(tid, '_trait_error')))

        self.emitter.emit_blank_line()


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

    def _organize_functions_by_bank(self, mir_program: MIRProgram) -> Dict[int, List[MIRFunction]]:
        """
        Organize functions by bank number.

        - Functions with #[bank(n)] attribute go to bank n.
        - Functions with no bank attribute default to bank 0.
        - Functions with #[bank(auto)] (bank_attr present, bank_number=None,
          guaranteed `far` by HIR validation) are placed together in
          `auto_bank_number(mir_program)`. wlalink reports an overflow if
          they do not fit; the user can then split with explicit #[bank(n)]
          directives.

        Returns:
            Dictionary mapping bank number to list of functions
        """
        by_bank: Dict[int, List[MIRFunction]] = {}
        auto_bank_funcs: List[MIRFunction] = []

        for func in mir_program.functions:
            if func.bank_attr is None:
                # No bank attribute: default to bank 0
                by_bank.setdefault(0, []).append(func)
            elif func.bank_attr.bank_number is not None:
                # Explicit bank
                by_bank.setdefault(func.bank_attr.bank_number, []).append(func)
            else:
                # #[bank(auto)] - defer placement until we know the explicit-bank max
                auto_bank_funcs.append(func)

        if auto_bank_funcs:
            by_bank.setdefault(auto_bank_number(mir_program), []).extend(auto_bank_funcs)

        return by_bank

    def _program_needs_interrupt_vectors(self, mir_program: MIRProgram) -> bool:
        """True iff the program needs an SNES vector table (and a
        bank-0 placeholder handler for unused vectors)."""
        for func in mir_program.functions:
            if func.interrupt_attr is not None:
                return True
            if func.is_entry:
                return True
        return False

    def _find_vector_handlers(self, mir_program: MIRProgram):
        """
        Scan MIR functions for the labels the SNES vector table points at.

        Returns (handlers, reset): `handlers` maps each vector that has an
        `#[interrupt(...)]` function — 'nmi', 'irq', 'brk', 'cop', 'abort'
        — to its label; reset is the `#[entry]` function, or None.
        """
        handlers = {}
        reset_handler = None

        for func in mir_program.functions:
            if func.interrupt_attr:
                # vector is an InterruptVector enum
                handlers[func.interrupt_attr.vector.value.lower()] = func.name

            # Check for entry point (becomes reset vector)
            if func.is_entry:
                reset_handler = func.name

        return handlers, reset_handler

    def _emit_interrupt_vectors(self, mir_program: MIRProgram):
        """
        Emit SNES ROM header and interrupt vector table.

        Scans functions for #[interrupt(vector)] attributes and
        generates the SNES header and interrupt vector table.

        Args:
            mir_program: MIR program
        """
        # Find interrupt handlers
        handlers, reset_handler = self._find_vector_handlers(mir_program)

        # Check for missing NMI handler - this is a common oversight
        # Only warn if there's an entry point (i.e., this is a standalone program)
        if reset_handler and 'nmi' not in handlers:
            self.warnings.append(
                "No NMI interrupt handler defined. "
                "The NMI fires every VBlank (~60Hz) and is typically used for game logic updates. "
                "Add an NMI handler with: #[interrupt(nmi)] fn nmi_handler() { ... }"
            )

        # Emit SNES header and vectors if any handlers found
        if handlers or reset_handler:
            # The empty interrupt handler was already emitted into
            # bank 0 alongside the other bank-0 functions; see the
            # `needs_empty_handler` block in the bank-emission loop.

            # For HiROM, all interrupt vectors must point to addresses >= $8000
            # because the CPU reads vectors from bank $00 where only $8000-$FFFF
            # maps to ROM. Emit trampolines at $FF00+ for each handler.
            is_hirom = False
            is_fastrom = False
            if mir_program.snesrom_config:
                is_hirom = mir_program.snesrom_config.hirom or mir_program.snesrom_config.exhirom
                is_fastrom = mir_program.snesrom_config.fastrom

            # The vectors point at the handlers themselves, or at the
            # trampolines into them: HiROM's at $FF00, or (LoROM FastROM)
            # the stubs emitted in the bank-0 window above, so entry lands
            # in the $80-$BF fast mirror.
            prefix = None
            if is_hirom:
                self.emitter.emit_hirom_vector_trampolines(
                    reset=reset_handler,
                    fastrom=is_fastrom,
                    **handlers
                )
                prefix = "__hirom_"
            elif is_fastrom:
                prefix = "__fast_"
            targets = {vector: f"{prefix}{vector}" if prefix else name
                       for vector, name in handlers.items()}
            actual_reset = reset_handler
            if prefix and reset_handler:
                actual_reset = f"{prefix}reset"

            # Emit SNES ROM header (use config from #[snesrom(...)] if present)
            self.emitter.emit_snes_header(
                snesrom_config=mir_program.snesrom_config,
                romsize_value=self.romsize_value
            )

            # Emit interrupt vectors
            self.emitter.emit_interrupt_vectors(reset=actual_reset, **targets)

    def _emit_rom_data_table(self, rom_data):
        """Emit a single ROM data table (label + .db rows of 16 bytes)."""
        self.emitter.emit_label(rom_data.label)
        data = rom_data.data
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            bytes_str = ', '.join(
                b.render() if isinstance(b, SymbolByte) else f'${b:02X}'
                for b in chunk
            )
            self.emitter.emit_directive(f".db {bytes_str}")

    def _emit_function_local_rom_data(self, mir_func: MIRFunction):
        """
        Emit a function's local array/string/struct literal init data
        immediately after its body, inside the function's ROM bank.

        Keeping the data in the same bank as the function lets the inlined
        MVN block-copy reach it via the data label's own bank (resolved by
        WLA-DX), instead of routing through __init_start and the global
        trailing ROM data sections (which land in a different bank).
        """
        if not mir_func.local_rom_data:
            return

        self.emitter.emit_comment(
            f"Local literal init data for {mir_func.name}"
        )
        for rom_data in mir_func.local_rom_data:
            self._emit_rom_data_table(rom_data)

    def _rom_data_by_bank(self, mir_program: MIRProgram) -> Dict[int, List[Any]]:
        """Group the program's ROM data tables by the WLA bank they belong in.

        A ROM static's table (`__NAME_data`) goes in the static's own bank -
        the rule include_bytes! data follows too (symbol_gen.rom_static_bank):
        its #[bank(n)], or the auto bank under #[bank(auto)]. A near `&T`
        reads through DBR, so a bank-0 static's bytes must be in bank 0.
        Everything else - a RAM static's initializer, an inline string -
        goes in bank 0.
        """
        auto_bank = auto_bank_number(mir_program)
        bank_of_label: Dict[str, int] = {}
        for decl in mir_program.statics:
            label = getattr(getattr(decl, 'symbol', None), 'rom_label', None)
            if label:
                bank = rom_static_bank(decl)
                bank_of_label[label] = auto_bank if bank is None else bank
        by_bank: Dict[int, List[Any]] = {}
        for rom_data in mir_program.rom_data_sections:
            by_bank.setdefault(bank_of_label.get(rom_data.label, 0), []).append(rom_data)
        return by_bank

    def _emit_rom_data_sections(self, rom_data_tables: List[Any]):
        """
        Emit ROM data tables into the open bank window.

        These hold ROM statics' bytes and the initializer data that
        __init_start copies to RAM with BlockCopy (MVN).

        Args:
            rom_data_tables: ROMDataRef tables placed in this bank
        """
        if not rom_data_tables:
            return

        self.emitter.emit_section_header("ROM Data Sections (array literal init data)")

        for rom_data in rom_data_tables:
            self._emit_rom_data_table(rom_data)

    # ========================================================================
    # Linkfile Generation
    # ========================================================================

    def _generate_linkfile(self, output_file: str):
        """
        Generate a WLA-DX linkfile alongside the assembly output.

        The linkfile references the .o object file (same stem as .asm).
        RAM section placement is handled by .RAMSECTION directives in
        the assembly source itself (BANK/SLOT/FORCE ORGA), so no
        [ramsections] block is needed in the linkfile.

        Args:
            output_file: Path to the .asm output file
        """
        import os

        base = os.path.splitext(output_file)[0]
        link_path = base + ".link"
        obj_path = base + ".o"

        with open(link_path, 'w') as f:
            f.write(f"[objects]\n{obj_path}\n")

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
        print(f"Debug info written to: {dbg_path}", file=sys.stderr)

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
        functions_by_bank = self._organize_functions_by_bank(mir_program)

        # Bank size in bytes
        bank_size = rom_bank_size(is_hirom)

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
                bank=self._cpu_bank(bank_num),
                ooffs=ooffs,
                seg_type="ro"
            )

    def _cpu_bank(self, bank_num: int) -> int:
        """The 65816 bank the CPU executes WLA bank `bank_num` from.

        `.BASE $80` (LoROM FastROM) / `$C0` (HiROM) shifts it off the WLA
        bank index. DbgSegment.bank holds this CPU bank, so a lookup by WLA
        bank index goes through here too.
        """
        return getattr(self.emitter, 'rom_base', 0) + bank_num

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

        # Calculate label addresses
        calc = AddressCalculator(base_address)
        _, label_addresses = calc.calculate_with_labels(nodes)

        # Add function symbols
        auto_bank = auto_bank_number(mir_program)
        for func in mir_program.functions:
            # Get address from label
            addr = label_addresses.get(func.name, base_address)

            # Determine segment (bank): the one _organize_functions_by_bank put it in
            bank_num = 0
            if func.bank_attr is not None:
                bank_num = func.bank_attr.bank_number
                if bank_num is None:
                    bank_num = auto_bank

            # Find segment ID for this bank (segments carry the CPU bank)
            seg_id = None
            cpu_bank = self._cpu_bank(bank_num)
            for seg in self.debug_info.segments:
                if seg.bank == cpu_bank:
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
            # Size is the distance to the next function; the last one gets 0x100
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
                span_ids=[],
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
        from r65.compiler.codegen.asm_nodes import Instruction

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
