# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Function Inlining Optimization Pass.

Replaces call sites with inlined function bodies to eliminate JSR/RTS overhead
(12 cycles) and JSL/RTL overhead (14 cycles).

The inlining pass operates on MIR after HIR lowering, before code generation.
"""

from typing import Dict, List, Set, Optional, Tuple

from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock,
    MIRInstruction, VirtualRegister, HardwareRegister, Immediate,
    Move, Jump, Return, Call, CondBranch, JumpTable, LookupTable,
    Load, Store, LoadIndirect, StoreIndirect, MemoryLocation,
    BinaryOp, UnaryOp, Compare, TypeConvert, ToBool,
    InlineAsm, ReturnFromInterrupt, TraitDispatch,
    Argument, ArgumentMechanism,
    MemoryFill, BlockCopy, BankByte, BitTest, Rotate,
    StatusFlagTest, StatusFlagSet, StatusFlagRead,
    SetMode, Push, Pull, SaveRegister, RestoreRegister,
    FarPtrStrategy,
)
from r65.compiler.hir import RegisterBinding, VariableBinding
from r65.compiler.hir.attributes import InlineMode
from r65.compiler.analysis.loop_analysis import compute_block_nesting


# ---------------------------------------------------------------------------
# Cycle-cost model
# ---------------------------------------------------------------------------
# Estimated 65816 cycles contributed by each MIR instruction type when
# the function is inlined (Return costs 0 because it is eliminated).
# Used in place of a flat instruction count so the threshold can be
# compared directly against the JSR/RTS overhead being saved.

_MIR_CYCLE_COSTS: Dict[type, int] = {
    Move:             2,
    Load:             3,
    Store:            3,
    LoadIndirect:     6,
    StoreIndirect:    6,
    BinaryOp:         3,
    UnaryOp:          2,
    Compare:          2,
    BitTest:          2,
    TypeConvert:      2,
    ToBool:           3,
    Rotate:           2,
    BankByte:         2,
    Jump:             3,
    CondBranch:       3,
    JumpTable:        6,
    LookupTable:      5,
    StatusFlagTest:   3,
    StatusFlagSet:    2,
    StatusFlagRead:   4,
    Call:            12,   # nested JSR(6)+RTS(6) — counts against callee body
    TraitDispatch:   16,
    SetMode:          2,
    Push:             3,
    Pull:             4,
    SaveRegister:     2,
    RestoreRegister:  2,
    MemoryFill:      20,
    BlockCopy:       20,
    Return:           0,   # eliminated by inlining
    ReturnFromInterrupt: 0,
}
_DEFAULT_CYCLE_COST = 3

# Cycle budget thresholds (compared against _estimate_cycle_cost())
CALL_OVERHEAD_CYCLES     = 12   # JSR(6) + RTS(6)
INLINE_COST_BREAK_EVEN   = 12   # body ≤ overhead → always worth inlining
INLINE_COST_WITH_ATTR    = 60   # ~30 MIR instrs at avg 2 cycles
INLINE_COST_NO_ATTR      = 12   # break-even for unmarked implicit inlining

# Per-nesting-depth multiplier on INLINE_COST_NO_ATTR (index = clamped depth)
_LOOP_DEPTH_MULTIPLIER = [1, 3, 6]

# Backward-compatibility aliases (previously raw instruction counts)
INLINE_THRESHOLD_WITH_ATTR = INLINE_COST_WITH_ATTR
INLINE_THRESHOLD_NO_ATTR   = INLINE_COST_NO_ATTR


# Instructions that produce their result in the A register. When the
# Return is preceded by such an instruction whose dest is the same vreg
# we're about to return, A still holds that value and the inliner can
# substitute HardwareRegister('A') for the vreg source, sparing the
# caller a redundant Move (often nothing emits at all once the codegen
# coalesces both endpoints to A).
def _value_is_in_a_at_end_of(instrs: List[MIRInstruction],
                              vreg: VirtualRegister) -> bool:
    """Conservative check: does A hold `vreg` after `instrs` execute?

    True iff the last MIR instruction in `instrs` produces `vreg` in A
    via an ALU/load operation that the codegen lowers as
    'compute-into-A then store to vreg's slot' (the common case for
    BinaryOp / UnaryOp / Load with a simple result vreg).

    Anything else — a Move from another vreg, a function call, a
    SetMode, a CondBranch — is treated as "A unknown". Likewise if
    there are no instructions, A's contents predate this block and
    can't be assumed.

    We deliberately keep this narrow. Loops, control flow that re-
    enters the block, or any work between the producer and the Return
    invalidates the assumption that the producer's result still lives
    in A.
    """
    if not instrs:
        return False
    last = instrs[-1]
    # The producing instruction must directly target the return-value
    # vreg, and must be one of the instruction kinds whose codegen
    # leaves the result in A en route to the destination slot.
    if not isinstance(last, (BinaryOp, UnaryOp, Load, LoadIndirect,
                              TypeConvert)):
        return False
    return getattr(last, 'dest', None) is vreg


class InlinabilityChecker:
    """
    Determines whether a function can and should be inlined.

    Hard requirements (must all be true):
    1. Not recursive (direct or mutual)
    2. Not an interrupt handler
    3. Not the entry point
    4. No inline assembly (asm!())
    5. If far: body must be bank-independent (see `_far_body_is_bank_safe`).
       Far fns whose bodies only use far indirects, DP / zeropage, or
       WRAM long-addressing can be inlined into any-bank callers.

    Heuristics for inlining decisions:
    - Explicit inlining (always checked):
      - Marked #[inline] or #[inline(always)] → inline if < 30 instructions
    - Implicit inlining (only when implicit_inline=True, i.e., -O2):
      - Called exactly once → always inline (no code size increase)
      - No attribute → inline only if very small (< 3 instructions)

    Note: Trivial getters/setters are auto-marked with #[inline] at HIR level.
    """

    def __init__(self, mir_program: MIRProgram, implicit_inline: bool = True):
        """
        Initialize the inlinability checker.

        Args:
            mir_program: The MIR program to analyze
            implicit_inline: If True, allow implicit inlining (called-once and small
                           functions without #[inline] attribute). If False, only
                           inline functions with explicit #[inline] or #[inline(always)].
        """
        self.mir_program = mir_program
        self.implicit_inline = implicit_inline
        self.func_map: Dict[str, MIRFunction] = {f.name: f for f in mir_program.functions}
        self.call_graph: Dict[str, Set[str]] = {}
        self.call_counts: Dict[str, int] = {}
        self.call_depths: Dict[str, int] = {}  # max loop nesting depth of any call site
        self._build_call_graph()
        self._compute_call_site_depths()
        self._find_recursive_functions()

    def _build_call_graph(self):
        """Build a call graph mapping function names to called functions."""
        for func in self.mir_program.functions:
            called = set()
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call) and isinstance(instr.function, str):
                        called.add(instr.function)
            self.call_graph[func.name] = called

    def _compute_call_site_depths(self):
        """Count call sites and record the maximum loop nesting depth of each."""
        self.call_counts = {f.name: 0 for f in self.mir_program.functions}
        self.call_depths = {f.name: 0 for f in self.mir_program.functions}
        for func in self.mir_program.functions:
            nesting = compute_block_nesting(func)
            for block_id, block in func.blocks.items():
                depth = nesting.get(block_id, 0)
                for instr in block.instructions:
                    if isinstance(instr, Call) and isinstance(instr.function, str):
                        name = instr.function
                        if name in self.call_counts:
                            self.call_counts[name] += 1
                            if depth > self.call_depths.get(name, 0):
                                self.call_depths[name] = depth

    def _find_recursive_functions(self):
        """Find all functions involved in recursion (direct or mutual)."""
        self.recursive_functions: Set[str] = set()

        # Check for direct recursion
        for func_name, called in self.call_graph.items():
            if func_name in called:
                self.recursive_functions.add(func_name)

        # Check for mutual recursion using transitive closure
        for start_func in self.call_graph:
            visited = set()
            stack = list(self.call_graph.get(start_func, set()))

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current == start_func:
                    self.recursive_functions.add(start_func)
                    break

                stack.extend(self.call_graph.get(current, set()))

    def _estimate_cycle_cost(self, func: MIRFunction) -> int:
        """Estimate the 65816 cycle cost of a function body when inlined.

        Uses a per-instruction-type weight table. Return costs 0 because it is
        eliminated by inlining. The result is compared against CALL_OVERHEAD_CYCLES
        (12 for JSR/RTS) to decide whether inlining pays off.
        """
        cost = 0
        for block in func.blocks.values():
            for instr in block.instructions:
                cost += _MIR_CYCLE_COSTS.get(type(instr), _DEFAULT_CYCLE_COST)
        return cost

    def _has_inline_asm(self, func: MIRFunction) -> bool:
        """Check if function contains inline assembly."""
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, InlineAsm):
                    return True
        return False

    def _far_body_is_bank_safe(self, func: MIRFunction) -> bool:
        """True iff every instruction in a far function's body would
        produce identical machine code if assembled in a different bank.

        Used by `can_inline` to decide whether to allow inlining a far
        callee into a caller in some other bank. Conservative — rejects
        anything we can't statically prove bank-independent. The cases
        that pass:

        - **Far indirects** (LoadIndirect/StoreIndirect with is_far=True):
          use [zp],Y / [zp] — fully bank-explicit via the pointer's bank
          byte.
        - **Zero-page Load/Store** (storage_type='zeropage'): DP-relative,
          DBR-independent.
        - **WRAM Load/Store with bank-resolved address ≥ $010000**
          (storage_type='ram', address >= 0x010000): codegen always emits
          long addressing because the address is wider than 16 bits.

        Rejected:

        - **Near indirects** (is_far=False): use DBR, which differs
          between caller's and callee's banks.
        - **Any Call** to a non-far function in the body — `JSR` targets
          the current PBR; after inlining that PBR is the caller's, so
          the JSR lands in the wrong bank.
        - **`#[mode(databank=...)]`** other than NONE: the callee relies
          on a prologue PHB/PLB to set DBR to its own bank, and may use
          absolute addressing inside that bracket; we'd need to emit
          equivalent MIR-level DBR management at the inline boundary,
          which isn't implemented.
        - **Anything else with a `MemoryLocation` operand** (Load/Store
          to ROM/HW/unresolved RAM, BinaryOp on a MemoryLocation,
          MemoryFill/BlockCopy): codegen may pick absolute or
          DBR-relative addressing based on the callee's expected DBR;
          inlining into a different bank could silently miscompile.

        Returns False on any unrecognized instruction kind too — failing
        closed is the right default for a cross-bank correctness check.
        """
        from r65.compiler.hir.attributes import DataBankMode

        # databank=inline / databank=caller can't be replicated in
        # body-only MIR without emitting explicit PHB/PLB sequences;
        # reject for now.
        if func.mode_attr is not None and getattr(
                func.mode_attr, 'databank', DataBankMode.NONE) != DataBankMode.NONE:
            return False

        # Far-pointer stack params force the codegen to emit a prologue
        # bracket (PHB / set DBR from ptr bank / PLB, or PHD/TSC/TCD)
        # that the body's `(d,S),Y` or `[dp],Y` indirect derefs depend on
        # for DBR. Inlining removes the prologue but the body's
        # addressing is still relative to that bank. Until we model the
        # DBR setup at the inline boundary, refuse. (Concretely: this
        # blocks put_str / put_num / similar utilities. Lifting it
        # requires emitting MIR-level Push DBR / set DBR / Pull DBR
        # around the inlined body — left for a follow-up.)
        if func.has_far_ptr_stack_params:
            return False

        for block in func.blocks.values():
            for instr in block.instructions:
                if not self._instr_is_bank_safe(instr):
                    return False
        return True

    def _instr_is_bank_safe(self, instr: MIRInstruction) -> bool:
        """Per-instruction bank-safety predicate; see `_far_body_is_bank_safe`."""
        # Calls: only far calls are bank-explicit. Near calls would JSR
        # in the caller's bank — wrong target.
        if isinstance(instr, Call):
            return instr.is_far
        if isinstance(instr, TraitDispatch):
            # Trait dispatch's bank semantics are complex and tied to
            # self_is_far; refuse for now.
            return False

        # Indirect memory access: bank-explicit only if far.
        if isinstance(instr, (LoadIndirect, StoreIndirect)):
            return instr.is_far

        # Direct memory access: bank-safe only when codegen will emit
        # long addressing (address >= 0x010000) or DP (zeropage).
        if isinstance(instr, (Load, Store)):
            loc = instr.source if isinstance(instr, Load) else instr.dest
            return self._mem_loc_is_bank_safe(loc)

        # Operands that may carry MemoryLocation references. Reject if any
        # bank-dependent location appears.
        for op in self._maybe_memory_operands(instr):
            if not self._mem_loc_is_bank_safe(op):
                return False

        # MemoryFill / BlockCopy: target is bank-pinned; codegen uses MVN
        # with explicit dest_bank, but inlining into a different bank
        # caller hasn't been validated. Refuse.
        if isinstance(instr, (MemoryFill, BlockCopy)):
            return False

        # Everything else (BinaryOp/UnaryOp on registers, Move, Jump,
        # CondBranch, SetMode, Return, etc.) is bank-independent at the
        # body level.
        return True

    def _maybe_memory_operands(self, instr: MIRInstruction):
        """Yield MemoryLocation operands of `instr`. Used to scan
        BinaryOp/UnaryOp/Compare/etc. whose source slots can hold a
        MemoryLocation directly (avoids loading into a temp vreg)."""
        for field in ('left', 'right', 'operand', 'source', 'value'):
            op = getattr(instr, field, None)
            if isinstance(op, MemoryLocation):
                yield op

    def _mem_loc_is_bank_safe(self, loc) -> bool:
        """Decide whether codegen will emit bank-independent addressing
        for this MemoryLocation."""
        if not isinstance(loc, MemoryLocation):
            return True  # non-memory operand, irrelevant
        if loc.storage_type == 'zeropage':
            return True  # DP — bank-independent
        if loc.storage_type == 'ram' and loc.address is not None and loc.address >= 0x010000:
            # WRAM with a resolved address >= 64K → codegen always uses
            # long addressing (the address itself is wider than the
            # absolute mode can encode).
            return True
        # ROM / HW / lowram / unresolved RAM: may use abs (3-byte) which
        # depends on DBR. Refuse.
        return False

    def can_inline(self, func_name: str) -> bool:
        """
        Check if a function can be inlined (hard requirements).

        Returns:
            True if the function passes all hard requirements for inlining.
        """
        if func_name not in self.func_map:
            return False

        func = self.func_map[func_name]

        # Not recursive
        if func_name in self.recursive_functions:
            return False

        # Far functions: allowed when the body is bank-independent, i.e.
        # every instruction would produce the same bytes regardless of
        # which PBR/DBR is active at runtime. The blanket "no far fn
        # inlining" rule was overly conservative — many far utilities
        # (string copy, indexed nametable write, etc.) only touch far
        # pointers or WRAM via long addressing, both of which work
        # unchanged in any caller's bank. The detailed safety check is
        # in `_far_body_is_bank_safe`.
        if func.is_far and not self._far_body_is_bank_safe(func):
            return False

        # Not an interrupt handler
        if func.interrupt_attr is not None:
            return False

        # Not an entry point
        if func.is_entry:
            return False

        # Not __init_start
        if func_name == "__init_start":
            return False

        # No inline assembly
        if self._has_inline_asm(func):
            return False

        # Trait methods receive self in Y register (`is_trait_method` plus
        # `self_y_vreg` pre-allocated to HW Y). Splicing the body into a
        # caller breaks the "self lives in Y" contract — the prologue setup
        # is callee-specific and the body's self_y_vreg has no meaning in
        # the caller's vreg space.
        if func.is_trait_method:
            return False

        # Scratch-param promotion (analyze_scratch_params / fixedstack_params)
        # reserves direct-page slots tied to the callee's signature. The
        # caller's frame doesn't reflect those reservations, and the global
        # scratch reservation set lives on the function being processed.
        # Rather than merge the maps (and risk overlapping reservations),
        # refuse to inline callees with promoted params. The expression
        # lowerer never reaches them anyway except via the call ABI.
        if func.scratch_param_addrs:
            return False
        if func.hw_param_regs:
            return False

        # Far-pointer access strategy (D=S vs SET_DBR) is chosen per-function
        # by analysis/far_ptr_strategy.py based on body characteristics
        # (prologue PHD/TSC/TCD vs PHB/PLB). LoadIndirect/StoreIndirect
        # lowering reads `caller.far_ptr_strategy`, so splicing a callee
        # configured for the opposite strategy yields wrong addressing.
        # Reject any callee with a non-default strategy.
        if func.far_ptr_strategy is not None:
            return False

        # Callee's #[preserves(...)] contract is enforced at the call boundary
        # by codegen (save/restore around the JSR). Inlining removes the
        # boundary; any caller continuation that observed the preserved
        # registers across the call is now broken. The fix requires
        # emitting equivalent SaveRegister/RestoreRegister around the
        # inlined body — until that's implemented, refuse to inline.
        if func.preserves_attr is not None:
            return False

        return True

    def _bank_compatible(self, caller: MIRFunction, callee: MIRFunction) -> bool:
        """True iff the callee can be inlined into the caller without
        breaking same-bank label assumptions.

        Per CLAUDE.md, near functions can only call near functions in the
        same bank. Inlining a NEAR callee whose `#[bank(n)]` differs
        from the caller's would assemble the body inside the caller's
        bank, breaking same-bank references the body relied on.

        Far callees are exempt: cross-bank inlining is exactly the point
        of relaxing the far-fn rule. `_far_body_is_bank_safe` has
        already verified the body produces identical bytes in any bank.

        Auto-banked callees (`bank_attr.is_auto`) are permitted for
        near callees too: the linker places them with the caller.
        """
        if callee.is_far:
            # Body-level bank safety is enforced by _far_body_is_bank_safe;
            # cross-bank is the whole point.
            return True
        callee_bank = callee.bank_attr
        if callee_bank is None or callee_bank.is_auto:
            return True
        caller_bank = caller.bank_attr
        if caller_bank is None or caller_bank.is_auto:
            # Caller has no fixed bank; inlining a fixed-bank callee
            # would pin the caller's containing block to that bank,
            # which the caller didn't ask for. Conservative: reject.
            return False
        return callee_bank.bank_number == caller_bank.bank_number

    def should_inline(self, func_name: str) -> bool:
        """
        Check if a function should be inlined (heuristics).

        Returns:
            True if the function should be inlined based on heuristics.
        """
        if not self.can_inline(func_name):
            return False

        func = self.func_map[func_name]
        body_cost = self._estimate_cycle_cost(func)

        # Check for #[inline(never)] - never inline these functions
        if func.inline_attr is not None and func.inline_attr.mode == InlineMode.NEVER:
            return False

        # Marked #[inline] or #[inline(always)] → inline if under cycle budget
        if func.inline_attr is not None:
            return body_cost < INLINE_COST_WITH_ATTR

        # Below this point is implicit inlining - only allowed when implicit_inline=True
        if not self.implicit_inline:
            return False

        # Called exactly once → inline if not absurdly large. The "no code
        # size increase" reasoning that lets us bypass the regular budget
        # still applies, but huge called-once callees inflate the caller's
        # frame past what the slot allocator can untangle (observed in
        # classickong.r65 with unpack_level — a 700-line function whose
        # locals collided with the caller's at slot $07,S). Capping at
        # the same limit as explicitly-attributed functions keeps the win
        # for small called-once helpers without breaking large ones.
        #
        # NOTE(slot-allocator-latent): the underlying liveness/slot-reuse
        # bug is still there — synthetic small repros (one large called-
        # once callee + caller with several u16 locals, see /tmp tests in
        # the inlining session) do NOT trigger it, so a targeted fix
        # would need a minimal reduction from the classickong.r65 case.
        # The cap is the practical mitigation; revisit if a smaller repro
        # surfaces.
        if self.call_counts.get(func_name, 0) == 1:
            return body_cost <= INLINE_COST_WITH_ATTR

        # Apply loop-depth multiplier: a call inside a loop is worth inlining
        # at a larger budget since JSR/RTS overhead repeats every iteration.
        depth = self.call_depths.get(func_name, 0)
        factor = _LOOP_DEPTH_MULTIPLIER[min(depth, 2)]
        limit = min(INLINE_COST_NO_ATTR * factor, INLINE_COST_WITH_ATTR)
        return body_cost <= limit


class BlockCloner:
    """
    Clones basic blocks with fresh virtual register and block IDs.

    Handles remapping of:
    - Virtual register IDs
    - Block IDs (for jump targets)
    - Preserves hardware register references
    """

    def __init__(self, caller_func: MIRFunction, callee_func: MIRFunction):
        """
        Initialize the block cloner.

        Args:
            caller_func: The function receiving the inlined code
            callee_func: The function being inlined
        """
        self.caller_func = caller_func
        self.callee_func = callee_func
        self.vreg_map: Dict[int, VirtualRegister] = {}
        self.block_map: Dict[int, int] = {}

    def _get_next_block_id(self) -> int:
        """Get the next available block ID in the caller function."""
        if not self.caller_func.blocks:
            return 0
        return max(self.caller_func.blocks.keys()) + 1

    def _remap_vreg(self, vreg: VirtualRegister) -> VirtualRegister:
        """
        Get or create a remapped virtual register.

        Args:
            vreg: Original virtual register from callee

        Returns:
            New virtual register in caller's address space
        """
        if vreg.id not in self.vreg_map:
            new_vreg = self.caller_func.vreg_allocator.alloc(
                vreg.type_info,
                f"inlined_{vreg.hint}" if vreg.hint else None
            )
            # Carry over dynamic attributes the expression lowerer sets on
            # vregs that hold address-of-ROM-data values (`.symbol` →
            # `rom_label`). Without this, the near→far codegen at
            # type_conversion_select.py:425 sees a vreg with no `.symbol`,
            # falls back to bank $00, and the resulting far pointer aims
            # at bank 0 instead of bank 8 (classickong.r65 -O2:
            # title_screen_press_start_str inlining → PRESS START
            # rendered from a wild far-ptr read).
            if hasattr(vreg, 'symbol') and vreg.symbol is not None:
                new_vreg.symbol = vreg.symbol
            self.vreg_map[vreg.id] = new_vreg
        return self.vreg_map[vreg.id]

    def _remap_operand(self, operand):
        """
        Remap an operand, handling virtual registers.

        Args:
            operand: Any MIR operand

        Returns:
            Remapped operand
        """
        if isinstance(operand, VirtualRegister):
            return self._remap_vreg(operand)
        elif isinstance(operand, (HardwareRegister, Immediate)):
            return operand
        else:
            # Memory locations and other operands pass through
            return operand

    def _remap_block_id(self, block_id: int) -> int:
        """
        Get or create a remapped block ID.

        Args:
            block_id: Original block ID from callee

        Returns:
            New block ID in caller's block space
        """
        if block_id not in self.block_map:
            # Find a new block ID that doesn't conflict
            new_id = self._get_next_block_id()
            # Get existing mapped values (excluding the one we're about to add)
            existing_mapped = set(self.block_map.values())
            # Find an ID that's not in caller's blocks and not already mapped
            while new_id in self.caller_func.blocks or new_id in existing_mapped:
                new_id += 1
            self.block_map[block_id] = new_id
        return self.block_map[block_id]

    def _clone_argument(self, arg: Argument) -> Argument:
        """Construct a fresh Argument with the value operand remapped.

        `location` / `param_type` / `scratch_addr` / `mechanism` are
        pointer-stable descriptors and may be shared between original
        and clone.
        """
        return Argument(
            value=self._remap_operand(arg.value),
            mechanism=arg.mechanism,
            location=arg.location,
            param_type=arg.param_type,
            scratch_addr=arg.scratch_addr,
        )

    def _clone_instruction(self, instr: MIRInstruction) -> MIRInstruction:
        """
        Clone an instruction with remapped operands.

        Constructs a fresh instance for each known MIR instruction type so
        that no part of the cloned instruction aliases the callee's
        objects. Raises NotImplementedError for unknown types — the
        previous `deepcopy` fallback silently preserved callee-side
        VirtualRegister objects in nested fields and caused vreg-id
        collisions; failing loudly is the right call.

        Args:
            instr: Original instruction from callee

        Returns:
            Cloned instruction with remapped operands
        """
        loc = instr.source_loc

        if isinstance(instr, Move):
            return Move(
                dest=self._remap_operand(instr.dest),
                source=self._remap_operand(instr.source),
                type_info=instr.type_info,
                persist_16bit_mode=instr.persist_16bit_mode,
                source_loc=loc,
            )

        if isinstance(instr, Load):
            return Load(
                dest=self._remap_operand(instr.dest),
                source=instr.source,  # MemoryLocation: shared
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, Store):
            return Store(
                source=self._remap_operand(instr.source),
                dest=instr.dest,  # MemoryLocation: shared
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, LoadIndirect):
            return LoadIndirect(
                dest=self._remap_operand(instr.dest),
                pointer=self._remap_operand(instr.pointer),
                is_far=instr.is_far,
                index_register=instr.index_register,
                type_info=instr.type_info,
                offset=instr.offset,
                source_loc=loc,
            )

        if isinstance(instr, StoreIndirect):
            return StoreIndirect(
                source=self._remap_operand(instr.source),
                pointer=self._remap_operand(instr.pointer),
                is_far=instr.is_far,
                index_register=instr.index_register,
                type_info=instr.type_info,
                offset=instr.offset,
                source_loc=loc,
            )

        if isinstance(instr, BinaryOp):
            return BinaryOp(
                dest=self._remap_operand(instr.dest),
                left=self._remap_operand(instr.left),
                right=self._remap_operand(instr.right),
                op=instr.op,
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, UnaryOp):
            return UnaryOp(
                dest=self._remap_operand(instr.dest),
                operand=self._remap_operand(instr.operand),
                op=instr.op,
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, Compare):
            return Compare(
                left=self._remap_operand(instr.left),
                right=self._remap_operand(instr.right),
                comparison=instr.comparison,
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, TypeConvert):
            return TypeConvert(
                dest=self._remap_operand(instr.dest),
                source=self._remap_operand(instr.source),
                source_type=instr.source_type,
                target_type=instr.target_type,
                source_loc=loc,
            )

        if isinstance(instr, BankByte):
            return BankByte(
                dest=self._remap_operand(instr.dest),
                source=self._remap_operand(instr.source),
                source_loc=loc,
            )

        if isinstance(instr, ToBool):
            return ToBool(
                dest=self._remap_operand(instr.dest),
                source=self._remap_operand(instr.source),
                source_type=instr.source_type,
                source_loc=loc,
            )

        if isinstance(instr, BitTest):
            return BitTest(
                value=self._remap_operand(instr.value),
                test_bit=instr.test_bit,
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, Rotate):
            return Rotate(
                dest=self._remap_operand(instr.dest),
                source=self._remap_operand(instr.source),
                direction=instr.direction,
                count=instr.count,
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, Jump):
            return Jump(
                target=self._remap_block_id(instr.target),
                source_loc=loc,
            )

        if isinstance(instr, CondBranch):
            return CondBranch(
                condition=self._remap_operand(instr.condition),
                true_target=self._remap_block_id(instr.true_target),
                false_target=self._remap_block_id(instr.false_target),
                comparison=instr.comparison,
                source_loc=loc,
            )

        if isinstance(instr, JumpTable):
            return JumpTable(
                scrutinee=self._remap_operand(instr.scrutinee),
                base_value=instr.base_value,
                targets=[self._remap_block_id(t) for t in instr.targets],
                default_target=self._remap_block_id(instr.default_target),
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, LookupTable):
            return LookupTable(
                dest=self._remap_operand(instr.dest),
                scrutinee=self._remap_operand(instr.scrutinee),
                base_value=instr.base_value,
                values=list(instr.values),
                default_value=instr.default_value,
                merge_target=self._remap_block_id(instr.merge_target),
                type_info=instr.type_info,
                source_loc=loc,
            )

        if isinstance(instr, StatusFlagTest):
            return StatusFlagTest(
                flag_name=instr.flag_name,
                bit_position=instr.bit_position,
                bit_mask=instr.bit_mask,
                source_loc=loc,
            )

        if isinstance(instr, StatusFlagSet):
            return StatusFlagSet(
                flag_name=instr.flag_name,
                value=instr.value,
                source_loc=loc,
            )

        if isinstance(instr, StatusFlagRead):
            return StatusFlagRead(
                dest=self._remap_operand(instr.dest),
                flag_name=instr.flag_name,
                bit_mask=instr.bit_mask,
                source_loc=loc,
            )

        if isinstance(instr, Call):
            return Call(
                function=(self._remap_operand(instr.function)
                          if isinstance(instr.function, VirtualRegister)
                          else instr.function),
                args=[self._clone_argument(a) for a in instr.args],
                returns=[self._remap_operand(r) for r in instr.returns],
                is_far=instr.is_far,
                mode_attr=instr.mode_attr,
                bank_attr=instr.bank_attr,
                builtin_name=instr.builtin_name,
                callee_entry_m_mode=instr.callee_entry_m_mode,
                callee_exit_m_mode=instr.callee_exit_m_mode,
                callee_return_type=instr.callee_return_type,
                preserves_attr=instr.preserves_attr,
                pascal_result_bytes=instr.pascal_result_bytes,
                source_loc=loc,
            )

        if isinstance(instr, TraitDispatch):
            # TraitDispatch shouldn't survive can_inline (callee is a trait
            # method or contains a dispatch — both rejected). Cloned here
            # only so a body containing a nested TraitDispatch to a
            # different trait can be inlined safely if policy ever
            # permits it.
            return TraitDispatch(
                trait_name=instr.trait_name,
                method_name=instr.method_name,
                method_index=instr.method_index,
                self_ptr=(self._remap_operand(instr.self_ptr)
                          if instr.self_ptr is not None else None),
                args=[self._clone_argument(a) for a in instr.args],
                returns=[self._remap_operand(r) for r in instr.returns],
                is_far=instr.is_far,
                self_is_far=instr.self_is_far,
                callee_return_type=instr.callee_return_type,
                self_chain_role=instr.self_chain_role,
                self_y_preloaded=instr.self_y_preloaded,
                source_loc=loc,
            )

        if isinstance(instr, Return):
            return Return(
                values=[self._remap_operand(v) for v in instr.values],
                source_loc=loc,
            )

        if isinstance(instr, ReturnFromInterrupt):
            return ReturnFromInterrupt(source_loc=loc)

        if isinstance(instr, SetMode):
            return SetMode(mask=instr.mask, is_set=instr.is_set, source_loc=loc)

        if isinstance(instr, Push):
            # register field is HardwareRegister (shareable)
            return Push(register=instr.register, source_loc=loc)

        if isinstance(instr, Pull):
            return Pull(register=instr.register, source_loc=loc)

        if isinstance(instr, SaveRegister):
            return SaveRegister(
                register=instr.register,
                save_location=self._remap_operand(instr.save_location),
                source_loc=loc,
            )

        if isinstance(instr, RestoreRegister):
            return RestoreRegister(
                register=instr.register,
                save_location=self._remap_operand(instr.save_location),
                source_loc=loc,
            )

        if isinstance(instr, MemoryFill):
            return MemoryFill(
                dest=instr.dest,  # MemoryLocation: shared
                fill_value=instr.fill_value,
                count=instr.count,
                element_size=instr.element_size,
                source_loc=loc,
            )

        if isinstance(instr, BlockCopy):
            return BlockCopy(
                dest=instr.dest,         # MemoryLocation: shared
                rom_data=instr.rom_data, # ROMDataRef: shared
                count=instr.count,
                source_loc=loc,
            )

        if isinstance(instr, InlineAsm):
            # can_inline() rejects callees containing InlineAsm; reaching
            # here means that check was bypassed. Fail loudly rather than
            # produce an aliased clone.
            raise NotImplementedError(
                "InlineAsm cannot be inlined; can_inline() should have "
                "rejected this callee"
            )

        raise NotImplementedError(
            f"FunctionInliner._clone_instruction: unhandled MIR instruction "
            f"type {type(instr).__name__}. Add an explicit clone branch "
            f"(or extend can_inline() to reject callees containing it)."
        )

    def clone_blocks(self) -> Dict[int, BasicBlock]:
        """
        Clone all blocks from the callee with remapped IDs.

        Returns:
            Dictionary of new block ID to cloned BasicBlock
        """
        cloned_blocks = {}

        # First pass: assign new block IDs
        for old_id in self.callee_func.blocks:
            self._remap_block_id(old_id)

        # Second pass: clone blocks with remapped instructions
        for old_id, block in self.callee_func.blocks.items():
            new_id = self.block_map[old_id]

            # entry_mode / exit_mode are deliberately NOT cloned. The MIR
            # built for the callee carried mode metadata computed from the
            # callee's own entry mode (set by MIRModeTracker during MIR
            # build). After splicing those blocks into the caller, the
            # join points and mode flow may be entirely different, so the
            # cached metadata is by definition stale. The caller side
            # of FunctionInliner.run() invalidates this metadata and
            # mode_tracker.reanalyze_function() recomputes it. See the
            # inliner's `mutated_funcs` contract.
            new_block = BasicBlock(
                block_id=new_id,
                instructions=[self._clone_instruction(i) for i in block.instructions],
                predecessors=[],  # Will be rebuilt
                successors=[],    # Will be rebuilt
            )

            cloned_blocks[new_id] = new_block

        # Third pass: update predecessor/successor relationships
        for old_id, block in self.callee_func.blocks.items():
            new_id = self.block_map[old_id]
            new_block = cloned_blocks[new_id]

            new_block.predecessors = [self.block_map[p] for p in block.predecessors
                                      if p in self.block_map]
            new_block.successors = [self.block_map[s] for s in block.successors
                                    if s in self.block_map]

        return cloned_blocks

    def get_entry_block_id(self) -> int:
        """Get the remapped entry block ID."""
        return self.block_map[self.callee_func.entry_block_id]


class FunctionInliner:
    """
    Main function inlining pass.

    Algorithm:
    1. Count calls to each function
    2. Find inline candidates (call sites where should_inline() returns true)
    3. For each candidate (bottom-up order):
       a. Clone callee's basic blocks with fresh vreg/block IDs
       b. Set up parameter bindings (map callee params to caller args)
       c. Replace Return instructions with Move + Jump to merge block
       d. Split call block: pre-call instructions → jump to inlined entry
       e. Create merge block with post-call instructions
       f. Add cloned blocks to caller's CFG
    """

    def __init__(self, verbose: bool = False, implicit_inline: bool = True):
        """
        Initialize the function inliner.

        Args:
            verbose: If True, print information about inlined functions
            implicit_inline: If True, allow implicit inlining (called-once and small
                           functions). If False, only inline explicitly marked functions.
        """
        self.verbose = verbose
        self.implicit_inline = implicit_inline
        # Names of caller functions whose MIR was mutated by inlining. After
        # run() returns, every name in this set has invalidated per-block
        # metadata (entry_mode / exit_mode) and the caller must re-run any
        # pass that decorates blocks — currently MIRModeTracker via
        # `mode_tracker.reanalyze_function()`. Mirrors the LLVM /
        # rustc convention of treating inlining as analysis-invalidating.
        self.mutated_funcs: Set[str] = set()

    def run(self, mir_program: MIRProgram) -> int:
        """
        Run the inlining pass on the MIR program.

        Args:
            mir_program: The MIR program to optimize

        Returns:
            Number of call sites inlined.

        After this method returns, every function named in
        `self.mutated_funcs` has invalidated per-block mode metadata.
        Callers MUST invoke `mode_tracker.reanalyze_function()` on each
        one before any pass that reads `BasicBlock.entry_mode` /
        `BasicBlock.exit_mode` runs.
        """
        checker = InlinabilityChecker(mir_program, implicit_inline=self.implicit_inline)
        inlined_count = 0

        # Process callers in reverse-topological order on the static call
        # graph (callees before their callers). This gives nested inlining
        # in a single pass: by the time we process function A that calls B
        # that calls C, B's body already has C expanded (if C was
        # inlinable), so cloning B into A also picks up the C expansion.
        # No need to iterate to a fixed point.
        order = self._reverse_topo_order(checker)

        for func_name in order:
            caller_func = checker.func_map.get(func_name)
            if caller_func is None:
                continue

            # Repeatedly inline call sites in this caller until none remain.
            # Each successful inline mutates the caller's CFG (new blocks,
            # shifted instruction indices), so we rescan the function rather
            # than try to maintain a stable index. The inner loop terminates
            # because each iteration either inlines a site (strictly reduces
            # the count of inlinable call sites pointing at non-recursive
            # callees) or finds no candidate.
            while True:
                site = self._find_first_inline_site(caller_func, checker)
                if site is None:
                    break
                block_id, instr_idx, call_instr = site
                callee_name = call_instr.function
                callee_func = checker.func_map.get(callee_name)
                if callee_func is None:
                    break
                if not self._inline_call(caller_func, block_id, instr_idx,
                                         call_instr, callee_func):
                    break
                inlined_count += 1
                self.mutated_funcs.add(caller_func.name)
                # Decrement the call count incrementally; this is a worklist
                # bookkeeping update, not a full checker rebuild.
                checker.call_counts[callee_name] = max(
                    0, checker.call_counts.get(callee_name, 1) - 1
                )
                if self.verbose:
                    print(f"  Inlined {callee_name} into {caller_func.name}")

        if self.verbose and inlined_count > 0:
            print(f"Function inlining: {inlined_count} call site(s) inlined")

        return inlined_count

    def _reverse_topo_order(self, checker: InlinabilityChecker) -> List[str]:
        """Return function names in reverse-topological order on the call
        graph.

        Postorder DFS: callees emit before callers. Cycles (recursive
        functions) are broken by an `in_stack` guard — those callees can
        never be inlined anyway (`can_inline` rejects them), so the order
        of their callers doesn't affect correctness.
        """
        visited: Set[str] = set()
        in_stack: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited or name in in_stack:
                return
            in_stack.add(name)
            for callee in checker.call_graph.get(name, set()):
                if callee in checker.func_map:
                    visit(callee)
            in_stack.discard(name)
            visited.add(name)
            order.append(name)

        for func in checker.mir_program.functions:
            visit(func.name)
        return order

    def _find_first_inline_site(
        self,
        func: MIRFunction,
        checker: InlinabilityChecker,
    ) -> Optional[Tuple[int, int, Call]]:
        """Return the first (block_id, instr_idx, Call) the inliner wants
        to expand in `func`, or None if no such site exists."""
        for block_id, block in func.blocks.items():
            for idx, instr in enumerate(block.instructions):
                if (isinstance(instr, Call)
                        and isinstance(instr.function, str)
                        and self._should_inline_at_site(checker, func, instr)):
                    return (block_id, idx, instr)
        return None

    def _all_args_constant(self, call: Call) -> bool:
        """Return True if every argument is a plain compile-time integer literal.

        Symbolic Immediates (e.g. from &ARR[i]) carry a 'symbol' attribute
        and are excluded — their .value is an offset, not an absolute address.
        """
        return bool(call.args) and all(
            isinstance(arg.value, Immediate)
            and not getattr(arg.value, 'symbol', None)
            for arg in call.args
        )

    def _callee_has_no_calls(self, func: MIRFunction) -> bool:
        """Return True if the callee body contains no nested Call instructions."""
        return not any(
            isinstance(instr, Call)
            for block in func.blocks.values()
            for instr in block.instructions
        )

    def _should_inline_at_site(
        self,
        checker: InlinabilityChecker,
        caller: MIRFunction,
        call: Call,
    ) -> bool:
        """Decide whether to inline this specific call site.

        Extends the function-level should_inline() heuristic with a call-site
        bypass: when every argument is a compile-time constant and the callee
        has no nested calls, inlining always pays off because the post-inline
        peephole can constant-fold the body down to 1-2 instructions.
        """
        func_name = call.function
        if not isinstance(func_name, str):
            return False
        callee = checker.func_map.get(func_name)
        if callee is None:
            return False
        # Bank compatibility is per-(caller,callee), not a property of the
        # callee alone — must be checked at the call site.
        if not checker._bank_compatible(caller, callee):
            return False
        if checker.should_inline(func_name):
            return True
        # Constant-arg bypass is an implicit heuristic — only at -O2
        if (checker.implicit_inline
                and checker.can_inline(func_name)
                and self._all_args_constant(call)
                and self._callee_has_no_calls(callee)):
            return True
        return False

    def _inline_call(
        self,
        caller: MIRFunction,
        block_id: int,
        instr_idx: int,
        call: Call,
        callee: MIRFunction
    ) -> bool:
        """
        Inline a single call site.

        Args:
            caller: Calling function
            block_id: Block containing the call
            instr_idx: Index of call instruction in block
            call: The Call instruction
            callee: Function being called

        Returns:
            True if inlining was successful
        """
        call_block = caller.blocks[block_id]

        # Clone callee's blocks
        cloner = BlockCloner(caller, callee)
        cloned_blocks = cloner.clone_blocks()
        inlined_entry_id = cloner.get_entry_block_id()

        # Override source_loc on all inlined instructions to use call site location
        # This ensures debug info points to where the function was called, not defined
        call_site_loc = call.source_loc
        if call_site_loc:
            for block in cloned_blocks.values():
                for instr in block.instructions:
                    instr.source_loc = call_site_loc

        # Create merge block for code after the call
        merge_block_id = max(caller.blocks.keys()) + 1
        while merge_block_id in cloned_blocks:
            merge_block_id += 1

        # Split instructions: before call, call itself, after call
        pre_call_instrs = call_block.instructions[:instr_idx]
        post_call_instrs = call_block.instructions[instr_idx + 1:]

        # Create merge block with post-call instructions
        merge_block = BasicBlock(
            block_id=merge_block_id,
            instructions=list(post_call_instrs),
            predecessors=[],  # Will be filled in
            successors=list(call_block.successors),
        )

        # Compute the mode the caller is in at the call site. The non-inlined
        # call path brackets JSR with SEP/REP to bridge caller→callee.entry
        # and back; we need the same bridging in MIR so merge blocks see
        # consistent predecessor modes. Walk pre_call_instrs and apply each
        # SetMode's effect — no other MIR instruction in an inlinable
        # function mutates the M flag (Push/Pull of STATUS only appears in
        # interrupt handlers and asm wrappers, both non-inlinable).
        caller_mode_at_call = self._compute_caller_mode_at_call(
            caller, call_block, pre_call_instrs
        )

        # Update call block: keep pre-call instructions, add jump to inlined
        # entry, and insert an entry-boundary SetMode just before the jump
        # if the callee expects a different M mode.
        call_block.instructions = list(pre_call_instrs)
        entry_setmode = self._make_boundary_setmode(
            from_mode=caller_mode_at_call.m_mode,
            to_mode=getattr(callee, 'entry_m_mode', None),
        )
        if entry_setmode is not None:
            call_block.instructions.append(entry_setmode)
        call_block.instructions.append(Jump(target=inlined_entry_id))
        call_block.successors = [inlined_entry_id]

        # Identify which hardware registers the callee observes. Two sources:
        #   (a) Save Moves at the top of the entry block — the MIR builder
        #       emits Move(saved_vreg, HW_REG) for every register-bound
        #       parameter referenced by symbol in the body.
        #   (b) Direct reads of the HW register anywhere in the body — the
        #       source can refer to a register by its hardware name (e.g.
        #       `self.lo = X` for `value @ X`), which never goes through the
        #       saved vreg.
        # If a param's HW register isn't observed by either path, the load is
        # dead and the inliner can skip it.
        inlined_entry = cloned_blocks[inlined_entry_id]
        reg_save_count = 0
        live_hw_regs: Set[str] = set()
        for instr in inlined_entry.instructions:
            if (isinstance(instr, Move) and
                isinstance(instr.source, HardwareRegister) and
                isinstance(instr.dest, VirtualRegister)):
                live_hw_regs.add(instr.source.name)
                reg_save_count += 1
            else:
                break
        live_hw_regs.update(self._collect_read_hw_reg_names(cloned_blocks))

        # Identify which stack-param vregs are actually read in the cloned
        # body. A stack param vreg with no reads is dead — skip its setup
        # Move too. We compute this on the cloned blocks, post-Return-rewrite
        # below would obscure it, so do it before that pass touches Returns.
        used_vreg_ids = self._collect_read_vreg_ids(cloned_blocks)

        # Set up parameter bindings for inlined code.
        # Returns two lists: register_instrs (set up hw regs) and stack_instrs
        # (bind stack params to vregs). Unused params are skipped.
        register_instrs, stack_instrs = self._create_param_bindings(
            call, callee, cloner,
            saved_hw_regs=live_hw_regs,
            used_vreg_ids=used_vreg_ids,
        )

        # Insert instructions in the correct order at the inlined entry block:
        #   1. register_instrs: Load argument values into hardware registers
        #      (e.g., LDA #33 for val @ A)
        #   2. reg_save instructions from callee: Move(dest=vreg, source=HardwareRegister)
        #      (e.g., save A to saved_val_vreg before anything clobbers A)
        #   3. stack_instrs: Bind stack param arguments to callee vregs
        #      (e.g., Move(dest=ptr_vreg, source=Immediate(addr)) — may clobber A in codegen)
        #   4. Rest of callee body
        inlined_entry.instructions = (
            register_instrs +
            inlined_entry.instructions[:reg_save_count] +
            stack_instrs +
            inlined_entry.instructions[reg_save_count:]
        )
        inlined_entry.predecessors.append(block_id)

        # Result vregs in the caller's space — one per return value.
        # Multi-return callees (e.g. `-> rA, rB` or `-> rA, rX`) plumb each
        # Return.values[i] into call.returns[i]; previously only [0] was
        # emitted, silently dropping the second value.
        result_vregs = list(call.returns)

        # Exit-boundary SetMode: switches back to the caller's mode at
        # every Return so the merge block has a single predecessor mode
        # (mirrors call_select.py's _emit_exit_mode_restore). Computed
        # once — every Return uses the same transition.
        exit_setmode = self._make_boundary_setmode(
            from_mode=getattr(callee, 'exit_m_mode', None),
            to_mode=caller_mode_at_call.m_mode,
        )

        # Replace Return instructions with Move + Jump to merge block
        return_blocks = []

        for cloned_block in cloned_blocks.values():
            new_instructions = []
            for instr in cloned_block.instructions:
                if isinstance(instr, Return):
                    # Pair each Return value with its caller-side
                    # destination. The A-substitution optimization
                    # (return-value still in A from the tail instruction)
                    # applies only to the primary return; subsequent
                    # values live in B/X/Y per the ABI's multi-return
                    # convention and aren't "in A" at the Return point.
                    # And it's disabled when an exit-boundary SetMode is
                    # pending: the mode change between this block and the
                    # merge can re-interpret A's width, and the codegen
                    # for the substituted Move may emit its own bracket
                    # SEP/REP that interacts badly. Falling back to the
                    # vreg path (LDA slot / STA dst) is always sound —
                    # the slot holds the producer's full-width result.
                    pairs = list(zip(result_vregs, instr.values))
                    for i, (dst, src) in enumerate(pairs):
                        return_value = src
                        if (i == 0
                                and exit_setmode is None
                                and isinstance(return_value, VirtualRegister)
                                and _value_is_in_a_at_end_of(
                                    new_instructions, return_value)):
                            return_value = HardwareRegister('A')
                        new_instructions.append(Move(
                            dest=dst,
                            source=return_value,
                            type_info=callee.return_type,
                        ))
                    if exit_setmode is not None:
                        new_instructions.append(exit_setmode)
                    # Jump to merge block
                    new_instructions.append(Jump(target=merge_block_id))
                    return_blocks.append(cloned_block.block_id)
                else:
                    new_instructions.append(instr)
            cloned_block.instructions = new_instructions
            # Update successors if this was a return block
            if cloned_block.block_id in return_blocks:
                cloned_block.successors = [merge_block_id]

        # Set merge block predecessors
        merge_block.predecessors = return_blocks

        # Add cloned blocks to caller
        for new_id, new_block in cloned_blocks.items():
            caller.blocks[new_id] = new_block

        # Add merge block
        caller.blocks[merge_block_id] = merge_block

        # Update original successors' predecessors
        for succ_id in merge_block.successors:
            if succ_id in caller.blocks:
                succ_block = caller.blocks[succ_id]
                if block_id in succ_block.predecessors:
                    succ_block.predecessors.remove(block_id)
                if merge_block_id not in succ_block.predecessors:
                    succ_block.predecessors.append(merge_block_id)

        # Keep exit_block_ids consistent. The original call_block no longer
        # terminates the function (it now ends with Jump to the inlined
        # entry), so drop it. The merge block inherits the post-call
        # instructions; if those end in Return / ReturnFromInterrupt it is
        # now an exit. Downstream passes (frame teardown, mode analysis)
        # rely on this list — stale entries cause silent miscompiles.
        if block_id in caller.exit_block_ids:
            caller.exit_block_ids.remove(block_id)
        if (merge_block.instructions
                and isinstance(merge_block.instructions[-1],
                               (Return, ReturnFromInterrupt))
                and merge_block_id not in caller.exit_block_ids):
            caller.exit_block_ids.append(merge_block_id)

        return True

    def _compute_caller_mode_at_call(
        self,
        caller: MIRFunction,
        call_block: BasicBlock,
        pre_call_instrs: List[MIRInstruction],
    ):
        """Return the ProcessorMode the CPU is in when control reaches the
        Call instruction.

        Starts from `call_block.entry_mode` (set by MIRModeTracker during
        MIR construction) and folds any SetMode in pre_call_instrs forward.
        Falls back to a freshly-built mode from `caller.entry_m_mode` when
        the block's entry_mode is None — that happens after a previous
        inline mutated the caller and we don't re-run mode tracking.
        """
        from r65.compiler.typeck.processor_mode import (
            ProcessorMode, ModeState, XModeState,
        )

        mode = call_block.entry_mode
        if mode is None:
            mode = ProcessorMode(
                caller.entry_m_mode or ModeState.M8, XModeState.X16
            )
        for instr in pre_call_instrs:
            if isinstance(instr, SetMode):
                if instr.is_set:
                    mode = mode.apply_sep(instr.mask)
                else:
                    mode = mode.apply_rep(instr.mask)
        return mode

    def _make_boundary_setmode(self, from_mode, to_mode):
        """Return a SetMode that transitions M flag from `from_mode` to
        `to_mode`, or None if no transition is needed.

        Both arguments are ModeState (M8/M16) or None; None is treated as
        ModeState.M8 to match MIRFunction's default. X is always X16 in
        R65, so we only ever flip the M bit.
        """
        from r65.compiler.typeck.processor_mode import ModeState

        from_m = from_mode or ModeState.M8
        to_m = to_mode or ModeState.M8
        if from_m == to_m:
            return None
        if to_m == ModeState.M8:
            return SetMode(mask=0x20, is_set=True)   # SEP #$20
        return SetMode(mask=0x20, is_set=False)      # REP #$20

    def _create_param_bindings(
        self,
        call: Call,
        callee: MIRFunction,
        cloner: BlockCloner,
        saved_hw_regs: Optional[Set[str]] = None,
        used_vreg_ids: Optional[Set[int]] = None,
    ) -> Tuple[List[MIRInstruction], List[MIRInstruction]]:
        """
        Create instructions to bind call arguments to callee parameters.

        Returns two lists:
        - register_instrs: Move arg value into hardware register (must run first)
        - stack_instrs: Move arg value into callee vreg (must run after reg saves)

        Setup is skipped for parameters the inlined body never observes:
        - Register params whose HW register isn't saved at entry (i.e., the
          MIR builder elided the save Move because no use exists).
        - Stack params whose vreg has no reads in the cloned body.

        Args:
            call: The Call instruction
            callee: Function being inlined
            cloner: Block cloner for vreg remapping
            saved_hw_regs: Names of HW registers saved by the entry-block
                leading Moves. Treated as "all live" when None (no pruning).
            used_vreg_ids: IDs of vregs (in caller's space) that are read in
                the cloned body. When None, all stack-param Moves are emitted.

        Returns:
            Tuple of (register_instrs, stack_instrs)
        """
        register_instrs = []
        stack_instrs = []

        for i, (arg, param) in enumerate(zip(call.args, callee.parameters)):
            if isinstance(param.binding, RegisterBinding):
                # Register parameter: the call lowering would have set up the
                # hardware register. After inlining, we must emit this setup
                # explicitly since the Call instruction is removed — unless
                # the body never reads this register, in which case the load
                # is dead. Detected via the absence of a save Move at entry.
                hw_reg = HardwareRegister(param.binding.register_name)
                if saved_hw_regs is not None and hw_reg.name not in saved_hw_regs:
                    continue
                register_instrs.append(Move(
                    dest=hw_reg,
                    source=arg.value,
                    type_info=param.param_type
                ))

            elif isinstance(param.binding, VariableBinding):
                # Variable-bound: caller already stored to the variable
                pass

            else:
                # Stack parameter: map caller's arg to callee's param vreg
                # Get the callee's vreg for this parameter
                param_idx = i
                if param_idx in callee.param_to_vreg:
                    callee_vreg = callee.param_to_vreg[param_idx]
                    # Get the remapped vreg
                    inlined_vreg = cloner._remap_vreg(callee_vreg)
                    # Skip if the body never reads this vreg.
                    if (used_vreg_ids is not None
                            and inlined_vreg.id not in used_vreg_ids):
                        continue
                    # When the argument's type is narrower than the parameter's
                    # type (e.g. u8 → u16), we need a TypeConvert so the high
                    # bytes get zero/sign-extended properly. A plain Move does
                    # a bit-copy and leaves the high byte garbage from whatever
                    # was last in A — observed as bogus walkmap reads after
                    # implicit inlining of test_map(x: u8, y: u16) in
                    # classickong.r65. The non-inlined call path handles this
                    # via emit_outgoing_stack_argument's source_size < param_size
                    # branch; the inline path must mirror it.
                    from r65.compiler.codegen.type_utils import get_type_size
                    arg_type = getattr(arg.value, 'type_info', None)
                    param_size = get_type_size(param.param_type) if param.param_type else None
                    arg_size = get_type_size(arg_type) if arg_type else None
                    if (arg_type is not None and param_size is not None
                            and arg_size is not None
                            and arg_size < param_size):
                        widened = cloner.caller_func.vreg_allocator.alloc(
                            param.param_type,
                            f"widened_{getattr(arg.value, 'hint', 'arg')}"
                        )
                        stack_instrs.append(TypeConvert(
                            dest=widened,
                            source=arg.value,
                            source_type=arg_type,
                            target_type=param.param_type,
                        ))
                        source = widened
                    else:
                        source = arg.value
                    # Move argument value to inlined vreg
                    stack_instrs.append(Move(
                        dest=inlined_vreg,
                        source=source,
                        type_info=param.param_type
                    ))

        # Sort register loads so any A-targeted load comes LAST. Several
        # non-A param loads route through A in codegen — Move to B uses
        # XBA on A, Move to DBR uses PHA/PLB, Move to D uses TCD — so
        # if we set A first then load another register through A, the
        # A-param value is clobbered before the body uses it. Matches the
        # non-inlined emit_call_args convention (see MEMORY.md "Call Arg
        # A-Coalescence"). Stable sort: False (non-A) sorts before True (A).
        register_instrs.sort(
            key=lambda m: getattr(m.dest, 'name', '') == 'A'
        )

        return register_instrs, stack_instrs

    def _collect_read_hw_reg_names(self, blocks: Dict[int, BasicBlock]) -> Set[str]:
        """Return the set of HardwareRegister names read anywhere in `blocks`.

        Liveness's _GET_USES table only tracks X/Y as HardwareRegisters (A is
        too volatile to track that way), so we walk every operand position
        explicitly. False positives — counting a HW reg that's redefined
        before any read — are safe; they just keep the param load alive.
        """
        names: Set[str] = set()

        def add(op):
            if isinstance(op, HardwareRegister):
                names.add(op.name)

        for block in blocks.values():
            for instr in block.instructions:
                if isinstance(instr, Move):
                    add(instr.source)
                elif isinstance(instr, Store):
                    add(instr.source)
                elif isinstance(instr, BinaryOp):
                    add(instr.left); add(instr.right)
                elif isinstance(instr, UnaryOp):
                    add(instr.operand)
                elif isinstance(instr, Compare):
                    add(instr.left); add(instr.right)
                elif isinstance(instr, BitTest):
                    add(instr.value)
                elif isinstance(instr, (Rotate, TypeConvert, ToBool, BankByte)):
                    add(instr.source)
                elif isinstance(instr, StoreIndirect):
                    add(instr.source); add(instr.pointer)
                elif isinstance(instr, (LookupTable, JumpTable)):
                    add(instr.scrutinee)
                elif isinstance(instr, Call):
                    if isinstance(instr.function, HardwareRegister):
                        add(instr.function)
                    for arg in instr.args:
                        add(arg.value)
                elif isinstance(instr, TraitDispatch):
                    for arg in instr.args:
                        add(arg.value)
                    add(instr.self_ptr)
                elif isinstance(instr, Return):
                    for v in instr.values:
                        add(v)
                elif isinstance(instr, CondBranch):
                    add(instr.condition)
        return names

    def _collect_read_vreg_ids(self, blocks: Dict[int, BasicBlock]) -> Set[int]:
        """Return the set of VirtualRegister IDs that appear as a read operand
        anywhere in `blocks`.

        Pure write positions (Move.dest, Load.dest, BinaryOp.dest, …) are
        excluded — only consumed operands count as a read. Used to detect
        dead stack-param Moves: if the inlined body never reads the param's
        vreg, the inserted setup Move is a dead store.
        """
        from r65.compiler.mir.liveness import _GET_USES
        used: Set[int] = set()
        for block in blocks.values():
            for instr in block.instructions:
                handler = _GET_USES.get(type(instr))
                if handler is None:
                    continue
                for op in handler(instr):
                    if isinstance(op, VirtualRegister):
                        used.add(op.id)
        return used
