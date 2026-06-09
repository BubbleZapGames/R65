# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Liveness analysis for MIR virtual registers.

Determines which virtual registers are live (in use) at each program point,
enabling optimizations like stack slot reuse and better register allocation.
"""

from typing import Set, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from r65.compiler.mir.nodes import (
    MIRInstruction, VirtualRegister, HardwareRegister,
    BasicBlock, MIRFunction,
    Load, Store, Move, BinaryOp, UnaryOp, Compare, BitTest, Rotate,
    Call, Return, Jump, CondBranch, JumpTable, LookupTable, TypeConvert, ToBool,
    LoadIndirect, StoreIndirect, StatusFlagRead, TraitDispatch, BankByte,
)

# Types that count as register operands for liveness
_REG_TYPES = (VirtualRegister, HardwareRegister)


# ============================================================================
# _get_uses dispatch helpers
# ============================================================================

def _uses_load(instr):
    """Load: only VR source (not HR)."""
    s = instr.source
    return [s] if isinstance(s, VirtualRegister) else []

def _uses_source_vr_hr(instr):
    """Store, Move: source may be VR or HR."""
    s = instr.source
    return [s] if isinstance(s, _REG_TYPES) else []

def _uses_left_right(instr):
    """BinaryOp, Compare: left and right may be VR or HR."""
    uses = []
    if isinstance(instr.left, _REG_TYPES):
        uses.append(instr.left)
    if isinstance(instr.right, _REG_TYPES):
        uses.append(instr.right)
    return uses

def _uses_operand(instr):
    """UnaryOp: operand may be VR or HR."""
    o = instr.operand
    return [o] if isinstance(o, _REG_TYPES) else []

def _uses_value(instr):
    """BitTest: value may be VR or HR."""
    v = instr.value
    return [v] if isinstance(v, _REG_TYPES) else []

def _uses_rotate_source(instr):
    """Rotate, TypeConvert, ToBool: source may be VR or HR."""
    s = instr.source
    return [s] if isinstance(s, _REG_TYPES) else []

def _uses_load_indirect(instr):
    """LoadIndirect: pointer (VR only)."""
    p = instr.pointer
    return [p] if isinstance(p, VirtualRegister) else []

def _uses_store_indirect(instr):
    """StoreIndirect: pointer (VR) + source (VR or HR)."""
    uses = []
    if isinstance(instr.pointer, VirtualRegister):
        uses.append(instr.pointer)
    if isinstance(instr.source, _REG_TYPES):
        uses.append(instr.source)
    return uses

def _uses_scrutinee(instr):
    """LookupTable, JumpTable: scrutinee may be VR or HR."""
    s = instr.scrutinee
    return [s] if isinstance(s, _REG_TYPES) else []

def _uses_call(instr):
    """Call: argument values (VR or HR), plus function pointer if indirect."""
    uses = []
    if isinstance(instr.function, _REG_TYPES):
        uses.append(instr.function)
    for arg in instr.args:
        if isinstance(arg.value, _REG_TYPES):
            uses.append(arg.value)
    return uses

def _uses_trait_dispatch(instr):
    """TraitDispatch: argument values + self_ptr."""
    uses = []
    for arg in instr.args:
        if isinstance(arg.value, _REG_TYPES):
            uses.append(arg.value)
    if isinstance(instr.self_ptr, VirtualRegister):
        uses.append(instr.self_ptr)
    return uses

def _uses_return(instr):
    """Return: return values (VR or HR)."""
    uses = []
    for val in instr.values:
        if isinstance(val, _REG_TYPES):
            uses.append(val)
    return uses

def _uses_condbranch(instr):
    """CondBranch: condition may be a VirtualRegister."""
    c = instr.condition
    return [c] if isinstance(c, _REG_TYPES) else []

def _uses_none(instr):
    return []


# Dispatch table: instruction type → uses extractor
_GET_USES = {
    Load: _uses_load,
    Store: _uses_source_vr_hr,
    Move: _uses_source_vr_hr,
    BinaryOp: _uses_left_right,
    UnaryOp: _uses_operand,
    Compare: _uses_left_right,
    BitTest: _uses_value,
    Rotate: _uses_rotate_source,
    TypeConvert: _uses_rotate_source,
    ToBool: _uses_rotate_source,
    BankByte: _uses_rotate_source,
    LoadIndirect: _uses_load_indirect,
    StoreIndirect: _uses_store_indirect,
    LookupTable: _uses_scrutinee,
    JumpTable: _uses_scrutinee,
    Call: _uses_call,
    TraitDispatch: _uses_trait_dispatch,
    Return: _uses_return,
    StatusFlagRead: _uses_none,
    CondBranch: _uses_condbranch,
    Jump: _uses_none,
}


# ============================================================================
# _get_defs dispatch helpers
# ============================================================================

def _defs_dest_vr_hr(instr):
    """Load, Move, BinaryOp, UnaryOp, Rotate, TypeConvert, ToBool: dest may be VR or HR."""
    d = instr.dest
    return [d] if isinstance(d, _REG_TYPES) else []

def _defs_load_indirect(instr):
    """LoadIndirect: dest (VR only)."""
    d = instr.dest
    return [d] if isinstance(d, VirtualRegister) else []

def _defs_lookup_table(instr):
    """LookupTable: dest (VR only)."""
    d = instr.dest
    return [d] if isinstance(d, VirtualRegister) else []

def _defs_call(instr):
    """Call, TraitDispatch: return registers (VR only)."""
    defs = []
    for ret in instr.returns:
        if isinstance(ret, VirtualRegister):
            defs.append(ret)
    return defs

def _defs_status_flag_read(instr):
    """StatusFlagRead: dest (VR or HR)."""
    d = instr.dest
    return [d] if isinstance(d, _REG_TYPES) else []

def _defs_none(instr):
    return []


# Dispatch table: instruction type → defs extractor
_GET_DEFS = {
    Load: _defs_dest_vr_hr,
    Move: _defs_dest_vr_hr,
    BinaryOp: _defs_dest_vr_hr,
    UnaryOp: _defs_dest_vr_hr,
    Rotate: _defs_dest_vr_hr,
    TypeConvert: _defs_dest_vr_hr,
    ToBool: _defs_dest_vr_hr,
    BankByte: _defs_dest_vr_hr,
    LoadIndirect: _defs_load_indirect,
    LookupTable: _defs_lookup_table,
    Call: _defs_call,
    TraitDispatch: _defs_call,
    StatusFlagRead: _defs_status_flag_read,
    Store: _defs_none,
    StoreIndirect: _defs_none,
    Compare: _defs_none,
    BitTest: _defs_none,
    Return: _defs_none,
    Jump: _defs_none,
    CondBranch: _defs_none,
    JumpTable: _defs_none,
}


@dataclass
class LivenessInfo:
    """
    Liveness information for a basic block.

    Tracks which variables are live at entry and exit of the block.
    Also tracks hardware register (X, Y) liveness for direct usage.
    Note: A register is NOT tracked here - it's only tracked via vreg bindings
    because A is constantly used for intermediate calculations.
    """
    live_in: Set[VirtualRegister] = field(default_factory=set)
    live_out: Set[VirtualRegister] = field(default_factory=set)

    # Variables used (read) in this block
    use: Set[VirtualRegister] = field(default_factory=set)

    # Variables defined (written) in this block
    define: Set[VirtualRegister] = field(default_factory=set)

    # Hardware register tracking (X and Y only, not A)
    hw_use: Set[str] = field(default_factory=set)  # {'X', 'Y'}
    hw_define: Set[str] = field(default_factory=set)
    hw_live_in: Set[str] = field(default_factory=set)
    hw_live_out: Set[str] = field(default_factory=set)


class LivenessAnalyzer:
    """
    Performs liveness analysis on MIR functions.

    Uses iterative dataflow analysis to compute live ranges for each
    virtual register, enabling stack slot reuse optimization.
    """

    def __init__(self, mir_func: MIRFunction):
        """
        Initialize liveness analyzer.

        Args:
            mir_func: MIR function to analyze
        """
        self.func = mir_func
        self.liveness: Dict[int, LivenessInfo] = {}
        self._live_ranges_cache: Optional[Dict[VirtualRegister, Set[int]]] = None

    def analyze(self) -> Dict[int, LivenessInfo]:
        """
        Perform liveness analysis on the function.

        Analyzes both VirtualRegister and HardwareRegister (X and Y) liveness.

        Returns:
            Dictionary mapping block IDs to liveness information
        """
        # Initialize liveness info for each block
        for block_id, block in self.func.blocks.items():
            self.liveness[block_id] = LivenessInfo()
            self._compute_use_def(block)

        # Iterative dataflow analysis (backward)
        changed = True
        iterations = 0
        max_iterations = 100  # Safety limit

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Process blocks in reverse order (backward analysis)
            for block_id in reversed(list(self.func.blocks.keys())):
                block = self.func.blocks[block_id]
                info = self.liveness[block_id]

                # Save old live_in for convergence check
                old_live_in = info.live_in.copy()
                old_hw_live_in = info.hw_live_in.copy()

                # live_out = union of successors' live_in
                info.live_out = set()
                info.hw_live_out = set()
                for succ_id in self._get_successors(block):
                    if succ_id in self.liveness:
                        succ_info = self.liveness[succ_id]
                        info.live_out.update(succ_info.live_in)
                        info.hw_live_out.update(succ_info.hw_live_in)

                # live_in = use ∪ (live_out - define)
                info.live_in = info.use | (info.live_out - info.define)
                info.hw_live_in = info.hw_use | (info.hw_live_out - info.hw_define)

                # Check for convergence
                if info.live_in != old_live_in or info.hw_live_in != old_hw_live_in:
                    changed = True

        if iterations >= max_iterations:
            print(f"Warning: Liveness analysis did not converge after {max_iterations} iterations")

        return self.liveness

    def _compute_use_def(self, block: BasicBlock):
        """
        Compute use and define sets for a basic block.

        Handles both VirtualRegister and HardwareRegister (X and Y only).

        Args:
            block: Basic block to analyze
        """
        info = self.liveness[block.block_id]

        for instr in block.instructions:
            # Get variables used by this instruction
            uses = self._get_uses(instr)
            for var in uses:
                if isinstance(var, VirtualRegister):
                    # If not yet defined in this block, it's used
                    if var not in info.define:
                        info.use.add(var)
                elif isinstance(var, HardwareRegister) and var.name in ('X', 'Y'):
                    # Track X/Y hardware register usage (not A)
                    if var.name not in info.hw_define:
                        info.hw_use.add(var.name)

            # Get variables defined by this instruction
            defs = self._get_defs(instr)
            for var in defs:
                if isinstance(var, VirtualRegister):
                    info.define.add(var)
                elif isinstance(var, HardwareRegister) and var.name in ('X', 'Y'):
                    # Track X/Y hardware register definitions (not A)
                    info.hw_define.add(var.name)

    def _get_uses(self, instr: MIRInstruction) -> List:
        """
        Get variables used (read) by an instruction.

        Uses dispatch dict for O(1) type lookup.

        Returns both VirtualRegister and HardwareRegister instances.
        Note: HardwareRegister tracking is only for X and Y; A is not tracked
        for direct usage because it's constantly used for intermediate calculations.
        """
        handler = _GET_USES.get(type(instr))
        if handler:
            return handler(instr)
        return []

    def _get_defs(self, instr: MIRInstruction) -> List:
        """
        Get variables defined (written) by an instruction.

        Uses dispatch dict for O(1) type lookup.

        Returns both VirtualRegister and HardwareRegister instances.
        Note: HardwareRegister tracking is only for X and Y; A is not tracked
        for direct usage because it's constantly used for intermediate calculations.
        """
        handler = _GET_DEFS.get(type(instr))
        if handler:
            return handler(instr)
        return []

    def _get_successors(self, block: BasicBlock) -> List[int]:
        """
        Get successor block IDs for a basic block.

        Args:
            block: Basic block

        Returns:
            List of successor block IDs
        """
        successors = []

        # Look at last instruction in block
        if block.instructions:
            last = block.instructions[-1]

            if isinstance(last, Jump):
                successors.append(last.target)

            elif isinstance(last, CondBranch):
                successors.append(last.true_target)
                successors.append(last.false_target)

            elif isinstance(last, JumpTable):
                successors.extend(last.targets)
                successors.append(last.default_target)

            elif isinstance(last, LookupTable):
                successors.append(last.merge_target)

            elif isinstance(last, Return):
                # No successors for return
                pass

        return successors

    def get_live_ranges(self) -> Dict[VirtualRegister, Set[int]]:
        """
        Compute live ranges for all virtual registers.

        Returns a dictionary mapping each virtual register to the set of
        basic block IDs where it is live. Results are cached for performance.

        Returns:
            Dictionary mapping VirtualRegister to set of live block IDs
        """
        # Return cached result if available
        if self._live_ranges_cache is not None:
            return self._live_ranges_cache

        live_ranges: Dict[VirtualRegister, Set[int]] = {}

        for block_id, info in self.liveness.items():
            # A variable is live in a block if it's in live_in or live_out
            live_vars = info.live_in | info.live_out

            for var in live_vars:
                if var not in live_ranges:
                    live_ranges[var] = set()
                live_ranges[var].add(block_id)

        # Cache the result
        self._live_ranges_cache = live_ranges
        return live_ranges

    def interferes(self, var1: VirtualRegister, var2: VirtualRegister) -> bool:
        """
        Check if two variables interfere (have overlapping live ranges).

        Variables interfere if they are both live at the same program point,
        meaning they cannot share the same memory location.

        Args:
            var1: First virtual register
            var2: Second virtual register

        Returns:
            True if variables interfere, False otherwise
        """
        ranges = self.get_live_ranges()

        # Fast non-interference check: if the block-level live ranges are
        # completely disjoint, the variables cannot interfere.
        var1_blocks = ranges.get(var1, set())
        var2_blocks = ranges.get(var2, set())
        shared_blocks = var1_blocks & var2_blocks
        # Also include blocks where either variable is defined or used
        # (a var may not be in live_in/live_out but still appear in the block)
        candidate_blocks = set(shared_blocks)
        for block_id, info in self.liveness.items():
            var1_relevant = var1 in info.use or var1 in info.define or var1 in info.live_in
            var2_relevant = var2 in info.use or var2 in info.define or var2 in info.live_in
            if var1_relevant and var2_relevant:
                candidate_blocks.add(block_id)

        if not candidate_blocks:
            return False

        # Precise per-instruction interference check within candidate blocks
        for block_id in candidate_blocks:
            info = self.liveness[block_id]
            block = self.func.blocks[block_id]

            # Determine if each variable is live on entry to this block
            var1_live_on_entry = var1 in info.use or var1 in info.live_in
            var2_live_on_entry = var2 in info.use or var2 in info.live_in

            # First pass: find last use of each variable
            var1_last_use = -1
            var2_last_use = -1
            for i, instr in enumerate(block.instructions):
                uses = self._get_uses(instr)
                if var1 in uses:
                    var1_last_use = i
                if var2 in uses:
                    var2_last_use = i

            # Also consider live_out - if a var is live out, its "last use" is the end
            if var1 in info.live_out:
                var1_last_use = len(block.instructions)
            if var2 in info.live_out:
                var2_last_use = len(block.instructions)

            # Second pass: check for interference at each instruction.
            #
            # A variable is live at instruction i if it was previously
            # defined (or live-in) and still has future uses (i <= last_use).
            #
            # Crucially, a pure definition (not also a use) at instruction i
            # makes the variable live starting AFTER i.  At i itself the old
            # value is dead and the new value hasn't been observed yet, so a
            # Move(dest, src) does NOT cause dest and src to interfere at
            # that instruction — this is the standard copy-coalescing
            # property.  We track this with *_just_defined flags.

            var1_live = var1_live_on_entry
            var2_live = var2_live_on_entry

            for i, instr in enumerate(block.instructions):
                defs = self._get_defs(instr)
                uses = self._get_uses(instr)

                var1_just_defined = False
                var2_just_defined = False

                # A variable becomes live when defined
                if var1 in defs:
                    if not var1_live:
                        var1_just_defined = True
                    var1_live = True
                if var2 in defs:
                    if not var2_live:
                        var2_just_defined = True
                    var2_live = True

                # A variable that was JUST defined at a Move instruction (and
                # not also used) is not yet observable — the Move's dest doesn't
                # interfere with the Move's source.  This is the standard
                # copy-coalescing property.
                #
                # For non-Move defs (BinaryOp, Load, etc.), the dest's physical
                # storage IS written during the instruction.  Even if the dest
                # vreg has no future uses (e.g., it was eliminated by copy
                # propagation), the write still occurs and would corrupt any
                # other variable sharing the same storage.  So we must NOT
                # suppress interference for non-Move defining instructions.
                from r65.compiler.mir.nodes import Move as MoveInstr
                is_move = isinstance(instr, MoveInstr)

                var1_live_here = var1_live and i <= var1_last_use
                if var1_just_defined and var1 not in uses and is_move:
                    var1_live_here = False
                # A dead def (no future uses) at a non-Move instruction still
                # physically writes to the storage location, so it must
                # interfere with any other live variable at this instruction.
                if var1 in defs and not is_move:
                    var1_live_here = True

                var2_live_here = var2_live and i <= var2_last_use
                if var2_just_defined and var2 not in uses and is_move:
                    var2_live_here = False
                if var2 in defs and not is_move:
                    var2_live_here = True

                if var1_live_here and var2_live_here:
                    return True

        return False


class InstructionLivenessAnalyzer:
    """
    Per-instruction liveness analysis.

    Provides fine-grained liveness information at the instruction level,
    enabling precise tracking of register liveness across calls.
    """

    def __init__(self, mir_func: MIRFunction):
        """
        Initialize instruction liveness analyzer.

        Args:
            mir_func: MIR function to analyze
        """
        self.func = mir_func
        self.block_analyzer = LivenessAnalyzer(mir_func)
        self.liveness = self.block_analyzer.analyze()

        # Cache for instruction-level liveness
        # Key: (block_id, instr_idx), Value: set of live vregs after instruction
        self._live_after_cache: Dict[Tuple[int, int], Set[VirtualRegister]] = {}

        # Cache for hardware register instruction-level liveness
        # Key: (block_id, instr_idx), Value: set of live hw reg names ('X', 'Y')
        self._hw_live_after_cache: Dict[Tuple[int, int], Set[str]] = {}

        # Cache for calls each vreg is live across
        self._live_across_calls_cache: Dict[int, List[Call]] = {}

        # Build instruction position index
        self._instruction_positions: Dict[id, Tuple[int, int]] = {}
        self._build_instruction_index()

    def _build_instruction_index(self):
        """Build index mapping instructions to their positions."""
        for block_id, block in self.func.blocks.items():
            for instr_idx, instr in enumerate(block.instructions):
                self._instruction_positions[id(instr)] = (block_id, instr_idx)

    def get_instruction_position(self, instr: MIRInstruction) -> Optional[Tuple[int, int]]:
        """
        Get the (block_id, instr_idx) position of an instruction.

        Args:
            instr: Instruction to find

        Returns:
            (block_id, instr_idx) tuple or None if not found
        """
        return self._instruction_positions.get(id(instr))

    def is_live_after(self, vreg: VirtualRegister,
                      block_id: int, instr_idx: int) -> bool:
        """
        Check if a virtual register is live after a given instruction.

        Args:
            vreg: Virtual register to check
            block_id: Block ID containing the instruction
            instr_idx: Index of instruction within the block

        Returns:
            True if vreg is live after the instruction
        """
        live_after = self._get_live_after(block_id, instr_idx)
        return vreg in live_after

    def _get_live_after(self, block_id: int, instr_idx: int) -> Set[VirtualRegister]:
        """
        Get the set of virtual registers live after an instruction.

        Uses backward scan within the block.
        """
        cache_key = (block_id, instr_idx)
        if cache_key in self._live_after_cache:
            return self._live_after_cache[cache_key]

        block = self.func.blocks.get(block_id)
        if not block:
            return set()

        info = self.liveness.get(block_id)
        if not info:
            return set()

        # Start with live_out and work backward
        live = info.live_out.copy()

        # Process instructions in reverse from end to instr_idx
        for i in range(len(block.instructions) - 1, instr_idx, -1):
            instr = block.instructions[i]
            # Remove definitions (they become live before, not after)
            for d in self.block_analyzer._get_defs(instr):
                if isinstance(d, VirtualRegister):
                    live.discard(d)
            # Add uses (they must be live before this instruction)
            for u in self.block_analyzer._get_uses(instr):
                if isinstance(u, VirtualRegister):
                    live.add(u)

        # Cache and return
        self._live_after_cache[cache_key] = live
        return live

    def _get_hw_live_after(self, block_id: int, instr_idx: int) -> Set[str]:
        """
        Get the set of hardware registers (X, Y) live after an instruction.

        Uses backward scan within the block.

        Args:
            block_id: Block ID
            instr_idx: Instruction index within the block

        Returns:
            Set of hardware register names ('X', 'Y') that are live after
        """
        cache_key = (block_id, instr_idx)
        if cache_key in self._hw_live_after_cache:
            return self._hw_live_after_cache[cache_key]

        block = self.func.blocks.get(block_id)
        if not block:
            return set()

        info = self.liveness.get(block_id)
        if not info:
            return set()

        # Start with hw_live_out and work backward
        live = info.hw_live_out.copy()

        # Process instructions in reverse from end to instr_idx
        for i in range(len(block.instructions) - 1, instr_idx, -1):
            instr = block.instructions[i]
            # Remove definitions (they become live before, not after)
            for d in self.block_analyzer._get_defs(instr):
                if isinstance(d, HardwareRegister) and d.name in ('X', 'Y'):
                    live.discard(d.name)
            # Add uses (they must be live before this instruction)
            for u in self.block_analyzer._get_uses(instr):
                if isinstance(u, HardwareRegister) and u.name in ('X', 'Y'):
                    live.add(u.name)

        # Cache and return
        self._hw_live_after_cache[cache_key] = live
        return live

    def is_hw_reg_live_after(self, hw_reg: str, block_id: int, instr_idx: int) -> bool:
        """
        Check if a hardware register (X/Y) is live after an instruction.

        This tracks DIRECT hardware register usage (e.g., X = 0; clobbers_x(); X + 1).
        Note: A register is NOT tracked here - it's only tracked via vreg bindings
        because A is constantly used for intermediate calculations.

        Args:
            hw_reg: Hardware register name ('X' or 'Y')
            block_id: Block ID containing the instruction
            instr_idx: Index of instruction within the block

        Returns:
            True if hw_reg is live after the instruction
        """
        if hw_reg not in ('X', 'Y'):
            return False

        hw_live_after = self._get_hw_live_after(block_id, instr_idx)
        return hw_reg in hw_live_after

    def is_live_across_any_call(self, vreg: VirtualRegister) -> bool:
        """
        Check if a virtual register is live across any Call instruction.

        A vreg is "live across a call" if it is:
        1. Live before the call (used after the call)
        2. Defined before the call

        Args:
            vreg: Virtual register to check

        Returns:
            True if vreg is live across at least one call
        """
        calls = self.get_calls_vreg_is_live_across(vreg)
        return len(calls) > 0

    def get_calls_vreg_is_live_across(self, vreg: VirtualRegister) -> List[Call]:
        """
        Get all Call instructions where a vreg is live across.

        Args:
            vreg: Virtual register to check

        Returns:
            List of Call instructions the vreg is live across
        """
        if vreg.id in self._live_across_calls_cache:
            return self._live_across_calls_cache[vreg.id]

        calls = []
        vreg_ranges = self.block_analyzer.get_live_ranges()

        # Get blocks where vreg is live (in live_in/live_out)
        vreg_blocks = vreg_ranges.get(vreg, set())

        # Also check blocks where vreg is defined (it may be live within the
        # block even if not in live_in/live_out)
        for block_id, block in self.func.blocks.items():
            for instr in block.instructions:
                defs = self.block_analyzer._get_defs(instr)
                if vreg in defs:
                    vreg_blocks = vreg_blocks | {block_id}
                    break

        for block_id in vreg_blocks:
            block = self.func.blocks.get(block_id)
            if not block:
                continue

            # Track if vreg is defined yet (only live across call if defined before)
            info = self.liveness.get(block_id)
            vreg_defined = vreg in info.live_in if info else False

            for instr_idx, instr in enumerate(block.instructions):
                # Check if this instruction defines vreg
                defs = self.block_analyzer._get_defs(instr)
                if vreg in defs:
                    vreg_defined = True

                # Check if this is a Call/TraitDispatch and vreg is live after it
                if isinstance(instr, (Call, TraitDispatch)) and vreg_defined:
                    # Check if vreg is live after this call
                    if self.is_live_after(vreg, block_id, instr_idx):
                        calls.append(instr)

        self._live_across_calls_cache[vreg.id] = calls
        return calls


@dataclass
class ClobberRegion:
    """
    A region where a hardware register needs to be spilled once.

    A clobber region is a maximal sequence of instructions where:
    1. The register is live at the start (defined before, used after)
    2. Contains one or more clobbering calls (calls that don't preserve the register)
    3. The register is not used until after the region ends

    This allows saving once before the first clobber and restoring once
    before the first use, rather than spilling around each individual call.
    """
    hw_reg: str                    # 'X' or 'Y'
    save_before_idx: int           # Instruction index to insert save before
    restore_before_idx: int        # Instruction index to insert restore before
    clobbering_calls: List[int] = field(default_factory=list)  # Indices of calls in this region


class ClobberRegionAnalyzer:
    """
    Analyzes clobber regions for hardware registers within basic blocks.

    A clobber region groups consecutive clobbering calls (without intervening
    uses of the register) so that we can save once before the first clobber
    and restore once before the first use.

    This is Phase 2 optimization - V1 keeps regions within single basic blocks.
    """

    def __init__(self, instr_liveness: InstructionLivenessAnalyzer):
        """
        Initialize clobber region analyzer.

        Args:
            instr_liveness: Instruction-level liveness analyzer
        """
        self.instr_liveness = instr_liveness
        self.func = instr_liveness.func
        self.block_analyzer = instr_liveness.block_analyzer
        self.liveness = instr_liveness.liveness

    def analyze_block(self, block_id: int, preserves_map: Dict[str, Set[str]] = None,
                     include_a: bool = False) -> Dict[str, List[ClobberRegion]]:
        """
        Analyze clobber regions for a single basic block.

        Args:
            block_id: Block ID to analyze
            preserves_map: Map from function name to set of preserved registers.
                          If None, assumes all calls clobber all registers.
            include_a: If True, also analyze A register regions

        Returns:
            Dictionary mapping register name ('A', 'X', 'Y') to list of ClobberRegions
        """
        if preserves_map is None:
            preserves_map = {}

        block = self.func.blocks.get(block_id)
        if not block:
            return {'A': [], 'X': [], 'Y': []} if include_a else {'X': [], 'Y': []}

        regions: Dict[str, List[ClobberRegion]] = {'X': [], 'Y': []}
        if include_a:
            regions['A'] = []

        regs_to_analyze = ('A', 'X', 'Y') if include_a else ('X', 'Y')
        for hw_reg in regs_to_analyze:
            regions[hw_reg] = self._analyze_register_regions(
                block, block_id, hw_reg, preserves_map
            )

        return regions

    def _analyze_register_regions(self, block: BasicBlock, block_id: int,
                                   hw_reg: str, preserves_map: Dict[str, Set[str]]
                                  ) -> List[ClobberRegion]:
        """
        Analyze clobber regions for a single register in a block.

        Algorithm:
        1. Find all clobbering calls where the register is live across
        2. Group consecutive clobbers (without intervening uses) into regions
        3. For each region, record save point (before first clobber) and
           restore point (before first use after last clobber)

        Args:
            block: Basic block to analyze
            block_id: Block ID
            hw_reg: Hardware register name ('X' or 'Y')
            preserves_map: Map from function name to preserved registers

        Returns:
            List of ClobberRegions for this register
        """
        regions: List[ClobberRegion] = []

        # Track state for building regions
        current_region: Optional[ClobberRegion] = None

        # Scan instructions to find clobbering calls and uses
        for instr_idx, instr in enumerate(block.instructions):
            instr_type = type(instr)

            # Check if this instruction uses the hw register
            uses = self.block_analyzer._get_uses(instr)
            is_use = any(
                isinstance(u, HardwareRegister) and u.name == hw_reg
                for u in uses
            )

            # Check if this instruction defines the hw register
            defs = self.block_analyzer._get_defs(instr)
            is_def = any(
                isinstance(d, HardwareRegister) and d.name == hw_reg
                for d in defs
            )

            # For Call/TraitDispatch instructions, also check if the call returns a value
            # in hw_reg. A call that defines a register via its return value
            # is NOT a clobber — it's a definition. Without this check, the
            # region analyzer would spill/restore the register around the call,
            # clobbering the return value.
            is_call = instr_type is Call or instr_type is TraitDispatch
            if is_call and not is_def:
                if self._call_returns_in_hw_reg(instr, hw_reg):
                    is_def = True

            # If we have a current region and hit a use, close the region
            if current_region is not None and is_use:
                # Restore point is before this use
                current_region.restore_before_idx = instr_idx
                regions.append(current_region)
                current_region = None

            # Check if this is a clobbering call (skip if call defines hw_reg)
            if is_call and not is_def:
                # Get callee's preserved registers
                preserved: Set[str] = set()
                if instr_type is Call and isinstance(instr.function, str):
                    preserved = preserves_map.get(instr.function, set())
                # Also check the preserves attribute on the call instruction
                if hasattr(instr, 'preserves_attr') and instr.preserves_attr:
                    preserved = preserved | set(instr.preserves_attr.registers)

                # Check if this call clobbers the register
                if hw_reg not in preserved:
                    # Check if register is live after this call
                    if self.instr_liveness.is_hw_reg_live_after(hw_reg, block_id, instr_idx):
                        # This is a clobbering call where register is live across

                        if current_region is None:
                            # Start a new region - save before this call
                            current_region = ClobberRegion(
                                hw_reg=hw_reg,
                                save_before_idx=instr_idx,
                                restore_before_idx=-1,  # Will be set when region closes
                                clobbering_calls=[instr_idx]
                            )
                        else:
                            # Extend current region
                            current_region.clobbering_calls.append(instr_idx)

            # If register is redefined, close any open region without restore
            # (the old value is dead, new value starts fresh)
            if is_def and current_region is not None:
                # This is unusual - register redefined while live across call
                # Close the region before this definition
                current_region.restore_before_idx = instr_idx
                regions.append(current_region)
                current_region = None

        # Handle region that extends to end of block
        if current_region is not None:
            # Register is live out of block - restore at end of block
            current_region.restore_before_idx = len(block.instructions)
            regions.append(current_region)

        return regions

    @staticmethod
    def _call_returns_in_hw_reg(instr: Call, hw_reg: str) -> bool:
        """
        Check if a Call instruction returns a value in a specific hardware register.

        Uses the callee's return type and mode to determine the return register
        ordering, then checks if hw_reg is among the registers set by the callee.

        Note: The Call MIR node's `returns` list may be empty even for multi-return
        functions (return values are captured by separate Move instructions after
        the call). So we determine the return count from callee_return_type instead.

        Args:
            instr: Call instruction
            hw_reg: Hardware register name ('A', 'X', 'Y')

        Returns:
            True if the call returns a value in hw_reg
        """
        from r65.compiler.codegen.constants import get_return_registers

        callee_return_type = getattr(instr, 'callee_return_type', None)
        if callee_return_type is None:
            return False

        callee_entry_mode = getattr(instr, 'callee_entry_m_mode', None)
        return_regs = get_return_registers(callee_return_type, callee_entry_mode)

        # Determine how many registers the callee actually sets from its return type
        from r65.compiler.hir.types import MultiReturnTypeInfo
        if isinstance(callee_return_type, MultiReturnTypeInfo):
            num_returns = len(callee_return_type.element_types)
        else:
            # Non-tuple: single return value in A
            num_returns = 1

        return hw_reg in return_regs[:num_returns]

    def analyze_function(self, preserves_map: Dict[str, Set[str]] = None,
                        include_a: bool = False) -> Dict[int, Dict[str, List[ClobberRegion]]]:
        """
        Analyze clobber regions for all blocks in a function.

        Args:
            preserves_map: Map from function name to set of preserved registers
            include_a: If True, also analyze A register regions

        Returns:
            Dictionary mapping block_id to register regions
        """
        if preserves_map is None:
            preserves_map = {}

        result: Dict[int, Dict[str, List[ClobberRegion]]] = {}

        for block_id in self.func.blocks:
            result[block_id] = self.analyze_block(block_id, preserves_map, include_a)

        return result
