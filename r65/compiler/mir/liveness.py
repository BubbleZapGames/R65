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
    Call, Return, Jump, CondBranch, TypeConvert, ToBool,
    LoadIndirect, StoreIndirect, StatusFlagRead
)


@dataclass
class LivenessInfo:
    """
    Liveness information for a basic block.

    Tracks which variables are live at entry and exit of the block.
    """
    live_in: Set[VirtualRegister] = field(default_factory=set)
    live_out: Set[VirtualRegister] = field(default_factory=set)

    # Variables used (read) in this block
    use: Set[VirtualRegister] = field(default_factory=set)

    # Variables defined (written) in this block
    define: Set[VirtualRegister] = field(default_factory=set)


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

                # live_out = union of successors' live_in
                info.live_out = set()
                for succ_id in self._get_successors(block):
                    if succ_id in self.liveness:
                        info.live_out.update(self.liveness[succ_id].live_in)

                # live_in = use ∪ (live_out - define)
                info.live_in = info.use | (info.live_out - info.define)

                # Check for convergence
                if info.live_in != old_live_in:
                    changed = True

        if iterations >= max_iterations:
            print(f"Warning: Liveness analysis did not converge after {max_iterations} iterations")

        return self.liveness

    def _compute_use_def(self, block: BasicBlock):
        """
        Compute use and define sets for a basic block.

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

            # Get variables defined by this instruction
            defs = self._get_defs(instr)
            for var in defs:
                if isinstance(var, VirtualRegister):
                    info.define.add(var)

    def _get_uses(self, instr: MIRInstruction) -> List:
        """
        Get variables used (read) by an instruction.

        Args:
            instr: Instruction to analyze

        Returns:
            List of used variables
        """
        uses = []

        if isinstance(instr, Load):
            # Load uses the source address
            if hasattr(instr, 'source') and isinstance(instr.source, VirtualRegister):
                uses.append(instr.source)

        elif isinstance(instr, Store):
            # Store uses both source value and destination address
            if isinstance(instr.source, VirtualRegister):
                uses.append(instr.source)

        elif isinstance(instr, Move):
            if isinstance(instr.source, VirtualRegister):
                uses.append(instr.source)

        elif isinstance(instr, BinaryOp):
            if isinstance(instr.left, VirtualRegister):
                uses.append(instr.left)
            if isinstance(instr.right, VirtualRegister):
                uses.append(instr.right)

        elif isinstance(instr, UnaryOp):
            if isinstance(instr.operand, VirtualRegister):
                uses.append(instr.operand)

        elif isinstance(instr, Compare):
            if isinstance(instr.left, VirtualRegister):
                uses.append(instr.left)
            if isinstance(instr.right, VirtualRegister):
                uses.append(instr.right)

        elif isinstance(instr, BitTest):
            if isinstance(instr.value, VirtualRegister):
                uses.append(instr.value)

        elif isinstance(instr, Rotate):
            if isinstance(instr.source, VirtualRegister):
                uses.append(instr.source)

        elif isinstance(instr, TypeConvert):
            if isinstance(instr.source, VirtualRegister):
                uses.append(instr.source)

        elif isinstance(instr, ToBool):
            if isinstance(instr.source, VirtualRegister):
                uses.append(instr.source)

        elif isinstance(instr, LoadIndirect):
            # LoadIndirect uses the pointer
            if isinstance(instr.pointer, VirtualRegister):
                uses.append(instr.pointer)

        elif isinstance(instr, StoreIndirect):
            # StoreIndirect uses both the pointer and the source value
            if isinstance(instr.pointer, VirtualRegister):
                uses.append(instr.pointer)
            if isinstance(instr.source, VirtualRegister):
                uses.append(instr.source)

        elif isinstance(instr, Call):
            # Call uses all argument registers
            for arg in instr.args:
                if isinstance(arg.value, VirtualRegister):
                    uses.append(arg.value)

        elif isinstance(instr, Return):
            # Return uses all return value registers
            for val in instr.values:
                if isinstance(val, VirtualRegister):
                    uses.append(val)

        return uses

    def _get_defs(self, instr: MIRInstruction) -> List:
        """
        Get variables defined (written) by an instruction.

        Args:
            instr: Instruction to analyze

        Returns:
            List of defined variables
        """
        defs = []

        if isinstance(instr, Load):
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, Move):
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, BinaryOp):
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, UnaryOp):
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, Rotate):
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, TypeConvert):
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, ToBool):
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, LoadIndirect):
            # LoadIndirect defines the destination register
            if isinstance(instr.dest, VirtualRegister):
                defs.append(instr.dest)

        elif isinstance(instr, Call):
            # Call defines all return value registers
            for ret in instr.returns:
                if isinstance(ret, VirtualRegister):
                    defs.append(ret)

        return defs

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

        # Check block-level interference
        if var1 in ranges and var2 in ranges:
            if ranges[var1] & ranges[var2]:
                return True

        # Check intra-block interference using precise per-instruction liveness
        for block_id, info in self.liveness.items():
            block = self.func.blocks[block_id]

            # Determine if each variable is relevant to this block
            var1_live_on_entry = var1 in info.use or var1 in info.live_in
            var2_live_on_entry = var2 in info.use or var2 in info.live_in
            var1_defined_in_block = var1 in info.define
            var2_defined_in_block = var2 in info.define

            # Skip if neither variable is relevant to this block
            if not (var1_live_on_entry or var1_defined_in_block):
                continue
            if not (var2_live_on_entry or var2_defined_in_block):
                continue

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

            # Second pass: check for interference at each instruction
            var1_live = var1_live_on_entry
            var2_live = var2_live_on_entry

            for i, instr in enumerate(block.instructions):
                defs = self._get_defs(instr)
                uses = self._get_uses(instr)

                # A variable becomes live when defined
                if var1 in defs:
                    var1_live = True
                if var2 in defs:
                    var2_live = True

                # Check if both variables are live at this instruction
                # A variable is live at instruction i if:
                # - It was live on entry and i <= last_use, OR
                # - It was defined at some j <= i and i <= last_use
                var1_live_here = var1_live and i <= var1_last_use
                var2_live_here = var2_live and i <= var2_last_use

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

    def is_live_across_indirect_call(self, vreg: VirtualRegister) -> bool:
        """
        Check if a virtual register is live across any indirect Call.

        An indirect call is a call through a function pointer, where
        we don't know statically which function will be called.

        Args:
            vreg: Virtual register to check

        Returns:
            True if vreg is live across at least one indirect call
        """
        calls = self.get_calls_vreg_is_live_across(vreg)
        for call in calls:
            # Indirect call: function is a VirtualRegister, not a string
            if not isinstance(call.function, str):
                return True
        return False

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

                # Check if this is a Call and vreg is live after it
                if isinstance(instr, Call) and vreg_defined:
                    # Check if vreg is live after this call
                    if self.is_live_after(vreg, block_id, instr_idx):
                        calls.append(instr)

        self._live_across_calls_cache[vreg.id] = calls
        return calls

    def get_live_vregs_at_call(self, call_instr: Call) -> Set[VirtualRegister]:
        """
        Get all virtual registers that are live across a specific call.

        Args:
            call_instr: The Call instruction

        Returns:
            Set of VirtualRegisters live across this call
        """
        pos = self.get_instruction_position(call_instr)
        if not pos:
            return set()

        block_id, instr_idx = pos

        # Get vregs live after the call
        live_after = self._get_live_after(block_id, instr_idx)

        # Filter to only those defined before the call
        result = set()
        block = self.func.blocks[block_id]
        info = self.liveness.get(block_id)

        for vreg in live_after:
            # Check if vreg was defined before this call
            vreg_defined = vreg in info.live_in if info else False

            for i in range(instr_idx):
                defs = self.block_analyzer._get_defs(block.instructions[i])
                if vreg in defs:
                    vreg_defined = True
                    break

            if vreg_defined:
                result.add(vreg)

        return result
