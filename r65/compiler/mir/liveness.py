"""
Liveness analysis for MIR virtual registers.

Determines which virtual registers are live (in use) at each program point,
enabling optimizations like stack slot reuse and better register allocation.
"""

from typing import Set, Dict, List, Optional
from dataclasses import dataclass, field
from r65.compiler.mir.nodes import (
    MIRInstruction, VirtualRegister, HardwareRegister,
    BasicBlock, MIRFunction,
    Load, Store, Move, BinaryOp, UnaryOp, Compare, BitTest, Rotate,
    Call, Return, Jump, CondBranch, TypeConvert, ToBool,
    LoadIndirect, StoreIndirect
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
