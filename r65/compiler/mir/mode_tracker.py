# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Mode tracking through MIR CFG.

In the simplified mode system:
- X/Y registers are always 16-bit (x16 mode)
- A register mode (m8/m16) is inferred from parameter types
- Auto REP/SEP is inserted around 16-bit A operations when in m8 mode

Performs dataflow analysis to track processor M mode (M8/M16)
through the control flow graph.
"""

from typing import Dict, List, Set, Optional
from r65.compiler.typeck.processor_mode import ProcessorMode, ModeState, XModeState
from r65.compiler.mir.nodes import *


class MIRModeTracker:
    """
    Tracks processor modes through MIR CFG using dataflow analysis.

    In R65:
    - X mode is always X16 (16-bit index registers)
    - M mode is inferred from A parameter type (m8 default, m16 if u16 @ A)

    Algorithm:
    1. Initialize entry block with function's inferred entry mode
    2. Propagate modes through CFG using worklist algorithm
    3. At each block:
       - Entry mode = join of predecessor exit modes
       - Exit mode = apply instructions (SetMode) to entry mode
    4. Detect mode conflicts at merge points
    """

    def __init__(self, mir_func: MIRFunction):
        self.mir_func = mir_func
        # Track which blocks have been processed
        self.visited: Set[int] = set()

    def analyze(self) -> bool:
        """
        Perform mode analysis on MIR function.

        Returns:
            True if analysis succeeded, False if mode conflicts detected

        Side effects:
            - Sets entry_mode and exit_mode on each BasicBlock
        """
        # Get entry mode from inferred entry_m_mode (based on A parameter type)
        if self.mir_func.entry_m_mode is not None:
            entry_mode = ProcessorMode(self.mir_func.entry_m_mode, XModeState.X16)
        else:
            # Default: m8, x16
            entry_mode = ProcessorMode.default()

        # Set entry block mode
        entry_block = self.mir_func.blocks[self.mir_func.entry_block_id]
        entry_block.entry_mode = entry_mode

        # Worklist algorithm: process blocks until convergence
        worklist = [self.mir_func.entry_block_id]
        self.visited.clear()

        while worklist:
            block_id = worklist.pop(0)
            block = self.mir_func.blocks[block_id]

            # If block has no entry mode yet, compute it from predecessors
            if block.entry_mode is None and block_id != self.mir_func.entry_block_id:
                entry_mode = self._compute_entry_mode(block)
                if entry_mode is None:
                    # Mode conflict at merge point
                    return False
                block.entry_mode = entry_mode

            # Propagate mode through block instructions
            exit_mode = self._propagate_through_block(block)
            block.exit_mode = exit_mode

            # Mark as visited
            self.visited.add(block_id)

            # Propagate to successors
            for succ_id in block.successors:
                succ_block = self.mir_func.blocks[succ_id]

                # Compute what the successor's entry mode should be
                new_entry_mode = self._compute_entry_mode(succ_block)
                if new_entry_mode is None:
                    # Mode conflict
                    return False

                # If successor's entry mode changed, re-process it
                if succ_block.entry_mode != new_entry_mode:
                    succ_block.entry_mode = new_entry_mode
                    if succ_id not in worklist:
                        worklist.append(succ_id)

        return True

    def _compute_entry_mode(self, block: BasicBlock) -> Optional[ProcessorMode]:
        """
        Compute entry mode for a block by joining predecessor exit modes.

        Args:
            block: Block to compute entry mode for

        Returns:
            Joined ProcessorMode, or None if modes conflict
        """
        if not block.predecessors:
            # No predecessors - use default mode (m8, x16)
            return ProcessorMode.default()

        # Get exit modes from all predecessors
        pred_modes = []
        for pred_id in block.predecessors:
            pred_block = self.mir_func.blocks[pred_id]
            if pred_block.exit_mode is not None:
                pred_modes.append(pred_block.exit_mode)

        if not pred_modes:
            # No predecessor modes computed yet
            return None

        # Join all predecessor modes
        result_mode = pred_modes[0]
        for mode in pred_modes[1:]:
            result_mode = result_mode.join(mode)
            if result_mode is None:
                # Mode conflict - predecessors have incompatible modes
                return None

        return result_mode

    def _propagate_through_block(self, block: BasicBlock) -> ProcessorMode:
        """
        Propagate mode through block instructions.

        Applies SetMode instructions to compute exit mode.

        Args:
            block: Block to propagate through

        Returns:
            Exit mode after applying all instructions
        """
        current_mode = block.entry_mode
        if current_mode is None:
            current_mode = ProcessorMode.default()

        # Apply each instruction's effect on mode
        for instr in block.instructions:
            if isinstance(instr, SetMode):
                if instr.is_set:
                    # SEP instruction - set bits (8-bit mode)
                    current_mode = current_mode.apply_sep(instr.mask)
                else:
                    # REP instruction - reset bits (16-bit mode)
                    current_mode = current_mode.apply_rep(instr.mask)

        return current_mode

    def print_mode_info(self):
        """Print mode information for debugging."""
        print(f"Mode tracking for function '{self.mir_func.name}':")
        print(f"  Function mode attribute: {self.mir_func.mode_attr}")
        print()

        for block_id in sorted(self.mir_func.blocks.keys()):
            block = self.mir_func.blocks[block_id]
            print(f"  Block {block_id}:")
            print(f"    Entry mode:  {block.entry_mode}")
            print(f"    Exit mode:   {block.exit_mode}")

            # Show SetMode instructions
            for instr in block.instructions:
                if isinstance(instr, SetMode):
                    op_name = "SEP" if instr.is_set else "REP"
                    print(f"      {op_name} #${instr.mask:02X}")
            print()


def reanalyze_function(mir_func: MIRFunction) -> bool:
    """
    Recompute per-block mode metadata for a function whose MIR has been
    mutated since the last analysis (currently: by the inliner).

    Called after MIR-mutating passes. Clears stale `entry_mode` /
    `exit_mode` on every block, then re-runs the fixpoint via a fresh
    `MIRModeTracker`. This is the canonical "invalidate analyses,
    recompute lazily" pattern (cf. LLVM `PreservedAnalyses::none()`):
    the inliner's job is to leave the CFG valid; this helper makes the
    decorating metadata catch up.

    Returns the same bool as `MIRModeTracker.analyze()` — True on
    success, False on irreconcilable mode conflict at a merge point.
    """
    for block in mir_func.blocks.values():
        block.entry_mode = None
        block.exit_mode = None
    return MIRModeTracker(mir_func).analyze()
