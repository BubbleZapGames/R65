# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Mode tracking through MIR CFG.

X/Y are always 16-bit (x16). A function enters in m16 if it has a
`u16 @ A` parameter, otherwise m8; SetMode instructions (the inliner's
SEP/REP around a spliced body) change the M mode from there.

Performs dataflow analysis to track processor M mode (M8/M16)
through the control flow graph.
"""

from typing import Optional
from r65.compiler.typeck.processor_mode import ProcessorMode, XModeState
from r65.compiler.mir.nodes import *


class MIRModeTracker:
    """
    Tracks processor modes through MIR CFG using dataflow analysis.

    Algorithm (worklist from the entry block, which takes the function's
    inferred entry mode):
       - Exit mode = entry mode with the block's SetMode instructions applied
       - A successor's entry mode = join of its predecessors' exit modes;
         when that changes, the successor is processed (again)
       - Predecessors whose modes cannot be joined are a conflict
    """

    def __init__(self, mir_func: MIRFunction):
        self.mir_func = mir_func

    def analyze(self) -> bool:
        """
        Perform mode analysis on MIR function.

        Returns:
            True if analysis succeeded, False if mode conflicts detected

        Side effects:
            - Sets entry_mode and exit_mode on each reachable BasicBlock
        """
        # Get entry mode from inferred entry_m_mode (based on A parameter type)
        if self.mir_func.entry_m_mode is not None:
            entry_mode = ProcessorMode(self.mir_func.entry_m_mode, XModeState.X16)
        else:
            # Default: m8, x16
            entry_mode = ProcessorMode.default()

        entry_block = self.mir_func.blocks[self.mir_func.entry_block_id]
        entry_block.entry_mode = entry_mode

        # Every block on the worklist has its entry mode set.
        worklist = [self.mir_func.entry_block_id]
        while worklist:
            block = self.mir_func.blocks[worklist.pop(0)]
            block.exit_mode = self._propagate_through_block(block)

            for succ_id in block.successors:
                succ_block = self.mir_func.blocks[succ_id]
                new_entry_mode = self._compute_entry_mode(succ_block)
                if new_entry_mode is None:
                    return False
                if succ_block.entry_mode != new_entry_mode:
                    succ_block.entry_mode = new_entry_mode
                    if succ_id not in worklist:
                        worklist.append(succ_id)

        return True

    def _compute_entry_mode(self, block: BasicBlock) -> Optional[ProcessorMode]:
        """
        Join the exit modes of *block*'s predecessors processed so far (the
        block just propagated from is always one of them).

        Returns:
            Joined ProcessorMode, or None if the modes conflict
        """
        pred_modes = [self.mir_func.blocks[pred_id].exit_mode
                      for pred_id in block.predecessors]
        pred_modes = [mode for mode in pred_modes if mode is not None]

        result_mode = pred_modes[0]
        for mode in pred_modes[1:]:
            result_mode = result_mode.join(mode)
            if result_mode is None:
                return None
        return result_mode

    def _propagate_through_block(self, block: BasicBlock) -> ProcessorMode:
        """The block's exit mode: its entry mode with each SetMode applied
        (SEP sets bits, 8-bit; REP clears them, 16-bit)."""
        current_mode = block.entry_mode
        for instr in block.instructions:
            if isinstance(instr, SetMode):
                if instr.is_set:
                    current_mode = current_mode.apply_sep(instr.mask)
                else:
                    current_mode = current_mode.apply_rep(instr.mask)
        return current_mode


def reanalyze_function(mir_func: MIRFunction) -> bool:
    """
    Clear and recompute every block's entry/exit mode. Run after building
    a function and again after any pass that edits its CFG (the inliner).

    Returns the same bool as `MIRModeTracker.analyze()` — True on
    success, False on irreconcilable mode conflict at a merge point.
    """
    for block in mir_func.blocks.values():
        block.entry_mode = None
        block.exit_mode = None
    return MIRModeTracker(mir_func).analyze()
