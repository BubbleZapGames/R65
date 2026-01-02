"""
Mode tracking through control flow for R65.

Uses dataflow analysis to propagate processor modes through the CFG.
"""

from typing import Dict
from r65.compiler.typeck.processor_mode import *


class ModeTracker:
    """
    Tracks processor modes through control flow.

    Uses worklist algorithm to propagate modes through CFG.
    """

    def __init__(self, cfg, entry_mode: ProcessorMode):
        self.cfg = cfg
        self.entry_mode = entry_mode

        # Maps block_id -> mode at block entry
        self.block_entry_modes: Dict[int, ProcessorMode] = {}
        self.block_exit_modes: Dict[int, ProcessorMode] = {}

    def analyze(self):
        """
        Perform mode analysis on the CFG.

        Stub implementation - just sets all blocks to entry mode.
        """
        # Initialize all blocks with entry mode
        for block_id in self.cfg.blocks:
            self.block_entry_modes[block_id] = self.entry_mode
            self.block_exit_modes[block_id] = self.entry_mode

    def get_mode_at_statement(self, stmt) -> ProcessorMode:
        """
        Get processor mode at a specific statement.

        Stub implementation - returns entry mode for now.
        """
        return self.entry_mode
