"""
Mode tracking through control flow for R65.

Current implementation is simplified - it returns the function's entry mode
for all statements. A full implementation would use dataflow analysis to
propagate processor modes through the CFG, tracking mode changes from
SEP/REP instructions.
"""

from r65.compiler.typeck.processor_mode import ProcessorMode


class ModeTracker:
    """
    Tracks processor modes through control flow.

    Current implementation returns entry mode for all statements.
    A full implementation would propagate modes through the CFG.
    """

    def __init__(self, cfg, entry_mode: ProcessorMode):
        self.cfg = cfg
        self.entry_mode = entry_mode

    def analyze(self):
        """
        Perform mode analysis on the CFG.

        Current implementation is a no-op since we return entry_mode everywhere.
        A full implementation would use a worklist algorithm to propagate modes.
        """
        pass

    def get_mode_at_statement(self, stmt) -> ProcessorMode:
        """
        Get processor mode at a specific statement.

        Returns the function's entry mode for all statements.
        """
        return self.entry_mode
