# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Mode tracking through control flow for R65.

CURRENT LIMITATION:
===================
This implementation is simplified - it returns the function's entry mode
for ALL statements within a function. This means:

1. WHAT WORKS CORRECTLY:
   - Functions with @ A: u8 parameter are type-checked as m8 mode throughout
   - Functions with @ A: u16 parameter are type-checked as m16 mode throughout
   - The compiler automatically generates SEP/REP at function entry/exit
   - Cross-function calls with mode transitions work correctly

2. WHAT MAY NOT BE CORRECTLY TYPE-CHECKED:
   - Explicit asm!("SEP #$20") or asm!("REP #$20") within a function
   - Code that manually switches modes and expects the type checker to track it
   - B register access after manual mode switches

3. WHY THIS LIMITATION EXISTS:
   The R65 language is designed to manage modes automatically via function
   signatures (@ A: u8 vs @ A: u16). Manual mode switching via inline assembly
   is considered an advanced/escape-hatch feature that bypasses the type system.

4. FUTURE IMPLEMENTATION:
   A full implementation would use dataflow analysis to propagate processor
   modes through the CFG, tracking mode changes from SEP/REP instructions.
   This would require:
   - Proper CFG with control flow edges (see cfg_builder.py)
   - Recognition of mode-changing instructions in inline assembly
   - Worklist-based dataflow analysis
   - Mode state merging at control flow join points
"""

from r65.compiler.typeck.processor_mode import ProcessorMode


class ModeTracker:
    """
    Tracks processor modes through control flow.

    CURRENT LIMITATION: Returns entry mode for all statements.
    See module docstring for details on what this means for type checking.

    A full implementation would propagate modes through the CFG using
    dataflow analysis, tracking mode changes from SEP/REP instructions.
    """

    def __init__(self, cfg, entry_mode: ProcessorMode):
        self.cfg = cfg
        self.entry_mode = entry_mode

    def analyze(self):
        """
        Perform mode analysis on the CFG.

        CURRENT IMPLEMENTATION: No-op - returns entry_mode everywhere.

        FUTURE: Use a worklist algorithm to propagate modes through the CFG,
        handling SEP/REP instructions and merging at join points.
        """
        pass

    def get_mode_at_statement(self, stmt) -> ProcessorMode:
        """
        Get processor mode at a specific statement.

        CURRENT IMPLEMENTATION: Returns entry mode for all statements.

        This is correct for R65 code that uses automatic mode management via
        function signatures. For code using manual SEP/REP via inline assembly,
        the returned mode may not reflect the actual CPU state.

        Args:
            stmt: The HIR statement to query

        Returns:
            ProcessorMode at the statement (currently always entry_mode)
        """
        return self.entry_mode
