"""
Simplified Control Flow Graph builder for mode tracking.

CURRENT LIMITATION:
===================
This is a minimal CFG that creates a SINGLE basic block containing all
statements. Control flow (if/else, loops, match) is NOT represented.

1. IMPLICATIONS:
   - Mode tracking cannot distinguish between different code paths
   - Dead code analysis is not possible at the HIR/typeck level
   - Loop-specific checks cannot be implemented
   - Register preservation across branches is not verified

2. WHY THIS IS ACCEPTABLE:
   - R65's automatic mode management via function signatures means most
     code doesn't need intra-function mode tracking
   - Code generation (MIR level) has its own control flow handling
   - Dead code elimination happens at the optimization level
   - The simplified CFG is sufficient for basic type checking

3. FUTURE IMPLEMENTATION:
   A full CFG implementation would need:
   - Separate blocks for if/else branches
   - Loop back edges with cycle detection
   - Proper successor/predecessor tracking
   - Integration with mode_tracker.py for dataflow analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    block_id: int
    statements: List = field(default_factory=list)
    # FUTURE: Add these fields for proper control flow
    # successors: List[int] = field(default_factory=list)
    # predecessors: List[int] = field(default_factory=list)


@dataclass
class CFG:
    """Control Flow Graph."""
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_block_id: int = 0
    exit_block_ids: List[int] = field(default_factory=list)


class CFGBuilder:
    """
    Builds a CFG from HIR statements.

    CURRENT LIMITATION: Creates a single basic block with all statements.
    See module docstring for details.
    """

    def __init__(self):
        self.cfg = CFG()
        self.next_block_id = 0

    def build(self, body) -> CFG:
        """
        Build CFG from function body.

        CURRENT IMPLEMENTATION: Creates a single basic block with all statements.
        This is sufficient for basic type checking but does not track control flow.

        FUTURE: Would recursively process the body, creating new blocks at:
        - If statement branches (then/else blocks)
        - Loop bodies (with back edges)
        - Match expression arms
        - Return/break/continue terminators

        Args:
            body: HIR block statement containing the function body

        Returns:
            CFG with single block (current) or proper control flow (future)
        """
        entry_id = self._new_block()
        self.cfg.entry_block_id = entry_id
        self.cfg.exit_block_ids = [entry_id]

        for stmt in body.statements:
            self.cfg.blocks[entry_id].statements.append(stmt)

        return self.cfg

    def _new_block(self) -> int:
        """Create a new basic block."""
        block_id = self.next_block_id
        self.next_block_id += 1
        self.cfg.blocks[block_id] = BasicBlock(block_id=block_id)
        return block_id
