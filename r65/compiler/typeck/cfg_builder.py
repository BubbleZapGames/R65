"""
Simplified Control Flow Graph builder for mode tracking.

This is a minimal CFG that creates a single basic block containing all statements.
A full CFG implementation with proper control flow handling would be needed for
sophisticated mode tracking through branches and loops.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    block_id: int
    statements: List = field(default_factory=list)


@dataclass
class CFG:
    """Control Flow Graph."""
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_block_id: int = 0
    exit_block_ids: List[int] = field(default_factory=list)


class CFGBuilder:
    """Builds a CFG from HIR statements."""

    def __init__(self):
        self.cfg = CFG()
        self.next_block_id = 0

    def build(self, body) -> CFG:
        """
        Build CFG from function body.

        Current implementation creates a single basic block with all statements.
        This is sufficient for basic type checking but does not track control flow.
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
