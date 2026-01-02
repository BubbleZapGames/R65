"""
Simplified Control Flow Graph builder for mode tracking.

This is a lightweight CFG focused on tracking modes, not full optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    block_id: int
    statements: List = field(default_factory=list)
    successors: List[int] = field(default_factory=list)  # Block IDs
    predecessors: List[int] = field(default_factory=list)

    # For mode tracking
    entry_mode: Optional['ProcessorMode'] = None
    exit_mode: Optional['ProcessorMode'] = None

    # For loop tracking
    is_loop_header: bool = False
    is_loop_exit: bool = False


@dataclass
class CFG:
    """Control Flow Graph."""
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_block_id: int = 0
    exit_block_ids: List[int] = field(default_factory=list)

    # Loop tracking
    break_targets: Dict[int, int] = field(default_factory=dict)
    continue_targets: Dict[int, int] = field(default_factory=dict)


class CFGBuilder:
    """Builds a CFG from HIR statements."""

    def __init__(self):
        self.cfg = CFG()
        self.next_block_id = 0
        self.loop_stack: List[tuple] = []  # (continue_target, break_target)

    def build(self, body) -> CFG:
        """Build CFG from function body."""
        # Stub implementation - creates single block for now
        entry_id = self._new_block()
        self.cfg.entry_block_id = entry_id
        self.cfg.exit_block_ids = [entry_id]

        # Add all statements to single block
        for stmt in body.statements:
            self.cfg.blocks[entry_id].statements.append(stmt)

        return self.cfg

    def _new_block(self) -> int:
        """Create a new basic block."""
        block_id = self.next_block_id
        self.next_block_id += 1
        self.cfg.blocks[block_id] = BasicBlock(block_id=block_id)
        return block_id
