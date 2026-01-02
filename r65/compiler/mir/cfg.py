"""
Control Flow Graph utilities for MIR.

Provides helper functions for building and manipulating CFG structures.
"""

from r65.compiler.mir.nodes import BasicBlock, MIRFunction, Return
from typing import List


class CFGBuilder:
    """
    Helper for building control flow graphs.

    Provides utilities for:
    - Creating basic blocks
    - Linking blocks with edges
    - Managing block predecessors/successors
    """

    def __init__(self, function: MIRFunction):
        self.function = function
        self.next_block_id = 0

    def new_block(self) -> BasicBlock:
        """
        Create a new basic block and add it to the function.

        Returns:
            Newly created BasicBlock
        """
        block = BasicBlock(block_id=self.next_block_id)
        self.next_block_id += 1
        self.function.blocks[block.block_id] = block
        return block

    def add_edge(self, from_block: BasicBlock, to_block: BasicBlock):
        """
        Add a control flow edge between two blocks.

        Args:
            from_block: Source block
            to_block: Destination block
        """
        if to_block.block_id not in from_block.successors:
            from_block.successors.append(to_block.block_id)
        if from_block.block_id not in to_block.predecessors:
            to_block.predecessors.append(from_block.block_id)

    def remove_edge(self, from_block: BasicBlock, to_block: BasicBlock):
        """
        Remove a control flow edge between two blocks.

        Args:
            from_block: Source block
            to_block: Destination block
        """
        if to_block.block_id in from_block.successors:
            from_block.successors.remove(to_block.block_id)
        if from_block.block_id in to_block.predecessors:
            to_block.predecessors.remove(from_block.block_id)

    def get_block(self, block_id: int) -> BasicBlock:
        """
        Get block by ID.

        Args:
            block_id: Block ID

        Returns:
            BasicBlock with given ID
        """
        return self.function.blocks[block_id]

    def compute_predecessors(self):
        """
        Recompute predecessor lists from successor lists.

        Useful after manually modifying successor lists.
        """
        # Clear all predecessor lists
        for block in self.function.blocks.values():
            block.predecessors.clear()

        # Rebuild from successors
        for block in self.function.blocks.values():
            for succ_id in block.successors:
                succ_block = self.function.blocks[succ_id]
                if block.block_id not in succ_block.predecessors:
                    succ_block.predecessors.append(block.block_id)

    def find_exit_blocks(self) -> List[int]:
        """
        Find all exit blocks (blocks with no successors or Return instructions).

        Returns:
            List of exit block IDs
        """
        exit_blocks = []
        for block in self.function.blocks.values():
            # Block with no successors is an exit
            if not block.successors:
                exit_blocks.append(block.block_id)
            # Block ending with Return is an exit
            elif block.instructions and isinstance(block.instructions[-1], Return):
                exit_blocks.append(block.block_id)
        return exit_blocks
