"""
Peephole optimization for R65 assembly code.

Applies local optimizations to AsmNode instruction sequences to eliminate
redundant operations and improve code quality.
"""

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from r65.compiler.codegen.asm_nodes import AsmNode


class NodePeepholeOptimizer:
    """
    Peephole optimizer that works directly on AsmNode objects.

    This avoids the string parsing overhead and provides type-safe access
    to instruction opcodes and operands.
    """

    def __init__(self):
        self.optimizations_applied = 0

    def optimize(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Apply peephole optimizations to AsmNode list.

        Args:
            nodes: List of AsmNode objects

        Returns:
            Optimized node list
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Address
        from r65.compiler.codegen.opcodes import Opcode, mnemonic

        # Apply optimization passes
        nodes = self._eliminate_redundant_load_after_store(nodes)
        nodes = self._eliminate_dead_stores(nodes)

        return nodes

    def _eliminate_redundant_load_after_store(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant LDA immediately after STA to same location.

        Pattern: STA $XX; LDA $XX -> STA $XX

        After storing A to memory, the value is still in A, so loading it back is redundant.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Address
        from r65.compiler.codegen.opcodes import Opcode, mnemonic

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            # Only process Instruction nodes
            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            # Check for STA followed by LDA to same address
            if mnemonic(node.opcode) == "STA" and i + 1 < len(nodes):
                next_node = nodes[i + 1]

                if (isinstance(next_node, Instruction) and
                    mnemonic(next_node.opcode) == "LDA" and
                    node.operand == next_node.operand):
                    # Redundant load after store - skip the LDA
                    optimized.append(node)
                    i += 2  # Skip both instructions (we keep STA, skip LDA)
                    self.optimizations_applied += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_dead_stores(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate dead stores that are immediately overwritten.

        Pattern: STA $XX; ... (no read of $XX); STA $XX -> ... ; STA $XX

        If a value is stored but then overwritten before being read, the first store is dead.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address
        from r65.compiler.codegen.opcodes import Opcode, mnemonic, is_branch, is_return

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            # Only process Instruction nodes
            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            # Check for STA
            if mnemonic(node.opcode) != "STA":
                optimized.append(node)
                i += 1
                continue

            store_operand = node.operand

            # Skip indexed addressing - can't reliably track
            if store_operand and isinstance(store_operand, Address):
                # Check if it looks like indexed addressing (contains ,X or ,Y in the value)
                if isinstance(store_operand.value, str) and ',' in store_operand.value:
                    optimized.append(node)
                    i += 1
                    continue

            # Look ahead to see if there's another store to same address
            j = i + 1
            is_dead = False

            while j < len(nodes):
                next_node = nodes[j]

                # If we hit a label, stop looking
                if isinstance(next_node, Label):
                    break

                # If not an instruction, skip
                if not isinstance(next_node, Instruction):
                    j += 1
                    continue

                # If we hit a control flow change, stop
                if is_return(next_node.opcode) or mnemonic(next_node.opcode) in ('JMP', 'BRA'):
                    break

                # If we find another store to same address, first store is dead
                if mnemonic(next_node.opcode) == "STA" and next_node.operand == store_operand:
                    is_dead = True
                    break

                # If we find a read of the address, store is not dead
                if (mnemonic(next_node.opcode) in ('LDA', 'ADC', 'SBC', 'AND', 'ORA', 'EOR', 'CMP') and
                    next_node.operand == store_operand):
                    break

                # If we find a branch, stop looking (conservative)
                if is_branch(next_node.opcode):
                    break

                j += 1

            if is_dead:
                # Skip this dead store
                i += 1
                self.optimizations_applied += 1
                continue

            optimized.append(node)
            i += 1

        return optimized


def optimize_nodes(nodes: List['AsmNode']) -> Tuple[List['AsmNode'], int]:
    """
    Apply peephole optimizations to AsmNode list.

    Args:
        nodes: List of AsmNode objects

    Returns:
        Tuple of (optimized nodes, number of optimizations applied)
    """
    optimizer = NodePeepholeOptimizer()
    optimized = optimizer.optimize(nodes)
    return optimized, optimizer.optimizations_applied
