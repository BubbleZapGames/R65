"""
Peephole optimization for R65 assembly code.

Applies local optimizations to AsmNode instruction sequences to eliminate
redundant operations and improve code quality.

Uses typed Opcode enum for efficient pattern matching without string parsing.
"""

from typing import List, Tuple, TYPE_CHECKING, Set
from dataclasses import dataclass

from r65.compiler.codegen.opcodes import (
    Opcode, is_branch, is_return, is_load, is_store,
    BRANCH_OPCODES, JUMP_OPCODES,
)

if TYPE_CHECKING:
    from r65.compiler.codegen.asm_nodes import AsmNode, Instruction, Label


# ============================================================================
# Opcode Categories for Pattern Matching
# ============================================================================

# Load instructions by register
LOAD_A_OPCODES: Set[Opcode] = {
    Opcode.LDA_IMMEDIATE, Opcode.LDA_DP, Opcode.LDA_DP_X,
    Opcode.LDA_ABSOLUTE, Opcode.LDA_ABSOLUTE_X, Opcode.LDA_ABSOLUTE_Y,
    Opcode.LDA_DP_INDIRECT, Opcode.LDA_DP_INDIRECT_X, Opcode.LDA_DP_INDIRECT_Y,
    Opcode.LDA_DP_INDIRECT_LONG, Opcode.LDA_DP_INDIRECT_LONG_Y,
    Opcode.LDA_LONG, Opcode.LDA_LONG_X,
    Opcode.LDA_STACK, Opcode.LDA_STACK_INDIRECT_Y,
}

LOAD_X_OPCODES: Set[Opcode] = {
    Opcode.LDX_IMMEDIATE, Opcode.LDX_DP, Opcode.LDX_DP_Y,
    Opcode.LDX_ABSOLUTE, Opcode.LDX_ABSOLUTE_Y,
}

LOAD_Y_OPCODES: Set[Opcode] = {
    Opcode.LDY_IMMEDIATE, Opcode.LDY_DP, Opcode.LDY_DP_X,
    Opcode.LDY_ABSOLUTE, Opcode.LDY_ABSOLUTE_X,
}

# Store instructions by register
STORE_A_OPCODES: Set[Opcode] = {
    Opcode.STA_DP, Opcode.STA_DP_X,
    Opcode.STA_ABSOLUTE, Opcode.STA_ABSOLUTE_X, Opcode.STA_ABSOLUTE_Y,
    Opcode.STA_DP_INDIRECT, Opcode.STA_DP_INDIRECT_X, Opcode.STA_DP_INDIRECT_Y,
    Opcode.STA_DP_INDIRECT_LONG, Opcode.STA_DP_INDIRECT_LONG_Y,
    Opcode.STA_LONG, Opcode.STA_LONG_X,
    Opcode.STA_STACK, Opcode.STA_STACK_INDIRECT_Y,
}

STORE_X_OPCODES: Set[Opcode] = {
    Opcode.STX_DP, Opcode.STX_DP_Y, Opcode.STX_ABSOLUTE,
}

STORE_Y_OPCODES: Set[Opcode] = {
    Opcode.STY_DP, Opcode.STY_DP_X, Opcode.STY_ABSOLUTE,
}

# Instructions that read A (for dead store analysis)
READS_A_OPCODES: Set[Opcode] = STORE_A_OPCODES | {
    Opcode.ADC_IMMEDIATE, Opcode.ADC_DP, Opcode.ADC_ABSOLUTE,
    Opcode.SBC_IMMEDIATE, Opcode.SBC_DP, Opcode.SBC_ABSOLUTE,
    Opcode.AND_IMMEDIATE, Opcode.AND_DP, Opcode.AND_ABSOLUTE,
    Opcode.ORA_IMMEDIATE, Opcode.ORA_DP, Opcode.ORA_ABSOLUTE,
    Opcode.EOR_IMMEDIATE, Opcode.EOR_DP, Opcode.EOR_ABSOLUTE,
    Opcode.CMP_IMMEDIATE, Opcode.CMP_DP, Opcode.CMP_ABSOLUTE,
    Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR,
    Opcode.TAX, Opcode.TAY, Opcode.PHA,
    Opcode.XBA,  # Exchanges A and B
}

# Instructions that modify A
MODIFIES_A_OPCODES: Set[Opcode] = LOAD_A_OPCODES | {
    Opcode.ADC_IMMEDIATE, Opcode.ADC_DP, Opcode.ADC_ABSOLUTE,
    Opcode.ADC_DP_X, Opcode.ADC_ABSOLUTE_X, Opcode.ADC_ABSOLUTE_Y,
    Opcode.SBC_IMMEDIATE, Opcode.SBC_DP, Opcode.SBC_ABSOLUTE,
    Opcode.SBC_DP_X, Opcode.SBC_ABSOLUTE_X, Opcode.SBC_ABSOLUTE_Y,
    Opcode.AND_IMMEDIATE, Opcode.AND_DP, Opcode.AND_ABSOLUTE,
    Opcode.AND_DP_X, Opcode.AND_ABSOLUTE_X, Opcode.AND_ABSOLUTE_Y,
    Opcode.ORA_IMMEDIATE, Opcode.ORA_DP, Opcode.ORA_ABSOLUTE,
    Opcode.ORA_DP_X, Opcode.ORA_ABSOLUTE_X, Opcode.ORA_ABSOLUTE_Y,
    Opcode.EOR_IMMEDIATE, Opcode.EOR_DP, Opcode.EOR_ABSOLUTE,
    Opcode.EOR_DP_X, Opcode.EOR_ABSOLUTE_X, Opcode.EOR_ABSOLUTE_Y,
    Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR,
    Opcode.INC, Opcode.DEC,
    Opcode.TXA, Opcode.TYA, Opcode.PLA,
    Opcode.XBA,  # Exchanges A and B
}

# Transfer instructions (for redundant transfer elimination)
TRANSFER_OPCODES: Set[Opcode] = {
    Opcode.TAX, Opcode.TAY, Opcode.TXA, Opcode.TYA,
    Opcode.TXY, Opcode.TYX,
}

# Push/pull pairs for redundant stack operation elimination
PUSH_PULL_PAIRS = {
    Opcode.PHA: Opcode.PLA,
    Opcode.PHX: Opcode.PLX,
    Opcode.PHY: Opcode.PLY,
    Opcode.PHP: Opcode.PLP,
    Opcode.PHD: Opcode.PLD,
    Opcode.PHB: Opcode.PLB,
}

# Map store opcode sets to corresponding load opcode sets (for redundant load elimination)
STORE_TO_LOAD_MAP = {
    # Each store opcode maps to the set of corresponding load opcodes
    **{op: LOAD_A_OPCODES for op in STORE_A_OPCODES},
    **{op: LOAD_X_OPCODES for op in STORE_X_OPCODES},
    **{op: LOAD_Y_OPCODES for op in STORE_Y_OPCODES},
}

# Instructions that read from a memory operand (for dead store analysis)
READS_FROM_MEMORY_OPCODES: Set[Opcode] = (
    LOAD_A_OPCODES | LOAD_X_OPCODES | LOAD_Y_OPCODES |
    # ADC variants
    {Opcode.ADC_DP, Opcode.ADC_DP_X, Opcode.ADC_ABSOLUTE,
     Opcode.ADC_ABSOLUTE_X, Opcode.ADC_ABSOLUTE_Y,
     Opcode.ADC_DP_INDIRECT, Opcode.ADC_DP_INDIRECT_X, Opcode.ADC_DP_INDIRECT_Y,
     Opcode.ADC_DP_INDIRECT_LONG, Opcode.ADC_DP_INDIRECT_LONG_Y,
     Opcode.ADC_LONG, Opcode.ADC_LONG_X, Opcode.ADC_STACK, Opcode.ADC_STACK_INDIRECT_Y} |
    # SBC variants
    {Opcode.SBC_DP, Opcode.SBC_DP_X, Opcode.SBC_ABSOLUTE,
     Opcode.SBC_ABSOLUTE_X, Opcode.SBC_ABSOLUTE_Y,
     Opcode.SBC_DP_INDIRECT, Opcode.SBC_DP_INDIRECT_X, Opcode.SBC_DP_INDIRECT_Y,
     Opcode.SBC_DP_INDIRECT_LONG, Opcode.SBC_DP_INDIRECT_LONG_Y,
     Opcode.SBC_LONG, Opcode.SBC_LONG_X, Opcode.SBC_STACK, Opcode.SBC_STACK_INDIRECT_Y} |
    # AND variants
    {Opcode.AND_DP, Opcode.AND_DP_X, Opcode.AND_ABSOLUTE,
     Opcode.AND_ABSOLUTE_X, Opcode.AND_ABSOLUTE_Y,
     Opcode.AND_DP_INDIRECT, Opcode.AND_DP_INDIRECT_X, Opcode.AND_DP_INDIRECT_Y,
     Opcode.AND_DP_INDIRECT_LONG, Opcode.AND_DP_INDIRECT_LONG_Y,
     Opcode.AND_LONG, Opcode.AND_LONG_X, Opcode.AND_STACK, Opcode.AND_STACK_INDIRECT_Y} |
    # ORA variants
    {Opcode.ORA_DP, Opcode.ORA_DP_X, Opcode.ORA_ABSOLUTE,
     Opcode.ORA_ABSOLUTE_X, Opcode.ORA_ABSOLUTE_Y,
     Opcode.ORA_DP_INDIRECT, Opcode.ORA_DP_INDIRECT_X, Opcode.ORA_DP_INDIRECT_Y,
     Opcode.ORA_DP_INDIRECT_LONG, Opcode.ORA_DP_INDIRECT_LONG_Y,
     Opcode.ORA_LONG, Opcode.ORA_LONG_X, Opcode.ORA_STACK, Opcode.ORA_STACK_INDIRECT_Y} |
    # EOR variants
    {Opcode.EOR_DP, Opcode.EOR_DP_X, Opcode.EOR_ABSOLUTE,
     Opcode.EOR_ABSOLUTE_X, Opcode.EOR_ABSOLUTE_Y,
     Opcode.EOR_DP_INDIRECT, Opcode.EOR_DP_INDIRECT_X, Opcode.EOR_DP_INDIRECT_Y,
     Opcode.EOR_DP_INDIRECT_LONG, Opcode.EOR_DP_INDIRECT_LONG_Y,
     Opcode.EOR_LONG, Opcode.EOR_LONG_X, Opcode.EOR_STACK, Opcode.EOR_STACK_INDIRECT_Y} |
    # CMP variants
    {Opcode.CMP_DP, Opcode.CMP_DP_X, Opcode.CMP_ABSOLUTE,
     Opcode.CMP_ABSOLUTE_X, Opcode.CMP_ABSOLUTE_Y,
     Opcode.CMP_DP_INDIRECT, Opcode.CMP_DP_INDIRECT_X, Opcode.CMP_DP_INDIRECT_Y,
     Opcode.CMP_DP_INDIRECT_LONG, Opcode.CMP_DP_INDIRECT_LONG_Y,
     Opcode.CMP_LONG, Opcode.CMP_LONG_X, Opcode.CMP_STACK, Opcode.CMP_STACK_INDIRECT_Y} |
    # CPX variants
    {Opcode.CPX_DP, Opcode.CPX_ABSOLUTE} |
    # CPY variants
    {Opcode.CPY_DP, Opcode.CPY_ABSOLUTE} |
    # BIT variants
    {Opcode.BIT_DP, Opcode.BIT_DP_X, Opcode.BIT_ABSOLUTE, Opcode.BIT_ABSOLUTE_X}
)

# Control flow instructions that end a basic block
CONTROL_FLOW_OPCODES: Set[Opcode] = (
    BRANCH_OPCODES | JUMP_OPCODES | {Opcode.RTS, Opcode.RTL, Opcode.RTI}
)


# ============================================================================
# Statistics Tracking
# ============================================================================

@dataclass
class OptimizationStats:
    """Track optimization statistics."""
    redundant_loads_eliminated: int = 0
    dead_stores_eliminated: int = 0
    redundant_transfers_eliminated: int = 0
    redundant_stack_ops_eliminated: int = 0
    redundant_mode_changes_eliminated: int = 0

    @property
    def total(self) -> int:
        return (
            self.redundant_loads_eliminated +
            self.dead_stores_eliminated +
            self.redundant_transfers_eliminated +
            self.redundant_stack_ops_eliminated +
            self.redundant_mode_changes_eliminated
        )


# ============================================================================
# Peephole Optimizer
# ============================================================================

class PeepholeOptimizer:
    """
    Peephole optimizer that works directly on AsmNode objects.

    Uses typed Opcode enum for efficient pattern matching without string parsing.
    Applies multiple optimization passes to eliminate redundant operations.
    """

    def __init__(self):
        self.stats = OptimizationStats()

    @property
    def optimizations_applied(self) -> int:
        """Total number of optimizations applied."""
        return self.stats.total

    def optimize(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Apply peephole optimizations to AsmNode list.

        Args:
            nodes: List of AsmNode objects

        Returns:
            Optimized node list
        """
        # Apply optimization passes until no more changes
        changed = True
        while changed:
            prev_total = self.stats.total
            nodes = self._eliminate_redundant_load_after_store(nodes)
            nodes = self._eliminate_dead_stores(nodes)
            nodes = self._eliminate_redundant_transfers(nodes)
            nodes = self._eliminate_redundant_stack_ops(nodes)
            nodes = self._eliminate_redundant_mode_changes(nodes)
            changed = self.stats.total > prev_total

        return nodes

    def _eliminate_redundant_load_after_store(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant load immediately after store to same location.

        Patterns:
            STA $XX; LDA $XX -> STA $XX  (A still contains the value)
            STX $XX; LDX $XX -> STX $XX
            STY $XX; LDY $XX -> STY $XX
        """
        from r65.compiler.codegen.asm_nodes import Instruction

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            # Check if this is a store opcode with a corresponding load set
            if node.opcode in STORE_TO_LOAD_MAP and i + 1 < len(nodes):
                next_node = nodes[i + 1]

                if isinstance(next_node, Instruction):
                    load_opcodes = STORE_TO_LOAD_MAP[node.opcode]

                    # Check if next is corresponding load with same operand
                    if (next_node.opcode in load_opcodes and
                        node.operand == next_node.operand and
                        self._same_addressing_mode(node.opcode, next_node.opcode)):
                        # Redundant load - skip it
                        optimized.append(node)
                        i += 2
                        self.stats.redundant_loads_eliminated += 1
                        continue

            optimized.append(node)
            i += 1

        return optimized

    def _same_addressing_mode(self, op1: Opcode, op2: Opcode) -> bool:
        """Check if two opcodes use the same addressing mode."""
        from r65.compiler.codegen.opcodes import addressing_mode
        return addressing_mode(op1) == addressing_mode(op2)

    def _eliminate_dead_stores(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate dead stores that are overwritten before being read.

        Pattern: STA $XX; ... (no read of $XX); STA $XX -> ... ; STA $XX
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            # Only check STA (most common case)
            if node.opcode not in STORE_A_OPCODES:
                optimized.append(node)
                i += 1
                continue

            store_operand = node.operand

            # Skip if no operand or if using indexed addressing (can't track reliably)
            if store_operand is None or self._is_indexed_addressing(node.opcode):
                optimized.append(node)
                i += 1
                continue

            # Look ahead to see if there's another store to same address
            # before the value is read
            is_dead = self._is_dead_store(nodes, i, store_operand)

            if is_dead:
                i += 1
                self.stats.dead_stores_eliminated += 1
                continue

            optimized.append(node)
            i += 1

        return optimized

    def _is_indexed_addressing(self, opcode: Opcode) -> bool:
        """Check if opcode uses indexed addressing."""
        name = opcode.name
        return '_X' in name or '_Y' in name

    def _is_dead_store(self, nodes: List['AsmNode'], store_idx: int, store_operand) -> bool:
        """Check if a store is dead (overwritten before read)."""
        from r65.compiler.codegen.asm_nodes import Instruction, Label

        j = store_idx + 1

        while j < len(nodes):
            next_node = nodes[j]

            # Label = potential branch target, stop analysis
            if isinstance(next_node, Label):
                return False

            if not isinstance(next_node, Instruction):
                j += 1
                continue

            # Control flow = stop analysis
            if next_node.opcode in CONTROL_FLOW_OPCODES:
                return False

            # Another store to same location = first store is dead
            if (next_node.opcode in STORE_A_OPCODES and
                next_node.operand == store_operand and
                not self._is_indexed_addressing(next_node.opcode)):
                return True

            # Read from same location = store is not dead
            if self._reads_from_location(next_node, store_operand):
                return False

            j += 1

        return False

    def _reads_from_location(self, instr: 'Instruction', operand) -> bool:
        """Check if instruction reads from the given memory location."""
        return instr.opcode in READS_FROM_MEMORY_OPCODES and instr.operand == operand

    def _eliminate_redundant_transfers(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant register transfers.

        Patterns:
            TAX; TXA -> TAX  (A still has value, TXA is redundant)
            TAY; TYA -> TAY
            TXA; TAX -> TXA
            TYA; TAY -> TYA
            TXY; TYX -> TXY
            TYX; TXY -> TYX
        """
        from r65.compiler.codegen.asm_nodes import Instruction

        # Map transfer to its reverse
        reverse_transfers = {
            Opcode.TAX: Opcode.TXA,
            Opcode.TAY: Opcode.TYA,
            Opcode.TXA: Opcode.TAX,
            Opcode.TYA: Opcode.TAY,
            Opcode.TXY: Opcode.TYX,
            Opcode.TYX: Opcode.TXY,
        }

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            if node.opcode in reverse_transfers and i + 1 < len(nodes):
                next_node = nodes[i + 1]

                if (isinstance(next_node, Instruction) and
                    next_node.opcode == reverse_transfers[node.opcode]):
                    # Redundant reverse transfer - keep first, skip second
                    optimized.append(node)
                    i += 2
                    self.stats.redundant_transfers_eliminated += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_redundant_stack_ops(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant push/pull pairs with nothing between them.

        Pattern: PHA; PLA -> (nothing, if A isn't needed on stack)

        Note: This is conservative - only removes immediately adjacent pairs.
        """
        from r65.compiler.codegen.asm_nodes import Instruction

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            if node.opcode in PUSH_PULL_PAIRS and i + 1 < len(nodes):
                next_node = nodes[i + 1]
                expected_pull = PUSH_PULL_PAIRS[node.opcode]

                if (isinstance(next_node, Instruction) and
                    next_node.opcode == expected_pull):
                    # Adjacent push/pull pair - remove both
                    i += 2
                    self.stats.redundant_stack_ops_eliminated += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_redundant_mode_changes(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant processor mode changes.

        Patterns:
            SEP #$XX; SEP #$XX -> SEP #$XX (same value)
            REP #$XX; REP #$XX -> REP #$XX (same value)
            SEP #$20; REP #$20 -> (cancel out for m flag)
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            if node.opcode in (Opcode.SEP, Opcode.REP) and i + 1 < len(nodes):
                next_node = nodes[i + 1]

                if isinstance(next_node, Instruction):
                    # Same instruction with same operand
                    if (next_node.opcode == node.opcode and
                        node.operand == next_node.operand):
                        # Duplicate - keep first, skip second
                        optimized.append(node)
                        i += 2
                        self.stats.redundant_mode_changes_eliminated += 1
                        continue

                    # SEP followed by REP (or vice versa) with same bits
                    if ((node.opcode == Opcode.SEP and next_node.opcode == Opcode.REP) or
                        (node.opcode == Opcode.REP and next_node.opcode == Opcode.SEP)):
                        if (isinstance(node.operand, Immediate) and
                            isinstance(next_node.operand, Immediate) and
                            node.operand.value == next_node.operand.value):
                            # Canceling mode changes - remove both
                            i += 2
                            self.stats.redundant_mode_changes_eliminated += 1
                            continue

            optimized.append(node)
            i += 1

        return optimized


# ============================================================================
# Public API
# ============================================================================

def optimize_nodes(nodes: List['AsmNode']) -> Tuple[List['AsmNode'], int]:
    """
    Apply peephole optimizations to AsmNode list.

    Args:
        nodes: List of AsmNode objects

    Returns:
        Tuple of (optimized nodes, number of optimizations applied)
    """
    optimizer = PeepholeOptimizer()
    optimized = optimizer.optimize(nodes)
    return optimized, optimizer.optimizations_applied
