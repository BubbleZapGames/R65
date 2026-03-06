"""
Peephole optimization for R65 assembly code.

Applies local optimizations to AsmNode instruction sequences to eliminate
redundant operations and improve code quality.

Uses typed Opcode enum for efficient pattern matching without string parsing.
"""

from typing import List, Tuple, TYPE_CHECKING, Set
from dataclasses import dataclass

from r65.compiler.codegen.opcodes import (
    Opcode,
    BRANCH_OPCODES, JUMP_OPCODES, CALL_OPCODES,
    LOAD_A_OPCODES, LOAD_X_OPCODES, LOAD_Y_OPCODES,
    STORE_A_OPCODES, STORE_X_OPCODES, STORE_Y_OPCODES,
)

if TYPE_CHECKING:
    from r65.compiler.codegen.asm_nodes import AsmNode, Instruction


# ============================================================================
# Opcode Categories for Pattern Matching
# ============================================================================

# Indirect addressing opcodes - these can read/write to any address
# Dead store analysis must be conservative when these are present
INDIRECT_ADDRESSING_OPCODES: Set[Opcode] = {
    # DP indirect
    Opcode.LDA_DP_INDIRECT, Opcode.LDA_DP_INDIRECT_X, Opcode.LDA_DP_INDIRECT_Y,
    Opcode.LDA_DP_INDIRECT_LONG, Opcode.LDA_DP_INDIRECT_LONG_Y,
    Opcode.STA_DP_INDIRECT, Opcode.STA_DP_INDIRECT_X, Opcode.STA_DP_INDIRECT_Y,
    Opcode.STA_DP_INDIRECT_LONG, Opcode.STA_DP_INDIRECT_LONG_Y,
    # Stack indirect
    Opcode.LDA_STACK_INDIRECT_Y, Opcode.STA_STACK_INDIRECT_Y,
    # ADC/AND/CMP/EOR/ORA/SBC indirect variants
    Opcode.ADC_DP_INDIRECT, Opcode.ADC_DP_INDIRECT_X, Opcode.ADC_DP_INDIRECT_Y,
    Opcode.ADC_DP_INDIRECT_LONG, Opcode.ADC_DP_INDIRECT_LONG_Y, Opcode.ADC_STACK_INDIRECT_Y,
    Opcode.AND_DP_INDIRECT, Opcode.AND_DP_INDIRECT_X, Opcode.AND_DP_INDIRECT_Y,
    Opcode.AND_DP_INDIRECT_LONG, Opcode.AND_DP_INDIRECT_LONG_Y, Opcode.AND_STACK_INDIRECT_Y,
    Opcode.CMP_DP_INDIRECT, Opcode.CMP_DP_INDIRECT_X, Opcode.CMP_DP_INDIRECT_Y,
    Opcode.CMP_DP_INDIRECT_LONG, Opcode.CMP_DP_INDIRECT_LONG_Y, Opcode.CMP_STACK_INDIRECT_Y,
    Opcode.EOR_DP_INDIRECT, Opcode.EOR_DP_INDIRECT_X, Opcode.EOR_DP_INDIRECT_Y,
    Opcode.EOR_DP_INDIRECT_LONG, Opcode.EOR_DP_INDIRECT_LONG_Y, Opcode.EOR_STACK_INDIRECT_Y,
    Opcode.ORA_DP_INDIRECT, Opcode.ORA_DP_INDIRECT_X, Opcode.ORA_DP_INDIRECT_Y,
    Opcode.ORA_DP_INDIRECT_LONG, Opcode.ORA_DP_INDIRECT_LONG_Y, Opcode.ORA_STACK_INDIRECT_Y,
    Opcode.SBC_DP_INDIRECT, Opcode.SBC_DP_INDIRECT_X, Opcode.SBC_DP_INDIRECT_Y,
    Opcode.SBC_DP_INDIRECT_LONG, Opcode.SBC_DP_INDIRECT_LONG_Y, Opcode.SBC_STACK_INDIRECT_Y,
}

# Instructions that read A (for dead store analysis)
READS_A_OPCODES: Set[Opcode] = STORE_A_OPCODES | {
    Opcode.ADC_IMMEDIATE, Opcode.ADC_DP, Opcode.ADC_ABSOLUTE, Opcode.ADC_STACK,
    Opcode.SBC_IMMEDIATE, Opcode.SBC_DP, Opcode.SBC_ABSOLUTE, Opcode.SBC_STACK,
    Opcode.AND_IMMEDIATE, Opcode.AND_DP, Opcode.AND_ABSOLUTE, Opcode.AND_STACK,
    Opcode.ORA_IMMEDIATE, Opcode.ORA_DP, Opcode.ORA_ABSOLUTE, Opcode.ORA_STACK,
    Opcode.EOR_IMMEDIATE, Opcode.EOR_DP, Opcode.EOR_ABSOLUTE, Opcode.EOR_STACK,
    Opcode.CMP_IMMEDIATE, Opcode.CMP_DP, Opcode.CMP_ABSOLUTE, Opcode.CMP_STACK,
    Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR,
    Opcode.TAX, Opcode.TAY, Opcode.PHA,
    Opcode.XBA,  # Exchanges A and B
}

# Instructions that modify A
MODIFIES_A_OPCODES: Set[Opcode] = LOAD_A_OPCODES | {
    Opcode.ADC_IMMEDIATE, Opcode.ADC_DP, Opcode.ADC_ABSOLUTE, Opcode.ADC_STACK,
    Opcode.ADC_DP_X, Opcode.ADC_ABSOLUTE_X, Opcode.ADC_ABSOLUTE_Y,
    Opcode.SBC_IMMEDIATE, Opcode.SBC_DP, Opcode.SBC_ABSOLUTE, Opcode.SBC_STACK,
    Opcode.SBC_DP_X, Opcode.SBC_ABSOLUTE_X, Opcode.SBC_ABSOLUTE_Y,
    Opcode.AND_IMMEDIATE, Opcode.AND_DP, Opcode.AND_ABSOLUTE, Opcode.AND_STACK,
    Opcode.AND_DP_X, Opcode.AND_ABSOLUTE_X, Opcode.AND_ABSOLUTE_Y,
    Opcode.ORA_IMMEDIATE, Opcode.ORA_DP, Opcode.ORA_ABSOLUTE, Opcode.ORA_STACK,
    Opcode.ORA_DP_X, Opcode.ORA_ABSOLUTE_X, Opcode.ORA_ABSOLUTE_Y,
    Opcode.EOR_IMMEDIATE, Opcode.EOR_DP, Opcode.EOR_ABSOLUTE, Opcode.EOR_STACK,
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
# CALL_OPCODES (JSR/JSL) are included because the callee may modify any memory location,
# so dead store analysis must stop at calls to avoid removing stores that are actually used
CONTROL_FLOW_OPCODES: Set[Opcode] = (
    BRANCH_OPCODES | JUMP_OPCODES | CALL_OPCODES | {Opcode.RTS, Opcode.RTL, Opcode.RTI}
)

# Instructions that modify the stack pointer — invalidate stack-relative operand tracking
STACK_MODIFYING_OPCODES: Set[Opcode] = {
    Opcode.PHA, Opcode.PHX, Opcode.PHY, Opcode.PHP, Opcode.PHD, Opcode.PHB,
    Opcode.PLA, Opcode.PLX, Opcode.PLY, Opcode.PLP, Opcode.PLD, Opcode.PLB,
    Opcode.TCS,
}

# LDA opcodes with deterministic addressing (value depends only on operand/SP)
TRACKABLE_LDA_OPCODES: frozenset = frozenset({
    Opcode.LDA_IMMEDIATE,
    Opcode.LDA_DP,
    Opcode.LDA_ABSOLUTE,
    Opcode.LDA_STACK,
})

# Indirect store opcodes that could alias any address
INDIRECT_STORE_OPCODES: frozenset = frozenset({
    Opcode.STA_DP_INDIRECT, Opcode.STA_DP_INDIRECT_X, Opcode.STA_DP_INDIRECT_Y,
    Opcode.STA_DP_INDIRECT_LONG, Opcode.STA_DP_INDIRECT_LONG_Y,
    Opcode.STA_STACK_INDIRECT_Y,
})


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
    redundant_and_before_sep_eliminated: int = 0
    branch_over_branch_eliminated: int = 0
    branch_to_next_eliminated: int = 0
    branch_threading_applied: int = 0
    tracked_loads_eliminated: int = 0
    identity_copies_eliminated: int = 0
    memory_inc_dec_folded: int = 0

    @property
    def total(self) -> int:
        return (
            self.redundant_loads_eliminated +
            self.dead_stores_eliminated +
            self.redundant_transfers_eliminated +
            self.redundant_stack_ops_eliminated +
            self.redundant_mode_changes_eliminated +
            self.redundant_and_before_sep_eliminated +
            self.branch_over_branch_eliminated +
            self.branch_to_next_eliminated +
            self.tracked_loads_eliminated +
            self.identity_copies_eliminated +
            self.memory_inc_dec_folded +
            self.branch_threading_applied
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

    def __init__(self, volatile_names: Set[str] = None, volatile_addresses: Set[int] = None):
        """
        Initialize peephole optimizer.

        Args:
            volatile_names: Set of variable names that are volatile (from #[hw] attributes).
                           Stores to these locations will not be eliminated.
            volatile_addresses: Set of hardware register addresses (from #[hw] attributes).
                               Stores to these addresses will not be eliminated.
        """
        self.stats = OptimizationStats()
        self.volatile_names = volatile_names or set()
        self.volatile_addresses = volatile_addresses or set()

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
            nodes = self._fold_memory_inc_dec(nodes)
            nodes = self._eliminate_redundant_load_after_store(nodes)
            nodes = self._eliminate_identity_copies(nodes)
            nodes = self._eliminate_redundant_loads_tracked(nodes)
            nodes = self._eliminate_dead_stores(nodes)
            nodes = self._eliminate_redundant_transfers(nodes)
            nodes = self._eliminate_redundant_stack_ops(nodes)
            nodes = self._eliminate_redundant_mode_changes(nodes)
            nodes = self._eliminate_redundant_and_before_sep(nodes)
            nodes = self._eliminate_branch_over_branch(nodes)
            nodes = self._thread_branches(nodes)
            nodes = self._eliminate_branch_to_next_label(nodes)
            changed = self.stats.total > prev_total

        return nodes

    def _fold_memory_inc_dec(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Fold load/add-1/store sequences into INC/DEC on memory.

        Patterns (m8 mode, DP addressing):
            LDA dp; CLC; ADC #$01; STA dp  →  INC dp  (same address)
            LDA dp; SEC; SBC #$01; STA dp  →  DEC dp  (same address)

        Also handles ABSOLUTE addressing. Skips volatile (hardware) addresses.
        Only applies in m8 mode — INC/DEC on memory is always 8-bit width.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Directive, Address, Immediate as AsmImmediate
        from r65.compiler.codegen.constants import DP_BOUNDARY

        optimized = []
        i = 0
        in_m16 = False

        while i < len(nodes):
            node = nodes[i]

            # Track accumulator mode via .ACCU directives
            if isinstance(node, Directive) and node.name == '.ACCU':
                if node.args and node.args[0] == '16':
                    in_m16 = True
                elif node.args and node.args[0] == '8':
                    in_m16 = False
                optimized.append(node)
                i += 1
                continue

            # Track mode changes via REP/SEP
            if isinstance(node, Instruction):
                if node.opcode == Opcode.REP_IMMEDIATE and isinstance(node.operand, AsmImmediate):
                    if isinstance(node.operand.value, int) and node.operand.value & 0x20:
                        in_m16 = True
                elif node.opcode == Opcode.SEP_IMMEDIATE and isinstance(node.operand, AsmImmediate):
                    if isinstance(node.operand.value, int) and node.operand.value & 0x20:
                        in_m16 = False

            # Only fold in m8 mode (INC/DEC on memory operates at current A width,
            # but the pattern we're matching — LDA/CLC/ADC #1/STA — is m8 specific)
            if in_m16 or i + 3 >= len(nodes):
                optimized.append(node)
                i += 1
                continue

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            # Match: LDA addr; CLC; ADC #$01; STA addr (increment)
            #    or: LDA addr; SEC; SBC #$01; STA addr (decrement)
            n1 = nodes[i + 1]
            n2 = nodes[i + 2]
            n3 = nodes[i + 3]

            if not (isinstance(n1, Instruction) and isinstance(n2, Instruction) and isinstance(n3, Instruction)):
                optimized.append(node)
                i += 1
                continue

            is_inc = (node.opcode in (Opcode.LDA_DP, Opcode.LDA_ABSOLUTE) and
                      n1.opcode == Opcode.CLC and
                      n2.opcode == Opcode.ADC_IMMEDIATE and
                      isinstance(n2.operand, AsmImmediate) and n2.operand.value == 1 and
                      n3.opcode in (Opcode.STA_DP, Opcode.STA_ABSOLUTE) and
                      node.operand == n3.operand)

            is_dec = (node.opcode in (Opcode.LDA_DP, Opcode.LDA_ABSOLUTE) and
                      n1.opcode == Opcode.SEC and
                      n2.opcode == Opcode.SBC_IMMEDIATE and
                      isinstance(n2.operand, AsmImmediate) and n2.operand.value == 1 and
                      n3.opcode in (Opcode.STA_DP, Opcode.STA_ABSOLUTE) and
                      node.operand == n3.operand)

            if (is_inc or is_dec) and not self._is_hardware_register(node.operand):
                # Determine DP vs ABSOLUTE opcode
                use_dp = node.opcode in (Opcode.LDA_DP,)
                if is_inc:
                    opcode = Opcode.INC_DP if use_dp else Opcode.INC_ABSOLUTE
                else:
                    opcode = Opcode.DEC_DP if use_dp else Opcode.DEC_ABSOLUTE
                optimized.append(Instruction(opcode, node.operand,
                                             comment=f"{'INC' if is_inc else 'DEC'} memory (folded)"))
                i += 4
                self.stats.memory_inc_dec_folded += 1
            else:
                optimized.append(node)
                i += 1

        return optimized

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

    def _eliminate_identity_copies(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate adjacent LDA/STA pairs where source and destination are identical.

        Patterns:
            LDA $NN; STA $NN -> (removed)  — DP identity copy
            LDA $NN,S; STA $NN,S -> (removed)  — Stack-relative identity copy
        """
        from r65.compiler.codegen.asm_nodes import Instruction

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if (isinstance(node, Instruction) and
                    node.opcode in LOAD_A_OPCODES and
                    i + 1 < len(nodes)):
                next_node = nodes[i + 1]
                if (isinstance(next_node, Instruction) and
                        next_node.opcode in STORE_A_OPCODES and
                        node.operand == next_node.operand and
                        self._same_addressing_mode(node.opcode, next_node.opcode)):
                    # Identity copy — skip both instructions
                    i += 2
                    self.stats.identity_copies_eliminated += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_dead_stores(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate dead stores that are overwritten before being read.

        Pattern: STA $XX; ... (no read of $XX); STA $XX -> ... ; STA $XX
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

            # Skip hardware registers - stores have side effects (I/O)
            # SNES hardware ranges: $2100-$213F (PPU), $4200-$437F (CPU I/O)
            if self._is_hardware_register(store_operand):
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

    def _is_hardware_register(self, operand) -> bool:
        """
        Check if operand is a hardware register (volatile).

        Hardware register writes have side effects (I/O operations) and
        must never be eliminated, even if the value is overwritten later.

        Checks volatile register names and addresses from #[hw] attributes.
        """
        from r65.compiler.codegen.asm_nodes import Address

        if operand is None:
            return False

        # Extract the actual value from Address objects
        value = operand
        if isinstance(operand, Address):
            value = operand.value

        # Check if this is a known volatile register name (from #[hw] attribute)
        if isinstance(value, str):
            val = value.strip()
            if val in self.volatile_names:
                return True
            # Also check for hex address format against known volatile addresses
            if val.startswith('$'):
                try:
                    addr = int(val[1:], 16)
                    if addr in self.volatile_addresses:
                        return True
                except ValueError:
                    pass
        elif isinstance(value, int):
            # Check if address is in known volatile addresses
            if value in self.volatile_addresses:
                return True

        return False

    def _is_dead_store(self, nodes: List['AsmNode'], store_idx: int, store_operand) -> bool:
        """
        Check if a store is dead (overwritten before read).

        For stack-relative stores ($XX,S), extends analysis past unconditional
        branches (BRA/BRL): if no instruction in the entire node list reads
        from the stored address, the store is dead. This catches temporaries
        whose only reader was eliminated by a prior pass.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, StackOffset

        is_stack_relative = isinstance(store_operand, StackOffset)

        j = store_idx + 1

        while j < len(nodes):
            next_node = nodes[j]

            # Label = potential branch target, stop analysis
            if isinstance(next_node, Label):
                return False

            if not isinstance(next_node, Instruction):
                j += 1
                continue

            # Control flow
            if next_node.opcode in CONTROL_FLOW_OPCODES:
                # For stack-relative stores, check if branch target reads value
                if is_stack_relative:
                    if next_node.opcode in (Opcode.BRA, Opcode.BRL):
                        if not self._any_instruction_reads(
                            nodes, store_operand):
                            return True
                return False

            # Stack-modifying instructions invalidate stack-relative analysis.
            # After PHA/PLX/etc., the same $XX,S offset refers to a different
            # physical address, so we can't compare operands across SP changes.
            if is_stack_relative and next_node.opcode in STACK_MODIFYING_OPCODES:
                return False

            # Mode change = stop analysis (16-bit mode can read adjacent bytes)
            # REP/SEP change how many bytes are accessed by subsequent instructions
            if next_node.opcode in (Opcode.REP_IMMEDIATE, Opcode.SEP_IMMEDIATE):
                return False

            # Indirect addressing = conservative, might read from any address
            # We can't track what address is being accessed through the pointer
            if next_node.opcode in INDIRECT_ADDRESSING_OPCODES:
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

    def _any_instruction_reads(self, nodes: List['AsmNode'],
                              store_operand) -> bool:
        """
        Check if any instruction in the node list reads from store_operand.

        Conservatively scans all nodes. This may miss optimization
        opportunities when multiple functions use the same stack offset,
        but is always safe (no false negatives).
        """
        from r65.compiler.codegen.asm_nodes import Instruction, StackOffset

        is_stack = isinstance(store_operand, StackOffset)

        for node in nodes:
            if not isinstance(node, Instruction):
                continue
            if self._reads_from_location(node, store_operand):
                return True
            # Indirect STA instructions (e.g. STA ($nn,S),Y) read the
            # pointer from their operand even though they are stores.
            # If the operand matches, the stored value IS being read.
            if (node.opcode in INDIRECT_ADDRESSING_OPCODES
                    and node.operand == store_operand):
                return True
            # Stack-modifying opcodes (PHB/PHA/PHX/PHY/PLB/PLA/PLX/PLY)
            # shift the stack pointer, so a store to $N,S may be read as
            # $(N+1),S after a push. We can't reliably match offsets across
            # SP changes, so conservatively assume the store is read.
            if is_stack and node.opcode in STACK_MODIFYING_OPCODES:
                return True

        return False

    def _reads_from_location(self, instr: 'Instruction', operand) -> bool:
        """Check if instruction reads from the given memory location.

        Also matches StackOffset(N) against Address(N) since they alias
        when D=S mode is active (far pointer functions use PHD/TSC/TCD
        to make stack offsets equivalent to direct page offsets).
        """
        if instr.opcode not in READS_FROM_MEMORY_OPCODES:
            return False
        if instr.operand == operand:
            return True
        # StackOffset(N) and Address(N) alias under D=S
        from r65.compiler.codegen.asm_nodes import StackOffset, Address
        if isinstance(operand, StackOffset) and isinstance(instr.operand, Address):
            return instr.operand.value == operand.offset
        if isinstance(operand, Address) and isinstance(instr.operand, StackOffset):
            return operand.value == instr.operand.offset
        return False

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

        Also handles directives (.ACCU) between SEP/REP pairs.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Directive

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            if node.opcode in (Opcode.SEP_IMMEDIATE, Opcode.REP_IMMEDIATE):
                # Find the next instruction, skipping over directives
                next_instr_idx = i + 1
                directives_between = []
                while next_instr_idx < len(nodes):
                    next_node = nodes[next_instr_idx]
                    if isinstance(next_node, Directive):
                        directives_between.append(next_node)
                        next_instr_idx += 1
                    elif isinstance(next_node, Instruction):
                        break
                    else:
                        # Comments, labels, etc. - stop looking
                        break

                if next_instr_idx < len(nodes):
                    next_instr = nodes[next_instr_idx]

                    if isinstance(next_instr, Instruction):
                        # Same instruction with same operand
                        if (next_instr.opcode == node.opcode and
                            node.operand == next_instr.operand):
                            # Duplicate - keep first, skip second (and directives between)
                            optimized.append(node)
                            optimized.extend(directives_between)
                            i = next_instr_idx + 1
                            self.stats.redundant_mode_changes_eliminated += 1
                            continue

                        # Opposite pair with same operand (SEP #$V → REP #$V or vice versa):
                        # The second instruction unconditionally determines the final mode,
                        # so the first is redundant. Keep only the second.
                        opposite = {Opcode.SEP_IMMEDIATE: Opcode.REP_IMMEDIATE,
                                    Opcode.REP_IMMEDIATE: Opcode.SEP_IMMEDIATE}
                        if (next_instr.opcode == opposite[node.opcode] and
                            node.operand == next_instr.operand):
                            # Skip first instruction and directives between;
                            # next iteration will naturally append the second
                            i = next_instr_idx
                            self.stats.redundant_mode_changes_eliminated += 1
                            continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_redundant_and_before_sep(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Remove AND #$00FF when followed by SEP #$20.

        In 8-bit accumulator mode, only the low byte is used, so masking
        off the high byte before switching to 8-bit is redundant.

        Pattern: AND #$00FF (or #$FF), SEP #$xx where xx & 0x20 != 0
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Directive, Immediate

        M_FLAG = 0x20
        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            # Check for AND #$00FF or AND #$FF (immediate)
            if isinstance(node, Instruction) and node.opcode == Opcode.AND_IMMEDIATE:
                if isinstance(node.operand, Immediate) and node.operand.value in (0x00FF, 0xFF):
                    # Look ahead for SEP with M flag, skipping directives
                    next_instr_idx = i + 1
                    directives_between = []

                    while next_instr_idx < len(nodes):
                        next_node = nodes[next_instr_idx]
                        if isinstance(next_node, Directive):
                            directives_between.append(next_node)
                            next_instr_idx += 1
                        elif isinstance(next_node, Instruction):
                            break
                        else:
                            break

                    if next_instr_idx < len(nodes):
                        next_instr = nodes[next_instr_idx]

                        if (isinstance(next_instr, Instruction) and
                            next_instr.opcode == Opcode.SEP_IMMEDIATE and
                            isinstance(next_instr.operand, Immediate) and
                            next_instr.operand.value & M_FLAG):
                            # Pattern matched - skip the AND, keep directives and SEP
                            optimized.extend(directives_between)
                            i = next_instr_idx  # Will emit SEP on next iteration
                            self.stats.redundant_and_before_sep_eliminated += 1
                            continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_branch_over_branch(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate conditional branch over unconditional branch when the
        conditional target is the immediately following label.

        Pattern:
            Bcc label_A       ; conditional branch
            [labels/directives/comments]
            BRA label_B       ; unconditional branch
            [labels/directives/comments]
            label_A:          ; conditional target is right here

        Becomes:
            B!cc label_B      ; inverted condition, target BRA's destination
            [labels/directives/comments preserved]
            label_A:          ; kept (may be targeted by other branches)

        Labels between Bcc and BRA are safe to skip because they contain
        no instructions — any branch targeting them reaches the same BRA.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address
        from r65.compiler.codegen.asm_nodes import invert_branch

        CONDITIONAL_BRANCHES = BRANCH_OPCODES - {Opcode.BRA, Opcode.BRL}

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if (isinstance(node, Instruction) and
                node.opcode in CONDITIONAL_BRANCHES and
                isinstance(node.operand, Address) and
                isinstance(node.operand.value, str)):

                cond_target = node.operand.value

                # Look ahead for BRA, skipping directives/comments AND labels.
                # Labels with no instructions between them and the BRA are
                # transparent — branches to them reach the same BRA destination.
                j = i + 1
                between = []
                while j < len(nodes) and not isinstance(nodes[j], Instruction):
                    between.append(nodes[j])
                    j += 1

                if (j < len(nodes) and
                    isinstance(nodes[j], Instruction) and
                    nodes[j].opcode == Opcode.BRA and
                    isinstance(nodes[j].operand, Address)):

                    bra_node = nodes[j]
                    bra_target = bra_node.operand

                    # Look ahead past the BRA for the conditional target label,
                    # skipping directives/comments and non-target labels
                    k = j + 1
                    between2 = []
                    found_target = False
                    while k < len(nodes):
                        if isinstance(nodes[k], Instruction):
                            break  # Hit an instruction before finding target
                        if isinstance(nodes[k], Label) and nodes[k].name == cond_target:
                            found_target = True
                            break
                        between2.append(nodes[k])
                        k += 1

                    if found_target:

                        inverted = invert_branch(node.opcode)
                        if inverted is not None:
                            # Emit inverted branch to BRA's target
                            optimized.append(Instruction(
                                inverted, bra_target, bra_node.comment,
                                node.source_loc))
                            # Keep labels/directives/comments that were between
                            optimized.extend(between)
                            optimized.extend(between2)
                            # Label will be appended on next iteration
                            i = k
                            self.stats.branch_over_branch_eliminated += 1
                            continue

            optimized.append(node)
            i += 1

        return optimized

    def _thread_branches(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Thread branches through intermediate unconditional branches.

        When a branch (conditional or BRA) targets a label whose first
        instruction is BRA, redirect it to the BRA's ultimate target.

        Example:
            BPL label_A       ->  BPL label_B
            ...                   ...
            label_A:              label_A:
            BRA label_B           BRA label_B  (now dead, removed by later passes)
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address

        ALL_BRANCHES = BRANCH_OPCODES

        # Build label -> BRA target map: for each label immediately followed
        # by a BRA (skipping other labels/directives), record the BRA target.
        label_to_bra_target: dict[str, str] = {}
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                # Find the first instruction after this label
                j = i + 1
                while j < len(nodes) and not isinstance(nodes[j], Instruction):
                    j += 1
                if (j < len(nodes) and
                    isinstance(nodes[j], Instruction) and
                    nodes[j].opcode == Opcode.BRA and
                    isinstance(nodes[j].operand, Address) and
                    isinstance(nodes[j].operand.value, str)):
                    label_to_bra_target[node.name] = nodes[j].operand.value

        # Follow chains (label_A -> BRA label_B -> BRA label_C)
        def resolve(label: str, depth: int = 0) -> str:
            if depth > 10:
                return label
            if label in label_to_bra_target:
                return resolve(label_to_bra_target[label], depth + 1)
            return label

        # Rewrite branch targets
        optimized = []
        for node in nodes:
            if (isinstance(node, Instruction) and
                node.opcode in ALL_BRANCHES and
                isinstance(node.operand, Address) and
                isinstance(node.operand.value, str)):

                target = node.operand.value
                resolved = resolve(target)
                if resolved != target:
                    optimized.append(Instruction(
                        node.opcode,
                        Address(resolved),
                        node.comment,
                        node.source_loc,
                    ))
                    self.stats.branch_threading_applied += 1
                    continue

            optimized.append(node)

        return optimized

    def _eliminate_branch_to_next_label(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate BRA instructions that branch to the immediately following label.

        Pattern: BRA label; label: -> label:
        Skips directives/comments between the BRA and the label.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            if (isinstance(node, Instruction) and
                node.opcode == Opcode.BRA and
                isinstance(node.operand, Address) and
                isinstance(node.operand.value, str)):

                target = node.operand.value

                # Look ahead past directives/comments/non-target labels for the target label.
                # Labels between the BRA and target contain no instructions, so
                # falling through them is equivalent to branching to the target.
                j = i + 1
                while j < len(nodes) and not isinstance(nodes[j], Instruction):
                    if isinstance(nodes[j], Label) and nodes[j].name == target:
                        break
                    j += 1

                if (j < len(nodes) and
                    isinstance(nodes[j], Label) and
                    nodes[j].name == target):
                    # BRA to next label — skip the BRA, keep directives/comments between
                    for k in range(i + 1, j):
                        optimized.append(nodes[k])
                    # Label will be appended on next iteration
                    i = j
                    self.stats.branch_to_next_eliminated += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_redundant_loads_tracked(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate redundant LDA instructions by tracking what A currently holds.

        Tracks the last LDA as (opcode, operand). When a subsequent LDA has
        the same opcode and operand, it's eliminated since A already holds
        that value.

        Only tracks deterministic addressing modes (immediate, DP, absolute,
        stack-relative) — indexed and indirect loads depend on register values
        that may change between loads.

        State is cleared on: labels, inline asm blocks, mode changes,
        A-modifying instructions (except trackable LDA which updates tracking),
        stack pointer changes when tracking a stack-relative address, stores to
        the tracked address by other registers (STX/STY), and indirect stores.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, RawAsm, StackOffset

        optimized = []
        # Track last LDA: (opcode, operand) or None
        known_a = None

        for node in nodes:
            if not isinstance(node, Instruction):
                if isinstance(node, (Label, RawAsm)):
                    # Label = unknown incoming state
                    # RawAsm = inline asm!() may modify any register
                    known_a = None
                optimized.append(node)
                continue

            opcode = node.opcode

            # Check for redundant LDA before any state updates
            if opcode in LOAD_A_OPCODES:
                if (known_a is not None and
                    known_a[0] == opcode and
                    known_a[1] == node.operand):
                    # A already holds this value — skip the LDA
                    self.stats.tracked_loads_eliminated += 1
                    continue
                # Update tracking only for deterministic addressing modes
                # Never track loads from hardware I/O registers (volatile)
                if opcode in TRACKABLE_LDA_OPCODES and not self._is_hardware_register(node.operand):
                    known_a = (opcode, node.operand)
                else:
                    # Non-deterministic load (indexed/indirect) — A changed
                    # but we can't track the value
                    known_a = None
                optimized.append(node)
                continue

            # Mode changes (REP/SEP affecting M flag) change A width
            if opcode in (Opcode.REP_IMMEDIATE, Opcode.SEP_IMMEDIATE):
                from r65.compiler.codegen.asm_nodes import Immediate
                if isinstance(node.operand, Immediate):
                    val = node.operand.value
                    if isinstance(val, int) and val & 0x20:
                        known_a = None
                optimized.append(node)
                continue

            # Instructions that modify A (but aren't LDA — those are handled above)
            if opcode in MODIFIES_A_OPCODES:
                known_a = None
                optimized.append(node)
                continue

            # Stack pointer modifications invalidate stack-relative tracking
            if opcode in STACK_MODIFYING_OPCODES:
                if known_a is not None and isinstance(known_a[1], StackOffset):
                    known_a = None
                optimized.append(node)
                continue

            # STX/STY to the tracked address invalidates tracking
            # (STA to same address is fine — writes A back)
            if opcode in (STORE_X_OPCODES | STORE_Y_OPCODES):
                if known_a is not None and node.operand == known_a[1]:
                    known_a = None
                optimized.append(node)
                continue

            # Indirect stores can alias any address
            if opcode in INDIRECT_STORE_OPCODES:
                known_a = None
                optimized.append(node)
                continue

            # Control flow (branches/jumps/calls) — clear tracking
            # Calls: callee may modify memory that A was loaded from
            # Branches/jumps: target may have different A state
            if opcode in CONTROL_FLOW_OPCODES:
                known_a = None
                optimized.append(node)
                continue

            # Stores to hw I/O registers may have side effects that change
            # the value at the tracked address (e.g., STA WRMPYB triggers
            # a multiply, changing RDMPYL/RDMPYH)
            if opcode in (STORE_A_OPCODES | STORE_X_OPCODES | STORE_Y_OPCODES):
                if known_a is not None and self._is_hardware_register(node.operand):
                    known_a = None
                optimized.append(node)
                continue

            # Other instructions (CMP, TAX, TAY, CLC, SEC, etc.)
            # don't modify A — tracking stays valid
            optimized.append(node)

        return optimized


# ============================================================================
# Public API
# ============================================================================

def optimize_nodes(nodes: List['AsmNode'], volatile_names: Set[str] = None,
                   volatile_addresses: Set[int] = None) -> Tuple[List['AsmNode'], int]:
    """
    Apply peephole optimizations to AsmNode list.

    Args:
        nodes: List of AsmNode objects
        volatile_names: Set of variable names that are volatile (from #[hw] attributes).
                       Stores to these locations will not be eliminated.
        volatile_addresses: Set of hardware register addresses (from #[hw] attributes).
                           Stores to these addresses will not be eliminated.

    Returns:
        Tuple of (optimized nodes, number of optimizations applied)
    """
    optimizer = PeepholeOptimizer(volatile_names, volatile_addresses)
    optimized = optimizer.optimize(nodes)
    return optimized, optimizer.optimizations_applied
