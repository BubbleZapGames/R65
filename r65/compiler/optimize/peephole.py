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
    loops_rotated: int = 0
    loop_invariant_loads_hoisted: int = 0
    count_down_loops: int = 0
    unreachable_nodes_eliminated: int = 0
    stz_conversions: int = 0
    inc_dec_folded: int = 0
    redundant_cmp_zero_eliminated: int = 0

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
            self.branch_threading_applied +
            self.loops_rotated +
            self.loop_invariant_loads_hoisted +
            self.count_down_loops +
            self.unreachable_nodes_eliminated +
            self.stz_conversions +
            self.inc_dec_folded +
            self.redundant_cmp_zero_eliminated
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
            nodes = self._eliminate_cross_block_mode_changes(nodes)
            nodes = self._eliminate_redundant_and_before_sep(nodes)
            nodes = self._eliminate_branch_over_branch(nodes)
            nodes = self._thread_branches(nodes)
            nodes = self._eliminate_branch_to_next_label(nodes)
            nodes = self._inline_branch_to_return(nodes)
            nodes = self._rotate_top_tested_loops(nodes)
            nodes = self._hoist_loop_invariant_loads(nodes)
            nodes = self._count_down_loops(nodes)
            nodes = self._hoist_loop_mode_switches(nodes)
            nodes = self._eliminate_unreachable_code(nodes)
            nodes = self._convert_zero_stores_to_stz(nodes)
            nodes = self._fold_inc_dec_accumulator(nodes)
            nodes = self._eliminate_redundant_cmp_zero(nodes)
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
        Eliminate redundant processor mode changes using mode state tracking.

        Tracks the current m-flag state (8-bit or 16-bit accumulator) and
        removes REP/SEP instructions that don't change the current mode.
        Also removes .ACCU directives that follow eliminated mode switches.

        Handles non-adjacent redundancies: if we know the mode is already m16,
        a REP #$20 several instructions later is redundant even with intervening
        instructions (as long as no branch target or mode change occurs between).
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Directive, Label

        optimized = []
        i = 0
        # None = unknown, 8 = m8, 16 = m16
        current_mode = None

        while i < len(nodes):
            node = nodes[i]

            # Labels invalidate mode tracking (branch target could arrive in any mode)
            if isinstance(node, Label):
                current_mode = None
                optimized.append(node)
                i += 1
                continue

            # Track mode from .ACCU directives
            if isinstance(node, Directive) and node.name == '.ACCU':
                if node.args:
                    if node.args[0] == '8':
                        current_mode = 8
                    elif node.args[0] == '16':
                        current_mode = 16
                optimized.append(node)
                i += 1
                continue

            if not isinstance(node, Instruction):
                optimized.append(node)
                i += 1
                continue

            # Branch/jump invalidates mode knowledge for what follows
            if node.opcode in BRANCH_OPCODES or node.opcode in JUMP_OPCODES:
                optimized.append(node)
                current_mode = None
                i += 1
                continue

            # Check REP/SEP for redundancy
            if node.opcode == Opcode.REP_IMMEDIATE and isinstance(node.operand, Immediate):
                if isinstance(node.operand.value, int) and node.operand.value & 0x20:
                    if current_mode == 16:
                        # Already in m16 — skip REP and trailing .ACCU 16
                        self.stats.redundant_mode_changes_eliminated += 1
                        i += 1
                        if (i < len(nodes) and isinstance(nodes[i], Directive)
                                and nodes[i].name == '.ACCU' and nodes[i].args
                                and nodes[i].args[0] == '16'):
                            i += 1
                        continue
                    current_mode = 16
                    optimized.append(node)
                    i += 1
                    continue

            if node.opcode == Opcode.SEP_IMMEDIATE and isinstance(node.operand, Immediate):
                if isinstance(node.operand.value, int) and node.operand.value & 0x20:
                    if current_mode == 8:
                        # Already in m8 — skip SEP and trailing .ACCU 8
                        self.stats.redundant_mode_changes_eliminated += 1
                        i += 1
                        if (i < len(nodes) and isinstance(nodes[i], Directive)
                                and nodes[i].name == '.ACCU' and nodes[i].args
                                and nodes[i].args[0] == '8'):
                            i += 1
                        continue
                    current_mode = 8
                    optimized.append(node)
                    i += 1
                    continue

            # PLP/RTI restore unknown mode
            if node.opcode in (Opcode.PLP, Opcode.RTI):
                current_mode = None

            optimized.append(node)
            i += 1

        return optimized

    def _eliminate_cross_block_mode_changes(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate SEP/REP at label targets when all predecessors arrive in the target mode.

        The linear mode tracker resets at labels (conservative). This pass collects
        the mode at each branch source and fallthrough, then removes SEP/REP that
        are provably redundant because all predecessors agree on the mode.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Directive, Label

        M_FLAG = 0x20

        # Pass 1: Collect mode at each branch source and fallthrough to each label.
        # label_arriving_modes[label_name] = set of modes (8, 16, or None for unknown)
        label_arriving_modes: dict[str, set] = {}
        current_mode = None
        dead_fallthrough = False  # True after terminal instructions (RTS/RTL/RTI/BRA/JMP)

        from r65.compiler.codegen.opcodes import RETURN_OPCODES
        TERMINAL_OPCODES = RETURN_OPCODES | JUMP_OPCODES | frozenset({Opcode.BRA})

        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                # Only record fallthrough mode if control can actually fall through
                if node.name not in label_arriving_modes:
                    label_arriving_modes[node.name] = set()
                if not dead_fallthrough:
                    label_arriving_modes[node.name].add(current_mode)
                dead_fallthrough = False
                # Don't reset current_mode here — we still know the fallthrough mode
                continue

            if isinstance(node, Directive) and node.name == '.ACCU':
                if node.args:
                    if node.args[0] == '8':
                        current_mode = 8
                    elif node.args[0] == '16':
                        current_mode = 16
                continue

            if not isinstance(node, Instruction):
                continue

            # Track mode changes
            if node.opcode == Opcode.REP_IMMEDIATE and isinstance(node.operand, Immediate):
                if isinstance(node.operand.value, int) and node.operand.value & M_FLAG:
                    current_mode = 16
            elif node.opcode == Opcode.SEP_IMMEDIATE and isinstance(node.operand, Immediate):
                if isinstance(node.operand.value, int) and node.operand.value & M_FLAG:
                    current_mode = 8
            elif node.opcode in (Opcode.PLP, Opcode.RTI):
                current_mode = None

            # Branch: record mode at branch target
            if node.opcode in BRANCH_OPCODES:
                target = node.operand.value if hasattr(node.operand, 'value') else None
                if target:
                    if target not in label_arriving_modes:
                        label_arriving_modes[target] = set()
                    label_arriving_modes[target].add(current_mode)
                # After unconditional branch, no fallthrough
                if node.opcode == Opcode.BRA:
                    current_mode = None
                    dead_fallthrough = True
            elif node.opcode in JUMP_OPCODES:
                current_mode = None
                dead_fallthrough = True

            # Return instructions: no fallthrough to next label
            if node.opcode in RETURN_OPCODES:
                dead_fallthrough = True

        # Pass 2: Remove SEP/REP at labels where all predecessors agree on mode
        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            # Look for Label followed by SEP/REP (possibly with .ACCU between)
            if isinstance(node, Label):
                modes = label_arriving_modes.get(node.name, set())
                # Check if next non-directive instruction is SEP or REP
                j = i + 1
                accu_idx = None
                sep_rep_idx = None
                while j < len(nodes):
                    if isinstance(nodes[j], Directive) and nodes[j].name == '.ACCU':
                        accu_idx = j
                        j += 1
                        continue
                    if isinstance(nodes[j], Instruction):
                        if (nodes[j].opcode == Opcode.SEP_IMMEDIATE and
                                isinstance(nodes[j].operand, Immediate) and
                                isinstance(nodes[j].operand.value, int) and
                                nodes[j].operand.value & M_FLAG):
                            sep_rep_idx = j
                        elif (nodes[j].opcode == Opcode.REP_IMMEDIATE and
                                isinstance(nodes[j].operand, Immediate) and
                                isinstance(nodes[j].operand.value, int) and
                                nodes[j].operand.value & M_FLAG):
                            sep_rep_idx = j
                    break
                    # Labels or other nodes: stop looking
                    break

                if sep_rep_idx is not None:
                    target_mode = 8 if nodes[sep_rep_idx].opcode == Opcode.SEP_IMMEDIATE else 16
                    # All predecessors must arrive in target_mode (no None/unknown).
                    # Function entry labels (after RTS/RTL) have no fallthrough
                    # predecessors recorded, so labels with empty modes or all-matching
                    # modes can be optimized.
                    # Labels with NO predecessors at all (empty set) get an .ACCU
                    # directive — trust it as the declared entry mode.
                    accu_declares_mode = False
                    if accu_idx is not None:
                        accu_node = nodes[accu_idx]
                        accu_val = int(accu_node.args[0]) if accu_node.args else None
                        accu_declares_mode = (accu_val == target_mode)
                    all_match = modes and all(m == target_mode for m in modes)
                    no_preds_but_declared = (not modes and accu_declares_mode)
                    if all_match or no_preds_but_declared:
                        # Eliminate the SEP/REP but KEEP the .ACCU directive.
                        # WLA-DX tracks accumulator size linearly (not by control
                        # flow), so the .ACCU directive is needed to inform the
                        # assembler of the correct mode for subsequent instructions
                        # even when the runtime SEP/REP is provably redundant.
                        optimized.append(node)  # Keep the label
                        # Keep the .ACCU directive before SEP/REP
                        if accu_idx is not None:
                            optimized.append(nodes[accu_idx])
                        i = sep_rep_idx + 1
                        # Keep trailing .ACCU directive after SEP/REP
                        if (i < len(nodes) and isinstance(nodes[i], Directive) and
                                nodes[i].name == '.ACCU'):
                            optimized.append(nodes[i])
                            i += 1
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

    def _rotate_top_tested_loops(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Rotate top-tested loops to bottom-tested when safe.

        Pattern (top-tested, count-up with X/Y):
            [LDX/LDY #0 or small const]
            HEADER:
                CPX/CPY #N          ; N > init value
                BCS EXIT            ; exit when >= N
            [BODY_LABEL:]
                ... body ...        ; no branches to HEADER except back-edge
                INX/INY/DEX/DEY
                BRA HEADER
            EXIT:

        Transforms to (bottom-tested):
            [LDX/LDY #init]
            BODY_LABEL:
                ... body ...
                INX/INY/DEX/DEY
                CPX/CPY #N
                BCC BODY_LABEL      ; continue while < N
            EXIT:

        Only applies when:
        - Loop init < bound (guaranteed to execute at least once)
        - No other branches target HEADER within the loop
        - The compare+branch pattern uses BCS (unsigned >=) for count-up
        """
        from r65.compiler.codegen.asm_nodes import (
            Instruction, Label, Address, Immediate as AsmImmediate, Directive, Comment
        )

        CMP_OPCODES = {Opcode.CPX_IMMEDIATE, Opcode.CPY_IMMEDIATE}
        INC_OPCODES = {Opcode.INX, Opcode.INY, Opcode.DEX, Opcode.DEY}
        INIT_OPCODES = {Opcode.LDX_IMMEDIATE, Opcode.LDY_IMMEDIATE}

        # Build label reference counts to check if header has other refs
        label_refs: dict[str, int] = {}
        for node in nodes:
            if (isinstance(node, Instruction) and
                isinstance(getattr(node, 'operand', None), Address) and
                isinstance(node.operand.value, str)):
                label_refs[node.operand.value] = label_refs.get(node.operand.value, 0) + 1

        optimized = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            # Look for pattern: LDX/LDY #init; HEADER: CPX/CPY #N; BCS EXIT
            if (isinstance(node, Instruction) and node.opcode in INIT_OPCODES and
                    isinstance(node.operand, AsmImmediate) and
                    isinstance(node.operand.value, int)):
                init_val = node.operand.value
                init_reg = 'X' if node.opcode == Opcode.LDX_IMMEDIATE else 'Y'

                # Next should be header label
                j = i + 1
                if (j < len(nodes) and isinstance(nodes[j], Label)):
                    header_label = nodes[j].name

                    # Header should only be referenced by the back-edge BRA
                    if label_refs.get(header_label, 0) != 1:
                        optimized.append(node)
                        i += 1
                        continue

                    # Next instructions: optional directives/mode switches, then CPX/CPY, then BCS
                    k = j + 1
                    pre_cmp_nodes = []
                    while k < len(nodes) and isinstance(nodes[k], (Directive, Comment)):
                        pre_cmp_nodes.append(nodes[k])
                        k += 1
                    # Also skip SEP/REP mode switches before the compare
                    # (these will be included in the rotated loop body)
                    if (k < len(nodes) and isinstance(nodes[k], Instruction) and
                            nodes[k].opcode in (Opcode.SEP_IMMEDIATE, Opcode.REP_IMMEDIATE)):
                        pre_cmp_nodes.append(nodes[k])
                        k += 1
                        # Skip trailing .ACCU directive
                        while k < len(nodes) and isinstance(nodes[k], (Directive, Comment)):
                            pre_cmp_nodes.append(nodes[k])
                            k += 1

                    if (k < len(nodes) and isinstance(nodes[k], Instruction) and
                            nodes[k].opcode in CMP_OPCODES and
                            isinstance(nodes[k].operand, AsmImmediate) and
                            isinstance(nodes[k].operand.value, int)):
                        cmp_instr = nodes[k]
                        bound_val = cmp_instr.operand.value
                        cmp_reg = 'X' if cmp_instr.opcode == Opcode.CPX_IMMEDIATE else 'Y'

                        if cmp_reg != init_reg:
                            optimized.append(node)
                            i += 1
                            continue

                        # Must be BCS (unsigned >=) for count-up
                        k2 = k + 1
                        if (k2 < len(nodes) and isinstance(nodes[k2], Instruction) and
                                nodes[k2].opcode == Opcode.BCS and
                                isinstance(nodes[k2].operand, Address) and
                                isinstance(nodes[k2].operand.value, str)):
                            exit_label_name = nodes[k2].operand.value
                            branch_instr = nodes[k2]

                            # Check loop executes at least once
                            if init_val >= bound_val:
                                optimized.append(node)
                                i += 1
                                continue

                            # Scan body: find BRA HEADER at end, collect body nodes
                            body_start = k2 + 1
                            # Skip optional body label
                            body_label_node = None
                            bs = body_start
                            if bs < len(nodes) and isinstance(nodes[bs], Label):
                                body_label_node = nodes[bs]
                                bs += 1

                            # Collect body up to BRA HEADER
                            body_nodes = []
                            found_bra = False
                            bra_idx = bs
                            while bra_idx < len(nodes):
                                n = nodes[bra_idx]
                                if (isinstance(n, Instruction) and
                                        n.opcode == Opcode.BRA and
                                        isinstance(n.operand, Address) and
                                        n.operand.value == header_label):
                                    found_bra = True
                                    break
                                # If we hit the exit label, stop
                                if isinstance(n, Label) and n.name == exit_label_name:
                                    break
                                body_nodes.append(n)
                                bra_idx += 1

                            if not found_bra or not body_nodes:
                                optimized.append(node)
                                i += 1
                                continue

                            # Verify last instruction of body is INX/INY
                            last_body_instr = None
                            for bn in reversed(body_nodes):
                                if isinstance(bn, Instruction):
                                    last_body_instr = bn
                                    break
                            if last_body_instr is None or last_body_instr.opcode not in INC_OPCODES:
                                optimized.append(node)
                                i += 1
                                continue

                            # Transform: emit init, body label, pre-cmp nodes, body, compare, BCC body
                            optimized.append(node)  # LDX/LDY #init
                            target_label = body_label_node.name if body_label_node else header_label
                            if body_label_node:
                                optimized.append(body_label_node)
                            else:
                                # Reuse header label as body target
                                optimized.append(nodes[j])  # header label
                            optimized.extend(pre_cmp_nodes)
                            optimized.extend(body_nodes)
                            optimized.append(cmp_instr)  # CPX/CPY #N
                            optimized.append(Instruction(
                                Opcode.BCC,
                                Address(target_label),
                                "Continue loop",
                                branch_instr.source_loc,
                            ))
                            # Skip to after BRA (exit label will be picked up next)
                            i = bra_idx + 1
                            self.stats.loops_rotated += 1
                            continue

            optimized.append(node)
            i += 1

        return optimized

    def _hoist_loop_invariant_loads(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Hoist loop-invariant LDA #imm out of bottom-tested loops.

        Pattern: a bottom-tested loop (body ending with BCC/BCS/BNE/BEQ to
        body label) where LDA #imm appears in the body and A is not modified
        by any other instruction in the loop. The LDA is moved before the
        loop header label.

        Only applies when:
        - Exactly one LDA in the loop body (the invariant load)
        - No other instruction in the loop modifies A
        - The LDA is LDA_IMMEDIATE
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Address

        # Find bottom-tested loops: label -> body -> BCC/BCS/BNE/BEQ label
        # Build label position map
        label_positions: dict[str, int] = {}
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                label_positions[node.name] = i

        # Find back-edges (conditional branches that jump backward to a label)
        loops = []  # (label_idx, branch_idx, label_name)
        for i, node in enumerate(nodes):
            if (isinstance(node, Instruction) and
                    node.opcode in BRANCH_OPCODES and
                    node.opcode != Opcode.BRA and
                    isinstance(getattr(node, 'operand', None), Address) and
                    isinstance(node.operand.value, str)):
                target = node.operand.value
                if target in label_positions and label_positions[target] < i:
                    loops.append((label_positions[target], i, target))

        if not loops:
            return nodes

        # Process each loop
        hoists = {}  # label_idx -> (lda_instruction, lda_body_offset)
        for label_idx, branch_idx, label_name in loops:
            # Collect body instructions (between label and branch, inclusive of branch)
            body_instrs = []
            lda_positions = []
            other_a_mods = False

            for j in range(label_idx + 1, branch_idx + 1):
                n = nodes[j]
                if not isinstance(n, Instruction):
                    continue
                body_instrs.append((j, n))
                if n.opcode == Opcode.LDA_IMMEDIATE:
                    lda_positions.append(j)
                elif n.opcode in MODIFIES_A_OPCODES:
                    other_a_mods = True

            # Only hoist if exactly one LDA #imm and nothing else modifies A
            if len(lda_positions) == 1 and not other_a_mods:
                hoists[lda_positions[0]] = label_idx

        if not hoists:
            return nodes

        # Rebuild node list with hoisted LDAs
        optimized = []
        for i, node in enumerate(nodes):
            if i in hoists:
                # Skip — already hoisted before the label
                self.stats.loop_invariant_loads_hoisted += 1
                continue
            optimized.append(node)
            # If this is a label that has a hoist, insert the LDA after it...
            # Actually, we need to insert BEFORE the label.
            # Let me rethink: collect all hoists per label_idx, insert before label
            pass

        # Redo: build insert-before map
        optimized = []
        insert_before: dict[int, list] = {}  # label_idx -> [instructions to insert]
        for lda_idx, label_idx in hoists.items():
            if label_idx not in insert_before:
                insert_before[label_idx] = []
            insert_before[label_idx].append(nodes[lda_idx])

        for i, node in enumerate(nodes):
            if i in insert_before:
                optimized.extend(insert_before[i])
            if i in hoists:
                self.stats.loop_invariant_loads_hoisted += 1
                continue
            optimized.append(node)

        return optimized

    def _count_down_loops(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Transform count-up loops to count-down when the counter is unused in body.

        Pattern (bottom-tested, count-up):
            LDX/LDY #0
            LABEL:
                ... body (no reads of X/Y) ...
                INX/INY
                CPX/CPY #N
                BCC LABEL

        Transforms to (count-down):
            LDX/LDY #N
            LABEL:
                ... body ...
                DEX/DEY
                BNE LABEL

        Saves 2 bytes and 2 cycles per iteration (CPX/CPY eliminated).
        Only applies when the counter register is not read in the body.
        """
        from r65.compiler.codegen.asm_nodes import (
            Instruction, Label, Address, Immediate as AsmImmediate, Directive, Comment
        )

        # Build sets of opcodes that read X or Y register value
        # (indexed addressing, transfers, stores, compares, push)
        def _reads_register(opcode: Opcode, reg: str) -> bool:
            name = opcode.name
            if reg == 'X':
                return (name.endswith('_X') or name.endswith('_DP_X') or
                        '_ABSOLUTE_X' in name or '_LONG_X' in name or
                        '_INDIRECT_X' in name or
                        name in ('TXA', 'TXY', 'STX_DP', 'STX_ABSOLUTE',
                                 'STX_DP_Y', 'PHX', 'INX', 'DEX',
                                 'CPX_IMMEDIATE', 'CPX_DP', 'CPX_ABSOLUTE'))
            else:  # Y
                return (name.endswith('_Y') or name.endswith('_DP_Y') or
                        '_ABSOLUTE_Y' in name or '_INDIRECT_Y' in name or
                        '_INDIRECT_LONG_Y' in name or '_STACK_INDIRECT_Y' in name or
                        name in ('TYA', 'TYX', 'STY_DP', 'STY_ABSOLUTE',
                                 'STY_DP_X', 'PHY', 'INY', 'DEY',
                                 'CPY_IMMEDIATE', 'CPY_DP', 'CPY_ABSOLUTE'))

        INC_TO_DEC = {
            Opcode.INX: Opcode.DEX, Opcode.INY: Opcode.DEY,
        }
        INC_REG = {
            Opcode.INX: 'X', Opcode.INY: 'Y',
        }
        CMP_OPCODES = {
            'X': Opcode.CPX_IMMEDIATE, 'Y': Opcode.CPY_IMMEDIATE,
        }
        INIT_OPCODES = {
            'X': Opcode.LDX_IMMEDIATE, 'Y': Opcode.LDY_IMMEDIATE,
        }

        # Build label position map
        label_positions: dict[str, int] = {}
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                label_positions[node.name] = i

        optimized = []
        skip_until = -1
        # Track which init instructions to replace (idx -> new_value)
        init_replacements: dict[int, int] = {}

        # Find bottom-tested loops: scan for BCC that targets a label before it
        # with trailing INX/CPX or INY/CPY pattern
        i = 0
        while i < len(nodes):
            node = nodes[i]

            if isinstance(node, Instruction) and node.opcode == Opcode.BCC:
                if (isinstance(node.operand, Address) and
                        isinstance(node.operand.value, str)):
                    target = node.operand.value
                    if target in label_positions:
                        label_idx = label_positions[target]
                        if label_idx < i:
                            # Found a bottom-tested loop: label_idx .. i
                            # Check for INX/CPX/BCC or INY/CPY/BCC at end
                            # BCC is at i, CPX/CPY should be at i-1, INX/INY at i-2
                            # (skipping directives)
                            cmp_idx = i - 1
                            while cmp_idx > label_idx and isinstance(nodes[cmp_idx], (Directive, Comment)):
                                cmp_idx -= 1
                            inc_idx = cmp_idx - 1
                            while inc_idx > label_idx and isinstance(nodes[inc_idx], (Directive, Comment)):
                                inc_idx -= 1

                            if (inc_idx > label_idx and
                                    isinstance(nodes[inc_idx], Instruction) and
                                    isinstance(nodes[cmp_idx], Instruction)):
                                inc_instr = nodes[inc_idx]
                                cmp_instr = nodes[cmp_idx]

                                if (inc_instr.opcode in INC_TO_DEC and
                                        cmp_instr.opcode == CMP_OPCODES.get(INC_REG.get(inc_instr.opcode)) and
                                        isinstance(cmp_instr.operand, AsmImmediate) and
                                        isinstance(cmp_instr.operand.value, int)):

                                    reg = INC_REG[inc_instr.opcode]
                                    bound = cmp_instr.operand.value

                                    # Check that counter register is not read in body
                                    # (between label and inc_idx, exclusive)
                                    body_uses_counter = False
                                    for j in range(label_idx + 1, inc_idx):
                                        n = nodes[j]
                                        if isinstance(n, Instruction):
                                            if _reads_register(n.opcode, reg):
                                                body_uses_counter = True
                                                break

                                    if not body_uses_counter and bound > 0:
                                        # Find the init instruction (LDX/LDY #0 before label)
                                        # May be separated by hoisted instructions (e.g. LDA #imm)
                                        init_opcode = INIT_OPCODES[reg]
                                        init_idx = None
                                        search_depth = 0
                                        for j in range(label_idx - 1, -1, -1):
                                            n = nodes[j]
                                            if isinstance(n, Instruction):
                                                if (n.opcode == init_opcode and
                                                        isinstance(n.operand, AsmImmediate) and
                                                        n.operand.value == 0):
                                                    init_idx = j
                                                    break
                                                search_depth += 1
                                                if search_depth > 3:
                                                    break
                                            elif isinstance(n, Label):
                                                break

                                        if init_idx is not None:
                                            # Replace init value with bound
                                            init_replacements[init_idx] = bound
                                            # Replace INX->DEX, remove CPX, replace BCC->BNE
                                            # We'll rebuild from inc_idx
                                            # Output everything up to inc_idx as-is
                                            # Then: DEX/DEY, BNE label
                                            # Skip cmp_instr and original BCC

                                            # Emit everything from current optimized position to inc_idx
                                            # (handled by the normal append below)
                                            # Actually, we need to handle this inline
                                            # Mark the nodes to transform
                                            # For simplicity, rebuild by replacing nodes in-place
                                            pass  # Will handle below with node replacement

            optimized.append(node)
            i += 1

        if not init_replacements:
            return nodes

        # Second pass: apply transformations
        optimized = []
        # Rebuild loop structures
        # For each identified count-down loop, track (init_idx, inc_idx, cmp_idx, bcc_idx, bound, reg)
        # Re-scan to collect full info
        transforms = []
        for i, node in enumerate(nodes):
            if isinstance(node, Instruction) and node.opcode == Opcode.BCC:
                if (isinstance(node.operand, Address) and
                        isinstance(node.operand.value, str)):
                    target = node.operand.value
                    if target in label_positions:
                        label_idx = label_positions[target]
                        if label_idx < i:
                            cmp_idx = i - 1
                            while cmp_idx > label_idx and isinstance(nodes[cmp_idx], (Directive, Comment)):
                                cmp_idx -= 1
                            inc_idx = cmp_idx - 1
                            while inc_idx > label_idx and isinstance(nodes[inc_idx], (Directive, Comment)):
                                inc_idx -= 1

                            if (inc_idx > label_idx and
                                    isinstance(nodes[inc_idx], Instruction) and
                                    isinstance(nodes[cmp_idx], Instruction)):
                                inc_instr = nodes[inc_idx]
                                cmp_instr = nodes[cmp_idx]

                                if (inc_instr.opcode in INC_TO_DEC and
                                        cmp_instr.opcode == CMP_OPCODES.get(INC_REG.get(inc_instr.opcode))):
                                    reg = INC_REG[inc_instr.opcode]
                                    init_opcode = INIT_OPCODES[reg]
                                    # Find init (may be separated by hoisted instrs)
                                    search_depth_2 = 0
                                    for j in range(label_idx - 1, -1, -1):
                                        n = nodes[j]
                                        if isinstance(n, Instruction):
                                            if j in init_replacements:
                                                transforms.append({
                                                    'init_idx': j,
                                                    'inc_idx': inc_idx,
                                                    'cmp_idx': cmp_idx,
                                                    'bcc_idx': i,
                                                    'bound': init_replacements[j],
                                                    'reg': reg,
                                                    'target': target,
                                                })
                                                break
                                            search_depth_2 += 1
                                            if search_depth_2 > 3:
                                                break
                                        elif isinstance(n, Label):
                                            break

        if not transforms:
            return nodes

        # Build sets of indices to skip/replace
        skip_indices = set()
        replace_map = {}  # idx -> replacement instruction

        for t in transforms:
            # Replace init LDX/LDY #0 with LDX/LDY #bound
            replace_map[t['init_idx']] = Instruction(
                nodes[t['init_idx']].opcode,
                AsmImmediate(t['bound']),
                nodes[t['init_idx']].comment,
                nodes[t['init_idx']].source_loc,
            )
            # Replace INX/INY with DEX/DEY
            replace_map[t['inc_idx']] = Instruction(
                INC_TO_DEC[nodes[t['inc_idx']].opcode],
                None,
                nodes[t['inc_idx']].comment,
                nodes[t['inc_idx']].source_loc,
            )
            # Skip CPX/CPY
            skip_indices.add(t['cmp_idx'])
            # Replace BCC with BNE
            replace_map[t['bcc_idx']] = Instruction(
                Opcode.BNE,
                nodes[t['bcc_idx']].operand,
                nodes[t['bcc_idx']].comment,
                nodes[t['bcc_idx']].source_loc,
            )
            self.stats.count_down_loops += 1

        for i, node in enumerate(nodes):
            if i in skip_indices:
                continue
            if i in replace_map:
                optimized.append(replace_map[i])
            else:
                optimized.append(node)

        return optimized

    def _hoist_loop_mode_switches(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Hoist SEP/REP from inside bottom-tested loops to before the loop header.

        Pattern: Label / SEP|REP / body / BCC|BNE Label
        When the back-edge arrives in the target mode already (body doesn't change
        it back), hoisting the SEP/REP before the label makes it execute once
        instead of every iteration. The cross-block mode pass then eliminates
        the now-redundant in-loop copy.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Directive, Label, Address

        M_FLAG = 0x20
        BACK_EDGE_OPCODES = {Opcode.BCC, Opcode.BNE, Opcode.BRA, Opcode.BCS, Opcode.BEQ,
                             Opcode.BPL, Opcode.BMI}

        # Build label reference counts
        label_refs: dict[str, int] = {}
        for node in nodes:
            if isinstance(node, Instruction) and node.opcode in (BRANCH_OPCODES | JUMP_OPCODES):
                if hasattr(node.operand, 'value') and isinstance(node.operand.value, str):
                    label_refs[node.operand.value] = label_refs.get(node.operand.value, 0) + 1

        # Find candidates: Label followed by SEP/REP (possibly with .ACCU between)
        hoists = []  # (label_idx, sep_idx, accu_idx_or_none, target_mode)

        for i, node in enumerate(nodes):
            if not isinstance(node, Label):
                continue

            label_name = node.name
            # Only handle labels with exactly 1 reference (the back-edge)
            if label_refs.get(label_name, 0) != 1:
                continue

            # Look for SEP/REP after label (skip .ACCU directives)
            j = i + 1
            accu_before = None
            sep_idx = None
            while j < len(nodes):
                if isinstance(nodes[j], Directive) and nodes[j].name == '.ACCU':
                    accu_before = j
                    j += 1
                    continue
                if isinstance(nodes[j], Instruction):
                    if (nodes[j].opcode == Opcode.SEP_IMMEDIATE and
                            isinstance(nodes[j].operand, Immediate) and
                            isinstance(nodes[j].operand.value, int) and
                            nodes[j].operand.value & M_FLAG):
                        sep_idx = j
                    elif (nodes[j].opcode == Opcode.REP_IMMEDIATE and
                            isinstance(nodes[j].operand, Immediate) and
                            isinstance(nodes[j].operand.value, int) and
                            nodes[j].operand.value & M_FLAG):
                        sep_idx = j
                break

            if sep_idx is None:
                continue

            target_mode = 8 if nodes[sep_idx].opcode == Opcode.SEP_IMMEDIATE else 16

            # Find the .ACCU directive AFTER the SEP/REP
            accu_after = None
            if (sep_idx + 1 < len(nodes) and isinstance(nodes[sep_idx + 1], Directive) and
                    nodes[sep_idx + 1].name == '.ACCU'):
                accu_after = sep_idx + 1

            # Find the back-edge branch that targets this label
            # Track mode through the loop body to verify it arrives in target_mode
            body_start = (accu_after + 1) if accu_after else sep_idx + 1
            current_mode = target_mode  # After the SEP/REP, we're in target_mode
            found_back_edge = False

            for k in range(body_start, len(nodes)):
                n = nodes[k]
                if isinstance(n, Label):
                    # Reached another label — might be fall-through from our loop
                    continue
                if isinstance(n, Directive) and n.name == '.ACCU':
                    if n.args:
                        if n.args[0] == '8':
                            current_mode = 8
                        elif n.args[0] == '16':
                            current_mode = 16
                    continue
                if not isinstance(n, Instruction):
                    continue

                # Track mode
                if n.opcode == Opcode.REP_IMMEDIATE and isinstance(n.operand, Immediate):
                    if isinstance(n.operand.value, int) and n.operand.value & M_FLAG:
                        current_mode = 16
                elif n.opcode == Opcode.SEP_IMMEDIATE and isinstance(n.operand, Immediate):
                    if isinstance(n.operand.value, int) and n.operand.value & M_FLAG:
                        current_mode = 8
                elif n.opcode in (Opcode.PLP, Opcode.RTI):
                    current_mode = None
                    break

                # Check for back-edge
                if (n.opcode in BACK_EDGE_OPCODES and
                        hasattr(n.operand, 'value') and n.operand.value == label_name):
                    if current_mode == target_mode:
                        found_back_edge = True
                    break

                # If we hit an unconditional branch/jump to elsewhere, stop
                if n.opcode == Opcode.BRA or n.opcode in JUMP_OPCODES:
                    break

            if found_back_edge:
                hoists.append((i, sep_idx, accu_after, target_mode))

        if not hoists:
            return nodes

        # Apply hoists: insert SEP/REP before label, remove from inside loop
        hoist_set = {h[1] for h in hoists}  # SEP/REP indices to remove
        accu_set = {h[2] for h in hoists if h[2] is not None}  # .ACCU indices to remove
        insert_before = {}  # label_idx -> (sep_node, accu_node_or_none)
        for label_idx, sep_idx, accu_idx, target_mode in hoists:
            insert_before[label_idx] = (nodes[sep_idx], nodes[accu_idx] if accu_idx else None)

        optimized = []
        for i, node in enumerate(nodes):
            if i in insert_before:
                sep_node, accu_node = insert_before[i]
                optimized.append(sep_node)
                if accu_node:
                    optimized.append(accu_node)
                self.stats.redundant_mode_changes_eliminated += 1
            if i in hoist_set or i in accu_set:
                continue
            optimized.append(node)

        return optimized

    def _eliminate_unreachable_code(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate unreachable code after unconditional control flow (RTS, RTI, RTL, BRA, JMP).

        After an unconditional transfer, any instructions before the next referenced
        label are dead code. Removes instructions, directives, and unreferenced labels.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label, Directive

        TERMINAL_OPCODES = ({Opcode.RTS, Opcode.RTI, Opcode.RTL, Opcode.BRA} |
                            JUMP_OPCODES)

        # Build set of referenced labels
        referenced = set()
        for node in nodes:
            if isinstance(node, Instruction):
                if hasattr(node.operand, 'value') and isinstance(node.operand.value, str):
                    referenced.add(node.operand.value)

        optimized = []
        skipping = False
        eliminated = 0

        for node in nodes:
            if skipping:
                if isinstance(node, Label):
                    # Stop skipping at any label — it may be referenced by
                    # expressions (e.g., LDA #>(__SCMP1 - 1)) not caught by
                    # simple operand scanning
                    skipping = False
                    optimized.append(node)
                    continue
                # Only skip Instructions — preserve Directives and Comments
                if isinstance(node, Instruction):
                    eliminated += 1
                    continue
                # Non-instruction node (Directive/Comment) — stop skipping
                skipping = False

            optimized.append(node)

            if isinstance(node, Instruction) and node.opcode in TERMINAL_OPCODES:
                skipping = True

        self.stats.unreachable_nodes_eliminated += eliminated
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

    def _inline_branch_to_return(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Replace BRA to a return instruction with the return instruction itself.

        Pattern: BRA label / ... / label: RTS → RTS (inline the return)
        Saves 1 cycle and allows subsequent dead code elimination of the
        now-unreferenced label.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Label

        RETURN_OPCODES = {Opcode.RTS, Opcode.RTI, Opcode.RTL}

        # Build label -> return opcode map
        label_to_return: dict[str, Opcode] = {}
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                # Find first instruction after label (skip directives/labels)
                j = i + 1
                while j < len(nodes) and not isinstance(nodes[j], Instruction):
                    j += 1
                if (j < len(nodes) and isinstance(nodes[j], Instruction) and
                        nodes[j].opcode in RETURN_OPCODES):
                    label_to_return[node.name] = nodes[j].opcode

        if not label_to_return:
            return nodes

        optimized = []
        for node in nodes:
            if (isinstance(node, Instruction) and
                    node.opcode == Opcode.BRA and
                    hasattr(node.operand, 'value') and
                    isinstance(node.operand.value, str) and
                    node.operand.value in label_to_return):
                # Replace BRA with the return instruction
                ret_opcode = label_to_return[node.operand.value]
                optimized.append(Instruction(ret_opcode, None, node.comment, node.source_loc))
                self.stats.branch_to_next_eliminated += 1
                continue
            optimized.append(node)

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
            # a multiply, changing RDMPYL/RDMPYH). But immediate loads are
            # unaffected — the constant value doesn't depend on memory.
            if opcode in (STORE_A_OPCODES | STORE_X_OPCODES | STORE_Y_OPCODES):
                if (known_a is not None and
                        known_a[0] not in (Opcode.LDA_IMMEDIATE,) and
                        self._is_hardware_register(node.operand)):
                    known_a = None
                optimized.append(node)
                continue

            # Other instructions (CMP, TAX, TAY, CLC, SEC, etc.)
            # don't modify A — tracking stays valid
            optimized.append(node)

        return optimized

    # ========================================================================
    # INC/DEC Accumulator Folding
    # ========================================================================

    # Instructions that use the carry flag as input
    _CARRY_INPUT_OPCODES = frozenset({
        Opcode.ADC_IMMEDIATE, Opcode.ADC_DP, Opcode.ADC_ABSOLUTE, Opcode.ADC_STACK,
        Opcode.ADC_DP_X, Opcode.ADC_ABSOLUTE_X, Opcode.ADC_ABSOLUTE_Y,
        Opcode.ADC_DP_INDIRECT, Opcode.ADC_DP_INDIRECT_X, Opcode.ADC_DP_INDIRECT_Y,
        Opcode.ADC_DP_INDIRECT_LONG, Opcode.ADC_DP_INDIRECT_LONG_Y, Opcode.ADC_STACK_INDIRECT_Y,
        Opcode.SBC_IMMEDIATE, Opcode.SBC_DP, Opcode.SBC_ABSOLUTE, Opcode.SBC_STACK,
        Opcode.SBC_DP_X, Opcode.SBC_ABSOLUTE_X, Opcode.SBC_ABSOLUTE_Y,
        Opcode.SBC_DP_INDIRECT, Opcode.SBC_DP_INDIRECT_X, Opcode.SBC_DP_INDIRECT_Y,
        Opcode.SBC_DP_INDIRECT_LONG, Opcode.SBC_DP_INDIRECT_LONG_Y, Opcode.SBC_STACK_INDIRECT_Y,
        Opcode.ROL, Opcode.ROR,
        Opcode.ROL_DP, Opcode.ROR_DP,
        Opcode.ROL_ABSOLUTE, Opcode.ROR_ABSOLUTE,
        Opcode.BCC, Opcode.BCS,
    })

    # Instructions that set the carry flag (overwriting previous carry)
    _CARRY_SETTING_OPCODES = frozenset({
        Opcode.CLC, Opcode.SEC,
        Opcode.CMP_IMMEDIATE, Opcode.CMP_DP, Opcode.CMP_ABSOLUTE, Opcode.CMP_STACK,
        Opcode.CMP_DP_X, Opcode.CMP_ABSOLUTE_X,
        Opcode.CPX_IMMEDIATE, Opcode.CPX_DP, Opcode.CPX_ABSOLUTE,
        Opcode.CPY_IMMEDIATE, Opcode.CPY_DP, Opcode.CPY_ABSOLUTE,
        Opcode.ASL, Opcode.LSR, Opcode.ASL_DP, Opcode.LSR_DP,
        Opcode.ASL_ABSOLUTE, Opcode.LSR_ABSOLUTE,
        Opcode.PLP,  # Restores all flags from stack
    }) | _CARRY_INPUT_OPCODES  # ADC/SBC/ROL/ROR also set carry

    def _carry_dead_after(self, nodes: List['AsmNode'], start: int) -> bool:
        """Check if the carry flag is dead (not used) after position start."""
        from r65.compiler.codegen.asm_nodes import Instruction, Label
        from r65.compiler.codegen.opcodes import RETURN_OPCODES
        for j in range(start, min(start + 30, len(nodes))):
            n = nodes[j]
            if isinstance(n, Label):
                return False  # Conservative: unknown predecessors could use carry
            if not isinstance(n, Instruction):
                continue
            if n.opcode in self._CARRY_INPUT_OPCODES:
                return False  # Carry is used
            if n.opcode in self._CARRY_SETTING_OPCODES:
                return True  # Carry is overwritten before use
            if n.opcode in RETURN_OPCODES or n.opcode in JUMP_OPCODES or n.opcode == Opcode.BRA:
                return True  # Control flow exit, carry not used on this path
        return False  # Conservative

    def _fold_inc_dec_accumulator(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Fold CLC/ADC #$01 → INC A and SEC/SBC #$01 → DEC A.

        INC/DEC are 1 instruction (2 cycles) vs CLC/ADC or SEC/SBC (4 cycles).
        Only safe when carry flag is not used after (INC/DEC don't affect carry).
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate

        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            if isinstance(node, Instruction):
                # CLC / ADC #$01 → INC A
                if (node.opcode == Opcode.CLC and
                        i + 1 < len(nodes) and
                        isinstance(nodes[i + 1], Instruction) and
                        nodes[i + 1].opcode == Opcode.ADC_IMMEDIATE and
                        isinstance(nodes[i + 1].operand, Immediate) and
                        nodes[i + 1].operand.value == 1):
                    if self._carry_dead_after(nodes, i + 2):
                        optimized.append(Instruction(Opcode.INC, comment="INC A (folded CLC/ADC #1)"))
                        self.stats.inc_dec_folded += 1
                        i += 2
                        continue

                # SEC / SBC #$01 → DEC A
                if (node.opcode == Opcode.SEC and
                        i + 1 < len(nodes) and
                        isinstance(nodes[i + 1], Instruction) and
                        nodes[i + 1].opcode == Opcode.SBC_IMMEDIATE and
                        isinstance(nodes[i + 1].operand, Immediate) and
                        nodes[i + 1].operand.value == 1):
                    if self._carry_dead_after(nodes, i + 2):
                        optimized.append(Instruction(Opcode.DEC, comment="DEC A (folded SEC/SBC #1)"))
                        self.stats.inc_dec_folded += 1
                        i += 2
                        continue

            optimized.append(node)
            i += 1

        return optimized

    # ========================================================================
    # Redundant CMP #$00 Elimination
    # ========================================================================

    # Instructions that set the Z and N flags based on their result
    _Z_N_SETTING_OPCODES = (
        LOAD_A_OPCODES | LOAD_X_OPCODES | LOAD_Y_OPCODES | {
        Opcode.AND_IMMEDIATE, Opcode.AND_DP, Opcode.AND_ABSOLUTE, Opcode.AND_STACK,
        Opcode.AND_DP_X, Opcode.AND_ABSOLUTE_X, Opcode.AND_ABSOLUTE_Y,
        Opcode.ORA_IMMEDIATE, Opcode.ORA_DP, Opcode.ORA_ABSOLUTE, Opcode.ORA_STACK,
        Opcode.ORA_DP_X, Opcode.ORA_ABSOLUTE_X, Opcode.ORA_ABSOLUTE_Y,
        Opcode.EOR_IMMEDIATE, Opcode.EOR_DP, Opcode.EOR_ABSOLUTE, Opcode.EOR_STACK,
        Opcode.EOR_DP_X, Opcode.EOR_ABSOLUTE_X, Opcode.EOR_ABSOLUTE_Y,
        Opcode.ADC_IMMEDIATE, Opcode.ADC_DP, Opcode.ADC_ABSOLUTE, Opcode.ADC_STACK,
        Opcode.SBC_IMMEDIATE, Opcode.SBC_DP, Opcode.SBC_ABSOLUTE, Opcode.SBC_STACK,
        Opcode.ASL, Opcode.LSR, Opcode.ROL, Opcode.ROR,
        Opcode.INC, Opcode.DEC,
        Opcode.INX, Opcode.DEX, Opcode.INY, Opcode.DEY,
        Opcode.TXA, Opcode.TYA, Opcode.PLA, Opcode.TAX, Opcode.TAY,
        Opcode.XBA,
        Opcode.BIT_IMMEDIATE, Opcode.BIT_DP, Opcode.BIT_ABSOLUTE,
    })

    # Branches that only check Z or N flags (not carry or overflow)
    _Z_N_BRANCH_OPCODES = frozenset({
        Opcode.BEQ, Opcode.BNE, Opcode.BMI, Opcode.BPL,
    })

    def _eliminate_redundant_cmp_zero(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Eliminate CMP #$00 when the previous instruction already sets Z/N flags
        and the next branch only checks Z or N (not carry or overflow).

        CMP #$00 is redundant after LDA, AND, ORA, EOR, etc. when the only
        consumers of the flags are BEQ/BNE/BMI/BPL.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Label, Directive

        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            if (isinstance(node, Instruction) and
                    node.opcode == Opcode.CMP_IMMEDIATE and
                    isinstance(node.operand, Immediate) and
                    node.operand.value == 0):
                # Check: previous real instruction sets Z/N
                prev_sets_zn = False
                for j in range(len(optimized) - 1, -1, -1):
                    prev = optimized[j]
                    if isinstance(prev, (Directive,)):
                        continue
                    if isinstance(prev, Label):
                        break
                    if isinstance(prev, Instruction):
                        prev_sets_zn = prev.opcode in self._Z_N_SETTING_OPCODES
                        break
                    break

                # Check: next real instruction only uses Z/N flags
                next_only_zn = False
                if prev_sets_zn:
                    for k in range(i + 1, len(nodes)):
                        n = nodes[k]
                        if isinstance(n, Directive):
                            continue
                        if isinstance(n, Instruction):
                            next_only_zn = n.opcode in self._Z_N_BRANCH_OPCODES
                        break

                if prev_sets_zn and next_only_zn:
                    self.stats.redundant_cmp_zero_eliminated += 1
                    i += 1
                    continue

            optimized.append(node)
            i += 1

        return optimized

    # ========================================================================
    # STZ Conversion
    # ========================================================================

    # Mapping from STA addressing modes to equivalent STZ opcodes
    _STA_TO_STZ = {
        Opcode.STA_DP: Opcode.STZ_DP,
        Opcode.STA_DP_X: Opcode.STZ_DP_X,
        Opcode.STA_ABSOLUTE: Opcode.STZ_ABSOLUTE,
        Opcode.STA_ABSOLUTE_X: Opcode.STZ_ABSOLUTE_X,
    }

    def _convert_zero_stores_to_stz(self, nodes: List['AsmNode']) -> List['AsmNode']:
        """
        Convert LDA #$00 / STA addr sequences to STZ addr.

        When A is loaded with zero solely for storing to memory, STZ is more
        efficient: it stores zero directly without occupying A (1 fewer instr).

        Converts all consecutive STZ-compatible STAs after LDA #$00 to STZ.
        The LDA #$00 is removed if A is overwritten before next use, or kept
        if subsequent code depends on A=0.
        """
        from r65.compiler.codegen.asm_nodes import Instruction, Immediate, Label, Directive

        optimized = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            # Match LDA #$00 (immediate zero)
            if (isinstance(node, Instruction) and
                    node.opcode == Opcode.LDA_IMMEDIATE and
                    isinstance(node.operand, Immediate) and
                    node.operand.value == 0):

                # Scan forward: convert all consecutive STZ-compatible STAs
                j = i + 1
                stz_indices = []
                while j < len(nodes):
                    n = nodes[j]
                    if isinstance(n, (Directive, Label)):
                        break
                    if not isinstance(n, Instruction):
                        j += 1
                        continue
                    if n.opcode in self._STA_TO_STZ:
                        stz_indices.append(j)
                        j += 1
                        continue
                    break  # Non-STA instruction

                if not stz_indices:
                    optimized.append(node)
                    i += 1
                    continue

                # Check if A=0 is needed after the last converted STA.
                # Skip over mode switches (SEP/REP) and directives since
                # they don't use A's value.
                a_needed = True
                k = stz_indices[-1] + 1
                while k < len(nodes):
                    n = nodes[k]
                    if isinstance(n, Label):
                        break
                    if isinstance(n, Directive):
                        k += 1
                        continue
                    if isinstance(n, Instruction):
                        if n.opcode in MODIFIES_A_OPCODES:
                            a_needed = False  # A is overwritten
                            break
                        # SEP/REP don't use A's value, skip them
                        if n.opcode in (Opcode.SEP_IMMEDIATE, Opcode.REP_IMMEDIATE):
                            k += 1
                            continue
                        break  # Other instruction uses A (or might)
                    k += 1

                # If A=0 is potentially needed, keep the LDA #$00
                if a_needed:
                    optimized.append(node)

                # Convert STAs to STZs
                for idx in stz_indices:
                    sta_node = nodes[idx]
                    stz_opcode = self._STA_TO_STZ[sta_node.opcode]
                    stz_node = Instruction(stz_opcode, sta_node.operand, sta_node.comment)
                    optimized.append(stz_node)
                    self.stats.stz_conversions += 1

                # Resume after the last converted STA
                i = stz_indices[-1] + 1
                continue

            optimized.append(node)
            i += 1

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
