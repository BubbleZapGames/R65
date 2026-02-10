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
# CALL_OPCODES (JSR/JSL) are included because the callee may modify any memory location,
# so dead store analysis must stop at calls to avoid removing stores that are actually used
CONTROL_FLOW_OPCODES: Set[Opcode] = (
    BRANCH_OPCODES | JUMP_OPCODES | CALL_OPCODES | {Opcode.RTS, Opcode.RTL, Opcode.RTI}
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
    redundant_and_before_sep_eliminated: int = 0

    @property
    def total(self) -> int:
        return (
            self.redundant_loads_eliminated +
            self.dead_stores_eliminated +
            self.redundant_transfers_eliminated +
            self.redundant_stack_ops_eliminated +
            self.redundant_mode_changes_eliminated +
            self.redundant_and_before_sep_eliminated
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
            nodes = self._eliminate_redundant_load_after_store(nodes)
            nodes = self._eliminate_dead_stores(nodes)
            nodes = self._eliminate_redundant_transfers(nodes)
            nodes = self._eliminate_redundant_stack_ops(nodes)
            nodes = self._eliminate_redundant_mode_changes(nodes)
            nodes = self._eliminate_redundant_and_before_sep(nodes)
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
        """Check if a store is dead (overwritten before read)."""
        from r65.compiler.codegen.asm_nodes import Instruction, Label
        import sys

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

                        # NOTE: We intentionally do NOT remove SEP→REP or REP→SEP pairs.
                        # These do NOT cancel out: SEP #$20 sets M flag (m8) and REP #$20
                        # clears M flag (m16). If the CPU is already in m8 before SEP, the
                        # SEP is a no-op but the REP is the actual mode switch. Removing
                        # both leaves the CPU in the wrong mode, causing the assembler to
                        # generate 3-byte m16 instructions that the m8 CPU misinterprets.

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
