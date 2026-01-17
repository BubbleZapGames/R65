"""
Branch fixup pass: handles conditional branches that exceed 127-byte range.

The 65816's conditional branch instructions (BEQ, BNE, BCC, BCS, BMI, BPL, BVC, BVS)
have an 8-bit signed offset, limiting them to ±127 bytes. This pass identifies
branches that exceed this limit and rewrites them using the inverted pattern:

    ; Original (broken if target > 127 bytes):
    BEQ far_target
    JMP other_target

    ; Rewritten:
    BNE __branch_skip_N    ; Inverted condition
    JMP far_target         ; JMP can reach anywhere
    __branch_skip_N:
    JMP other_target

This pass runs after peephole optimization to work with final instruction sizes.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set

from r65.compiler.codegen.opcodes import (
    Opcode, mnemonic, instruction_size,
)
from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Directive,
    Address, invert_branch,
)


# ============================================================================
# Constants
# ============================================================================

# Maximum branch distance (signed 8-bit: -128 to +127)
# Use conservative threshold to account for potential size calculation differences
# between our estimation and actual assembled output
MAX_BRANCH_DISTANCE = 100

# Conditional branches that can be inverted for long branch fixup
# BRA and BRL are unconditional and don't need fixup
CONDITIONAL_BRANCH_OPCODES: Set[Opcode] = {
    Opcode.BEQ, Opcode.BNE,
    Opcode.BCC, Opcode.BCS,
    Opcode.BMI, Opcode.BPL,
    Opcode.BVC, Opcode.BVS,
}


# ============================================================================
# Statistics Tracking
# ============================================================================

@dataclass
class BranchFixupStats:
    """Track branch fixup statistics."""
    branches_analyzed: int = 0
    branches_fixed: int = 0
    labels_created: int = 0


# ============================================================================
# Branch Fixup Pass
# ============================================================================

class BranchFixup:
    """
    Branch fixup that works directly on AsmNode objects.

    Uses typed Opcode enum for efficient pattern matching.
    Identifies conditional branches that exceed 127-byte range and rewrites
    them using the inverted branch + JMP pattern.
    """

    def __init__(self):
        self.stats = BranchFixupStats()
        self._skip_label_counter = 0
        self._acc_16bit = False
        self._idx_16bit = False

    @property
    def branches_fixed(self) -> int:
        """Number of branches that were fixed."""
        return self.stats.branches_fixed

    def fixup(self, nodes: List[AsmNode]) -> List[AsmNode]:
        """
        Apply branch fixup to AsmNode list.

        Args:
            nodes: List of AsmNode objects

        Returns:
            Fixed node list
        """
        # Calculate initial label offsets
        label_offsets = self._calculate_label_offsets(nodes)

        # Process nodes, fixing long branches
        fixed: List[AsmNode] = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            # Track mode directives
            self._track_mode_directive(node)

            # Check for conditional branch that might need fixup
            if self._is_fixable_branch(node):
                assert isinstance(node, Instruction)
                target_label = self._get_branch_target(node)

                if target_label and target_label in label_offsets:
                    self.stats.branches_analyzed += 1

                    # Calculate branch distance
                    branch_offset = self._calculate_current_offset(fixed)
                    branch_size = instruction_size(node.opcode, self._acc_16bit, self._idx_16bit)
                    target_offset = label_offsets[target_label]
                    distance = target_offset - (branch_offset + branch_size)

                    if abs(distance) > MAX_BRANCH_DISTANCE:
                        # Fix this long branch
                        fixed_nodes = self._rewrite_long_branch(node)
                        fixed.extend(fixed_nodes)
                        self.stats.branches_fixed += 1

                        # Recalculate label offsets with the new nodes
                        label_offsets = self._calculate_label_offsets(fixed + nodes[i + 1:])
                        i += 1
                        continue

            fixed.append(node)
            i += 1

        return fixed

    def _is_fixable_branch(self, node: AsmNode) -> bool:
        """Check if node is a conditional branch that can be fixed."""
        if not isinstance(node, Instruction):
            return False
        return node.opcode in CONDITIONAL_BRANCH_OPCODES

    def _get_branch_target(self, instr: Instruction) -> str | None:
        """Extract target label from a branch instruction."""
        if isinstance(instr.operand, Address) and isinstance(instr.operand.value, str):
            return instr.operand.value
        return None

    def _track_mode_directive(self, node: AsmNode):
        """Track .ACCU/.INDEX directives for instruction sizing."""
        if isinstance(node, Directive):
            if node.name == '.ACCU':
                self._acc_16bit = '16' in ''.join(node.args)
            elif node.name == '.INDEX':
                self._idx_16bit = '16' in ''.join(node.args)

    def _calculate_label_offsets(self, nodes: List[AsmNode]) -> Dict[str, int]:
        """Calculate byte offsets for all labels."""
        label_offsets: Dict[str, int] = {}
        current_offset = 0
        acc_16 = False
        idx_16 = False

        for node in nodes:
            if isinstance(node, Label):
                label_offsets[node.name] = current_offset
            elif isinstance(node, Instruction):
                current_offset += instruction_size(node.opcode, acc_16, idx_16)
            elif isinstance(node, Directive):
                if node.name == '.ACCU':
                    acc_16 = '16' in ''.join(node.args)
                elif node.name == '.INDEX':
                    idx_16 = '16' in ''.join(node.args)

        return label_offsets

    def _calculate_current_offset(self, nodes: List[AsmNode]) -> int:
        """Calculate byte offset at current position."""
        offset = 0
        acc_16 = False
        idx_16 = False

        for node in nodes:
            if isinstance(node, Instruction):
                offset += instruction_size(node.opcode, acc_16, idx_16)
            elif isinstance(node, Directive):
                if node.name == '.ACCU':
                    acc_16 = '16' in ''.join(node.args)
                elif node.name == '.INDEX':
                    idx_16 = '16' in ''.join(node.args)

        return offset

    def _rewrite_long_branch(self, branch: Instruction) -> List[AsmNode]:
        """
        Rewrite a long branch using the inverted pattern.

        Original:
            BEQ far_target

        Rewritten:
            BNE __branch_skip_N
            JMP far_target
            __branch_skip_N:
        """
        # Generate unique skip label
        skip_label = f"__branch_skip_{self._skip_label_counter}"
        self._skip_label_counter += 1
        self.stats.labels_created += 1

        # Get inverted branch opcode
        inverted_opcode = invert_branch(branch.opcode)
        if inverted_opcode is None:
            # Can't invert - shouldn't happen for conditional branches
            return [branch]

        original_target = branch.operand

        result: List[AsmNode] = []

        # 1. Inverted branch to skip label
        inverted_branch = Instruction(
            opcode=inverted_opcode,
            operand=Address(skip_label),
            comment=f"Long branch fixup (was {mnemonic(branch.opcode)})"
        )
        result.append(inverted_branch)

        # 2. JMP to original target
        jmp_instr = Instruction(
            opcode=Opcode.JMP_ABSOLUTE,
            operand=original_target
        )
        result.append(jmp_instr)

        # 3. Skip label
        skip_label_node = Label(name=skip_label)
        result.append(skip_label_node)

        return result


# ============================================================================
# Public API
# ============================================================================

def fixup_nodes(nodes: List[AsmNode]) -> Tuple[List[AsmNode], int]:
    """
    Apply long branch fixup to AsmNode list.

    This function should be called after peephole optimization and before
    final assembly output.

    Runs multiple passes because fixing one branch can push other branches
    over the 127-byte limit.

    Args:
        nodes: List of AsmNode objects

    Returns:
        Tuple of (fixed nodes, number of branches fixed)
    """
    total_fixed = 0
    max_iterations = 10  # Prevent infinite loop

    for _ in range(max_iterations):
        fixup = BranchFixup()
        nodes = fixup.fixup(nodes)

        if fixup.branches_fixed == 0:
            break  # No more branches need fixing

        total_fixed += fixup.branches_fixed

    return nodes, total_fixed
