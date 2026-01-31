"""
Branch fixup pass: handles conditional branches that exceed 127-byte range.

The 65816's conditional branch instructions (BEQ, BNE, BCC, BCS, BMI, BPL, BVC, BVS)
have an 8-bit signed offset, limiting them to -128 to +127 bytes. This pass identifies
branches that exceed this limit and rewrites them using the inverted pattern:

    ; Original (broken if target > 127 bytes away):
    BEQ far_target

    ; Rewritten:
    BNE __skip_N       ; Inverted condition, skip over JMP
    JMP far_target     ; JMP can reach anywhere in bank
    __skip_N:

Algorithm:
1. Build branch/label index for the code
2. Calculate initial offsets assuming all branches are short (2 bytes)
3. Mark branches exceeding 127 bytes as "long" (will become 5 bytes)
4. Iterate until stable:
   - Recalculate offsets with long branches at 5 bytes
   - Check if any branches changed status (short↔long)
5. Single rewrite pass: expand all "long" branches

This approach avoids the inefficiency of recalculating after each individual fixup.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional

from r65.compiler.codegen.opcodes import (
    Opcode, mnemonic, instruction_size,
)
from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Directive,
    Address, invert_branch,
)
from r65.compiler.errors import compiler_assert


# ============================================================================
# Constants
# ============================================================================

# Maximum branch distance (signed 8-bit: -128 to +127)
# Use exact limit - the algorithm handles cascading correctly
MAX_BRANCH_DISTANCE = 127

# Size of short branch instruction (e.g., BEQ rel8)
SHORT_BRANCH_SIZE = 2

# Size of long conditional branch replacement (inverted branch + JMP + label)
# BNE skip (2) + JMP target (3) + skip: (0) = 5 bytes
LONG_CONDITIONAL_BRANCH_SIZE = 5

# Size of long unconditional branch replacement (just JMP)
# BRA → JMP = 3 bytes (was 2)
LONG_UNCONDITIONAL_BRANCH_SIZE = 3

# Conditional branches that can be inverted for long branch fixup
CONDITIONAL_BRANCH_OPCODES: Set[Opcode] = {
    Opcode.BEQ, Opcode.BNE,
    Opcode.BCC, Opcode.BCS,
    Opcode.BMI, Opcode.BPL,
    Opcode.BVC, Opcode.BVS,
}

# All branch opcodes that have limited range (need fixup if too far)
ALL_BRANCH_OPCODES: Set[Opcode] = CONDITIONAL_BRANCH_OPCODES | {Opcode.BRA}


# ============================================================================
# Branch Info
# ============================================================================

@dataclass
class BranchInfo:
    """Information about a conditional branch instruction."""
    index: int              # Index in node list
    target_label: str       # Target label name
    opcode: Opcode          # Original opcode
    is_long: bool = False   # Whether this needs long form


# ============================================================================
# Branch Fixup Pass
# ============================================================================

class BranchFixup:
    """
    Optimized branch fixup using iterative convergence.

    Analyzes all branches first, determines which need long form,
    then applies fixups in a single pass.
    """

    def __init__(self):
        self._skip_label_counter = 0
        self.branches_fixed = 0

    def fixup(self, nodes: List[AsmNode]) -> List[AsmNode]:
        """
        Apply branch fixup to AsmNode list.

        Args:
            nodes: List of AsmNode objects

        Returns:
            Fixed node list with long branches expanded
        """
        if not nodes:
            return nodes

        # Phase 1: Build indices
        branches, label_indices = self._build_indices(nodes)

        if not branches:
            return nodes  # No conditional branches to fix

        # Phase 2: Determine which branches need long form (iterative)
        self._analyze_branches(nodes, branches, label_indices)

        # Phase 3: Apply fixups
        return self._apply_fixups(nodes, branches)

    def _build_indices(
        self, nodes: List[AsmNode]
    ) -> Tuple[List[BranchInfo], Dict[str, int]]:
        """
        Build indices for branches and labels.

        Returns:
            Tuple of (branch_infos, label_to_index mapping)
        """
        branches: List[BranchInfo] = []
        label_indices: Dict[str, int] = {}

        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                label_indices[node.name] = i
            elif isinstance(node, Instruction):
                if node.opcode in ALL_BRANCH_OPCODES:
                    target = self._get_branch_target(node)
                    if target:
                        branches.append(BranchInfo(
                            index=i,
                            target_label=target,
                            opcode=node.opcode
                        ))

        return branches, label_indices

    def _get_branch_target(self, instr: Instruction) -> Optional[str]:
        """Extract target label from a branch instruction."""
        if isinstance(instr.operand, Address) and isinstance(instr.operand.value, str):
            return instr.operand.value
        return None

    def _analyze_branches(
        self,
        nodes: List[AsmNode],
        branches: List[BranchInfo],
        label_indices: Dict[str, int]
    ):
        """
        Iteratively determine which branches need long form.

        Continues until no branch changes status (convergence).
        """
        max_iterations = 20  # Safety limit

        for _ in range(max_iterations):
            # Calculate byte offsets for all nodes
            offsets = self._calculate_offsets(nodes, branches)

            # Check each branch and update is_long status
            changed = False
            for branch in branches:
                if branch.target_label not in label_indices:
                    continue  # Target not in this scope

                target_idx = label_indices[branch.target_label]

                # Calculate distance from end of branch instruction to target
                branch_end_offset = offsets[branch.index + 1] if branch.index + 1 < len(offsets) else offsets[-1]
                target_offset = offsets[target_idx]
                distance = target_offset - branch_end_offset

                needs_long = abs(distance) > MAX_BRANCH_DISTANCE

                if needs_long != branch.is_long:
                    branch.is_long = needs_long
                    changed = True

            if not changed:
                break  # Converged

    def _calculate_offsets(
        self,
        nodes: List[AsmNode],
        branches: List[BranchInfo]
    ) -> List[int]:
        """
        Calculate byte offset for each node position.

        Accounts for branch instructions that are marked as long:
        - Conditional branches: 5 bytes (inverted branch + JMP + label)
        - BRA: 3 bytes (just JMP)
        - Short branches: 2 bytes

        Returns:
            List where offsets[i] is byte offset at start of nodes[i]
        """
        # Build map of long branch indices to their info for O(1) lookup
        long_branch_map = {b.index: b for b in branches if b.is_long}

        offsets: List[int] = []
        current_offset = 0
        acc_16 = False
        idx_16 = True  # X/Y are always 16-bit in R65

        for i, node in enumerate(nodes):
            offsets.append(current_offset)

            if isinstance(node, Instruction):
                if i in long_branch_map:
                    # This branch will be expanded to long form
                    branch_info = long_branch_map[i]
                    if branch_info.opcode == Opcode.BRA:
                        # BRA → JMP is just 3 bytes
                        current_offset += LONG_UNCONDITIONAL_BRANCH_SIZE
                    else:
                        # Conditional branch → inverted + JMP + label is 5 bytes
                        current_offset += LONG_CONDITIONAL_BRANCH_SIZE
                else:
                    current_offset += instruction_size(node.opcode, acc_16, idx_16)
            elif isinstance(node, Directive):
                if node.name == '.ACCU':
                    acc_16 = '16' in ''.join(node.args)
                elif node.name == '.INDEX':
                    idx_16 = '16' in ''.join(node.args)
                # Handle data directives
                current_offset += self._directive_size(node)

        # Add final offset (end of code)
        offsets.append(current_offset)

        return offsets

    def _directive_size(self, directive: Directive) -> int:
        """Calculate size in bytes of a directive."""
        name = directive.name.upper()
        args = directive.args

        if name in ('.DB', '.BYTE'):
            return len(args)
        elif name in ('.DW', '.WORD'):
            return len(args) * 2
        elif name in ('.DL', '.LONG', '.FARADDR'):
            return len(args) * 3
        elif name in ('.DD', '.DWORD'):
            return len(args) * 4
        elif name == '.DSB':
            # .DSB count - reserve count bytes
            if args:
                try:
                    return int(args[0], 0)
                except (ValueError, IndexError):
                    pass
        elif name == '.DSW':
            # .DSW count - reserve count words
            if args:
                try:
                    return int(args[0], 0) * 2
                except (ValueError, IndexError):
                    pass

        return 0

    def _apply_fixups(
        self,
        nodes: List[AsmNode],
        branches: List[BranchInfo]
    ) -> List[AsmNode]:
        """
        Apply fixups to all branches marked as long.

        Single pass through nodes, expanding long branches.
        """
        # Build set of indices that need expansion
        long_indices = {b.index: b for b in branches if b.is_long}

        if not long_indices:
            return nodes  # Nothing to fix

        result: List[AsmNode] = []

        for i, node in enumerate(nodes):
            if i in long_indices:
                # Expand this branch
                branch_info = long_indices[i]
                compiler_assert(
                    isinstance(node, Instruction),
                    f"long_indices[{i}] references non-Instruction node: {type(node).__name__}"
                )
                expanded = self._expand_long_branch(node, branch_info)
                result.extend(expanded)
                self.branches_fixed += 1
            else:
                result.append(node)

        return result

    def _expand_long_branch(
        self,
        branch: Instruction,
        info: BranchInfo
    ) -> List[AsmNode]:
        """
        Expand a long branch to reach far targets.

        For BRA (unconditional):
            BRA far_target  →  JMP far_target

        For conditional branches (BEQ, BNE, etc.):
            BEQ far_target  →  BNE __skip_N       ; Inverted, jumps over the JMP
                               JMP far_target     ; Unconditional jump to original target
                               __skip_N:          ; Continue here if condition was false
        """
        # Handle BRA specially - just convert to JMP
        if branch.opcode == Opcode.BRA:
            return [Instruction(
                opcode=Opcode.JMP_ABSOLUTE,
                operand=branch.operand,
                comment=branch.comment
            )]

        # For conditional branches, use the inverted pattern
        # Generate unique skip label
        skip_label = f"__skip_{self._skip_label_counter}"
        self._skip_label_counter += 1

        # Get inverted branch opcode
        inverted_opcode = invert_branch(branch.opcode)
        if inverted_opcode is None:
            # Shouldn't happen for conditional branches
            return [branch]

        result: List[AsmNode] = []

        # 1. Inverted branch to skip label (skips over JMP if condition false)
        result.append(Instruction(
            opcode=inverted_opcode,
            operand=Address(skip_label),
            comment=f"long branch (was {mnemonic(branch.opcode)})"
        ))

        # 2. JMP to original target (taken if original condition was true)
        result.append(Instruction(
            opcode=Opcode.JMP_ABSOLUTE,
            operand=branch.operand
        ))

        # 3. Skip label (continue point if original condition was false)
        result.append(Label(name=skip_label))

        return result


# ============================================================================
# Public API
# ============================================================================

def fixup_nodes(nodes: List[AsmNode]) -> Tuple[List[AsmNode], int]:
    """
    Apply long branch fixup to AsmNode list.

    Uses iterative convergence to determine optimal branch sizing,
    then applies fixups in a single pass.

    Args:
        nodes: List of AsmNode objects

    Returns:
        Tuple of (fixed nodes, number of branches fixed)
    """
    fixup = BranchFixup()
    fixed_nodes = fixup.fixup(nodes)
    return fixed_nodes, fixup.branches_fixed
