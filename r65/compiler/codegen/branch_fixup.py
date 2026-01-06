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
    Opcode, mnemonic, instruction_size, is_branch,
    BRANCH_OPCODES,
)
from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Directive, Comment, BlankLine,
    Address, invert_branch,
)


# ============================================================================
# Constants
# ============================================================================

# Maximum branch distance (signed 8-bit: -128 to +127)
MAX_BRANCH_DISTANCE = 127

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

    Args:
        nodes: List of AsmNode objects

    Returns:
        Tuple of (fixed nodes, number of branches fixed)
    """
    fixup = BranchFixup()
    fixed = fixup.fixup(nodes)
    return fixed, fixup.branches_fixed


# ============================================================================
# String-Based Branch Fixup (Test Support)
# ============================================================================
# The following code provides string-based branch fixup for testing purposes.
# Production code should use fixup_nodes() with AsmNode objects.

from enum import Enum, auto
from typing import Optional


class AsmElementKind(Enum):
    """Types of assembly elements (for string-based parsing)."""
    INSTRUCTION = auto()
    LABEL = auto()
    DIRECTIVE = auto()
    COMMENT = auto()
    BLANK = auto()
    RAW = auto()


@dataclass
class AsmElement:
    """Base class for assembly elements (string-based)."""
    kind: AsmElementKind
    original_line: str
    offset: int = 0
    size: int = 0


@dataclass
class AsmInstruction(AsmElement):
    """An assembly instruction (string-based)."""
    mnemonic: str = ""
    operand: Optional[str] = None
    comment: Optional[str] = None

    def __post_init__(self):
        self.kind = AsmElementKind.INSTRUCTION


@dataclass
class AsmLabel(AsmElement):
    """A label definition (string-based)."""
    name: str = ""

    def __post_init__(self):
        self.kind = AsmElementKind.LABEL


@dataclass
class AsmDirective(AsmElement):
    """An assembler directive (string-based)."""
    directive: str = ""
    args: str = ""

    def __post_init__(self):
        self.kind = AsmElementKind.DIRECTIVE


# String-based branch mnemonics
STRING_CONDITIONAL_BRANCHES = {'BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS'}
STRING_BRANCH_INVERSION = {
    'BEQ': 'BNE', 'BNE': 'BEQ',
    'BCC': 'BCS', 'BCS': 'BCC',
    'BMI': 'BPL', 'BPL': 'BMI',
    'BVC': 'BVS', 'BVS': 'BVC',
}

# Backwards-compatible aliases for tests
CONDITIONAL_BRANCHES = STRING_CONDITIONAL_BRANCHES
BRANCH_INVERSION = STRING_BRANCH_INVERSION


class AssemblyParser:
    """Parses assembly text into structured IR (for testing)."""

    def __init__(self):
        self.acc_16bit = False
        self.idx_16bit = False

    def parse_lines(self, lines: List[str]) -> List[AsmElement]:
        """Parse assembly lines into structured elements."""
        elements = []
        for line in lines:
            element = self._parse_line(line)
            if element:
                elements.append(element)
                if isinstance(element, AsmDirective):
                    self._update_mode(element)
        return elements

    def _parse_line(self, line: str) -> Optional[AsmElement]:
        """Parse a single assembly line."""
        stripped = line.strip()

        if not stripped:
            return AsmElement(kind=AsmElementKind.BLANK, original_line=line)

        if stripped.startswith(';'):
            return AsmElement(kind=AsmElementKind.COMMENT, original_line=line)

        if stripped.endswith(':') and not stripped.startswith('.'):
            return AsmLabel(kind=AsmElementKind.LABEL, original_line=line, name=stripped[:-1])

        if stripped.startswith('.'):
            parts = stripped.split(None, 1)
            return AsmDirective(
                kind=AsmElementKind.DIRECTIVE,
                original_line=line,
                directive=parts[0],
                args=parts[1] if len(parts) > 1 else ""
            )

        return self._parse_instruction(line, stripped)

    def _parse_instruction(self, original: str, stripped: str) -> AsmInstruction:
        """Parse an instruction line."""
        comment = None
        if ';' in stripped:
            code_part, comment_part = stripped.split(';', 1)
            stripped = code_part.strip()
            comment = comment_part.strip()

        parts = stripped.split(None, 1)
        mnemonic_str = parts[0].upper()
        operand = parts[1] if len(parts) > 1 else None

        instr = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=original,
            mnemonic=mnemonic_str,
            operand=operand,
            comment=comment
        )
        instr.size = self._calculate_instruction_size(mnemonic_str, operand)
        return instr

    def _calculate_instruction_size(self, mnemonic_str: str, operand: Optional[str]) -> int:
        """Calculate instruction size in bytes."""
        if mnemonic_str in STRING_CONDITIONAL_BRANCHES or mnemonic_str == 'BRA':
            return 2
        if mnemonic_str == 'BRL':
            return 3
        if mnemonic_str == 'JMP':
            return 3
        if mnemonic_str == 'JML':
            return 4
        if mnemonic_str in ('JSR', 'JSL'):
            return 4 if mnemonic_str == 'JSL' else 3
        if mnemonic_str in ('RTS', 'RTL', 'RTI'):
            return 1
        if mnemonic_str in ('REP', 'SEP'):
            return 2
        if operand is None:
            return 1

        if operand.startswith('#'):
            base_size = 2
            if mnemonic_str in ('LDA', 'ADC', 'SBC', 'AND', 'ORA', 'EOR', 'CMP', 'BIT'):
                if self.acc_16bit:
                    base_size = 3
            elif mnemonic_str in ('LDX', 'LDY', 'CPX', 'CPY'):
                if self.idx_16bit:
                    base_size = 3
            return base_size

        if ',S' in operand:
            return 2
        if operand.startswith('f:') or '>>' in operand:
            return 4
        if ',X' in operand or ',Y' in operand:
            addr_part = operand.split(',')[0].strip()
            if addr_part.startswith('$') and len(addr_part) > 3:
                return 3
            return 2
        if operand.startswith('$'):
            addr = operand[1:].split()[0]
            return 2 if len(addr) <= 2 else 3

        return 3

    def _update_mode(self, directive: AsmDirective):
        """Update mode tracking from .ACCU/.INDEX directives."""
        if directive.directive == '.ACCU':
            self.acc_16bit = directive.args.strip() == '16'
        elif directive.directive == '.INDEX':
            self.idx_16bit = directive.args.strip() == '16'


class StringBranchFixup:
    """String-based branch fixup (for testing)."""

    def __init__(self):
        self.skip_label_counter = 0

    def fixup(self, lines: List[str]) -> Tuple[List[str], int]:
        """Apply branch fixup to assembly lines."""
        parser = AssemblyParser()
        elements = parser.parse_lines(lines)
        label_offsets = self._calculate_offsets(elements)
        fixed_elements, num_fixups = self._fix_long_branches(elements, label_offsets)
        fixed_lines = self._elements_to_lines(fixed_elements)
        return fixed_lines, num_fixups

    def _calculate_offsets(self, elements: List[AsmElement]) -> Dict[str, int]:
        """Calculate byte offsets for all elements and build label map."""
        label_offsets: Dict[str, int] = {}
        current_offset = 0

        for elem in elements:
            elem.offset = current_offset
            if isinstance(elem, AsmLabel):
                label_offsets[elem.name] = current_offset
            elif isinstance(elem, AsmInstruction):
                current_offset += elem.size

        return label_offsets

    def _fix_long_branches(self, elements: List[AsmElement],
                           label_offsets: Dict[str, int]) -> Tuple[List[AsmElement], int]:
        """Find and fix branches that exceed 127-byte range."""
        fixed: List[AsmElement] = []
        num_fixups = 0
        i = 0

        while i < len(elements):
            elem = elements[i]

            if isinstance(elem, AsmInstruction) and elem.mnemonic in STRING_CONDITIONAL_BRANCHES:
                target_label = elem.operand
                if target_label and target_label in label_offsets:
                    target_offset = label_offsets[target_label]
                    branch_end = elem.offset + elem.size
                    distance = target_offset - branch_end

                    if abs(distance) > MAX_BRANCH_DISTANCE:
                        fixed_elements = self._rewrite_branch(elem)
                        fixed.extend(fixed_elements)
                        num_fixups += 1
                        label_offsets = self._calculate_offsets(fixed + elements[i + 1:])
                        i += 1
                        continue

            fixed.append(elem)
            i += 1

        return fixed, num_fixups

    def _rewrite_branch(self, branch: AsmInstruction) -> List[AsmElement]:
        """Rewrite a long branch using the inverted pattern."""
        skip_label = f"__branch_skip_{self.skip_label_counter}"
        self.skip_label_counter += 1

        inverted_mnemonic = STRING_BRANCH_INVERSION[branch.mnemonic]
        original_target = branch.operand

        result: List[AsmElement] = []

        inverted_branch = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=f"    {inverted_mnemonic} {skip_label}",
            mnemonic=inverted_mnemonic,
            operand=skip_label,
            comment=f"Long branch fixup (was {branch.mnemonic})",
            size=2
        )
        result.append(inverted_branch)

        jmp_instr = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=f"    JMP {original_target}",
            mnemonic="JMP",
            operand=original_target,
            comment=None,
            size=3
        )
        result.append(jmp_instr)

        skip_label_elem = AsmLabel(
            kind=AsmElementKind.LABEL,
            original_line=f"{skip_label}:",
            name=skip_label
        )
        result.append(skip_label_elem)

        return result

    def _elements_to_lines(self, elements: List[AsmElement]) -> List[str]:
        """Convert elements back to assembly lines."""
        lines: List[str] = []

        for elem in elements:
            if isinstance(elem, AsmInstruction):
                line = f"    {elem.mnemonic}"
                if elem.operand:
                    line += f" {elem.operand}"
                if elem.comment:
                    padding = max(1, 32 - len(line))
                    line += " " * padding + f"; {elem.comment}"
                lines.append(line)
            elif isinstance(elem, AsmLabel):
                lines.append(f"{elem.name}:")
            else:
                lines.append(elem.original_line)

        return lines


def fixup_long_branches(assembly_lines: List[str]) -> Tuple[List[str], int]:
    """
    Apply long branch fixup to assembly source (string-based, for testing).

    Args:
        assembly_lines: List of assembly source lines

    Returns:
        Tuple of (fixed lines, number of branches fixed)
    """
    fixup = StringBranchFixup()
    return fixup.fixup(assembly_lines)
