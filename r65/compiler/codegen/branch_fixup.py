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

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum, auto


# =============================================================================
# Instruction Size Calculation
# =============================================================================

# Instruction sizes indexed by opcode byte
# Size may vary based on accumulator/index register width (m16/x16 modes)
INSTRUCTION_SIZES = [
    2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # x0
    2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3, 4,  # x1
    3, 2, 4, 2, 2, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # x2
    2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3, 4,  # x3
    1, 2, 2, 2, 3, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # x4
    2, 2, 2, 2, 3, 2, 2, 2, 1, 3, 1, 1, 4, 3, 3, 4,  # x5
    1, 2, 3, 2, 2, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # x6
    2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3, 4,  # x7
    2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # x8
    2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3, 4,  # x9
    2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # xA
    2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3, 4,  # xB
    2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # xC
    2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3, 4,  # xD
    2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 3, 3, 3, 4,  # xE
    2, 2, 2, 2, 3, 2, 2, 2, 1, 3, 1, 1, 3, 3, 3, 4,  # xF
]

# Opcodes that have variable size based on accumulator width (m16 mode)
ACC_VARIABLE_OPCODES = {0x09, 0x69, 0x29, 0x89, 0xC9, 0x49, 0xE9, 0xA9}

# Opcodes that have variable size based on index width (x16 mode)
INDEX_VARIABLE_OPCODES = {0xE0, 0xC0, 0xA2, 0xA0}

# Mnemonic to opcode mapping for common instructions
# We only need this for instructions we emit and need to size
MNEMONIC_TO_OPCODE = {
    # Branches (all 2 bytes: opcode + relative offset)
    'BEQ': 0xF0, 'BNE': 0xD0, 'BCC': 0x90, 'BCS': 0xB0,
    'BMI': 0x30, 'BPL': 0x10, 'BVC': 0x50, 'BVS': 0x70,
    'BRA': 0x80,
    # Jumps
    'JMP': 0x4C,  # Absolute (3 bytes)
    'JML': 0x5C,  # Long (4 bytes)
    'JSR': 0x20,  # Absolute (3 bytes)
    'JSL': 0x22,  # Long (4 bytes)
    # Returns
    'RTS': 0x60, 'RTL': 0x6B, 'RTI': 0x40,
    # Loads (immediate forms - size varies with mode)
    'LDA': 0xA9,  # Immediate
    'LDX': 0xA2,  # Immediate
    'LDY': 0xA0,  # Immediate
    # Stores (zero-page forms)
    'STA': 0x85,  # Zero-page (2 bytes)
    'STX': 0x86,  # Zero-page (2 bytes)
    'STY': 0x84,  # Zero-page (2 bytes)
    # Implied (1 byte)
    'NOP': 0xEA, 'SEC': 0x38, 'CLC': 0x18, 'SEI': 0x78, 'CLI': 0x58,
    'SED': 0xF8, 'CLD': 0xD8, 'PHP': 0x08, 'PLP': 0x28,
    'PHA': 0x48, 'PLA': 0x68, 'PHX': 0xDA, 'PLX': 0xFA,
    'PHY': 0x5A, 'PLY': 0x7A, 'PHB': 0x8B, 'PLB': 0xAB,
    'PHD': 0x0B, 'PLD': 0x2B, 'PHK': 0x4B,
    'TAX': 0xAA, 'TXA': 0x8A, 'TAY': 0xA8, 'TYA': 0x98,
    'TSX': 0xBA, 'TXS': 0x9A, 'TXY': 0x9B, 'TYX': 0xBB,
    'TCD': 0x5B, 'TDC': 0x7B, 'TCS': 0x1B, 'TSC': 0x3B,
    'INX': 0xE8, 'DEX': 0xCA, 'INY': 0xC8, 'DEY': 0x88,
    'INC': 0x1A, 'DEC': 0x3A,  # Accumulator forms
    'ASL': 0x0A, 'LSR': 0x4A, 'ROL': 0x2A, 'ROR': 0x6A,  # Accumulator
    'XBA': 0xEB, 'WAI': 0xCB, 'STP': 0xDB,
    'REP': 0xC2, 'SEP': 0xE2,  # 2 bytes (opcode + immediate)
    # Compare
    'CMP': 0xC9, 'CPX': 0xE0, 'CPY': 0xC0,  # Immediate forms
    # Arithmetic
    'ADC': 0x69, 'SBC': 0xE9,  # Immediate forms
    'AND': 0x29, 'ORA': 0x09, 'EOR': 0x49,  # Immediate forms
    'BIT': 0x89,  # Immediate
}

# Branch instructions that need fixup
CONDITIONAL_BRANCHES = {'BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS'}

# Branch inversion table
BRANCH_INVERSION = {
    'BEQ': 'BNE', 'BNE': 'BEQ',
    'BCC': 'BCS', 'BCS': 'BCC',
    'BMI': 'BPL', 'BPL': 'BMI',
    'BVC': 'BVS', 'BVS': 'BVC',
}

# Maximum branch distance (signed 8-bit: -128 to +127)
MAX_BRANCH_DISTANCE = 127


# =============================================================================
# Intermediate Representation
# =============================================================================

class AsmElementKind(Enum):
    """Types of assembly elements."""
    INSTRUCTION = auto()
    LABEL = auto()
    DIRECTIVE = auto()
    COMMENT = auto()
    BLANK = auto()
    RAW = auto()  # Raw text that doesn't fit other categories


@dataclass
class AsmElement:
    """Base class for assembly elements."""
    kind: AsmElementKind
    original_line: str
    offset: int = 0  # Byte offset from start of function
    size: int = 0    # Size in bytes (0 for non-code elements)


@dataclass
class AsmInstruction(AsmElement):
    """An assembly instruction."""
    mnemonic: str = ""
    operand: Optional[str] = None
    comment: Optional[str] = None

    def __post_init__(self):
        self.kind = AsmElementKind.INSTRUCTION


@dataclass
class AsmLabel(AsmElement):
    """A label definition."""
    name: str = ""

    def __post_init__(self):
        self.kind = AsmElementKind.LABEL


@dataclass
class AsmDirective(AsmElement):
    """An assembler directive."""
    directive: str = ""
    args: str = ""

    def __post_init__(self):
        self.kind = AsmElementKind.DIRECTIVE


# =============================================================================
# Assembly Parser
# =============================================================================

class AssemblyParser:
    """Parses assembly text into structured IR."""

    def __init__(self):
        self.acc_16bit = False  # Track accumulator width
        self.idx_16bit = False  # Track index register width

    def parse_lines(self, lines: List[str]) -> List[AsmElement]:
        """Parse assembly lines into structured elements."""
        elements = []

        for line in lines:
            element = self._parse_line(line)
            if element:
                elements.append(element)
                # Track mode directives
                if isinstance(element, AsmDirective):
                    self._update_mode(element)

        return elements

    def _parse_line(self, line: str) -> Optional[AsmElement]:
        """Parse a single assembly line."""
        stripped = line.strip()

        # Blank line
        if not stripped:
            return AsmElement(
                kind=AsmElementKind.BLANK,
                original_line=line
            )

        # Comment line
        if stripped.startswith(';'):
            return AsmElement(
                kind=AsmElementKind.COMMENT,
                original_line=line
            )

        # Label (ends with :)
        if stripped.endswith(':') and not stripped.startswith('.'):
            return AsmLabel(
                kind=AsmElementKind.LABEL,
                original_line=line,
                name=stripped[:-1]
            )

        # Directive (starts with .)
        if stripped.startswith('.'):
            parts = stripped.split(None, 1)
            directive = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            return AsmDirective(
                kind=AsmElementKind.DIRECTIVE,
                original_line=line,
                directive=directive,
                args=args
            )

        # Instruction
        return self._parse_instruction(line, stripped)

    def _parse_instruction(self, original: str, stripped: str) -> AsmInstruction:
        """Parse an instruction line."""
        # Remove inline comment
        comment = None
        if ';' in stripped:
            code_part, comment_part = stripped.split(';', 1)
            stripped = code_part.strip()
            comment = comment_part.strip()

        # Split mnemonic and operand
        parts = stripped.split(None, 1)
        mnemonic = parts[0].upper()
        operand = parts[1] if len(parts) > 1 else None

        instr = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=original,
            mnemonic=mnemonic,
            operand=operand,
            comment=comment
        )

        # Calculate size
        instr.size = self._calculate_instruction_size(mnemonic, operand)

        return instr

    def _calculate_instruction_size(self, mnemonic: str, operand: Optional[str]) -> int:
        """Calculate instruction size in bytes."""
        # Handle known simple cases first
        if mnemonic in CONDITIONAL_BRANCHES or mnemonic == 'BRA':
            return 2  # All relative branches are 2 bytes

        if mnemonic == 'JMP':
            # Check for indirect modes
            if operand and '(' in operand:
                if operand.startswith('(') and operand.endswith(')'):
                    return 3  # JMP (addr) - indirect
                elif ',X)' in operand:
                    return 3  # JMP (addr,X) - indexed indirect
            return 3  # JMP addr - absolute

        if mnemonic == 'JML':
            return 4  # Always long

        if mnemonic in ('JSR', 'JSL'):
            return 4 if mnemonic == 'JSL' else 3

        if mnemonic in ('RTS', 'RTL', 'RTI'):
            return 1

        if mnemonic in ('REP', 'SEP'):
            return 2

        if mnemonic == 'BRL':
            return 3  # Branch long relative

        # For other instructions, use operand to determine addressing mode
        if operand is None:
            return 1  # Implied/accumulator

        # Immediate mode
        if operand.startswith('#'):
            base_size = 2
            # Check for 16-bit immediate
            if mnemonic in ('LDA', 'ADC', 'SBC', 'AND', 'ORA', 'EOR', 'CMP', 'BIT'):
                if self.acc_16bit:
                    base_size = 3
            elif mnemonic in ('LDX', 'LDY', 'CPX', 'CPY'):
                if self.idx_16bit:
                    base_size = 3
            return base_size

        # Stack relative
        if ',S' in operand:
            if '),Y' in operand:
                return 2  # (d,S),Y
            return 2  # d,S

        # Long addressing (24-bit)
        if operand.startswith('f:') or '>>' in operand:
            return 4

        # Absolute long indexed
        if ',X' in operand or ',Y' in operand:
            # Check if it looks like a 16-bit address
            addr_part = operand.split(',')[0].strip()
            if addr_part.startswith('$') and len(addr_part) > 3:
                return 3  # Absolute indexed
            return 2  # Zero-page indexed

        # Direct/Zero-page vs Absolute
        if operand.startswith('$'):
            addr = operand[1:].split()[0]  # Remove any trailing comment reference
            if len(addr) <= 2:
                return 2  # Zero-page
            else:
                return 3  # Absolute

        # Symbol reference - assume absolute (3 bytes)
        return 3

    def _update_mode(self, directive: AsmDirective):
        """Update mode tracking from .ACCU/.INDEX directives."""
        if directive.directive == '.ACCU':
            self.acc_16bit = directive.args.strip() == '16'
        elif directive.directive == '.INDEX':
            self.idx_16bit = directive.args.strip() == '16'


# =============================================================================
# Branch Fixup Pass
# =============================================================================

class BranchFixup:
    """
    Identifies and fixes conditional branches that exceed 127-byte range.
    """

    def __init__(self):
        self.skip_label_counter = 0

    def fixup(self, lines: List[str]) -> Tuple[List[str], int]:
        """
        Apply branch fixup to assembly lines.

        Args:
            lines: Assembly source lines

        Returns:
            Tuple of (fixed lines, number of fixups applied)
        """
        # Parse into IR
        parser = AssemblyParser()
        elements = parser.parse_lines(lines)

        # Calculate offsets and find labels
        label_offsets = self._calculate_offsets(elements)

        # Find and fix long branches
        fixed_elements, num_fixups = self._fix_long_branches(elements, label_offsets)

        # Convert back to lines
        fixed_lines = self._elements_to_lines(fixed_elements)

        return fixed_lines, num_fixups

    def _calculate_offsets(self, elements: List[AsmElement]) -> Dict[str, int]:
        """Calculate byte offsets for all elements and build label map."""
        label_offsets = {}
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
        fixed = []
        num_fixups = 0

        i = 0
        while i < len(elements):
            elem = elements[i]

            if isinstance(elem, AsmInstruction) and elem.mnemonic in CONDITIONAL_BRANCHES:
                # Check if this branch needs fixup
                target_label = elem.operand
                if target_label and target_label in label_offsets:
                    target_offset = label_offsets[target_label]
                    # Branch offset is from the instruction AFTER the branch
                    branch_end = elem.offset + elem.size
                    distance = target_offset - branch_end

                    if abs(distance) > MAX_BRANCH_DISTANCE:
                        # Need to fix this branch
                        fixed_elements = self._rewrite_branch(elem, elements, i)
                        fixed.extend(fixed_elements)
                        num_fixups += 1

                        # Recalculate offsets after fixup
                        # The fixup adds bytes, shifting everything after
                        label_offsets = self._calculate_offsets(fixed + elements[i+1:])

                        i += 1
                        continue

            fixed.append(elem)
            i += 1

        return fixed, num_fixups

    def _rewrite_branch(self, branch: AsmInstruction, elements: List[AsmElement],
                        index: int) -> List[AsmElement]:
        """
        Rewrite a long branch using the inverted pattern.

        Original:
            BEQ far_target
            JMP other_target  (or next instruction)

        Rewritten:
            BNE __branch_skip_N
            JMP far_target
            __branch_skip_N:
            JMP other_target  (or next instruction)
        """
        # Generate unique skip label
        skip_label = f"__branch_skip_{self.skip_label_counter}"
        self.skip_label_counter += 1

        # Get inverted branch
        inverted_mnemonic = BRANCH_INVERSION[branch.mnemonic]
        original_target = branch.operand

        # Create new elements
        result = []

        # 1. Inverted branch to skip label
        inverted_branch = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=f"    {inverted_mnemonic} {skip_label}",
            mnemonic=inverted_mnemonic,
            operand=skip_label,
            comment=f"Long branch fixup (was {branch.mnemonic})",
            size=2
        )
        result.append(inverted_branch)

        # 2. JMP to original target
        jmp_instr = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=f"    JMP {original_target}",
            mnemonic="JMP",
            operand=original_target,
            comment=None,
            size=3
        )
        result.append(jmp_instr)

        # 3. Skip label
        skip_label_elem = AsmLabel(
            kind=AsmElementKind.LABEL,
            original_line=f"{skip_label}:",
            name=skip_label
        )
        result.append(skip_label_elem)

        return result

    def _elements_to_lines(self, elements: List[AsmElement]) -> List[str]:
        """Convert elements back to assembly lines."""
        lines = []

        for elem in elements:
            if isinstance(elem, AsmInstruction):
                # Reconstruct instruction line
                if elem.operand:
                    line = f"    {elem.mnemonic} {elem.operand}"
                else:
                    line = f"    {elem.mnemonic}"

                if elem.comment:
                    # Pad to comment column
                    padding = max(1, 32 - len(line))
                    line += " " * padding + f"; {elem.comment}"

                lines.append(line)
            elif isinstance(elem, AsmLabel):
                lines.append(f"{elem.name}:")
            else:
                # Use original line for directives, comments, blanks
                lines.append(elem.original_line)

        return lines


# =============================================================================
# Public API
# =============================================================================

def fixup_long_branches(assembly_lines: List[str]) -> Tuple[List[str], int]:
    """
    Apply long branch fixup to assembly source.

    This function should be called after peephole optimization and before
    final assembly output.

    Args:
        assembly_lines: List of assembly source lines

    Returns:
        Tuple of (fixed lines, number of branches fixed)
    """
    fixup = BranchFixup()
    return fixup.fixup(assembly_lines)
