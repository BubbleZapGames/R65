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
# It parses assembly text to AsmNode objects with proper Opcodes, applies
# the typed BranchFixup, then converts back to strings.

from enum import Enum, auto
from typing import Optional
from r65.compiler.codegen.asm_nodes import emit_nodes


# ============================================================================
# Opcode Parsing from Assembly Text
# ============================================================================

# Map mnemonic to branch Opcode
MNEMONIC_TO_BRANCH_OPCODE: Dict[str, Opcode] = {
    'BCC': Opcode.BCC, 'BCS': Opcode.BCS,
    'BEQ': Opcode.BEQ, 'BNE': Opcode.BNE,
    'BMI': Opcode.BMI, 'BPL': Opcode.BPL,
    'BVC': Opcode.BVC, 'BVS': Opcode.BVS,
    'BRA': Opcode.BRA, 'BRL': Opcode.BRL,
}

# Map mnemonic to implied (no operand) Opcode
MNEMONIC_TO_IMPLIED_OPCODE: Dict[str, Opcode] = {
    'NOP': Opcode.NOP, 'RTS': Opcode.RTS, 'RTL': Opcode.RTL, 'RTI': Opcode.RTI,
    'SEC': Opcode.SEC, 'CLC': Opcode.CLC, 'SEI': Opcode.SEI, 'CLI': Opcode.CLI,
    'SED': Opcode.SED, 'CLD': Opcode.CLD,
    'PHP': Opcode.PHP, 'PLP': Opcode.PLP, 'PHA': Opcode.PHA, 'PLA': Opcode.PLA,
    'PHX': Opcode.PHX, 'PLX': Opcode.PLX, 'PHY': Opcode.PHY, 'PLY': Opcode.PLY,
    'PHB': Opcode.PHB, 'PLB': Opcode.PLB, 'PHD': Opcode.PHD, 'PLD': Opcode.PLD,
    'PHK': Opcode.PHK,
    'TAX': Opcode.TAX, 'TXA': Opcode.TXA, 'TAY': Opcode.TAY, 'TYA': Opcode.TYA,
    'TSX': Opcode.TSX, 'TXS': Opcode.TXS, 'TXY': Opcode.TXY, 'TYX': Opcode.TYX,
    'TCD': Opcode.TCD, 'TDC': Opcode.TDC, 'TCS': Opcode.TCS, 'TSC': Opcode.TSC,
    'INX': Opcode.INX, 'DEX': Opcode.DEX, 'INY': Opcode.INY, 'DEY': Opcode.DEY,
    'INC': Opcode.INC, 'DEC': Opcode.DEC,
    'ASL': Opcode.ASL, 'LSR': Opcode.LSR, 'ROL': Opcode.ROL, 'ROR': Opcode.ROR,
    'XBA': Opcode.XBA, 'WAI': Opcode.WAI, 'STP': Opcode.STP,
}

# Map mnemonic to jump Opcode (absolute addressing)
MNEMONIC_TO_JUMP_OPCODE: Dict[str, Opcode] = {
    'JMP': Opcode.JMP_ABSOLUTE,
    'JML': Opcode.JMP_LONG,
    'JSR': Opcode.JSR,
    'JSL': Opcode.JSL,
}


def parse_opcode(mnemonic_str: str, operand: Optional[str]) -> Opcode:
    """
    Parse mnemonic and operand text to determine the correct Opcode.

    Args:
        mnemonic_str: Instruction mnemonic (e.g., "LDA", "BEQ")
        operand: Operand text (e.g., "#$42", "$20", "label")

    Returns:
        The appropriate Opcode enum value
    """
    mnemonic_str = mnemonic_str.upper()

    # Branch instructions
    if mnemonic_str in MNEMONIC_TO_BRANCH_OPCODE:
        return MNEMONIC_TO_BRANCH_OPCODE[mnemonic_str]

    # Implied (no operand)
    if operand is None and mnemonic_str in MNEMONIC_TO_IMPLIED_OPCODE:
        return MNEMONIC_TO_IMPLIED_OPCODE[mnemonic_str]

    # Jump instructions
    if mnemonic_str in MNEMONIC_TO_JUMP_OPCODE:
        return MNEMONIC_TO_JUMP_OPCODE[mnemonic_str]

    # REP/SEP (immediate)
    if mnemonic_str == 'REP':
        return Opcode.REP
    if mnemonic_str == 'SEP':
        return Opcode.SEP

    # Determine addressing mode from operand
    if operand is None:
        # Try accumulator mode for ASL, LSR, etc.
        return MNEMONIC_TO_IMPLIED_OPCODE.get(mnemonic_str, Opcode.NOP)

    # Parse operand to determine addressing mode
    operand = operand.strip()

    # Immediate: #value
    if operand.startswith('#'):
        return _get_immediate_opcode(mnemonic_str)

    # Stack relative: d,S
    if ',S' in operand.upper():
        return _get_stack_opcode(mnemonic_str)

    # Indexed modes
    if ',X' in operand.upper():
        addr_part = operand.split(',')[0].strip()
        if addr_part.startswith('$') and len(addr_part) > 3:
            return _get_absolute_x_opcode(mnemonic_str)
        return _get_dp_x_opcode(mnemonic_str)

    if ',Y' in operand.upper():
        addr_part = operand.split(',')[0].strip()
        if addr_part.startswith('$') and len(addr_part) > 3:
            return _get_absolute_y_opcode(mnemonic_str)
        return _get_dp_y_opcode(mnemonic_str)

    # Long addressing: f:addr or >>
    if operand.startswith('f:') or '>>' in operand:
        return _get_long_opcode(mnemonic_str)

    # Direct page vs absolute based on address size
    if operand.startswith('$'):
        addr = operand[1:].split()[0]
        if len(addr) <= 2:
            return _get_dp_opcode(mnemonic_str)
        return _get_absolute_opcode(mnemonic_str)

    # Label reference - assume absolute
    return _get_absolute_opcode(mnemonic_str)


def _get_immediate_opcode(mnemonic: str) -> Opcode:
    """Get immediate addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_IMMEDIATE, 'LDX': Opcode.LDX_IMMEDIATE, 'LDY': Opcode.LDY_IMMEDIATE,
        'ADC': Opcode.ADC_IMMEDIATE, 'SBC': Opcode.SBC_IMMEDIATE,
        'AND': Opcode.AND_IMMEDIATE, 'ORA': Opcode.ORA_IMMEDIATE, 'EOR': Opcode.EOR_IMMEDIATE,
        'CMP': Opcode.CMP_IMMEDIATE, 'CPX': Opcode.CPX_IMMEDIATE, 'CPY': Opcode.CPY_IMMEDIATE,
        'BIT': Opcode.BIT_IMMEDIATE,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_dp_opcode(mnemonic: str) -> Opcode:
    """Get direct page addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_DP, 'LDX': Opcode.LDX_DP, 'LDY': Opcode.LDY_DP,
        'STA': Opcode.STA_DP, 'STX': Opcode.STX_DP, 'STY': Opcode.STY_DP, 'STZ': Opcode.STZ_DP,
        'ADC': Opcode.ADC_DP, 'SBC': Opcode.SBC_DP,
        'AND': Opcode.AND_DP, 'ORA': Opcode.ORA_DP, 'EOR': Opcode.EOR_DP,
        'CMP': Opcode.CMP_DP, 'CPX': Opcode.CPX_DP, 'CPY': Opcode.CPY_DP,
        'BIT': Opcode.BIT_DP,
        'INC': Opcode.INC_DP, 'DEC': Opcode.DEC_DP,
        'ASL': Opcode.ASL_DP, 'LSR': Opcode.LSR_DP, 'ROL': Opcode.ROL_DP, 'ROR': Opcode.ROR_DP,
        'TRB': Opcode.TRB_DP, 'TSB': Opcode.TSB_DP,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_dp_x_opcode(mnemonic: str) -> Opcode:
    """Get direct page,X addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_DP_X, 'LDY': Opcode.LDY_DP_X,
        'STA': Opcode.STA_DP_X, 'STY': Opcode.STY_DP_X, 'STZ': Opcode.STZ_DP_X,
        'ADC': Opcode.ADC_DP_X, 'SBC': Opcode.SBC_DP_X,
        'AND': Opcode.AND_DP_X, 'ORA': Opcode.ORA_DP_X, 'EOR': Opcode.EOR_DP_X,
        'CMP': Opcode.CMP_DP_X,
        'BIT': Opcode.BIT_DP_X,
        'INC': Opcode.INC_DP_X, 'DEC': Opcode.DEC_DP_X,
        'ASL': Opcode.ASL_DP_X, 'LSR': Opcode.LSR_DP_X, 'ROL': Opcode.ROL_DP_X, 'ROR': Opcode.ROR_DP_X,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_dp_y_opcode(mnemonic: str) -> Opcode:
    """Get direct page,Y addressing mode opcode."""
    mapping = {
        'LDX': Opcode.LDX_DP_Y,
        'STX': Opcode.STX_DP_Y,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_absolute_opcode(mnemonic: str) -> Opcode:
    """Get absolute addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_ABSOLUTE, 'LDX': Opcode.LDX_ABSOLUTE, 'LDY': Opcode.LDY_ABSOLUTE,
        'STA': Opcode.STA_ABSOLUTE, 'STX': Opcode.STX_ABSOLUTE, 'STY': Opcode.STY_ABSOLUTE, 'STZ': Opcode.STZ_ABSOLUTE,
        'ADC': Opcode.ADC_ABSOLUTE, 'SBC': Opcode.SBC_ABSOLUTE,
        'AND': Opcode.AND_ABSOLUTE, 'ORA': Opcode.ORA_ABSOLUTE, 'EOR': Opcode.EOR_ABSOLUTE,
        'CMP': Opcode.CMP_ABSOLUTE, 'CPX': Opcode.CPX_ABSOLUTE, 'CPY': Opcode.CPY_ABSOLUTE,
        'BIT': Opcode.BIT_ABSOLUTE,
        'INC': Opcode.INC_ABSOLUTE, 'DEC': Opcode.DEC_ABSOLUTE,
        'ASL': Opcode.ASL_ABSOLUTE, 'LSR': Opcode.LSR_ABSOLUTE, 'ROL': Opcode.ROL_ABSOLUTE, 'ROR': Opcode.ROR_ABSOLUTE,
        'TRB': Opcode.TRB_ABSOLUTE, 'TSB': Opcode.TSB_ABSOLUTE,
        'JMP': Opcode.JMP_ABSOLUTE, 'JSR': Opcode.JSR,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_absolute_x_opcode(mnemonic: str) -> Opcode:
    """Get absolute,X addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_ABSOLUTE_X, 'LDY': Opcode.LDY_ABSOLUTE_X,
        'STA': Opcode.STA_ABSOLUTE_X, 'STZ': Opcode.STZ_ABSOLUTE_X,
        'ADC': Opcode.ADC_ABSOLUTE_X, 'SBC': Opcode.SBC_ABSOLUTE_X,
        'AND': Opcode.AND_ABSOLUTE_X, 'ORA': Opcode.ORA_ABSOLUTE_X, 'EOR': Opcode.EOR_ABSOLUTE_X,
        'CMP': Opcode.CMP_ABSOLUTE_X,
        'BIT': Opcode.BIT_ABSOLUTE_X,
        'INC': Opcode.INC_ABSOLUTE_X, 'DEC': Opcode.DEC_ABSOLUTE_X,
        'ASL': Opcode.ASL_ABSOLUTE_X, 'LSR': Opcode.LSR_ABSOLUTE_X, 'ROL': Opcode.ROL_ABSOLUTE_X, 'ROR': Opcode.ROR_ABSOLUTE_X,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_absolute_y_opcode(mnemonic: str) -> Opcode:
    """Get absolute,Y addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_ABSOLUTE_Y, 'LDX': Opcode.LDX_ABSOLUTE_Y,
        'STA': Opcode.STA_ABSOLUTE_Y,
        'ADC': Opcode.ADC_ABSOLUTE_Y, 'SBC': Opcode.SBC_ABSOLUTE_Y,
        'AND': Opcode.AND_ABSOLUTE_Y, 'ORA': Opcode.ORA_ABSOLUTE_Y, 'EOR': Opcode.EOR_ABSOLUTE_Y,
        'CMP': Opcode.CMP_ABSOLUTE_Y,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_long_opcode(mnemonic: str) -> Opcode:
    """Get long addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_LONG,
        'STA': Opcode.STA_LONG,
        'ADC': Opcode.ADC_LONG, 'SBC': Opcode.SBC_LONG,
        'AND': Opcode.AND_LONG, 'ORA': Opcode.ORA_LONG, 'EOR': Opcode.EOR_LONG,
        'CMP': Opcode.CMP_LONG,
        'JMP': Opcode.JMP_LONG, 'JML': Opcode.JMP_LONG,
        'JSR': Opcode.JSL, 'JSL': Opcode.JSL,
    }
    return mapping.get(mnemonic, Opcode.NOP)


def _get_stack_opcode(mnemonic: str) -> Opcode:
    """Get stack relative addressing mode opcode."""
    mapping = {
        'LDA': Opcode.LDA_STACK,
        'STA': Opcode.STA_STACK,
        'ADC': Opcode.ADC_STACK, 'SBC': Opcode.SBC_STACK,
        'AND': Opcode.AND_STACK, 'ORA': Opcode.ORA_STACK, 'EOR': Opcode.EOR_STACK,
        'CMP': Opcode.CMP_STACK,
    }
    return mapping.get(mnemonic, Opcode.NOP)


# ============================================================================
# Operand Parsing
# ============================================================================

def parse_operand(operand_str: Optional[str]) -> Optional[Address]:
    """
    Parse operand text to an Address operand.

    Args:
        operand_str: Operand text (e.g., "#$42", "$20", "label")

    Returns:
        Address operand or None if no operand
    """
    if operand_str is None:
        return None

    operand_str = operand_str.strip()

    # For branch targets and label references, use string value
    # The Address class can hold either int or str
    if operand_str.startswith('#'):
        # Immediate value
        value_str = operand_str[1:].strip()
        if value_str.startswith('$'):
            return Address(int(value_str[1:], 16))
        elif value_str.startswith('%'):
            return Address(int(value_str[1:], 2))
        elif value_str.isdigit():
            return Address(int(value_str))
        return Address(value_str)  # Label or expression

    if operand_str.startswith('$'):
        # Hex address
        addr_part = operand_str[1:].split(',')[0].split()[0]
        return Address(int(addr_part, 16))

    # Label reference
    return Address(operand_str.split(',')[0].split()[0])


# ============================================================================
# String-Based Element Types (for test compatibility)
# ============================================================================

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
    """An assembly instruction (string-based, wraps typed Instruction)."""
    mnemonic: str = ""
    operand: Optional[str] = None
    comment: Optional[str] = None
    _typed_instr: Optional[Instruction] = None  # Typed instruction

    def __post_init__(self):
        self.kind = AsmElementKind.INSTRUCTION
        # Create typed instruction if mnemonic is set
        if self.mnemonic:
            opcode = parse_opcode(self.mnemonic, self.operand)
            typed_operand = parse_operand(self.operand) if self.operand else None
            self._typed_instr = Instruction(
                opcode=opcode,
                operand=typed_operand,
                comment=self.comment
            )

    @property
    def typed_instruction(self) -> Optional[Instruction]:
        """Get the typed Instruction."""
        return self._typed_instr


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


# Backwards-compatible aliases for tests (using Opcode enum)
CONDITIONAL_BRANCHES = {'BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS'}
BRANCH_INVERSION = {
    'BEQ': 'BNE', 'BNE': 'BEQ',
    'BCC': 'BCS', 'BCS': 'BCC',
    'BMI': 'BPL', 'BPL': 'BMI',
    'BVC': 'BVS', 'BVS': 'BVC',
}


# ============================================================================
# Assembly Parser (String to Typed Nodes)
# ============================================================================

class AssemblyParser:
    """
    Parses assembly text into typed AsmNode objects.

    Uses the Opcode enum internally for proper type safety.
    """

    def __init__(self):
        self.acc_16bit = False
        self.idx_16bit = False

    def parse_lines(self, lines: List[str]) -> List[AsmElement]:
        """Parse assembly lines into structured elements (for test compatibility)."""
        elements = []
        for line in lines:
            element = self._parse_line(line)
            if element:
                elements.append(element)
                if isinstance(element, AsmDirective):
                    self._update_mode(element)
        return elements

    def parse_to_nodes(self, lines: List[str]) -> List[AsmNode]:
        """Parse assembly lines into typed AsmNode objects."""
        nodes: List[AsmNode] = []
        for line in lines:
            node = self._parse_line_to_node(line)
            if node:
                nodes.append(node)
                if isinstance(node, Directive):
                    self._update_mode_from_directive(node)
        return nodes

    def _parse_line(self, line: str) -> Optional[AsmElement]:
        """Parse a single assembly line to element."""
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

    def _parse_line_to_node(self, line: str) -> Optional[AsmNode]:
        """Parse a single assembly line to typed node."""
        stripped = line.strip()

        if not stripped:
            return BlankLine()

        if stripped.startswith(';'):
            return Comment(text=stripped[1:].strip())

        if stripped.endswith(':') and not stripped.startswith('.'):
            return Label(name=stripped[:-1])

        if stripped.startswith('.'):
            parts = stripped.split(None, 1)
            return Directive(
                name=parts[0],
                args=[parts[1]] if len(parts) > 1 else []
            )

        return self._parse_instruction_to_node(stripped)

    def _parse_instruction(self, original: str, stripped: str) -> AsmInstruction:
        """Parse an instruction line to element."""
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

        # Calculate size using typed opcode
        if instr._typed_instr:
            instr.size = instruction_size(instr._typed_instr.opcode, self.acc_16bit, self.idx_16bit)
        else:
            instr.size = 1

        return instr

    def _parse_instruction_to_node(self, stripped: str) -> Instruction:
        """Parse an instruction line to typed node."""
        comment = None
        if ';' in stripped:
            code_part, comment_part = stripped.split(';', 1)
            stripped = code_part.strip()
            comment = comment_part.strip()

        parts = stripped.split(None, 1)
        mnemonic_str = parts[0].upper()
        operand_str = parts[1] if len(parts) > 1 else None

        opcode = parse_opcode(mnemonic_str, operand_str)
        operand = parse_operand(operand_str)

        return Instruction(opcode=opcode, operand=operand, comment=comment)

    def _update_mode(self, directive: AsmDirective):
        """Update mode tracking from .ACCU/.INDEX directives."""
        if directive.directive == '.ACCU':
            self.acc_16bit = directive.args.strip() == '16'
        elif directive.directive == '.INDEX':
            self.idx_16bit = directive.args.strip() == '16'

    def _update_mode_from_directive(self, directive: Directive):
        """Update mode tracking from typed Directive."""
        if directive.name == '.ACCU':
            self.acc_16bit = '16' in ''.join(directive.args)
        elif directive.name == '.INDEX':
            self.idx_16bit = '16' in ''.join(directive.args)


# ============================================================================
# String-Based Branch Fixup (Test Support)
# ============================================================================

class StringBranchFixup:
    """
    String-based branch fixup for testing.

    Parses string input, converts to typed AsmNode objects with Opcodes,
    applies the typed BranchFixup, then converts back to strings.
    """

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

            if isinstance(elem, AsmInstruction) and elem.mnemonic in CONDITIONAL_BRANCHES:
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
        """Rewrite a long branch using the inverted pattern (using Opcodes)."""
        skip_label = f"__branch_skip_{self.skip_label_counter}"
        self.skip_label_counter += 1

        # Get inverted opcode using typed system
        original_opcode = parse_opcode(branch.mnemonic, branch.operand)
        inverted_opcode = invert_branch(original_opcode)
        inverted_mnemonic = mnemonic(inverted_opcode) if inverted_opcode else BRANCH_INVERSION.get(branch.mnemonic, branch.mnemonic)

        original_target = branch.operand

        result: List[AsmElement] = []

        # Create inverted branch instruction
        inverted_branch = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=f"    {inverted_mnemonic} {skip_label}",
            mnemonic=inverted_mnemonic,
            operand=skip_label,
            comment=f"Long branch fixup (was {branch.mnemonic})",
        )
        # Size is calculated in __post_init__
        if inverted_branch._typed_instr:
            inverted_branch.size = instruction_size(inverted_branch._typed_instr.opcode, False, False)
        else:
            inverted_branch.size = 2
        result.append(inverted_branch)

        # Create JMP instruction
        jmp_instr = AsmInstruction(
            kind=AsmElementKind.INSTRUCTION,
            original_line=f"    JMP {original_target}",
            mnemonic="JMP",
            operand=original_target,
            comment=None,
        )
        jmp_instr.size = instruction_size(Opcode.JMP_ABSOLUTE, False, False)
        result.append(jmp_instr)

        # Create skip label
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

    Uses the typed Opcode enum internally for instruction parsing and size
    calculation.

    Args:
        assembly_lines: List of assembly source lines

    Returns:
        Tuple of (fixed lines, number of branches fixed)
    """
    fixup = StringBranchFixup()
    return fixup.fixup(assembly_lines)
