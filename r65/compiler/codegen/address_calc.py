# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Calculate byte addresses for assembly nodes.

Provides address calculation for generating debug info with accurate
instruction addresses for source-level debugging.
"""

from typing import List, Dict, Tuple

from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Directive, ModeChange, BlankLine, Comment,
    RawAsm
)
from r65.compiler.codegen.opcodes import instruction_size


class AddressCalculator:
    """
    Calculates addresses for assembly nodes within a segment.

    Walks through the assembly nodes and assigns addresses based on
    instruction sizes and directive data sizes.
    """

    def __init__(self, base_address: int, m16: bool = False, x16: bool = True):
        """
        Initialize address calculator.

        Args:
            base_address: Starting address for the segment
            m16: Whether accumulator is in 16-bit mode (affects immediate sizes)
            x16: Whether index registers are in 16-bit mode (affects immediate sizes)
        """
        self.base_address = base_address
        self.m16 = m16
        self.x16 = x16

    def calculate(self, nodes: List[AsmNode]) -> Dict[int, Tuple[int, int]]:
        """
        Calculate addresses for all nodes.

        Tracks .ACCU and .INDEX mode changes to correctly size immediate
        instructions based on the current processor mode.

        Args:
            nodes: List of AsmNode objects

        Returns:
            Dict mapping node index to (address, size) tuple.
            Only nodes with non-zero size are included.
        """
        result = {}
        offset = 0

        # Track current processor mode (may change via .ACCU/.INDEX directives)
        current_m16 = self.m16
        current_x16 = self.x16

        for i, node in enumerate(nodes):
            # Track mode-changing directives — sizing of accumulator
            # / index immediate operands depends on m / x flag width.
            if isinstance(node, ModeChange):
                if node.flag == 'ACCU':
                    current_m16 = node.bits == 16
                elif node.flag == 'INDEX':
                    current_x16 = node.bits == 16

            size = self._node_size_with_mode(node, current_m16, current_x16)
            if size > 0:
                result[i] = (self.base_address + offset, size)
            offset += size

        return result

    def calculate_with_labels(self, nodes: List[AsmNode]) -> Tuple[Dict[int, Tuple[int, int]], Dict[str, int]]:
        """
        Calculate addresses for all nodes and collect label addresses.

        Tracks .ACCU and .INDEX mode changes to correctly size immediate
        instructions based on the current processor mode.

        Args:
            nodes: List of AsmNode objects

        Returns:
            Tuple of:
            - Dict mapping node index to (address, size) tuple
            - Dict mapping label names to addresses
        """
        node_addresses = {}
        label_addresses = {}
        offset = 0

        # Track current processor mode (may change via .ACCU/.INDEX directives)
        current_m16 = self.m16
        current_x16 = self.x16

        for i, node in enumerate(nodes):
            # Track mode-changing directives — sizing of accumulator
            # / index immediate operands depends on m / x flag width.
            if isinstance(node, ModeChange):
                if node.flag == 'ACCU':
                    current_m16 = node.bits == 16
                elif node.flag == 'INDEX':
                    current_x16 = node.bits == 16

            # Record label address before processing
            if isinstance(node, Label):
                label_addresses[node.name] = self.base_address + offset

            size = self._node_size_with_mode(node, current_m16, current_x16)
            if size > 0:
                node_addresses[i] = (self.base_address + offset, size)
            offset += size

        return node_addresses, label_addresses


    def _node_size(self, node: AsmNode) -> int:
        """
        Get size of a single node in bytes.

        Uses the instance's default m16/x16 modes.

        Args:
            node: AsmNode to measure

        Returns:
            Size in bytes (0 for labels, comments, blank lines)
        """
        return self._node_size_with_mode(node, self.m16, self.x16)

    def _node_size_with_mode(self, node: AsmNode, m16: bool, x16: bool) -> int:
        """
        Get size of a single node in bytes with explicit mode.

        Args:
            node: AsmNode to measure
            m16: True if accumulator is 16-bit
            x16: True if index registers are 16-bit

        Returns:
            Size in bytes (0 for labels, comments, blank lines)
        """
        match node:
            case Instruction(opcode, _, _, _):
                return instruction_size(opcode, m16, x16)
            case Directive(name, args, _):
                return self._directive_size(name, args)
            case RawAsm(text):
                return self._raw_asm_size(text)
            case Label() | Comment() | BlankLine() | ModeChange(_, _, _):
                return 0
            case _:
                return 0

    def _directive_size(self, name: str, args: List[str]) -> int:
        """
        Estimate size of data directives.

        Args:
            name: Directive name (e.g., ".db", ".dw")
            args: Directive arguments

        Returns:
            Size in bytes
        """
        name_lower = name.lower()

        if name_lower == ".db":
            # Count bytes - each arg could be a single byte or a string
            size = 0
            for arg in args:
                arg = arg.strip()
                if arg.startswith('"') and arg.endswith('"'):
                    # String literal - count characters
                    size += len(arg) - 2  # Subtract quotes
                else:
                    # Single byte value
                    size += 1
            return size
        elif name_lower == ".dw":
            return len(args) * 2
        elif name_lower == ".dl":
            return len(args) * 3
        elif name_lower == ".dd":
            return len(args) * 4
        elif name_lower == ".dsb":
            # .dsb size [, fill]
            if args:
                try:
                    return self._parse_int(args[0])
                except ValueError:
                    return 0
            return 0
        elif name_lower == ".dsw":
            # .dsw count - reserves count * 2 bytes
            if args:
                try:
                    return self._parse_int(args[0]) * 2
                except ValueError:
                    return 0
            return 0
        elif name_lower in (".accu", ".index", ".org", ".bank", ".define", ".equ"):
            # Control directives - no code size
            return 0
        else:
            # Unknown directive - assume no size
            return 0

    def _raw_asm_size(self, text: str) -> int:
        """
        Estimate size of raw assembly text.

        This is a rough estimate since raw assembly can contain anything.
        We try to detect common patterns.

        Args:
            text: Raw assembly text

        Returns:
            Estimated size in bytes
        """
        text = text.strip()

        # Skip empty lines and comments
        if not text or text.startswith(';'):
            return 0

        # Skip directives (start with .)
        if text.startswith('.'):
            return 0

        # Skip labels (end with :)
        if text.endswith(':'):
            return 0

        # Skip WLA-DX control directives (inside .MEMORYMAP, .ROMBANKMAP, etc.)
        # These don't start with '.' but produce no code
        first_word = text.split()[0].upper() if text.split() else ''
        wla_control_directives = {
            'DEFAULTSLOT', 'SLOTSIZE', 'SLOT',
            'BANKSTOTAL', 'BANKSIZE', 'BANKS',
            'LOROM', 'HIROM', 'EXLOROM', 'EXHIROM',
            'SLOWROM', 'FASTROM',
            'CARTRIDGETYPE', 'ROMSIZE', 'SRAMSIZE',
            'COUNTRY', 'LICENSEECODE', 'VERSION',
            'ID', 'NAME',
            'COMPUTESNESCHECKSUM', 'COMPUTEGBCHECKSUM',
            'FORCE', 'BACKGROUND', 'INTERRUPT',
            'NMI', 'IRQ', 'BRK', 'COP', 'ABORT', 'UNUSED',
            'EMUNMI', 'EMUIRQ', 'EMUCOP', 'EMUABORT', 'EMUUNUSED', 'EMURESET',
            'NATIVECOP', 'NATIVEBRK', 'NATIVEABORT', 'NATIVENMI',
            'NATIVEIRQ', 'NATIVERESET', 'NATIVEUNUSED',
        }
        if first_word in wla_control_directives:
            return 0

        # Single-byte implied instructions (no operand)
        single_byte_implied = {
            # Flag operations
            'CLC', 'CLD', 'CLI', 'CLV', 'SEC', 'SED', 'SEI',
            # Transfer operations
            'TAX', 'TAY', 'TXA', 'TYA', 'TSX', 'TXS',
            'TCD', 'TDC', 'TCS', 'TSC', 'TXY', 'TYX',
            # Stack operations
            'PHA', 'PHP', 'PHX', 'PHY', 'PHB', 'PHD', 'PHK',
            'PLA', 'PLP', 'PLX', 'PLY', 'PLB', 'PLD',
            # Increment/Decrement
            'INX', 'INY', 'DEX', 'DEY', 'INC', 'DEC',  # INC A / DEC A
            # Returns and misc
            'RTS', 'RTI', 'RTL', 'NOP', 'WAI', 'STP', 'WDM',
            # 65816 specific
            'XBA', 'XCE',
        }
        if first_word in single_byte_implied:
            return 1

        # For other instructions, estimate based on instruction type
        # 2 bytes for most instructions with operands, 3 for absolute addressing
        return 2

    def _parse_int(self, value: str) -> int:
        """
        Parse an integer value from various formats.

        Handles:
        - Decimal: 123
        - Hex with $: $1A
        - Hex with 0x: 0x1A

        Args:
            value: String value to parse

        Returns:
            Integer value
        """
        value = value.strip()
        if value.startswith('$'):
            return int(value[1:], 16)
        elif value.startswith('0x') or value.startswith('0X'):
            return int(value[2:], 16)
        else:
            return int(value)


