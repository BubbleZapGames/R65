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
from r65.compiler.codegen.asm_size import data_directive_size, raw_asm_size


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
                return data_directive_size(name, args)
            case RawAsm(text):
                return raw_asm_size(text, m16, x16)
            case Label() | Comment() | BlankLine() | ModeChange(_, _, _):
                return 0
            case _:
                return 0


