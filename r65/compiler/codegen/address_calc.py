# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Calculate byte addresses for assembly nodes.

Provides address calculation for generating debug info with accurate
instruction addresses for source-level debugging.
"""

from typing import List, Dict, Tuple

from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Label, Directive, ModeChange, BlankLine, Comment,
    RawAsm, Immediate
)
from r65.compiler.codegen.opcodes import instruction_size, mnemonic
from r65.compiler.codegen.asm_size import (
    asm_line_size, data_directive_size, raw_asm_size
)


class AddressCalculator:
    """
    Calculates addresses for assembly nodes, as wlalink lays them out.

    Walks through the assembly nodes and assigns addresses based on
    instruction sizes and directive data sizes. `.ORG` sets the offset into
    the bank; `.BANK` alone keeps it, as WLA-DX does, even when that bank was
    left somewhere else. An address is base_address plus that offset, and
    node_banks / label_banks say which WLA bank.
    """

    def __init__(self, base_address: int, m16: bool = False, x16: bool = True):
        """
        Initialize address calculator.

        Args:
            base_address: The CPU address of a bank's first byte
            m16: Whether accumulator is in 16-bit mode (affects immediate sizes)
            x16: Whether index registers are in 16-bit mode (affects immediate sizes)
        """
        self.base_address = base_address
        self.m16 = m16
        self.x16 = x16
        # Filled by calculate_with_labels: the WLA bank of each sized node
        # and label, and the address after the last byte placed in each bank
        self.node_banks: Dict[int, int] = {}
        self.label_banks: Dict[str, int] = {}
        self.bank_ends: Dict[int, int] = {}

    def calculate(self, nodes: List[AsmNode]) -> Dict[int, Tuple[int, int]]:
        """Node addresses only: calculate_with_labels without the labels."""
        return self.calculate_with_labels(nodes)[0]

    def calculate_with_labels(self, nodes: List[AsmNode]) -> Tuple[Dict[int, Tuple[int, int]], Dict[str, int]]:
        """
        Calculate addresses for all nodes and collect label addresses.

        Follows the register widths the way WLA-DX sizes immediates: the
        .ACCU/.INDEX directives, and also REP/SEP - an asm! block after a
        `SEP #$20` with no `.ACCU 8` assembles 8-bit.

        Args:
            nodes: List of AsmNode objects

        Returns:
            Tuple of:
            - Dict mapping node index to (address, size) tuple
            - Dict mapping label names to addresses
        """
        node_addresses = {}
        label_addresses = {}
        self.node_banks = {}
        self.label_banks = {}
        self.bank_ends = {}
        self._bank = 0
        self._offset = 0
        self._m16 = self.m16
        self._x16 = self.x16

        for i, node in enumerate(nodes):
            match node:
                case ModeChange():
                    if node.flag == 'ACCU':
                        self._m16 = node.bits == 16
                    elif node.flag == 'INDEX':
                        self._x16 = node.bits == 16
                    continue
                case Label():
                    label_addresses[node.name] = self.base_address + self._offset
                    self.label_banks[node.name] = self._bank
                    continue
                case RawAsm(text):
                    start, size = self._place_text(text)
                case _:
                    start = self._offset
                    size = self._node_size_with_mode(node, self._m16, self._x16)
                    self._advance(size)
                    if isinstance(node, Instruction):
                        self._follow_rep_sep(mnemonic(node.opcode), node.operand)

            if size > 0:
                node_addresses[i] = (self.base_address + start, size)
                self.node_banks[i] = self._bank

        return node_addresses, label_addresses

    def _advance(self, size: int):
        self._offset += size
        end = self.base_address + self._offset
        if end > self.bank_ends.get(self._bank, 0):
            self.bank_ends[self._bank] = end

    def _place_text(self, text: str) -> Tuple[int, int]:
        """Lay out raw assembly a line at a time; (start offset, bytes)."""
        start = None
        size = 0
        for line in text.splitlines():
            code = line.split(';', 1)[0].strip()
            word, _, rest = code.partition(' ')
            word = word.upper()
            rest = rest.strip()
            if word == '.BANK':
                self._bank = _parse_int(rest.split()[0]) if rest else 0
                continue
            if word == '.ORG' and rest:
                self._offset = _parse_int(rest)
                continue
            if word == '.ORGA' and rest:
                self._offset = _parse_int(rest) - self.base_address
                continue
            if word in ('.ACCU', '.INDEX') and rest:
                if word == '.ACCU':
                    self._m16 = rest == '16'
                else:
                    self._x16 = rest == '16'
                continue
            line_size = asm_line_size(line, self._m16, self._x16)
            if line_size > 0:
                if start is None:
                    start = self._offset
                self._advance(line_size)
                size += line_size
            mnem = word.partition('.')[0]
            if mnem in ('REP', 'SEP') and rest.startswith('#'):
                self._follow_rep_sep(mnem, rest[1:])
        return (start if start is not None else self._offset), size

    def _follow_rep_sep(self, mnem: str, operand):
        """REP/SEP of a constant mask: WLA-DX sizes what follows by it."""
        if mnem not in ('REP', 'SEP'):
            return
        value = operand.value if isinstance(operand, Immediate) else operand
        if isinstance(value, str):
            try:
                value = _parse_int(value.strip())
            except ValueError:
                return
        if not isinstance(value, int):
            return
        wide = mnem == 'REP'
        if value & 0x20:
            self._m16 = wide
        if value & 0x10:
            self._x16 = wide

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


def _parse_int(text: str) -> int:
    """A WLA-DX number: $FF00, 0xFF00, %1010 or decimal."""
    text = text.strip()
    if text.startswith('$'):
        return int(text[1:], 16)
    if text.startswith('%'):
        return int(text[1:], 2)
    return int(text, 0)
