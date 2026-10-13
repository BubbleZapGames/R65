# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Bank size validation for R65 compiler.

Validates that the generated code fits within SNES bank size limits:
- LoROM: 32KB (0x8000 bytes) per bank
- HiROM: 64KB (0x10000 bytes) per bank

The header bank (bank 0) has reduced capacity due to the SNES header
and interrupt vectors at the end of the bank.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from r65.compiler.codegen.asm_nodes import (
    AsmNode, Instruction, Directive, ModeChange, Label, Comment, BlankLine,
    RawAsm
)
from r65.compiler.errors import CompilerError
from r65.compiler.codegen.asm_size import data_directive_size, raw_asm_size


# Bank size constants
LOROM_BANK_SIZE = 0x8000   # 32KB
HIROM_BANK_SIZE = 0x10000  # 64KB

# SNES header + interrupt vectors size (at end of header bank)
# Header: $xxC0-$xxDF (32 bytes) + Vectors: $xxE0-$xxFF (32 bytes) = 64 bytes
SNES_HEADER_SIZE = 64


class BankSizeError(CompilerError):
    """Error when bank code exceeds size limit."""
    pass


@dataclass
class BankInfo:
    """Information about a bank's code size."""
    bank_number: int
    size: int
    limit: int
    is_header_bank: bool

    @property
    def overflow(self) -> int:
        """Bytes over the limit (0 if within limit)."""
        return max(0, self.size - self.limit)

    @property
    def usage_percent(self) -> float:
        """Percentage of bank capacity used."""
        return (self.size / self.limit) * 100 if self.limit > 0 else 0


def calculate_bank_sizes(
    nodes: List[AsmNode],
    is_hirom: bool = False,
    has_header: bool = True
) -> Dict[int, BankInfo]:
    """
    Calculate the code/data size for each bank.

    Args:
        nodes: List of assembly nodes after optimization
        is_hirom: True for HiROM (64KB banks), False for LoROM (32KB banks)
        has_header: True if ROM has SNES header (reduces bank 0 capacity)

    Returns:
        Dictionary mapping bank number to BankInfo
    """
    bank_size = HIROM_BANK_SIZE if is_hirom else LOROM_BANK_SIZE
    current_bank = 0
    bank_sizes: Dict[int, int] = {0: 0}

    # Track processor mode for instruction size calculation
    m16 = False
    x16 = False

    for node in nodes:
        if isinstance(node, ModeChange):
            # Tells us the accumulator / index width WLA-DX believes is
            # in effect for sizing immediates that follow.
            if node.flag == 'ACCU':
                m16 = node.bits == 16
            elif node.flag == 'INDEX':
                x16 = node.bits == 16
            continue

        if isinstance(node, Directive):
            # Check for bank switch
            if node.name == ".BANK":
                current_bank = int(node.args[0]) if node.args else 0
                if current_bank not in bank_sizes:
                    bank_sizes[current_bank] = 0

            else:
                bank_sizes[current_bank] += data_directive_size(node.name, node.args)

        elif isinstance(node, Instruction):
            # Add instruction size
            bank_sizes[current_bank] += node.size(m16=m16, x16=x16)

        elif isinstance(node, RawAsm):
            bank_sizes[current_bank] += raw_asm_size(node.text, m16, x16)

        # Labels, comments, blank lines don't consume space

    # Build BankInfo for each bank
    result: Dict[int, BankInfo] = {}
    for bank_num, size in bank_sizes.items():
        # Header bank has reduced capacity
        is_header_bank = bank_num == 0 and has_header
        limit = bank_size
        if is_header_bank:
            limit -= SNES_HEADER_SIZE

        result[bank_num] = BankInfo(
            bank_number=bank_num,
            size=size,
            limit=limit,
            is_header_bank=is_header_bank
        )

    return result


def validate_bank_sizes(
    nodes: List[AsmNode],
    is_hirom: bool = False,
    has_header: bool = True
) -> None:
    """
    Validate that all banks fit within size limits.

    Args:
        nodes: List of assembly nodes
        is_hirom: True for HiROM, False for LoROM
        has_header: True if ROM has SNES header

    Raises:
        BankSizeError: If any bank exceeds its size limit
    """
    bank_infos = calculate_bank_sizes(nodes, is_hirom, has_header)

    errors = []
    for bank_num, info in sorted(bank_infos.items()):
        if info.overflow > 0:
            mode = "HiROM" if is_hirom else "LoROM"
            limit_kb = info.limit / 1024
            size_kb = info.size / 1024

            msg = f"bank {bank_num} exceeds {mode} size limit: " \
                  f"{info.size} bytes ({size_kb:.1f}KB) > {info.limit} bytes ({limit_kb:.1f}KB)"

            if info.is_header_bank:
                msg += f" (header bank, {SNES_HEADER_SIZE} bytes reserved for SNES header)"

            errors.append(msg)

    if errors:
        raise BankSizeError(
            "ROM bank size overflow:\n  " + "\n  ".join(errors),
            hint="Split code across multiple banks using #[bank(n)] directive"
        )
