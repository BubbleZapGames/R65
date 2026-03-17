# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
STATUS register flag definitions for the 65816 processor.

Provides metadata for each STATUS flag including bit positions, branch instructions,
and write instructions for optimized code generation.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class StatusFlag:
    """Metadata for a STATUS register flag."""
    name: str              # e.g., "Carry"
    bit_position: int      # 0-7
    bit_mask: int          # e.g., 0x01
    branch_set: Optional[str]    # Branch if set, e.g., "BCS" (None if not branchable)
    branch_clear: Optional[str]  # Branch if clear, e.g., "BCC" (None if not branchable)
    set_instruction: Optional[str]    # Instruction to set flag, e.g., "SEC"
    clear_instruction: Optional[str]  # Instruction to clear flag, e.g., "CLC"
    is_writable: bool      # Can be written directly (False for Zero, Overflow, Negative)


# All STATUS flags with their metadata
# Bit layout: NVMXDIZC (Negative, Overflow, M-flag, X-flag, Decimal, IRQ-disable, Zero, Carry)
STATUS_FLAGS: Dict[str, StatusFlag] = {
    'Carry': StatusFlag(
        name='Carry',
        bit_position=0,
        bit_mask=0x01,
        branch_set='BCS',
        branch_clear='BCC',
        set_instruction='SEC',
        clear_instruction='CLC',
        is_writable=True
    ),
    'Zero': StatusFlag(
        name='Zero',
        bit_position=1,
        bit_mask=0x02,
        branch_set='BEQ',
        branch_clear='BNE',
        set_instruction=None,  # Set by CPU operations
        clear_instruction=None,
        is_writable=False
    ),
    'Irq': StatusFlag(
        name='Irq',
        bit_position=2,
        bit_mask=0x04,
        branch_set=None,  # No dedicated branch instruction
        branch_clear=None,
        set_instruction='SEI',
        clear_instruction='CLI',
        is_writable=True
    ),
    'Decimal': StatusFlag(
        name='Decimal',
        bit_position=3,
        bit_mask=0x08,
        branch_set=None,  # No dedicated branch instruction
        branch_clear=None,
        set_instruction='SED',
        clear_instruction='CLD',
        is_writable=True
    ),
    'XY16': StatusFlag(
        name='XY16',
        bit_position=4,
        bit_mask=0x10,
        branch_set=None,  # No dedicated branch instruction
        branch_clear=None,
        # Note: XY16=true means 16-bit mode (X flag=0), XY16=false means 8-bit (X flag=1)
        # So set_instruction (for XY16=false->8-bit) is SEP, clear_instruction (for XY16=true->16-bit) is REP
        set_instruction='SEP',  # SEP #$10 - sets X flag to 1 (8-bit mode, XY16=false)
        clear_instruction='REP',  # REP #$10 - clears X flag to 0 (16-bit mode, XY16=true)
        is_writable=True
    ),
    'A16': StatusFlag(
        name='A16',
        bit_position=5,
        bit_mask=0x20,
        branch_set=None,  # No dedicated branch instruction
        branch_clear=None,
        # Note: A16=true means 16-bit mode (M flag=0), A16=false means 8-bit (M flag=1)
        # So set_instruction (for A16=false->8-bit) is SEP, clear_instruction (for A16=true->16-bit) is REP
        set_instruction='SEP',  # SEP #$20 - sets M flag to 1 (8-bit mode, A16=false)
        clear_instruction='REP',  # REP #$20 - clears M flag to 0 (16-bit mode, A16=true)
        is_writable=True
    ),
    'Overflow': StatusFlag(
        name='Overflow',
        bit_position=6,
        bit_mask=0x40,
        branch_set='BVS',
        branch_clear='BVC',
        set_instruction=None,  # Set by CPU operations (CLV exists but no SEV)
        clear_instruction='CLV',  # CLV exists
        is_writable=False  # No SEV instruction
    ),
    'Negative': StatusFlag(
        name='Negative',
        bit_position=7,
        bit_mask=0x80,
        branch_set='BMI',
        branch_clear='BPL',
        set_instruction=None,  # Set by CPU operations
        clear_instruction=None,
        is_writable=False
    ),
}


def get_status_flag(name: str) -> Optional[StatusFlag]:
    """
    Get STATUS flag metadata by name.

    Args:
        name: Flag name (e.g., "Carry", "Zero")

    Returns:
        StatusFlag metadata or None if not found
    """
    return STATUS_FLAGS.get(name)


def is_branchable_flag(name: str) -> bool:
    """
    Check if a STATUS flag has dedicated branch instructions.

    Branchable flags: Carry, Zero, Overflow, Negative
    Non-branchable flags: Irq, Decimal, XY16, A16

    Args:
        name: Flag name

    Returns:
        True if flag has dedicated branch instructions
    """
    flag = STATUS_FLAGS.get(name)
    return flag is not None and flag.branch_set is not None


def get_all_flag_names() -> list:
    """Get list of all valid STATUS flag names."""
    return list(STATUS_FLAGS.keys())
