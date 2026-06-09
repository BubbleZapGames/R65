# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Processor mode representation and tracking for 65816.

Tracks M and X mode bits which control register sizes:
- M bit (bit 5): Accumulator (A) size - m8 (8-bit) or m16 (16-bit)
- X bit (bit 4): Index register (X/Y) size - always x16 (16-bit) in R65

Mode inference:
- Default mode: m8 (8-bit A), x16 (16-bit X/Y)
- Function entry mode: m16 if A parameter is u16, otherwise m8
- X/Y registers are always u16 (x16 mode)
- Auto REP/SEP is inserted around 16-bit A operations
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from enum import Enum
from r65.compiler.hir import *

if TYPE_CHECKING:
    pass


class ModeState(Enum):
    """Accumulator mode values."""
    M8 = "m8"       # 8-bit (M=1)
    M16 = "m16"     # 16-bit (M=0)


class XModeState(Enum):
    """Index register mode values. X/Y are always 16-bit in R65."""
    X16 = "x16"     # 16-bit (X=0) - the only supported mode


@dataclass(frozen=True)  # Immutable for hashing
class ProcessorMode:
    """
    Represents the processor mode state at a program point.

    Immutable to enable safe sharing across CFG nodes.

    In R65:
    - X/Y registers are always 16-bit (x16 mode)
    - A register is 8-bit by default, switches to 16-bit for u16 operations
    - Mode inference happens based on parameter types
    """
    m_mode: ModeState    # Accumulator mode
    x_mode: XModeState = XModeState.X16  # Always x16 in R65

    @staticmethod
    def default() -> 'ProcessorMode':
        """Create the default mode: m8 (8-bit A), x16 (16-bit X/Y)."""
        return ProcessorMode(ModeState.M8, XModeState.X16)


    def is_fully_known(self) -> bool:
        """Check if mode is fully known (always True in new design)."""
        # Mode is always known in the new design - m8 or m16, always x16
        return True

    def get_a_type(self) -> TypeInfo:
        """Get type of A register in this mode."""
        if self.m_mode == ModeState.M8:
            return BasicTypeInfo("u8")
        else:  # M16
            return BasicTypeInfo("u16")

    def get_x_type(self) -> TypeInfo:
        """Get type of X register (always u16 in R65)."""
        return BasicTypeInfo("u16")

    def get_y_type(self) -> TypeInfo:
        """Get type of Y register (always u16 in R65)."""
        return BasicTypeInfo("u16")

    def get_register_type(self, reg_name: str) -> Optional[TypeInfo]:
        """Get type of any register in this mode."""
        if reg_name == 'A':
            return self.get_a_type()
        elif reg_name == 'X':
            return self.get_x_type()
        elif reg_name == 'Y':
            return self.get_y_type()
        elif reg_name == 'B':
            # B register is always u8 - it's the high byte of the 16-bit accumulator
            # Programmer is responsible for ensuring proper mode via STATUS.A16 = false
            return BasicTypeInfo("u8")
        elif reg_name == 'STATUS':
            return BasicTypeInfo("u8")
        elif reg_name == 'D':
            return BasicTypeInfo("u16")
        elif reg_name == 'DBR':
            return BasicTypeInfo("u8")
        elif reg_name == 'PBR':
            return BasicTypeInfo("u8")
        elif reg_name == 'S':
            return BasicTypeInfo("u16")
        else:
            return None

    def apply_sep(self, mask: int) -> 'ProcessorMode':
        """
        Apply SEP (Set Processor Status) instruction.
        SEP sets bits (1 = 8-bit mode).

        Bit 5 (0x20): M bit (accumulator)
        Bit 4 (0x10): X bit (index) - ignored in R65 (always x16)
        """
        new_m = self.m_mode

        if mask & 0x20:  # Set M bit
            new_m = ModeState.M8
        # Ignore X bit - always x16 in R65

        return ProcessorMode(new_m, XModeState.X16)

    def apply_rep(self, mask: int) -> 'ProcessorMode':
        """
        Apply REP (Reset Processor Status) instruction.
        REP clears bits (0 = 16-bit mode).

        Bit 5 (0x20): M bit (accumulator)
        Bit 4 (0x10): X bit (index) - ignored in R65 (always x16)
        """
        new_m = self.m_mode

        if mask & 0x20:  # Clear M bit
            new_m = ModeState.M16
        # Ignore X bit - always x16 in R65

        return ProcessorMode(new_m, XModeState.X16)

    def is_compatible(self, other: 'ProcessorMode') -> bool:
        """
        Check if this mode is compatible with another.

        In the new design, modes are compatible if:
        - M modes match (both m8 or both m16)
        - X modes always match (both x16)
        """
        return self.m_mode == other.m_mode

    def join(self, other: 'ProcessorMode') -> Optional['ProcessorMode']:
        """
        Join two modes at a control flow merge point.

        Returns:
        - ProcessorMode if modes are compatible (m_mode matches)
        - None if modes conflict (different m_mode values)
        """
        if self.m_mode == other.m_mode:
            return self
        return None  # Conflict - different M modes

    def __str__(self) -> str:
        m_str = self.m_mode.value
        return f"({m_str}, x16)"
