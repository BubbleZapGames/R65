"""
Processor mode representation and tracking for 65816.

Tracks M and X mode bits which control register sizes:
- M bit (bit 5): Accumulator (A) size - m8 (8-bit) or m16 (16-bit)
- X bit (bit 4): Index register (X/Y) size - x8 (8-bit) or x16 (16-bit)
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
from r65.compiler.hir import *


class ModeState(Enum):
    """Tri-state mode values for flow analysis."""
    M8 = "m8"       # 8-bit (M=1)
    M16 = "m16"     # 16-bit (M=0)
    UNKNOWN = "?"   # Unknown (for uninitialized or multiple paths)


class XModeState(Enum):
    """Tri-state X mode values."""
    X8 = "x8"
    X16 = "x16"
    UNKNOWN = "?"


@dataclass(frozen=True)  # Immutable for hashing
class ProcessorMode:
    """
    Represents the processor mode state at a program point.

    Immutable to enable safe sharing across CFG nodes.
    """
    m_mode: ModeState    # Accumulator mode
    x_mode: XModeState   # Index register mode

    @staticmethod
    def from_attribute(mode_attr) -> 'ProcessorMode':
        """Create ProcessorMode from ModeAttribute."""
        if mode_attr is None:
            return ProcessorMode.unknown()

        m_state = ModeState.UNKNOWN
        if mode_attr.m_mode == MMode.M8:
            m_state = ModeState.M8
        elif mode_attr.m_mode == MMode.M16:
            m_state = ModeState.M16

        x_state = XModeState.UNKNOWN
        if mode_attr.x_mode == XMode.X8:
            x_state = XModeState.X8
        elif mode_attr.x_mode == XMode.X16:
            x_state = XModeState.X16

        return ProcessorMode(m_state, x_state)

    @staticmethod
    def unknown() -> 'ProcessorMode':
        """Create a fully unknown mode."""
        return ProcessorMode(ModeState.UNKNOWN, XModeState.UNKNOWN)

    def is_fully_known(self) -> bool:
        """Check if both modes are known (not UNKNOWN)."""
        return (self.m_mode != ModeState.UNKNOWN and
                self.x_mode != XModeState.UNKNOWN)

    def get_a_type(self) -> Optional[TypeInfo]:
        """Get type of A register in this mode."""
        if self.m_mode == ModeState.M8:
            return BasicTypeInfo("u8")
        elif self.m_mode == ModeState.M16:
            return BasicTypeInfo("u16")
        else:
            return None  # Unknown mode

    def get_x_type(self) -> Optional[TypeInfo]:
        """Get type of X register in this mode."""
        if self.x_mode == XModeState.X8:
            return BasicTypeInfo("u8")
        elif self.x_mode == XModeState.X16:
            return BasicTypeInfo("u16")
        else:
            return None

    def get_y_type(self) -> Optional[TypeInfo]:
        """Get type of Y register in this mode."""
        return self.get_x_type()  # Y follows X

    def get_register_type(self, reg_name: str) -> Optional[TypeInfo]:
        """Get type of any register in this mode."""
        if reg_name == 'A':
            return self.get_a_type()
        elif reg_name == 'X':
            return self.get_x_type()
        elif reg_name == 'Y':
            return self.get_y_type()
        elif reg_name == 'B':
            # B register only available in m8 mode
            if self.m_mode == ModeState.M8:
                return BasicTypeInfo("u8")
            else:
                # B not available in m16 mode or unknown mode
                return None
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
        Bit 4 (0x10): X bit (index)
        """
        new_m = self.m_mode
        new_x = self.x_mode

        if mask & 0x20:  # Set M bit
            new_m = ModeState.M8
        if mask & 0x10:  # Set X bit
            new_x = XModeState.X8

        return ProcessorMode(new_m, new_x)

    def apply_rep(self, mask: int) -> 'ProcessorMode':
        """
        Apply REP (Reset Processor Status) instruction.
        REP clears bits (0 = 16-bit mode).

        Bit 5 (0x20): M bit (accumulator)
        Bit 4 (0x10): X bit (index)
        """
        new_m = self.m_mode
        new_x = self.x_mode

        if mask & 0x20:  # Clear M bit
            new_m = ModeState.M16
        if mask & 0x10:  # Clear X bit
            new_x = XModeState.X16

        return ProcessorMode(new_m, new_x)

    def is_compatible(self, other: 'ProcessorMode') -> bool:
        """
        Check if this mode is compatible with another.

        Compatible means:
        - Both modes are equal, OR
        - One mode has UNKNOWN in a dimension where the other is known

        Used for function calls with partial mode annotations.
        """
        m_compat = (self.m_mode == other.m_mode or
                    self.m_mode == ModeState.UNKNOWN or
                    other.m_mode == ModeState.UNKNOWN)

        x_compat = (self.x_mode == other.x_mode or
                    self.x_mode == XModeState.UNKNOWN or
                    other.x_mode == XModeState.UNKNOWN)

        return m_compat and x_compat

    def join(self, other: 'ProcessorMode') -> Optional['ProcessorMode']:
        """
        Join two modes at a control flow merge point.

        Returns:
        - ProcessorMode if modes are compatible (same or one unknown)
        - None if modes conflict (different known values)
        """
        if self == other:
            return self

        # Join M mode
        if self.m_mode == other.m_mode:
            new_m = self.m_mode
        elif self.m_mode == ModeState.UNKNOWN:
            new_m = other.m_mode
        elif other.m_mode == ModeState.UNKNOWN:
            new_m = self.m_mode
        else:
            return None  # Conflict

        # Join X mode
        if self.x_mode == other.x_mode:
            new_x = self.x_mode
        elif self.x_mode == XModeState.UNKNOWN:
            new_x = other.x_mode
        elif other.x_mode == XModeState.UNKNOWN:
            new_x = self.x_mode
        else:
            return None  # Conflict

        return ProcessorMode(new_m, new_x)

    def __str__(self) -> str:
        m_str = self.m_mode.value
        x_str = self.x_mode.value
        return f"({m_str}, {x_str})"
